#!/usr/bin/env python3
"""Run the exact one-warmup/three-score JSpark3 v1 health witness with no retries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

PAYLOAD = {
    "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 400,
    "messages": [{"content": "Write a Python function is_prime(n) with tests. No markdown fences.", "role": "user"}],
    "model": "glm-5.3-flash", "seed": 20260830, "stream": True,
    "stream_options": {"include_usage": True}, "temperature": 0, "top_p": 1,
}
PAYLOAD_BYTES = json.dumps(PAYLOAD, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
PAYLOAD_SHA = "681a284f6d441734ccaade4d7b7c4731c736e7fb7ce87cabff1312326c5187a7"
FLOOR = 59.513845
ADVISORY = 67.261727
COUNTERS = ("port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets")


class Refusal(RuntimeError):
    pass


def parse_host(value: str) -> tuple[str, str]:
    label, mark, host = value.partition("=")
    if not mark or label not in {"rank0", "rank1", "rank2"} or not host:
        raise argparse.ArgumentTypeError("counter host must be rank0=HOST, rank1=HOST, or rank2=HOST")
    return label, host


def parse_hcas(value: str) -> tuple[str, tuple[str, str]]:
    label, mark, raw = value.partition("=")
    hcas = tuple(part.strip() for part in raw.split(","))
    if (not mark or label not in {"rank0", "rank1", "rank2"} or len(hcas) != 2 or
            len(set(hcas)) != 2 or any(re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", item) is None for item in hcas)):
        raise argparse.ArgumentTypeError("counter HCAs must be rankN=HCA0,HCA1 with two safe, distinct names")
    return label, (hcas[0], hcas[1])


def counter_snapshot(host: str, hcas: tuple[str, str]) -> dict[str, dict[str, int]]:
    code = (
        "import json,pathlib,sys; names=" + repr(COUNTERS) + "; out={}; "
        "[(out.__setitem__(h,{n:int((pathlib.Path('/sys/class/infiniband')/h/'ports/1/counters'/n).read_text()) for n in names})) for h in sys.argv[1:]]; "
        "print(json.dumps(out,sort_keys=True))"
    )
    command = shlex.join(["python3", "-c", code, *hcas])
    process = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "--", host, command,
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=15)
    if process.returncode:
        raise Refusal("rank counter snapshot failed")
    value = json.loads(process.stdout)
    if (set(value) != set(hcas) or any(set(row) != set(COUNTERS) for row in value.values()) or
            any(type(number) is not int or number < 0 for row in value.values() for number in row.values())):
        raise Refusal("rank counter snapshot schema drift")
    return value


def snapshot_all(bindings: dict[str, tuple[str, tuple[str, str]]]) -> dict[str, dict[str, dict[str, int]]]:
    labels = ("rank0", "rank1", "rank2")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {label: pool.submit(counter_snapshot, *bindings[label]) for label in labels}
        return {label: futures[label].result() for label in labels}


def positive_counter_delta(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> bool:
    return (set(before) == set(after) and all(
        set(before[hca]) == set(COUNTERS) == set(after[hca]) and
        all(after[hca][name] > before[hca][name] for name in COUNTERS)
        for hca in before
    ))


def one_request(url: str, api_key: str) -> float:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=PAYLOAD_BYTES, headers=headers)
    first = last = None
    done = False
    usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise Refusal("witness response is not HTTP 200")
        for wire in response:
            line = wire.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.monotonic_ns()
                    first = now if first is None else first
                    last = now
    if not done or usage is None or first is None or last is None:
        raise Refusal("witness SSE/usage/timing gate failed")
    completion = int(usage["completion_tokens"])
    if completion < 2 or last <= first:
        raise Refusal("witness has no measurable N-1 decode interval")
    return (completion - 1) / ((last - first) / 1e9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--counter-host", action="append", type=parse_host, default=[])
    parser.add_argument("--counter-hcas", action="append", type=parse_hcas, default=[])
    args = parser.parse_args()
    try:
        if hashlib.sha256(PAYLOAD_BYTES).hexdigest() != PAYLOAD_SHA:
            raise Refusal("embedded witness payload hash drift")
        labels = {"rank0", "rank1", "rank2"}
        if bool(args.counter_host) != bool(args.counter_hcas):
            raise Refusal("counter bracket requires both host and HCA mappings")
        if args.counter_host and (len(args.counter_host) != 3 or len(args.counter_hcas) != 3 or
                                  {label for label, _ in args.counter_host} != labels or
                                  {label for label, _ in args.counter_hcas} != labels):
            raise Refusal("counter bracket requires each symbolic rank exactly once")
        hosts = dict(args.counter_host)
        hcas = dict(args.counter_hcas)
        bindings = {label: (hosts[label], hcas[label]) for label in labels} if hosts else {}
        rates = []
        brackets = []
        key = os.environ.get(args.api_key_env, "")
        url = args.base_url.rstrip("/") + "/v1/chat/completions"
        one_request(url, key)
        for _ in range(3):
            before = snapshot_all(bindings) if bindings else {}
            rates.append(one_request(url, key))
            after = snapshot_all(bindings) if bindings else {}
            brackets.append({label: positive_counter_delta(before[label], after[label]) for label in labels}
                            if bindings else {})
        median = statistics.median(rates)
        if median < FLOOR:
            raise Refusal(f"witness median {median:.6f} is below admission floor {FLOOR:.6f}")
        if bindings and not all(all(row.values()) for row in brackets):
            raise Refusal("all-three request-window causality bracket failed")
        observed = {label: all(row[label] for row in brackets) for label in labels} if bindings else None
        print(json.dumps({
            "schema_version": 1, "grade": "ENGINEERING-EVIDENCE",
            "request_body_sha256": PAYLOAD_SHA, "warmups": 1, "scored_requests": 3,
            "automatic_retries": 0, "decode_tok_s": rates, "median_decode_tok_s": median,
            "admission_floor": FLOOR, "pass": True,
            "advisory_recheck_ruler": median > ADVISORY,
            "request_window_rank_causality": brackets if bindings else None,
            "all_three_ranks_observed": observed,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError,
            urllib.error.URLError, subprocess.TimeoutExpired, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
