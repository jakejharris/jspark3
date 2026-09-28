#!/usr/bin/env python3
"""Pre-admission prefill warmup and gate for v1.6 (S9.6).

Warmup sends one fresh prompt per mHC ``n_splits`` size class, plus a few requests with
server-default sampling and Pi-shaped turns. That compiles every per-shape TileLang/Triton
kernel before users arrive. The gate pass then replays Pi-shaped turns and a second size
sweep. It FAILs if any rank logs a kernel compile after the warmup, or if any Pi-shaped turn
(2560 reusable tokens + 2.6-3.7K new) prefills below the floor. The default floor is 1100
tok/s for the post-hygiene admission pass. First-pass canaries are recorded
only; quick is diagnostic; post-hygiene canaries remain blocking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v16_common import QAError, fetch_remote_logs, metric_delta, metrics_snapshot, sha256_json, write_json, diagnostics  # noqa: E402

COMPILE_RE = re.compile(r"TileLang begins to compile|JIT compilation during inference")
# One size per mhc compute_num_split class on a 48-SM GB10 (grid = ceil(tokens/64); split = 48 // grid).
CLASS_SIZES = (40, 100, 160, 230, 290, 350, 450, 540, 700, 900, 1300, 2000, 3000)
CLASS_EDGES = (1, 65, 129, 193, 257, 321, 385, 513, 577, 769, 1025, 1537, 2561, 4000)
WORDS = ("system state kernel cache request token model server stream deploy config branch commit review "
         "latency budget schedule prefix decode prefill memory layer expert router rank batch window epoch "
         "gate ladder quality evidence runbook boot seal worker tool result message context session answer "
         "river mountain garden window library market harbor station bridge forest valley meadow island").split()


def filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def count_tokens(base: str, model: str, text: str) -> int:
    req = urllib.request.Request(base.rstrip("/") + "/tokenize",
                                 data=json.dumps({"model": model, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return int(json.loads(diagnostics.private_read(urllib.request.urlopen(req, timeout=30)))["count"])


def sized_text(base: str, model: str, rng: random.Random, target: int) -> str:
    text = filler(rng, max(5, target))
    n = count_tokens(base, model, text)
    return filler(rng, max(5, int(len(text.split()) * target / max(n, 1))))


def stream_chat(base: str, body: dict) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    request_started_at = datetime.now(timezone.utc).isoformat()
    request_ids, finish_reasons, usage = set(), set(), None
    first = None
    raw_events = []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                raw_events.append(raw.decode("utf-8", "replace"))
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                if data.get("id"):
                    request_ids.add(diagnostics.fingerprint(data["id"]))
                if data.get("usage"):
                    usage = diagnostics.numeric_usage(data["usage"])
                for choice in data.get("choices", []):
                    if choice.get("finish_reason"):
                        if choice["finish_reason"] not in ("stop", "length", "tool_calls", "content_filter", "function_call"):
                            raise QAError("invalid finish reason: " + repr(choice))
                        finish_reasons.add(choice["finish_reason"])
                    delta = choice.get("delta", {})
                    if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                        first = time.monotonic() - t0
        return {"client_ttft_s": first, "client_e2e_s": time.monotonic() - t0,
                "request_started_at": request_started_at, "response_ended_at": datetime.now(timezone.utc).isoformat(),
                "request_ids": sorted(request_ids), "finish_reasons": sorted(finish_reasons), "usage": usage}
    finally:
        diagnostics.retain("".join(raw_events))


def measured(base: str, body: dict, tag: str) -> dict:
    metrics_before_at = datetime.now(timezone.utc).isoformat()
    before = metrics_snapshot(base)
    client = stream_chat(base, body)
    time.sleep(0.3)
    delta = metric_delta(before, metrics_snapshot(base))
    get = lambda name: sum(v for k, v in delta.items() if k.split("{", 1)[0] == "vllm:" + name)  # noqa: E731
    new = get("prefix_cache_queries_total") - get("prefix_cache_hits_total")
    prefill = get("request_prefill_time_seconds_sum")
    return {"tag": tag, **client, "request_sha256": sha256_json(body),
            "metrics_before_at": metrics_before_at, "metrics_after_at": datetime.now(timezone.utc).isoformat(),
            "metrics_delta": delta,
            "requests": get("request_success_total"), "prompt_tokens": get("prompt_tokens_total"),
            "cached_tokens": get("prefix_cache_hits_total"), "new_tokens": new, "prefill_s": round(prefill, 4),
            "prefill_tok_s": round(new / prefill, 1) if prefill else None}


_HEAD: dict[str, str] = {}


def pi_system(base: str, model: str, nonce: str) -> str:
    """~5.1K tokens; the nonce sits past the first 2560-token block (>= 2700 tokens), as in a real Pi turn."""
    if "head" not in _HEAD:
        fixed = random.Random(7)
        head = "You are an expert coding assistant operating inside pi.\n\n# Project context\n" + filler(fixed, 1900)
        while count_tokens(base, model, head) < 2700:
            head += " " + filler(fixed, 200)
        _HEAD["head"], _HEAD["tail"] = head, filler(fixed, 1500)
    return _HEAD["head"] + f"\n\n[turn-id {nonce}]\n\n# Session notes\n" + _HEAD["tail"]


def pi_turn(base: str, model: str, rng: random.Random, salt: str, turn: int) -> tuple[dict, str]:
    system = pi_system(base, model, f"{salt}-{turn}")
    new_target = rng.randint(2600, 3700)
    user_tokens = max(20, new_target - (count_tokens(base, model, system) - 2560))
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": filler(rng, user_tokens) + "\nOne sentence summary."}],
            "max_tokens": 16, "stream": True, "stream_options": {"include_usage": True}}
    return body, f"pi-turn-{turn}"


def class_body(base: str, model: str, rng: random.Random, salt: str, size: int) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": f"[{salt}-{size}] " + sized_text(base, model, rng, size)
                                          + "\nReply with one word."}],
            "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}, "cache_salt": f"{salt}-{size}"}


def compile_lines(log_text: str) -> list[str]:
    return [match.group(0) for line in log_text.splitlines() if (match := COMPILE_RE.search(line))]


def evaluate(gate_rows: list[dict], compiles: list[str], floor: float) -> list[str]:
    findings = [f"kernel compile after warmup: {line}" for line in compiles]
    for row in gate_rows:
        if row["tag"].startswith("pi-turn") and row["new_tokens"] >= 2500:
            if row["cached_tokens"] < 2560:
                findings.append(f"{row['tag']}: expected 2560 reusable tokens, got {row['cached_tokens']}")
            if not row["prefill_tok_s"] or row["prefill_tok_s"] < floor:
                findings.append(f"{row['tag']}: prefill {row['prefill_tok_s']} tok/s < floor {floor}")
    if not any(row["tag"].startswith("pi-turn") for row in gate_rows):
        findings.append("gate pass measured no Pi-shaped turns")
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--container-prefix", default="jspark3-v16-rank")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--floor-tok-s", type=float, default=1100.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    salt = f"prefill-gate-{time.time_ns()}"
    warm, gate = [], []
    try:
        for size in CLASS_SIZES:
            warm.append(measured(args.base_url, class_body(args.base_url, args.model, rng, salt + "-w", size), f"warm-{size}"))
        for i in range(2):   # server-default sampling (Pi sends no temperature) compiles the top-k/top-p kernels
            body = {"model": args.model, "messages": [{"role": "user", "content": f"[{salt}-s{i}] Say hello in a sentence."}],
                    "max_tokens": 48, "stream": True, "stream_options": {"include_usage": True}}
            warm.append(measured(args.base_url, body, f"warm-sampled-{i}"))
        for turn in range(3):
            body, tag = pi_turn(args.base_url, args.model, rng, salt + "-w", turn)
            warm.append(measured(args.base_url, body, "warm-" + tag))
        time.sleep(2)
        gate_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        time.sleep(1)
        for turn in range(args.turns):
            body, tag = pi_turn(args.base_url, args.model, rng, salt + "-g", turn)
            gate.append(measured(args.base_url, body, tag))
        for low, high in zip(CLASS_EDGES, CLASS_EDGES[1:]):
            size = rng.randint(low + 10, high - 20) if high - low > 40 else low + 10
            gate.append(measured(args.base_url, class_body(args.base_url, args.model, rng, salt + "-g", size), f"class-{size}"))
        logs, sources = fetch_remote_logs(args.env_file, args.container_prefix, since=gate_start)
        diagnostics.private_text(args.out, logs)
        compiles = compile_lines(logs)
    except (QAError, OSError, ValueError, KeyError) as exc:
        diagnostics.report_failure(exc, args.out, command='prefill gate')
        return 2
    findings = evaluate(gate, compiles, args.floor_tok_s)
    pi = [row["prefill_tok_s"] for row in gate if row["tag"].startswith("pi-turn")]
    doc = {"schema": "jspark3-v16-prefill-gate/1", "base_url": args.base_url, "gate_start": gate_start,
           "floor_tok_s": args.floor_tok_s, "log_sources": sources, "compile_lines_after_warmup": compiles,
           "pi_turn_prefill_tok_s": {"min": min(pi), "max": max(pi)} if pi else None,
           "warmup": warm, "gate": gate, "findings": findings, "verdict": "FAIL" if findings else "PASS"}
    doc["receipt_sha256"] = sha256_json(doc)
    write_json(args.out, doc)
    print(json.dumps({"verdict": doc["verdict"], "pi_turn_prefill_tok_s": doc["pi_turn_prefill_tok_s"],
                      "compiles_after_warmup": len(compiles), "findings": findings[:5]}))
    return 0 if not findings else 1


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
