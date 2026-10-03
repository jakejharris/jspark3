#!/usr/bin/env python3
"""Manually invoked forced cancellation gate. No hook runs in the normal server.

Launch each rank with `serve --gate-dir DIR -- <its existing tensorfold args>`.
Run `capture` in rank zero's container, sharing DIR, against a supplied
prefix/warm request pair. Specify its base and candidate token counts and expected token hash. Use SESSION_HASH_GATE=1 on all ranks. These are correctness
receipts, never latency measurements. The user owns isolation/resources.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


class GateMismatch(ValueError):
    """A valid forced case disagreed with its control."""


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def publish_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, value)
    os.link(tmp, path)  # atomic visibility and refusal to overwrite a receipt
    tmp.unlink()


def install_hook(directory, timeout=30):
    from tensorfold.families.glm5_next.cuda import batched, decode

    directory.mkdir(parents=True, exist_ok=True)
    original = batched.BatchedDecoder._fill

    def fill(self, sid, stop, layer_stop=0):
        result = original(self, sid, stop, layer_stop)
        stream = self.streams.get(sid)
        if (self.rank != 0 or stream is None or not getattr(stream, "reply_prefill", False)
                or stream.done or not stream.fill_layer or len(set(stream.st.cur)) < 2):
            return result
        arm_path = directory / "arm.json"
        if not arm_path.exists():
            return result
        arm = json.loads(arm_path.read_text())
        trial = arm["trial"]
        if not isinstance(trial, str) or re.fullmatch(r"\d{3}", trial) is None:
            raise ValueError("invalid preemption gate trial")
        ready = directory / f"{trial}-ready.json"
        if (ready.exists() or stream.reply_base != arm["base"] or len(stream.prompt) != arm["candidate"]
                or stream.fill_pos != stream.cached or stream.fill_layer < arm.get("min_layer", 1)):
            return result
        record = {"trial": trial, "sid": sid, "base": stream.reply_base, "candidate": len(stream.prompt),
                  "cached": stream.cached, "fill_pos": stream.fill_pos, "layer": stream.fill_layer,
                  "selectors": list(stream.st.cur), "already_cancelled": stream.reply_cancel.is_set(),
                  "decode_source_sha256": hashlib.sha256(Path(decode.__file__).read_bytes()).hexdigest()}
        publish_json(ready, record)
        # Releases the GIL. HTTP preparation/submit can set this event while the
        # worker is held at an actual mixed-bank layer boundary on every rank.
        cancelled = stream.reply_cancel.wait(timeout)
        publish_json(directory / f"{trial}-released.json", {**record, "cancelled": cancelled})
        return result

    batched.BatchedDecoder._fill = fill


def fingerprint(reply):
    stats = reply["tensorfold"]
    ids, states = stats.get("token_ids"), stats.get("session_state_sha256")
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError("full output token IDs are required")
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if stats.get("token_sha") != digest[:12]:
        raise ValueError("token IDs disagree with token_sha")
    if (not isinstance(states, list) or len(states) != 3
            or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in states)):
        raise ValueError("SESSION_HASH_GATE=1 is required on all three ranks")
    if stats.get("policy") != "fc7:0.3" or stats.get("drafts") is not True:
        raise ValueError("the frozen gate requires explicit fc7:0.3")
    return {"token_sha256": digest, "state_sha256": states, "tokens": len(ids)}


def check_case(control, actual, ready, released, *, base, candidate):
    if (ready.get("already_cancelled") is not False or released.get("cancelled") is not True
            or ready.get("trial") != released.get("trial") or ready.get("selectors") != released.get("selectors")
            or len(set(ready.get("selectors", []))) != 2 or set(ready["selectors"]) != {0, 1}
            or ready.get("base") != base or ready.get("candidate") != candidate
            or ready.get("cached") != base or ready.get("fill_pos") != base or ready.get("layer", 0) <= 0):
        raise ValueError("the requested mixed-bank preemption did not occur")
    for reply in (control, actual):
        stats = reply["tensorfold"]
        if (stats.get("cached") != base or stats.get("session_cache_source") != "memory"
                or stats.get("reply_prefill_hit_tokens") != 0):
            raise ValueError("warm request did not use the original memory prefix")
    want, got = fingerprint(control), fingerprint(actual)
    if got != want or actual["tensorfold"]["token_ids"] != control["tensorfold"]["token_ids"]:
        raise GateMismatch("forced preemption changed output IDs or canonical state")
    return got


def wait_json(path, timeout):
    until = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= until:
            raise TimeoutError(f"missing gate receipt: {path.name}")
        time.sleep(0.01)
    # Hook publishes the ready file atomically; release is complete by
    # the time the subsequent warm response is returned.
    return json.loads(path.read_text())


def capture(args):
    def body(path, reply_prefill):
        raw = json.loads(path.read_text())
        request = dict(raw.get("request", raw))
        request.update(stream=False, return_token_ids=True, tf_policy="fc7:0.3", tf_reply_prefill=reply_prefill)
        return request

    prefix_off, prefix_on = body(args.prefix, False), body(args.prefix, True)
    warm = body(args.warm, False)
    if not 1 <= args.repeats <= 999 or args.candidate_tokens <= args.base_tokens:
        raise ValueError("invalid repeat count or prefix sizes")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "repeats": args.repeats, "requests": 2 + 2 * args.repeats,
                          "base_tokens": args.base_tokens, "candidate_tokens": args.candidate_tokens,
                          "writes_receipts": False, "requires_state_hashes": True}))
        return
    if not args.gate_dir.is_dir() or any(args.gate_dir.iterdir()):
        raise ValueError("gate directory must exist and be empty for this cohort")
    args.out.mkdir(parents=True, exist_ok=False)
    requests = 0
    results = []

    def post(request, name):
        nonlocal requests
        payload = json.dumps(request).encode()
        write_json(args.out / f"{name}-request.json", request)
        req = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions", payload,
                                     {"Content-Type": "application/json"})
        requests += 1
        with urllib.request.urlopen(req, timeout=600) as response:
            reply = json.load(response)
        write_json(args.out / f"{name}.json", reply)
        return reply

    def metrics():
        with urllib.request.urlopen(args.base.rstrip("/") + "/metrics", timeout=10) as response:
            return {k: float(v) for k, v in (line.split() for line in response.read().decode().splitlines()
                                            if line and not line.startswith("#"))}

    before = None
    try:
        before = metrics()
        if any(before.get(key, 0) for key in ("tensorfold:running", "vllm:num_requests_waiting", "tensorfold:inflight")):
            raise ValueError("server is not idle")
        if "process_start_time_seconds" not in before:
            raise ValueError("missing process identity metric")
        post(prefix_off, "control-prefix")
        control = post(warm, "control-warm")
        if fingerprint(control)["token_sha256"][:12] != args.expected_token_sha:
            raise GateMismatch("Reply-prefill-off control does not match the supplied expected token hash")
        for index in range(args.repeats):
            trial = f"{index:03}"
            arm = {"trial": trial, "base": args.base_tokens, "candidate": args.candidate_tokens,
                   "min_layer": args.min_layer}
            tmp = args.gate_dir / "arm.tmp"
            write_json(tmp, arm)
            tmp.replace(args.gate_dir / "arm.json")
            first = post(prefix_on, f"{trial}-prefix")
            prediction = first["tensorfold"].get("reply_prefill", {})
            if prediction.get("status") != "queued" or prediction.get("candidate_tokens") != args.candidate_tokens:
                raise ValueError("prefix did not queue the specified reply prefill extension")
            ready = wait_json(args.gate_dir / f"{trial}-ready.json", 30)
            actual = post(warm, f"{trial}-warm")
            released = wait_json(args.gate_dir / f"{trial}-released.json", 5)
            try:
                hashes = check_case(control, actual, ready, released,
                                    base=args.base_tokens, candidate=args.candidate_tokens)
            except GateMismatch:
                results.append({"trial": trial, "pass": False, "ready": ready, "released": released})
                raise
            results.append({"trial": trial, "pass": True, **hashes, "ready": ready, "released": released})
            print(json.dumps({"trial": trial, "pass": True, "layer": ready["layer"]}), flush=True)
    except GateMismatch as exc:
        (args.out / "FAIL.md").write_text(str(exc) + "\n")
        raise
    except BaseException as exc:
        (args.out / "INVALID.md").write_text(f"Incomplete/invalid cohort: {type(exc).__name__}\n")
        raise
    finally:
        valid = False
        try:
            if before is not None:
                after = metrics()
                audit = {"expected_requests": requests,
                         "actual_requests": after["tensorfold:requests_total"] - before["tensorfold:requests_total"],
                         "same_process": after.get("process_start_time_seconds") == before.get("process_start_time_seconds")}
                valid = audit["actual_requests"] == requests and audit["same_process"]
                write_json(args.out / "traffic.json", dict(audit, valid=valid))
                if not valid:
                    raise ValueError("traffic or process accounting mismatch")
        except Exception:
            (args.out / "INVALID.md").write_text("Traffic/process audit failed.\n")
            raise
        finally:
            write_json(args.out / "result.json", {"pass": valid and len(results) == args.repeats and all(r["pass"] for r in results),
                                                  "scope": "forced timing correctness; no performance claim",
                                                  "repetitions": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--gate-dir", type=Path, required=True)
    serve.add_argument("server_args", nargs=argparse.REMAINDER)
    run = commands.add_parser("capture")
    run.add_argument("--base", default="http://127.0.0.1:8002")
    run.add_argument("--gate-dir", type=Path, required=True)
    run.add_argument("--prefix", type=Path, required=True)
    run.add_argument("--warm", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--base-tokens", type=int, required=True)
    run.add_argument("--candidate-tokens", type=int, required=True)
    run.add_argument("--min-layer", type=int, default=1)
    run.add_argument("--expected-token-sha", required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        install_hook(args.gate_dir)
        from tensorfold.cli import main as server
        return server(args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args)
    capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
