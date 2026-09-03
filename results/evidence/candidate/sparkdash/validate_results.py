#!/usr/bin/env python3
"""Validate completed author-instrument receipts and extract reported values."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def parse_jsonl(relative: str):
    return [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line.startswith("{")]


def main() -> int:
    failures: list[str] = []
    idle_paths = ["idle-before-A/IDLE-GATE.json", "idle-after-A-before-B/IDLE-GATE.json", "idle-after-B-before-C/IDLE-GATE.json", "idle-after-C/IDLE-GATE.json"]
    idles = {path: load(path) for path in idle_paths}
    for path, record in idles.items():
        if record.get("verdict") != "PASS" or record.get("nonzero_delta"):
            failures.append(f"idle gate failed: {path}")

    counters = {}
    for block in ("A", "B", "C"):
        before = load(f"block-{block}/counters/before.json")
        after = load(f"block-{block}/counters/after.json")
        evaluation = after.get("window_evaluation") or {}
        if after.get("window_verdict") != "PASS":
            failures.append(f"block {block} counter hook failed")
        memory = evaluation.get("memory_checks") or []
        if len(memory) != 3 or any(row.get("swap_delta", 1) > 0 or not row.get("same_container") for row in memory):
            failures.append(f"block {block} swap/container delta failed")
        if any(node.get("oom_killed") or any(node.get("memory_events", {}).get(k, 0) for k in ("oom", "oom_kill", "oom_group_kill")) for node in after.get("memory", {}).values()):
            failures.append(f"block {block} OOM evidence")
        counters[block] = {"engine_delta": evaluation.get("engine_delta"), "memory_checks": memory, "verdict": after.get("window_verdict")}

    proxy_expect = {"A": (32, 32), "B": (37, 13), "C": (9, 9)}
    proxies = {}
    for block, (total_expected, post_expected) in proxy_expect.items():
        record = load(f"block-{block}/raw/PROXY-SUMMARY.json")
        posts = [row for row in record.get("exchanges", []) if row.get("method") == "POST"]
        if record.get("exchange_count") != total_expected or len(posts) != post_expected:
            failures.append(f"block {block} request count mismatch")
        if any(row.get("status") != 200 or row.get("error") or row.get("usage_events") != 1 for row in posts):
            failures.append(f"block {block} POST status/error/usage failure")
        if block in ("A", "B") and any(row.get("finish_reasons") != ["length"] for row in posts):
            failures.append(f"block {block} abnormal finish")
        if block == "C":
            if any(row.get("finish_reasons") != ["stop"] for row in posts[:3]) or any(row.get("finish_reasons") != ["length"] for row in posts[3:]):
                failures.append("block C abnormal finish")
        proxies[block] = {"exchange_count": record.get("exchange_count"), "post_count": len(posts), "all_post_http_200": all(row.get("status") == 200 for row in posts), "all_post_usage": all(row.get("usage_events") == 1 for row in posts)}

    sparkdash = load("block-A/SPARKDASH-RESULT.json")
    if len(sparkdash.get("jobs", [])) != 4:
        failures.append("sparkDash prompt-type count mismatch")
    spark_rows = {}
    for job in sparkdash.get("jobs", []):
        prompt_type = job["config"]["promptType"]
        if job.get("status") != "completed" or job.get("error") or len(job.get("results", [])) != 3:
            failures.append(f"sparkDash job invalid: {prompt_type}")
        spark_rows[prompt_type] = [{key: row.get(key) for key in ("concurrency", "meanDecodeTps", "medianDecodeTps", "aggregateDecodeTps", "meanTtftMs", "medianTtftMs", "streamsOk", "streamsFailed", "totalCompletionTokens")} for row in job.get("results", [])]

    mia = {phase: load(f"block-B/{phase}.json") for phase in ("structured", "prose")}
    for phase, record in mia.items():
        if record.get("health_code") != 200 or record.get("health_code_after") != 200 or len(record.get("runs", [])) != 5 or record.get("any_nan"):
            failures.append(f"Mia {phase} top-level validation failed")
        if any(row.get("http") != 200 or row.get("finish_reason") != "length" or row.get("completion_tokens") != 400 for row in record.get("runs", [])):
            failures.append(f"Mia {phase} run validation failed")

    fly = {phase: parse_jsonl(f"block-C/{phase}.stdout") for phase in ("hello", "structured", "code")}
    for phase, rows in fly.items():
        expected_finish = "stop" if phase == "hello" else "length"
        expected_tokens = 17 if phase == "hello" else 200
        if len(rows) != 3 or any(row.get("finish_reason") != expected_finish or row.get("completion_tokens") != expected_tokens for row in rows):
            failures.append(f"Fly {phase} validation failed")
    fly_engine = counters["C"]["engine_delta"]
    fly_acceptance = fly_engine["accepted_tokens"] / fly_engine["draft_tokens"] if fly_engine and fly_engine.get("draft_tokens") else None

    result = {
        "schema_version": 1,
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "idle_gates": {path: record["verdict"] for path, record in idles.items()},
        "proxies": proxies,
        "counters": counters,
        "results": {
            "sparkdash": spark_rows,
            "mia_bench_decode": {phase: {key: record.get(key) for key in ("tok_s_median", "ttft_median_s", "accept_ratio_median", "accepted_per_step_median", "completion_tokens_median")} for phase, record in mia.items()},
            "fly_t0": fly,
            "fly_global_acceptance_ratio": fly_acceptance,
        },
    }
    (ROOT / "VALIDATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
