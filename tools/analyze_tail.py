#!/usr/bin/env python3
"""Compensation-aware inter-token pacing analyzer, calibrated to a matched control battery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


SINGLE_PHASES = {"warmup", "c1-prose", "c1-code", "c1-count", "legacy-control"}
MIN_STEP_MS = 1.0
SLOW_MARGIN_MS = 20.0
IDLE_GAP_MS = 1000.0
POST_IDLE_STEPS = 5


class Refusal(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {
        "count": len(values),
        "median_ms": median,
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "max_ms": max(values),
    }


def load_requests(run_dir: Path) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    for path in sorted((run_dir / "results").glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("phase") not in SINGLE_PHASES:
            continue
        timestamps = [int(row["timestamp_ns"]) for row in result["sse_event_receipts"]]
        steps = [
            {"ms": (right - left) / 1_000_000, "end_ns": right}
            for left, right in zip(timestamps, timestamps[1:])
            if (right - left) / 1_000_000 >= MIN_STEP_MS
        ]
        if not steps:
            raise Refusal(f"{path.name}: no visible step intervals")
        delta_path = run_dir / "counters" / f"{result['request_id']}.delta.json"
        reconciliation = None
        if delta_path.is_file():
            delta = json.loads(delta_path.read_text())
            iterations = int(delta["metrics"]["engine"]["draft_iterations"])
            difference = len(steps) - iterations
            reconciliation = {
                "draft_iterations": iterations,
                "visible_steps": len(steps),
                "difference": difference,
            }
            if difference not in {0, -1}:
                raise Refusal(
                    f"{result['request_id']}: {len(steps)} visible SSE steps "
                    f"do not reconcile to {iterations} draft iterations"
                )
        requests.append({
            "request_id": result["request_id"],
            "phase": result["phase"],
            "workload": result.get("workload"),
            "start_ns": timestamps[0],
            "last_ns": timestamps[-1],
            "steps": steps,
            "reconciliation": reconciliation,
        })
    if len(requests) != 15:
        raise Refusal(f"expected 15 single-stream requests, observed {len(requests)}")
    return sorted(requests, key=lambda row: int(row["start_ns"]))


def analyze(run_dir: Path) -> dict[str, object]:
    terminal = json.loads((run_dir / "RUN.json").read_text())
    matrix = json.loads((run_dir / "MATRIX.json").read_text())
    if terminal.get("status") != "COMPLETE" or matrix.get("fatal_error") is not None:
        raise Refusal("matrix is not terminal and error-free")
    if not matrix.get("functional_scored_checks_all_pass"):
        raise Refusal("functional scored checks did not all pass")
    if not matrix.get("counter_brackets_all_pass"):
        raise Refusal("counter brackets did not all pass")

    requests = load_requests(run_dir)
    all_steps = [
        float(step["ms"])
        for request in requests
        for step in request["steps"]  # type: ignore[index]
    ]
    dist = distribution(all_steps)
    slow_threshold = float(dist["median_ms"]) + SLOW_MARGIN_MS
    slow_count = 0
    uncompensated_interior_count = 0
    max_uncompensated_interior_run = 0
    spike_count = 0
    request_rows: list[dict[str, object]] = []
    previous_last_ns: int | None = None

    for request in requests:
        steps = request["steps"]  # type: ignore[assignment]
        values = [float(step["ms"]) for step in steps]
        request_median = statistics.median(values)
        start_ns = int(request["start_ns"])
        idle_gap_ms = None if previous_last_ns is None else (start_ns - previous_last_ns) / 1_000_000
        post_idle = idle_gap_ms is not None and idle_gap_ms >= IDLE_GAP_MS
        previous_last_ns = int(request["last_ns"])
        run = 0
        row_slow = 0
        row_compensated = 0
        row_uncompensated_interior = 0
        row_max_run = 0
        post_idle_slow = 0
        post_idle_max_excess = 0.0
        row_spikes = 0

        for index, value in enumerate(values):
            if value >= 250.0:
                spike_count += 1
                row_spikes += 1
            if value <= slow_threshold:
                run = 0
                continue
            slow_count += 1
            row_slow += 1
            excess = value - request_median
            previous_value = values[index - 1] if index > 0 else None
            next_value = values[index + 1] if index + 1 < len(values) else None
            deficit = max(
                request_median - previous_value if previous_value is not None else 0.0,
                request_median - next_value if next_value is not None else 0.0,
            )
            compensated = deficit >= 0.5 * excess
            if compensated:
                row_compensated += 1
            interior = not (post_idle and index < POST_IDLE_STEPS) and index != len(values) - 1
            if interior and not compensated:
                uncompensated_interior_count += 1
                row_uncompensated_interior += 1
                run += 1
                row_max_run = max(row_max_run, run)
                max_uncompensated_interior_run = max(max_uncompensated_interior_run, run)
            else:
                run = 0

        # A post-idle ramp is a leading contiguous slow sequence, not an
        # isolated delivery spike that happens to land among the first five
        # positions. Other isolated events remain covered by the tail and
        # >=250 ms gates.
        if post_idle:
            for value in values[:POST_IDLE_STEPS]:
                if value <= slow_threshold:
                    break
                post_idle_slow += 1
                post_idle_max_excess = max(post_idle_max_excess, value - request_median)

        request_rows.append({
            "request_id": request["request_id"],
            "phase": request["phase"],
            "workload": request["workload"],
            "step_count": len(values),
            "request_median_ms": request_median,
            "idle_gap_ms": idle_gap_ms,
            "post_idle": post_idle,
            "slow_count": row_slow,
            "compensated_slow_count": row_compensated,
            "uncompensated_interior_count": row_uncompensated_interior,
            "max_uncompensated_interior_run": row_max_run,
            "post_idle_leading_ramp_count": post_idle_slow,
            "post_idle_leading_ramp_max_excess_ms": post_idle_max_excess,
            "spike_count_ge_250ms": row_spikes,
            "reconciliation": request["reconciliation"],
        })

    return {
        "source": {
            "run_dir": str(run_dir),
            "RUN.json": digest(run_dir / "RUN.json"),
            "MATRIX.json": digest(run_dir / "MATRIX.json"),
            "REQUESTS.json": digest(run_dir / "REQUESTS.json"),
        },
        "distribution": dist,
        "slow_threshold_ms": slow_threshold,
        "slow_count": slow_count,
        "slow_fraction": slow_count / len(all_steps),
        "uncompensated_interior_count": uncompensated_interior_count,
        "uncompensated_interior_fraction": uncompensated_interior_count / len(all_steps),
        "max_uncompensated_interior_run": max_uncompensated_interior_run,
        "spike_count_ge_250ms": spike_count,
        "requests": request_rows,
        "rates": {
            key: value["median_rate"] for key, value in matrix["rate_phases"].items()
        },
        "matrix_flags": {
            "fatal_error": matrix["fatal_error"],
            "functional_scored_checks_all_pass": matrix["functional_scored_checks_all_pass"],
            "counter_brackets_all_pass": matrix["counter_brackets_all_pass"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("REFUSE: output exists")
    try:
        candidate = analyze(args.run_dir.resolve())
        control = analyze(args.control_run_dir.resolve())
    except (OSError, KeyError, ValueError, Refusal) as error:
        receipt = {
            "schema_version": 2,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "grade": "ENGINEERING-EVIDENCE",
            "arm": args.arm,
            "run_id": args.run_id,
            "verdict": "REFUSE_INADMISSIBLE",
            "reason": str(error),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True))
        return 9

    candidate_dist = candidate["distribution"]
    control_dist = control["distribution"]
    assert isinstance(candidate_dist, dict) and isinstance(control_dist, dict)
    requests = candidate["requests"]
    assert isinstance(requests, list)
    post_idle_rows = [row for row in requests if row["post_idle"]]
    gates = {
        "sustained_mode": candidate["max_uncompensated_interior_run"] < 5,
        "post_idle_ramp": all(
            row["post_idle_leading_ramp_count"] <= 5
            and row["post_idle_leading_ramp_max_excess_ms"] <= 60.0
            for row in post_idle_rows
        ),
        "slow_fraction": candidate["slow_fraction"] <= 0.034,
        "uncompensated_interior_fraction": candidate["uncompensated_interior_fraction"] <= 0.010,
        "spike_count_ge_250ms": candidate["spike_count_ge_250ms"] <= 2,
        "median": candidate_dist["median_ms"] <= control_dist["median_ms"],
        "p95": candidate_dist["p95_ms"] <= 1.10 * control_dist["p95_ms"],
        "p99": candidate_dist["p99_ms"] <= 1.15 * control_dist["p99_ms"],
        "mad": candidate_dist["mad_ms"] <= control_dist["mad_ms"] + 1.0,
    }
    passed = all(gates.values())
    receipt = {
        "schema_version": 2,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "grade": "ENGINEERING-EVIDENCE",
        "arm": args.arm,
        "run_id": args.run_id,
        "method": {
            "step_min_ms": MIN_STEP_MS,
            "slow_step": "candidate own single-stream median + 20 ms",
            "compensated": "adjacent interval deficit >= half slow-step excess over request median",
            "post_idle": "leading contiguous slow steps, capped to first five, after >=1 second request gap",
            "interior": "not a post-idle first-five step and not the final request step",
        },
        "candidate": candidate,
        "matched_control": control,
        "gates": gates,
        "calibration_only": args.calibration_only,
        "verdict": "CALIBRATION" if args.calibration_only else ("PASS" if passed else "FAIL"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "verdict": receipt["verdict"],
        "gates": gates,
        "candidate_distribution": candidate_dist,
        "slow_fraction": candidate["slow_fraction"],
        "uncompensated_interior_fraction": candidate["uncompensated_interior_fraction"],
        "max_uncompensated_interior_run": candidate["max_uncompensated_interior_run"],
        "spike_count_ge_250ms": candidate["spike_count_ge_250ms"],
    }, sort_keys=True))
    return 0 if args.calibration_only or passed else 8


if __name__ == "__main__":
    sys.exit(main())
