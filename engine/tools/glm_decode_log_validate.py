"""Validate decode JSONL and sampled tap metadata before using it in a price or sampled-row study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_record(record: dict) -> None:
    if record.get("observer_dropped", 0):
        raise ValueError("observer dropped receipts; rerun before qualifying timings")
    streams = record["streams"]
    if len(streams) != record["concurrency"] or len({s["sid"] for s in streams}) != len(streams):
        raise ValueError("duplicate or missing stream in round")
    if sum(s["rows"] for s in streams) != record["verify_rows"]:
        raise ValueError("verify row total differs from stream windows")
    if record["slot"] == "nonzero" and record["graph_paths"] is not None:
        raise ValueError("pooled eager round cannot claim engine graph counters")
    if record["slot"] == "zero" and record["graph_paths"] is None:
        raise ValueError("original-engine graph counters are unavailable")
    if record["path"].startswith("graph-") and record["slot"] != "zero":
        raise ValueError("graph path cannot cover a pooled slot")
    if record["path"] not in ("graph-main", "eager") and not record["path"].startswith("graph-sparse:"):
        raise ValueError("unrecognized decode path")
    for s in streams:
        if len(s["accepted_by_position"]) != s["rows"] - 1:
            raise ValueError("per-position outcomes do not match row count")


def validate_rows(record: dict, rows: list[dict]) -> None:
    seen = set()
    expected = {(s["sid"], i) for s in record["streams"] for i in range(s["rows"])}
    for row in rows:
        if row["round"] != record["round"] or row["rank"] != record["rank"]:
            raise ValueError("tap row belongs to another rank or round")
        if "tap_row" in row and not 0 <= row["tap_row"] < record["verify_rows"]:
            raise ValueError("tap row offset is outside executed window")
        key = row["sid"], row["row"]
        if key in seen:
            raise ValueError("duplicate tap row")
        seen.add(key)
    if seen != expected:
        raise ValueError("tap row stream or row ids differ from round")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    args = p.parse_args()
    records = 0
    for line in args.jsonl.read_text().splitlines():
        record = json.loads(line)
        validate_record(record)
        if "tap_file" in record:
            import torch
            payload = torch.load(args.jsonl.parent / record["tap_file"], map_location="cpu", weights_only=False)
            validate_rows(record, payload["rows"])
        records += 1
    if not records:
        raise ValueError("no decode receipts; observer may have failed")
    print(f"validated {records} decode rounds")


if __name__ == "__main__":
    main()
