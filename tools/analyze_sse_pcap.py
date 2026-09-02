#!/usr/bin/env python3
"""Extract passive localhost SSE timing and apply analyzer-v2's sustained gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def packets(path: Path):
    with path.open("rb") as handle:
        magic = handle.read(4)
        formats = {
            b"\xd4\xc3\xb2\xa1": ("<", 1_000_000), b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
            b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000), b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
        }
        if magic not in formats:
            raise ValueError("not classic pcap")
        endian, scale = formats[magic]
        rest = handle.read(20)
        linktype = struct.unpack(endian + "I", rest[-4:])[0]
        while True:
            header = handle.read(16)
            if not header:
                return
            sec, frac, captured, _original = struct.unpack(endian + "IIII", header)
            data = handle.read(captured)
            yield sec + frac / scale, linktype, data


def tcp_payload(linktype: int, data: bytes):
    if linktype == 1:
        offset = 14
        ethertype = int.from_bytes(data[12:14], "big") if len(data) >= 14 else 0
        while ethertype in {0x8100, 0x88A8} and len(data) >= offset + 4:
            ethertype = int.from_bytes(data[offset + 2:offset + 4], "big")
            offset += 4
        if ethertype != 0x0800:
            return None
    elif linktype == 113:
        if len(data) < 16 or int.from_bytes(data[14:16], "big") != 0x0800:
            return None
        offset = 16
    elif linktype == 276:
        if len(data) < 20 or int.from_bytes(data[0:2], "big") != 0x0800:
            return None
        offset = 20
    else:
        raise ValueError(f"unsupported pcap linktype {linktype}")
    if len(data) < offset + 20 or data[offset] >> 4 != 4 or data[offset + 9] != 6:
        return None
    ip_len = (data[offset] & 0x0F) * 4
    tcp = offset + ip_len
    if len(data) < tcp + 20:
        return None
    src_port = int.from_bytes(data[tcp:tcp + 2], "big")
    dst_port = int.from_bytes(data[tcp + 2:tcp + 4], "big")
    sequence = int.from_bytes(data[tcp + 4:tcp + 8], "big")
    tcp_len = (data[tcp + 12] >> 4) * 4
    payload = data[tcp + tcp_len:]
    src = data[offset + 12:offset + 16]
    dst = data[offset + 16:offset + 20]
    return src, dst, src_port, dst_port, sequence, payload


def extract(path: Path) -> list[list[int]]:
    flows: dict[tuple[bytes, bytes, int, int], list[tuple[int, float, bytes]]] = {}
    seen = set()
    for timestamp, linktype, frame in packets(path):
        row = tcp_payload(linktype, frame)
        if row is None:
            continue
        src, dst, src_port, dst_port, sequence, payload = row
        if src_port != 8000 or not payload:
            continue
        key = (src, dst, src_port, dst_port)
        identity = (key, sequence, payload)
        if identity in seen:
            continue
        seen.add(identity)
        flows.setdefault(key, []).append((sequence, timestamp, payload))
    requests: list[list[int]] = []
    for segments in flows.values():
        segments.sort(key=lambda item: item[0])
        stream = bytearray()
        times: list[int] = []
        next_sequence = None
        for sequence, timestamp, payload in segments:
            if next_sequence is not None and sequence < next_sequence:
                trim = next_sequence - sequence
                if trim >= len(payload):
                    continue
                payload = payload[trim:]
                sequence = next_sequence
            if next_sequence is not None and sequence > next_sequence:
                stream.extend(b"\n")
                times.append(int(timestamp * 1e9))
            stream.extend(payload)
            times.extend([int(timestamp * 1e9)] * len(payload))
            next_sequence = sequence + len(payload)
        cursor = 0
        current: list[int] = []
        while True:
            start = stream.find(b"data: ", cursor)
            if start < 0:
                break
            end = stream.find(b"\n", start)
            if end < 0:
                break
            raw = bytes(stream[start + 6:end]).strip()
            cursor = end + 1
            if raw == b"[DONE]":
                if len(current) >= 2:
                    requests.append(current)
                current = []
                continue
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict) and event.get("choices"):
                current.append(times[end])
        if len(current) >= 2:
            requests.append(current)
    return sorted(requests, key=lambda request: request[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("REFUSE: output exists")
    requests = extract(args.pcap)
    intervals = [[(b - a) / 1e6 for a, b in zip(ts, ts[1:]) if (b - a) / 1e6 >= 1.0] for ts in requests]
    intervals = [values for values in intervals if values]
    if not intervals:
        raise SystemExit("REFUSE: no SSE intervals extracted")
    all_values = [value for values in intervals for value in values]
    median = statistics.median(all_values)
    threshold = median + 20.0
    rows = []
    maximum = 0
    previous_end = None
    for index, (timestamps, values) in enumerate(zip(requests, intervals), 1):
        request_median = statistics.median(values)
        post_idle = previous_end is not None and (timestamps[0] - previous_end) / 1e6 >= 1000.0
        previous_end = timestamps[-1]
        run = 0
        row_max = 0
        uncompensated = 0
        for position, value in enumerate(values):
            if value <= threshold:
                run = 0
                continue
            excess = value - request_median
            before = values[position - 1] if position else None
            after = values[position + 1] if position + 1 < len(values) else None
            deficit = max(request_median - before if before is not None else 0.0,
                          request_median - after if after is not None else 0.0)
            compensated = deficit >= 0.5 * excess
            interior = not (post_idle and position < 5) and position != len(values) - 1
            if interior and not compensated:
                uncompensated += 1
                run += 1
                row_max = max(row_max, run)
                maximum = max(maximum, run)
            else:
                run = 0
        rows.append({"request": index, "visible_intervals": len(values),
                     "median_ms": request_median, "post_idle": post_idle,
                     "uncompensated_interior_count": uncompensated,
                     "max_uncompensated_interior_run": row_max})
    receipt = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "grade": "ENGINEERING-EVIDENCE",
        "source": {"pcap": str(args.pcap), "sha256": digest(args.pcap)},
        "method": "analyzer-v2 sustained-mode gate over passive localhost SSE packet timestamps",
        "request_count": len(rows),
        "visible_interval_count": len(all_values),
        "distribution": {"median_ms": median, "p95_ms": percentile(all_values, 0.95),
                         "p99_ms": percentile(all_values, 0.99), "max_ms": max(all_values)},
        "slow_threshold_ms": threshold,
        "max_uncompensated_interior_run": maximum,
        "requests": rows,
        "gate": {"sustained_mode": maximum < 5},
        "verdict": "PASS" if maximum < 5 else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": receipt["verdict"], "requests": len(rows),
                      "visible_intervals": len(all_values), "max_run": maximum}, sort_keys=True))
    return 0 if maximum < 5 else 8


if __name__ == "__main__":
    raise SystemExit(main())
