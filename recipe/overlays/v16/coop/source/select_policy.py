"""Bind a complete real-weight geometry profile to its measured native build.

Input: nine profile_shapes.py JSONL logs (three EP ranges x three geometries).
This is component profiling, not serving or model-quality qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import manifest

ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 28, 30, 32, 35, 40, 48, 56, 64)
PATTERNS = ("ep_uniform", "concentrated")


def select(logs, native_hash):
    cases, compared, completed = {}, set(), set()
    hashes = []
    for path in logs:
        raw = Path(path).read_bytes()
        hashes.append(hashlib.sha256(raw).hexdigest())
        identities = []
        for line in raw.decode().splitlines():
            if not line.startswith("{"):
                continue
            item = json.loads(line)
            if item.get("stage") == "profile_identity":
                identities.append(item.get("native_sha256"))
            elif item.get("stage") == "compare":
                if item.get("pass") is not True or any(
                    not isinstance(item.get(key), (int, float))
                    or not math.isfinite(item[key]) or not 0 <= item[key] <= tolerance
                    for key, tolerance in (("peak_rel", .003), ("row_rel", .05), ("rel_l2", .05))
                ):
                    raise ValueError("numerical failure in profile")
                if item["label"] in compared:
                    raise ValueError("duplicate numerical comparison")
                compared.add(item["label"])
            elif item.get("stage") == "profile_complete":
                pair = (item["rank"], item["geometry"])
                if pair in completed:
                    raise ValueError("duplicate profile completion")
                completed.add(pair)
            elif item.get("stage") == "profile":
                key = (item["rank"], item["geometry"], item["rows"], item["pattern"])
                if key in cases:
                    raise ValueError("duplicate profile case")
                stock, candidate = item["stock"]["median_ms"], item["candidate"]["median_ms"]
                if not (0 < stock < float("inf") and 0 < candidate < float("inf")):
                    raise ValueError("nonfinite/nonpositive timing")
                cases[key] = candidate / stock
        if identities != [native_hash]:
            raise ValueError("profile log is not bound to this native binary")
    expected = {(r, g, n, p) for r in range(3) for g in range(3) for n in ROWS for p in PATTERNS}
    if set(cases) != expected or completed != {(r, g) for r in range(3) for g in range(3)}:
        raise ValueError("require all 468 cases and nine completed profiles")
    for r, g, n, p in expected:
        if f"profile/rank={r}/geometry={g}/rows={n}/{p}" not in compared:
            raise ValueError("missing numerical comparison")
    worst = {str(n): {str(g): max(cases[r, g, n, p] for r in range(3) for p in PATTERNS)
                     for g in range(3)} for n in ROWS}
    rows = {str(n): "stock" for n in range(1, 65)}
    for n, values in worst.items():
        geometry = min(values, key=values.get)
        if values[geometry] <= .97:
            rows[n] = int(geometry)
    return {"schema": 1, "rows": rows, "native_sha256": native_hash,
            "basis": "minimum worst-case ratio across three EP ranges and two routing patterns; require <=0.97",
            "profile_log_sha256": hashes, "worst_ratio_by_row_geometry": worst}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("logs", type=Path, nargs=9)
    args = parser.parse_args()
    # Validate the existing artifact before updating only its policy/manifest.
    original = json.loads((args.bundle / "manifest.json").read_text())
    for relative, expected in original["files"].items():
        path = (args.bundle / relative).resolve()
        if not path.is_relative_to(args.bundle.resolve()) or manifest.digest(path) != expected:
            raise ValueError(f"artifact changed: {relative}")
    policy = select(args.logs, manifest.digest(args.bundle / "cooperative_moe.so"))
    (args.bundle / "dispatch_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    manifest.create(args.bundle)
    print("Policy bound to measured build; full-model qualification remains separate.")
