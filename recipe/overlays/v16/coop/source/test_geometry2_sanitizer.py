#!/usr/bin/env python3
"""Bounded real-weight geometry-2 memory/race qualification; no serving."""
import hashlib
import json
import os

import torch
import test_cuda_integration as gate


ROWS = (2, 3, 4, 5, 8, 10, 12, 16, 18, 20, 24, 28, 32, 64)


def main():
    if os.environ.get("GLM53_COOP_QUALIFICATION") != "1" or os.environ.get("GLM53_COOP_GEOMETRY") != "2":
        raise RuntimeError("this gate requires the explicit geometry-2 qualification override")
    torch.manual_seed(20260916)
    print(json.dumps({"stage": "geometry2_sanitizer_contract", "rank": gate.EP_RANK,
                      "rows": ROWS, "production_rows": ROWS, "geometry": 2,
                      "peak": gate.PEAK_TOL, "row_peak": gate.ROW_PEAK_TOL,
                      "rel_l2": gate.REL_L2_TOL, "graph": "bitwise",
                      "native_sha256": hashlib.sha256((gate.BUNDLE / "cooperative_moe.so").read_bytes()).hexdigest(),
                      "manifest_sha256": hashlib.sha256((gate.BUNDLE / "manifest.json").read_bytes()).hexdigest()}), flush=True)
    result = gate.real_expert_layer(torch.device("cuda"), 96)
    if result is None:
        raise RuntimeError("existing real-weight snapshot unavailable")
    layer, hidden, intermediate = result
    if (hidden, intermediate) != (4096, 2048):
        raise AssertionError("wrong real-weight dimensions")
    gate.attach_map(layer)
    native = layer._glm53_coop_native
    original_launch = native.launch
    launch_count = 0

    def counted(*args):
        nonlocal launch_count
        if int(args[5]) != 2:
            raise AssertionError(f"wrong native geometry: {args[5]}")
        launch_count += 1
        return original_launch(*args)

    native.launch = counted
    try:
        for rows in ROWS:
            for pattern in ("concentrated", "mixed"):
                # Includes changed x/weights/routes, nonlocal/invalid/zero rows,
                # three graph replays and exact graph/eager equality.
                gate.run_case(layer, rows, "geometry2-real96", pattern)
        # Return to the small capacity after the 64-row specialization.
        gate.run_case(layer, 2, "geometry2-capacity-reuse", "concentrated")
    finally:
        native.launch = original_launch
    print(json.dumps({"stage": "geometry2_sanitizer_complete", "rank": gate.EP_RANK,
                      "geometry": 2, "rows": ROWS, "comparisons": len(gate.records),
                      "host_launches": launch_count, "graph_replays": 3 * (2 * len(ROWS) + 1),
                      "pass": True}), flush=True)


if __name__ == "__main__":
    main()
