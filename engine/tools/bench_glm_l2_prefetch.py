"""One-Spark L2 prefetch microbench: a synthetic gather wait then the unchanged Q4 projection.

No NCCL latency or serving speed is inferred from this model. Run with the shared
MEASURING lock absent and a memory guard; the runner writes raw paired timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch

from tensorfold.families.glm5_next.cuda import l2_prefetch as l2, qmm


def fingerprint(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="1,6,8,32,64")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--wait-cycles", type=int, default=80000)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.n % 64 or args.k % 64 or min(args.n, args.k, args.repeats) < 1 or args.wait_cycles < 0:
        parser.error("positive group-64 shapes/repeats and a nonnegative wait are required")
    rows = [int(n) for n in args.rows.split(",")]
    if not rows or min(rows) < 1 or max(rows) > 64:
        parser.error("rows must be in 1..64")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(940)
    matrix = qmm.quantize4(torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16) * .02)
    original_weight = fingerprint(matrix.weight)
    flush = torch.empty((256 * 2**20,), device="cuda", dtype=torch.uint8)
    hint = l2.Prefetch(NS(device=torch.device("cuda:0"), layers=[]))
    report = {"kind": "synthetic-gather-wait-plus-Q4-projection", "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0), "shape": [args.n, args.k],
              "hint_bytes": min(matrix.weight.numel() * matrix.weight.element_size(), l2.MAX_BYTES),
              "wait_cycles": args.wait_cycles, "flush_bytes": flush.numel(), "rows": [],
              "started_unix": time.time(), "claims_real_collective_or_serving_speed": False}
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for count in rows:
        send = torch.randn((count, args.k), device="cuda", dtype=torch.float32) * .1
        gathered = torch.empty((3, count, args.k), device="cuda", dtype=torch.float32)
        summed = torch.empty_like(send)
        x = torch.empty_like(send, dtype=torch.bfloat16)
        out = torch.empty((count, args.n), device="cuda", dtype=torch.bfloat16)
        # The normal GLM scratch bound; every arm uses the same reduction kernel.
        part = torch.empty((8 * count * max(16384, args.n),), device="cuda", dtype=torch.float32)

        def run(enabled):
            if enabled:
                hint.pending = matrix.weight
                parent = hint.start()
            torch.cuda._sleep(args.wait_cycles)
            for rank in range(3):
                gathered[rank].copy_(send)
            if enabled:
                hint.finish(parent)
            torch.add(gathered[0], gathered[1], out=summed)
            summed.add_(gathered[2])
            x.copy_(summed)
            qmm.matmul(x, matrix, out=out, part=part)

        for enabled in (False, True):
            for _ in range(3):
                run(enabled)
        torch.cuda.synchronize()
        graphs = {}
        for enabled in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(enabled)
            graphs[enabled] = graph
        for mode in ("eager", "graph"):
            samples = {False: [], True: []}
            hashes = {}
            orders = []
            for repeat in range(args.repeats):
                order = (False, True) if repeat % 2 == 0 else (True, False)
                orders.append([int(flag) for flag in order])
                for enabled in order:
                    flush.fill_(repeat % 256)
                    start.record()
                    if mode == "graph":
                        graphs[enabled].replay()
                    else:
                        run(enabled)
                    stop.record()
                    stop.synchronize()
                    samples[enabled].append(start.elapsed_time(stop) * 1000)
                    if repeat == 0:
                        hashes[enabled] = fingerprint(out)
                if repeat == 0:
                    assert hashes[False] == hashes[True], (count, mode, "bits differ")
                    broken = out.clone()
                    broken.view(torch.uint8).flatten()[0] ^= 1
                    assert fingerprint(broken) != hashes[False], "negative control did not fail"
            plain, prefetched = (statistics.median(samples[flag]) for flag in (False, True))
            report["rows"].append({"rows": count, "mode": mode, "order": orders,
                                   "off_us": samples[False], "on_us": samples[True],
                                   "median_off_us": plain, "median_on_us": prefetched,
                                   "paired_delta_us": [b-a for a,b in zip(samples[False], samples[True])],
                                   "median_speed_ratio": plain / prefetched,
                                   "output_sha256": hashes[False], "bits_equal": True,
                                   "corruption_control_detected": True})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["rows"][-1]), flush=True)
    assert fingerprint(matrix.weight) == original_weight, "hint changed weight bytes"
    report.update(finished_unix=time.time(), weight_bytes_unchanged=True,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
