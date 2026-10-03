"""Qwen3.8 Flash Next's decode forward on one GPU: step times for 1..8 rows (CUDA graphs), and the CUDA kernels of a
1-row and a 4-row step by total time (eager).

    python tools/profile_flashnext_decode.py MODEL_DIR
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    eng = FlashNextEngine(Path(sys.argv[1]), max_len=8192, context_explicit=True)
    e = eng.e
    rng = np.random.default_rng(0)
    vocab = e.w.cfg.vocab
    prefill(e, [int(t) for t in rng.integers(0, vocab - 1000, 3000)], None)
    for R in range(1, 9):
        toks = [int(t) for t in rng.integers(0, vocab - 1000, R)]
        for _ in range(3):
            e.forward(toks)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            e.forward(toks)
        torch.cuda.synchronize()
        print(f"rows {R}: {(time.perf_counter() - t) / 10 * 1e3:.2f} ms", flush=True)
    from torch.profiler import ProfilerActivity, profile

    graphs, e.graphs = e.graphs, None
    for R in (1, 4):
        toks = [int(t) for t in rng.integers(0, vocab - 1000, R)]
        e.forward(toks)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                e.forward(toks)
            torch.cuda.synchronize()
        by = defaultdict(float)
        count = defaultdict(int)
        for ev in p.events():
            if ev.device_type.name == "CUDA":
                by[ev.name[:70]] += ev.device_time / 5
                count[ev.name[:70]] += 1
        total = sum(by.values())
        print(f"\n== rows {R}: GPU kernel time {total / 1e3:.2f} ms a forward (eager), "
              f"{sum(count.values()) // 5} kernels")
        for name, us in sorted(by.items(), key=lambda x: -x[1])[:24]:
            print(f"  {us / 1e3:7.3f} ms  {100 * us / total:5.1f}%  x{count[name] // 5:<4d} {name}")
    e.graphs = graphs


if __name__ == "__main__":
    main()
