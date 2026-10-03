"""One rank of GLM-5.3-Flash's decode forward without the network (the all-gathers hand back two copies of this
rank's partials): forward times for 1..8 rows, and the CUDA kernels of a 1-row and a 4-row step by total time.

    python tools/profile_glm_decode.py RANK0_FOLDER
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class TwoCopies:
    rank, world = 0, 2

    def all_gather(self, send, recv):
        n = send.numel()
        recv.view(-1)[:n].copy_(send.reshape(-1))
        recv.view(-1)[n:2 * n].copy_(send.reshape(-1))

    def barrier(self):
        torch.cuda.synchronize()


def main() -> None:
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = GlmEngine(Path(sys.argv[1]), rank=0, master="", port=0, comm=TwoCopies(), context=8192)
    eng = e.e
    rng = np.random.default_rng(0)
    prompt = [int(t) for t in rng.integers(0, 150000, 3000)]
    prefill(eng, prompt, None)
    for R in range(1, 9):
        toks = [int(t) for t in rng.integers(0, 150000, R)]
        for _ in range(3):
            eng.forward(toks)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            eng.forward(toks)
        torch.cuda.synchronize()
        print(f"rows {R}: {(time.perf_counter() - t) / 10 * 1e3:.2f} ms (graph replays {eng.replays})", flush=True)
    from torch.profiler import ProfilerActivity, profile

    graphs, eng.graphs = eng.graphs, None                 # eager, so the profiler sees every kernel
    for R in (1, 4):
        toks = [int(t) for t in rng.integers(0, 150000, R)]
        eng.forward(toks)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                eng.forward(toks)
            torch.cuda.synchronize()
        by = defaultdict(float)
        for ev in p.events():
            if ev.device_type.name == "CUDA":
                by[ev.name[:70]] += ev.device_time / 5
        total = sum(by.values())
        print(f"\n== rows {R}: GPU kernel time {total / 1e3:.2f} ms a forward (eager)")
        for name, us in sorted(by.items(), key=lambda x: -x[1])[:22]:
            print(f"  {us / 1e3:7.3f} ms  {100 * us / total:5.1f}%  {name}")
    eng.graphs = graphs


if __name__ == "__main__":
    main()
