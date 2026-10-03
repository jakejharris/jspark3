"""The all-gathers of one GLM-5.3-Flash forward between two ranks: 90 (attention and FFN of 45 layers) of R rows of
fp32 partials, eager and in a CUDA graph. Run on both machines (rank 1 first):

    python tools/bench_glm_comm.py --rank R --master ADDRESS [--port 29561]
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from tensorfold.cuda.comm import NCCL

D, CALLS = 4096, 90


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29561)
    a = p.parse_args()
    torch.cuda.set_device(0)
    comm = NCCL(a.rank, 2, a.master, a.port)
    comm.barrier()
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k not in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA",
                                                                                     "NCCL_IB_GID_INDEX")}
    for rows in (1, 4, 8):
        send = torch.randn(rows * D, device="cuda")
        recv = torch.empty(2 * rows * D, device="cuda")

        def forward() -> None:
            for _ in range(CALLS):
                comm.all_gather(send, recv)

        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            forward()
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t) / 10
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        captured = (time.perf_counter() - t) / 10
        if a.rank == 0:
            print(f"{env or 'NCCL defaults'} rows {rows}: {CALLS} all-gathers {eager * 1e3:.2f} ms eager "
                  f"({eager / CALLS * 1e6:.0f} us each), {captured * 1e3:.2f} ms in a graph "
                  f"({captured / CALLS * 1e6:.0f} us each)", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
