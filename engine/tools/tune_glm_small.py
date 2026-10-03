"""GLM-5.3-Flash's small decode kernels, in CUDA graphs over enough copies of their weights to miss L2 (as a real step
reads 45 layers' worth): hyper-connection mixing (hc_pre) by K blocks, and the router by K slices and expert tiles.

    python tools/tune_glm_small.py
"""

from __future__ import annotations

import torch
import triton

from tensorfold.families.glm5_next.cuda import glue

D, S, E = 4096, 4, 288


def graph_us(fn, calls: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 20 * 1e3 / calls


def hc(rows: int, nb: int, sub: int, warps: int) -> float:
    calls = 90
    fns = [torch.randn(24, S * D, device="cuda").bfloat16() * 0.01 for _ in range(calls)]
    x = torch.randn(rows, S * D, device="cuda").bfloat16()
    base, scale = torch.randn(24, device="cuda"), torch.ones(3, device="cuda")
    nw = torch.ones(D, device="cuda").bfloat16()
    out = torch.empty(rows, D, device="cuda").bfloat16()
    xs = torch.empty(rows, D // 64, device="cuda")
    post, comb = torch.empty(rows, S, device="cuda"), torch.empty(rows, S * S, device="cuda")
    part = torch.empty(rows, nb, 32, device="cuda")

    def run():
        for fn in fns:
            glue._hc_partial[(rows, nb)](x, fn, part, WIDE=S * D, NB=nb, SUB=sub, num_warps=warps)
            glue._hc_finish[(rows,)](x, part, base, scale, nw, out, xs, post, comb, 1e-5, 1e-6, D=D, S=S, NB=nb,
                                     ITERS=20, BLOCK=D, num_warps=8)
    return graph_us(run, calls)


def router(rows: int, ks: int, be: int) -> float:
    calls = 42
    ws = [torch.randn(E, D, device="cuda").bfloat16() for _ in range(calls)]
    x = torch.randn(rows, D, device="cuda").bfloat16()
    out = torch.empty(rows, E, device="cuda")
    bm = 16
    part = torch.empty(ks, rows, E, device="cuda")
    total = rows * E

    def run():
        for w in ws:
            glue._router_part[(triton.cdiv(rows, bm), triton.cdiv(E, be), ks)](x, w, part, rows, x.stride(0), D=D, NE=E,
                                                                            BM=bm, BLOCK_E=be, BK=64, KS=ks,
                                                                            num_warps=4, num_stages=3)
            glue._router_sum[(triton.cdiv(total, 1024),)](part, out, total, KS=ks, BLOCK=1024, num_warps=4)
    return graph_us(run, calls)


for rows in (1, 4):
    print(f"== {rows} rows")
    for nb, sub, warps in ((16, 128, 4), (32, 128, 4), (64, 128, 4), (128, 128, 4), (64, 64, 2), (128, 64, 2)):
        us = hc(rows, nb, sub, warps)
        print(f"  hc_pre   K blocks {nb:3d} sub {sub:3d} warps {warps}: {us:5.2f} us a call, {us * 90 / 1e3:.2f} ms a step"
              f"{'  (now)' if (nb, sub, warps) == (16, 128, 4) else ''}", flush=True)
    for ks, be in ((8, 32), (16, 32), (32, 32), (16, 16), (32, 16), (64, 16)):
        us = router(rows, ks, be)
        print(f"  router   K slices {ks:3d} experts a tile {be}: {us:5.2f} us a call, {us * 42 / 1e3:.2f} ms a step"
              f"{'  (now)' if (ks, be) == (8, 32) else ''}", flush=True)
