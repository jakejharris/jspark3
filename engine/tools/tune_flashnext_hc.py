"""Qwen3.8 Flash Next's fused hyper-connection kernels (qmm._qmm_hcdown, qmm._qmm_upmix) by launch settings that keep
the bits (warps, stages, groups a step, column tile): time over 97 calls on weights past L2, and each setting's output
against the one the engine uses.

    python tools/tune_flashnext_hc.py
"""

from __future__ import annotations

import itertools

import torch
import triton

from tensorfold.families.qwen4_exp.cuda import qmm

S, D, R_LOW = 4, 2560, 320
K_DOWN, CALLS = S * D, 97


def rand_q4(n: int, k: int) -> qmm.Q4:
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
    s = (torch.rand(n, k // 32, device="cuda") * 0.02).bfloat16()
    b = (torch.rand(n, k // 32, device="cuda") * -0.1).bfloat16()
    return qmm.make_q4(w, s, b, layout="tiled")


def graph_us(fn) -> float:
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
    return a.elapsed_time(b) / 20 * 1e3 / CALLS


def down(rows: int):
    qs = [rand_q4(324, K_DOWN) for _ in range(CALLS)]
    h = torch.randn(rows, K_DOWN, device="cuda").bfloat16()
    pss = torch.rand(rows, 1, S, device="cuda") * D
    scale = torch.randn(K_DOWN, device="cuda").bfloat16()
    normed = torch.empty(rows, K_DOWN, device="cuda").bfloat16()
    sk = qmm.split_for(324, K_DOWN)
    per = (K_DOWN // 32) // sk
    out = torch.empty(rows, 324, device="cuda").bfloat16()
    part = torch.empty(sk * rows * 324, device="cuda")

    def run(gpi, warps, stages, bn):
        def f():
            for q in qs:
                qmm._qmm_hcdown[(triton.cdiv(rows, 16), triton.cdiv(q.n, bn), sk)](
                    h, pss, scale, normed, q.weight, q.scales, q.biases, out, part, rows, 1e-6, N=q.n, K=K_DOWN,
                    D=D, NC=1, SS=S, SK=sk, BM=16, BLOCK_N=bn, GPI=gpi, SBN=qmm.BN, num_warps=warps,
                    num_stages=stages)
        return f
    now = qmm.SHAPES16[(324, K_DOWN)]
    base = run(qmm.gpi_for(per, now[1]), now[2], now[3], now[4])
    base()
    ref = part.clone()
    print(f"== hc_down, {rows} rows (K slices {sk}, {per} groups a slice)")
    for gpi, warps, stages, bn in itertools.product((1, 2, 5, 10), (2, 4, 8), (2, 3, 4), (32, 64)):
        if per % gpi:
            continue
        f = run(gpi, warps, stages, bn)
        us = graph_us(f)
        f()
        same = torch.equal(part, ref)
        mark = "  (now)" if (gpi, warps, stages, bn) == (qmm.gpi_for(per, now[1]), now[2], now[3], now[4]) else ""
        print(f"  gpi {gpi:2d} warps {warps} stages {stages} tile {bn}: {us:6.2f} us{'' if same else '  BITS DIFFER'}{mark}")


def upmix(rows: int):
    qs = [rand_q4(S * D, R_LOW) for _ in range(CALLS)]
    act = torch.randn(rows, R_LOW, device="cuda").bfloat16()
    xs = torch.randn(rows, R_LOW // 32, device="cuda")
    normed = torch.randn(rows, S * D, device="cuda").bfloat16()
    mixed = torch.empty(rows, D, device="cuda").bfloat16()
    xsm = torch.empty(rows, D // 32, device="cuda")
    kg = R_LOW // 32

    def run(gpi, warps, stages):
        def f():
            for q in qs:
                qmm._qmm_upmix[(triton.cdiv(rows, 16), D // 32)](act, xs, q.weight, q.scales, q.biases, normed, mixed,
                                                                  xsm, rows, N=q.n, K=R_LOW, D=D, SS=S, BM=16, DB=32,
                                                                  GPI=gpi, SBN=qmm.BN, num_warps=warps,
                                                                  num_stages=stages)
        return f
    now = (qmm.gpi_for(kg, 2), 4, 3)
    run(*now)()
    ref = mixed.clone()
    print(f"== hc_upmix, {rows} rows")
    for gpi, warps, stages in itertools.product((1, 2, 5, 10), (1, 2, 4, 8), (1, 2, 3, 4)):
        if kg % gpi:
            continue
        f = run(gpi, warps, stages)
        us = graph_us(f)
        f()
        same = torch.equal(mixed, ref)
        print(f"  gpi {gpi:2d} warps {warps} stages {stages}: {us:6.2f} us{'' if same else '  BITS DIFFER'}"
              f"{'  (now)' if (gpi, warps, stages) == now else ''}")


for rows in (1, 4):
    down(rows)
    upmix(rows)
