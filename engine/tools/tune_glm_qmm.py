"""K slices of GLM-5.3-Flash's 4-bit decode projections: the shapes one decode forward runs (rank 0's weights, no
network), then each shape's time at 1 and 4 rows for every K split, against the one ``split_k`` picks.

    python tools/tune_glm_qmm.py RANK0_FOLDER
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from profile_glm_decode import TwoCopies  # noqa: E402


def timed(fn, reps=400) -> float:
    for _ in range(5):
        fn()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(40):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps // 40):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3            # us a call


def main() -> None:
    from tensorfold.cuda.kernels import qmm as shared
    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    e = GlmEngine(Path(sys.argv[1]), rank=0, master="", port=0, comm=TwoCopies(), context=4096)
    eng = e.e
    seen: Counter = Counter()
    real = shared.matmul

    def spy(x, q, xs=None, *, sk=None, f32=False, out=None, **kw):
        seen[(q.n, q.k, bool(f32), sk)] += 1
        return real(x, q, xs, sk=sk, f32=f32, out=out, **kw)

    shared.matmul = spy
    graphs, eng.graphs = eng.graphs, None
    eng.forward([1])
    shared.matmul = real
    print("decode projections a forward (n x k, f32 out, K slices): calls")
    for key, c in sorted(seen.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1]):
        print(f"  {key}: {c}")
    rng = np.random.default_rng(0)
    total_now = total_best = 0.0
    for (n, k, f32, sk_now), calls in sorted(seen.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1]):
        w = torch.from_numpy(rng.integers(-2**31, 2**31, (n, k // 8), dtype=np.int64).astype(np.int32)).cuda()
        s = (torch.rand(n, k // 64, device="cuda") * 0.01).bfloat16()
        b = (torch.rand(n, k // 64, device="cuda") * -0.05).bfloat16()
        q = qmm.make_q4(w, s, b)
        copies = [q] + [qmm.Q4(q.weight.clone(), q.scales.clone(), q.biases.clone(), q.n, q.k)
                        for _ in range(max(0, -(-(160 << 20) // q.nbytes()) - 1))]   # past L2: every call reads DRAM
        line = f"{n}x{k}{' f32' if f32 else ''} x{calls}:"
        for rows in (1, 4):
            x = torch.randn(rows, k, device="cuda").bfloat16()
            xs = qmm.group_sums(x)
            out = torch.empty(rows, n, device="cuda", dtype=torch.float32 if f32 else torch.bfloat16)
            res = {}
            for sk in (1, 2, 4, 8, 16):
                if (k // 64) % sk:
                    continue
                try:
                    it = iter(range(10 ** 9))
                    res[sk] = timed(lambda: real(x, copies[next(it) % len(copies)], xs, sk=sk, f32=f32, out=out))
                except Exception as exc:  # noqa: BLE001
                    res[sk] = float("inf")
            best = min(res, key=res.get)
            gbs = q.nbytes() / res[sk_now] / 1e3
            line += f"  rows {rows}: now sk {sk_now} {res[sk_now]:.1f} us ({gbs:.0f} GB/s), best sk {best} {res[best]:.1f} us"
            if rows == 1:
                total_now += res[sk_now] * calls
                total_best += res[best] * calls
        print(line, flush=True)
    print(f"a 1-row forward's projections: {total_now / 1e3:.2f} ms now, {total_best / 1e3:.2f} ms with the best K slices")
    eng.graphs = graphs


if __name__ == "__main__":
    main()
