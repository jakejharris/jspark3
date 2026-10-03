"""Read bandwidth this GPU reaches: torch sums and a Triton streaming read over 4 GB (past every cache)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _read(X, OUT, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in range(ITERS):
        off = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(X + off, mask=off < n, other=0).to(tl.float32)
    tl.store(OUT + pid, tl.sum(acc))


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3


x = torch.ones(2 * 2 ** 30, dtype=torch.bfloat16, device="cuda")        # 4 GB
nbytes = x.numel() * 2
s = timed(lambda: x.sum(dtype=torch.float32))
print(f"torch sum: {nbytes / s / 1e9:.0f} GB/s")
for block, iters, warps in ((4096, 16, 8), (8192, 8, 8), (2048, 64, 4), (16384, 4, 16)):
    grid = triton.cdiv(x.numel(), block * iters)
    out = torch.empty(grid, device="cuda")
    s = timed(lambda: _read[(grid,)](x, out, x.numel(), BLOCK=block, ITERS=iters, num_warps=warps))
    print(f"triton read block {block} x{iters} warps {warps}: {nbytes / s / 1e9:.0f} GB/s")
y = torch.empty_like(x)
s = timed(lambda: y.copy_(x))
print(f"copy (read + write): {2 * nbytes / s / 1e9:.0f} GB/s")
