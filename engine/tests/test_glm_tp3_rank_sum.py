"""Three ranks' partials are summed by TensorFold's own kernels in rank order, (p0 + p1) + p2 in fp32 per element, with
one bf16 rounding; NCCL only all-gathers them (no arithmetic). With three operands the grouping changes bits, so this is
what keeps a k-row verify window bit-equal to k one-row serial steps. Host side: the real Triton kernels run in
Triton's interpreter (TRITON_INTERPRET=1, set before triton is imported, so in a subprocess)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

CHECK = r'''
import torch
from tensorfold.families.glm5_next.cuda import glue

D, S, W = 256, 4, 3
g = torch.Generator().manual_seed(0)
bits = lambda t: t.view(torch.int16)                                          # noqa: E731


def post(parts, x, p, c):
    out = torch.empty_like(x)
    glue.hc_post(x, out, parts.contiguous(), p, c)
    return out


def resid(parts, x):
    out = torch.empty_like(x)
    glue.residual_add(x, out, parts.contiguous())
    return out


# 1. any row of a k-row window equals the same row alone (hc_post: every layer; residual_add: the MTP layer)
for R in (2, 3, 5, 8):
    parts = torch.randn(W, R, D, generator=g) * torch.logspace(-4, 4, D)       # partials across 8 decades
    x = torch.randn(R, S * D, generator=g).to(torch.bfloat16)
    p, c = torch.rand(R, S, generator=g), torch.rand(R, 16, generator=g)
    many = post(parts, x, p, c)
    xm = torch.randn(R, D, generator=g).to(torch.bfloat16)
    many_r = resid(parts, xm)
    for r in range(R):
        one = post(parts[:, r:r + 1], x[r:r + 1].clone(), p[r:r + 1].clone(), c[r:r + 1].clone())
        assert torch.equal(bits(many[r:r + 1]), bits(one)), ("hc_post", R, r)
        assert torch.equal(bits(many_r[r:r + 1]), bits(resid(parts[:, r:r + 1], xm[r:r + 1].clone()))), ("resid", R, r)

# 2. the branch is bf16((p0 + p1) + p2): the rank-order left fold in fp32, rounded once. (Triton's interpreter
#    narrows fp32 to bf16 by truncation where the GPU rounds to nearest even; the reference narrows the same way.)
trunc = lambda t: (t.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)   # noqa: E731
parts = torch.randn(W, 4, D, generator=g) * torch.logspace(-4, 4, D)
zero = torch.zeros(4, D, dtype=torch.bfloat16)
assert torch.equal(bits(resid(parts, zero)), bits(trunc((parts[0] + parts[1]) + parts[2])))

# 3. three operands whose groupings disagree under any rounding: (2^24 + 1) - 2^24 = 0, 2^24 + (1 - 2^24) = 1
a, b, c = 2.0 ** 24, 1.0, -(2.0 ** 24)
assert float((torch.tensor(a) + torch.tensor(b)) + torch.tensor(c)) == 0.0
assert float(torch.tensor(a) + (torch.tensor(b) + torch.tensor(c))) == 1.0
tri = torch.tensor([a, b, c], dtype=torch.float32).view(3, 1, 1).expand(3, 5, D).contiguous()
got = resid(tri, torch.zeros(5, D, dtype=torch.bfloat16))
assert torch.all(got == 0.0), got.unique()                                      # every row: rank 0, then 1, then 2
assert torch.equal(bits(resid(tri[:, :1].contiguous(), torch.zeros(1, D, dtype=torch.bfloat16))), bits(got[:1]))
print("OK")
'''


def test_rank_order_sum_is_the_same_for_one_row_and_k_rows():
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": f"{src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    run = subprocess.run([sys.executable, "-c", CHECK], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and run.stdout.strip().endswith("OK"), run.stdout[-2000:] + run.stderr[-4000:]
