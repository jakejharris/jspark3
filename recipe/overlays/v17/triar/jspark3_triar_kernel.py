"""One local reduction kernel. Native bf16 adds match NCCL's __hadd/__hadd2.

The interpreter branch is an explicitly rounded CPU mirror, not GPU evidence.
Keep inputs/output as int16 views: transport and comparison preserve all bits.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _add(a, b, INTERPRET: tl.constexpr):
    if INTERPRET:
        af = (a.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
        bf = (b.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
        f = af + bf
        u = f.to(tl.uint32, bitcast=True)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return tl.where((u & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, r).to(tl.uint16)
    else:
        return tl.inline_asm_elementwise(
            "add.rn.bf16 $0, $1, $2;", constraints="=h,h,h",
            args=[a.to(tl.uint16), b.to(tl.uint16)], dtype=tl.uint16,
            is_pure=True, pack=1,
        )


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def _reduce(X0, X1, X2, Labels, Out, N,
            INTERPRET: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X0 + i, mask, 0).to(tl.uint16)
    y = tl.load(X1 + i, mask, 0).to(tl.uint16)
    z = tl.load(X2 + i, mask, 0).to(tl.uint16)
    c = tl.load(Labels + i, mask, 0)
    a = tl.where(c == 0, y, x)
    b = tl.where(c == 2, y, z)
    last = tl.where(c == 0, x, tl.where(c == 1, y, z))
    result = _add(_add(a, b, INTERPRET), last, INTERPRET)
    tl.store(Out + i, result.to(tl.int16, bitcast=True), mask)


def reduce(partials, labels, output):
    n = output.numel()
    return _reduce[(triton.cdiv(n, 256),)](
        *(x.view(torch.int16) for x in partials), labels,
        output.view(torch.int16), n,
        INTERPRET=os.environ.get("TRITON_INTERPRET") == "1", BLOCK=256,
        enable_fp_fusion=False,
    )


def prepare(device):
    """Prepare the one served signature before model execution/API readiness.

    N is runtime i32 across the admitted [8, 524288] element domain. Grid is
    launch state, not a specialization. Admitted pointers are 16-byte aligned;
    dtype/options are fixed. This allocation prepares, rather than proves, coverage.
    """
    if triton.__version__ != "3.7.1":
        raise RuntimeError("TRIAR runtime-N proof requires pinned Triton 3.7.1")
    if (_reduce.params[5].is_constexpr or not _reduce.params[5].do_not_specialize
            or not _reduce.params[5].do_not_specialize_on_alignment):
        raise RuntimeError("TRIAR N must remain unspecialized at runtime")
    probe = torch.zeros(8, dtype=torch.bfloat16, device=device)
    labels = torch.zeros(8, dtype=torch.uint8, device=device)
    reduce([probe] * 3, labels, torch.empty_like(probe))
    torch.cuda.current_stream(device).synchronize()
    return {"triton": triton.__version__, "N_runtime": True,
            "minimum_n": 8, "maximum_n": 524288, "multiple_n": 8}
