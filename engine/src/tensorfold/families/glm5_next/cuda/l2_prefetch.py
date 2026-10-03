"""Optional read-only weight hints alongside GLM's existing ordered gathers.

No result of this kernel enters model arithmetic. A fork before the gather and
join after it make both eager execution and CUDA graph capture own the work.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _enabled() -> bool:
    value = os.environ.get("TF_GLM_L2_PREFETCH", "0")
    if value not in ("0", "1"):
        raise ValueError("TF_GLM_L2_PREFETCH must be 0 or 1")
    return value == "1"


ENABLED = _enabled()
MAX_BYTES = int(os.environ.get("TF_GLM_L2_PREFETCH_BYTES", str(4 * 2**20)))
if not 32 <= MAX_BYTES <= 16 * 2**20:
    raise ValueError("TF_GLM_L2_PREFETCH_BYTES must be in 32..16777216")


@triton.jit
def _lines(P, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)) * 32
    address = P.to(tl.pointer_type(tl.uint8)) + offset
    # One hint per 32-byte sector; all addresses, including the tail, are valid.
    # Side effects keep the compiler from deleting a kernel with no output.
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.u32 $0, 0; }",
        constraints="=r,l,r", args=[address, (offset < N).to(tl.int32)],
        dtype=tl.int32, is_pure=False, pack=1,
    )


def tensor(weight: torch.Tensor, max_bytes: int = MAX_BYTES):
    """Hint at most max_bytes of a contiguous, already resident weight; no copy."""
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("L2 prefetch needs a contiguous CUDA weight")
    n = min(weight.numel() * weight.element_size(), max_bytes)
    if n <= 0:
        return None
    return _lines[(triton.cdiv(n, 32 * 128),)](weight, N=n, BLOCK=128, num_warps=4)


def targets(w) -> dict:
    """One next projection per gather, without predicting routed expert ids."""
    plan = {}
    for i, layer in enumerate(w.layers):
        after_attention = layer.mlp.gu.weight if layer.mlp is not None else layer.moe.router
        following = w.layers[i + 1] if i + 1 < len(w.layers) else None
        after_ffn = ((following.kda if following.kind == "kda" else following.dsa).proj.weight
                     if following is not None else w.head.weight)
        plan[layer.index] = after_attention, after_ffn
    return plan


class Prefetch:
    """One buffer owner's capture-safe stream; each gather consumes one hint."""

    def __init__(self, w) -> None:
        self.plan = targets(w)
        self.pending = None
        self.device = w.device
        self.stream = torch.cuda.Stream(device=w.device)
        self.ready, self.done = torch.cuda.Event(), torch.cuda.Event()

    def select(self, layer: int, phase: int) -> None:
        self.pending = self.plan[layer][phase]

    def start(self):
        weight, self.pending = self.pending, None
        if weight is None:
            return None
        parent = torch.cuda.current_stream(self.device)
        self.ready.record(parent)
        self.stream.wait_event(self.ready)
        try:
            with torch.cuda.stream(self.stream):
                tensor(weight)
                self.done.record(self.stream)
        except BaseException:
            # Also rejoin on a launch/compile error, before the worker fails.
            self.done.record(self.stream)
            parent.wait_event(self.done)
            raise
        return parent

    def finish(self, parent) -> None:
        if parent is not None:
            parent.wait_event(self.done)


def make(w, *, prefill: bool):
    # Both prompt and decode buffers use the shared block selectors. Overlap
    # halves share their buffer's controller: each tail consumes its hint before
    # the next half is queued. start/finish fork and join the actual tail stream.
    # MTP/drafter gathers select no target hint. Flag-off allocates no events.
    return Prefetch(w) if ENABLED and w.world > 1 else None
