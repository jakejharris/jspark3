"""MTP reads final-normed main rows, chains its own shared_head.norm output, zeros the position-0 embedding, and trims draft cache entries on absorb."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import glue, qmm
from .forward import Buffers, State, check_room, dsa_block, mm, moe_block
from .weights import Weights


def mtp_stage(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor) -> int:
    """Host work before an MTP step: the next tokens and the input hidden rows into the static buffers."""

    return mtp_stage_streams(w, b, [(st, next_tokens, hidden)])[-1][2]


def mtp_stage_streams(w: Weights, b: Buffers, windows) -> list:
    """Several streams' MTP rows (state, next tokens, hidden rows) into consecutive rows -> segments (state, a0, a1)."""

    segs, ids, a0, zero = [], [], 0, []
    for st, tokens, _ in windows:
        check_room(w, st, len(tokens), pos=st.mtp_len)
        if st.mtp_len == 0:
            zero.append(a0)
        segs.append((st, a0, a0 + len(tokens)))
        ids.extend(tokens)
        a0 += len(tokens)
    b.zero_rows = zero
    b.zero_first = bool(zero) and zero[0] == 0
    b.staged.synchronize()
    host = b.ids_host[:a0].numpy()
    host[:] = ids
    np.copyto(host, w.cfg.image_token, where=host < 0)
    b.ids[:a0].copy_(b.ids_host[:a0], non_blocking=True)
    for (st, tokens, hidden), (_, s0, s1) in zip(windows, segs):
        if hidden.data_ptr() != b.hin[s0:s1].data_ptr():
            b.hin[s0:s1].copy_(hidden)
    b.staged.record()
    return segs


def mtp_caches(segs, eager: bool) -> list:
    """The MTP layer's cache views of each stream (host positions only in eager steps)."""
    return [(st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, st.index[-1] if st.index is not None else None,
             st.mtp_len if eager else None, a0, a1) for st, a0, a1 in segs]


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, *, last_only: bool = True,
                nch: int | None = None, host_pos: int | None = None, sparse_np: int | None = None) -> torch.Tensor:
    """The MTP head's GPU work on staged rows (capturable)."""

    return mtp_compute_streams(w, [(st, 0, n)], b, last=[n - 1] if last_only else None, nch=nch,
                               eager=host_pos is not None, sparse_np=sparse_np)


def mtp_compute_streams(w: Weights, segs, b: Buffers, *, last: list[int] | None = None, nch: int | None = None,
                        eager: bool = True, sparse_np: int | None = None) -> torch.Tensor:
    """The MTP head on every segment's rows at once; logits of the rows ``last`` (all rows when None), in that order."""

    c = w.cfg
    m = w.mtp
    D = c.hidden
    n = segs[-1][2]
    glue.embed(b.ids[:n], w.embed, D, 1, b.me[:n])
    for r in b.zero_rows:
        b.me[r].zero_()
    if b.overlay is not None:
        rows, emb = b.overlay
        b.me[rows] = emb
    glue.rmsnorm(b.me[:n], m.enorm, c.eps, b.mcat[:n, :D])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.eps, b.mcat[:n, D:])
    mm(b, b.mcat[:n], m.eh, None if b.prefill else qmm.group_sums(b.mcat[:n], b.mxs[:n]), b.mx[:n])
    layer = m.layer
    glue.rmsnorm(b.mx[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    g = dsa_block(layer, w, mtp_caches(segs, eager), b, n, nch, sparse_np)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    glue.rmsnorm(b.mx[:n], layer.post_norm, c.eps, b.normed[:n], b.xs[:n])
    g = moe_block(layer, w, b, n)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    if last is None:
        rows, k = b.mx[:n], n
    elif len(last) == 1 or last == list(range(last[0], last[0] + len(last))):
        rows, k = b.mx[last[0]:last[0] + len(last)], len(last)
    else:
        k = len(last)
        rows = b.mx[:n].index_select(0, torch.tensor(last, device=b.mx.device))
    glue.rmsnorm(rows, m.norm, c.eps, b.fnormed[:k], b.fxs[:k])
    head = w.draft_head if w.draft_head is not None else w.head        # the head's logits only draft
    return mm(b, b.fnormed[:k], head, b.fxs[:k], b.logits[:k, :w.head.n])


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Write hidden/token rows into cache slots mtp_len onward and expose logits and b.mx; the caller advances st.mtp_len."""

    n = mtp_stage(w, st, b, next_tokens, hidden)
    from .attention import CHUNK

    return mtp_compute(w, st, b, n, last_only=last_only, nch=-(-(st.mtp_len + n) // CHUNK), host_pos=st.mtp_len)
