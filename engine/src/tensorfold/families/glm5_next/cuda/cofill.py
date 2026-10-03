"""Finish several small, text-only EXL3 appends in one prefill forward.

This pass is atomic: it never runs while either prefill lane owns a suspended
chunk. With the express lane on, only atomic express prompts (at most
TF_GLM_EXPRESS_ATOMIC_ROWS uncached rows, TF_GLM_EXPRESS_COFILL=1) cofill, on the
express buffer, and the pass is capped at that many rows while a reply decodes;
sliced prompts keep their stepper. MTP/image requests and partial/large chunks
retain the ordinary stepper. The same prefill buffers/math are used; each stream
retains its own KDA chain, cache and head row.
"""

from __future__ import annotations

import os
import time

import torch

from .decode import restore
from .forward import commit, compute_streams, stage_streams
from .multi import sample_streams
from . import prefill_options as p1

ADMIT_COFILL, ADMIT_COPY_COFILL, FILL_COFILL = 23, 24, 25
EXPRESS_IDLE_ROWS = 1023


def enabled(stream) -> bool:
    value = getattr(stream, "cofill", None)
    return (os.environ.get("TF_GLM_COFILL", "0") == "1" or p1.COFILL_BIG) if value is None else value


def complete(decoder, s, first: list[int]) -> None:
    """Finish a fully committed prompt after sampling its own last head row."""
    s.engine.images = None
    snapshot = decoder._remember(s) if s.draft else None
    if decoder.sessions is not None:
        decoder.sessions.completed(s, snapshot)
    if s.segment_tracker is not None:
        s.segment = s.segment_tracker.feed(first)
    decoder._propose(s, first[0], s.engine.last_hidden, first, s.count - 1)
    s.prefill_s, s.started = time.perf_counter() - s.prefill_started, time.perf_counter()
    s.take(first, s.eos)


def decoding(decoder) -> bool:
    return any(not s.done and s.sid not in decoder.filling for s in decoder.streams.values())


def candidates(decoder, first) -> list:
    express = getattr(first, "express", False)
    if (any(getattr(decoder, name, None) is not None for name in ("fill_owner", "express_owner"))
            or express and not getattr(decoder, "express_cofill", False)
            or not getattr(first, "cofill_ready", False)):
        return []
    # Keep the scheduler's first choice; no waiting, no new admission fairness.
    available = [first] + sorted((decoder.streams[sid] for sid in decoder.filling if sid != first.sid),
                                 key=lambda s: (len(s.prompt) - s.fill_pos, s.sid))
    chosen, total = [], 0
    budget = min(2048, first.engine.prefill_rows)
    if express:
        # Idle: stay below the g64 four-warp gate (R >= 1024), so a cofilled row
        # runs the expert kernel it would alone. Decoding: bound the incumbents' pause.
        busy = min(decoder.express_atomic_rows, getattr(decoder, "express_busy_rows", decoder.express_atomic_rows))
        budget = min(budget, busy if decoding(decoder) else EXPRESS_IDLE_ROWS)
    for s in available:
        n = len(s.prompt) - s.fill_pos
        if (getattr(s, "cofill_ready", False) and getattr(s, "express", False) == express
                and not getattr(s, "slice_layers", 0) and n <= s.fill_rows and total + n <= budget
                and s.engine.pbuf is first.engine.pbuf):
            chosen.append(s)
            total += n
    return chosen if len(chosen) >= 2 else []


@torch.no_grad()
def run(decoder, sids: list[int]) -> list:
    if (len(sids) < 2 or len(sids) != len(set(sids))
            or any(getattr(decoder, name, None) is not None for name in ("fill_owner", "express_owner"))
            or any(sid not in decoder.filling for sid in sids)):
        raise RuntimeError("invalid cofill participants or suspended prefill owner")
    streams = [decoder.streams[sid] for sid in sids]
    decoder.price_rounds.contaminate()
    b = streams[0].engine.pbuf
    sizes = [len(s.prompt) - s.fill_pos for s in streams]
    if (sum(sizes) > min(2048, b.rows) or any(not getattr(s, "cofill_ready", False) or s.engine.pbuf is not b
            or getattr(s, "express", False) != getattr(streams[0], "express", False)
            or getattr(s, "express", False) and not getattr(decoder, "express_cofill", False)
            or getattr(s, "slice_layers", 0)
            or not 1 <= n <= s.fill_rows for s, n in zip(streams, sizes))):
        raise RuntimeError("invalid cofill shape")
    # Allocate only when this default-off path is actually used. KDA's shared
    # scratch output is reused per stream, so each segment needs a separate copy.
    if not hasattr(b, "cofill_kout"):
        b.cofill_kout = torch.empty_like(b.ka[:min(2048, b.rows)])
    for s in streams:
        decoder.filling.pop(s.sid).close()  # all participants are unstarted generators
        s.cofill_ready = False
        snap, s.cofill_snap = s.cofill_snap, None
        if snap is None:
            s.engine.reset()
            if s.use_dflash:
                s.drafter.reset()
        else:
            restore(s.engine, snap, s.drafter if s.use_dflash else None)
        del snap
    windows = [(s.st, s.prompt[s.fill_pos:]) for s in streams]
    segs = stage_streams(decoder.w, b, windows)
    logits = compute_streams(decoder.w, segs, b, eager=True)
    firsts, _ = sample_streams(decoder.w, logits, [[i] for i in range(len(streams))],
                              [[len(s.prompt)] for s in streams], [s.sampling for s in streams])
    for i, (s, (_, a0, a1), first) in enumerate(zip(streams, segs, firsts)):
        s.engine.last_hidden = b.fnormed[a1 - 1:a1].clone()
        if p1.EXACT_REPLAY and not s.use_mtp:
            s.engine.last_logits = logits[i:i + 1].clone()   # the head row sample_streams just read for this stream
        if s.use_dflash:
            s.drafter.add_taps(torch.cat([t[a0:a1] for t in b.taps], dim=1))
        commit(decoder.w, s.st, b, a1 - a0, a1 - a0)
        s.fill_pos = len(s.prompt)
        s.cofill_stats = {"streams": len(streams), "total_rows": segs[-1][2], "rows": a1 - a0}
        complete(decoder, s, first)
    return [s for s in streams if s.done]
