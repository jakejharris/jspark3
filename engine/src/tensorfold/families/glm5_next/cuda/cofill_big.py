"""Co-prefill tiled passes: rank zero assigns bounded chunks; each prompt commits privately.

Every pass ends at a full-layer boundary. Express and sliced requests keep
their existing steppers; no suspended prefill scratch may be overwritten.
"""

from __future__ import annotations

import time

import torch

from . import cofill, prefill_options as p1, prof
from .decode import restore, take_snapshot
from .forward import commit, compute_streams, stage_streams
from .multi import sample_streams

FILL_BIG = 27


class Prefill:
    """A lazy, unsliced prompt continuation whose next chunk may share a pass."""

    def __init__(self, resume, checkpoints) -> None:
        self.resume, self.checkpoints = resume, checkpoints
        self.started = False

    def start(self, s) -> None:
        if self.started:
            return
        snap, self.resume = self.resume, None
        if snap is None:
            s.engine.reset()
            if s.use_dflash:
                s.drafter.reset()
        else:
            if (len(snap.ids) != s.fill_pos or len(snap.ids) >= len(s.prompt)
                    or s.prompt[:s.fill_pos] != snap.ids
                    or s.use_dflash and snap.drafter_end != s.fill_pos):
                raise RuntimeError("invalid big cofill resume prefix")
            restore(s.engine, snap, s.drafter if s.use_dflash else None)
        self.started = True

    def close(self) -> None:
        self.resume = None
        self.checkpoints = {}


def plan(decoder, first) -> list[tuple[int, int]]:
    if (not isinstance(decoder.filling.get(first.sid), Prefill)
            or any(getattr(decoder, name) is not None for name in ("fill_owner", "express_owner"))):
        return []
    # Preserve the scheduler's first choice. Equal shares let long prompts share
    # the expert pass; unused short shares flow to the following participants.
    b = first.engine.pbuf
    others = sorted((decoder.streams[sid] for sid, pending in decoder.filling.items()
                     if sid != first.sid and isinstance(pending, Prefill)
                     and decoder.streams[sid].engine.pbuf is b),
                    key=lambda s: (len(s.prompt) - s.fill_pos, s.sid))
    budget = decoder._prefill_budget(first) if getattr(decoder, "prefill_rows_idle", 0) else b.rows
    streams, remaining, chosen = [first, *others], min(b.rows, budget), []
    for i, s in enumerate(streams):
        rows = min(len(s.prompt) - s.fill_pos, remaining // (len(streams) - i))
        if rows < 1:
            continue
        stop = s.fill_pos + rows
        if decoder.sessions is not None:
            stop = decoder.sessions.stop(s, s.fill_pos, stop)
        remaining -= stop - s.fill_pos
        chosen.append((s.sid, stop))
    return chosen


@torch.no_grad()
def run(decoder, pieces: list[tuple[int, int]]) -> list:
    if (not p1.COFILL_BIG or not p1.ATTENTION_TILES or not pieces
            or len({sid for sid, _ in pieces}) != len(pieces)
            or any(getattr(decoder, name) is not None for name in ("fill_owner", "express_owner"))
            or any(not isinstance(decoder.filling.get(sid), Prefill) for sid, _ in pieces)):
        raise RuntimeError("invalid big cofill participants or suspended prefill owner")
    streams = [decoder.streams[sid] for sid, _ in pieces]
    b = streams[0].engine.pbuf
    sizes = [stop - s.fill_pos for s, (_, stop) in zip(streams, pieces)]
    if (sum(sizes) > b.rows or any(not 1 <= n <= s.engine.prefill_rows or stop > len(s.prompt)
            or s.engine.pbuf is not b or s.express or getattr(s, "slice_layers", 0) or s.use_mtp
            or s.engine.images is not None or s.done
            for s, n, (_, stop) in zip(streams, sizes, pieces))):
        raise RuntimeError("invalid big cofill shape")
    # Validate every checkpoint stop before restoring or writing any stream.
    for s, n, (_, stop) in zip(streams, sizes, pieces):
        s.fill_rows = n
        pending = decoder.filling[s.sid]
        mark = pending.checkpoints.get("mark", lambda pos: None)(s.fill_pos)
        if mark is not None and s.fill_pos < mark < stop:
            raise RuntimeError("big cofill crosses a checkpoint")
        if pending.started and s.st.pos != s.fill_pos:
            raise RuntimeError("big cofill committed position differs")
    decoder.price_rounds.contaminate()
    if not hasattr(b, "cofill_kout") or b.cofill_kout.shape[0] < b.rows:
        b.cofill_kout = torch.empty_like(b.ka)
    for s in streams:
        decoder.filling[s.sid].start(s)
    windows = [(s.st, s.prompt[s.fill_pos:stop]) for s, (_, stop) in zip(streams, pieces)]
    prof.active = True
    try:
        segs = stage_streams(decoder.w, b, windows)
        logits = compute_streams(decoder.w, segs, b, eager=True)
    finally:
        prof.active = False
    final = [i for i, (s, (_, stop)) in enumerate(zip(streams, pieces)) if stop == len(s.prompt)]
    firsts = []
    if final:
        firsts, _ = sample_streams(decoder.w, logits, [[i] for i in final],
                                  [[len(streams[i].prompt)] for i in final], [streams[i].sampling for i in final])
    for s, (_, a0, a1), (_, stop) in zip(streams, segs, pieces):
        pending = decoder.filling[s.sid]
        s.engine.last_hidden = b.fnormed[a1 - 1:a1].clone()
        if s.use_dflash:
            s.drafter.add_taps(torch.cat([t[a0:a1] for t in b.taps], dim=1))
        commit(decoder.w, s.st, b, a1 - a0, a1 - a0)
        if s.st.pos != stop:
            raise RuntimeError("big cofill committed position differs")
        mark = pending.checkpoints.get("mark", lambda pos: None)(s.fill_pos)
        if mark == stop and stop < len(s.prompt):
            pending.checkpoints["keep"](take_snapshot(s.engine, s.prompt[:stop], None, mtp=False,
                                                     drafter=s.drafter if s.use_dflash else None))
        s.fill_pos, s.fill_stop, s.fill_served = stop, stop, time.perf_counter()
        s.fill_started = True
        stats = s.cofill_stats
        stats.update(streams=len(streams), total_rows=segs[-1][2], rows=a1 - a0)
        stats["passes"] += 1
        stats["cofilled_passes"] += int(len(streams) > 1)
        stats["prompt_rows"] += a1 - a0
        stats["max_total_rows"] = max(stats["max_total_rows"], segs[-1][2])
    # All participants' target/tap state is committed before any model proposal.
    for i, first in zip(final, firsts):
        s = streams[i]
        decoder.filling.pop(s.sid).close()
        cofill.complete(decoder, s, first)
    return [s for s in streams if s.done]
