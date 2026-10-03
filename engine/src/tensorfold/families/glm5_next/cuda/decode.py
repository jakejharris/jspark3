"""CUDA decode accepts drafts only when they match the serial keyed sample; all ranks sample identical gathered candidates without a broadcast."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import glue, prefill_options as p1, prof, qmm
from .forward import Buffers, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_forward, mtp_stage
from .sparse import pool_bucket
from .weights import Weights


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
        w.comm.all_gather(packed.view(-1), got)
        g = got.view(w.world, R, 2 * k).cpu()
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Return each chosen token's top-k/top-p probability, using temperature 1 for greedy draft confidence."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


PREFILL_ROWS = 2048      # rows of a prompt chunk


def path_for(e, rows: int) -> str:
    """The same graph decision used by forward and the round price lookup."""
    st, graphs = e.st, getattr(e, "graphs", None)
    if graphs is None or rows > 8:
        return "eager"
    if not hasattr(e.w.cfg, "dense_limit"):
        return "graph-main"  # host-only stub; production configs always supply the limit
    if st.pos + rows <= e.w.cfg.dense_limit:
        return "graph-main" if (rows, st.parity) in graphs.main else "eager"
    if st.pos >= e.w.cfg.dense_limit and st.index is not None:
        bucket = pool_bucket(st.pos, rows, st.index[0][2].shape[0] - 2)
        return f"graph-sparse:{bucket}" if (rows, st.parity, bucket) in graphs.sparse else "eager"
    return "eager"


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    images = None       # this prompt's image rows: (positions, rows [n, D] bf16), set around a prefill

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = ()) -> None:
        self.w = w
        w.meta["long_context"] = long_context
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.buf = Buffers(w, max_rows, capacity)
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        if taps:
            self.buf.set_taps(tuple(taps), w.cfg.hidden)         # before any graph capture
            self.pbuf.set_taps(tuple(taps), w.cfg.hidden)
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.st = State(w, capacity, max_rows)
        self.last_hidden: torch.Tensor | None = None
        self.draft_n = w.head.n
        self.graphs = None
        self.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}   # steps by path
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, graph_rows, graph_rows)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def overlay(self, first: int, n: int):
        """The image rows among prompt positions [first, first + n): (row indices, their rows), or None."""

        if self.images is None:
            return None
        pos, rows = self.images
        lo, hi = np.searchsorted(pos, first), np.searchsorted(pos, first + n)
        if lo == hi:
            return None
        idx = torch.from_numpy(pos[lo:hi] - first).to(rows.device, non_blocking=True)
        return idx, rows[lo:hi]

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        path = path_for(self, R)
        if path != "eager":
            kind = "main" if path == "graph-main" else "sparse"
            g = (self.graphs.main[(R, self.st.parity)] if kind == "main" else
                 self.graphs.sparse[(R, self.st.parity, int(path.split(":", 1)[1]))])
            self.replays[kind] += 1
            g.replay()
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        return compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world]."""

        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        g, kind = None, "mtp"
        if self.graphs is not None and not self.mbuf.zero_first and dense:
            g = self.graphs.mtp.get(n)
        elif (self.graphs is not None and not self.mbuf.zero_first and self.st.index is not None
              and self.st.mtp_len >= self.w.cfg.dense_limit):
            bucket = pool_bucket(self.st.mtp_len, n, self.st.index[-1][2].shape[0] - 2)
            g, kind = self.graphs.sparse_mtp.get((n, bucket)), "sparse_mtp"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.mbuf.logits[:1, :self.draft_n]
        from .attention import CHUNK

        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // CHUNK),
                           host_pos=self.st.mtp_len)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def tap_rows(self, n: int, b: Buffers | None = None) -> torch.Tensor:
        """The last forward's first n rows of DFlash2 taps, concatenated in layer order: [n, taps * D]."""

        return torch.cat([t[:n] for t in (b or self.buf).taps], dim=1)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """Return the final-normed main-model rows that the MTP head reads after a forward with logits."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """Absorb hidden rows and their next tokens into the MTP cache in independent chunks; return the last logits."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept positions and chain drafts until cumulative confidence fails, always keeping the first draft."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Copy committed KDA state and retain attention caches whose valid prefixes survive resume; pending MTP rows await next tokens from the new prompt."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    drafter_end: int
    rows: list | None = None      # target and drafter rows, saved when another conversation took the live caches
    nbytes: int = 0
    drafter_rows: list[torch.Tensor] | None = None
    image_digests: tuple[bytes, ...] = ()
    # Exact replay (TF_GLM_EXACT_REPLAY): the prompt's last head row, padded to a rank-invariant width
    # (head_cols valid columns), and its final-normed hidden row. Memory snapshots of whole prompts only.
    last_logits: torch.Tensor | None = None
    head_cols: int = 0
    last_hidden: torch.Tensor | None = None


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool,
                  drafter=None) -> Snapshot:
    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    snap = Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    st.mtp_len - st.mtp_drafted if mtp and pending is not None else -1,
                    drafter.context_end if drafter is not None else -1)
    if drafter is not None and getattr(drafter, "ring", 0):
        snap.drafter_rows = _ring_window(drafter, snap.drafter_end)
    return snap


def _ring_slots(drafter, n: int) -> torch.Tensor:
    # The block may read the tile before its window boundary; preserve its bits too.
    lo = max(0, (max(0, n - drafter.window) // 64) * 64)
    return torch.arange(lo, n, device=drafter.kc[0].device) % drafter.ring


def _ring_window(drafter, n: int) -> list[torch.Tensor]:
    idx = _ring_slots(drafter, n)
    return [c.index_select(1, idx) for c in (*drafter.kc, *drafter.vc)]


def ring_bytes(drafter, n: int) -> int:
    if drafter is None or not getattr(drafter, "ring", 0):
        return 0
    count = n - (max(0, n - drafter.window) // 64) * 64
    return sum(c.shape[0] * count * c.shape[2] * c.element_size() for c in (*drafter.kc, *drafter.vc))


def _put_ring_window(drafter, n: int, rows: list[torch.Tensor]) -> None:
    idx = _ring_slots(drafter, n)
    caches = (*drafter.kc, *drafter.vc)
    if len(caches) != len(rows) or any(r.shape[1] != idx.numel() for r in rows):
        raise ValueError("saved DFlash2 ring does not fit the drafter")
    for cache, saved in zip(caches, rows):
        cache.index_copy_(1, idx, saved)


def _row_views(st, n: int, m: int) -> list[torch.Tensor]:
    """Views of the attention rows a snapshot of n tokens (m in the MTP head) depends on: latents or keys and values, indexer keys, gates and pools."""
    views = [kc[:n] for kc in st.kc] + [vc[:n] for vc in st.vc if vc is not None]
    idx = st.index or []
    main_idx = idx[:len(st.kc)]
    for ik, ig, pk in main_idx:
        views += [ik[:n], ig[:n], pk[:n // 4 + 1]]
    if m > 0 and hasattr(st, "mtp_kc"):
        views.append(st.mtp_kc[:m])
        if getattr(st, "mtp_vc", None) is not None:
            views.append(st.mtp_vc[:m])
        if len(idx) > len(st.kc):
            ik, ig, pk = idx[-1]
            views += [ik[:m], ig[:m], pk[:m // 4 + 1]]
    return views


def _snapshot_row_views(e: Engine, snap: Snapshot, drafter=None) -> list[torch.Tensor]:
    """Target rows, then this rank's committed DFlash2 keys and values (when the snapshot has them)."""
    views = _row_views(e.st, len(snap.ids), max(snap.mtp_len, 0))
    if snap.drafter_end >= 0 and not getattr(drafter, "ring", 0):
        if drafter is None:
            raise ValueError("a DFlash2 snapshot needs its drafter to save or load rows")
        # Keep the whole prefix: _dattn_kernel loads even out-of-window values before its masked dot.
        # Copying those bits too avoids assuming that masked values are finite (zero * NaN is NaN).
        views += [c[:, :snap.drafter_end] for c in (*drafter.kc, *drafter.vc)]
    return views


def save_rows(e: Engine, snap: Snapshot, drafter=None) -> None:
    """Copy a snapshot's target and drafter rows before another conversation overwrites the live caches."""
    views = _snapshot_row_views(e, snap, drafter)
    snap.rows = [v.clone() for v in views]
    snap.nbytes = sum(r.numel() * r.element_size() for r in snap.rows)


def extend_rows(e: Engine, old: Snapshot, snap: Snapshot, drafter=None) -> None:
    """Transfer saved prefix rows to a newer snapshot, copying only the tail."""
    if old.rows is None or old.ids != snap.ids[:len(old.ids)]:
        raise ValueError("only a saved strict prefix can be extended")
    views = _snapshot_row_views(e, snap, drafter)
    if len(views) != len(old.rows):
        raise ValueError("snapshot layouts differ")
    rows = []
    for dst, prior in zip(views, old.rows):
        parts = prior if isinstance(prior, tuple) else (prior,)
        used = sum(part.shape[0] for part in parts)
        if used > dst.shape[0]:
            raise ValueError("saved prefix exceeds new rows")
        rows.append((*parts, dst[used:].clone()))
    snap.rows = rows
    snap.nbytes = sum(part.numel() * part.element_size() for parts in rows for part in parts)
    old.rows, old.nbytes = None, 0


def row_bytes(e: Engine, snap: Snapshot, drafter=None) -> int:
    """What ``save_rows`` would copy for this snapshot."""
    return sum(v.numel() * v.element_size() for v in _snapshot_row_views(e, snap, drafter))


def snapshot_bytes(snap: Snapshot) -> int:
    """Device memory a kept snapshot holds: its KDA states, conv windows, pending MTP rows and any saved rows."""
    held = [snap.rec, snap.conv] + ([snap.pending] if snap.pending is not None else []) + (snap.drafter_rows or [])
    held += [t for t in (snap.last_logits, snap.last_hidden) if t is not None]
    return sum(t.numel() * t.element_size() for t in held) + (snap.nbytes if snap.rows is not None else 0)


def load_rows(e: Engine, snap: Snapshot, drafter=None) -> None:
    """Put a saved snapshot's target and drafter rows back into the live caches."""
    for dst, src in zip(_snapshot_row_views(e, snap, drafter), snap.rows, strict=True):
        if isinstance(src, tuple):
            at = 0
            for part in src:
                dst[at:at + part.shape[0]].copy_(part)
                at += part.shape[0]
        else:
            dst.copy_(src)


def restore(e: Engine, snap: Snapshot, drafter=None) -> None:
    st = e.st
    if st.cur:
        # A cancelled layer-sliced prefill can leave KDA layers on different
        # banks. The snapshot holds one complete bank; every layer must select
        # it before the next prefill, snapshot or decode commit uses the state.
        st.cur = [st.cur[0]] * len(st.cur)
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0
    if drafter is not None:
        if getattr(drafter, "ring", 0):
            if snap.drafter_rows is None:
                raise ValueError("this snapshot has no DFlash2 ring window")
            _put_ring_window(drafter, snap.drafter_end, snap.drafter_rows)
        drafter.context_end = snap.drafter_end
        drafter.pos_dev.fill_(snap.drafter_end)


# -- prefill ----------------------------------------------------------------------------------------------------
def dead_tap_prefix(start: int, end: int, needed_at: int, drafter) -> int:
    """Rows before the next kept window, preserving its leading 64-key tile and original tap projection groups."""

    if drafter is None:
        return end - start
    if not getattr(drafter, "ring", 0) or drafter.window < 0:
        return 0
    lo = max(0, needed_at - drafter.window) // 64 * 64
    if end <= lo:
        return end - start
    # add_taps projects groups of tap_in.shape[0] rows; keep every surviving group's original row count.
    group = drafter.tap_in.shape[0]
    return max(0, lo - start) // group * group


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
            resume: Snapshot | None = None, mark=None, keep=None) -> int:
    steps = prefill_steps(e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume, mark=mark, keep=keep)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def prefill_steps(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
                  resume: Snapshot | None = None, mark=None, keep=None, rows: int | Callable[[], int] | None = None,
                  slice_end: Callable[[], int] | None = None, sample_first: bool = True,
                  live_start: int | None = None):
    """Commit the prompt in chunks and sample its first token; a resumed prompt ends in a fresh prefill's state.

    ``mark(pos)``: the next checkpoint position after pos (or None). Chunks end there, and ``keep`` gets a snapshot
    of the prompt up to it, as the snapshot of a prompt ending there would be (its last row's MTP input pending).
    ``slice_end`` names an exclusive layer stop (zero disables slicing). A tuple yield is (chunk end, layer end),
    with state only partly written: the caller must retain exclusive pbuf ownership until the integer chunk yield.
    """

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.pbuf
    use_mtp = mtp and w.mtp is not None
    begin = 0
    if live_start is not None:
        # Reply admission has restored exact complete-chunk state, including the already
        # absorbed next-token MTP input. Prompt-snapshot restore would absorb it twice.
        if resume is not None or not 0 <= live_start < len(prompt) or st.pos != live_start:
            raise ValueError("invalid live prefill continuation")
        begin = live_start
    elif resume is None:
        e.reset()
        if drafter is not None:
            drafter.reset()
    else:
        begin = len(resume.ids)
        if begin >= len(prompt) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        restore(e, resume, drafter)
        if use_mtp:
            k = resume.pending.shape[0]
            b.overlay = e.overlay(begin - k + 1, k)
            try:
                _absorb_rows(e, resume.pending, list(prompt[begin - k + 1:begin + 1]))
            finally:
                b.overlay = None
        # The restored rows are private to this stream. Do not pin an evicted
        # snapshot for the rest of a long, interleaved prefill.
        resume = None
    last = None
    start = begin
    while start < len(prompt):
        step_rows = rows() if callable(rows) else rows or e.prefill_rows
        if not 1 <= step_rows <= e.prefill_rows:
            raise ValueError("invalid prefill step size")
        end = min(start + step_rows, len(prompt))
        at = mark(start) if mark is not None else None
        point = at is not None and start < at <= end and at < len(prompt)
        if point:
            end = at
        chunk = list(prompt[start:end])
        R = len(chunk)
        tap_start, dead_chunk = 0, False
        if p1.DEAD_WORK and not use_mtp:
            needed_at = min(len(prompt), at) if at is not None and at > start else len(prompt)
            tap_start = dead_tap_prefix(start, end, needed_at, drafter)
            dead_chunk = tap_start == R and end < needed_at
        prof.active = True
        try:
            b.overlay = e.overlay(start, R)
            staged = stage(w, st, b, chunk)
            options = dict(tap_start=tap_start, skip_final_ffn=dead_chunk, logits=not dead_chunk) if tap_start else {}
            stop_layer = slice_end() if slice_end is not None else 0
            if stop_layer:
                layer = 0
                while True:
                    result = compute(w, st, b, staged, nch=chunks_for(st, R), host_pos=st.pos,
                                     layer_start=layer, layer_stop=stop_layer, **options)
                    if stop_layer == len(w.layers):
                        break
                    layer = stop_layer
                    prof.active = False         # interleaved decode is not part of prefill profiling
                    yield end, layer
                    prof.active = True
                    stop_layer = slice_end()
            else:
                result = compute(w, st, b, staged, nch=chunks_for(st, R), host_pos=st.pos, **options)
            if not dead_chunk:
                last = result.clone()
            b.overlay = None
            if not dead_chunk:
                e.last_hidden = b.fnormed[R - 1:R].clone()
            if drafter is not None:
                if tap_start:
                    drafter.context_end += tap_start
                    drafter.pos_dev += tap_start
                    if tap_start < R:
                        drafter.add_taps(torch.cat([t[tap_start:R] for t in b.taps], dim=1))
                else:
                    drafter.add_taps(e.tap_rows(R, b))
            held = None
            if use_mtp:
                nxt = list(prompt[start + 1:start + R + 1])
                if point and nxt:                         # the last row waits, as at a prompt's end
                    held = b.fnormed[R - 1:R].clone()
                    nxt = nxt[:-1]
                if nxt:
                    with prof.timed("mtp absorb"):
                        b.overlay = e.overlay(start + 1, len(nxt))
                        _absorb_rows(e, b.fnormed[:len(nxt)], nxt)
                        b.overlay = None
            with prof.timed("commit"):
                commit(w, st, b, R, R)
            if point:
                with prof.timed("checkpoint"):
                    keep(take_snapshot(e, prompt[:end], held if use_mtp else None, mtp=use_mtp, drafter=drafter))
                if held is not None:
                    b.overlay = e.overlay(end, 1)
                    _absorb_rows(e, held, [prompt[end]])
                    b.overlay = None
            start = end
        finally:
            prof.active = False
            b.overlay = None
        if start < len(prompt):
            yield start
    prof.active = False
    prof.report(len(prompt) - begin)
    if p1.EXACT_REPLAY:
        e.last_logits = last        # the prompt-end head row a whole-prompt snapshot keeps (already a copy)
    return e.sample(last, [len(prompt)], sampling)[0] if sample_first else None


def head_width(w: Weights, cols: int) -> int:
    """The widest vocabulary shard over all ranks (at least this rank's ``cols``): a rank-invariant row width."""

    vocab, world = getattr(w.cfg, "vocab", None), getattr(w, "world", 1) or 1
    if not vocab:
        return cols
    from .weights import vocab_share

    return max(cols, max(hi - lo for lo, hi in (vocab_share(vocab, r, world) for r in range(world))))


def keep_head_row(w: Weights, logits: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Copy the last head row into a buffer as wide on every rank, so cache budgets agree across ranks."""

    cols = logits.shape[-1]
    row = logits.new_zeros((1, head_width(w, cols)))
    row[:, :cols].copy_(logits[-1:])
    return row, cols


def replay_ready(snap: Snapshot | None, prompt: Sequence[int]) -> bool:
    """A snapshot of exactly this prompt that can produce its first token without any prefill row."""

    return (snap is not None and snap.last_logits is not None and snap.head_cols > 0
            and len(snap.ids) == len(prompt) > 0 and list(prompt) == snap.ids)


@torch.no_grad()
def replay_first(e: Engine, prompt: Sequence[int], sampling: Sampling | None, snap: Snapshot, *,
                 drafter=None) -> int:
    """An exact replay: the prompt-end state, then its first token from the stored head row (no forward).

    The snapshot was taken right after this prompt's prefill finished, before its first token was sampled,
    so the restored target and drafter state equal the cold prefill's and the stored row is its last logits.
    The caller has already loaded the snapshot's attention rows (as for a strict-prefix resume).
    """

    if not replay_ready(snap, prompt):
        raise ValueError("an exact replay needs a whole-prompt snapshot with its head row")
    if snap.mtp_len >= 0 or (drafter is not None and snap.drafter_end != len(prompt)):
        raise ValueError("this snapshot's draft caches do not fit an exact replay")
    restore(e, snap, drafter)
    e.last_hidden = snap.last_hidden.clone() if snap.last_hidden is not None else None
    e.last_logits = None
    return sample_rows(e.w, snap.last_logits[:, :snap.head_cols], [len(prompt)], sampling)[0]   # = Engine.sample


def replay_steps(e: Engine, prompt: Sequence[int], sampling: Sampling | None, snap: Snapshot, *, drafter=None):
    """``replay_first`` as a prefill stepper that finishes on its first step (a zero-row FILL)."""

    return replay_first(e, prompt, sampling, snap, drafter=drafter)
    yield  # noqa: unreachable; makes this a generator


def _absorb_rows(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> None:
    """A prompt's rows into the MTP cache through the prefill buffers (the prefill arithmetic, like the prompt)."""

    st = e.st
    mtp_forward(e.w, st, e.pbuf, next_tokens, hidden)
    st.set_mtp_len(st.mtp_len + len(next_tokens))


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    arms: str = ""                          # auto_decode: the drafter of each round, "m" (MTP) or "f" (DFlash2)
    pending: torch.Tensor | None = None     # auto_decode: MTP input rows of committed positions not absorbed yet

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = e.forward([out[-1]])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Choose a fixed draft count or adapt it to running acceptance."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.rate = 0.8

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify pending and MTP draft rows through the first mismatch, starting with every prompt position except the last in the MTP cache."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    drafts = draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence) if depth > 0 else []
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                        policy.confidence) if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


@torch.no_grad()
def dflash_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                  policy: DepthPolicy | None = None, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify pending and DFlash2 draft rows through the first mismatch and absorb kept taps, starting from prefill's drafter state."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy(3, fixed=True)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        drafts = drafter.propose(out[-1], depth, sampling, policy.confidence) if depth > 0 else []
        t1 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        drafter.add_taps(e.tap_rows(keep))
        t5 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)
