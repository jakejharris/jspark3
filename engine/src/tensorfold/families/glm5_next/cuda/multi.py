"""GLM-5.3-Flash's concurrent rounds on two ranks: every stream keeps exactly its own accepted prefix, so it equals its serial decoding."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from .decode import PREFILL_ROWS, DepthPolicy, Engine, _probability, draft, prefill, restore, take_snapshot
from .forward import Buffers, State, commit, compute_streams, stage_streams
from .mtp import mtp_compute_streams, mtp_stage_streams
from .weights import Weights

ADMIT, ROUND, DONE = 1, 2, 3            # rank 0's messages


def _slot(w: Weights, st: State, buf: Buffers, mbuf: Buffers | None, pbuf: Buffers) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.rows, e.prefill_rows = w, buf.rows, pbuf.rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.last_hidden, e.draft_n = None, w.head.n
    e.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}
    return e


def pack_sampling(s: Sampling | None) -> list[int]:
    if s is None or s.temperature <= 0:
        return [0, 0, 0, 0, 0, 0, 0]
    seed = s.seed & 0xFFFFFFFFFFFFFFFF
    return [1, seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62, round(s.temperature * 1e6), int(s.top_k or 0),
            round(s.top_p * 1e6)]


def unpack_sampling(v: Sequence[int]) -> Sampling | None:
    if not v[0]:
        return None
    return Sampling((v[3] << 62) | (v[2] << 31) | v[1], v[4] / 1e6, v[5], v[6] / 1e6)


def sample_streams(w: Weights, logits: torch.Tensor, rows: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
                   samplings: Sequence[Sampling | None], probs: Sequence[bool] = ()) -> tuple[list, list]:
    """Each stream's rows of (this rank's vocabulary slice of) logits -> its tokens, as ``sample_rows`` picks them; one gather per candidate width."""

    probs = list(probs) or [False] * len(rows)
    width = logits.shape[1]

    def k_of(s, p):
        greedy = s is None or s.temperature <= 0
        k = 1 if greedy else min(width, int(s.top_k) + MARGIN)
        return min(width, 20 + MARGIN) if p and greedy else k

    groups: dict[int, list[int]] = {}
    for i, (s, p) in enumerate(zip(samplings, probs)):
        groups.setdefault(k_of(s, p), []).append(i)
    cand: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, members in groups.items():
        idx = [r for i in members for r in rows[i]]
        sel = logits[torch.tensor(idx, device=logits.device)] if len(idx) != logits.shape[0] else logits
        vals, ids = torch.topk(sel.float(), k, dim=-1)
        ids = (ids + w.vocab_offset).to(torch.int32)
        if w.comm is None:
            values, tokens = vals.cpu().numpy().astype(np.float32), ids.cpu().numpy().astype(np.int64)
        else:
            packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
            got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
            w.comm.all_gather(packed.view(-1), got)
            g = got.view(w.world, len(idx), 2 * k).cpu()
            values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
            tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
            tokens = tokens.astype(np.int64)
        at = 0
        for i in members:
            n = len(rows[i])
            cand[i] = (values[at:at + n], tokens[at:at + n])
            at += n
    chosen, chances = [], []
    for i, s in enumerate(samplings):
        values, tokens = cand[i]
        if s is None or s.temperature <= 0:
            order = np.lexsort((tokens, -values), axis=-1)
            got = [int(tokens[r, order[r, 0]]) for r in range(len(rows[i]))]
        else:
            got = choose_rows(values, tokens, positions[i], s)
        chosen.append(got)
        chances.append(_probability(values, tokens, got, s) if probs[i] else None)
    return chosen, chances


class MultiDecoder:
    """The ``Scheduler``'s decoder as ``rank`` of two: ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w: Weights, *, slots: int, capacity: int, depth: int = 3, keep: int = 8, rank: int = 0,
                 share=None, long_context: bool = False) -> None:
        self.w, self.depth, self.capacity, self.rank, self.share = w, depth, capacity, rank, share
        self.eos = tuple(w.cfg.eos)
        w.meta["long_context"] = long_context
        rows = slots * (depth + 1)
        self.buf = Buffers(w, rows, capacity)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, PREFILL_ROWS, capacity, prefill=True)
        self.slots = [State(w, capacity, depth + 1) for _ in range(slots)]
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.slots[0]))
        self.free = list(range(slots))
        self.kept: list[tuple[list[int], int, object]] = []           # (prompt ids, slot, snapshot), oldest first
        self.keep = keep
        self.streams: dict[int, Stream] = {}
        self.next_id = 0
        self.broken: Exception | None = None

    # -- rank 0 -> rank 1 -------------------------------------------------------------------------------------
    def _send(self, values: list[int]) -> None:
        if self.share is not None and self.rank == 0 and self.broken is None:
            self.share(values)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    def live(self) -> int:
        return len(self.streams)

    # -- slots ------------------------------------------------------------------------------------------------
    def _busy(self) -> set[int]:
        return {s.slot for s in self.streams.values()}

    def _pick(self, prompt: list[int], reuse: bool) -> tuple[int, int]:
        """(slot, cached length): the idle kept prompt the new one extends furthest, else a free slot, else the oldest idle kept one."""

        busy = self._busy()
        best = None
        for ids, slot, _ in self.kept if reuse else []:
            if slot not in busy and len(ids) < len(prompt) and prompt[:len(ids)] == ids and (
                    best is None or len(ids) > len(best[0])):
                best = (ids, slot)
        if best is not None:
            return best[1], len(best[0])
        if self.free:
            return self.free[0], 0
        idle = next((slot for _, slot, _ in self.kept if slot not in busy), None)
        if idle is None:
            raise RuntimeError("no free stream slot")
        return idle, 0

    def _claim(self, slot: int, cached: int) -> object | None:
        """Take ``slot`` for a new stream; its kept snapshot of ``cached`` ids when resuming, and no other kept entry keeps it."""

        snap = next((sn for ids, sl, sn in self.kept if sl == slot and len(ids) == cached), None) if cached else None
        self.kept = [k for k in self.kept if k[1] != slot]
        if slot in self.free:
            self.free.remove(slot)
        return snap

    def _remember(self, ids: list[int], slot: int, snap) -> None:
        """Keep a prompt's end in its slot; a slot another entry for the same ids (or the oldest entry) held goes back to the free list."""
        gone = [k[1] for k in self.kept if k[0] == ids and k[1] != slot]
        self.kept = [k for k in self.kept if k[0] != ids and k[1] != slot] + [(ids, slot, snap)]
        while len(self.kept) > self.keep:
            gone.append(self.kept.pop(0)[1])
        busy = self._busy()
        for old in gone:
            if old != slot and old not in busy and old not in self.free and all(k[1] != old for k in self.kept):
                self.free.append(old)

    # -- streams ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Rank 0: prefill a request in a slot (resuming a kept prompt it extends), draft its first chain and emit its first token."""

        self._check()
        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        slot, cached = self._pick(list(s.prompt), s.draft)
        s.sid = self.next_id
        self.next_id += 1
        stop = int(getattr(s, "stop_eos", True))
        self._send([ADMIT, s.sid, s.count, int(s.draft), stop, slot, cached, *pack_sampling(s.sampling)])
        self._send(list(s.prompt))
        try:
            self._admit(s, slot, cached)
        except Exception as exc:
            self.broken = exc
            raise

    def _admit(self, s: Stream, slot: int, cached: int) -> None:
        t0 = time.perf_counter()
        snap = self._claim(slot, cached)
        st = self.slots[slot]
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        if snap is None:
            cached = 0
        first = prefill(e, s.prompt, s.sampling, mtp=mtp, resume=snap)
        s.slot, s.st, s.cached = slot, st, cached
        s.eos = self.eos if getattr(s, "stop_eos", True) else ()
        greedy = s.sampling is None or s.sampling.temperature <= 0
        s.policy = DepthPolicy(self.depth, fixed=True, confidence=0.35) if greedy else DepthPolicy(self.depth, low=0.6,
                                                                                                   high=0.85)
        if s.draft:
            self._remember(list(s.prompt), slot, take_snapshot(e, s.prompt, e.last_hidden if mtp else None, mtp=mtp))
        s.context = list(s.prompt)
        depth = min(s.policy.next(0, 0), s.count - 1) if mtp else 0
        s.drafts = draft(e, e.last_hidden, [first], st.pos + 1, depth, s.sampling, s.policy.confidence) if depth > 0 \
            else []
        s.prefill_s, s.started = time.perf_counter() - t0, time.perf_counter()
        self.streams[s.sid] = s
        s.take([first], s.eos)

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams (rank 0 names them for rank 1); returns the ones that finished."""

        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return []
        self._check()
        self._send([ROUND, len(live), *[s.sid for s in live]])
        try:
            self._round(live)
        except Exception as exc:
            self.broken = exc
            raise
        return [s for s in live if s.done]

    def _round(self, live: list[Stream]) -> None:
        w = self.w
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage_streams(w, self.buf, windows)
        logits = compute_streams(w, segs, self.buf, eager=True)
        rows = [list(range(a0, a1)) for _, a0, a1 in segs]
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled, _ = sample_streams(w, logits, rows, positions, [s.sampling for s in live])
        kept = []
        for s, (_, tokens), (st, a0, a1), got in zip(live, windows, segs, sampled):
            keep = 1
            for i, d in enumerate(tokens[1:]):
                if got[i] != d or got[i] in s.eos:
                    break
                keep += 1
            keep = min(keep, s.count - len(s.out))
            commit(w, st, self.buf, a1 - a0, keep)
            s.counted(len(tokens))
            s.last = (len(tokens) - 1, keep - 1)
            kept.append((s, a0, got[:keep]))
        self._draft_all([(s, a0, new) for s, a0, new in kept if s.draft and len(s.out) + len(new) < s.count
                         and new[-1] not in s.eos])
        for s, _, new in kept:
            s.take(new, s.eos)

    def _draft_all(self, todo: list) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one MTP step a depth."""

        w = self.w
        for s, _, _ in todo:
            s.drafts = []
        todo = [(s, a0, new, min(s.policy.next(*s.last), s.count - len(s.out) - len(new))) for s, a0, new in todo]
        todo = [t for t in todo if t[3] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, new, self.buf.fnormed[a0:a0 + len(new)]) for s, a0, new, _ in todo]
        segs = mtp_stage_streams(w, self.mbuf, windows)
        logits = mtp_compute_streams(w, segs, self.mbuf, last=[a1 - 1 for _, _, a1 in segs], eager=True)
        for (s, _, new, _), _ in zip(todo, segs):
            s.st.set_mtp_len(s.st.mtp_len + len(new))
        active = [(s, i, depth, 1.0) for i, (s, _, _, depth) in enumerate(todo)]
        for j in range(self.depth):
            conf = [s.policy.confidence > 0 for s, _, _, _ in active]
            picks, chances = sample_streams(w, logits, [[i] for _, i, _, _ in active],
                                            [[s.st.pos + 1 + j] for s, _, _, _ in active],
                                            [s.sampling for s, _, _, _ in active], conf)
            nxt = []
            for (s, i, depth, chain), (d,), p, c in zip(active, picks, chances, conf):
                if c and j > 0 and chain * p[0] < s.policy.confidence:
                    continue
                s.drafts.append(d)
                if c:
                    chain *= p[0]
                    if chain < s.policy.confidence:
                        continue
                if j + 1 < depth:
                    nxt.append((s, i, depth, chain, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.fnormed[i:i + 1].clone()) for s, i, _, _, d in nxt]
            segs = mtp_stage_streams(w, self.mbuf, windows)
            logits = mtp_compute_streams(w, segs, self.mbuf, last=[a1 - 1 for _, _, a1 in segs], eager=True)
            for s, _, _, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, k, depth, chain) for k, (s, _, depth, chain, _) in enumerate(nxt)]

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams (rank 0 names them for rank 1); a slot whose prompt is kept stays with it, the rest are free again."""

        if done:
            self._send([DONE, len(done), *[s.sid for s in done]])
        self._finish([s.sid for s in done])

    def _finish(self, sids: list[int]) -> None:
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is not None and all(k[1] != s.slot for k in self.kept) and s.slot not in self.free:
                self.free.append(s.slot)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            self.streams.pop(s.sid, None)
            self.kept = [k for k in self.kept if k[1] != s.slot]
            if s.slot not in self.free:
                self.free.append(s.slot)
        return live

    # -- rank 1 -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def follow(self) -> None:
        """Rank 1: mirror rank 0's admissions, rounds and finished streams, forever."""

        while True:
            msg = self.share(None)
            if msg[0] == ADMIT:
                sid, count, drafts, stop, slot, cached, *samp = msg[1:]
                s = Stream(self.share(None), count, unpack_sampling(samp), draft=bool(drafts))
                s.stop_eos, s.sid = bool(stop), sid
                self.next_id = sid + 1
                self._admit(s, slot, cached)
            elif msg[0] == ROUND:
                self._round([self.streams[sid] for sid in msg[2:2 + msg[1]]])
            elif msg[0] == DONE:
                self._finish(list(msg[2:2 + msg[1]]))


def _tensors(st: State):
    for value in vars(st).values():
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, torch.Tensor):
                    yield v
                elif isinstance(v, tuple):
                    yield from (t for t in v if isinstance(t, torch.Tensor))
