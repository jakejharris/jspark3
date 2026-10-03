"""TF_GLM_SP=1 (off by default; prompt chunks only): every rank does a pass's row-local work (hyper-connections, their
norms, the stream means) on its own share of the rows; the tensor-parallel blocks still run on every row everywhere.

Class E by construction, since no arithmetic moves:
- those kernels launch one program per row with the same constexprs, so a row's bits do not depend on which rows
  share its launch (the property drafted == serial already rests on);
- the cross-rank sum stays in ``glue.hc_post``: rank k's fp32 partial rows travel to the rows' owner and land at offset
  k, never in arrival order, so the owner adds (p0 + p1) + p2 and rounds to bf16 at the point every rank did before
  (its own partial read where its block wrote it, in its turn);
- what the blocks read (normed rows and their 64-group sums) and what the engine reads after a pass (the final hidden
  rows and the DFlash2 taps) are copied byte for byte from each row's owner. The residual streams ``b.x`` are not:
  during and after a planned pass a rank's other rows there are stale, so nothing may read them.

Bytes a rank receives at each of a layer's two reductions, for R rows of width D: 2/3 R D x 4 (partials) + 2/3 R D x 2
(normed) + 2/3 R D/64 x 4 (sums), against 2 R D x 4 for the all-gather of every partial. The collectives are groups of
point-to-point calls (one hop on the full mesh) whatever TF_COMM_ONEHOP says. TF_GLM_SP_MIN_ROWS: smaller passes keep
the all-gather (both are exact; this only picks the faster).

TF_GLM_SP_OVERLAP=1 (off by default, with or without TF_GLM_SP) pipelines a prompt chunk as two halves, rows [0, h) and
[h, R), so each half's comm hides under the other half's work: the main stream runs only the big blocks (attention, the
FFN) in the order A, B, A, B; one side stream runs each half's tail after its block (reduce, hc_post, taps, the next
hc_pre and share), every NCCL call of the pass on it in one order on every rank. Class E as two consecutive chunks are:
every kernel is chunk-invariant, and each half's layer L (KDA state, latent and indexer writes) runs after the first
half's. TF_GLM_SP_OVERLAP_MIN_ROWS: smaller chunks run whole."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

import torch

from . import prof, tp3_probe

ENABLED = os.environ.get("TF_GLM_SP", "0") == "1"
MIN_ROWS = int(os.environ.get("TF_GLM_SP_MIN_ROWS", "256"))
OVERLAP = os.environ.get("TF_GLM_SP_OVERLAP", "0") == "1"
OVERLAP_MIN_ROWS = int(os.environ.get("TF_GLM_SP_OVERLAP_MIN_ROWS", "1024"))


@dataclass(frozen=True)
class Plan:
    """A pass's rows in contiguous, balanced shares: rank k owns rows starts[k] .. starts[k] + counts[k]."""

    rank: int
    starts: tuple[int, ...]
    counts: tuple[int, ...]

    def rows(self, k: int) -> slice:
        return slice(self.starts[k], self.starts[k] + self.counts[k])

    @property
    def own(self) -> tuple[int, int]:
        return self.starts[self.rank], self.starts[self.rank] + self.counts[self.rank]

    @property
    def peers(self) -> list[int]:
        return [k for k in range(len(self.counts)) if k != self.rank]


def split(R: int, world: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shares of R rows: the first R % world ranks take one row more; starts in rank order."""

    base, extra = divmod(R, world)
    counts = tuple(base + (k < extra) for k in range(world))
    return tuple(sum(counts[:k]) for k in range(world)), counts


def plan(w, b, R: int) -> Plan | None:
    """This pass's plan, or None for the all-gather path (flag off, decode or MTP buffers, one rank, the NCCL probe)."""

    if not ENABLED or not b.prefill or w.comm is None or w.world < 2 or tp3_probe.REDUCE != "gather":
        return None
    if R < max(MIN_ROWS, w.world):        # every rank owns at least one row
        return None
    starts, counts = split(R, w.world)
    return Plan(w.rank, starts, counts)


def _swap(w, sp: Plan, tensors: list[torch.Tensor], start: int = 0) -> None:
    """Each tensor's own rows (from row ``start`` on) to every peer and every peer's rows into place, in one group."""

    def rows(k: int) -> slice:
        r = sp.rows(k)
        return slice(max(r.start, start), max(r.stop, start))

    sends = [(k, t[rows(sp.rank)]) for k in sp.peers for t in tensors]
    recvs = [(k, t[rows(k)]) for k in sp.peers for t in tensors]
    w.comm.exchange(sends, recvs)


def share(w, b) -> None:
    """After a hc_pre on own rows: every rank's normed rows and their sums to every rank (a no-op without a plan)."""

    if b.sp is not None:
        _swap(w, b.sp, [b.normed, b.xs])


class Reduced(NamedTuple):
    """``reduce``'s result, which ``glue.hc_post`` sums in rank order: every peer k's fp32 partial of this rank's n
    rows at slot k of ``peers`` [world, n, D] (slot ``rank`` is left unwritten), and this rank's own partial ``own``
    [n, D], read where its block wrote it (b.part): no copy into the view."""

    peers: torch.Tensor
    rank: int
    own: torch.Tensor


def reduce(w, b, R: int) -> Reduced:
    """The ordered reduce-scatter of b.part[:R]: rank k's partial of this rank's n rows at k."""

    sp = b.sp
    n, d = sp.counts[sp.rank], b.part.shape[1]
    out = b.gath[:len(sp.counts) * n * d].view(len(sp.counts), n, d)
    w.comm.exchange([(k, b.part[sp.rows(k)]) for k in sp.peers], [(k, out[k]) for k in sp.peers])
    return Reduced(out, sp.rank, b.part[sp.rows(sp.rank)])


def finish(w, b, tap_start: int = 0) -> None:
    """The pass's last exchanges: every rank's final hidden rows, and its DFlash2 tap rows from ``tap_start`` on (the
    dead-work skip computes no earlier ones), to every rank."""

    _swap(w, b.sp, [b.hidden])
    if b.taps:
        _swap(w, b.sp, b.taps, tap_start)


def halves(w, b, segs, R: int, eager: bool) -> int | None:
    """TF_GLM_SP_OVERLAP: the first half's rows of this pass (a 64-row cut where it has one), or None to run it whole
    (flag off, decode or MTP buffers, several streams, one rank, the NCCL probe or its recorder, the profiler)."""

    if not OVERLAP or not b.prefill or not eager or len(segs) != 1 or w.comm is None or w.world < 2:
        return None
    if tp3_probe.REDUCE != "gather" or tp3_probe.recorder is not None or prof.ENABLED:
        return None                       # the profiler synchronizes every block: nothing could overlap
    if R < max(OVERLAP_MIN_ROWS, 2):
        return None
    return R // 2 // 64 * 64 or R // 2


class Side:
    """A half's two events on the pass's side stream: ``tail`` runs work there after the main stream's work so far;
    ``join`` makes the main stream wait for the half's last tail. Without CUDA (interpreter tests) it runs in order."""

    stream = None                         # one side stream a process, made on first use

    def __init__(self, device) -> None:
        self.cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        if self.cuda:
            if Side.stream is None:       # high priority: a comm kernel starts as soon as an SM frees
                Side.stream = torch.cuda.Stream(device=device, priority=-1)
            self.main, self.big, self.done = torch.cuda.current_stream(device), torch.cuda.Event(), torch.cuda.Event()

    def tail(self, fn) -> None:
        if not self.cuda:
            fn()
            return
        self.big.record(self.main)
        with torch.cuda.stream(Side.stream):
            Side.stream.wait_event(self.big)
            fn()
            self.done.record(Side.stream)

    def join(self) -> None:
        if self.cuda:
            self.main.wait_event(self.done)
