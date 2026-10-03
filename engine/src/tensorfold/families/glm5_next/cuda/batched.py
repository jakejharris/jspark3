"""Pooled GLM streams: shared target rows, private recurrent state and exact keyed sampling.

Only the worker thread touches this decoder. Rank zero names every admission,
round and release; followers run the same forwards and choose the same tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from typing import Sequence

import torch

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling

from .decode import (Snapshot, draft, head_width, keep_head_row, load_rows, path_for, prefill, prefill_steps, replay_first,
                     replay_ready, replay_steps, row_bytes, save_rows, snapshot_bytes, take_snapshot)
from .decode_observer import DecodeObserver, PriceRounds
from .draft_pricing import (Calibration, PriceTable, SegmentTracker, choose_depths, context_class, cumulative,
                            load_json_env, segment_patterns, threshold_depth, SRC_N, SRC_D, SRC_C, SRC_F,
                            )
from .engine import DFLASH_POLICY, _f64_ints, _ints_f64, decode_policy, encode_policy
from .forward import Buffers, State, commit, compute_streams, stage_streams
from .multi import sample_streams
from . import cofill, cofill_big, prefill_options as p1, prof
from .copy_control import ADMIT_COPY, CopyController
from .reply_reservations import MESSAGE as RESERVATION_MESSAGE

ADMIT, ROUND, DONE, FILL, FILL_SLICE, SCORE, ROUND2 = 1, 2, 3, 4, 5, 6, 7
ROUND_QUANTUM = 10  # 8 is ADMIT_REPLY; 9 is the reservation protocol.
ADMIT_REPLY = 8
SAMPLING_INTS = 9
PREFILL_BUSY_ROWS = 1024
PREFILL_IDLE_ROWS = 2048
PREFILL_EXPRESS_ATOMIC_ROWS = p1.EXPRESS_ATOMIC_ROWS  # TF_GLM_EXPRESS_ATOMIC_ROWS; configurable atomic threshold
PREFILL_AGE_ROWS_PER_SECOND = 8192
PREFILL_PRIORITY_ROWS = {None: 0, "interactive": -4096, "background": 4096}


def pack_sampling(s: Sampling | None) -> list[int]:
    """Carry the exact binary64 temperature/top-p and all 64 seed bits in int32 messages."""
    if s is None or s.temperature <= 0:
        return [0] * SAMPLING_INTS
    seed = s.seed & 0xFFFFFFFFFFFFFFFF
    return [1, seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
            *_f64_ints(s.temperature), int(s.top_k or 0), *_f64_ints(s.top_p)]


def unpack_sampling(v: Sequence[int]) -> Sampling | None:
    if len(v) != SAMPLING_INTS:
        raise ValueError("invalid batched sampling message")
    if not v[0]:
        return None
    return Sampling((v[3] << 62) | (v[2] << 31) | v[1], _ints_f64(v[4], v[5]), v[6], _ints_f64(v[7], v[8]))


class BatchedDecoder:
    """An existing initialized engine and drafter, partitioned among at most ``slots`` streams.

    Reservations include guards and are held through finish. Saved prompt ends own
    complete copies, so revisits never pin a live reservation or retain stale rows.
    Residual automatic policies use a fixed available drafter; EXL3's normal auto
    resolution still happens in ``owner._effective`` first.
    """

    admit_per_round = 1
    fatal_errors = True

    def __init__(self, owner, slots: int) -> None:
        from .pool import TokenPool

        if not 2 <= slots <= 8:
            raise ValueError("batched GLM needs between two and eight slots")
        if not owner.e.st.latent:
            raise ValueError("batched GLM needs the latent cache (TF_GLM_LATENT=1)")
        if p1.COFILL_BIG and not p1.ATTENTION_TILES:
            raise ValueError("TF_GLM_COFILL_BIG requires TF_GLM_A2_ATTENTION_TILES=1")
        self.owner, self.w = owner, owner.w
        self.prefill_rows_idle = getattr(owner, "prefill_rows_idle", 0)
        self.prefill_rows_busy = getattr(owner, "prefill_rows_busy", 4096)
        self.rank = owner.rank
        self.share = owner._share
        self.capacity = owner.e.st.capacity
        self.max_rows = owner.e.rows
        self.guard = max(self.max_rows, owner.drafter.block if owner.drafter is not None else 0)
        if self.guard > 64:
            raise ValueError("batched GLM supports DFlash2 blocks up to 64 rows")
        self.pool = TokenPool(self.capacity, guard=self.guard)
        self.buf = Buffers(self.w, slots * self.max_rows, self.capacity)
        if owner.drafter is not None:
            self.buf.set_taps(tuple(owner.drafter.tap_layers), self.w.cfg.hidden)
        self.slots: list[State | None] = [owner.e.st] + [None] * (slots - 1)
        self.streams: dict[int, Stream] = {}
        self.filling: dict[int, object] = {}
        self.fill_owner: int | None = None
        self.prefill_decode_ticks = 0
        self.express_owner: int | None = None
        self.express_buf = None
        if p1.EXPRESS:
            self.express_buf = Buffers(self.w, min(owner.e.prefill_rows, p1.EXPRESS_ROWS),
                                       self.capacity, prefill=True)
            if owner.drafter is not None:
                self.express_buf.set_taps(tuple(owner.drafter.tap_layers), self.w.cfg.hidden)
        self.express_atomic_rows = PREFILL_EXPRESS_ATOMIC_ROWS
        self.express_cofill = p1.EXPRESS_COFILL and self.express_buf is not None
        self.prefill_slice_ms = p1.PREFILL_SLICE_MS
        self.express_busy_rows = p1.EXPRESS_BUSY_ROWS
        self.express_gather_s = p1.EXPRESS_GATHER_MS / 1000.0
        self.prefill_slice_layers = int(os.environ.get("TF_GLM_PREFILL_SLICE_LAYERS", "0"))
        if self.prefill_slice_layers < 0:
            raise ValueError("TF_GLM_PREFILL_SLICE_LAYERS must be nonnegative")
        self.fair_schedule = os.environ.get("TF_GLM_FAIR_SCHED", "0") == "1"
        self.cache: list[Snapshot] = []
        self.next_id = 0
        self.broken: Exception | None = None
        self.observer = DecodeObserver(self.rank)
        self.price_error = None
        try:
            table = load_json_env("TF_GLM_PRICE_TABLE") if self.rank == 0 else None
            self.price_table = PriceTable(table, weights=os.environ.get("TF_GLM_WEIGHT_ID", ""),
                                          drafter=os.environ.get("TF_GLM_DRAFTER_ID", ""),
                                          build=os.environ.get("TF_GLM_BUILD_ID", ""),
                                          build_ok=os.environ.get("TF_GLM_PRICE_TABLE_BUILD_OK", "")) if table is not None else None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.price_error = str(exc)
            self.price_table = None
        self.prior = load_json_env("TF_GLM_DRAFT_PRIOR") if self.rank == 0 else None
        self.carried = Calibration(self.prior, copy_depth=self.max_rows - 1) if self.rank == 0 else None
        self.segment_patterns = None
        self.segment_patterns_loaded = False
        self.round_id = 0
        self.price_rounds = PriceRounds(self.rank, self.observer.enabled, False)
        self.recent_sessions = [] if self.observer.enabled and self.rank == 0 else None
        self.sessions = getattr(owner, "session_cache", None)
        self.copy_control = CopyController(self)
        if cofill.enabled(None):
            self.admit_per_round = slots
        self.reply_prefill = bool(getattr(owner, "reply_prefill", False))
        self.reply_prefill_rows = getattr(owner, "reply_prefill_rows", 256)
        self.growth = None
        if getattr(getattr(owner, "reply_config", None), "enabled", False):
            from .reply_reservations import ReplyReservations

            self.growth = ReplyReservations(self, owner.reply_config)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the GLM ranks are out of step after an error; restart all ranks") from self.broken

    def _send(self, values: list[int], *, payload: bool = False) -> None:
        if not payload:
            self._writer_idle(False)
        if self.share is not None and self.rank == 0:
            bell = getattr(self.owner, "follower_doorbell", None)
            # Every top-level command while streams is empty wakes both followers.
            # Payload frames belong to the preceding command; active rounds never ring.
            if bell is not None and not payload and not self.streams:
                bell.ring()
            self.share(values)

    def _writer_idle(self, idle):
        if self.sessions is not None and self.sessions.store is not None:
            self.sessions.store.write_gate.set_idle(idle)

    def live(self) -> int:
        return len(self.streams)

    def _request(self, s: Stream) -> tuple[list[int], int]:
        """Host-only validation, before ADMIT can leave a follower waiting in a collective."""
        self._check()
        priority = getattr(s, "priority", None)
        if priority not in (None, "interactive", "background"):
            raise ValueError("priority must be interactive or background")
        layers = getattr(s, "prefill_slice_layers", None)
        layers = self.prefill_slice_layers if layers is None else layers
        if type(layers) is not int or layers < 0:
            raise ValueError("prefill_slice_layers must be a nonnegative integer")
        if not s.prompt:
            raise ValueError("prefill requires at least one token")
        count = int(s.count)
        if count < 1 or len(s.prompt) + count > self.owner.limit:
            raise ValueError(f"{len(s.prompt)} prompt tokens and {count} output tokens do not fit the "
                             f"{self.owner.limit}-token context")
        if self.growth is not None and (self.pool._size(len(s.prompt) + count) > self.pool.capacity
                or len(s.prompt) + count > getattr(self.owner, "pool_limit", self.owner.limit)):
            raise ValueError("the full reply promise must fit alone, including speculative guards")
        code = self.owner._effective(encode_policy(getattr(s, "policy", self.owner.policy))) if s.draft else [0]*4
        if code[0] == 16 and self.price_table is None and self.rank == 0:
            code = encode_policy("fc7:0.3")
            s.price_table_missing = "invalid" if self.price_error is not None else "missing"
        if code[0] in (4, 5):
            code = encode_policy(DFLASH_POLICY if self.owner.drafter is not None else "3")
        if self.fair_schedule and priority == "background" and (code[0] % 10 in (1, 2, 3)
                                                                 or code[0] in (16, 17)):
            code = [code[0], min(code[1], 1), *code[2:]]
        if code[0] >= 10 and self.owner.drafter is None:
            raise ValueError("the requested DFlash2 policy needs a loaded drafter")
        if code[0] in (1, 2, 3) and self.w.mtp is None:
            raise ValueError("the requested MTP policy needs an MTP head")
        images = getattr(s, "images", None)
        if self.growth is not None and any(
                not isinstance(getattr(image, "digest", None), bytes) or len(image.digest) != 32
                for image in images or ()):
            raise ValueError("reply growth requires complete image fingerprints before admission")
        if hasattr(self.owner, "_image_starts"):
            self.owner._image_starts(s.prompt, images)
        if images and getattr(self.owner, "tower", None) is None:
            raise ValueError("image input needs a loaded vision tower")
        return code, len(s.prompt) + count

    def can_admit(self, s: Stream) -> bool:
        """Nonmutating admission: both usable context and a contiguous guarded span must fit."""
        if hasattr(s, "quality_score_start"):
            return not self.live() and not self.filling and self.fill_owner is None and self.express_owner is None
        _, tokens = self._request(s)
        if self.growth is not None:
            if self.streams and (self.growth.drain or self.growth.parked):
                return False
            tokens = self.growth.initial(s)
        if self.live() == len(self.slots):
            return False
        if sum(s.reserved_tokens for s in self.streams.values()
               if self.growth is None or s.sid not in self.growth.parked) + tokens > getattr(self.owner, "pool_limit", self.owner.limit):
            return False
        return self.pool.can_allocate(tokens)

    def _saved(self, prompt: Sequence[int], code: Sequence[int], cached: int | None = None,
               images=None, image_digests=None, *, memory_only: bool = False):
        mtp, dflash = code[0] in (1, 2, 3), code[0] >= 10
        digests = (image_digests if image_digests is not None else
                   tuple(getattr(image, "digest", None) for image in images or ()) if images is not None else None)
        # Exact replay: an entry equal to the whole prompt fits when it kept its head row (not MTP or reply prefill).
        exact = p1.EXACT_REPLAY and not memory_only and not mtp
        fits = [c for c in self.cache if (len(c.ids) < len(prompt) or exact and replay_ready(c, prompt))
                and list(prompt[:len(c.ids)]) == c.ids
                and (not mtp or c.mtp_len >= 0) and (not dflash or c.drafter_end == len(c.ids))
                and (digests is None or c.image_digests == digests[:len(c.image_digests)]
                     and len(c.image_digests) == sum(token < 0 and (i == 0 or c.ids[i - 1] != token)
                                                     for i, token in enumerate(c.ids)))
                and (cached is None or len(c.ids) == cached)]
        best = max(fits, key=lambda c: len(c.ids), default=None)
        if best is not None:
            return best  # warm memory never waits on a longer disk prefix
        if self.sessions is not None and not memory_only:
            disk = self.sessions.saved(prompt, code, cached=cached, digests=tuple(digests or ()))
            if disk is not None and (best is None or len(disk.ids) > len(best.ids)):
                return disk
        return best

    def _engine(self, slot: int, start: int, length: int, eager: bool = False):
        """Bind views without replacing any pointer captured by the original slot-zero graphs."""
        base = self.owner.e
        if slot == 0 and start == 0 and not eager:
            return base
        if self.slots[slot] is None:
            self.slots[slot] = State(self.w, 0, self.max_rows)
        st = copy.copy(self.slots[slot])
        # Two index-pool fence rows occupy eight tokens of the reservation. All
        # requested rows fit before those fences; verification never exceeds them.
        st.capacity = length - 8
        end = start + st.capacity
        parent = base.st
        st.kc = [c[start:end] for c in parent.kc]
        st.vc = [c[start:end] if c is not None else None for c in parent.vc]
        if hasattr(parent, "mtp_kc"):
            st.mtp_kc = parent.mtp_kc[start:end]
            st.mtp_vc = parent.mtp_vc[start:end] if parent.mtp_vc is not None else None
        if parent.index is not None:
            st.index = [(ik[start:end], ig[start:end], pk[start // 4:(start + length) // 4])
                        for ik, ig, pk in parent.index]
        e = copy.copy(base)
        e.st, e.buf, e.graphs = st, self.buf, None
        e.last_hidden = None
        e.replays = {key: 0 for key in base.replays}
        return e

    def _drafter(self, slot: int, start: int, length: int, eager: bool = False):
        parent = self.owner.drafter
        if parent is None or (slot == 0 and start == 0 and not eager):
            return parent
        bank = getattr(self.owner, "draft_graph_bank", None)
        if bank is not None:
            # Ring addresses depend only on the slot, even when the target's
            # reservation moved. No capture or collective is allowed here.
            d = copy.copy(bank[slot])
            d.context_end = 0
            if self.observer.enabled:
                d.observe = True
            return d
        d = copy.copy(parent)
        if getattr(parent, "ring", 0):
            first, last = slot * parent.ring, (slot + 1) * parent.ring
        else:
            first, last = start, start + length
        d.kc = [c[:, first:last] for c in parent.kc]
        d.vc = [c[:, first:last] for c in parent.vc]
        d.pos_dev = torch.zeros_like(parent.pos_dev)
        d.context_end = 0
        d.block_graph, d.tap_graphs = None, {}
        # _dattn_kernel uses CAP as the head stride. A dimension-one slice keeps
        # the parent's head stride, even though its visible length is smaller.
        d.cap = parent.cap
        if self.observer.enabled:
            d.observe = True
        return d

    def held_bytes(self) -> int:
        return sum(snapshot_bytes(c) for c in self.cache)

    def _remember(self, s: Stream) -> Snapshot | None:
        """Budget the entire prompt copy before allocating its recurrent or KV tensors."""
        if s.sid in self.filling or self.fill_owner == s.sid or self.express_owner == s.sid:
            raise RuntimeError("an incomplete GLM prefill cannot be snapshotted")
        if self.owner.cache_entries <= 0 or self.owner.cache_bytes <= 0:
            return
        e, st = s.engine, s.st
        d = s.drafter if s.use_dflash else None
        pending = e.last_hidden if s.use_mtp else None
        probe = Snapshot(list(s.prompt), st.rec[st.cur[0]] if st.cur else st.rec[0], st.conv, pending,
                         st.mtp_len - st.mtp_drafted if pending is not None else -1,
                         d.context_end if d is not None else -1)
        probe.image_digests = getattr(s, "image_digests", tuple(
            getattr(image, "digest", None) for image in getattr(s, "images", None) or ()))
        # Exact replay's head row and hidden row: budgeted here by size, copied only once the entry is kept.
        logits = e.last_logits if p1.EXACT_REPLAY and not s.use_mtp else None
        head_need = 0
        if logits is not None:
            head_need = head_width(self.w, logits.shape[-1]) * logits.element_size() + (
                e.last_hidden.numel() * e.last_hidden.element_size() if e.last_hidden is not None else 0)
        old = next((c for c in reversed(self.cache) if c.rows is not None and c.ids == probe.ids[:len(c.ids)]
                    and c.image_digests == probe.image_digests[:len(c.image_digests)]
                    and c.drafter_end == len(c.ids) and c.mtp_len < 0
                    and getattr(d, "ring", 0)), None)
        from .decode import ring_bytes, extend_rows
        need = snapshot_bytes(probe) + ring_bytes(d, len(probe.ids)) + row_bytes(e, probe, d) + head_need
        if old is not None:
            need -= old.nbytes
        old_held = snapshot_bytes(old) if old is not None else 0
        pinned = lambda kept: self.sessions.extra_rows(kept) if self.sessions is not None else 0
        parent = [old] if old is not None else []
        if need + old_held + pinned(parent) > self.owner.cache_bytes:
            return
        retained = [c for c in self.cache if c is not old and c.ids != probe.ids]
        while retained and (len(retained) >= self.owner.cache_entries or
                            sum(snapshot_bytes(c) for c in retained) + need + old_held
                            + pinned(retained + parent)
                            > self.owner.cache_bytes):
            retained.pop(0)
        self.cache = retained + ([old] if old is not None else [])
        snap = take_snapshot(e, s.prompt, pending, mtp=s.use_mtp, drafter=d)
        snap.image_digests = probe.image_digests
        if logits is not None:
            snap.last_logits, snap.head_cols = keep_head_row(self.w, logits)
            snap.last_hidden = e.last_hidden.clone() if e.last_hidden is not None else None
        if old is not None:
            # The parent's unused pool fence may now be committed, even on a
            # plain flags-off resume. Transfer the freshly computed live row.
            from .session_state import refresh_transferred_pool_fences
            refresh_transferred_pool_fences(e, old, d)
            extend_rows(e, old, snap, d)
        else:
            save_rows(e, snap, d)
        if getattr(s, "reply_prefill", False):
            snap.reply_base = s.reply_base
        self.cache = retained + [snap]
        return snap

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        return self._begin_admit(s, stepped=False)

    @torch.no_grad()
    def begin_admit(self, s: Stream) -> None:
        """Reserve and announce a request; prefill advances in named worker rounds."""
        return self._begin_admit(s, stepped=True)

    def _begin_admit(self, s: Stream, *, stepped: bool) -> None:
        if (self.fill_owner is not None or self.express_owner is not None) and (not stepped or hasattr(s, "quality_score_start")):
            raise ValueError("a sliced GLM chunk must finish before synchronous admission")
        if hasattr(s, "quality_score_start"):
            from .quality_score import score_prefill

            try:
                self._send([SCORE, s.quality_score_start])
                self._send(list(s.prompt), payload=True)
                s.started = time.perf_counter()
                s.quality_result.extend(score_prefill(self.owner.e, s.prompt, s.quality_score_start))
                s.finished = time.perf_counter()
                s.done = True
            except Exception as exc:
                self.broken = exc
                raise
            return
        code, tokens = self._request(s)
        reply = getattr(s, "reply_prefill", False)
        copies = not reply and self.copy_control.enabled(s, code)
        cofilled = not reply and cofill.enabled(s)
        if type(cofilled) is not bool:
            raise ValueError("cofill must be a boolean")
        if reply and (not self.reply_prefill or not stepped or self.live()):
            raise ValueError("reply prefill requires an idle pooled worker")
        if not self.can_admit(s):
            raise ValueError("the request needs to wait for a free GLM token reservation")
        slot = next(i for i in range(len(self.slots)) if all(s.slot != i for s in self.streams.values()))
        sid = self.next_id
        if self.growth is not None:
            if time.monotonic() < self.growth.retry_at:
                return False
            tokens = self.growth.initial(s)
        snap = self._saved(s.prompt, code, images=getattr(s, "images", None), memory_only=reply) if s.draft else None
        cached = len(snap.ids) if snap is not None else 0
        if reply and cached < s.reply_base:
            raise ValueError("reply prefill requires its completed prompt in memory")
        del snap
        span = self.pool.allocate(sid, tokens)
        if span is None:
            raise RuntimeError("GLM reservation changed during admission")
        start, length = span
        # Allocation and view construction have no collectives and can fail safely
        # before the admission message. Followers construct their matching views.
        try:
            e = self._engine(slot, start, length)
            d = self._drafter(slot, start, length)
            samp = pack_sampling(s.sampling)
            step_rows = min(PREFILL_BUSY_ROWS, getattr(e, "prefill_rows", PREFILL_BUSY_ROWS))
            if stepped and self.express_buf is not None and len(s.prompt) - cached <= self.express_buf.rows:
                step_rows = min(step_rows, self.express_buf.rows)
        except Exception:
            self.pool.release(sid)
            raise
        if self.growth is not None:
            from .reply_reservations import CREDIT

            if not self.growth.command(CREDIT, sid, len(s.prompt) + s.count):
                self.pool.release(sid)
                return False
            if not self.streams:
                self.growth.drain, self.growth.error = False, None
        s.sid = sid
        s.slice_layers = getattr(s, "prefill_slice_layers", None)
        if s.slice_layers is None:
            s.slice_layers = self.prefill_slice_layers
        automatic_slicing = self.express_buf is not None and not s.slice_layers
        if automatic_slicing:
            s.slice_layers = 1             # the express lane needs bounded layer yields
        s.slice_auto = automatic_slicing
        # An automatically sliced newcomer may still become an atomic express
        # prefill after the restore vote (which can only shrink ``cached``);
        # _admit makes the final cofill choice on every rank.
        express_cofill = (automatic_slicing and stepped and self.express_cofill
                          and len(s.prompt) - cached <= self.express_atomic_rows)
        cofilled = cofilled and (not s.slice_layers or express_cofill)
        self.next_id += 1
        try:
            s.image_digests = tuple(getattr(image, "digest", None) for image in getattr(s, "images", None) or ())
            digest_bytes = b"".join(digest if digest is not None else bytes(32) for digest in s.image_digests)
            kind = (cofill.ADMIT_COPY_COFILL if copies else cofill.ADMIT_COFILL) if cofilled else ADMIT_COPY if copies else ADMIT
            kind = ADMIT_REPLY if reply else kind
            self._send([kind, sid, s.count, int(s.draft), int(s.stop_eos),
                        slot, start, length, cached, int(stepped), step_rows, *code, *samp,
                        len(s.image_digests), *digest_bytes, *([s.reply_base] if reply else [])])
            self._send(list(s.prompt), payload=True)
            self._admit(s, slot, start, length, cached, code, e, d, stepped=stepped, step_rows=step_rows,
                        copies=copies, cofilled=cofilled)
            # Tiny prefills must join the decode batch before incumbents finish.
            # Decide after the session restore vote: a named hit can become a long cold miss.
            if automatic_slicing and getattr(s, "express", False) and len(s.prompt) - s.cached <= self.express_atomic_rows:
                s.slice_layers = 0
        except Exception as exc:
            self.broken = exc
            self.pool.release(sid)
            raise

    def _admit(self, s: Stream, slot: int, start: int, length: int, cached: int, code: list[int], e=None, d=None,
               *, stepped: bool = False, step_rows: int = 1024, copies: bool = False, cofilled: bool = False):
        if not stepped and (self.fill_owner is not None or self.express_owner is not None):
            raise RuntimeError("synchronous admission would overwrite a sliced GLM chunk")
        self.price_rounds.contaminate()
        t0 = time.perf_counter()
        s.slot, s.start, s.span = slot, start, length
        s.request_limit = len(s.prompt) + s.count
        s.reserved_tokens = self.growth.initial(s) if self.growth is not None else s.request_limit
        if self.growth is not None:
            s.reply_grows = s.reply_parks = 0
            s.reply_park_seconds = 0.0
            s.reply_initial_tokens = s.reserved_tokens
        s.engine = e if e is not None else self._engine(slot, start, length)
        if p1.EXACT_REPLAY:
            s.engine.last_logits = None     # slot zero may share owner.e; never keep another prompt's head row
        s.st = s.engine.st
        s.drafter = d if d is not None else self._drafter(slot, start, length)
        s.use_mtp, s.use_dflash = code[0] in (1, 2, 3), code[0] >= 10
        s.depth_policy = decode_policy(code)
        s.policy_code = list(code)
        s.priced = code[0] in (16, 17)
        s.price_mode = code[0]
        if self.rank == 0 and (s.priced or self.observer.enabled) and not self.segment_patterns_loaded:
            self.segment_patterns = segment_patterns(getattr(self.owner, "model_dir", None))
            self.segment_patterns_loaded = True
        carry = code[0] == 16 and bool(code[2] & 4)
        s.calibration = (copy.deepcopy(self.carried) if carry else Calibration(self.prior, copy_depth=self.max_rows - 1)) if \
            self.rank == 0 and s.priced else None
        s.segment_tracker = (SegmentTracker(self.segment_patterns, s.prompt)
                             if self.rank == 0 and (s.priced or self.observer.enabled) else None)
        s.segment = s.segment_tracker.segment if s.segment_tracker is not None else "text"
        s.segment_detection = bool(self.segment_patterns)
        if self.observer.enabled:
            s.prompt_sha256 = hashlib.sha256(json.dumps(s.prompt, separators=(",", ":")).encode()).hexdigest()
        if self.observer.enabled and self.rank == 0:
            s.pass_economics_rounds = []
        if s.drafter is not None:
            s.drafter.observe = self.observer.enabled or (self.rank == 0 and s.priced)
        kind, most, low, high = code
        prefix = "f" if kind >= 10 else ""
        kind %= 10
        s.policy = ("fp" + str(most) if code[0] == 16 else "fq" + str(most) if code[0] == 17 else
                    prefix + f"c{most}:{low / 1e6:g}" if kind == 3 else
                    prefix + f"a:{low / 1e6:g}:{high / 1e6:g}" if kind == 2 else
                    prefix + str(most) if kind == 1 else "0")
        s.eos = tuple(self.w.cfg.eos) if s.stop_eos else ()
        s.cached = cached
        s.cached_named = cached
        s.context = list(s.prompt)
        s.drafted_tokens = s.accepted_draft_tokens = 0
        if self.observer.enabled and s.drafter is not None:
            s.drafter.observe = True
        s.depths, s.keeps = [], []
        s.last = (0, 0)
        self.copy_control.admit(s, copies)
        reply = getattr(s, "reply_prefill", False)
        snap = self._saved(s.prompt, code, cached, getattr(s, "images", None),
                           getattr(s, "image_digests", None), memory_only=reply) if cached else None
        if self.recent_sessions is not None:
            s.session_key = hashlib.sha256(json.dumps(s.prompt[:64], separators=(",", ":")).encode()).hexdigest()[:16]
            s.reply_prefix_hit_tokens = 0
            s.cache_miss_reason = "none" if snap is not None else "boot"
            for old_prompt, old_reply, old_images in reversed(self.recent_sessions):
                common = next((i for i, (a, b) in enumerate(zip(old_prompt, s.prompt)) if a != b),
                              min(len(old_prompt), len(s.prompt)))
                if common == len(old_prompt):
                    tail = s.prompt[common:]
                    s.reply_prefix_hit_tokens = max(s.reply_prefix_hit_tokens, next(
                        (i for i, (a, b) in enumerate(zip(tail, old_reply)) if a != b),
                        min(len(tail), len(old_reply))))
                    if snap is None:
                        s.cache_miss_reason = ("image-changed" if old_images != s.image_digests
                                               else "store-evicted")
                    break
                if common >= min(64, len(old_prompt), len(s.prompt)) and snap is None:
                    s.cache_miss_reason = "compaction-like" if len(s.prompt) < len(old_prompt) else "fork"
        if reply:
            if snap is not None:
                load_rows(s.engine, snap, s.drafter)
            present = self.owner._gather_ints([int(snap is not None)])
            if not all(row[0] for row in present):
                # Never start a cold 100k prefill for an optional prediction.
                s.cached, s.fill_pos = 0, 0
                s.done, s.started, s.finished = True, time.perf_counter(), time.perf_counter()
                self.streams[s.sid] = s
                return
        elif cached and self.sessions is not None:
            selected = snap
            snap = self.sessions.restore(s, snap)
            cached = s.cached = len(snap.ids) if snap is not None else 0
            if snap is not None and any(c is selected for c in self.cache):
                self.cache = [c for c in self.cache if c is not selected] + [selected]
        else:
            if cached and snap is None:
                raise RuntimeError("a rank lost the named GLM prompt snapshot")
            if snap is not None:
                load_rows(s.engine, snap, s.drafter)
                self.cache = [c for c in self.cache if c is not snap] + [snap]
        # Decided after the session restore vote, which every rank takes: a failed vote is a cold miss.
        s.exact_replay = (p1.EXACT_REPLAY and not reply and s.cached == len(s.prompt)
                          and replay_ready(snap, s.prompt))
        if self.reply_prefill and not reply:
            s.reply_prefill_hit_tokens = max(0, s.cached - getattr(snap, "reply_base", s.cached))
        s.express = stepped and self.express_buf is not None and len(s.prompt) - cached <= self.express_buf.rows
        if s.express:
            # Slot zero may otherwise return owner.e itself. Never rebind the
            # owner's pbuf: existing long generators retain its live tensors.
            s.engine = copy.copy(s.engine)
            s.engine.pbuf = self.express_buf
            s.engine.prefill_rows = self.express_buf.rows
        if self.sessions is not None and not reply:
            source = "disk" if getattr(snap, "disk", None) is not None else "memory" if snap is not None else "cold"
            self.sessions.begin(s, source=source, failed_load=bool(cached == 0 and getattr(s, "cached_named", 0)))
        checkpoints = self.sessions.prefill_kwargs(s) if self.sessions is not None and not reply else {}
        if reply:
            checkpoints["sample_first"] = False
        # Every rank calls the overlay exchange after ADMIT, on the same worker.
        s.engine.images = self.owner._image_rows(s.prompt, cached, getattr(s, "images", None))
        if stepped:
            if not 1 <= step_rows <= s.engine.prefill_rows:
                raise RuntimeError("invalid GLM prefill chunk size")
            s.fill_pos, s.fill_rows = cached, step_rows
            s.fill_layer = s.fill_layer_stop = 0
            s.fill_stop = cached
            if self.express_buf is None:
                lane = not s.express and self.w.cfg.quant == "exl3"
            else:
                # Express on: only atomic express prompts cofill, on the express
                # buffer, for either quantization (rank zero then clears slicing).
                lane = (s.express and self.express_cofill
                        and len(s.prompt) - cached <= self.express_atomic_rows)
            eligible = (cofilled and not reply and lane and not s.use_mtp and s.engine.images is None
                        and not s.exact_replay)
            big = eligible and p1.COFILL_BIG and not s.express
            s.cofill_ready = eligible and not big and len(s.prompt) - cached <= step_rows
            if s.cofill_ready and checkpoints:
                mark = checkpoints["mark"](cached)
                s.cofill_ready = mark is None or mark >= len(s.prompt)
            s.cofill_snap = snap if s.cofill_ready else None
            s.fill_started = False
            if big:
                self.filling[s.sid] = cofill_big.Prefill(snap, checkpoints)
                s.cofill_stats = {"big": True, "passes": 0, "cofilled_passes": 0,
                                  "prompt_rows": 0, "max_total_rows": 0}
            elif s.exact_replay:
                self.filling[s.sid] = replay_steps(s.engine, s.prompt, s.sampling, snap,
                                                   drafter=s.drafter if s.use_dflash else None)
            else:
                self.filling[s.sid] = prefill_steps(s.engine, s.prompt, s.sampling, mtp=s.use_mtp,
                                                  drafter=s.drafter if s.use_dflash else None, resume=snap,
                                                  rows=lambda: s.fill_rows, slice_end=lambda: s.fill_layer_stop,
                                                  **checkpoints)
            self.streams[s.sid] = s
            s.prefill_started = t0
            s.fill_served = t0
            return
        try:
            with self.observer.phase("prefill", stream_id=s.sid):
                if s.exact_replay:
                    first = replay_first(s.engine, s.prompt, s.sampling, snap,
                                         drafter=s.drafter if s.use_dflash else None)
                else:
                    first = prefill(s.engine, s.prompt, s.sampling, mtp=s.use_mtp,
                                    drafter=s.drafter if s.use_dflash else None, resume=snap, **checkpoints)
        finally:
            s.engine.images = None
        # Do not retain a restored snapshot through cache eviction and replacement.
        del snap
        # An exact replay keeps the entry it restored (already moved to most recent); no second copy.
        snapshot = self._remember(s) if s.draft and not s.exact_replay else None
        if s.segment_tracker is not None:
            s.segment = s.segment_tracker.feed([first])
        if self.sessions is not None:
            self.sessions.completed(s, snapshot)
        self._propose(s, first, s.engine.last_hidden, [first], s.count - 1)
        s.prefill_s, s.started = time.perf_counter() - t0, time.perf_counter()
        self.streams[s.sid] = s
        s.take([first], s.eos)

    def _prefill_budget(self, s: Stream, *, queued: bool = False) -> int:
        decoding = any(not other.done and other.sid not in self.filling for other in self.streams.values())
        if self.prefill_rows_idle:
            return min(s.engine.prefill_rows, self.prefill_rows_busy if decoding else self.prefill_rows_idle)
        return (s.engine.prefill_rows if p1.ATTENTION_TILES else
                min(s.engine.prefill_rows, PREFILL_BUSY_ROWS if queued or decoding or len(self.filling) > 1
                    else PREFILL_IDLE_ROWS))

    def prefill_step(self, *, queued: bool = False) -> list[Stream]:
        """Advance the chunk owner, or select a prompt by remaining work and optional priority/aging."""
        if not self.filling:
            return []
        cancelled = []
        for sid in list(self.filling):
            s = self.streams[sid]
            check = getattr(s, "cancelled", None)
            if check is None:
                continue
            try:
                stop = check()
            except Exception as exc:
                s.error, stop = exc, True
            if stop:
                s.done, s.finished = True, time.perf_counter()
                cancelled.append(s)
                self.filling.pop(sid).close()
        # Followers still own the cancelled partial chunk until DONE. Do not send
        # another stream's FILL before the scheduler sends that release.
        if any(owner is not None and owner not in self.filling for owner in (self.fill_owner, self.express_owner)):
            return cancelled
        if not self.filling:
            return cancelled
        now = time.perf_counter()
        if self._gathering(now):
            time.sleep(0.001)  # rank zero sends nothing; followers keep waiting for the next command
            return cancelled

        def order(i):
            stream = self.streams[i]
            score = len(stream.prompt) - stream.fill_pos
            if self.fair_schedule:
                score += PREFILL_PRIORITY_ROWS[getattr(stream, "priority", None)]
                score -= PREFILL_AGE_ROWS_PER_SECOND * (now - stream.fill_served)
            return score, i

        # Each lane retains its owner across slices. An eligible short prompt can
        # pass a suspended long chunk using its own complete buffer set.
        express = [i for i in self.filling if self.streams[i].express]
        sid = (self.express_owner if self.express_owner is not None else min(express, key=order) if express else
               self.fill_owner if self.fill_owner is not None else min(self.filling, key=order))
        s = self.streams[sid]
        pieces = cofill_big.plan(self, s)
        if pieces:
            self._send([cofill_big.FILL_BIG, len(pieces), *[x for piece in pieces for x in piece]])
            try:
                with self.observer.phase("prefill"):
                    return cancelled + cofill_big.run(self, pieces)
            except Exception as exc:
                self.broken = exc
                raise
        group = cofill.candidates(self, s)
        if group:
            self._send([cofill.FILL_COFILL, len(group), *[p.sid for p in group]])
            try:
                with self.observer.phase("prefill"):
                    return cancelled + cofill.run(self, [p.sid for p in group])
            except Exception as exc:
                self.broken = exc
                raise
        rows = self._prefill_budget(s, queued=queued)
        if getattr(s, "reply_prefill", False):
            rows = min(rows, self.reply_prefill_rows)
        lane_owner = self.express_owner if s.express else self.fill_owner
        stop = s.fill_stop if lane_owner is not None else min(len(s.prompt), s.fill_pos + rows)
        if self.sessions is not None and lane_owner is None and not getattr(s, "reply_prefill", False):
            stop = self.sessions.stop(s, s.fill_pos, stop)
        if getattr(s, "exact_replay", False):
            stop, layer_stop = len(s.prompt), 0     # a zero-row FILL: no forward to slice
        else:
            if (not s.slice_layers and s.express and getattr(s, "slice_auto", False) and lane_owner is None
                    and stop - s.fill_pos > self.express_busy_rows and cofill.decoding(self)):
                s.slice_layers = 1  # too large to pause live replies in one pass: slice this prompt (rank zero)
            layer_stop = (min(len(self.w.layers), s.fill_layer + self._slice_step(s, stop - s.fill_pos))
                          if s.slice_layers else 0)
        try:
            self._send([FILL_SLICE, sid, stop, layer_stop] if layer_stop else [FILL, sid, stop])
            self._fill(sid, stop, layer_stop)
        except Exception as exc:
            self.broken = exc
            raise
        return cancelled + ([s] if s.done else [])

    def _gathering(self, now: float) -> bool:
        """Idle gather: wait (bounded) for more atomic express arrivals before the first cofill pass."""
        if not self.express_gather_s or not self.express_cofill or cofill.decoding(self):
            return False
        waiting = [self.streams[i] for i in self.filling]
        if not all(s.express and getattr(s, "cofill_ready", False) for s in waiting):
            return False  # never delay a long or sliced prompt
        if len(self.streams) >= len(self.slots):
            return False
        if sum(len(s.prompt) - s.fill_pos for s in waiting) >= cofill.EXPRESS_IDLE_ROWS:
            return False
        return now - min(s.prefill_started for s in waiting) < self.express_gather_s

    def _slice_step(self, s: Stream, rows: int) -> int:
        """Layers for this step: explicit slices as requested; automatic ones by the estimated ms budget."""
        if not getattr(s, "slice_auto", False) or not self.prefill_slice_ms:
            return s.slice_layers
        layers = len(self.w.layers)
        per_layer = (p1.SLICE_FIXED_MS + 1000.0 * rows / p1.SLICE_ROWS_PER_S) / layers
        return max(1, min(layers, int(self.prefill_slice_ms / per_layer)))

    @torch.no_grad()
    def _fill(self, sid: int, stop: int, layer_stop: int = 0) -> None:
        self.price_rounds.contaminate()
        s = self.streams[sid]
        if isinstance(self.filling.get(sid), cofill_big.Prefill):
            raise RuntimeError("big cofill requires rank-zero chunk assignments")
        s.cofill_ready, s.cofill_snap = False, None
        rows = stop - s.fill_pos
        replay = getattr(s, "exact_replay", False)
        if replay and (rows or stop != len(s.prompt) or layer_stop):
            raise RuntimeError("an exact replay finishes in one zero-row step")
        if not (replay or 1 <= rows <= s.engine.prefill_rows) or stop > len(s.prompt):
            raise RuntimeError("invalid GLM prefill stop position")
        owner_attr = "express_owner" if s.express else "fill_owner"
        lane_owner = getattr(self, owner_attr)
        if lane_owner is not None and (sid != lane_owner or stop != s.fill_stop or not layer_stop):
            raise RuntimeError("a sliced GLM chunk owns the prefill buffers until completion")
        if layer_stop:
            if not s.fill_layer < layer_stop <= len(self.w.layers):
                raise RuntimeError("invalid GLM prefill layer stop")
            setattr(self, owner_attr, sid)
        s.fill_stop, s.fill_layer_stop = stop, layer_stop
        s.fill_rows = rows
        s.fill_started = True
        try:
            with self.observer.phase("prefill", stream_id=sid):
                at = next(self.filling[sid])
            if isinstance(at, tuple):
                if at != (stop, layer_stop) or not 0 < layer_stop < len(self.w.layers) or s.st.pos != s.fill_pos:
                    raise RuntimeError("GLM prefill slice positions differ across ranks")
                s.fill_layer = layer_stop
                return
            if at != stop or s.st.pos != stop:
                raise RuntimeError("GLM prefill positions differ across ranks")
            s.fill_pos = stop
            setattr(self, owner_attr, None)
            s.fill_layer = 0
            s.fill_served = time.perf_counter()
        except StopIteration as result:
            if stop != len(s.prompt) or s.st.pos != stop:
                raise RuntimeError("GLM prefill ended at the wrong prompt position")
            s.fill_pos = stop
            setattr(self, owner_attr, None)
            s.fill_layer = 0
            del self.filling[sid]
            s.engine.images = None
            snapshot = self._remember(s) if s.draft and not replay else None
            if getattr(s, "reply_prefill", False):
                s.prefill_s = time.perf_counter() - s.prefill_started
                s.done, s.started, s.finished = True, time.perf_counter(), time.perf_counter()
                if self.rank == 0:
                    self._reply_prefill_receipt(s, "completed" if any(c.ids == s.prompt for c in self.cache)
                                               else "store-full")
                return
            if self.sessions is not None:
                self.sessions.completed(s, snapshot)
            first = result.value
            if s.segment_tracker is not None:
                s.segment = s.segment_tracker.feed([first])
            self._propose(s, first, s.engine.last_hidden, [first], s.count - 1)
            s.prefill_s, s.started = time.perf_counter() - s.prefill_started, time.perf_counter()
            s.take([first], s.eos)

    def _propose(self, s: Stream, pending: int, hidden, new: list[int], room: int, *, quantum=False) -> None:
        if quantum:
            # _round already saved every committed target tap. Do not generate
            # proposals that the next one-token turn would immediately discard.
            s.drafts = []
            return
        depth = min(s.depth_policy.next(*s.last), room) if s.depth_policy is not None else 0
        if s.priced:
            s.generation_cap = depth if pending not in s.eos else 0
            s.useful_cap = min(s.generation_cap, max(0, room - 1))
        s.drafts = []
        if self.observer.enabled or s.priced:
            s.draft_observation = None
        if s.copy_enabled:
            self.copy_control.propose(s, pending, hidden, new, room, depth)
            return
        if depth <= 0 or pending in s.eos:
            return
        self._model_propose(s, pending, hidden, new, depth)

    def _model_propose(self, s: Stream, pending: int, hidden, new: list[int], depth: int) -> None:
        if s.use_dflash:
            if s.drafter.context_end + s.drafter.block > s.span:
                raise RuntimeError("DFlash2 block would cross its token reservation")
            confidence = 0.0 if s.priced else s.depth_policy.confidence
            s.drafts = s.drafter.propose(pending, depth, s.sampling, confidence)
        elif s.use_mtp:
            if (self.growth is not None and
                    s.st.mtp_len - s.st.mtp_drafted + len(new) + depth - 1 > s.span - 8):
                raise RuntimeError("MTP proposal would cross its token reservation")
            s.drafts = draft(s.engine, hidden, new, s.st.pos + 1, depth, s.sampling, s.depth_policy.confidence)
        if self.observer.enabled or s.priced:
            s.draft_observation = getattr(s.drafter, "last_observation", None) if s.use_dflash else None

    def _priced_depths(self, live: list[Stream]) -> tuple[list[int], list[int]]:
        priced = [s for s in live if s.priced]
        depths = {s.sid: len(s.drafts) for s in live}
        sources = {s.sid: SRC_N for s in live}
        if not priced:
            return [depths[s.sid] for s in live], [sources[s.sid] for s in live]
        offers = []
        for s in priced:
            sources[s.sid] = SRC_D
            claims = (s.draft_observation or {}).get("claims", [])[:len(s.drafts)]
            if len(claims) != len(s.drafts):
                raise RuntimeError("priced draft has no claim for every proposed row")
            raw = s.price_mode == 16 and bool(s.policy_code[2] & 1)
            segments = s.segment_tracker.row_segments(s.drafts) if s.segment_tracker is not None else ["text"] * len(claims)
            s.priced_segments = segments
            chances = s.calibration.chances_rows(segments, claims, raw=raw)
            s.priced_chances = chances
            s.price_ratios = ([1.0] * len(claims) if raw else
                              [s.calibration.ratio(seg, i) for i, seg in enumerate(segments)])
            if s.price_mode == 17:
                depths[s.sid] = threshold_depth(chances, s.policy_code[2] / 1e6)
            else:
                offers.append({"sid": s.sid, "pass_chances": cumulative(chances[:s.useful_cap])})
        if offers:
            base = len(live) + sum(depths[s.sid] for s in live if not s.priced or s.price_mode == 17)
            other = sum(1 + (s.accepted_draft_tokens / max(1, s.rounds))
                        for s in live if not s.priced or s.price_mode == 17)
            context = context_class(max(s.st.pos for s in live))
            original = len(live) == 1 and live[0].engine is self.owner.e
            path_at = (lambda rows: path_for(self.owner.e, rows)) if original else (lambda rows: "eager")
            try:
                for offer in offers:
                    s = self.streams[offer["sid"]]
                    copied, offer["pass_chances"] = self.copy_control.joint_offer(
                        s, offer["pass_chances"], base_rows=base, base_expected=len(offers) + other,
                        context=context, load=len(live), path_for_rows=path_at)
                    if copied:
                        sources[s.sid] = SRC_C
                chosen, detail = choose_depths(offers, self.price_table, context=context, load=len(live),
                                               base_rows=base, other_expected=other,
                                               solo=any(s.policy_code[2] & 2 for s in priced if s.price_mode == 16),
                                               path_for_rows=path_at)
            except ValueError as exc:
                # A present but incomplete table is never extrapolated. The cap chain
                # is the same prefix as fc7:0.3, so rank zero can name that cut.
                chosen = {s.sid: threshold_depth((s.draft_observation or {})["claims"][:len(s.drafts)], .3)
                          for s in priced if s.price_mode == 16}
                detail = {"fallback": "fc7:0.3", "reason": str(exc)}
                for s in priced:
                    s.price_coverage_fallback = True
                    if s.price_mode == 16:
                        sources[s.sid] = SRC_F
            depths.update(chosen)
            for s in priced:
                s.price_decision = detail
            assert base + sum(chosen.values()) == sum(1 + depths[s.sid] for s in live)
        for s in live:
            s.priced_source = sources[s.sid]
        return [depths[s.sid] for s in live], [sources[s.sid] for s in live]

    @staticmethod
    def _apply_instruction(live: list[Stream], sources: list[int], depths: list[int]) -> None:
        for s, source, depth in zip(live, sources, depths):
            if source not in ((SRC_D, SRC_C, SRC_F) if s.priced else (SRC_N,)):
                raise RuntimeError("priced draft source differs from the admitted policy")
            chain = s.drafts if source in (SRC_N, SRC_D, SRC_F) else getattr(s, "copy_drafts", None)
            cap = getattr(s, "copy_cap", 0) if source == SRC_C else getattr(s, "generation_cap", len(s.drafts))
            if chain is None or depth < 0 or depth > len(chain) or (s.priced and depth > cap):
                raise RuntimeError("priced draft source or depth exceeds the local proposal")
            s.drafts = chain[:depth]
            CopyController.applied(s, source, depth)

    def _cancel_requests(self):
        cancelled = []
        for s in self.streams.values():
            check = getattr(s, "cancelled", None)
            if s.done or check is None:
                continue
            try:
                stop = check()
            except Exception as exc:
                # A client callback failure ends only its request. It must still
                # receive the normal DONE message, so followers release its span.
                s.error, stop = exc, True
            if stop:
                s.done, s.finished = True, time.perf_counter()
                cancelled.append(s)
        return cancelled

    @torch.no_grad()
    def round(self) -> list[Stream]:
        self._check()
        previous_done = {s.sid for s in self.streams.values() if s.done} if self.growth is not None else set()
        cancelled = self._cancel_requests()
        if self.growth is not None:
            try:
                self.growth.service()
            except Exception as exc:
                self.broken = exc
                raise
            self._cancel_requests()  # a client may leave during bounded disk IO
            cancelled = [s for s in self.streams.values() if s.done and s.sid not in previous_done]
        live = [s for s in self.streams.values() if not s.done and s.sid not in self.filling
                and (self.growth is None or self.growth.safe(s))]
        if not live:
            if self.growth is not None and not self.filling and not cancelled:
                time.sleep(0.05)  # storage backpressure keeps requests open without busy-spinning
            return cancelled
        quantum = (p1.PREFILL_DECODE_QUANTUM and all(self._quantum_eligible(s) for s in live)
                   and any(not self.streams[sid].done and self.streams[sid].slice_layers
                           and not getattr(self.streams[sid], "reply_prefill", False) for sid in self.filling))
        if quantum:
            self.prefill_decode_ticks += 1
            if (self.prefill_decode_ticks - 1) % p1.PREFILL_DECODE_QUANTUM:
                for s in live:
                    s.prefill_quantum_skips = getattr(s, "prefill_quantum_skips", 0) + 1
                return cancelled
        else:
            self.prefill_decode_ticks = 0
        try:
            sids = [s.sid for s in live]
            priced = any(s.priced for s in live)
            if quantum:
                self.price_rounds.contaminate()
                self._send([ROUND_QUANTUM, len(live), *sids])
            elif priced:
                depths, sources = self._priced_depths(live)
                self.price_rounds.begin(live, context=context_class(max(s.st.pos for s in live)),
                                        slot="zero" if len(live) == 1 and live[0].engine is self.owner.e else "nonzero")
                self._send([ROUND2, len(live), *sids, *sources, *depths])
                self._apply_instruction(live, sources, depths)
            else:
                self.price_rounds.begin(live, context=context_class(max(s.st.pos for s in live)),
                                        slot="zero" if len(live) == 1 and live[0].engine is self.owner.e else "nonzero")
                self._send([ROUND, len(live), *sids])
            with self.observer.phase("decode", stream_id=None):
                if quantum:
                    self._round(live, quantum=True)
                else:
                    self._round(live)
            self.round_id += 1
        except Exception as exc:
            self.broken = exc
            raise
        return cancelled + [s for s in live if s.done]

    @staticmethod
    def _quantum_eligible(s: Stream) -> bool:
        # MTP, copy and priced policies retain their existing cache/controller
        # semantics. A mixed batch falls back as a whole to ordinary rounds.
        return not s.use_mtp and not s.copy_enabled and not s.priced

    def _round(self, live: list[Stream], *, quantum=False) -> None:
        if any(s.sid in self.filling for s in live):
            raise RuntimeError("a partially prefilled GLM stream cannot decode")
        if quantum:
            if not p1.PREFILL_DECODE_QUANTUM or not all(self._quantum_eligible(s) for s in live):
                raise RuntimeError("GLM one-token prefill turn has an ineligible policy or disabled flag")
            for s in live:
                s.drafts = []
                s.prefill_quantum_rounds = getattr(s, "prefill_quantum_rounds", 0) + 1
        observer = self.observer
        observer.begin_round()
        t0 = observer.clock()
        replays = dict(self.owner.e.replays) if observer.enabled else None
        context_tokens = {str(s.sid): s.st.pos for s in live} if observer.enabled else None
        windows = [(s.st, [s.out[-1], *s.drafts]) for s in live]
        for s, (_, tokens) in zip(live, windows):
            if s.st.pos + len(tokens) > s.reserved_tokens:
                raise RuntimeError("target verification would cross its token reservation")
        original_engine = len(live) == 1 and live[0].engine is self.owner.e
        prof.decode_timer = observer.block if observer.block_sample else None
        try:
            if original_engine:
                b = self.owner.e.buf
                logits = self.owner.e.forward(windows[0][1])
                segs = [(live[0].st, 0, len(windows[0][1]))]
            else:
                b = self.buf
                segs = stage_streams(self.w, b, windows)
                logits = compute_streams(self.w, segs, b, eager=True)
        finally:
            prof.decode_timer = None
        t1 = observer.clock()
        actual_path = path_for(self.owner.e, segs[-1][2]) if original_engine else "eager"
        self.price_rounds.executed(segs[-1][2], actual_path)
        rows = [list(range(a0, a1)) for _, a0, a1 in segs]
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled, _ = sample_streams(self.w, logits, rows, positions, [s.sampling for s in live])
        t2 = observer.clock()
        kept = []
        diagnostics = []
        captured_rows = []
        for s, (_, tokens), (st, a0, a1), got in zip(live, windows, segs, sampled):
            keep = 1
            for i, token in enumerate(tokens[1:]):
                if got[i] != token or got[i] in s.eos:
                    break
                keep += 1
            raw_keep = keep
            mismatch = (raw_keep < len(tokens) and got[raw_keep - 1] != tokens[raw_keep])
            keep = min(keep, s.count - len(s.out))
            rejected = keep - 1 if mismatch and keep == raw_keep else None
            if s.calibration is not None:
                claims = (getattr(s, "draft_observation", None) or {}).get("claims", [])[:len(tokens) - 1]
                segments = (getattr(s, "priced_segments", None) or
                            s.segment_tracker.row_segments(tokens[1:]))[:len(claims)]
                s.last_outcomes = s.calibration.update_rows(
                    segments, claims, tokens[1:], got, s.count - len(s.out), s.eos)
            if observer.enabled:
                info = getattr(s, "draft_observation", None) or {}
                candidates = info.get("candidates", [])
                row_segments = (s.segment_tracker.row_segments(tokens[1:]) if s.segment_tracker is not None
                                else ["text"] * (len(tokens) - 1))
                rank = (candidates[rejected].index(got[rejected]) + 1
                        if rejected is not None and rejected < len(candidates)
                        and got[rejected] in candidates[rejected] else None)
                outcomes = (s.last_outcomes if s.calibration is not None else
                            ["match"] * (keep - 1) + (["mismatch"] if rejected is not None else []) +
                            ["censored"] * (len(tokens) - 1 - (keep - 1) - int(rejected is not None)))
                diagnostics.append({"sid": s.sid, "rows": len(tokens), "accepted": keep - 1,
                                    "claims": info.get("claims", [])[:len(tokens) - 1],
                                    "claim_convention": info.get("claim_convention", "selected-selector-softmax-conditional"),
                                    "calibrated_chances": getattr(s, "priced_chances", [])[:len(tokens) - 1],
                                    "calibration_ratios": getattr(s, "price_ratios", [])[:len(tokens) - 1],
                                    "generation_cap": getattr(s, "generation_cap", None),
                                    "useful_cap": getattr(s, "useful_cap", None),
                                    "copy_cap": getattr(s, "copy_cap", None),
                                    "selected_depth": len(tokens) - 1,
                                    "source": (getattr(s, "copy_source", SRC_N) if s.copy_enabled and not s.priced else
                                               getattr(s, "priced_source", SRC_N)),
                                    "price_decision": getattr(s, "price_decision", None),
                                    "accepted_by_position": outcomes,
                                    "rejected_position": rejected, "target_candidate_rank": rank,
                                    "target_candidate_status": ("present" if rank is not None else
                                                                "absent" if rejected is not None else "not_rejected"),
                                    "segment": s.segment, "segment_detected": s.segment_detection,
                                    "row_segments": row_segments,
                                    "family": getattr(s, "family", os.environ.get("TF_GLM_DECODE_FAMILY", "unknown")),
                                    "prompt_sha256": getattr(s, "prompt_sha256", None),
                                    "session_key": getattr(s, "session_key", None),
                                    "cache_miss_reason": getattr(s, "cache_miss_reason", None),
                                    "reply_prefix_hit_tokens": getattr(s, "reply_prefix_hit_tokens", None),
                                    "price_table": ("invalid" if getattr(s, "price_table_invalid", False) else
                                                    getattr(s, "price_table_missing", None) or
                                                    ("loaded" if s.priced else "inactive"))})
                if observer.sample_every and observer.round % observer.sample_every == 0:
                    for i, token in enumerate(tokens):
                        captured_rows.append({"sid": s.sid, "row": i, "tap_row": a0 + i,
                                              "prompt_sha256": s.prompt_sha256,
                                              "family": getattr(s, "family", os.environ.get("TF_GLM_DECODE_FAMILY", "unknown")),
                                              "segment": row_segments[i - 1] if i else s.segment, "load": len(live),
                                              "context": context_class(st.pos), "position": st.pos + i,
                                              "draft_token": token, "claim": info.get("claims", [])[i - 1]
                                              if i and i - 1 < len(info.get("claims", [])) else None,
                                              "candidate_top8": candidates[i - 1][:8] if i and i - 1 < len(candidates) else [],
                                              "target_pick": got[i],
                                              "outcome": "pending" if i == 0 else outcomes[i - 1]})
            self.copy_control.observe(s, keep, got)
            commit(self.w, st, b, a1 - a0, keep)
            if s.use_dflash:
                if s.drafter.context_end + keep > s.reserved_tokens:
                    raise RuntimeError("DFlash2 taps would cross their token reservation")
                taps = torch.cat([t[a0:a0 + keep] for t in b.taps], dim=1)
                s.drafter.add_taps(taps)
            s.counted(len(tokens))
            s.last = (len(tokens) - 1, keep - 1)
            s.drafted_tokens += s.last[0]
            s.accepted_draft_tokens += s.last[1]
            if s.segment_tracker is not None:
                s.segment = s.segment_tracker.feed(got[:keep])
            s.depths.append(s.last[0])
            s.keeps.append(keep)
            s.committed.extend(tokens[:keep])
            kept.append((s, a0, got[:keep]))
        t3 = observer.clock()
        taps = observer.capture(b.taps, segs[-1][2]) if observer.enabled and b.taps else None
        # MTP reads the target buffer before any following forward can reuse it.
        for s, a0, new in kept:
            args = (s, new[-1], b.fnormed[a0:a0 + len(new)], new, s.count - len(s.out) - len(new))
            if quantum:
                self._propose(*args, quantum=True)
            else:
                self._propose(*args)
        t4 = observer.clock()
        if observer.enabled:
            economics = {"round_id": self.round_id, "rows": segs[-1][2],
                         "active_set": [str(s.sid) for s in live], "context_tokens": context_tokens,
                         "path": actual_path, "timing_basis": "host-unsynchronized",
                         "target_forward_s": t1 - t0,
                         "sampling_commit_tap_s": t3 - t1,
                         "draft_generation_s": t4 - t3,
                         "round_wall_s": t4 - t0}
            if self.rank == 0:
                for s in live:
                    s.pass_economics_rounds.append(economics)
            observer.record({"concurrency": len(live), "verify_rows": segs[-1][2],
                             "context": context_class(max(s.st.pos for s in live)),
                             "path": actual_path,
                             "slot": "zero" if original_engine else "nonzero",
                             "ms": {"forward": 1000 * (t1 - t0), "sampling": 1000 * (t2 - t1),
                                    "commit_taps": 1000 * (t3 - t2), "drafter": 1000 * (t4 - t3)},
                             "graph_paths": ({k: self.owner.e.replays[k] - replays[k] for k in replays}
                                             if original_engine else None),
                             "streams": diagnostics, "pass_economics": economics,
                             **({"reply_reservations": self.growth.metrics()} if self.growth is not None else {})},
                            taps, captured_rows)
        for s, _, new in kept:
            s.take(new, s.eos)

    def finish(self, done: list[Stream]) -> None:
        done = [s for s in done if not hasattr(s, "quality_score_start")]
        if not done:
            self._writer_idle(not self.streams and self.broken is None)
            return
        self._check()
        self.price_rounds.flush()
        sids = [s.sid for s in done]
        try:
            self._send([DONE, len(sids), *sids])
            self._finish(sids)
        except Exception as exc:
            self.broken = exc
            raise

    def _finish(self, sids: list[int]) -> None:
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is not None and getattr(s, "reply_prefill", False) and self.rank == 0:
                if getattr(s, "fill_pos", 0) < len(s.prompt):
                    self._reply_prefill_receipt(s, "preempted" if s.cached else "prefix-miss")
            if s is not None and self.recent_sessions is not None and not getattr(s, "reply_prefill", False):
                self.recent_sessions.append((list(s.prompt), list(s.out), getattr(s, "image_digests", ())))
                self.recent_sessions = self.recent_sessions[-8:]
            if (s is not None and s.calibration is not None and s.price_mode == 16
                    and s.policy_code[2] & 4):
                self.carried = copy.deepcopy(s.calibration)
            continuation = self.filling.pop(sid, None)
            try:
                if continuation is not None:
                    continuation.close()
                if s is not None and s.engine is not None:
                    s.engine.images = None
                    s.cofill_snap = None
                    if self.sessions is not None:
                        self.sessions.finish(s, broken=self.broken is not None or getattr(s, "reply_prefill", False))
            finally:
                if self.fill_owner == sid:
                    self.fill_owner = None
                if self.express_owner == sid:
                    self.express_owner = None
                self.pool.release(sid)
                if self.growth is not None:
                    self.growth.finish(sid)
        if self.growth is not None and not self.streams:
            self.growth.drain, self.growth.error = False, None
        self._writer_idle(not self.streams and self.broken is None)

    @staticmethod
    def _reply_prefill_receipt(s: Stream, status: str) -> None:
        """Bounded metadata for useful/wasted-work gates; no prompt or reply text."""
        receipt = {
            "status": status, "candidate_sha256": getattr(s, "reply_fingerprint", ""),
            "base_tokens": s.reply_base, "candidate_tokens": len(s.prompt),
            "committed_tokens": max(0, getattr(s, "fill_pos", 0) - s.cached),
            "partial_layer": getattr(s, "fill_layer", 0),
            "elapsed_s": round(time.perf_counter() - getattr(s, "prefill_started", s.started), 6),
        }
        try:
            print("[tensorfold] reply_prefill: " + json.dumps(receipt, sort_keys=True), flush=True)
        except OSError:
            pass  # a closed log pipe must not strand followers awaiting DONE

    def drop(self) -> list[Stream]:
        self.price_rounds.contaminate()
        streams = list(self.streams.values())
        self._finish([s.sid for s in streams])
        return streams

    @torch.no_grad()
    def follow(self) -> None:
        """Mirror only rank zero's named work; malformed or failed work poisons the decoder."""
        self._check()
        try:
            while True:
                if not self.streams:
                    self._writer_idle(True)
                bell = getattr(self.owner, "follower_doorbell", None)
                if bell is not None and not self.streams:
                    bell.wait()
                    self._writer_idle(False)  # stop DMA before the idle admission collective
                msg = self.share(None)
                self._writer_idle(False)
                if not msg:
                    raise RuntimeError("empty GLM worker message")
                if msg[0] == RESERVATION_MESSAGE:
                    if self.growth is None:
                        raise RuntimeError("reply growth message on a flag-off rank")
                    self.growth.follow(msg)
                elif msg[0] == SCORE:
                    if len(msg) != 2 or not self.owner.quality_score_enabled:
                        raise RuntimeError("invalid GLM quality scoring message")
                    from .quality_score import score_prefill

                    score_prefill(self.owner.e, self.share(None), msg[1])
                elif msg[0] in (ADMIT, ADMIT_REPLY, ADMIT_COPY, cofill.ADMIT_COFILL, cofill.ADMIT_COPY_COFILL):
                    reply = msg[0] == ADMIT_REPLY
                    count_at = 15 + SAMPLING_INTS
                    if (len(msg) < count_at + 1 or len(msg) != count_at + 1 + 32 * msg[count_at] + int(reply)
                            or reply and not self.reply_prefill):
                        raise RuntimeError("invalid GLM admission message")
                    sid, count, drafts, stop, slot, start, length, cached, stepped, step_rows = msg[1:11]
                    code = msg[11:15]
                    s = Stream(self.share(None), count, unpack_sampling(msg[15:count_at]), draft=bool(drafts),
                               stop_eos=bool(stop))
                    s.image_digests = tuple(bytes(msg[count_at + 1 + 32 * i:count_at + 33 + 32 * i])
                                            for i in range(msg[count_at]))
                    s.sid = sid
                    if reply:
                        s.reply_prefill, s.reply_base = True, msg[-1]
                        if not stepped or not 0 < s.reply_base <= cached < len(s.prompt) or self.live():
                            raise RuntimeError("invalid GLM reply prefill admission")
                    if self.growth is not None and sid not in self.growth.store.store.live_credits:
                        raise RuntimeError("reply admission has no backed live-state credit")
                    tokens = self.growth.initial(s) if self.growth is not None else len(s.prompt) + count
                    span = self.pool.allocate(sid, tokens)
                    if span != (start, length):
                        raise RuntimeError("GLM token reservations differ across ranks")
                    self.next_id = sid + 1
                    self._admit(s, slot, start, length, cached, code, stepped=bool(stepped), step_rows=step_rows,
                                copies=msg[0] in (ADMIT_COPY, cofill.ADMIT_COPY_COFILL),
                                cofilled=msg[0] in (cofill.ADMIT_COFILL, cofill.ADMIT_COPY_COFILL))
                elif msg[0] == cofill_big.FILL_BIG:
                    if len(msg) < 2 or msg[1] < 1 or len(msg) != 2 + 2 * msg[1]:
                        raise RuntimeError("invalid GLM big cofill message")
                    with self.observer.phase("prefill"):
                        cofill_big.run(self, list(zip(msg[2::2], msg[3::2])))
                elif msg[0] == cofill.FILL_COFILL:
                    if len(msg) < 2 or len(msg) != 2 + msg[1]:
                        raise RuntimeError("invalid GLM cofill message")
                    with self.observer.phase("prefill"):
                        cofill.run(self, msg[2:])
                elif msg[0] in (FILL, FILL_SLICE):
                    if len(msg) != (4 if msg[0] == FILL_SLICE else 3) or msg[1] not in self.filling:
                        raise RuntimeError("invalid GLM prefill step message")
                    if msg[0] == FILL_SLICE and msg[3] <= 0:
                        raise RuntimeError("invalid GLM prefill layer stop")
                    self._fill(msg[1], msg[2], msg[3] if msg[0] == FILL_SLICE else 0)
                elif msg[0] in (ROUND, ROUND2, ROUND_QUANTUM, DONE):
                    if len(msg) < 2 or msg[1] < 1:
                        raise RuntimeError("invalid GLM round or finish message")
                    if msg[0] in (ROUND, ROUND2, ROUND_QUANTUM):
                        n = msg[1]
                        priced = msg[0] == ROUND2
                        if len(msg) != (2 + 3 * n if priced else 2 + n):
                            raise RuntimeError("invalid GLM round layout")
                        sids = msg[2:2 + n]
                        if len(set(sids)) != n or any(sid not in self.streams for sid in sids):
                            raise RuntimeError("invalid GLM round stream ids")
                        live = [self.streams[sid] for sid in sids]
                        # Rank zero may have cancelled another stream this round;
                        # its DONE follows the survivor's ROUND. Named ready streams
                        # are authoritative for both the legacy and priced frame.
                        if any(s.done or s.sid in self.filling for s in live):
                            raise RuntimeError("GLM round names a finished or filling follower stream")
                        if self.growth is not None:
                            if any(s.sid in self.growth.parked or not self.growth.safe(s) for s in live):
                                raise RuntimeError("round names a parked or under-reserved response")
                        if priced:
                            sources = msg[2 + n:2 + 2 * n]
                            depths = msg[2 + 2 * n:2 + 3 * n]
                            self._apply_instruction(live, sources, depths)
                        elif any(s.priced for s in live):
                            raise RuntimeError("priced draft round has no rank-zero depth decision")
                        with self.observer.phase("decode", stream_id=None):
                            if msg[0] == ROUND_QUANTUM:
                                self._round(live, quantum=True)
                            else:
                                self._round(live)
                        self.round_id += 1
                    else:
                        if len(msg) != 2 + msg[1] or len(set(msg[2:])) != msg[1]:
                            raise RuntimeError("invalid GLM finish message")
                        self._finish(msg[2:])
                else:
                    raise RuntimeError(f"unknown GLM worker message {msg[0]}")
        except Exception as exc:
            self.broken = exc
            raise
