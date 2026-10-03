"""Default-off physical reply growth with rank-named, lossless pressure parking."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from .live_store import ABI, LiveStore

MESSAGE = 9  # ADMIT_REPLY owns 8; growth transactions must have a distinct opcode.
CREDIT, GROW, PREPARE, PARK, LOAD, RESUME = range(1, 7)


@dataclass
class ReplyConfig:
    enabled: bool = False
    block: int = 2048

    @classmethod
    def from_env(cls, parallel, sessions):
        value = os.environ.get("TF_GLM_GROW_REPLIES", "0")
        if value not in ("0", "1"):
            raise ValueError("TF_GLM_GROW_REPLIES must be 0 or 1")
        cfg = cls(value == "1", int(os.environ.get("TF_GLM_REPLY_BLOCK_TOKENS", "2048")))
        if cfg.block < 64 or cfg.block % 4:
            raise ValueError("TF_GLM_REPLY_BLOCK_TOKENS must be a multiple of four, at least 64")
        if cfg.enabled and (parallel < 2 or not sessions.disk):
            raise ValueError("reply growth requires pooled serving and TF_GLM_SESSION_DISK=1")
        return cfg

    @property
    def reserve(self):
        # Session tensor_chunks can hold raw, pending, split bytes and a CPU tensor
        # concurrently, in addition to the mmap and optional device tile. Charge
        # 128 MiB rather than the design's unverified 24 MiB estimate.
        return 128 * 2**20 if self.enabled else 0

    def agreement(self):
        budget = os.environ.get("TF_GLM_DISK_GIB", "64")
        return [int(self.enabled), self.block, ABI, *hashlib.sha256(budget.encode()).digest()]


class ReplyReservations:
    def __init__(self, decoder, config):
        self.d, self.config = decoder, config
        self.serial = 0
        self.parked = set()
        self.prepared = set()
        self.loaded = {}
        self.drain = False
        self.error = None
        self.retry_at = 0.0
        self.evicted_copy_bytes = 0
        owner = decoder.owner
        epoch = list(os.urandom(16)) if decoder.rank == 0 else None
        epoch = owner._share(epoch) if owner._share is not None else epoch
        error = None
        try:
            if decoder.sessions is None or decoder.sessions.store is None:
                raise ValueError("reply growth has no backed live-continuation store")
            if owner.drafter is not None and not getattr(owner.drafter, "ring", 0):
                raise ValueError("reply growth requires bounded private drafter rings")
            self.store = LiveStore(decoder.sessions.store, bytes(epoch).hex(), owner.rank, owner.world)
        except Exception as exc:
            error = exc
        if not self.vote(error is None):
            raise RuntimeError("one or more ranks lack live-continuation storage") from error

    def vote(self, ready):
        return all(row[0] for row in self.d.owner._gather_ints([int(ready)]))

    def initial(self, s):
        return len(s.prompt) + min(s.count, self.config.block)

    def command(self, op, *args):
        msg = [MESSAGE, self.serial + 1, op, *args]
        try:
            self.d._send(msg)
            return self.follow(msg)
        except Exception as exc:
            self.d.broken = exc
            raise

    def follow(self, msg):
        sizes = {CREDIT: 2, GROW: 5, PREPARE: 3, PARK: 1, LOAD: 5, RESUME: 1}
        valid = (len(msg) >= 3 and msg[0] == MESSAGE and msg[1] == self.serial + 1
                 and msg[2] in sizes and len(msg) == 3 + sizes[msg[2]])
        if not self.vote(valid):
            raise RuntimeError("reply reservation transaction epoch/layout mismatch")
        self.serial = msg[1]
        op, args = msg[2], msg[3:]
        if op == CREDIT:
            sid, limit = args
            valid = (sid == self.d.next_id and sid not in self.d.streams
                     and sid not in self.store.store.live_credits and limit > 0
                     and limit <= min(self.d.owner.limit, self.d.owner.pool_limit)
                     and self.d.pool._size(limit) <= self.d.pool.capacity)
            if not self.vote(valid):
                raise RuntimeError("reply credit names a duplicate owner or an impossible full cap")
            error = None
            try:
                size = self.store.credit_size(self.d, limit)
                ready = self.store.store.reserve_live(sid, size)
            except Exception as exc:
                error, ready = exc, False
            if not self.vote(ready):
                self.store.release(sid)
                self.failure(error or RuntimeError("a rank could not back the reply's disk credit"))
                return False
            return True
        sid = args[0]
        s = self.d.streams.get(sid)
        if not self.vote(s is not None):
            raise RuntimeError("reply transaction names a missing response")
        if op == GROW:
            _, start, old, length, tokens = args
            valid = (sid not in self.parked and self.d.pool.spans.get(sid) == (start, old)
                     and s.start == start and s.span == old and s.reserved_tokens <= tokens <= s.request_limit
                     and self.charged_except(sid) + tokens <= self.d.owner.pool_limit
                     and self.d.pool.growth(sid, tokens) == (start, length))
            if not self.vote(valid):
                raise RuntimeError("rank-local token ownership differs before grow")
            self.d.pool.grow(sid, (start, old), tokens)
            fresh = self.d._engine(s.slot, start, length)
            if fresh is not s.engine:
                for field in ("capacity", "kc", "vc", "index", "mtp_kc", "mtp_vc"):
                    if hasattr(fresh.st, field):
                        setattr(s.st, field, getattr(fresh.st, field))
            s.span, s.reserved_tokens = length, tokens
            s.reply_grows += 1
            return True
        if op == PREPARE:
            _, start, span = args
            valid = (sid not in self.parked and sid not in self.prepared
                     and self.d.fill_owner is None and self.d.express_owner is None
                     and not s.done and self.d.pool.spans.get(sid) == (start, span)
                     and (s.start, s.span) == (start, span)
                     and (sid not in self.d.filling or getattr(s, "fill_started", False)))
            if not self.vote(valid):
                raise RuntimeError("cannot park an incomplete or unowned live boundary")
            error = None
            try:
                # Staged session entries contain live target views. Persist/drop them
                # while ownership is retained; none may survive span reuse.
                self.d.sessions.flush(s)
                self.store.prepare(s, self.serial, filling=self.d.filling)
            except Exception as exc:
                error = exc
            if not self.vote(error is None):
                self.store.discard(sid)
                self.failure(error or RuntimeError("a rank failed durable park preparation"))
                return False
            self.prepared.add(sid)
            return True
        if op == PARK:
            valid = sid in self.prepared and sid in self.d.pool.spans and sid not in self.parked
            if not self.vote(valid):
                raise RuntimeError("duplicate or unprepared PARK_COMMIT")
            self.prepared.remove(sid)
            s.parked_prefill = sid in self.d.filling
            continuation = self.d.filling.pop(sid, None)
            from .cofill_big import Prefill
            s.parked_big_prefill = isinstance(continuation, Prefill)
            if continuation is not None:
                continuation.close()
            s.parked_images = s.engine.images
            s.reply_eager = s.engine is not self.d.owner.e
            s.cofill_ready, s.cofill_snap = False, None
            s.engine.images = None
            s.engine = s.st = s.drafter = None
            self.d.pool.release(sid)
            self.parked.add(sid)
            s.reply_parks += 1
            s.parked_at = time.perf_counter()
            return True
        if op == LOAD:
            _, slot, start, span, tokens = args
            valid = (sid in self.parked and sid not in self.loaded and tokens == s.request_limit
                     and sid not in self.d.pool.spans
                     and self.charged_except(sid) + tokens <= self.d.owner.pool_limit
                     and 0 <= slot < len(self.d.slots)
                     and all(t.slot != slot for t in self.d.streams.values() if t.sid not in self.parked)
                     and self.d.pool._find(tokens) == (start, span))
            if not self.vote(valid):
                raise RuntimeError("live restore has no matching exclusive destination")
            self.d.pool.allocate(sid, tokens)
            error = None
            try:
                eager = s.reply_eager or (slot, start) != (s.slot, s.start)
                e = self.d._engine(slot, start, span, eager)
                d = self.d._drafter(slot, start, span, eager)
                self.store.load(s, e, d)
            except Exception as exc:
                error = exc
            if not self.vote(error is None):
                self.d.pool.release(sid)
                self.failure(error or RuntimeError("a rank failed committed live restore"))
                return False
            self.loaded[sid] = (e, d, slot, start, span, tokens)
            return True
        if not self.vote(sid in self.loaded and sid in self.parked):
            raise RuntimeError("duplicate or unverified RESUME_COMMIT")
        e, d, s.slot, s.start, s.span, s.reserved_tokens = self.loaded.pop(sid)
        s.engine, s.st, s.drafter = e, e.st, d
        if d is not None:
            d.observe = self.d.observer.enabled or (self.d.rank == 0 and s.priced)
        e.images, s.parked_images = s.parked_images, None
        if s.parked_prefill:
            from .decode import prefill_steps
            if s.express:
                e.pbuf = self.d.express_buf
                e.prefill_rows = self.d.express_buf.rows
            kwargs = self.d.sessions.prefill_kwargs(s)
            if s.parked_big_prefill:
                from .cofill_big import Prefill
                pending = Prefill(None, kwargs)
                pending.started = True  # exact live state is loaded; never reset or re-absorb it
                self.d.filling[sid] = pending
            else:
                self.d.filling[sid] = prefill_steps(e, s.prompt, s.sampling, mtp=s.use_mtp,
                    drafter=d if s.use_dflash else None, live_start=s.fill_pos,
                    rows=lambda: s.fill_rows, slice_end=lambda: s.fill_layer_stop, **kwargs)
        s.reply_park_seconds += time.perf_counter() - s.parked_at
        self.parked.remove(sid)
        self.store.discard(sid)
        return True

    def failure(self, error):
        self.error = error
        self.retry_at = time.monotonic() + 1.0
        self.drain = True
        if self.d.rank == 0:
            print(f"[tensorfold] reply-growth storage blocked: {type(error).__name__}: {error}", flush=True)

    def needed(self, s):
        # Reserve this verification plus the *next* proposal before any forward.
        # ROUND2 can select the wider pending copy chain after this check. Model
        # proposals still cap at seven drafts; their eight-row allowance also
        # covers MTP absorption after a wide verify. DFlash's full internal block
        # is separately protected by the span guard.
        verify = 1 + max(len(s.drafts), len(getattr(s, "copy_drafts", ())))
        return min(s.request_limit, s.st.pos + verify + 8)

    def safe(self, s):
        return s.sid not in self.parked and self.needed(s) <= s.reserved_tokens

    def grow(self, s, tokens):
        if self.charged_except(s.sid) + tokens > self.d.owner.pool_limit:
            return False
        span = self.d.pool.growth(s.sid, tokens)
        if span is None:
            return False
        return self.command(GROW, s.sid, s.start, s.span, span[1], tokens)

    def charged_except(self, sid):
        return sum(s.reserved_tokens for s in self.d.streams.values()
                   if s.sid != sid and s.sid not in self.parked)

    def complete_chunk(self, s):
        from .batched import FILL, FILL_SLICE
        for lane in ("express_owner", "fill_owner"):
            sid = getattr(self.d, lane)
            if sid is not None:
                owner = self.d.streams[sid]
                self.d._send([FILL_SLICE, owner.sid, owner.fill_stop, len(self.d.w.layers)])
                self.d._fill(owner.sid, owner.fill_stop, len(self.d.w.layers))
        if s.sid in self.d.filling and not getattr(s, "fill_started", False):
            stop = min(len(s.prompt), s.fill_pos + s.fill_rows)
            if self.d.sessions is not None:
                stop = self.d.sessions.stop(s, s.fill_pos, stop)
            from . import cofill_big
            if isinstance(self.d.filling[s.sid], cofill_big.Prefill):
                self.d._send([cofill_big.FILL_BIG, 1, s.sid, stop])
                cofill_big.run(self.d, [(s.sid, stop)])
            else:
                self.d._send([FILL, s.sid, stop])
                self.d._fill(s.sid, stop)

    def park(self, s):
        self.complete_chunk(s)
        if s.done:
            return False
        if not self.command(PREPARE, s.sid, s.start, s.span):
            return False
        return self.command(PARK, s.sid)

    def resume(self, s):
        if self.charged_except(s.sid) + s.request_limit > self.d.owner.pool_limit:
            return False
        span = self.d.pool._find(s.request_limit)
        if span is None:
            return False
        slot = next(i for i in range(len(self.d.slots)) if all(
            t.slot != i for t in self.d.streams.values() if t.sid not in self.parked))
        if not self.command(LOAD, s.sid, slot, *span, s.request_limit):
            return False
        return self.command(RESUME, s.sid)

    def service(self):
        if time.monotonic() < self.retry_at:
            return
        active = [s for s in self.d.streams.values() if not s.done and s.sid not in self.parked]
        if not active:
            pending = [s for s in self.d.streams.values() if not s.done and s.sid in self.parked]
            if pending and not self.d.pool.spans:
                self.resume(min(pending, key=lambda s: s.sid))
            elif not pending and not self.d.streams:
                self.drain, self.error = False, None
            return
        wants = [s for s in active if s.sid not in self.d.filling and not self.safe(s)]
        if not wants and not self.drain:
            return
        if not self.drain:
            for s in wants:
                tokens = min(s.request_limit, s.reserved_tokens + self.config.block)
                if not self.grow(s, tokens):
                    self.drain = True
                    break
            if not self.drain:
                return
        # A failed growth stops admissions and grants the oldest request its
        # entire original promise. Complete/cancelled owners leave via DONE first.
        if any(s.done for s in self.d.streams.values()):
            return
        # Finished idle copies already attempted session persistence at DONE. Flush
        # still-live prompt anchors while their append-only rows remain owned.
        for s in active:
            self.d.sessions.flush(s)
        self.evicted_copy_bytes += self.d.held_bytes()
        self.d.cache.clear()  # independent prompt copies free RAM, never pool rows
        oldest = min(active, key=lambda s: s.sid)
        if oldest.reserved_tokens == oldest.request_limit:
            return
        while not self.grow(oldest, oldest.request_limit):
            active = [s for s in self.d.streams.values() if not s.done and s.sid not in self.parked]
            victim = max(active, key=lambda s: s.sid)
            if not self.park(victim):
                return
            if victim is oldest:
                self.resume(oldest)
                return

    def finish(self, sid):
        self.prepared.discard(sid)
        self.parked.discard(sid)
        self.loaded.pop(sid, None)
        self.store.release(sid)

    def metrics(self):
        return {"physical_tokens": sum(s.reserved_tokens for s in self.d.streams.values()
                                        if s.sid not in self.parked),
                "promised_tokens": sum(s.request_limit for s in self.d.streams.values()),
                "parked": len(self.parked), "disk_credit_bytes": self.store.store.credit_bytes(),
                "live_disk_bytes": sum(self.store.store.live_used.values()),
                "orphan_live_bytes": self.store.store.orphan_live_bytes,
                "evicted_copy_bytes": self.evicted_copy_bytes, "idle_copy_pool_rows_freed": 0,
                "draining": self.drain, "storage_blocked": self.error is not None}
