"""Default-off hooks for the pooled decoder; no CUDA math or scheduler policy here."""

from __future__ import annotations

import hashlib
import os
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .decode import snapshot_bytes, take_snapshot
from .session_state import image_count, state_hash


@dataclass
class SessionConfig:
    disk: bool = False
    checkpoints: bool = False
    cancel: bool = False
    hash_gate: bool = False
    stage_bytes: int = 1024 * 2**20
    math_version: str = ""

    @classmethod
    def from_env(cls, parallel):
        def flag(name):
            value = os.environ.get("TF_GLM_SESSION_" + name, "0")
            if value not in ("0", "1"):
                raise ValueError("TF_GLM_SESSION_" + name + " must be 0 or 1")
            return value == "1"
        cfg = cls(flag("DISK"), flag("CHECKPOINTS"), flag("CANCEL"), flag("HASH_GATE"))
        if cfg.checkpoints or cfg.cancel:
            if not cfg.disk:
                raise ValueError("session checkpoints/cancel preservation require TF_GLM_SESSION_DISK=1")
        if cfg.disk or cfg.hash_gate:
            if parallel < 2:
                raise ValueError("the session tier and hash gate require pooled GLM serving")
        if cfg.disk:
            cfg.stage_bytes = int(os.environ.get("TF_GLM_SESSION_STAGE_MIB", "1024")) * 2**20
            cfg.math_version = os.environ.get("TF_GLM_DISK_MATH_VERSION", "")
            if not cfg.math_version.startswith(("E:", "S:")) or not cfg.math_version[2:].strip():
                raise ValueError("TF_GLM_DISK_MATH_VERSION must be E:<version> or S:<version>")
            if not os.environ.get("TF_GLM_DISK_DIR") or cfg.stage_bytes <= 0:
                raise ValueError("session persistence requires TF_GLM_DISK_DIR and a positive staging budget")
        return cfg

    @property
    def reserve(self):
        # Includes a temporary fixed/ring snapshot and bounded host/device IO
        # buffers. The staged queue itself is separately capped, across all slots.
        return (self.stage_bytes if self.disk else 0) + (256 * 2**20 if self.disk or self.hash_gate else 0)

    def agreement(self):
        digest = hashlib.sha256(self.math_version.encode()).digest()
        return [int(self.disk), int(self.checkpoints), int(self.cancel), int(self.hash_gate),
                self.stage_bytes >> 20, *digest]


def configure(owner, cfg, model_dir, drafter_path, options, *, slots):
    owner.session_cache = None
    if not cfg.disk and not cfg.hash_gate:
        return
    store, role_ids, assistant_id = None, [], None
    error = None
    try:
        if cfg.disk:
            st, d = owner.e.st, owner.drafter
            fixed = sum(t.numel() * t.element_size() for t in (st.rec[0], st.conv)) + owner.w.cfg.hidden * 2
            if d is not None:
                if not getattr(d, "ring", 0):
                    raise ValueError("pooled session persistence requires a bounded DFlash2 ring")
                fixed += sum(t.shape[0] * min(d.ring, d.window + 63) * t.shape[2] * t.element_size()
                             for t in (*d.kc, *d.vc))
            if cfg.stage_bytes < slots * fixed:
                raise ValueError("session staging budget must retain one completed checkpoint per active slot")
            from .session_disk import SessionStore, exact_fingerprint
            stamp = exact_fingerprint(model_dir, owner.rank, owner.world, math_version=cfg.math_version,
                                      options=options, drafter=drafter_path)
            folder = Path(os.environ["TF_GLM_DISK_DIR"]) / "sessions-v2" / f"rank{owner.rank}"
            store = SessionStore(folder, int(float(os.environ.get("TF_GLM_DISK_GIB", "64")) * 2**30), stamp)
        if cfg.disk and owner.rank == 0:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
            assistant_id = tok.token_to_id("<|assistant|>")
            role_ids = [n for role in ("system", "user", "assistant", "observation")
                        if (n := tok.token_to_id(f"<|{role}|>")) is not None]
    except Exception as exc:
        error = exc
    # Even a single rank's unavailable filesystem must not strand its peers.
    good = owner._gather_ints([int(error is None)])
    if not all(row[0] for row in good):
        raise RuntimeError("one or more ranks could not initialize session persistence") from error
    if cfg.disk:
        role_ids = owner._share(role_ids if owner.rank == 0 else None)
        assistant_id = owner._share([assistant_id if assistant_id is not None else -1]
                                    if owner.rank == 0 else None)[0]
    owner.session_cache = SessionCache(owner, cfg, store, role_ids, assistant_id=assistant_id)


class SessionCache:
    def __init__(self, owner, config, store=None, role_ids=(), *, assistant_id=None):
        self.owner, self.config, self.store = owner, config, store
        self.role_ids = frozenset(role_ids)
        self.assistant_id = assistant_id if assistant_id is not None and assistant_id >= 0 else None
        self.staged = []                 # (stream id, snapshot, durable anchor)
        self.dropped = 0
        self.history = []    # at most 32 prior prompts as int32; no raw text in receipts
        self.sources = {}    # borrowed memory snapshots for writes after live-span release

    def held_bytes(self):
        pending = self.store.pending if self.store is not None else None
        return sum(snapshot_bytes(snap) - (snap.nbytes if snap.rows is not None else 0)
                   for _, snap, _ in self.staged) + (pending.fixed_bytes if pending else 0)

    def extra_rows(self, cached):
        """Writer-held rows evicted from memory still count against the cache byte cap."""
        from .session_writer import storages

        if self.store is not None:
            self.store.poll()
        def rows(snaps):
            return [part for snap in snaps for row in snap.rows or ()
                    for part in (row if isinstance(row, tuple) else (row,))]
        pinned = storages(rows(self.sources.values()))
        if self.store is not None and self.store.pending is not None:
            pinned.update(self.store.pending.row_storages)
        current = storages(rows(cached))
        return sum(size for key, size in pinned.items() if key not in current)

    def saved(self, prompt, code, *, cached=None, digests=()):
        if self.store is None:
            return None
        entry = self.store.resume(prompt, mtp=code[0] in (1, 2, 3), dflash=code[0] >= 10,
                                  image_digests=digests, cached=cached)
        return entry.stub() if entry is not None else None

    def restore(self, stream, snap):
        from .decode import load_rows
        if snap is not None:
            if getattr(snap, "disk", None) is not None:
                snap = self.store.load(stream.engine, snap.disk, stream.drafter)
            else:
                load_rows(stream.engine, snap, stream.drafter)
        present = self.owner._gather_ints([int(snap is not None)])
        return snap if all(row[0] for row in present) else None

    def begin(self, s, *, source="cold", failed_load=False):
        import numpy as np

        s.session_boundaries = (tuple(i for i, token in enumerate(s.prompt) if i > 0 and token in self.role_ids)
                                if self.config.checkpoints and self.role_ids else ())
        # Rolling image clients rewrite the oldest tool result into archive text.
        # Pi also moves its images into a following user message, so a checkpoint
        # at the image itself is too late. Keep one anchor before the preceding
        # assistant/tool turn, independently of the optional dense checkpoints.
        first_image = (next((i for i, token in enumerate(s.prompt) if token < 0), None)
                       if s.image_digests else None)
        s.session_image_boundary = None
        if first_image is not None and self.role_ids:
            before = range(first_image - 1, 0, -1)
            s.session_image_boundary = next((i for i in before if s.prompt[i] == self.assistant_id), None)
            if s.session_image_boundary is None:
                s.session_image_boundary = next((i for i in before if s.prompt[i] in self.role_ids), None)
        s.session_cache_source = source
        s.session_miss_reason = "none"
        s.session_reason_evidence = "hit" if s.cached else "unknown"
        s.session_cache_detail = "rank-miss-or-corruption" if failed_load else "none"
        if source == "memory" and s.cached and not failed_load:
            return  # a proven memory hit needs no disk scan or historical-prefix diagnostics
        ids, digests = np.asarray(s.prompt, dtype=np.int32), s.image_digests
        event = getattr(s, "session_event", None)
        if event in ("boot", "compaction", "fork"):
            s.session_miss_reason, s.session_reason_evidence = event, "client-reported"
            return
        invalidated = self.store.invalidated if self.store is not None else []
        for prior, images in invalidated:
            if len(prior) < len(ids) and np.array_equal(prior, ids[:len(prior)]) and digests[:len(images)] == images:
                s.session_miss_reason, s.session_reason_evidence = "boot", "invalidated-disk-identity"
                return
        disk_history = ([(e.ids, e.image_digests) for e in self.store.entries.values()][-32:]
                        if self.store is not None else [])
        for prior, images in reversed(disk_history + self.history):
            n = min(len(prior), len(ids))
            if len(prior) < len(ids) and np.array_equal(prior, ids[:n]) and images == digests[:len(images)]:
                if source != "memory":
                    s.session_miss_reason, s.session_reason_evidence = "store-evicted", "observed-prior-prompt"
                return
            # Require the whole old prefix to differ only at image placeholders,
            # rather than guessing from a shared system prompt and a new image.
            if (len(prior) < len(ids) and images != digests[:len(images)] and
                    np.array_equal(np.where(prior < 0, -1, prior), np.where(ids[:n] < 0, -1, ids[:n]))):
                s.session_miss_reason, s.session_reason_evidence = "image-changed", "prefix-and-image-digests"
                return
            if s.cached and n > s.cached and np.array_equal(prior[:s.cached], ids[:s.cached]):
                s.session_miss_reason, s.session_reason_evidence = "fork", "shared-prefix-divergence"
                return

    def _record(self, s):
        import json
        import numpy as np

        completed = min(getattr(s, "fill_pos", len(s.prompt)), len(s.prompt))
        if completed:
            ids = np.asarray(s.prompt[:completed], dtype=np.int32)
            self.history = (self.history + [(ids, s.image_digests[:image_count(ids)])])[-32:]
        if self.owner.rank == 0:
            keys = ("session_cache_source", "session_miss_reason", "session_reason_evidence", "session_cache_detail")
            record = {key: getattr(s, key) for key in keys}
            record.update(sid=s.sid, prompt_tokens=len(s.prompt), cached=s.cached,
                          disk_bytes=self.store.held() if self.store else 0,
                          staged_bytes=self.held_bytes(), dropped_anchors=self.dropped)
            try:
                print("[tensorfold] session-cache " + json.dumps(record, sort_keys=True), flush=True)
            except OSError:
                pass  # A diagnostic sink must not poison rank control.

    def next_mark(self, s, pos):
        if not s.draft or self.store is None:
            return None
        points = []
        image_boundary = getattr(s, "session_image_boundary", None)
        if image_boundary is not None and image_boundary > pos:
            points.append(image_boundary)
        if self.config.checkpoints:
            points.append(self.owner.checkpoint_after(pos))
            points += [i for i in s.session_boundaries if i > pos][:1]
        if self.config.cancel:
            points.append(pos + getattr(s, "fill_rows", s.engine.prefill_rows))
        return min(points) if points else None

    def prefill_kwargs(self, s):
        if not s.draft or self.store is None or not (self.config.checkpoints or self.config.cancel or
                                                    getattr(s, "session_image_boundary", None) is not None):
            return {}
        return {"mark": lambda pos: self.next_mark(s, pos), "keep": lambda snap: self.stage(s, snap)}

    def stop(self, s, start, end):
        s.fill_rows = end - start
        mark = self.next_mark(s, start)
        return min(end, mark) if mark is not None else end

    def stage(self, s, snap, *, final=False):
        if self.store is None:
            return
        snap.image_digests = tuple(s.image_digests[:image_count(snap.ids)])
        at = len(snap.ids)
        durable = final or at == getattr(s, "session_image_boundary", None) or (self.config.checkpoints and
                            (at in s.session_boundaries or self.owner.checkpoint_after(at - 1) == at))
        # Checkpoint retention needs only the last finished chunk in addition to durable anchors.
        self.staged = [(sid, old, keep) for sid, old, keep in self.staged
                       if sid != s.sid or keep and len(old.ids) != at]
        need = snapshot_bytes(snap) - (snap.nbytes if snap.rows is not None else 0)
        # A long prefill may discard older fork anchors, but cannot evict the
        # only completed checkpoint (or prompt end) of another active stream.
        latest = {sid: id(old) for sid, old, _ in self.staged if sid != s.sid}
        while self.staged and self.held_bytes() + need > self.config.stage_bytes:
            victim = next((i for i, (sid, old, _) in enumerate(self.staged)
                           if sid == s.sid or latest[sid] != id(old)), None)
            if victim is None:
                break
            self.staged.pop(victim)
            self.dropped += 1
        if self.held_bytes() + need <= self.config.stage_bytes:
            self.staged.append((s.sid, snap, durable))
        else:
            self.dropped += 1

    def completed(self, s, snapshot=None):
        if snapshot is None and getattr(s, "session_cache_source", None) == "memory" and not self.config.hash_gate:
            # Under cache pressure, keep a warm reply cheap instead of cloning a
            # new fixed state solely to feed the synchronous no-cache fallback.
            self.dropped += int(self.store is not None and s.draft)
            return
        snap = (copy.copy(snapshot) if snapshot is not None else
                take_snapshot(s.engine, s.prompt, s.engine.last_hidden if s.use_mtp else None,
                              mtp=s.use_mtp, drafter=s.drafter if s.use_dflash else None))
        snap.image_digests = s.image_digests
        if snap.rows is not None and self.store is not None:
            self.sources[s.sid] = SimpleNamespace(ids=snap.ids, rows=snap.rows, mtp_len=snap.mtp_len)
        if self.config.hash_gate:
            started = time.perf_counter()
            digest = bytes.fromhex(state_hash(s.engine, snap))
            all_hashes = self.owner._gather_ints(list(digest))
            s.session_state_sha256 = [bytes(row).hex() for row in all_hashes]
            s.session_hash_s = time.perf_counter() - started
        if s.draft:
            self.stage(s, snap, final=True)

    def flush(self, s, *, broken=False):
        staged = [snap for sid, snap, _ in self.staged if sid == s.sid]
        self.staged = [(sid, snap, keep) for sid, snap, keep in self.staged if sid != s.sid]
        source = self.sources.pop(s.sid, None)
        partial = getattr(s, "fill_pos", len(s.prompt)) < len(s.prompt)
        if not broken and self.store is not None and (not partial or self.config.cancel):
            if source is not None:
                try:
                    if not self.store.submit(s.engine, sorted(staged, key=lambda snap: len(snap.ids)), source,
                                             stage_bytes=self.config.stage_bytes - self.held_bytes()):
                        self.dropped += len(staged)
                except (OSError, ValueError, RuntimeError):
                    self.store.errors += 1
                return
            if not partial and getattr(s, "session_cache_source", None) == "memory":
                self.dropped += len(staged)
                return  # never force a warm memory reply through synchronous persistence
            for snap in sorted(staged, key=lambda snap: len(snap.ids)):
                try:
                    self.store.put(s.engine, snap)
                except OSError:
                    self.store.errors += 1  # filesystem failure is a future coordinated cache miss

    def finish(self, s, *, broken=False):
        self.flush(s, broken=broken)
        if hasattr(s, "session_cache_source"):
            self._record(s)
