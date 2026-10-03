"""Pooled GLM session chains with strict identity and bounded direct IO.

The longest-written-prefix chain design is adapted from upstream TensorFold
0.6.0's Apache-2.0 GLM disk.py. This format also owns bounded DFlash ring state,
full image digests, checksums and a hard budget. Legacy .prompt files are separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .decode import Snapshot
from .disk_io import TensorReader, file_digest, file_size, write_tensors
from .session_state import fixed_state, image_count, saved_target_rows, snapshot_key, target_rows

FORMAT = 2


def exact_fingerprint(model_dir: Path, rank: int, world: int, *, math_version: str, options: dict,
                      drafter: Path | None = None) -> str:
    """Hash loaded checkpoint content + explicit arithmetic ABI, never implementation source.

    E:<version> persists across exact builds; S:<version> uses a new namespace on
    every boot. The user must bump the arithmetic ABI when math changes.
    The content scan deliberately has no mtime-only cache or unverified shortcut.
    """
    from .split import rank_files

    if not math_version.startswith(("E:", "S:")) or not math_version[2:].strip():
        raise ValueError("TF_GLM_DISK_MATH_VERSION must be E:<version> or S:<version>")
    model_dir = Path(model_dir)
    mine = rank_files(model_dir, rank, world)
    files = set(mine or model_dir.glob("*.safetensors"))
    # Include the rank-zero vision tower and any separate grafts conservatively.
    files.update(p for p in model_dir.glob("*.safetensors") if ".rank" not in p.name)
    files.add(model_dir / "config.json")
    files.update(p for p in model_dir.glob("*.json") if p.name != "tokenizer.json")
    if len(files) < 2:
        raise ValueError("session identity requires checkpoint weight files")
    identity = {"target/" + p.name: file_digest(p) for p in sorted(files)}
    if drafter is not None:
        for name in ("config.json", "model.safetensors"):
            identity["drafter/" + name] = file_digest(Path(drafter) / name)
    data = {"format": FORMAT, "weights": identity, "rank": rank, "world": world,
            "math_version": math_version, "options": options}
    if math_version.startswith("S:"):
        data["cold_boot"] = os.urandom(32).hex()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class SessionEntry:
    key: str
    ids: np.ndarray
    mtp_len: int
    drafter_end: int
    image_digests: tuple
    parent: str | None
    path: Path
    nbytes: int
    used: float
    children: set = field(default_factory=set)

    def stub(self):
        snap = Snapshot(self.ids.tolist(), None, None, None, self.mtp_len, self.drafter_end)
        snap.image_digests, snap.disk = self.image_digests, self
        return snap


class SessionStore:
    """One worker owns a rank-local folder. Missing/bad files are cache misses."""

    def __init__(self, folder: Path, budget: int, stamp: str, *, min_free: int = 150 * 2**30):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.budget, self.stamp, self.min_free = int(budget), stamp, int(min_free)
        if self.budget <= 0 or self.min_free < 0:
            raise ValueError("invalid session disk budget")
        self.entries: dict[str, SessionEntry] = {}
        self.live_credits: dict[int, int] = {}
        self.live_used: dict[int, int] = {}
        self.orphan_live_bytes = sum(p.stat().st_size for p in self.dir.glob("live-continuation-v1/**/*")
                                     if p.is_file())
        self.errors = 0
        self.pending = None
        from .session_writer import WriteGate
        self.write_gate = WriteGate()
        self.invalidated = []  # bounded boot-miss evidence, never emitted as token content
        self._scan()

    def held(self):
        return sum(e.nbytes for e in self.entries.values()) + (self.pending.nbytes if self.pending else 0)

    def credit_bytes(self):
        return sum(self.live_credits.values())

    def unspent_credit(self):
        return self.credit_bytes() - sum(self.live_used.values())

    def reserve_live(self, sid, size):
        self.poll()  # finished writer parents are ordinary evictable cache entries again
        if sid in self.live_credits or size <= 0:
            raise ValueError("invalid or duplicate live-state credit")
        if not self._evict(set(), needed=size):
            return False
        pending_bytes = self.pending.nbytes if self.pending else 0
        protected = self.pending.protected if self.pending else set()
        while shutil.disk_usage(self.dir).free - self.unspent_credit() - pending_bytes - size < self.min_free:
            leaves = [e for e in self.entries.values() if not e.children and e.key not in protected]
            if not leaves:
                return False
            self._drop_tree(min(leaves, key=lambda e: e.used).key)
        self.live_credits[sid] = size
        return True

    def _scan(self):
        for path in self.dir.glob("*.part"):
            path.unlink(missing_ok=True)
        for path in self.dir.glob("*.session"):
            try:
                with TensorReader(path) as r:
                    meta = r.meta
                    ids = r.get("ids").numpy()
                    digests = tuple(bytes.fromhex(d) for d in meta["images"])
                    if meta["format"] != FORMAT or meta["stamp"] != self.stamp:
                        self.invalidated = (self.invalidated + [(ids, digests)])[-32:]
                        raise ValueError("session math/weights identity mismatch")
                    if (ids.dtype != np.int32 or ids.ndim != 1 or not len(ids) or
                            len(digests) != image_count(ids) or any(len(d) != 32 for d in digests) or
                            snapshot_key(ids, digests) != meta["key"] or path.stem != meta["key"]):
                        raise ValueError("invalid session identity")
                st = path.stat()
                self.entries[meta["key"]] = SessionEntry(meta["key"], ids, int(meta["mtp_len"]),
                    int(meta["drafter_end"]), digests, meta["parent"], path, st.st_size, st.st_mtime)
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                path.unlink(missing_ok=True)
                self.errors += 1
        # Strictly shorter parents prohibit cycles, and must share ids and images.
        for entry in list(self.entries.values()):
            try:
                self._chain(entry)
            except ValueError:
                self._drop_tree(entry.key)
        for entry in self.entries.values():
            if entry.parent:
                self.entries[entry.parent].children.add(entry.key)
        self._evict(set())

    @staticmethod
    def _extends(ids, digests, parent):
        return (len(parent.ids) < len(ids) and np.array_equal(ids[:len(parent.ids)], parent.ids)
                and digests[:len(parent.image_digests)] == parent.image_digests)

    def _chain(self, entry):
        chain = [entry]
        while chain[-1].parent is not None:
            child = chain[-1]
            parent = self.entries.get(child.parent)
            if parent is None or not self._extends(child.ids, child.image_digests, parent):
                raise ValueError("broken session prefix chain")
            chain.append(parent)
        return list(reversed(chain))

    def _drop_tree(self, key):
        keys = {key}
        while True:
            more = {e.key for e in self.entries.values() if e.parent in keys} - keys
            if not more:
                break
            keys |= more
        for k in keys:
            entry = self.entries.pop(k, None)
            if entry is not None:
                entry.path.unlink(missing_ok=True)
        for e in self.entries.values():
            e.children -= keys

    def _evict(self, protect, needed=0):
        protect = set(protect) | (self.pending.protected if self.pending else set())
        while self.held() + self.credit_bytes() + self.orphan_live_bytes + needed > self.budget:
            leaves = [e for e in self.entries.values() if not e.children and e.key not in protect]
            if not leaves:
                return False
            self._drop_tree(min(leaves, key=lambda e: e.used).key)
        return True

    def resume(self, prompt, *, mtp: bool, dflash: bool, image_digests=(), cached=None):
        self.poll()
        ids = np.asarray(prompt, dtype=np.int32)
        fits = [e for e in self.entries.values() if self._extends(ids, image_digests, e)
                and (not mtp or e.mtp_len >= 0) and (not dflash or e.drafter_end == len(e.ids))
                and (cached is None or len(e.ids) == cached)]
        return max(fits, key=lambda e: len(e.ids), default=None)

    def _plan(self, e, snap, *, source=None, entries=None):
        entries = self.entries if entries is None else entries
        ids = np.asarray(snap.ids, dtype=np.int32)
        digests = snap.image_digests
        if (not len(ids) or len(digests) != image_count(ids) or any(len(d) != 32 for d in digests)
                or snap.rec is None or (snap.drafter_end >= 0 and snap.drafter_rows is None)):
            return None
        key = snapshot_key(ids, digests)
        existing = entries.get(key)
        if existing is not None:
            if existing.mtp_len == snap.mtp_len and existing.drafter_end == snap.drafter_end:
                existing.used = time.time()
            return None  # never replace a published mode from a background writer
        parents = [p for p in entries.values() if self._extends(ids, digests, p)
                   and (snap.mtp_len < 0 or p.mtp_len >= 0)
                   and (snap.drafter_end < 0 or p.drafter_end == len(p.ids))]
        parent = max(parents, key=lambda p: len(p.ids), default=None)
        views = dict(saved_target_rows(e, source, snap) if source is not None else
                     target_rows(e.st, len(ids), max(snap.mtp_len, 0)))
        before = dict(target_rows(e.st, len(parent.ids), max(parent.mtp_len, 0))) if parent else {}
        if parent and set(before) != set(views):
            parent, before = None, {}
        starts = {name: before[name].shape[0] if parent else 0 for name in views}
        tensors = [(name, t.slice(starts[name], t.shape[0]) if source is not None else t[starts[name]:])
                   for name, t in views.items()]
        rows = [t for _, t in tensors]
        fixed = [("ids", torch.from_numpy(ids))] + fixed_state(snap)
        tensors += fixed
        meta = {"format": FORMAT, "stamp": self.stamp, "key": key, "images": [d.hex() for d in digests],
                "mtp_len": snap.mtp_len, "drafter_end": snap.drafter_end,
                "parent": parent.key if parent else None, "starts": starts,
                "draft_count": len(snap.drafter_rows or [])}
        size = file_size(meta, tensors)
        entry = SessionEntry(key, ids.copy(), snap.mtp_len, snap.drafter_end, digests,
                             parent.key if parent else None, self.dir / f"{key}.session", size, time.time())
        return dict(entry=entry, meta=meta, tensors=tensors, rows=rows, fixed=[t for _, t in fixed])

    def _publish(self, entry):
        if entry.parent is not None and entry.parent not in self.entries:
            try:
                entry.path.unlink(missing_ok=True)
            except OSError:
                self.orphan_live_bytes += entry.nbytes
            self.errors += 1
            return
        self.entries[entry.key] = entry
        if entry.parent:
            self.entries[entry.parent].children.add(entry.key)

    def poll(self):
        """Publish completed files only on the serving worker; never wait for IO."""
        pending = self.pending
        if pending is not None and pending.done.is_set():
            self.pending = None
            self.errors += pending.errors
            self.orphan_live_bytes += pending.orphan_bytes
            for entry in pending.written:
                self._publish(entry)

    def wait_pending(self, timeout=30):
        """Explicit drain for offline gates/shutdown, never a serving request hook."""
        if self.pending is not None:
            self.pending.drain()
            self.pending.thread.join(timeout)
            if not self.pending.done.is_set():
                raise TimeoutError("session writer still running")
            self.poll()

    def submit(self, e, snapshots, source, *, stage_bytes):
        """One optional write batch per rank, borrowing already-owned cache rows."""
        from .session_writer import SessionWrite

        self.poll()
        if self.pending is not None:
            return False  # bounded backpressure: keep the memory hit, skip optional persistence
        virtual, plans = dict(self.entries), []
        for snap in snapshots:
            plan = self._plan(e, snap, source=source, entries=virtual)
            if plan is not None:
                plans.append(plan)
                virtual[plan["entry"].key] = plan["entry"]
        if not plans:
            return True
        if sum(t.numel() * t.element_size() for plan in plans for t in plan["fixed"]) > stage_bytes:
            return False
        protected = set()
        for plan in plans:
            entry = plan["entry"]
            while entry is not None:
                protected.add(entry.key)
                entry = virtual.get(entry.parent)
        size = sum(plan["entry"].nbytes for plan in plans)
        if (size + sum(self.entries[k].nbytes for k in protected if k in self.entries) > self.budget or
                shutil.disk_usage(self.dir).free - self.unspent_credit() - size < self.min_free or
                not self._evict(protected, needed=size)):
            return False
        self.pending = SessionWrite(plans, protected, min_free=self.min_free, gate=self.write_gate)
        return True

    def put(self, e, snap) -> bool:
        """Synchronous fallback when no owned memory rows exist (e.g. cache disabled)."""
        self.poll()
        if self.pending is not None:
            return False
        key = snapshot_key(snap.ids, snap.image_digests)
        existing = self.entries.get(key)
        if existing is not None:
            if existing.mtp_len == snap.mtp_len and existing.drafter_end == snap.drafter_end:
                existing.used = time.time()
                return True
            self._drop_tree(key)
        plan = self._plan(e, snap)
        if plan is None:
            return False
        entry, size = plan["entry"], plan["entry"].nbytes
        parent = self.entries.get(entry.parent)
        protected = {link.key for link in self._chain(parent)} if parent else set()
        if size + sum(self.entries[k].nbytes for k in protected) > self.budget:
            return False
        if shutil.disk_usage(self.dir).free - self.unspent_credit() - size < self.min_free:
            return False
        if not self._evict(protected, needed=size):
            return False
        try:
            write_tensors(entry.path, plan["meta"], plan["tensors"])
        except (OSError, ValueError):
            self.errors += 1
            return False
        self._publish(entry)
        self._evict(protected | {key})
        return True

    def load(self, e, entry, drafter=None):
        """Checksum every restored tensor; failure is handled by all-rank fallback."""
        try:
            chain, snap = self._chain(entry), entry.stub()
            dev = e.st.kc[0].device
            for link in chain:
                with TensorReader(link.path) as r:
                    if (r.meta["stamp"] != self.stamp or r.meta["key"] != link.key or
                            r.meta["format"] != FORMAT or r.meta["parent"] != link.parent or
                            r.meta["mtp_len"] != link.mtp_len or r.meta["drafter_end"] != link.drafter_end or
                            r.meta["images"] != [d.hex() for d in link.image_digests]):
                        raise ValueError("session identity changed")
                    views = dict(target_rows(e.st, len(link.ids), max(link.mtp_len, 0)))
                    starts = r.meta["starts"]
                    parent = self.entries.get(link.parent)
                    before = dict(target_rows(e.st, len(parent.ids), max(parent.mtp_len, 0))) if parent else {}
                    expected = {name: before[name].shape[0] if parent else 0 for name in views}
                    if starts != expected:
                        raise ValueError("session delta does not fit its parent")
                    for name, dst in views.items():
                        r.copy_into(name, dst[starts[name]:])
                    if link is entry:
                        expected = {"rec": e.st.rec[0], "conv": e.st.conv}
                        from .disk_io import _DT
                        for name, tensor in expected.items():
                            if (r.layout[name]["shape"] != list(tensor.shape) or
                                    r.layout[name]["dtype"] != _DT[tensor.dtype]):
                                raise ValueError("session recurrent state shape/dtype mismatch")
                        if snap.mtp_len >= 0:
                            pending = r.layout["pending"]
                            if (len(pending["shape"]) != 2 or pending["shape"][0] != 1 or
                                    pending["dtype"] != "bf16" or not 0 <= snap.mtp_len < len(snap.ids)):
                                raise ValueError("invalid pending MTP state")
                            hidden = getattr(getattr(getattr(e, "w", None), "cfg", None), "hidden", None)
                            if hidden is not None and pending["shape"][1] != hidden:
                                raise ValueError("pending MTP width mismatch")
                        count = 0
                        if snap.drafter_end >= 0:
                            from .decode import _ring_slots
                            if (drafter is None or not getattr(drafter, "ring", 0) or
                                    snap.drafter_end != len(snap.ids)):
                                raise ValueError("session requires the bounded drafter ring")
                            n = len(_ring_slots(drafter, snap.drafter_end))
                            caches = (*drafter.kc, *drafter.vc)
                            count = len(caches)
                            for i, t in enumerate(caches):
                                saved = r.layout[f"draft{i}"]
                                if (saved["shape"] != [t.shape[0], n, t.shape[2]] or
                                        saved["dtype"] != _DT[t.dtype]):
                                    raise ValueError("session drafter state shape/dtype mismatch")
                        if r.meta["draft_count"] != count:
                            raise ValueError("session drafter segment count mismatch")
                        snap.rec, snap.conv = r.get("rec", dev), r.get("conv", dev)
                        snap.pending = r.get("pending", dev) if snap.mtp_len >= 0 else None
                        snap.drafter_rows = [r.get(f"draft{i}", dev) for i in range(count)] or None
                link.used = time.time()
            return snap
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError):
            self.errors += 1
            self._drop_tree(entry.key)
            return None
