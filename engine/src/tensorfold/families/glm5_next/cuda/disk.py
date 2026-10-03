"""Kept prompts on disk: each prompt's attention rows written once, as the rows it adds to the longest prompt already
written that it extends, with its KDA state, conv windows and pending MTP rows.

A conversation's turns form a chain of small files, so a conversation that leaves the device's cache (another one
took the caches) or a server restart resumes from disk in seconds instead of prefilling again. The rows are the
bits a prefill wrote, so a prompt resumed from disk ends in the state a fresh prefill leaves. Files carry a
fingerprint of the engine's sources, the checkpoint config and the rank's settings; others are removed at start.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .decode import Snapshot, _row_views


def fingerprint(model_dir: Path, rank: int, world: int, extra: dict) -> str:
    """The engine's sources (this folder), the checkpoint config, the rank and settings that shape the rows."""

    h = hashlib.sha256()
    folder = Path(__file__).parent
    for path in sorted(p for p in folder.iterdir() if p.suffix in (".py", ".cu", ".cpp")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update((Path(model_dir) / "config.json").read_bytes())
    h.update(json.dumps({"rank": rank, "world": world, **extra}, sort_keys=True).encode())
    return h.hexdigest()[:20]


def _key(ids: np.ndarray) -> str:
    return hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest()[:32]


@dataclass
class Entry:
    key: str
    ids: np.ndarray                 # int32
    mtp_len: int
    parent: str | None
    path: Path
    nbytes: int
    used: float = 0.0
    children: set = field(default_factory=set)

    def stub(self) -> Snapshot:
        """A snapshot standing for this entry (no tensors until ``Store.load``)."""

        snap = Snapshot([int(t) for t in self.ids], None, None, None, self.mtp_len, -1)
        snap.disk = self
        return snap


CHUNK = 256 * 2 ** 20        # bytes through host memory at a time: a 449k-token prompt's 8.5 GB never sits there whole
_DT = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32", torch.float16: "f16", torch.int64: "i64"}
_TD = {v: k for k, v in _DT.items()}


def _write(path: Path, meta: dict, tensors: list[tuple[str, torch.Tensor]]) -> int:
    """A file of ``meta`` and raw tensors (rows copied to the host CHUNK bytes at a time); its size."""

    layout, offset = {}, 0
    for name, t in tensors:
        n = t.numel() * t.element_size()
        layout[name] = {"dtype": _DT[t.dtype], "shape": list(t.shape), "offset": offset}
        offset += n
    head = json.dumps({"meta": meta, "tensors": layout}).encode()
    head += b" " * (-len(head) % 64)
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for _, t in tensors:
            t = t.contiguous()
            rows = max(1, CHUNK // max(1, t[0].numel() * t.element_size())) if t.dim() else 1
            for a in range(0, max(t.shape[0], 1) if t.dim() else 1, rows):
                part = (t[a:a + rows] if t.dim() else t).cpu()
                f.write(part.view(torch.uint8).numpy().tobytes() if part.dtype != torch.uint8 else part.numpy().tobytes())
        f.flush()
        os.fsync(f.fileno())
        _forget_pages(f.fileno())
    os.replace(tmp, path)
    return 8 + len(head) + offset


def _forget_pages(fd: int) -> None:
    """Drop a file's pages from the page cache: on GB10 the GPU's free memory is MemFree, which counts them as used."""

    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


class _Reader:
    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            info = json.loads(f.read(n))
        self.meta, self.layout, self.base = info["meta"], info["tensors"], 8 + n
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def close(self) -> None:
        path = self.map.filename
        del self.map
        with open(path, "rb") as f:
            _forget_pages(f.fileno())

    def __contains__(self, name: str) -> bool:
        return name in self.layout

    def _raw(self, name: str):
        t = self.layout[name]
        dtype = _TD[t["dtype"]]
        size = int(np.prod(t["shape"])) * torch.tensor([], dtype=dtype).element_size()
        return t, dtype, self.map[self.base + t["offset"]:self.base + t["offset"] + size]

    def get(self, name: str, device=None) -> torch.Tensor:
        t, dtype, raw = self._raw(name)
        x = torch.from_numpy(np.array(raw)).view(dtype).reshape(t["shape"])
        return x.to(device) if device is not None else x

    def copy_into(self, name: str, dst: torch.Tensor) -> None:
        """dst[:rows] <- the tensor, CHUNK bytes at a time."""

        t, dtype, raw = self._raw(name)
        shape = t["shape"]
        if not shape or shape[0] == 0:
            return
        row = int(np.prod(shape[1:])) * torch.tensor([], dtype=dtype).element_size()
        step = max(1, CHUNK // max(1, row))
        for a in range(0, shape[0], step):
            b = min(shape[0], a + step)
            part = torch.from_numpy(np.array(raw[a * row:b * row])).view(dtype).reshape([b - a] + shape[1:])
            dst[a:b].copy_(part.to(dst.device))


class Store:
    """One rank's kept prompts in ``folder``, within ``budget`` bytes (least recently used chain ends go first)."""

    def __init__(self, folder: Path, budget: int, stamp: str) -> None:
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget, self.stamp = int(budget), stamp
        self.entries: dict[str, Entry] = {}
        self._scan()

    # -- index ----------------------------------------------------------------------------------------------------
    def _scan(self) -> None:
        for p in self.dir.glob("*.part"):
            p.unlink(missing_ok=True)
        for path in sorted(self.dir.glob("*.prompt")):
            try:
                r = _Reader(path)
                if r.meta.get("stamp") != self.stamp:
                    raise ValueError("another engine's prompt")
                ids = r.get("ids").numpy()
                meta = r.meta
                r.close()
            except Exception:  # noqa: BLE001 - unreadable or foreign: not ours to keep
                path.unlink(missing_ok=True)
                continue
            st = path.stat()
            self.entries[meta["key"]] = Entry(meta["key"], ids, int(meta["mtp_len"]), meta.get("parent") or None,
                                              path, st.st_size, st.st_mtime)
        changed = True                                  # a chain whose parent is gone cannot be loaded
        while changed:
            changed = False
            for e in list(self.entries.values()):
                if e.parent is not None and e.parent not in self.entries:
                    self._drop(e)
                    changed = True
        for e in self.entries.values():
            if e.parent is not None:
                self.entries[e.parent].children.add(e.key)

    def _drop(self, e: Entry) -> None:
        self.entries.pop(e.key, None)
        if e.parent in self.entries:
            self.entries[e.parent].children.discard(e.key)
        e.path.unlink(missing_ok=True)

    def held(self) -> int:
        return sum(e.nbytes for e in self.entries.values())

    def _longest(self, ids: np.ndarray, *, mtp: bool) -> Entry | None:
        best = None
        for e in self.entries.values():
            n = len(e.ids)
            if n < len(ids) and (not mtp or e.mtp_len >= 0) and (best is None or n > len(best.ids)) \
                    and np.array_equal(ids[:n], e.ids):
                best = e
        return best

    def resume(self, prompt, *, mtp: bool) -> Entry | None:
        """The longest written strict prefix of ``prompt`` (with its MTP rows when ``mtp``)."""

        return self._longest(np.asarray(prompt, dtype=np.int64), mtp=mtp)

    def find(self, ids) -> Entry | None:
        """Rank 1: the entry of exactly these ids."""

        return self.entries.get(_key(np.asarray(ids, dtype=np.int64)))

    def has(self, ids) -> bool:
        return _key(np.asarray(ids, dtype=np.int64)) in self.entries

    # -- write ----------------------------------------------------------------------------------------------------
    def put(self, e, snap: Snapshot) -> None:
        """Write ``snap`` (just taken: its rows are in the live caches) as the rows past its longest written prefix."""

        if snap.rec is None:
            return
        ids = np.asarray(snap.ids, dtype=np.int64)
        key = _key(ids)
        if key in self.entries:
            self.entries[key].used = time.time()
            return
        parent = self._longest(ids, mtp=False)
        n, m = len(ids), max(snap.mtp_len, 0)
        views = _row_views(e.st, n, m)
        starts = [0] * len(views)
        if parent is not None:
            before = _row_views(e.st, len(parent.ids), max(parent.mtp_len, 0))
            if len(before) == len(views):              # the last row again: a pool the prefix left partial
                starts = [max(0, v.shape[0] - 1) for v in before]
            else:                                      # MTP rows appeared or went: write them whole
                parent = None
        tensors = [(f"row{i}", v[s:]) for i, (v, s) in enumerate(zip(views, starts))]
        tensors += [("ids", torch.from_numpy(ids.astype(np.int32))), ("rec", snap.rec), ("conv", snap.conv)]
        if snap.pending is not None:
            tensors.append(("pending", snap.pending))
        meta = {"stamp": self.stamp, "key": key, "mtp_len": snap.mtp_len, "parent": parent.key if parent else "",
                "starts": starts}
        path = self.dir / f"{key}.prompt"
        try:
            nbytes = _write(path, meta, tensors)
        except OSError:
            path.with_suffix(".part").unlink(missing_ok=True)
            return
        entry = Entry(key, ids.astype(np.int32), snap.mtp_len, parent.key if parent else None, path, nbytes,
                      time.time())
        self.entries[key] = entry
        if parent is not None:
            parent.children.add(key)
            parent.used = entry.used
        self._evict(keep=key)

    def _evict(self, keep: str) -> None:
        """Remove least recently used chain ends (never ``keep`` or its ancestors) until within the budget."""

        protect, k = set(), keep
        while k is not None and k in self.entries:
            protect.add(k)
            k = self.entries[k].parent
        while self.held() > self.budget:
            leaves = [e for e in self.entries.values() if not e.children and e.key not in protect]
            if not leaves:
                return
            self._drop(min(leaves, key=lambda e: e.used))

    # -- read -----------------------------------------------------------------------------------------------------
    def load(self, e, entry: Entry) -> Snapshot:
        """Write the entry's rows (its chain, root first) into the live caches; its snapshot, ready to resume."""

        chain, k = [], entry.key
        while k is not None:
            chain.append(self.entries[k])
            k = self.entries[k].parent
        dev = e.st.kc[0].device
        snap = entry.stub()
        for link in reversed(chain):
            r = _Reader(link.path)
            starts = r.meta["starts"]
            views = _row_views(e.st, len(link.ids), max(link.mtp_len, 0))
            for i, (v, s) in enumerate(zip(views, starts)):
                r.copy_into(f"row{i}", v[s:])
            if link is entry:
                snap.rec, snap.conv = r.get("rec", dev), r.get("conv", dev)
                snap.pending = r.get("pending", dev) if "pending" in r else None
            torch.cuda.synchronize()
            r.close()
            link.used = time.time()
        torch.cuda.synchronize()
        return snap
