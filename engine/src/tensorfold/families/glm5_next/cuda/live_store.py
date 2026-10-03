"""Pinned, epoch-local live continuations. Never a reusable prompt cache entry."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from .disk_io import HEADER_LIMIT, TensorReader, aligned, file_size, write_tensors

ABI = 1


def row_views(st, rows, pools):
    result = [(f"kc{i}", t[:rows]) for i, t in enumerate(st.kc)]
    result += [(f"vc{i}", t[:rows]) for i, t in enumerate(st.vc) if t is not None]
    for name in ("mtp_kc", "mtp_vc"):
        t = getattr(st, name, None)
        if t is not None:
            result.append((name, t[:rows]))
    for i, (ik, ig, pk) in enumerate(st.index or ()):
        result += [(f"ik{i}", ik[:rows]), (f"ig{i}", ig[:rows]), (f"pk{i}", pk[:pools])]
    return result


def tensor_views(e, d, rows, pools):
    """Full owned rows, both recurrence banks, and every private ring cell.

    Split rings by head and flatten contiguous views without allocating. This
    keeps the session store's direct-IO tiles bounded even when a recurrent-state row is >8 MiB.
    No whole-prefix clone or contiguous copy is permitted here.
    """
    tensors = row_views(e.st, rows, pools) + [("rec", e.st.rec), ("conv", e.st.conv)]
    if e.last_hidden is not None:
        tensors.append(("hidden", e.last_hidden))
    if d is not None:
        for i, t in enumerate((*d.kc, *d.vc)):
            # Slot zero exposes the parent's other rings too. Only its first
            # ring belongs to this request; nonzero views expose one ring.
            for h in range(t.shape[0]):
                tensors.append((f"draft{i}h{h}", t[h, :d.ring]))
    result = []
    for name, t in tensors:
        if not t.is_contiguous():
            raise ValueError("live checkpoint needs a contiguous per-head tensor view")
        result.append((name, t.view(-1)))
    return result


def request_identity(s):
    from .batched import pack_sampling

    data = {"prompt": s.prompt, "count": s.count, "sampling": pack_sampling(s.sampling),
            "policy": s.policy_code, "images": [d.hex() for d in s.image_digests],
            "eos": list(s.eos), "draft": s.draft}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class LiveStore:
    def __init__(self, store, epoch, rank, world):
        self.store, self.epoch, self.rank, self.world = store, epoch, rank, world
        self.dir = store.dir / "live-continuation-v1" / epoch
        self.dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.records = {}

    def guard(self):
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        # Stop a copy before reaching the fleet's 8 GiB floor, retaining the
        # source span (park) or durable record (restore).
        if available < 9 * 2**30:
            raise OSError("live-state IO requires at least 9 GiB MemAvailable")
        if shutil.disk_usage(self.dir).free < self.store.min_free:
            raise OSError("live-state IO reached the disk free-space floor")
        if self.store.held() + self.store.credit_bytes() + self.store.orphan_live_bytes > self.store.budget:
            raise OSError("retained live-state files exhausted the aggregate disk budget")

    def unlink(self, path, size):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # The response already has another owned copy, but deletion can
            # fail on a read-only filesystem. Account this unreachable epoch
            # artifact rather than making its bytes disappear from the cap.
            self.store.orphan_live_bytes += size
            self.store.errors += 1

    def credit_size(self, decoder, limit):
        # Two complete files cover old/new/temporary overlap. HEADER_LIMIT also
        # covers all JSON identities; no compression or sparse-file assumption.
        span = decoder.pool._size(limit)
        e, d = decoder.owner.e, decoder.owner.drafter
        tensors = tensor_views(e, d, span - 8, span // 4)
        size = HEADER_LIMIT + sum(aligned(t.numel() * t.element_size()) for _, t in tensors)
        # A pending hidden row may not exist during boot or first admission.
        size += aligned(decoder.w.cfg.hidden * 2)
        return 2 * size

    def meta(self, s, filling):
        from .disk_io import _DT

        st, d = s.st, s.drafter if s.use_dflash else None
        return {"abi": ABI, "epoch": self.epoch, "rank": self.rank, "world": self.world,
                "stamp": self.store.stamp, "sid": s.sid, "identity": request_identity(s),
                "rows": s.span - 8, "pools": s.span // 4, "pos": st.pos,
                "mtp_len": st.mtp_len, "mtp_drafted": st.mtp_drafted, "cur": list(st.cur),
                "hidden": list(s.engine.last_hidden.shape) if s.engine.last_hidden is not None else None,
                "hidden_dtype": _DT[s.engine.last_hidden.dtype] if s.engine.last_hidden is not None else None,
                "draft_end": d.context_end if d is not None else None,
                "draft_cap": d.cap if d is not None else None,
                "fill": s.sid in filling, "out": len(s.out), "drafts": list(s.drafts)}

    def prepare(self, s, transaction, *, filling):
        self.guard()
        if s.sid in self.records:
            raise RuntimeError("a live continuation already exists")
        meta = self.meta(s, filling)
        meta["transaction"] = transaction
        path = self.dir / f"{s.sid}-{transaction}.live"
        tensors = tensor_views(s.engine, s.drafter if s.use_dflash else None, meta["rows"], meta["pools"])
        size = file_size(meta, tensors)
        if 2 * size > self.store.live_credits.get(s.sid, 0):
            raise RuntimeError("live continuation exceeds its backed disk credit")
        try:
            write_tensors(path, meta, tensors, guard=self.guard)
            # Verify the durable bytes without copying them over the still-live
            # source. Reads remain bounded and hash every named payload.
            with TensorReader(path) as reader:
                if reader.meta != meta:
                    raise ValueError("live continuation metadata changed during write")
                from .disk_io import CHUNK
                for info in reader.layout.values():
                    h = hashlib.sha256()
                    for at in range(0, info["size"], CHUNK):
                        self.guard()
                        h.update(reader.f.read(reader.base + info["offset"] + at,
                                               min(CHUNK, info["size"] - at)))
                    if h.hexdigest() != info["sha256"]:
                        raise ValueError("live continuation write verification failed")
            self.records[s.sid] = (path, meta)
            self.store.live_used[s.sid] = size
        except BaseException:
            self.unlink(path, size)
            self.unlink(path.with_suffix(".part"), size)
            raise

    def load(self, s, e, d):
        self.guard()
        path, meta = self.records[s.sid]
        if (meta["identity"] != request_identity(s) or meta["out"] != len(s.out)
                or meta["drafts"] != s.drafts):
            raise RuntimeError("parked response identity or output position changed")
        if meta["hidden"] is not None:
            from .disk_io import _TD
            e.last_hidden = torch.empty(meta["hidden"], dtype=_TD[meta["hidden_dtype"]], device=e.st.rec.device)
        else:
            e.last_hidden = None
        if meta["draft_end"] is not None and (d is None or d.cap != meta["draft_cap"]):
            raise RuntimeError("live drafter head stride changed")
        tensors = tensor_views(e, d if s.use_dflash else None, meta["rows"], meta["pools"])
        with TensorReader(path) as reader:
            if reader.meta != meta or set(reader.layout) != {name for name, _ in tensors}:
                raise ValueError("committed live continuation identity/layout changed")
            for name, dst in tensors:
                reader.copy_into(name, dst, guard=self.guard)
        st = e.st
        st.cur = list(meta["cur"])
        st.set_pos(meta["pos"])
        st.set_mtp_len(meta["mtp_len"])
        st.mtp_drafted = meta["mtp_drafted"]
        if s.use_dflash:
            d.context_end = meta["draft_end"]
            d.pos_dev.fill_(d.context_end)
        return meta

    def discard(self, sid):
        record = self.records.pop(sid, None)
        if record is not None:
            self.unlink(record[0], self.store.live_used.get(sid, self.store.live_credits.get(sid, 0)))
        self.store.live_used.pop(sid, None)

    def release(self, sid):
        self.discard(sid)
        self.store.live_credits.pop(sid, None)
