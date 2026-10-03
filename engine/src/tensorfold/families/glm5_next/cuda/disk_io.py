"""Aligned, bounded O_DIRECT tensor files. No buffered fallback on unified memory."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
from pathlib import Path

import torch

ALIGN = 4096
CHUNK = 8 * 2**20
MAGIC = b"TFSESS02"
HEADER_LIMIT = 2**20
_DT = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32", torch.float16: "f16",
       torch.int64: "i64", torch.uint8: "u8"}
_TD = {v: k for k, v in _DT.items()}


class TensorParts:
    """An axis-zero concatenation of owned rows, without allocating another prefix."""

    def __init__(self, parts):
        self.parts = tuple(parts)
        first = self.parts[0]
        if any(t.dtype != first.dtype or t.shape[1:] != first.shape[1:] for t in self.parts):
            raise ValueError("incompatible saved tensor segments")
        self.dtype, self.ndim = first.dtype, first.ndim
        self.shape = (sum(t.shape[0] for t in self.parts), *first.shape[1:])

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.parts[0].element_size()

    def slice(self, start, stop):
        parts, offset = [], 0
        for t in self.parts:
            a, b = max(0, start - offset), min(t.shape[0], stop - offset)
            if a < b:
                parts.append(t[a:b])
            offset += t.shape[0]
        return TensorParts(parts or [self.parts[0][:0]])


def aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


class DirectFile:
    """An anonymous aligned transfer buffer; never mmap the backing file."""

    def __init__(self, path: Path, *, write: bool = False):
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
        self.fd = os.open(path, flags | os.O_DIRECT | os.O_CLOEXEC, 0o600)
        try:
            self.buffer = mmap.mmap(-1, CHUNK)
        except BaseException:
            os.close(self.fd)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.buffer.close()
        os.close(self.fd)

    def read(self, offset: int, length: int, *, eof: bool = False) -> bytes:
        if offset % ALIGN or length > CHUNK or length < 0:
            raise ValueError("unaligned or oversized direct read")
        if length == 0:
            return b""
        with memoryview(self.buffer)[:aligned(length)] as view:
            got = os.preadv(self.fd, [view], offset)
            if got < length and not eof:
                raise ValueError("truncated session file")
            return bytes(view[:min(got, length)])

    def write(self, offset: int, data: bytes) -> None:
        if offset % ALIGN or len(data) > CHUNK:
            raise ValueError("unaligned or oversized direct write")
        if not data:
            return
        n = aligned(len(data))
        with memoryview(self.buffer)[:n] as view:
            view[:len(data)] = data
            view[len(data):] = bytes(n - len(data))
            if os.pwritev(self.fd, [view], offset) != n:
                raise OSError("short direct write")


def file_digest(path: Path) -> str:
    """Hash actual checkpoint bytes, not names/mtimes or an asserted revision label."""
    h = hashlib.sha256()
    with DirectFile(path) as f:
        before = os.fstat(f.fd)
        size = before.st_size
        for at in range(0, size, CHUNK):
            h.update(f.read(at, min(CHUNK, size - at)))
        after = os.fstat(f.fd)
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (size, before.st_mtime_ns, before.st_ctime_ns):
            raise ValueError("checkpoint changed during identity scan")
    return h.hexdigest()


def tensor_chunks(t: torch.Tensor, *, before_copy=None):
    """Bounded host copies even for noncontiguous rank views; split on axis zero."""
    if isinstance(t, TensorParts):
        pending = b""
        for part in t.parts:
            for raw in tensor_chunks(part, before_copy=before_copy):
                pending += raw
                while len(pending) >= CHUNK:
                    yield pending[:CHUNK]
                    pending = pending[CHUNK:]
        if pending:
            yield pending
        return
    if t.numel() == 0:
        return
    if t.ndim == 0:
        t = t.reshape(1)
    row = math.prod(t.shape[1:]) * t.element_size()
    step = max(1, CHUNK // max(1, row))
    if row > CHUNK:
        raise ValueError("a session tensor row exceeds the transfer buffer")
    # CHUNK-sized chunks below also keep every file offset aligned, even when
    # the tensor's row width does not divide the direct-IO block size.
    pending = b""
    for start in range(0, t.shape[0], step):
        if before_copy is not None:
            before_copy()
        raw = t[start:start + step].contiguous().cpu().view(torch.uint8).numpy().tobytes()
        pending += raw
        while len(pending) >= CHUNK:
            yield pending[:CHUNK]
            pending = pending[CHUNK:]
    if pending:
        yield pending


def _layout(tensors):
    layout, offset = {}, 0
    for name, t in tensors:
        if name in layout:
            raise ValueError("duplicate session tensor")
        size = t.numel() * t.element_size()
        layout[name] = {"dtype": _DT[t.dtype], "shape": list(t.shape), "offset": offset, "size": size,
                        "sha256": "0" * 64}
        offset += aligned(size)
    return layout, offset


def file_size(meta: dict, tensors: list) -> int:
    layout, size = _layout(tensors)
    return aligned(48 + len(json.dumps({"meta": meta, "tensors": layout}, sort_keys=True).encode())) + size


def write_tensors(path: Path, meta: dict, tensors: list, *, guard=None, before_copy=None) -> int:
    layout, size = _layout(tensors)
    header_size = aligned(48 + len(json.dumps({"meta": meta, "tensors": layout}, sort_keys=True).encode()))
    if header_size > HEADER_LIMIT:
        raise ValueError("session header too large")
    tmp = path.with_suffix(".part")
    try:
        with DirectFile(tmp, write=True) as f:
            for name, t in tensors:
                h, at = hashlib.sha256(), header_size + layout[name]["offset"]
                for raw in tensor_chunks(t, before_copy=before_copy):
                    if guard is not None:
                        guard()
                    h.update(raw)
                    f.write(at, raw)
                    at += len(raw)
                layout[name]["sha256"] = h.hexdigest()
            head = json.dumps({"meta": meta, "tensors": layout}, sort_keys=True).encode()
            f.write(0, MAGIC + len(head).to_bytes(8, "little") + hashlib.sha256(head).digest() + head)
            os.ftruncate(f.fd, header_size + size)
            os.fsync(f.fd)
        os.replace(tmp, path)
        # Durability includes the rename, not just the data pages.
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return header_size + size
    finally:
        tmp.unlink(missing_ok=True)


class TensorReader:
    def __init__(self, path: Path):
        self.f = DirectFile(path)
        try:
            first = self.f.read(0, ALIGN)
            n = int.from_bytes(first[8:16], "little")
            if first[:8] != MAGIC or not 0 < n <= HEADER_LIMIT - 48:
                raise ValueError("unknown session file format")
            head = self.f.read(0, aligned(n + 48))[48:n + 48]
            if hashlib.sha256(head).digest() != first[16:48]:
                raise ValueError("session header checksum mismatch")
            info = json.loads(head)
            self.meta, self.layout, self.base = info["meta"], info["tensors"], aligned(n + 48)
            size, offset = os.fstat(self.f.fd).st_size, 0
            for t in sorted(self.layout.values(), key=lambda item: (item["offset"], item["size"])):
                if (t["dtype"] not in _TD or not isinstance(t["shape"], list) or
                        any(type(d) is not int or d < 0 for d in t["shape"])):
                    raise ValueError("invalid session tensor layout")
                expected = math.prod(t["shape"]) * torch.empty((), dtype=_TD[t["dtype"]]).element_size()
                if t["offset"] != offset or t["size"] != expected:
                    raise ValueError("invalid session tensor extent")
                offset += aligned(expected)
            if self.base + offset != size:
                raise ValueError("session file size mismatch")
        except BaseException:
            self.f.__exit__()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.__exit__(*exc)

    def get(self, name: str, device=None) -> torch.Tensor:
        t = self.layout[name]
        out = torch.empty(t["shape"], dtype=_TD[t["dtype"]], device=device or "cpu")
        self.copy_into(name, out)
        return out

    def copy_into(self, name: str, dst: torch.Tensor, *, guard=None) -> None:
        t = self.layout[name]
        if list(dst.shape) != t["shape"] or dst.dtype != _TD[t["dtype"]]:
            raise ValueError("session tensor does not fit the live state")
        if not dst.is_contiguous():
            raise ValueError("session restore requires contiguous target rows")
        out, h = dst.reshape(-1).view(torch.uint8), hashlib.sha256()
        for at in range(0, t["size"], CHUNK):
            if guard is not None:
                guard()
            raw = self.f.read(self.base + t["offset"] + at, min(CHUNK, t["size"] - at))
            h.update(raw)
            part = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            out[at:at + len(raw)].copy_(part)
        if h.hexdigest() != t["sha256"]:
            raise ValueError("session tensor checksum mismatch")
