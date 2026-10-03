"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal.

TF_COMM_ONEHOP=1 gathers with one send to and one receive from every peer at once (NCCL point-to-point in one group)
instead of the ring: one hop on a full mesh such as three cabled Sparks, the same bytes at the same places (each
peer's at its rank's offset, never in arrival order)."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}
ONEHOP = os.environ.get("TF_COMM_ONEHOP", "0") == "1"


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int) -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        for name in ("ncclSend", "ncclRecv"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_void_p]
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        if ONEHOP:
            return self.all_gather_onehop(send, recv)
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def all_gather_onehop(self, send: torch.Tensor, recv: torch.Tensor, peers: list[int] | None = None) -> None:
        """``all_gather`` as one group of point-to-point calls: send to and receive from every peer at once, each
        peer's bytes to its rank's offset; ``peers`` only reorders the calls (tests: the bytes may not change)."""

        n, dt = send.numel(), _DTYPES[send.dtype]
        stream = torch.cuda.current_stream().cuda_stream
        flat = recv.view(-1)
        flat[self.rank * n:(self.rank + 1) * n].copy_(send.reshape(-1))
        self._check(self.lib.ncclGroupStart())
        for peer in (peers if peers is not None else [r for r in range(self.world) if r != self.rank]):
            self._check(self.lib.ncclSend(send.data_ptr(), n, dt, peer, self.comm, stream))
            self._check(self.lib.ncclRecv(flat[peer * n:].data_ptr(), n, dt, peer, self.comm, stream))
        self._check(self.lib.ncclGroupEnd())

    def exchange(self, sends: list[tuple[int, torch.Tensor]], recvs: list[tuple[int, torch.Tensor]]) -> None:
        """One group of point-to-point calls on the current stream: each (peer, tensor) of ``sends`` to that peer and
        of ``recvs`` from it, in list order (both ranks list a pair's tensors in the same order and sizes). Bytes only,
        each to the place its receiver names, never in arrival order."""

        if any(not t.is_contiguous() for _, t in sends + recvs):
            raise ValueError("exchange: tensors must be contiguous")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclGroupStart())
        for call, ops in ((self.lib.ncclSend, sends), (self.lib.ncclRecv, recvs)):
            for peer, t in ops:
                if t.numel():
                    self._check(call(t.data_ptr(), t.numel(), _DTYPES[t.dtype], peer, self.comm, stream))
        self._check(self.lib.ncclGroupEnd())

    def all_reduce(self, x: torch.Tensor) -> None:
        """x <- the sum of every rank's x, by NCCL (its ring's grouping; for the TF_TP3_REDUCE=nccl probe only)."""

        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllReduce(x.data_ptr(), x.data_ptr(), x.numel(), _DTYPES[x.dtype], 0, self.comm,
                                           stream))

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
