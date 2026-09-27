"""Experimental TP3 triangle all-reduce, boot gated and fail closed.

Only the sealed CudaCommunicator/PyNccl single-call path is in scope. A size
must pass six order probes, adversarial bit checks, and a captured/replayed
NCCL comparison on all ranks before admission. Unknown capture sizes use the
original method. Target FULL graphs are pre-captured as OFF/ON banks; runtime epochs select a bank.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import torch
import torch.distributed as dist
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources, PTXASError

from .jspark3_triar_arithmetic import PROBES, infer_labels
from .jspark3_triar_kernel import reduce, prepare
from . import jspark3_triar_dual as dual

MAX_BYTES = 1 << 20
MAX_SIZES = 64
ENV_KEYS = ("JSPARK3_TRIAR", "NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
            "NCCL_P2P_LL_THRESHOLD", "NCCL_ALLOC_P2P_NET_LL_BUFFERS",
            "NCCL_P2P_NET_CHUNKSIZE", "NCCL_GRAPH_MIXING_SUPPORT",
            "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS",
            "NCCL_LL_BUFFSIZE", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", "TRITON_INTERPRET")


def environment():
    return {key: os.environ.get(key) for key in ENV_KEYS}


def bits(tensor):
    return tensor.detach().view(torch.int16).flatten().cpu().numpy().tobytes()


class GroupTracker:
    """Observe Python PyNccl grouping; outer groups must use the original AR."""
    def __init__(self, library):
        self.library = library
        self.local = threading.local()

    @property
    def depth(self):
        return getattr(self.local, "depth", 0)

    def ncclGroupStart(self):
        result = self.library.ncclGroupStart()
        self.local.depth = self.depth + 1
        return result

    def ncclGroupEnd(self):
        result = self.library.ncclGroupEnd()
        self.local.depth = self.depth - 1
        return result

    def __getattr__(self, name):
        return getattr(self.library, name)


class Triangle:
    def __init__(self, pynccl, *, fault=None):
        self.nccl = pynccl
        if not isinstance(pynccl.nccl, GroupTracker):
            pynccl.nccl = GroupTracker(pynccl.nccl)
        self.rank = pynccl.rank
        self.env = environment()
        self.fault = fault  # hardware/CPU harness only; no served environment knob
        self.entries = {}
        self.events = []
        self.fast_calls = 0
        self.fallback_calls = 0
        self.identity = self.comm_identity()

    def comm_identity(self):
        p = self.nccl
        return (id(p.comm), getattr(p.comm, "value", p.comm), p.nccl_version, p.device)

    def agree(self, value):
        values = [None] * 3
        dist.all_gather_object(values, value, group=self.nccl.group)
        return values

    def exchange(self, tensor, labels):
        # Per-call buffers avoid aliasing outputs, graph pools, or overlapping
        # calls. Both peers' traffic is in ONE NCCL group, with matching counts.
        partials = [None] * 3
        partials[self.rank] = tensor
        output = torch.empty_like(tensor)
        for peer in range(3):
            if peer != self.rank:
                partials[peer] = torch.empty_like(tensor)
        stream = torch.cuda.current_stream(tensor.device)
        self.nccl.nccl.ncclGroupStart()
        try:
            for peer in range(3):
                if peer != self.rank:
                    self.nccl.send(tensor, peer, stream=stream)
                    self.nccl.recv(partials[peer], peer, stream=stream)
        finally:
            self.nccl.nccl.ncclGroupEnd()
        reduce(partials, labels, output)
        if self.fault == "ulp":
            output.view(torch.int16).flatten()[0].bitwise_xor_(1)
        return output

    def calibrate(self, tensor):
        n = tensor.numel()
        probe = torch.empty_like(tensor)
        outputs = []
        for triple in PROBES:
            probe.view(torch.int16).fill_(triple[self.rank] if triple[self.rank] < 32768
                                         else triple[self.rank] - 65536)
            outputs.append(self.nccl.all_reduce(probe).view(torch.int16).flatten().cpu().tolist())
        try:
            label_bytes = infer_labels(outputs)
            signature = hashlib.sha256(label_bytes).hexdigest()
        except ValueError:
            label_bytes, signature = bytes(n), None
        signatures = self.agree(signature)
        if None in signatures or len(set(signatures)) != 1:
            return self.receipt(n, False, "order-probes-or-rank-disagreement")
        labels = torch.tensor(list(label_bytes), device=tensor.device, dtype=torch.uint8)
        if self.fault == "rotate":
            labels = (labels + 1) % 3

        # A compiler refusal is recoverable before any candidate transport or
        # capture. Vote first so peers cannot enter p2p after a local failure.
        compiled = True
        try:
            reduce([probe] * 3, labels, torch.empty_like(probe))
        except (CompilationError, OutOfResources, PTXASError):
            compiled = False
        if not all(self.agree(compiled)):
            return self.receipt(n, False, "kernel-compilation-refused", signature)

        # Warm the exact p2p peers and compile before capture; no first-use p2p
        # connection setup or Triton compilation is allowed inside capture.
        self.exchange(probe, labels)
        torch.cuda.current_stream(tensor.device).synchronize()
        capture_stream = torch.cuda.Stream(device=tensor.device)
        capture_stream.wait_stream(torch.cuda.current_stream(tensor.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_reference = self.nccl.all_reduce(probe)
            graph_candidate = self.exchange(probe, labels)

        # Both eager and replayed output must match all bits, including signed
        # zero, subnormals, infinities, NaN encodings, and overflow/cancellation.
        ok = True
        special = [0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x7F7F,
                   0xFF7F, 0x7F80, 0xFF80, 0x7F81, 0x7FC0, 0x7FFF, 0xFFFF]
        cases = []
        for triple in PROBES:
            cases.append(torch.full((n,), triple[self.rank], dtype=torch.int32))
        for seed in range(4):
            generator = torch.Generator().manual_seed(0x713A + 3 * seed + self.rank)
            cases.append(torch.randint(0, 65536, (n,), generator=generator, dtype=torch.int32))
        index = torch.arange(n)
        table = torch.tensor(special, dtype=torch.int32)
        for rotation in range(3):
            cases.append(table[(index // (len(special) ** self.rank) + rotation) % len(special)])
        cases.append(tensor.view(torch.int16).flatten().cpu().to(torch.int32) & 65535)
        for case in cases:
            probe.view(torch.int16).flatten().copy_(case.to(torch.int16))
            reference = self.nccl.all_reduce(probe)
            candidate = self.exchange(probe, labels)
            ok = bits(reference) == bits(candidate) and ok
            graph.replay()
            torch.cuda.synchronize(tensor.device)
            ok = bits(reference) == bits(graph_reference) == bits(graph_candidate) and ok
        ok = all(self.agree(ok))
        self.receipt(n, ok, "eager-and-graph-bit-check", signature, len(cases))
        return labels if ok else None

    def receipt(self, n, ok, reason, signature=None, cases=0):
        event = {"triar": "qualified" if ok else "rejected", "rank": self.rank,
                 "bytes": n * 2, "reason": reason, "label_sha256": signature,
                 "cases": cases, "nccl_version": self.nccl.nccl_version}
        self.events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
        return None

    def all_reduce(self, tensor):
        # None means call the original CudaCommunicator method. Never swallow
        # a transport/CUDA error and try another collective on a poisoned comm.
        if self.identity != self.comm_identity():
            raise RuntimeError("TRIAR communicator changed: restart and recapture")
        if self.env != environment():
            raise RuntimeError("TRIAR environment changed: restart all ranks and recapture")
        n = tensor.numel()
        if (self.nccl.nccl.depth or self.nccl.disabled or tensor.device != self.nccl.device or
                tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or
                tensor.data_ptr() % 16 or n < 8 or n % 8 or n * 2 > MAX_BYTES):
            self.fallback_calls += 1
            return None
        if n not in self.entries:
            if torch.cuda.is_current_stream_capturing() or len(self.entries) >= MAX_SIZES:
                self.fallback_calls += 1
                return None
            self.entries[n] = self.calibrate(tensor)
        labels = self.entries[n]
        if labels is None:
            self.fallback_calls += 1
            return None
        self.fast_calls += 1
        return self.exchange(tensor, labels)


def attach(coordinator):
    if coordinator.unique_name != "tp:0" or coordinator.world_size != 3:
        return None
    # Every rank participates even when its local PyNccl backend is disabled.
    # Otherwise a local rejection could leave its peers in the admission vote.
    group = coordinator.cpu_group
    if not isinstance(group, dist.ProcessGroup):
        return None
    from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
    p = coordinator.pynccl_comm
    other_backends = ("qr_comm", "fi_ar_comm", "aiter_ar_comm", "ca_comm", "symm_mem_comm")
    allowed = (p is not None and not p.disabled and p.world_size == 3
               and isinstance(p.group, dist.ProcessGroup) and not is_symmetric_memory_enabled()
               and p.nccl_version == 23007 and torch.cuda.get_device_capability(p.device) == (12, 1)
               and all(getattr(coordinator, name, None) is None or
                       getattr(getattr(coordinator, name), "disabled", False)
                       for name in other_backends)
               and os.environ.get("NCCL_NVLS_ENABLE") == "0"
               and os.environ.get("NCCL_TUNER_PLUGIN") in (None, "none")
               and not any(os.environ.get(k) for k in
                           ("NCCL_ALGO", "NCCL_PROTO", "TRITON_INTERPRET")))
    states = [None] * 3
    dist.all_gather_object(states, (allowed, environment()), group=group)
    if not all(row == states[0] and row[0] for row in states):
        print(json.dumps({"triar": "rejected", "reason": "unsupported-runtime-or-rank-config"}), flush=True)
        return None
    triangle = Triangle(p)
    # All ranks vote during construction, before any candidate transport.
    prepared, error = None, None
    try:
        prepared = prepare(p.device)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    if not all(triangle.agree(prepared is not None)):
        raise RuntimeError("TRIAR startup preparation refused: " + str(error))
    print(json.dumps({"triar": "runtime-n-prepared", "rank": p.rank,
        "kernel_sha256": hashlib.sha256(Path(__file__).with_name(
            "jspark3_triar_kernel.py").read_bytes()).hexdigest(), **prepared}, sort_keys=True), flush=True)
    return triangle


def install(cls):
    if os.environ.get("JSPARK3_TRIAR", "0") == "0":
        return
    if os.environ.get("JSPARK3_TRIAR") != "1":
        raise RuntimeError("JSPARK3_TRIAR must be 0 or 1")
    if getattr(cls, "_jspark3_triar_installed", False):
        return
    original_init, original_reduce = cls.__init__, cls.all_reduce

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._jspark3_triar = attach(self)
        dual.register(self._jspark3_triar)

    def all_reduce(self, input_):
        triangle = self._jspark3_triar
        if triangle is not None and dual.capturing_on():
            out = triangle.all_reduce(input_)
            if out is not None:
                return out
        return original_reduce(self, input_)

    cls.__init__, cls.all_reduce = initialize, all_reduce
    cls._jspark3_triar_installed = True
    dual.install()
