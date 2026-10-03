"""One bounded session write batch; only its owner thread changes the disk index."""

import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .disk_io import TensorParts, write_tensors


def storages(tensors):
    """Count pinned allocations once, including segmented views shared with newer snapshots."""
    result = {}
    for t in tensors:
        for part in t.parts if isinstance(t, TensorParts) else (t,):
            storage = part.untyped_storage()
            if storage.nbytes():
                result[(part.device, storage.data_ptr())] = storage.nbytes()
    return result


class WriteGate:
    """Persistence uses idle bandwidth; serving only changes state and never waits.

    The half-second quiet interval lets a back-to-back turn finish tokenizing
    before an optional multi-GiB save starts. Arrival stops new IO tiles; at most
    the tile already in flight can overlap serving. Offline drains bypass this.
    """

    def __init__(self, quiet_s=0.5):
        self.condition = threading.Condition()
        self.quiet_s = quiet_s
        self.idle = True
        self.preparing = 0
        self.since = time.monotonic()

    @contextmanager
    def prepare(self):
        """Rank-zero HTTP preparation also needs CPU and unified-memory bandwidth."""
        with self.condition:
            self.preparing += 1
            self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.preparing -= 1
                self.since = time.monotonic()
                self.condition.notify_all()

    def set_idle(self, idle):
        with self.condition:
            if self.idle != idle:
                self.idle, self.since = idle, time.monotonic()
                self.condition.notify_all()

    def wait(self, draining):
        with self.condition:
            while not draining.is_set():
                remaining = self.since + self.quiet_s - time.monotonic()
                available = self.idle and not self.preparing
                if available and remaining <= 0:
                    return
                self.condition.wait(remaining if available else None)


class SessionWrite:
    def __init__(self, plans, protected, *, min_free, gate):
        self.plans, self.protected, self.min_free = plans, protected, min_free
        self.gate, self.draining = gate, threading.Event()
        self.nbytes = sum(plan["entry"].nbytes for plan in plans)
        self.row_storages = storages(t for plan in plans for t in plan["rows"])
        self.fixed_bytes = sum(t.numel() * t.element_size() for plan in plans for t in plan["fixed"])
        self.written, self.errors, self.orphan_bytes = [], 0, 0
        self.done = threading.Event()
        self.ready = None
        tensors = [t for plan in plans for _, t in plan["tensors"]]
        devices = [part.device for t in tensors for part in (t.parts if isinstance(t, TensorParts) else (t,))
                   if part.is_cuda]
        self.device = devices[0] if devices else None
        if self.device is not None:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(self.device))
        self.thread = threading.Thread(target=self._run, name="glm-session-writer", daemon=True)
        self.thread.start()

    def drain(self):
        with self.gate.condition:
            self.draining.set()
            self.gate.condition.notify_all()

    def _idle(self):
        self.gate.wait(self.draining)

    def _guard(self, path):
        self._idle()
        if self.min_free:
            free = shutil.disk_usage(path.parent).free
            available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            if free < self.min_free or available < 9 * 2**30:
                raise OSError("session writer reached its disk or memory floor")

    def _write(self):
        for plan in self.plans:
            path = plan["entry"].path
            try:
                self._idle()
                write_tensors(path, plan["meta"], plan["tensors"], guard=lambda: self._guard(path),
                              before_copy=self._idle)
            except Exception:
                # A failure after rename (e.g. directory fsync) may leave a file.
                # Remove it or keep its reservation charged after this batch ends.
                for leftover in (path, path.with_suffix(".part")):
                    try:
                        leftover.unlink(missing_ok=True)
                    except OSError:
                        self.orphan_bytes += plan["entry"].nbytes
                raise
            self.written.append(plan["entry"])

    def _run(self):
        try:
            self._idle()
            if self.device is None:
                self._write()
            else:
                # Never make the serving stream wait for disk copies or fsync.
                with torch.cuda.device(self.device), torch.cuda.stream(torch.cuda.Stream(device=self.device)):
                    torch.cuda.current_stream().wait_event(self.ready)
                    self._write()
        except Exception:
            self.errors += 1  # optional persistence failure is a later cache miss
        finally:
            self.done.set()
