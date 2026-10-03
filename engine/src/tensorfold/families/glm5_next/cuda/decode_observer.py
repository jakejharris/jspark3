"""Opt-in, bounded decode receipts; CUDA waits and file I/O stay off the serving thread."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class DecodeObserver:
    def __init__(self, rank: int) -> None:
        root = os.environ.get("TF_GLM_DECODE_OBSERVE_DIR")
        self.enabled = bool(root)
        self.rank, self.round = rank, 0
        self.sample_every = self.block_every = 0
        self.dropped = self.tap_dropped = 0
        self._marks, self._blocks = [], []
        self._worker = None
        if not root:
            return
        self.sample_every = int(os.environ.get("TF_GLM_DECODE_TAP_EVERY", "64"))
        self.block_every = int(os.environ.get("TF_GLM_DECODE_BLOCK_EVERY", "64"))
        if min(self.sample_every, self.block_every) < 0:
            raise ValueError("TF_GLM_DECODE_TAP_EVERY and TF_GLM_DECODE_BLOCK_EVERY must be nonnegative")
        self._cuda = torch.cuda.is_available()
        self.root = Path(root)
        self.file = self.root / f"rank{rank}.jsonl"
        self.phase_file = self.root / f"rank{rank}-phases.jsonl"
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Do not append a new run to old receipts (or overwrite its tap files).
            self.file.touch(mode=0o600, exist_ok=False)
            self.phase_file.touch(mode=0o600, exist_ok=False)
        except OSError as exc:
            self.error, self.enabled = str(exc), False
            return
        self._jobs = queue.Queue()
        # Include the in-flight writer job in the bound. Never wait for disk/CUDA.
        self._slots = threading.BoundedSemaphore(128)
        self._tap_slots = queue.Queue()
        # Retain storage even if inference raises or logging is disabled between
        # capture() and record(); an enqueued D2H must never outlive its target.
        self._capture_buffers = [[], []]
        for buffers in self._capture_buffers:
            self._tap_slots.put(buffers)
        self._worker = threading.Thread(target=self._write_loop, name=f"glm-observer-{rank}", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def _event(self):
        if not self._cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextmanager
    def phase(self, name: str, *, stream_id: int | None = None):
        if not self.enabled:
            yield
            return
        start = self._event()
        begin = time.perf_counter()
        try:
            yield
        finally:
            end = self._event()
            self._submit("phase", {"rank": self.rank, "phase": name, "sid": stream_id,
                                   "host_start_s": begin, "host_end_s": time.perf_counter(),
                                   "observer_dropped": self.dropped}, [start, end], [], None)

    def begin_round(self) -> None:
        self._marks, self._blocks = [], []

    def clock(self) -> float:
        if not self.enabled:
            return 0.0
        self._marks.append(self._event())
        return time.perf_counter()

    @property
    def block_sample(self) -> bool:
        return self.enabled and bool(self.block_every) and self.round % self.block_every == 0

    @contextmanager
    def block(self, name: str):
        start = self._event()
        try:
            yield
        finally:
            self._blocks.append((name, start, self._event()))

    def capture(self, taps: list, count: int):
        """Copy on the producer stream before its next buffer reuse; no GPU clones.

        Each of two slots owns pinned CPU buffers until its completion event and
        disk write finish. Saturation skips captures, never stalls inference.
        """
        if not self.enabled or not self.sample_every or self.round % self.sample_every:
            return None
        try:
            buffers = self._tap_slots.get_nowait()
        except queue.Empty:
            self.tap_dropped += 1
            return None
        ready = None
        try:
            if (len(buffers) != len(taps) or
                    any(dst.shape != src.shape or dst.dtype != src.dtype for dst, src in zip(buffers, taps))):
                buffers[:] = [torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=t.is_cuda) for t in taps]
            for dst, src in zip(buffers, taps):
                dst[:count].copy_(src[:count].detach(), non_blocking=src.is_cuda)
            ready = self._event()
            return buffers, count, ready
        except (RuntimeError, OSError) as exc:
            # A failed CUDA copy may have earlier copies in flight: retire the
            # slot instead of reusing potentially unfinished host storage.
            self.error, self.enabled = str(exc), False
            return None

    def record(self, data: dict, taps=None, rows: list | None = None) -> None:
        data.update(rank=self.rank, round=self.round, observer_dropped=self.dropped,
                    tap_dropped=self.tap_dropped, block_sampled=self.block_sample,
                    timing_basis="cuda-stream-events", host_ms=data.pop("ms"))
        self._submit("round", data, self._marks, self._blocks, taps, rows)
        self.round += 1

    def _submit(self, kind, data, marks, blocks, taps, rows=None):
        if not self.enabled or not self._slots.acquire(blocking=False):
            self.dropped += 1
            if taps is not None:
                # Recycle only after D2H completes, without a serving-thread wait.
                self._jobs.put(("recycle", None, [], [], taps, None))
            return
        self._jobs.put((kind, data, marks, blocks, taps, rows))

    @staticmethod
    def _elapsed(start, end):
        return start.elapsed_time(end) if start is not None else None

    def _write_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            kind, data, marks, blocks, taps, rows = job
            try:
                if taps is not None and taps[2] is not None:
                    taps[2].synchronize()
                if kind == "recycle" or not self.enabled:
                    continue
                # Only this background writer waits. Stream order completes all
                # preceding block events too; the model never reads these events.
                if marks and marks[-1] is not None:
                    marks[-1].synchronize()
                if kind == "phase":
                    data["cuda_stream_ms"] = self._elapsed(*marks)
                    path = self.phase_file
                else:
                    data["ms"] = {name: self._elapsed(a, b) for name, a, b in
                                  zip(("forward", "sampling", "commit_taps", "drafter"), marks, marks[1:])}
                    totals = {}
                    for name, start, end in blocks:
                        elapsed = self._elapsed(start, end)
                        if elapsed is not None:
                            totals[name] = totals.get(name, 0.0) + elapsed
                    data["forward_blocks_ms"] = totals or None
                    experts = ("moe: routed experts", "moe: shared expert", "moe: gate/up", "moe: down")
                    data["forward_components_ms"] = ({
                        "expert_matmul_proxy": (sum(totals.get(k, 0) for k in experts)
                                                if any(k in totals for k in experts) else None),
                        "collective": totals.get("moe: all-gather"), "launch_host": None,
                    } if totals else None)
                    if taps is not None:
                        captured = [t[:taps[1]] for t in taps[0]]
                        target = self.root / f"rank{self.rank}-round{data['round']}.pt"
                        payload = [{**row, "rank": self.rank, "round": data["round"]} for row in rows or []]
                        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "wb") as f:
                            # Compact views: serializing a view otherwise saves
                            # unused backing storage (including unrelated rows).
                            torch.save({"taps": [t.clone() for t in captured], "rows": payload}, f)
                        data["tap_file"] = target.name
                        data["tap_sha256"] = [hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes())
                                              .hexdigest() for t in captured]
                    path = self.file
                # Reflect tail drops too, even if no later serving round fits.
                data["observer_dropped"] = self.dropped
                if kind == "round":
                    data["tap_dropped"] = self.tap_dropped
                with path.open("a") as f:
                    f.write(json.dumps(data, separators=(",", ":")) + "\n")
            except Exception as exc:  # a receipt failure must not kill the worker and strand its bounded queue
                self.error, self.enabled = str(exc), False
            finally:
                if taps is not None:
                    self._tap_slots.put(taps[0])
                if kind != "recycle":
                    self._slots.release()
                self._jobs.task_done()

    def flush(self) -> None:
        """Explicit offline/test drain. Never call from a decode or admission path."""
        if self._worker is not None:
            self._jobs.join()

    def close(self) -> None:
        if self._worker is not None:
            self.flush()
            self.enabled = False
            self._jobs.put(None)
            self._worker.join()
            self._worker = None
            atexit.unregister(self.close)


class PriceRounds:
    """Low-cost ROUND-to-ROUND rows for ruler.py price-v1; only rank zero writes."""

    def __init__(self, rank: int, hooks: bool, vote: bool) -> None:
        name = os.environ.get("TF_GLM_PRICE_ROUNDS") if rank == 0 else None
        self.enabled = bool(name)
        self.pending = None
        if not name:
            return
        self.path = Path(name)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self.hooks, self.vote = hooks, vote
        self.repeat = int(os.environ.get("TF_GLM_PRICE_REPEAT", "0"))

    def begin(self, live: list, *, context: str, slot: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self.flush(now)
        self.pending = {"load": len(live), "ctx": context, "slot": slot,
                        "ctx_tokens": [s.st.pos for s in live], "repeat": self.repeat,
                        "source": "hooked" if self.hooks else "plain", "hooks": self.hooks,
                        "vote": self.vote, "started": now}

    def executed(self, rows: int, path: str) -> None:
        if self.pending is not None:
            self.pending.update(rows=rows, path=path)

    def contaminate(self) -> None:
        self.pending = None

    def flush(self, now: float | None = None) -> None:
        if self.pending is None or not self.enabled:
            return
        row = dict(self.pending)
        if "rows" not in row:
            self.pending = None
            return
        row["total_round_ms"] = 1000 * ((now if now is not None else time.perf_counter()) - row.pop("started"))
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError as exc:
            self.error = str(exc)
            self.enabled = False
        self.pending = None
