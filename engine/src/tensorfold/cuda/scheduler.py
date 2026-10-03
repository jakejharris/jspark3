"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable

from .streams import Stream


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4, admit_per_round: int | None = None) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        self.waiting: queue.Queue = queue.Queue()
        self.boxes: dict[int, queue.Queue] = {}
        self.admit_per_round = admit_per_round
        self.pending = None
        self.deferred = deque()
        self.failed: Exception | None = None
        self.state_lock = threading.Lock()
        self.reply_prefill = bool(getattr(decoder, "reply_prefill", False))
        if self.reply_prefill and not getattr(decoder, "fair_schedule", False):
            raise ValueError("reply prefill requires fair scheduling")
        self.reply_epoch = 0
        self.reply_pending = self.reply_active = None
        self.reply_render_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, **request) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos)
        cancel = threading.Event()
        external = request.pop("cancelled", None)
        callback_error = None

        def cancelled():
            nonlocal callback_error
            if cancel.is_set():
                return True
            try:
                if external is not None and external():
                    cancel.set()
                return cancel.is_set()
            except Exception as exc:
                callback_error = exc
                cancel.set()
                return True                # preserve ownership until the worker acknowledges

        for name, value in request.items():
            setattr(stream, name, value)
        stream.cancelled = cancelled
        stream.emit = lambda new: True if stream.cancelled() else (box.put(("tokens", new)), False)[1]
        with self.state_lock:
            if self.failed is not None:
                raise self.failed
            if self.reply_prefill:
                self.reply_epoch += 1
                stream.reply_epoch = self.reply_epoch
                self.reply_pending = None
                if self.reply_active is not None:
                    self.reply_active.reply_cancel.set()
            self.waiting.put((stream, box))
        while True:
            try:
                kind, value = box.get(timeout=0.05) if external is not None else box.get()
            except queue.Empty:
                cancelled()
                continue                    # release request resources only after the worker acknowledges
            if kind == "tokens":
                if not cancel.is_set():
                    try:
                        if emit(value):
                            cancel.set()
                    except Exception as exc:        # finish GPU ownership before unwinding the caller
                        callback_error = exc
                        cancel.set()
            elif kind == "error":
                if callback_error is not None:
                    raise callback_error
                raise value
            else:
                if callback_error is not None:
                    raise callback_error
                return value

    def submit_reply_prefill(self, prompt: list[int], *, base: int, policy: str, epoch: int | None,
                            fingerprint: str = "") -> bool:
        """At most one pending idle job. A newer foreground request invalidates the prediction."""
        if not self.reply_prefill:
            return False
        stream = Stream(list(prompt), 1, draft=True, stop_eos=False)
        stream.reply_prefill, stream.reply_base = True, base
        stream.reply_fingerprint = fingerprint
        stream.policy, stream.priority = policy, "background"
        stream.copy_request = stream.cofill = False
        stream.prefill_slice_layers = 1
        stream.reply_cancel = threading.Event()
        stream.cancelled = stream.reply_cancel.is_set
        with self.state_lock:
            if self.failed is not None or epoch != self.reply_epoch:
                return False
            if self.reply_active is not None:
                self.reply_active.reply_cancel.set()
            self.reply_pending = stream
        return True

    def _start_reply_prefill(self) -> None:
        with self.state_lock:
            if self.pending is not None or self.deferred or not self.waiting.empty() or self.decoder.live():
                return
            stream, self.reply_pending = self.reply_pending, None
            self.reply_active = stream
        if stream is None:
            return
        try:
            if not stream.cancelled():
                if self.decoder.begin_admit(stream) is False:
                    self.reply_active = None
                    return  # optional prediction has no backed admission; drop it
                if stream.done:
                    self.decoder.finish([stream])
                    self.reply_active = None
            else:
                self.reply_active = None
        except Exception:
            self.reply_active = None
            if getattr(self.decoder, "broken", None) is not None:
                raise
            # A missing memory prefix, full store or a non-fitting prediction is optional work.

    def _preempt_reply_prefill(self) -> None:
        stream = self.reply_active
        if stream is not None and stream.cancelled():
            stream.done, stream.finished = True, time.perf_counter()
            self.decoder.finish([stream])        # all ranks close the partial layer before any foreground fill
            self.reply_active = None

    def _admit(self, first=None) -> list[Stream]:
        if getattr(self.decoder, "fair_schedule", False):
            return self._admit_fair(first)
        done = []
        attempts = 0
        if first is not None and self.decoder.live() >= self.max_streams:
            self.pending = first
            return done
        while self.decoder.live() < self.max_streams:
            if self.admit_per_round is not None and attempts >= self.admit_per_round:
                break
            if first is not None:
                (stream, box), first = first, None
            else:
                try:
                    stream, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
            self.boxes[id(stream)] = box
            try:
                if stream.cancelled():
                    from .server import RequestCancelled

                    raise RequestCancelled("the client left before the request started")
                fits = getattr(self.decoder, "can_admit", None)
                if fits is not None and not fits(stream):
                    if not self.decoder.live():
                        raise ValueError("this request does not fit the idle engine's token pool")
                    self.boxes.pop(id(stream))
                    self.pending = (stream, box)
                    break
                attempts += 1
                begin = getattr(self.decoder, "begin_admit", None)
                if (begin or self.decoder.admit)(stream) is False:
                    self.boxes.pop(id(stream))
                    self.pending = (stream, box)
                    break
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                if getattr(self.decoder, "fatal_errors", False) and getattr(self.decoder, "broken", None) is not None:
                    raise
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _admit_fair(self, first=None) -> list[Stream]:
        """Scan one finite queue snapshot for fitting work; old blocked owners eventually drain the pool."""
        from .server import RequestCancelled

        if first is not None:
            self.deferred.appendleft(first)
        with self.state_lock:
            while True:
                try:
                    self.deferred.append(self.waiting.get_nowait())
                except queue.Empty:
                    break
        if self.reply_prefill:
            # Every submit in this snapshot has already cancelled the idle job.
            # It may have arrived during ADMIT or since the loop's first check.
            self._preempt_reply_prefill()
        done, blocked, attempts = [], [], 0
        try:
            for _ in range(len(self.deferred)):
                if (self.decoder.live() >= self.max_streams or
                        self.admit_per_round is not None and attempts >= self.admit_per_round):
                    break
                stream, box = self.deferred.popleft()
                self.boxes[id(stream)] = box
                try:
                    if stream.cancelled():
                        raise RequestCancelled("the client left before the request started")
                    fits = getattr(self.decoder, "can_admit", None)
                    if fits is not None and not fits(stream):
                        if not self.decoder.live():
                            raise ValueError("this request does not fit the idle engine's token pool")
                        self.boxes.pop(id(stream))
                        blocked.append((stream, box))
                        now = time.monotonic()
                        if not hasattr(stream, "admission_blocked_at"):
                            stream.admission_blocked_at = now
                        if now - stream.admission_blocked_at >= 8.0:
                            break           # stop backfilling until this reservation fits (finite replies)
                        continue
                    attempts += 1
                    begin = getattr(self.decoder, "begin_admit", None)
                    if (begin or self.decoder.admit)(stream) is False:
                        self.boxes.pop(id(stream))
                        blocked.append((stream, box))
                        break
                except Exception as exc:
                    if getattr(self.decoder, "fatal_errors", False) and getattr(self.decoder, "broken", None) is not None:
                        raise
                    self.boxes.pop(id(stream)).put(("error", exc))
                    continue
                if stream.done:
                    done.append(stream)
        finally:
            self.deferred.extendleft(reversed(blocked))
        return done

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            try:
                if self.reply_prefill:
                    self._preempt_reply_prefill()
                self._cancel_waiters()
                first, self.pending = self.pending, None
                if first is None and self.deferred:
                    first = self.deferred.popleft()
                if first is None and not self.decoder.live():
                    if self.reply_prefill:
                        self._start_reply_prefill()
                        if not self.decoder.live():
                            try:
                                first = self.waiting.get(timeout=0.05)
                            except queue.Empty:
                                continue
                    else:
                        first = self.waiting.get()     # flags off: original blocking idle path
                if self.failed is not None:
                    first[1].put(("error", self.failed))
                    continue
                done = self._admit(first)
                try:
                    fill = getattr(self.decoder, "prefill_step", None)
                    if fill is not None:
                        done += fill(queued=self.pending is not None or bool(self.deferred) or not self.waiting.empty())
                    done += self.decoder.round()
                except Exception as exc:             # recoverable decoders may continue after dropping streams
                    if getattr(self.decoder, "fatal_errors", False) and getattr(self.decoder, "broken", None) is not None:
                        raise
                    for s in self.decoder.drop():
                        self._reply(s, "error", exc)
                self.decoder.finish(done)
                for s in done:
                    if s is self.reply_active:
                        self.reply_active = None
                    stats = s.stats()
                    if hasattr(s, "reply_epoch"):
                        stats["_reply_prefill_epoch"] = s.reply_epoch
                    self._reply(s, *(("error", s.error) if s.error is not None else ("done", stats)))
                if not self.decoder.live() and (self.pending is not None or self.deferred):
                    time.sleep(0.05)  # backed admission may wait for storage; keep cancellation responsive
            except Exception as exc:
                self._poison(exc)
                return                  # failed submitters are rejected; never retry a poisoned decoder

    def _poison(self, exc: Exception) -> None:
        """Answer every owner if distributed state or the worker's cleanup can no longer progress."""
        failure = RuntimeError("the CUDA serving worker failed; restart both/all ranks before another request")
        failure.__cause__ = exc
        with self.state_lock:
            self.failed = failure
            self.reply_pending = None
            self.reply_active = None
            active = list(self.boxes.values())
            self.boxes.clear()
            if self.pending is not None:
                self.pending[1].put(("error", exc))
                self.pending = None
            while self.deferred:
                _, box = self.deferred.popleft()
                box.put(("error", exc))
            while True:
                try:
                    _, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
                box.put(("error", exc))
        try:
            self.decoder.drop()
        except Exception:
            pass                           # no additional GPU/protocol calls can repair a poisoned worker
        for box in active:
            box.put(("error", exc))         # the caller may now release its prepared request

    def _cancel_waiters(self) -> None:
        """Disconnected unstarted requests release ownership even while every GPU slot is busy."""
        from .server import RequestCancelled

        with self.state_lock:
            waiting = []
            if self.pending is not None:
                waiting.append(self.pending)
                self.pending = None
            waiting.extend(self.deferred)
            self.deferred.clear()
            while True:
                try:
                    waiting.append(self.waiting.get_nowait())
                except queue.Empty:
                    break
            for stream, box in waiting:
                try:
                    cancelled = stream.cancelled()
                except Exception as exc:
                    box.put(("error", exc))
                    continue
                if cancelled:
                    box.put(("error", RequestCancelled("the client left before the request started")))
                else:
                    if getattr(self.decoder, "fair_schedule", False):
                        self.deferred.append((stream, box))
                    else:
                        self.waiting.put((stream, box))
