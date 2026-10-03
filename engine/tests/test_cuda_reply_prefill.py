"""Foreground arrival races around the optional worker-owned idle job."""

import queue
import threading

import pytest

from tensorfold.cuda.scheduler import Scheduler
from test_cuda_pooled_scheduler import Decoder, request, answer, WAIT


class IdleDecoder(Decoder):
    fair_schedule = reply_prefill = True

    def __init__(self):
        super().__init__()
        self.idle_entered, self.idle_release = threading.Event(), threading.Event()
        self.hold = "layer"
        self.completed = threading.Event()
        self.finishes = []

    def begin_admit(self, s):
        if getattr(s, "reply_prefill", False):
            assert not self.streams
            assert s.priority == "background" and s.prefill_slice_layers == 1
            assert s.copy_request is s.cofill is False
            self.streams.append(s)
            if self.hold == "admit":
                self.idle_entered.set()
                assert self.idle_release.wait(WAIT)
        else:
            assert not any(getattr(p, "reply_prefill", False) for p in self.streams), "foreground admitted before preemption"
            super().admit(s)

    def prefill_step(self, **kw):
        idle = [s for s in self.streams if getattr(s, "reply_prefill", False)]
        if idle:
            self.idle_entered.set()
            assert self.idle_release.wait(WAIT)
            s = idle[0]
            s.done = True
            return [s]
        return []

    def round(self):
        assert all(s.done or not getattr(s, "reply_prefill", False) for s in self.streams)
        return [s for s in super().round() if not getattr(s, "reply_prefill", False)]

    def finish(self, done):
        self.finishes.extend(done)
        super().finish(done)
        if any(getattr(s, "reply_prefill", False) for s in done):
            self.completed.set()


def scheduler():
    d = IdleDecoder()
    s = Scheduler(d, max_streams=1, admit_per_round=1)
    result = answer(request(s, [1], 1))
    return d, s, result["_reply_prefill_epoch"]


@pytest.mark.parametrize("stage", ["admit", "layer"])
def test_arrival_during_idle_work_releases_before_foreground(stage):
    d, s, epoch = scheduler()
    d.hold = stage
    try:
        assert s.submit_reply_prefill([1, 2, 3], base=1, policy="0", epoch=epoch)
        assert d.idle_entered.wait(WAIT)
        job = s.reply_active
        foreground = request(s, [4], 2)
        assert job.reply_cancel.wait(WAIT)
        d.idle_release.set()
        assert isinstance(answer(foreground), dict)
        assert job in d.finishes and not job.out and job.rounds == 0
        assert s.reply_active is None and not d.streams and not s.boxes
        assert not s.submit_reply_prefill([1, 2], base=1, policy="0", epoch=epoch)
    finally:
        d.idle_release.set()
        s._poison(RuntimeError("test shutdown"))


def test_completion_retains_no_response_box_or_decode():
    d, s, epoch = scheduler()
    try:
        d.idle_release.set()
        assert s.submit_reply_prefill([1, 2], base=1, policy="0", epoch=epoch)
        assert d.completed.wait(WAIT)
        job = d.finishes[-1]
        assert job.out == [] and job.rounds == 0
        assert isinstance(answer(request(s, [4], 1)), dict)
        assert not s.boxes and not d.streams
    finally:
        s._poison(RuntimeError("test shutdown"))


def test_pending_queue_is_bounded_and_foreground_invalidates_it():
    d, s, epoch = scheduler()
    try:
        d.gate.clear()
        d.entered.clear()
        active = request(s, [7], 3)
        # The submit owns epoch 2 before any optional render can queue work.
        assert d.entered.wait(WAIT)
        with s.state_lock:
            epoch = s.reply_epoch
        for n in range(12):
            assert s.submit_reply_prefill([7, n], base=1, policy="0", epoch=epoch)
        assert s.reply_pending.prompt == [7, 11] and s.reply_active is None
        enqueued = threading.Event()
        put = s.waiting.put
        def observed_put(item):
            put(item)
            enqueued.set()
        s.waiting.put = observed_put
        next_request = request(s, [8], 1)
        assert enqueued.wait(WAIT)
        d.idle_release.set()
        d.gate.set()
        assert isinstance(answer(active), dict) and isinstance(answer(next_request), dict)
        assert s.reply_pending is None and not d.idle_entered.is_set()
    finally:
        d.gate.set()
        d.idle_release.set()
        s._poison(RuntimeError("test shutdown"))


def test_nonfatal_idle_miss_does_not_poison_next_foreground():
    d, s, epoch = scheduler()
    original = d.begin_admit
    def missing(stream):
        if getattr(stream, "reply_prefill", False):
            d.idle_entered.set()
            raise ValueError("completed prefix evicted")
        return original(stream)
    d.begin_admit = missing
    try:
        assert s.submit_reply_prefill([1, 2], base=1, policy="0", epoch=epoch)
        assert d.idle_entered.wait(WAIT)
        assert isinstance(answer(request(s, [5], 1)), dict)
        assert not s.failed and not d.streams
    finally:
        s._poison(RuntimeError("test shutdown"))


def test_optional_job_without_backing_credit_leaves_no_active_owner():
    # begin_admit may decline without raising when reply admission cannot reserve backing.
    # Run one worker operation directly so the next loop cannot hide a stale job.
    from collections import deque
    from tensorfold.cuda.streams import Stream

    d = IdleDecoder()
    d.begin_admit = lambda stream: False
    s = Scheduler.__new__(Scheduler)
    s.decoder, s.state_lock, s.waiting = d, threading.Lock(), queue.Queue()
    s.pending, s.deferred = None, deque()
    job = Stream([1, 2], 1)
    job.cancelled = lambda: False
    s.reply_pending, s.reply_active = job, None
    s._start_reply_prefill()
    assert s.reply_active is s.reply_pending is None
    assert not d.streams and not d.finishes


def test_capability_is_required_and_flags_off_cannot_queue():
    d = Decoder()
    d.reply_prefill, d.fair_schedule = True, False
    with pytest.raises(ValueError, match="fair scheduling"):
        Scheduler(d)
    d.reply_prefill = False
    s = Scheduler(d)
    assert not s.submit_reply_prefill([1, 2], base=1, policy="0", epoch=0)
    assert isinstance(answer(request(s, [3], 1)), dict)


def test_arrival_between_loop_check_and_admission_snapshot():
    # Exercise the exact race deterministically with no worker thread.
    d = IdleDecoder()
    s = Scheduler.__new__(Scheduler)
    s.decoder, s.reply_prefill, s.max_streams, s.admit_per_round = d, True, 1, 1
    s.waiting, s.boxes, s.state_lock = queue.Queue(), {}, threading.Lock()
    from collections import deque
    from tensorfold.cuda.streams import Stream
    s.deferred = deque()
    idle = Stream([1, 2], 1)
    idle.reply_prefill, idle.reply_cancel = True, threading.Event()
    idle.cancelled = idle.reply_cancel.is_set
    s.reply_active = idle
    d.streams.append(idle)
    s._preempt_reply_prefill()  # arrives immediately after this check
    idle.reply_cancel.set()
    foreground = Stream([3], 1)
    foreground.cancelled = lambda: False
    s.waiting.put((foreground, queue.Queue()))
    assert s._admit_fair() == [foreground]
    assert d.finishes == [idle] and s.reply_active is None
