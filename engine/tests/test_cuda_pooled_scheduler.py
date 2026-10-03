"""A capacity waiter, cancel or worker failure cannot strand another request."""

import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.server import RequestCancelled
from tensorfold.cuda.streams import Stream

WAIT = 5


class Decoder:
    def __init__(self):
        self.streams = []
        self.gate, self.entered = threading.Event(), threading.Event()
        self.gate.set()
        self.seen = []
        self.fail_finish = False

    def live(self):
        return len(self.streams)

    def can_admit(self, s):
        return not self.streams or s.prompt != [99]

    def admit(self, s):
        self.seen.append((list(s.prompt), getattr(s, "policy", None), getattr(s, "images", None)))
        self.streams.append(s)
        s.take([1])

    def round(self):
        self.entered.set()
        assert self.gate.wait(WAIT)
        for s in self.streams:
            if not s.done:
                s.counted(1)
                s.take([2])
        return [s for s in self.streams if s.done]

    def finish(self, done):
        if self.fail_finish:
            raise RuntimeError("finish failed")
        self.streams = [s for s in self.streams if s not in done]

    def drop(self):
        streams, self.streams = self.streams, []
        return streams


@pytest.fixture(autouse=True, params=[False, True], ids=["fifo", "fair"])
def admission_mode(request, monkeypatch):
    monkeypatch.setattr(Decoder, "fair_schedule", request.param, raising=False)


def request(scheduler, prompt, count, **kw):
    result, done = [], threading.Event()

    def run():
        try:
            result.append(scheduler.submit(prompt, count, None, False, kw.pop("emit", lambda _: False), **kw))
        except Exception as exc:
            result.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return SimpleNamespace(result=result, done=done)


def answer(req):
    assert req.done.wait(WAIT), "request stranded without a worker reply"
    return req.result[0]


def test_capacity_waiter_finishes_after_peer_and_metadata_reaches_decoder():
    dec = Decoder()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=2, admit_per_round=1)
    a = request(sched, [1], 3, policy="fc7:0.3", images=["owned"])
    assert dec.entered.wait(WAIT)
    b = request(sched, [99], 2)
    dec.gate.set()
    assert isinstance(answer(a), dict)
    assert isinstance(answer(b), dict)
    assert dec.seen == [([1], "fc7:0.3", ["owned"]), ([99], None, None)]


def test_cancelled_capacity_waiter_releases_without_ever_entering_decoder():
    dec = Decoder()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=2, admit_per_round=1)
    a = request(sched, [1], 3)
    assert dec.entered.wait(WAIT)
    cancelled = threading.Event()
    b = request(sched, [99], 2, cancelled=cancelled.is_set)
    cancelled.set()
    dec.gate.set()
    assert isinstance(answer(a), dict)
    assert isinstance(answer(b), RequestCancelled)
    assert dec.seen == [([1], None, None)]


def test_finish_failure_answers_all_owners_and_refuses_future_requests():
    dec = Decoder()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=2, admit_per_round=1)
    a = request(sched, [1], 2)
    assert dec.entered.wait(WAIT)
    b = request(sched, [2], 4)
    dec.fail_finish = True
    dec.gate.set()
    assert isinstance(answer(a), RuntimeError)
    assert isinstance(answer(b), RuntimeError)
    assert "restart" in str(answer(request(sched, [3], 1)))


def test_callback_failure_waits_for_gpu_owner_before_unwinding():
    dec = Decoder()
    sched = Scheduler(dec, max_streams=2, admit_per_round=1)

    def fail(_):
        raise OSError("socket closed")

    a = request(sched, [1], 5, emit=fail)
    assert isinstance(answer(a), OSError)
    assert not dec.streams
    assert isinstance(answer(request(sched, [2], 1)), dict)


@pytest.mark.parametrize("stage", ["admit", "round"])
def test_distributed_failure_poison_drains_queue(stage):
    class Distributed(Decoder):
        fatal_errors = True
        broken = None

        def admit(self, s):
            super().admit(s)
            if stage == "admit":
                self.entered.set()
                assert self.gate.wait(WAIT)
                self.broken = RuntimeError("admission protocol failed")
                raise self.broken

        def round(self):
            self.entered.set()
            assert self.gate.wait(WAIT)
            self.broken = RuntimeError("round protocol failed")
            raise self.broken

    dec = Distributed()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=2, admit_per_round=1)
    a = request(sched, [1], 3)
    assert dec.entered.wait(WAIT)
    b = request(sched, [2], 3)
    dec.gate.set()
    assert isinstance(answer(a), RuntimeError)
    assert isinstance(answer(b), RuntimeError)
    assert sched.failed is not None
    assert "restart" in str(answer(request(sched, [3], 1)))
    assert not dec.streams


def test_cancelled_waiter_is_acknowledged_while_every_slot_stays_full():
    import queue

    class Stepped(Decoder):
        def __init__(self):
            super().__init__()
            self.steps, self.rounds = queue.Queue(), queue.Queue()

        def round(self):
            self.rounds.put(True)
            self.steps.get(timeout=WAIT)
            return super().round()

    dec = Stepped()
    sched = Scheduler(dec, max_streams=1)
    live_cancel = threading.Event()
    a = request(sched, [1], 100, cancelled=live_cancel.is_set)
    dec.rounds.get(timeout=WAIT)
    waiting_cancel = threading.Event()
    b = request(sched, [2], 2, cancelled=waiting_cancel.is_set)
    waiting_cancel.set()
    dec.steps.put(True)
    dec.rounds.get(timeout=WAIT)
    assert isinstance(answer(b), RequestCancelled)
    assert not a.done.is_set() and dec.live() == 1
    live_cancel.set()
    dec.steps.put(True)
    assert isinstance(answer(a), dict)


@pytest.mark.parametrize("stage", ["admit", "finish"])
def test_fatal_reply_waits_until_decoder_releases_active_ownership(stage):
    class BlockedDrop(Decoder):
        fatal_errors = True
        broken = None

        def __init__(self):
            super().__init__()
            self.dropping, self.release_drop = threading.Event(), threading.Event()

        def admit(self, s):
            super().admit(s)
            if stage == "admit":
                self.broken = RuntimeError("admission failed after ownership")
                raise self.broken

        def finish(self, done):
            raise RuntimeError("finish failed")

        def drop(self):
            self.dropping.set()
            assert self.release_drop.wait(WAIT)
            return super().drop()

    dec = BlockedDrop()
    sched = Scheduler(dec, max_streams=1)
    a = request(sched, [1], 2)
    assert dec.dropping.wait(WAIT)
    assert dec.live() == 1 and not a.done.is_set()
    dec.release_drop.set()
    assert isinstance(answer(a), RuntimeError)
    assert not dec.streams


def test_poisoned_worker_stops_even_if_decoder_cleanup_raises():
    class BadCleanup(Decoder):
        def finish(self, done):
            raise RuntimeError("finish failed")

        def drop(self):
            raise RuntimeError("cleanup failed")

    dec = BadCleanup()
    sched = Scheduler(dec, max_streams=1)
    assert isinstance(answer(request(sched, [1], 2)), RuntimeError)
    sched.thread.join(WAIT)
    assert not sched.thread.is_alive()
    assert "restart" in str(answer(request(sched, [2], 1)))


def test_cancel_callback_error_waits_for_blocked_admission_acknowledgement():
    class BlockedAdmission(Decoder):
        def admit(self, s):
            self.streams.append(s)
            self.entered.set()
            assert self.gate.wait(WAIT)
            s.take([1])

    dec = BlockedAdmission()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=2)
    failed = threading.Event()

    def cancelled():
        if threading.current_thread() is not sched.thread:
            failed.set()
            raise OSError("cancel probe failed")
        return False

    a = request(sched, [1], 5, cancelled=cancelled)
    assert dec.entered.wait(WAIT) and failed.wait(WAIT)
    assert not a.done.is_set() and dec.live() == 1
    dec.gate.set()
    assert isinstance(answer(a), OSError)
    assert not dec.streams and sched.failed is None
    assert isinstance(answer(request(sched, [2], 1)), dict)


def test_cancel_callback_error_during_emission_does_not_poison_peers():
    dec = Decoder()
    sched = Scheduler(dec, max_streams=2)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise OSError("cancel probe failed")
        return False

    assert isinstance(answer(request(sched, [1], 5, cancelled=cancelled)), OSError)
    assert not dec.streams and sched.failed is None
    assert isinstance(answer(request(sched, [2], 1)), dict)
