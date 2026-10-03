"""Fair prompt service, finite admission bypass, and rank-agreed priority policies."""

from collections import deque
from types import SimpleNamespace
import queue
import threading

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.server import RequestCancelled
from tensorfold.families.glm5_next.cuda.pool import TokenPool
from test_cuda_pooled_scheduler import Decoder

# These fixtures use CPU Torch only; no model or GPU is loaded.
pytest.importorskip("torch")
from test_glm_batched_host import setup_decoder, _stream, _drain
from test_cuda_geometry import allocations
from test_glm_prefill_slices import sliced_decoder, drain_steps, replay


def scheduler(fair=True):
    s = Scheduler.__new__(Scheduler)
    s.decoder = Decoder()
    s.decoder.fair_schedule = fair
    s.max_streams, s.admit_per_round = 4, 1
    s.waiting, s.boxes, s.deferred = queue.Queue(), {}, deque()
    s.pending = s.failed = None
    s.state_lock = threading.Lock()
    s.reply_prefill = False
    return s


def enqueue(s, prompt):
    stream = _stream(prompt, 10)
    stream.cancelled = lambda: False
    box = queue.Queue()
    s.waiting.put((stream, box))
    return stream, box


@pytest.mark.parametrize("fair", [False, True])
def test_fitting_short_bypasses_blocked_large_only_when_enabled(fair):
    s = scheduler(fair)
    s.decoder.streams = [_stream([1], 10)]
    large, large_box = enqueue(s, [99])
    small, small_box = enqueue(s, [2])
    s._admit()
    assert [item[0] for item in s.decoder.seen] == ([[2]] if fair else [])
    assert large_box.empty() and small_box.empty()
    if fair:
        assert list(s.deferred) == [(large, large_box)]
    else:
        assert s.pending == (large, large_box)


def test_old_blocked_reservation_stops_backfill_then_runs_when_pool_drains(monkeypatch):
    s = scheduler()
    s.decoder.streams = [_stream([1], 10)]
    now = [0.0]
    monkeypatch.setattr("tensorfold.cuda.scheduler.time.monotonic", lambda: now[0])
    large, box = enqueue(s, [99])
    s._admit()
    for second in (1, 4, 7):
        now[0] = second
        small, _ = enqueue(s, [2])
        s._admit()
        assert small in s.decoder.streams
        s.decoder.streams.remove(small)
    now[0] = 8
    small, _ = enqueue(s, [3])
    s._admit()
    assert small not in s.decoder.streams and s.deferred[0] == (large, box)
    s.decoder.streams.clear()
    s._admit()
    assert s.decoder.streams == [large]
    s._admit()
    assert small in s.decoder.streams


@pytest.mark.parametrize("action", ["cancel", "callback_error", "poison"])
def test_deferred_owners_are_replied_to_during_cancel_or_failure(action):
    s = scheduler()
    s.decoder.streams = [_stream([1], 10)]
    stream, box = enqueue(s, [99])
    s._admit()
    assert len(s.deferred) == 1
    if action == "poison":
        s._poison(RuntimeError("worker failed"))
    else:
        def cancelled():
            if action == "callback_error":
                raise OSError("disconnected")
            return True
        stream.cancelled = cancelled
        s._cancel_waiters()
    kind, error = box.get_nowait()
    assert kind == "error" and isinstance(error, {"cancel": RequestCancelled, "callback_error": OSError,
                                                  "poison": RuntimeError}[action])
    assert not s.deferred and s.waiting.empty() and not s.boxes


def test_priority_and_aging_pick_prompt_work_without_starving_background(sliced_decoder, monkeypatch):
    mod, build = sliced_decoder
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    d = build()
    now = [0.0]
    monkeypatch.setattr(mod.time, "perf_counter", lambda: now[0])
    background, interactive = _stream([2] * 40, 4), _stream([3] * 16, 4)
    background.priority, interactive.priority = "background", "interactive"
    d.begin_admit(background)
    d.begin_admit(interactive)
    d.prefill_step()
    assert interactive.fill_pos == 8 and background.fill_pos == 0
    # Reset the interactive wait after service, but let the unserved background age.
    now[0] = 2.0
    interactive.fill_served = now[0]
    d.prefill_step()
    assert background.fill_pos == 8
    drain_steps(d)
    assert all(o == {background.sid: background.out, interactive.sid: interactive.out}
               for o in replay(mod, build, d.messages))


@pytest.mark.parametrize("fair", [False, True])
@pytest.mark.parametrize("priority", [None, "interactive", "background"])
def test_priority_caps_only_opted_in_background_depth_and_preserves_tokens(setup_decoder, monkeypatch, fair, priority):
    _, make = setup_decoder
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", str(int(fair)))
    d = make()
    s = _stream([3, 5, 8], 30, policy="f7")
    s.priority = priority
    d.admit(s)
    expected_depth = 1 if fair and priority == "background" else 7
    assert s.policy_code[1] == expected_depth and s.depth_policy.most == expected_depth
    _drain(d)
    reference = make()
    serial = _stream(s.prompt, s.count, draft=False)
    reference.admit(serial)
    _drain(reference)
    assert s.out == serial.out
    assert max(s.depths) <= expected_depth


def test_untagged_prefill_retains_shortest_remaining_when_flag_off(sliced_decoder):
    _, build = sliced_decoder
    d = build()
    long, short = _stream([2] * 40, 4), _stream([3] * 16, 4)
    long.priority = "interactive"
    short.priority = "background"
    d.begin_admit(long)
    d.begin_admit(short)
    long.fill_served -= 1_000_000
    d.prefill_step()
    assert short.fill_pos == 8 and long.fill_pos == 0
    d.drop()


@pytest.mark.parametrize("aging", [False, True])
def test_continuing_short_arrivals_obey_aging_bound_and_negative_control(sliced_decoder, monkeypatch, aging):
    mod, build = sliced_decoder
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    if not aging:
        monkeypatch.setattr(mod, "PREFILL_AGE_ROWS_PER_SECOND", 0)
    now = [0.0]
    monkeypatch.setattr(mod.time, "perf_counter", lambda: now[0])
    d = build()
    long = _stream([2] * 200, 1)
    long.priority = "background"
    d.begin_admit(long)
    selected_at = None
    # New interactive jobs continually arrive. Their finite 8192-row priority advantage
    # plus the prompt length is overtaken after <= (200 + 8192) / 8192 seconds.
    for tick in range(1, 30):
        now[0] = tick / 10
        short = _stream([3], 1)
        short.priority = "interactive"
        d.begin_admit(short)
        d.finish(d.prefill_step())
        if long.fill_pos:
            selected_at = now[0]
            break
    if aging:
        assert selected_at is not None and selected_at <= 1.1
    else:
        assert selected_at is None  # the identical progress check would fail with aging disabled
    d.drop()


def test_real_fragmented_pool_admits_fitting_request_and_preserves_blocked_owner(setup_decoder, monkeypatch):
    _, make = setup_decoder
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    d = make(4, limit=128)
    # Allocate three separated spans, then release the middle. The largest tail is 56,
    # although there are 92 total free tokens. The large request needs a 72-token span.
    d.capacity = 164
    d.pool = TokenPool(d.capacity)
    a, hole, c = _stream([1], 24), _stream([2], 24), _stream([3], 24)
    for stream in (a, hole, c):
        d.admit(stream)
    d.finish([hole])
    large, small = _stream([4], 63), _stream([5], 15)
    before = dict(d.pool.spans)
    assert d.pool.available >= 72 and not d.can_admit(large) and d.can_admit(small)
    assert d.pool.spans == before
    s = scheduler()
    s.decoder = d
    # Use the synchronous path here: isolate real reservation/admission from prefill timing.
    d.begin_admit = d.admit
    for stream in (large, small):
        stream.cancelled = lambda: False
        s.waiting.put((stream, queue.Queue()))
    s._admit()
    assert small.sid in d.streams and large not in d.streams.values()
    assert s.deferred[0][0] is large
    assert small.start == hole.start and small.reserved_tokens == len(small.prompt) + small.count
    d.drop()


@pytest.mark.parametrize("body,valid", [({}, True), ({"priority": "interactive"}, True),
    ({"priority": "background", "prefill_slice_layers": 6}, True), ({"prefill_slice_layers": 0}, True),
    ({"priority": "urgent"}, False), ({"priority": []}, False), ({"prefill_slice_layers": True}, False),
    ({"prefill_slice_layers": -1}, False), ({"prefill_slice_layers": 1.5}, False)])
def test_request_fields_validate_before_streaming(body, valid):
    from tensorfold.families.glm5_next.cuda.app import GlmApp

    app = GlmApp.__new__(GlmApp)
    app.engine = SimpleNamespace()
    assert (app._check_fields(body) is None) is valid


@pytest.mark.parametrize("fail", [False, True])
def test_request_metadata_reaches_scheduler_and_is_cleared_after_run(monkeypatch, fail):
    from tensorfold.cuda.server import App, PreparedRequest
    from tensorfold.families.glm5_next.cuda.app import GlmApp
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    submitted = []
    engine = GlmEngine.__new__(GlmEngine)
    engine.limit, engine.concurrent, engine.serial_only, engine.policy = 128, True, False, "fc7:0.3"
    engine.request = threading.local()
    engine.tower = None
    engine.scheduler = SimpleNamespace(submit=lambda *a, **kw: submitted.append(kw) or {})
    app = GlmApp.__new__(GlmApp)
    app.engine = engine

    def run(self, *args, **kwargs):
        engine.generate([2, 3], 8, None, lambda _: False)
        if fail:
            raise RuntimeError("request failed")
        return {}

    monkeypatch.setattr(App, "run", run)
    prepared = PreparedRequest([2, 3], 8, [], False)
    body = dict(priority="background", prefill_slice_layers=4)
    if fail:
        with pytest.raises(RuntimeError, match="request failed"):
            app.run(body, False, lambda _: False, prepared=prepared)
    else:
        app.run(body, False, lambda _: False, prepared=prepared)
    assert submitted[-1]["priority"] == "background" and submitted[-1]["prefill_slice_layers"] == 4
    assert engine.request.priority is None and engine.request.prefill_slice_layers is None
    if not fail:
        app.run({}, False, lambda _: False, prepared=prepared)
        assert submitted[-1]["priority"] is None and submitted[-1]["prefill_slice_layers"] is None
