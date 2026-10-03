"""Tiny tiled attention prefills must become ready before incumbent short replies finish."""

import copy
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.scheduler import Scheduler
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream
from test_glm_prefill_slices import sliced_decoder, replay


@pytest.fixture
def wide_decoder(sliced_decoder, setup_decoder):
    mod, make = setup_decoder
    from tensorfold.families.glm5_next.cuda import decode

    def build(express=True):
        d = make(8, limit=8192, cache_bytes=10_000_000)
        d.w.layers = list(range(45))
        d.owner.e.prefill_rows = 512
        d.owner.e.pbuf = SimpleNamespace(rows=512, x=torch.zeros(512, 2), fnormed=torch.zeros(512, 2),
                                        overlay=None, taps=[torch.zeros(512, 2), torch.zeros(512, 2)])
        original = d._engine

        def engine(*args):
            e = original(*args)
            e.reset = e.st.reset
            e.overlay = lambda first, n: (first, n)
            e.tap_rows = lambda n, b: torch.cat([t[:n] for t in b.taps], dim=1)
            e.sample = lambda logits, pos, sampling: decode.sample_rows(e.w, logits, pos, sampling)
            return e

        d._engine = engine
        if express:
            d.express_buf = copy.deepcopy(d.owner.e.pbuf)
        return d

    return mod, build


def serve_eight(d, staggered, slices):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.decoder, scheduler.max_streams, scheduler.admit_per_round = d, 8, d.admit_per_round
    scheduler.waiting, scheduler.boxes, scheduler.deferred = queue.Queue(), {}, deque()
    scheduler.pending = scheduler.failed = None
    scheduler.state_lock = threading.Lock()
    scheduler.reply_prefill = False
    requests = [_stream([i + 1] * 27, 96, policy="fc7:0.3") for i in range(8)]

    def enqueue(items):
        for request in items:
            request.cancelled = lambda: False
            request.cofill = False  # Exercise individual prefill, with production admit_per_round=8.
            request.prefill_slice_layers = slices
            scheduler.waiting.put((request, queue.Queue()))

    enqueue(requests[:1] if staggered else requests)
    first_decode, completed = {}, {}
    max_admitted = max_decoding = 0
    for turn in range(450):
        if staggered and turn == 1:
            enqueue(requests[1:])
        scheduler._cancel_waiters()
        first, scheduler.pending = scheduler.pending, None
        if first is None and scheduler.deferred:
            first = scheduler.deferred.popleft()
        done = scheduler._admit(first)
        max_admitted = max(max_admitted, len(d.streams))
        done += d.prefill_step(queued=scheduler.pending is not None or bool(scheduler.deferred)
                              or not scheduler.waiting.empty())
        ready = [s.sid for s in d.streams.values() if not s.done and s.sid not in d.filling]
        max_decoding = max(max_decoding, len(ready))
        for sid in ready:
            first_decode.setdefault(sid, turn)
        done += d.round()
        completed.update((s.sid, turn) for s in done)
        d.finish(done)
        if not d.live() and not scheduler.deferred and scheduler.waiting.empty():
            break
    assert all(s.done and len(s.out) == 96 for s in requests)
    assert max_admitted == 8
    assert max_decoding == (1 if slices else 8)
    assert (max(first_decode.values()) < min(completed.values())) is (not bool(slices))
    return {s.sid: s.out for s in requests}


@pytest.mark.parametrize("fair", [False, True])
@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("slices", [None, 1], ids=["automatic", "forced-slice-negative-control"])
def test_eight_shorts_batch_with_same_tokens_and_rank_commands(wide_decoder, monkeypatch, fair, staggered, slices):
    mod, build = wide_decoder
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", str(int(fair)))
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    baseline = serve_eight(build(express=False), staggered, slices)
    d = build()
    observed = serve_eight(d, staggered, slices)
    assert observed == baseline
    assert all(result == observed for result in replay(mod, build, d.messages))


@pytest.mark.parametrize("tokens,requested,default,layers,atomic", [
    (32, None, 0, 0, 32), (33, None, 0, 1, 32), (513, None, 0, 1, 32),
    (27, 2, 0, 2, 32), (27, None, 3, 3, 32), (27, 0, 3, 0, 32),
    (33, None, 0, 0, 512), (512, None, 0, 0, 512), (513, None, 0, 1, 512), (130, 1, 0, 1, 512),
])
def test_atomic_limit_and_explicit_slices(wide_decoder, tokens, requested, default, layers, atomic):
    mod, build = wide_decoder
    d = build()
    d.express_atomic_rows = atomic          # 32 is the previous threshold; 512 is the TF_GLM_EXPRESS_ATOMIC_ROWS default
    d.prefill_slice_layers = default
    s = _stream([2] * tokens, 8)
    s.prefill_slice_layers = requested
    d.begin_admit(s)
    assert s.slice_layers == layers
    d.prefill_step()
    assert s.fill_layer == layers
    assert d.messages[-1][0] == (mod.FILL_SLICE if layers else mod.FILL)
    assert bool(s.out) is (layers == 0)


def _layer_stops(mod, d, sid):
    return [m[3] for m in d.messages if m[0] == mod.FILL_SLICE and m[1] == sid]


@pytest.mark.parametrize("budget,stops", [(0, list(range(1, 46))), (120, [15, 30, 45]), (30, [3 * i for i in range(1, 16)])])
def test_slice_budget_sets_layers_per_step_and_keeps_tokens(wide_decoder, budget, stops):
    mod, build = wide_decoder
    outputs = []
    for ms in (0, budget):
        d = build()
        d.prefill_slice_ms = ms
        s = _stream([(7 * i) % 60 for i in range(600)], 24, policy="fc7:0.3")
        d.begin_admit(s)
        assert s.slice_auto and s.slice_layers == 1 and not s.express     # 600 rows: main lane
        while d.live():
            d.finish(d.prefill_step() + d.round())
        outputs.append(s.out)
        if ms == budget:
            first = _layer_stops(mod, d, s.sid)
            assert first[:len(stops)] == stops                             # chunk 1 (512 rows)
            assert all(result == [s.out] for result in
                       [[o[s.sid]] for o in replay(mod, build, d.messages)])
    assert outputs[0] == outputs[1]


def test_slice_budget_never_overrides_explicit_slices(wide_decoder):
    mod, build = wide_decoder
    d = build()
    d.prefill_slice_ms = 500
    s = _stream([3] * 600, 8)
    s.prefill_slice_layers = 2
    d.begin_admit(s)
    assert not s.slice_auto
    d.prefill_step()
    assert d.messages[-1][:1] == [mod.FILL_SLICE] and d.messages[-1][3] == 2
