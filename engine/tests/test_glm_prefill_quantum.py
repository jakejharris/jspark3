"""Sliced prefill turn budgets preserve target tokens and rank-zero protocol decisions."""
import runpy

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream, _drain  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps, replay  # noqa: F401
from test_glm_reply_reservations import enable


def test_quantum_default_off_and_bounded(monkeypatch):
    monkeypatch.delenv("TF_GLM_PREFILL_DECODE_QUANTUM", raising=False)
    assert runpy.run_path(p1.__file__)["PREFILL_DECODE_QUANTUM"] == 0
    for value in ("1", "2", "3", "4"):
        monkeypatch.setenv("TF_GLM_PREFILL_DECODE_QUANTUM", value)
        assert runpy.run_path(p1.__file__)["PREFILL_DECODE_QUANTUM"] == int(value)
    for value in ("-1", "5", "yes", "1.5"):
        monkeypatch.setenv("TF_GLM_PREFILL_DECODE_QUANTUM", value)
        with pytest.raises(ValueError):
            runpy.run_path(p1.__file__)


@pytest.mark.parametrize("stride", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("sampling", [None, Sampling(123, .71, 16, .93)])
def test_turn_budget_equals_serial_and_rank_replay(sliced_decoder, monkeypatch, stride, sampling):
    mod, build = sliced_decoder
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", stride)
    d = build()
    incumbent = _stream([2, 5], 90, sampling=sampling)
    newcomer = _stream([1] * 33, 15, sampling=sampling)
    newcomer.prefill_slice_layers = 2
    expected = []
    for s in (incumbent, newcomer):
        plain = build()
        serial = _stream(s.prompt, s.count, draft=False, sampling=sampling)
        plain.admit(serial)
        _drain(plain)
        expected.append(serial.out)
    d.admit(incumbent)
    d.begin_admit(newcomer)
    services = 0
    steps = 0
    proposals = len(incumbent.drafter.proposals)
    while newcomer.sid in d.filling:
        done = d.prefill_step()
        active = newcomer.sid in d.filling
        before = len(incumbent.out)
        done += d.round()
        if active:
            steps += 1
            if stride:
                serviced = (steps - 1) % stride == 0
                assert len(incumbent.out) - before == int(serviced)
                assert len(incumbent.drafter.proposals) == proposals
                assert incumbent.drafter.context_end == incumbent.st.pos
                services += serviced
        d.finish(done)
    drain_steps(d)
    assert [incumbent.out, newcomer.out] == expected
    assert all(observed == {incumbent.sid: incumbent.out, newcomer.sid: newcomer.out}
               for observed in replay(mod, build, d.messages))
    frames = [m for m in d.messages if m[0] == mod.ROUND_QUANTUM]
    if stride:
        assert len(frames) == services > 0
        assert incumbent.stats()["prefill_quantum_rounds"] == services
        assert getattr(incumbent, "prefill_quantum_skips", 0) == steps - services
        assert len(incumbent.drafter.proposals) > proposals  # drafting resumes
    else:
        assert not frames and "prefill_quantum_rounds" not in incumbent.stats()


@pytest.mark.parametrize("cancel_filling", [False, True])
def test_cancel_on_skipped_turn_releases_and_resumes_normal_service(sliced_decoder, monkeypatch, cancel_filling):
    mod, build = sliced_decoder
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", 3)
    d = build()
    incumbent, newcomer = _stream([2, 5], 90), _stream([1] * 33, 15)
    newcomer.prefill_slice_layers = 1
    d.admit(incumbent)
    d.begin_admit(newcomer)
    d.prefill_step()
    d.round()
    assert incumbent.prefill_quantum_rounds == 1
    victim, survivor = (newcomer, incumbent) if cancel_filling else (incumbent, newcomer)
    victim.cancelled = lambda: True
    d.finish(d.prefill_step() + d.round())
    assert victim.sid not in d.streams
    drain_steps(d)
    assert len(survivor.out) == survivor.count and len(victim.out) < victim.count
    assert d.pool.available == d.capacity
    assert all(observed == {incumbent.sid: incumbent.out, newcomer.sid: newcomer.out}
               for observed in replay(mod, build, d.messages))


@pytest.mark.parametrize("field", ["use_mtp", "copy_enabled", "priced"])
def test_ineligible_quantum_rejected_before_forward(sliced_decoder, monkeypatch, field):
    _, build = sliced_decoder
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", 2)
    d = build()
    incumbent = _stream([2, 5], 90)
    d.admit(incumbent)
    setattr(incumbent, field, True)
    before = incumbent.st.pos
    with pytest.raises(RuntimeError, match="ineligible policy"):
        d._round([incumbent], quantum=True)
    assert incumbent.st.pos == before


def test_quantum_frame_rejected_with_flag_off(sliced_decoder, monkeypatch):
    mod, build = sliced_decoder
    d = build()
    s = _stream([2, 5], 20)
    d.admit(s)
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", 0)
    incoming = iter(d.messages + [[mod.ROUND_QUANTUM, 1, s.sid]])
    follower = build()
    follower.rank = follower.owner.rank = 1
    follower.share = lambda _: next(incoming)
    with pytest.raises(RuntimeError, match="disabled flag"):
        follower.follow()


def test_reservation_pressure_preserves_quantum_taps_and_full_responses(sliced_decoder, monkeypatch, tmp_path):
    _, build = sliced_decoder
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", 2)
    d = build()
    parent = d.owner.drafter
    parent.ring, parent.window, parent.cap = 512, 448, 4 * 512
    parent.kc = [torch.zeros(2, parent.cap, 2)]
    parent.vc = [torch.zeros(2, parent.cap, 2)]
    grow = enable(d, tmp_path, capacity=248)
    incumbent, newcomer = _stream([2, 5], 200), _stream([1] * 49, 180)
    newcomer.prefill_slice_layers = 1
    d.admit(incumbent)
    while incumbent.st.pos < 45:
        d.round()
    d.begin_admit(newcomer)
    drain_steps(d)
    assert incumbent.prefill_quantum_rounds > 0
    assert incumbent.reply_grows and newcomer.reply_parks
    assert not grow.parked and not grow.store.store.live_credits
    for s in (incumbent, newcomer):
        reference = build()
        serial = _stream(s.prompt, s.count, draft=False)
        reference.admit(serial)
        _drain(reference)
        assert s.out == serial.out and len(s.out) == s.count


@pytest.mark.parametrize("field", ["use_mtp", "copy_enabled", "priced"])
def test_mixed_policy_batch_keeps_ordinary_service(sliced_decoder, monkeypatch, field):
    mod, build = sliced_decoder
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", 3)
    d = build()
    a, b, newcomer = _stream([2], 50), _stream([3], 50), _stream([1] * 17, 10)
    newcomer.prefill_slice_layers = 1
    d.admit(a)
    d.admit(b)
    d.begin_admit(newcomer)
    setattr(b, field, True)
    d._priced_depths = lambda live: ([len(s.drafts) for s in live], [mod.SRC_N, mod.SRC_D])
    seen = []
    d._round = lambda live: seen.append([s.sid for s in live])
    for _ in range(2):
        d.prefill_step()
        d.round()
    assert seen == [[a.sid, b.sid]] * 2
    assert not any(m[0] == mod.ROUND_QUANTUM for m in d.messages)


@pytest.mark.parametrize("layers", [0, 2])
def test_saved_prefix_refreshes_pool_row_when_it_becomes_committed(sliced_decoder, layers):
    """Both flags-off and sliced prefill transfers repair a newly committed boundary."""
    _, build = sliced_decoder
    d = build()
    parent = d.owner.drafter
    parent.ring, parent.window, parent.cap = 512, 448, 4 * 512
    parent.kc = [torch.zeros(2, parent.cap, 2)]
    parent.vc = [torch.zeros(2, parent.cap, 2)]
    prefix = _stream([1] * 5, 1)
    prefix.prefill_slice_layers = layers
    d.begin_admit(prefix)
    drain_steps(d)
    old = d.cache[-1]
    old_pool = old.rows[3]
    assert len(old_pool) == 2  # one committed pool and the future fence
    old_pool[1].fill_(999)
    appended = _stream([1] * 5 + [2] * 8, 1)
    appended.prefill_slice_layers = layers
    d.begin_admit(appended)
    assert appended.cached == 5
    drain_steps(d)
    saved = d.cache[-1]
    row = saved.rows[3][0][1]
    assert not bool((row == 999).any())
    assert torch.equal(row, d.owner.e.st.index[0][2][1])
