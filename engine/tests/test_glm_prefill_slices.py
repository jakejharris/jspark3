"""Sliced prefill protocol/ownership gates with CPU tensor state; GPU arithmetic has a separate TP3 gate."""

from types import SimpleNamespace
from unittest.mock import patch
from itertools import count

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.engine.exact_sampling import Sampling
from test_glm_batched_host import setup_decoder, _stream, _drain, _logits
from test_cuda_geometry import allocations  # fixture used by setup_decoder


@pytest.fixture
def sliced_decoder(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    from tensorfold.families.glm5_next.cuda import decode

    def stage(w, st, b, tokens):
        b.tokens = list(tokens)
        return len(tokens)

    def compute(w, st, b, n, *, layer_start=0, layer_stop=None, **kw):
        stop = len(w.layers) if layer_stop is None else layer_stop
        data = torch.tensor([[token, st.pos + i] for i, token in enumerate(b.tokens)], dtype=torch.float32)
        if layer_start == 0:
            b.x[:n].copy_(data)
        assert torch.equal(b.x[:n], data + layer_start), "another prefill overwrote a paused chunk"
        for layer in range(layer_start, stop):
            b.x[:n].add_(1)
            if layer in (0, 3):
                b.taps[(0, 3).index(layer)][:n].copy_(b.x[:n] - layer - 1)
            st.conv.fill_(st.pos + n)   # simulate early per-layer recurrent writes
        if stop < len(w.layers):
            return None
        b.fnormed[:n].copy_(data)
        st.kc[0][st.pos:st.pos + n].copy_(data)
        for ik, ig, pk in st.index:
            ik[st.pos:st.pos + n].copy_(data)
            ig[st.pos:st.pos + n].copy_(data)
            pk[:(st.pos + n) // 4 + 1].fill_(st.pos + n)
        return _logits(b.tokens[-1:], st.pos + n - 1)

    def commit(w, st, b, n, keep):
        st.set_pos(st.pos + keep)
        st.conv.fill_(st.pos)
        st.rec[st.cur[0]].fill_(st.pos)

    monkeypatch.setattr(decode, "stage", stage)
    monkeypatch.setattr(decode, "compute", compute)
    monkeypatch.setattr(decode, "chunks_for", lambda st, n: 1)
    monkeypatch.setattr(decode, "commit", commit)

    def build(slots=4):
        d = make(slots)
        d.w.layers = list(range(6))
        d.owner.e.prefill_rows = 8
        d.owner.e.pbuf = SimpleNamespace(x=torch.zeros(8, 2), fnormed=torch.zeros(8, 2), overlay=None,
                                         taps=[torch.zeros(8, 2), torch.zeros(8, 2)])
        original = d._engine

        def engine(*args):
            e = original(*args)
            e.reset = e.st.reset
            e.overlay = lambda first, n: (first, n)
            e.tap_rows = lambda n, b: torch.cat([t[:n] for t in b.taps], dim=1)
            e.sample = lambda logits, pos, sampling: decode.sample_rows(e.w, logits, pos, sampling)
            return e

        d._engine = engine
        return d

    return mod, build


def drain_steps(d):
    while d.live():
        done = d.prefill_step()
        done += d.round()
        d.finish(done)


def replay(mod, build, messages):
    observations = []
    for rank in (1, 2):
        d = build()
        # Followers intentionally have different defaults. Only rank-0 messages decide slices/depth.
        d.rank = d.owner.rank = rank
        d.prefill_slice_layers = 5
        d.fair_schedule = False
        original = d._finish
        observed = {}

        def finish(sids):
            observed.update({sid: list(d.streams[sid].out) for sid in sids})
            original(sids)

        d._finish = finish
        incoming = iter(messages + [[99]])
        d.share = lambda _: next(incoming)
        clock = count(1e9 if rank == 1 else -1e9, 17.0 if rank == 1 else .001)
        with patch.object(mod.time, "perf_counter", side_effect=lambda: next(clock)):
            with pytest.raises(RuntimeError, match="unknown GLM worker message"):
                d.follow()
        assert d.fill_owner is None and not d.filling and d.pool.available == d.capacity
        observations.append(observed)
    return observations


@pytest.mark.parametrize("layers", [1, 2, 4, 6, 9])
@pytest.mark.parametrize("sampling", [None, Sampling(123, .71, 16, .93)])
def test_slices_equal_serial_coserved_and_replayed_on_both_followers(sliced_decoder, layers, sampling):
    mod, build = sliced_decoder
    d = build()
    a = _stream([2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8], 25, sampling=sampling)
    b = _stream([5, 2], 23, sampling=sampling)
    expected = []
    for s in (a, b):
        plain = build()
        serial = _stream(s.prompt, s.count, draft=False, sampling=sampling)
        plain.admit(serial)
        _drain(plain)
        expected.append(serial.out)
    d.admit(b)
    a.prefill_slice_layers = layers
    d.begin_admit(a)
    drain_steps(d)
    assert [a.out, b.out] == expected
    assert all(observed == {a.sid: a.out, b.sid: b.out} for observed in replay(mod, build, d.messages))
    slices = [m for m in d.messages if len(m) == 4 and m[0] == mod.FILL_SLICE]
    assert slices and all(m[3] <= 6 for m in slices)
    assert torch.equal(d.cache[-1].rows[0], torch.tensor([[v, i] for i, v in enumerate(a.prompt)]))


def test_chunk_owner_blocks_short_prefill_but_allows_decode(sliced_decoder):
    mod, build = sliced_decoder
    d = build()
    live, long, short = _stream([2, 5], 30), _stream([1] * 17, 4), _stream([3], 4)
    d.admit(live)
    long.prefill_slice_layers = 2
    d.begin_admit(long)
    d.prefill_step()
    assert long.fill_layer == 2 and long.st.pos == long.fill_pos == 0
    assert long.drafter.context_end == 0
    d.begin_admit(short)
    for end in (4, 6):
        before = len(live.out)
        d.round()
        assert len(live.out) > before and not long.out and short.fill_pos == 0
        d.prefill_step()
        assert d.messages[-1] == [mod.FILL_SLICE, long.sid, 8, end]
    assert d.fill_owner is None and long.fill_pos == 8 and long.drafter.context_end == 8
    d.prefill_step()
    assert short.out       # ownership lasts one chunk, not the whole long prompt
    drain_steps(d)
    assert all(o == {s.sid: s.out for s in (live, long, short)} for o in replay(mod, build, d.messages))


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5])
def test_cancel_at_each_boundary_closes_taps_overlay_and_reuses_span(sliced_decoder, cut):
    mod, build = sliced_decoder
    d = build()
    a = _stream([1] * 12, 4)
    a.prefill_slice_layers = cut
    d.begin_admit(a)
    d.prefill_step()
    old_span = (a.start, a.span)
    assert d.owner.e.pbuf.overlay is not None and not d.cache
    a.cancelled = lambda: True
    done = d.prefill_step()
    assert done == [a] and d.owner.e.pbuf.overlay is None and a.drafter.context_end == 0
    d.finish(done)
    b = _stream([4] * 12, 4)
    b.prefill_slice_layers = 1
    d.begin_admit(b)
    assert (b.start, b.span) == old_span
    drain_steps(d)
    assert all(o == {a.sid: [], b.sid: b.out} for o in replay(mod, build, d.messages))


@pytest.mark.parametrize("cut", [1, 3, 5])
def test_resume_and_fresh_match_sliced_prompt_state_and_tokens(sliced_decoder, cut):
    _, build = sliced_decoder
    prompt = [2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8]
    d = build()
    prefix = _stream(prompt[:5], 1)
    prefix.prefill_slice_layers = cut
    d.begin_admit(prefix)
    drain_steps(d)
    resumed = _stream(prompt, 12, sampling=Sampling(123, .7, 16, .95))
    resumed.prefill_slice_layers = cut
    d.begin_admit(resumed)
    assert resumed.cached == 5
    drain_steps(d)
    fresh = build()
    full = _stream(prompt, 12, sampling=resumed.sampling)
    full.prefill_slice_layers = cut
    fresh.begin_admit(full)
    drain_steps(fresh)
    assert full.out == resumed.out
    a, b = d.cache[-1], fresh.cache[-1]
    for x, y in [(a.rec, b.rec), (a.conv, b.conv), *zip(a.rows, b.rows)]:
        assert torch.equal(x, y)
    assert a.drafter_end == b.drafter_end == len(prompt)
    assert a.drafter_rows is b.drafter_rows is None  # full-cache drafter tensors are included in rows above


@pytest.mark.parametrize("mutation", ["stream", "token_stop", "layer_stop", "legacy"])
def test_negative_control_rejects_divergent_slice_messages(sliced_decoder, mutation):
    _, build = sliced_decoder
    d = build()
    a, b = _stream([1] * 12, 4), _stream([2] * 12, 4)
    a.prefill_slice_layers = 2
    d.begin_admit(a)
    d.begin_admit(b)
    d.prefill_step()
    args = {"stream": (b.sid, 8, 4), "token_stop": (a.sid, 7, 4),
            "layer_stop": (a.sid, 8, 2), "legacy": (a.sid, 8, 0)}[mutation]
    with pytest.raises(RuntimeError):
        d._fill(*args)
    d.drop()


def test_negative_control_detects_paused_buffer_corruption(sliced_decoder):
    _, build = sliced_decoder
    d = build()
    a = _stream([1] * 12, 4)
    a.prefill_slice_layers = 2
    d.begin_admit(a)
    d.prefill_step()
    d.owner.e.pbuf.x.add_(1)
    with pytest.raises(AssertionError, match="overwrote"):
        d.prefill_step()
    assert d.broken is not None
    d.drop()


def test_default_off_and_request_override_leave_legacy_fill_messages(sliced_decoder, monkeypatch):
    mod, build = sliced_decoder
    monkeypatch.setenv("TF_GLM_PREFILL_SLICE_LAYERS", "2")
    d = build()
    s = _stream([1] * 12, 4)
    s.prefill_slice_layers = 0
    d.begin_admit(s)
    drain_steps(d)
    assert any(m[0] == mod.FILL for m in d.messages)
    assert not any(m[0] == mod.FILL_SLICE for m in d.messages)


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5])
def test_actual_forward_range_loop_preserves_overlay_taps_and_head(allocations, monkeypatch, cut):
    from tensorfold.families.glm5_next.cuda import forward as f

    w = SimpleNamespace(cfg=SimpleNamespace(hidden=2, streams=2, eps=1e-5), embed=None, norm=None, head=None,
                        layers=[SimpleNamespace(index=i) for i in range(6)])
    events = []

    def buffers(prefill=True, overlay=True):
        return SimpleNamespace(prefill=prefill, ids=torch.tensor([1, 2, 3]), x=torch.zeros(3, 4),
                               overlay=(torch.tensor([1]), torch.tensor([[41., 43.]])) if overlay else None,
                               hidden=torch.zeros(3, 2), fnormed=torch.zeros(3, 2), fxs=torch.zeros(3, 2),
                               logits=torch.zeros(3, 2), tap_at={0: [0], 2: [1], 5: [2]},
                               taps=[torch.zeros(3, 2) for _ in range(3)])

    def embed(ids, weights, hidden, streams, out):
        events.append("embed")
        out.copy_(ids[:, None].expand(-1, 4))

    def layer(layer, w, segs, b, n, *args):
        events.append((id(b), layer.index))
        b.x[:n].mul_(2).add_(layer.index)

    def mean(x, out):
        out.copy_(x.reshape(len(x), 2, 2).mean(1))

    def norm(x, weights, eps, out, xs):
        out.copy_(x / 2)

    def mm(b, x, weights, xs, out):
        events.append("head")
        out.copy_(x + 1)
        return out

    probe = SimpleNamespace(calls=None, begin_step=lambda *a: setattr(probe, "calls", []),
                            end_step=lambda: setattr(probe, "calls", None))
    monkeypatch.setattr(f.glue, "embed", embed)
    monkeypatch.setattr(f.glue, "stream_mean", mean)
    monkeypatch.setattr(f.glue, "rmsnorm", norm)
    monkeypatch.setattr(f, "layer_forward", layer)
    monkeypatch.setattr(f, "mm", mm)
    monkeypatch.setattr(f.tp3_probe, "recorder", probe)
    segs = [(SimpleNamespace(pos=17), 0, 3)]
    full, sliced = buffers(), buffers()
    expected = f.compute_streams(w, segs, full).clone()
    events.clear()
    assert f.compute_streams(w, segs, sliced, layer_start=0, layer_stop=cut) is None
    held = sliced.x.clone()
    assert probe.calls is None
    # Another stream's eager decode uses its own buffer, even in a lone nonzero slot.
    other = buffers(prefill=False, overlay=False)
    f.compute_streams(w, [(SimpleNamespace(pos=39), 0, 3)], other)
    assert torch.equal(held, sliced.x) and probe.calls is None
    got = f.compute_streams(w, segs, sliced, layer_start=cut, layer_stop=6)
    assert torch.equal(expected, got)
    assert all(torch.equal(a, b) for a, b in zip(full.taps, sliced.taps))
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == id(sliced)] == list(range(6))
    assert events.count("embed") == events.count("head") == 2  # one prefill, one decode
    # Negative control: replacing the live residual with a different prompt must change the result.
    damaged = buffers()
    f.compute_streams(w, segs, damaged, layer_start=0, layer_stop=cut)
    damaged.x.copy_(other.x)
    bad = f.compute_streams(w, segs, damaged, layer_start=cut, layer_stop=6)
    assert not torch.equal(expected, bad)


@pytest.mark.parametrize("failure", ["stage", "compute", "taps", "commit", "sample"])
def test_sliced_errors_clear_profile_and_continuation_before_release(sliced_decoder, monkeypatch, failure):
    _, build = sliced_decoder
    from tensorfold.families.glm5_next.cuda import decode, prof

    d = build()
    s = _stream([3] * 5, 4)
    s.prefill_slice_layers = 2
    d.begin_admit(s)

    def fail(*args, **kw):
        raise RuntimeError("injected failure")

    if failure in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, failure, fail)
    elif failure == "taps":
        s.drafter.add_taps = fail
    else:
        s.engine.sample = fail
    with pytest.raises(RuntimeError, match="injected failure"):
        while d.filling:
            d.prefill_step()
            assert not prof.active
    assert not prof.active and d.owner.e.pbuf.overlay is None
    assert not d.cache
    d.drop()
    assert d.fill_owner is None and not d.filling and d.pool.available == d.capacity


def test_incomplete_snapshot_is_rejected_and_mixed_parity_hash_fails(sliced_decoder):
    _, build = sliced_decoder
    from tensorfold.families.glm5_next.cuda.decode import take_snapshot
    import hashlib

    d = build()
    s = _stream([1] * 12, 4)
    s.prefill_slice_layers = 1
    d.begin_admit(s)
    d.prefill_step()
    with pytest.raises(RuntimeError, match="incomplete"):
        d._remember(s)
    assert not d.cache and s.st.pos == 0
    # Two recurrent layers have written different banks before the global chunk commit.
    st = SimpleNamespace(cur=[1, 0], rec=torch.tensor([[0., 10.], [20., 0.]]), conv=torch.tensor([20., 10.]),
                         mtp_len=0, mtp_drafted=0)
    mixed = take_snapshot(SimpleNamespace(st=st), [1], None, mtp=False)
    st.cur[1] = 1
    st.rec[1, 1] = 30
    st.conv[1] = 30
    complete = take_snapshot(SimpleNamespace(st=st), [1], None, mtp=False)
    state_hash = lambda snap: hashlib.sha256(snap.rec.numpy().tobytes() + snap.conv.numpy().tobytes()).hexdigest()
    assert state_hash(mixed) != state_hash(complete)  # the forbidden snapshot cannot pass a state gate
    d.drop()


def test_profile_is_disabled_at_every_layer_and_chunk_yield(sliced_decoder):
    _, build = sliced_decoder
    from tensorfold.families.glm5_next.cuda import prof

    d = build()
    s = _stream([2] * 11, 4)
    s.prefill_slice_layers = 1
    d.begin_admit(s)
    while d.filling:
        d.prefill_step()
        assert not prof.active
    _drain(d)


def test_partial_stream_and_synchronous_admission_cannot_reenter_gpu_work(sliced_decoder):
    _, build = sliced_decoder
    d = build()
    s = _stream([1] * 12, 4)
    s.prefill_slice_layers = 1
    d.begin_admit(s)
    d.prefill_step()
    before = len(d.messages)
    with pytest.raises(ValueError, match="synchronous admission"):
        d.admit(_stream([3], 2))
    assert len(d.messages) == before and d.live() == 1
    with pytest.raises(RuntimeError, match="partially prefilled"):
        d._round([s])
    d.drop()


@pytest.mark.parametrize("bad_stop", [0, -1, 7, 2])
def test_bad_follower_endpoint_poisons_and_releases_continuation(sliced_decoder, bad_stop):
    mod, build = sliced_decoder
    leader = build()
    s = _stream([3] * 12, 4)
    s.prefill_slice_layers = 2
    leader.begin_admit(s)
    leader.prefill_step()
    messages = iter(leader.messages + [[mod.FILL_SLICE, s.sid, 8, bad_stop]])
    follower = build()
    follower.rank = follower.owner.rank = 1
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="layer stop"):
        follower.follow()
    assert follower.broken is not None
    follower.drop()
    assert follower.fill_owner is None and follower.pool.available == follower.capacity
    leader.drop()


def test_slice_announcement_failure_poisons_before_more_work(sliced_decoder):
    _, build = sliced_decoder
    d = build()
    s = _stream([2] * 11, 4)
    s.prefill_slice_layers = 2
    d.begin_admit(s)

    def fail(_):
        raise RuntimeError("control transport failed")

    d.share = fail
    with pytest.raises(RuntimeError, match="control transport failed"):
        d.prefill_step()
    assert d.broken is not None and s.fill_pos == 0
    d.drop()
    assert d.pool.available == d.capacity
