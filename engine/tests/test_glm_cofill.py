"""Co-prefill host gates: appends alone/co-served, state ownership, rank traces and controls."""

import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _Buffers, _drain, _stream  # noqa: F401


@pytest.fixture
def cofill_decoder(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    co = mod.cofill
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    baseline = mod.prefill
    forwards = []

    def steps(e, prompt, sampling, *, mtp, drafter, resume, rows, slice_end, **checkpoints):
        # Test appends fit one ordinary chunk. Keep construction lazy, as the
        # production stepper is: no shared scratch touched until next().
        if False:
            yield
        return baseline(e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume)

    def compute(w, segs, b, **kwargs):
        forwards.append([(st.pos, a1 - a0) for st, a0, a1 in segs])
        logits = mod.compute_streams(w, segs, b, **kwargs)
        for st, a0, a1 in segs:
            stop = st.pos + a1 - a0
            for ik, ig, pk in st.index:
                ik[st.pos:stop].copy_(st.kc[0][st.pos:stop])
                ig[st.pos:stop].copy_(st.kc[0][st.pos:stop])
                pk[:stop // 4 + 1].fill_(stop)
        return torch.cat([logits[a1 - 1:a1] for _, _, a1 in segs])

    def commit(w, st, b, rows, keep):
        assert rows == keep
        st.set_pos(st.pos + keep)
        st.conv.fill_(st.pos)
        st.rec[st.cur[0]].fill_(st.pos)

    monkeypatch.setattr(mod, "prefill_steps", steps)
    monkeypatch.setattr(co, "stage_streams", mod.stage_streams)
    monkeypatch.setattr(co, "compute_streams", compute)
    monkeypatch.setattr(co, "commit", commit)

    def build(slots=4, *, rows=128, limit=512, pool_limit=2048):
        d = make(slots, limit=limit, pool_limit=pool_limit)
        b = _Buffers(d.w, rows, d.capacity)
        b.set_taps((0, 1), 2)
        b.prefill, b.ka = True, torch.empty(rows, 2)
        d.owner.e.pbuf, d.owner.e.prefill_rows = b, rows
        original = d._engine

        def engine(*args):
            e = original(*args)
            e.reset = e.st.reset
            return e

        d._engine = engine
        return d

    return mod, build, forwards


def state_hash(s):
    arrays = [s.st.kc[0][:s.st.pos], s.st.rec[s.st.cur[0]], s.st.conv]
    arrays += [t[:s.st.pos] for t in s.st.index[0][:2]]
    return hashlib.sha256(b"".join(t.contiguous().numpy().tobytes() for t in arrays)).hexdigest()


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("slots", [2, 4, 8])
def test_cold_and_cached_appends_equal_alone_including_state_and_taps(cofill_decoder, slots, sampled):
    mod, make, forwards = cofill_decoder
    cases = []
    for i in range(slots):
        sampling = Sampling(2**64 - i - 1, .73, 20, .92) if sampled else None
        cases.append(([3 + i] * (8 + i), [11, 12, 13][:1 + i % 3], sampling))
    for cached in (False, True):
        refs = []
        for prefix, suffix, sampling in cases:
            d = make(slots)
            if cached:
                warm = _stream(prefix, 1)
                d.admit(warm)
                d.finish([warm])
            s = _stream(prefix + suffix, 12, sampling=sampling)
            s.cofill = False
            d.begin_admit(s)
            d.prefill_step()
            refs.append((s.out[:], state_hash(s), s.drafter.context_end))
            _drain(d)
            refs[-1] += (s.out[:],)
        d = make(slots)
        if cached:
            for prefix, _, _ in cases:
                warm = _stream(prefix, 1)
                d.admit(warm)
                d.finish([warm])
        streams = []
        for prefix, suffix, sampling in cases:
            s = _stream(prefix + suffix, 12, sampling=sampling)
            d.begin_admit(s)
            streams.append(s)
        before = len(forwards)
        d.prefill_step()
        assert len(forwards) == before + 1 and len(forwards[-1]) == slots
        assert not d.filling
        assert [(s.out, state_hash(s), s.drafter.context_end) for s in streams] == [r[:3] for r in refs]
        assert all(s.cached == (len(cases[i][0]) if cached else 0) for i, s in enumerate(streams))
        _drain(d)
        assert [s.out for s in streams] == [r[3] for r in refs]
        assert all(s.cofill_stats["streams"] == slots for s in streams)


def test_cofill_followers_replay_frozen_participants(cofill_decoder):
    mod, make, _ = cofill_decoder
    lead = make(2)
    streams = [_stream([3] * 9, 16, sampling=Sampling(42, .73, 20, .92)), _stream([7] * 13, 17)]
    for s in streams:
        lead.begin_admit(s)
    lead.prefill_step()
    _drain(lead)
    assert any(m[0] == mod.cofill.FILL_COFILL for m in lead.messages)
    for rank in (1, 2):
        follower = make(2)
        follower.rank = follower.owner.rank = rank
        outputs, finish = {}, follower._finish

        def record(sids):
            outputs.update({sid: list(follower.streams[sid].out) for sid in sids})
            finish(sids)

        follower._finish = record
        messages = iter(lead.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == {s.sid: s.out for s in streams}


@pytest.mark.parametrize("owner", ["fill_owner", "express_owner"])
def test_cofill_never_overwrites_a_suspended_chunk_and_cancel_reuses_span(cofill_decoder, owner):
    mod, make, forwards = cofill_decoder
    d = make(4)
    streams = [_stream([i + 1] * 12, 5) for i in range(3)]
    for s in streams:
        d.begin_admit(s)
    setattr(d, owner, streams[0].sid)
    assert mod.cofill.candidates(d, streams[0]) == []
    with pytest.raises(RuntimeError, match="suspended"):
        mod.cofill.run(d, [streams[1].sid, streams[2].sid])
    setattr(d, owner, None)
    streams[0].cancelled = lambda: True
    done = d.prefill_step()
    assert streams[0] in done and not streams[0].out
    assert len(forwards[-1]) == 2
    d.finish(done)
    fresh = _stream([9] * 12, 5)
    d.begin_admit(fresh)
    assert fresh.slot == 0
    d.prefill_step()
    _drain(d)
    assert d.pool.available == d.capacity


def test_sliced_requests_keep_m1_instead_of_atomic_cofill(cofill_decoder):
    mod, make, _ = cofill_decoder
    d = make(2)
    streams = [_stream([3] * 9, 8), _stream([7] * 13, 8)]
    for s in streams:
        s.prefill_slice_layers = 1
        d.begin_admit(s)
    assert all(not s.cofill_ready for s in streams)
    assert mod.cofill.candidates(d, streams[0]) == []
    assert all(m[0] != mod.cofill.ADMIT_COFILL for m in d.messages)


def test_cofill_marks_prefill_phase_and_invalidates_round_price(cofill_decoder, monkeypatch):
    from contextlib import contextmanager

    _, make, _ = cofill_decoder
    d, phases = make(2), []

    @contextmanager
    def phase(name, *, stream_id=None):
        phases.append((name, stream_id))
        yield

    monkeypatch.setattr(d.observer, "phase", phase)
    for token in (3, 7):
        d.begin_admit(_stream([token] * 12, 8))
    d.price_rounds.pending = {"rows": 2}
    d.prefill_step()
    assert phases == [("prefill", None)] and d.price_rounds.pending is None


@pytest.mark.parametrize("inner_mark", [False, True])
def test_cofill_preserves_session_completion_and_internal_checkpoint_boundaries(cofill_decoder, inner_mark):
    mod, make, _ = cofill_decoder
    d, completed = make(2), []
    def complete(s, snapshot):
        assert snapshot is d.cache[-1]  # Session persistence reuses the memory snapshot for async persistence
        completed.append((s.sid, s.st.pos, s.drafter.context_end))
    d.sessions = SimpleNamespace(
        store=None, saved=lambda *args, **kwargs: None, begin=lambda *args, **kwargs: None,
        extra_rows=lambda cached: 0, completed=complete,
        prefill_kwargs=lambda s: {"mark": lambda pos: pos + 5, "keep": lambda snap: None} if inner_mark else {})
    streams = [_stream([token] * 12, 8) for token in (3, 7)]
    for s in streams:
        d.begin_admit(s)
    if inner_mark:
        assert all(not s.cofill_ready for s in streams)
        assert mod.cofill.candidates(d, streams[0]) == [] and completed == []
    else:
        d.prefill_step()
        assert completed == [(s.sid, len(s.prompt), len(s.prompt)) for s in streams]


def test_wrong_output_head_negative_control_breaks_alone_cofilled_gate(cofill_decoder, monkeypatch):
    mod, make, _ = cofill_decoder
    d = make(2)
    original = mod.cofill.compute_streams

    def wrong_head(*args, **kwargs):
        logits = original(*args, **kwargs)
        return logits[-1:].expand_as(logits).clone()  # give both requests the last stream's head

    monkeypatch.setattr(mod.cofill, "compute_streams", wrong_head)
    a, b = _stream([1] * 8, 1), _stream([9] * 12, 1)
    d.begin_admit(a)
    d.begin_admit(b)
    d.prefill_step()
    reference = make(2)
    alone = _stream(a.prompt, 1)
    reference.admit(alone)
    with pytest.raises(AssertionError):
        assert a.out == alone.out


def test_cofill_real_kda_branch_keeps_per_stream_prefill_state(setup_decoder, monkeypatch):
    from tensorfold.families.glm5_next.cuda import forward as f

    w = SimpleNamespace(cfg=SimpleNamespace(eps=1e-5, lower=0))
    k = SimpleNamespace(proj="proj", fb="fb", gb="gb", o="out", fa_off=0, ga_off=128, b_off=0,
                        conv=None, a_log=None, dt_bias=None, norm=None)
    layer = SimpleNamespace(index=0, kda=k)

    def buffers(rows):
        return SimpleNamespace(prefill=True, kproj=torch.zeros(1, rows, 256),
                               normed=torch.arange(rows * 256).reshape(rows, 256).float(), xs=torch.zeros(rows, 4),
                               ka=torch.zeros(rows, 128), kg=torch.zeros(rows, 128), kscratch=object(),
                               cofill_kout=torch.zeros(rows, 128))

    def state(value):
        return SimpleNamespace(kda_index={0: 0}, cur=[0], conv=torch.full((1, 3, 128), value),
                               rec=torch.full((2, 1, 128), value), proj=torch.empty(1, 8, 256))

    def mm(b, x, weight, xs, out):
        out.copy_(x)
        return out

    def chain(p, b_off, a, g, conv, kernel, rec, log, bias, norm, eps, lower, n, scratch, next_rec):
        out = p[:, :128] + rec
        next_rec.copy_(out[-1])
        return out

    def shift(conv, proj, n):
        conv.copy_(proj[:, n - 3:n, :128])

    monkeypatch.setattr(f, "mm", mm)
    monkeypatch.setattr(f.kda_mod, "chain", chain)
    monkeypatch.setattr(f, "_shift_conv", shift)
    monkeypatch.setattr(f, "out_proj", lambda w, b, out, *args: out.clone())
    n1, n2 = 11, 13  # both exceed State.proj's eight decode rows
    b, a, c = buffers(n1 + n2), state(2.), state(5.)
    actual = f.kda_block(layer, w, [(a, 0, n1), (c, n1, n1 + n2)], b, n1 + n2)
    pieces = []
    for start, n, value, got in ((0, n1, 2., a), (n1, n2, 5., c)):
        single, st = buffers(n), state(value)
        single.normed.copy_(b.normed[start:start + n])
        pieces.append(f.kda_block(layer, w, [(st, 0, n)], single, n))
        assert st.cur == got.cur and torch.equal(st.rec, got.rec) and torch.equal(st.conv, got.conv)
    assert torch.equal(actual, torch.cat(pieces))


def test_cofill_real_forward_uses_each_streams_last_row_and_one_row_head(setup_decoder, monkeypatch):
    from tensorfold.families.glm5_next.cuda import forward as f

    n = 7
    b = SimpleNamespace(ids=torch.arange(n), x=torch.zeros(n, 1), hidden=torch.empty(n, 1),
                        fnormed=torch.empty(n, 1), fxs=torch.empty(n, 1), logits=torch.empty(1, 1),
                        overlay=None, prefill=True)
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=1, streams=1, eps=1e-5), layers=[], embed=None,
                        norm=None, head=None)
    monkeypatch.setattr(f.glue, "embed", lambda ids, embed, hidden, streams, out: out.copy_(ids[:, None]))
    monkeypatch.setattr(f.glue, "stream_mean", lambda x, out: out.copy_(x))
    monkeypatch.setattr(f.glue, "rmsnorm", lambda x, norm, eps, out, xs: out.copy_(x))
    sizes = []

    def head(buf, x, weight, xs, out):
        sizes.append(len(x))
        out.copy_(x + 100)
        return out

    monkeypatch.setattr(f, "mm", head)
    got = f.compute_streams(w, [(None, 0, 3), (None, 3, 7)], b)
    assert got.tolist() == [[102.], [106.]] and sizes == [1, 1]
