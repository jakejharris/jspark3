"""Reply prefill gates use the real prefill iterator/protocol with CPU arithmetic stand-ins."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from test_glm_a2_prefill import express_decoder  # noqa: F401


def enable(build):
    d = build()
    d.reply_prefill, d.fair_schedule, d.reply_prefill_rows = True, True, 3
    d.owner._gather_ints = lambda values: [values] * 3
    return d


def background(prompt, base):
    s = _stream(prompt, 1)
    s.reply_prefill, s.reply_base = True, base
    s.priority, s.prefill_slice_layers = "background", 1
    return s


def warm(d, prompt):
    s = _stream(prompt, 1)
    d.begin_admit(s)
    drain_steps(d)
    return s


@pytest.mark.parametrize("sampling", [None, Sampling(301, .7, 16, .93)])
@pytest.mark.parametrize("policy", ["0", "fc7:0.3"])
def test_completed_reply_is_prefill_only_and_resumed_equals_fresh(sliced_decoder, sampling, policy, capsys):
    mod, build = sliced_decoder
    prompt = [2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8, 7, 2, 6, 1, 5]
    d = enable(build)
    warm(d, prompt[:5])
    job = background(prompt[:13], 5)
    job.policy = policy
    d.begin_admit(job)
    # Neither keyed sampling nor drafting is needed to retain a prompt anchor.
    original = job.engine.sample
    job.engine.sample = lambda *a: pytest.fail("idle prefill sampled an output")
    drain_steps(d)
    job.engine.sample = original
    assert not job.out and job.rounds == 0 and d.pool.available == d.capacity
    assert d.cache[-1].ids == job.prompt and d.cache[-1].reply_base == 5
    assert '"status": "completed"' in capsys.readouterr().out
    actual = _stream(prompt, 12, sampling=sampling, policy=policy)
    d.begin_admit(actual)
    assert actual.cached == 13 and actual.reply_prefill_hit_tokens == 8
    drain_steps(d)
    fresh = enable(build)
    reference = _stream(prompt, 12, sampling=sampling, policy=policy)
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert actual.out == reference.out
    a, b = d.cache[-1], fresh.cache[-1]
    assert all(torch.equal(x, y) for x, y in [(a.rec, b.rec), (a.conv, b.conv), *zip(a.rows, b.rows)])
    # Both followers replay the named idle work without an HTTP request or local policy.
    for rank in (1, 2):
        follower = enable(build)
        follower.rank = follower.owner.rank = rank
        messages = iter(d.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert follower.pool.available == follower.capacity and not follower.filling
        assert all(torch.equal(x, y) for x, y in zip(a.rows, follower.cache[-1].rows))
    frames = [m for m in d.messages if m[0] == mod.ADMIT_REPLY]
    assert len(frames) == 1 and frames[0][-1] == 5


@pytest.mark.parametrize("boundary", range(1, 13))
def test_cancel_every_layer_and_chunk_releases_before_foreground(sliced_decoder, boundary):
    _, build = sliced_decoder
    d = enable(build)
    prompt = [2, 4, 5, 6, 3, 7, 1, 9, 3, 4, 1, 2, 8]
    warm(d, prompt[:3])
    job = background(prompt, 3)
    d.begin_admit(job)
    for _ in range(boundary):
        assert not d.prefill_step()
    job.cancelled = lambda: True
    d.finish(d.prefill_step())
    assert d.fill_owner is None and not d.filling and d.pool.available == d.capacity
    assert all(c.ids != prompt for c in d.cache)  # no partial speculative checkpoint
    assert d.owner.e.pbuf.overlay is None
    actual = _stream([2, 4, 5, 8, 9, 2], 11)
    d.begin_admit(actual)
    drain_steps(d)
    fresh = enable(build)
    reference = _stream(actual.prompt, actual.count)
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert actual.out == reference.out


def test_rank_prefix_miss_skips_without_disk_or_cold_prefill(sliced_decoder):
    mod, build = sliced_decoder
    d = enable(build)
    warm(d, [2, 3, 4])
    d.sessions = SimpleNamespace(store=None, saved=lambda *a, **k: pytest.fail("Reply prefill read disk"),
                                 finish=lambda s, broken: None)
    d.owner._gather_ints = lambda values: [[1], [0], [1]]
    job = background([2, 3, 4, 5], 3)
    d.begin_admit(job)
    assert job.done and not d.filling and job.cached == 0
    d.finish([job])
    assert d.pool.available == d.capacity and not job.out
    assert not any(m[0] in (mod.FILL, mod.FILL_SLICE) and m[1] == job.sid for m in d.messages)


def test_no_memory_prefix_is_rejected_before_protocol(sliced_decoder):
    _, build = sliced_decoder
    d = enable(build)
    with pytest.raises(ValueError, match="completed prompt in memory"):
        d.begin_admit(background([2, 3, 4], 2))
    assert not d.messages and not d.broken and d.pool.available == d.capacity


def test_wrong_prediction_is_a_normal_prefix_miss(sliced_decoder):
    _, build = sliced_decoder
    d = enable(build)
    warm(d, [2, 3, 4])
    d.begin_admit(background([2, 3, 4, 5, 6], 3))
    drain_steps(d)
    actual = _stream([2, 3, 4, 9, 6, 8], 9)
    d.begin_admit(actual)
    assert actual.cached <= 3 and actual.reply_prefill_hit_tokens == 0
    drain_steps(d)
    fresh = enable(build)
    reference = _stream(actual.prompt, actual.count)
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert actual.out == reference.out


def test_reply_never_joins_copy_or_cofill_or_session_disk(sliced_decoder, monkeypatch):
    mod, build = sliced_decoder
    d = enable(build)
    warm(d, [2, 3, 4])
    monkeypatch.setenv("TF_GLM_COPY_DRAFTS", "1")
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    d.sessions = SimpleNamespace(store=None, saved=lambda *a, **k: pytest.fail("disk read"),
                                 begin=lambda *a, **k: pytest.fail("session begin"),
                                 extra_rows=lambda cached: 0,
                                 finish=lambda s, broken: None,
                                 completed=lambda *a: pytest.fail("session snapshot"))
    job = background([2, 3, 4, 5, 6], 3)
    d.begin_admit(job)
    assert not job.copy_enabled and not job.cofill_ready
    drain_steps(d)
    assert d.messages[-1][0] == mod.DONE and d.pool.available == d.capacity


@pytest.mark.parametrize("mutation", ["base", "layout", "flag"])
def test_malformed_reply_admission_fails_closed(sliced_decoder, mutation):
    mod, build = sliced_decoder
    leader = enable(build)
    warm(leader, [2, 3, 4])
    leader.begin_admit(background([2, 3, 4, 5], 3))
    frames = [list(m) for m in leader.messages]
    header = next(m for m in frames if m[0] == mod.ADMIT_REPLY)
    if mutation == "base":
        header[-1] = 999
    elif mutation == "layout":
        header.append(7)
    follower = enable(build)
    follower.rank = follower.owner.rank = 1
    follower.reply_prefill = mutation != "flag"
    incoming = iter(frames)
    follower.share = lambda _: next(incoming)
    with pytest.raises(RuntimeError, match="admission"):
        follower.follow()
    follower.drop()
    leader.drop()
    assert follower.pool.available == follower.capacity


@pytest.mark.parametrize("copy_rows", [16, 32])
@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("suffix", [3, 7], ids=["express", "main"])
def test_reply_respects_both_lanes_tiles_and_wide_copy_geometry(
        express_decoder, monkeypatch, copy_rows, tiles, suffix):
    mod, build = express_decoder
    monkeypatch.setenv("TF_GLM_COPY_WIDE_ROWS", str(copy_rows))
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    monkeypatch.setattr(mod.p1, "ATTENTION_TILES", tiles)

    def configured():
        d = enable(build)
        d.reply_prefill_rows = 2
        return d

    d = configured()
    prompt = [2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8, 7]
    warm(d, prompt[:5])
    main = d.owner.e.pbuf
    job = background(prompt[:5 + suffix], 5)
    d.begin_admit(job)
    assert d.max_rows == d.guard == copy_rows
    assert job.express is (suffix <= d.express_buf.rows)
    assert not job.copy_enabled and not job.cofill_ready
    assert job.engine.pbuf is (d.express_buf if job.express else main)
    untouched = main if job.express else d.express_buf
    before = untouched.x.clone()
    while d.filling:
        assert not job.out
        d.prefill_step()
        assert job.fill_stop - job.fill_pos <= 2
        assert torch.equal(untouched.x, before)
        assert (d.fill_owner if job.express else d.express_owner) is None
    d.finish([job])
    assert job.out == [] and d.fill_owner is d.express_owner is None
    assert d.owner.e.pbuf is main and d.pool.available == d.capacity
    actual = _stream(job.prompt + [3, 6], 12, sampling=Sampling(307, .7, 16, .93))
    d.begin_admit(actual)
    assert actual.cached == len(job.prompt) and actual.reply_prefill_hit_tokens == suffix
    drain_steps(d)
    fresh = configured()
    reference = _stream(actual.prompt, actual.count, sampling=actual.sampling)
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert actual.out == reference.out
    a, b = d.cache[-1], fresh.cache[-1]
    assert all(torch.equal(x, y) for x, y in [(a.rec, b.rec), (a.conv, b.conv), *zip(a.rows, b.rows)])
    from test_glm_prefill_slices import replay
    assert all(o[job.sid] == [] and o[actual.sid] == actual.out for o in replay(mod, configured, d.messages))
    assert not any(m[0] == mod.cofill.FILL_COFILL for m in d.messages)


@pytest.mark.parametrize("suffix", [3, 7], ids=["express", "main"])
@pytest.mark.parametrize("boundary", [1, 5, 7, 11])
def test_preempt_reply_releases_selected_lane_before_foreground(
        express_decoder, monkeypatch, suffix, boundary):
    mod, build = express_decoder
    monkeypatch.setenv("TF_GLM_COPY_WIDE_ROWS", "32")
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    d = enable(build)
    d.reply_prefill_rows = 2
    prefix = [2, 3, 4, 8, 9]
    warm(d, prefix)
    job = background(prefix + [5] * suffix, len(prefix))
    d.begin_admit(job)
    for _ in range(boundary):
        assert not d.prefill_step()
    assert (d.express_owner if job.express else d.fill_owner) == job.sid
    assert mod.cofill.candidates(d, job) == []
    # Use the actual scheduler preemption path, which must close either lane.
    from tensorfold.cuda.scheduler import Scheduler
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.decoder, scheduler.reply_active = d, job
    job.cancelled = lambda: True
    scheduler._preempt_reply_prefill()
    assert scheduler.reply_active is None and not d.filling
    assert d.express_owner is d.fill_owner is None
    assert job.engine.pbuf.overlay is None and d.pool.available == d.capacity
    assert not any(c.ids == job.prompt for c in d.cache)
    foreground = _stream(prefix + [7, 8], 9)
    d.begin_admit(foreground)
    assert foreground.express and foreground.start == job.start
    drain_steps(d)
    fresh = enable(build)
    reference = _stream(foreground.prompt, foreground.count)
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert foreground.out == reference.out
    from test_glm_prefill_slices import replay
    assert all(o[job.sid] == [] and o[foreground.sid] == foreground.out
               for o in replay(mod, lambda: enable(build), d.messages))


@pytest.mark.parametrize("express", [False, True])
def test_reply_refuses_admission_while_foreground_lanes_or_cofill_wait(
        express_decoder, monkeypatch, express):
    mod, build = express_decoder
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    d = enable(build)
    if not express:
        d.express_buf = None
    warm(d, [2, 3, 4])
    long, short = _stream([5] * (17 if express else 4), 4), _stream([3, 7, 8], 4)
    if express:
        short.prefill_slice_layers = 1  # Keep both lanes suspended for this admission gate.
    d.begin_admit(long)
    if express:
        d.prefill_step()
    d.begin_admit(short)
    if express:
        d.prefill_step()
        assert d.fill_owner == long.sid and d.express_owner == short.sid
    else:
        assert long.cofill_ready and short.cofill_ready
        assert mod.cofill.candidates(d, long) == [long, short]
    frames = len(d.messages)
    spans = dict(d.pool.spans)
    with pytest.raises(ValueError, match="idle pooled worker"):
        d.begin_admit(background([2, 3, 4, 5], 3))
    assert len(d.messages) == frames and d.pool.spans == spans
    d.drop()
    assert d.express_owner is d.fill_owner is None and d.pool.available == d.capacity
