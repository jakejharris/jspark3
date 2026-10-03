"""Tiled co-prefill controls: independent long prompts, checkpoints, ranks and cancellation."""

import runpy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import geometry
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import cofill_big, prefill_options as p1
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream, _drain  # noqa: F401
from test_glm_cofill import cofill_decoder, state_hash  # noqa: F401
from test_glm_session_batched import attach, finish_writers  # noqa: F401


@pytest.fixture
def big_decoder(cofill_decoder, monkeypatch):
    mod, build, forwards = cofill_decoder
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    monkeypatch.setattr(p1, "COFILL_BIG", True)
    monkeypatch.setattr(cofill_big, "stage_streams", mod.cofill.stage_streams)
    monkeypatch.setattr(cofill_big, "compute_streams", mod.cofill.compute_streams)
    monkeypatch.setattr(cofill_big, "commit", mod.cofill.commit)

    def make(slots=4, rows=8192):
        d = build(slots, rows=rows, limit=65536, pool_limit=300000)
        d.owner.cache_bytes = 10_000_000
        return d

    return mod, make, forwards


def fill(d):
    for _ in range(200):
        if not d.filling:
            return
        d.finish(d.prefill_step())
    pytest.fail("big cofill did not finish")


def test_big_default_off_and_requires_tiles_and_pooled_server(monkeypatch, tmp_path):
    monkeypatch.delenv("TF_GLM_COFILL_BIG", raising=False)
    assert not runpy.run_path(p1.__file__)["COFILL_BIG"]
    monkeypatch.setenv("TF_GLM_COFILL_BIG", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        runpy.run_path(p1.__file__)
    monkeypatch.setattr(p1, "COFILL_BIG", True)
    for tiles, parallel in ((False, 2), (True, 1)):
        monkeypatch.setattr(p1, "ATTENTION_TILES", tiles)
        with pytest.raises(ValueError, match="requires"):
            GlmEngine(tmp_path, rank=0, master="", port=0, parallel=parallel)


@pytest.mark.parametrize("slots", [2, 4, 8])
@pytest.mark.parametrize("rows", [4096, 8192])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("sampled", [False, True])
def test_big_partial_prompts_match_alone_state_taps_and_continuation(big_decoder, slots, rows, cached, sampled):
    _, make, forwards = big_decoder
    prompts = [[3 + i] * (rows + 37 + 11 * i) for i in range(slots)]
    sampling = [Sampling(22 + i, .73, 20, .92) if sampled else None for i in range(slots)]
    expected = []
    for prompt, sampler in zip(prompts, sampling):
        ref = make(slots, rows)
        s = _stream(prompt, 9, sampling=sampler)
        s.cofill = False
        ref.admit(s)
        expected.append((state_hash(s), s.drafter.context_end))
        _drain(ref)
        expected[-1] += (s.out,)
    d = make(slots, rows)
    if cached:
        for prompt in prompts:
            prefix = _stream(prompt[:513], 1)
            d.admit(prefix)
            d.finish([prefix])
    streams = [_stream(prompt, 9, sampling=sampler) for prompt, sampler in zip(prompts, sampling)]
    for s in streams:
        d.begin_admit(s)
        assert isinstance(d.filling[s.sid], cofill_big.Prefill)
    at = len(forwards)
    d.prefill_step()
    assert len(forwards) == at + 1 and len(forwards[-1]) == slots
    assert sum(n for _, n in forwards[-1]) == rows
    assert all(not s.out and 0 < s.fill_pos < len(s.prompt) for s in streams)
    fill(d)
    assert [(state_hash(s), s.drafter.context_end) for s in streams] == [r[:2] for r in expected]
    _drain(d)
    assert [s.out for s in streams] == [r[2] for r in expected]
    for s in streams:
        assert s.cofill_stats["max_total_rows"] == rows
        assert s.cofill_stats["cofilled_passes"] >= 2
        assert s.cofill_stats["prompt_rows"] == len(s.prompt) - (513 if cached else 0)
        assert s.cached == (513 if cached else 0)


def test_32k_prompts_share_big_pass_and_new_arrival_joins_between_passes(big_decoder):
    _, make, forwards = big_decoder
    d = make(4)
    long = [_stream([token] * 32768, 5) for token in (3, 7)]
    for s in long:
        d.begin_admit(s)
    d.prefill_step()
    assert [s.fill_pos for s in long] == [4096, 4096]
    short = _stream([11] * 1100, 5)
    d.begin_admit(short)
    d.prefill_step()
    assert short.out and not any(s.out for s in long)
    assert len(forwards[-1]) == 3 and sum(n for _, n in forwards[-1]) == 8192
    before = len(short.out)
    d.round()
    assert len(short.out) > before  # existing decode can run between complete passes
    fill(d)
    _drain(d)
    assert not d.filling and not d.streams


def test_request_off_and_explicit_slices_keep_original_stepper(big_decoder):
    _, make, _ = big_decoder
    d = make(2)
    for off in (True, False):
        s = _stream([7] * 1200, 5)
        if off:
            s.cofill = False
        else:
            s.prefill_slice_layers = 1
        d.begin_admit(s)
        assert not isinstance(d.filling[s.sid], cofill_big.Prefill)
        assert not s.cofill_ready
    d.drop()


def test_big_boot_flag_batches_admission_without_small_flag(big_decoder, monkeypatch):
    _, make, _ = big_decoder
    monkeypatch.setenv("TF_GLM_COFILL", "0")
    d = make(8)
    assert d.admit_per_round == 8
    s = _stream([3] * 1100, 5)
    d.begin_admit(s)
    assert isinstance(d.filling[s.sid], cofill_big.Prefill)


@pytest.mark.parametrize("kind", ["q4", "images", "mtp", "reply"])
def test_other_prefill_modes_keep_the_original_stepper(big_decoder, kind):
    _, make, _ = big_decoder
    d = make(2)
    s = _stream([3] * 1200, 5)
    if kind == "q4":
        d.w.cfg.quant = "q4"
    elif kind == "images":
        d.owner._image_rows = lambda *args: (torch.tensor([0]), torch.ones(1, 2))
    elif kind == "mtp":
        d.w.mtp, d.owner.drafter, s.policy = SimpleNamespace(), None, "2"
    else:
        d.reply_prefill, d.fair_schedule = True, True
        d.owner._gather_ints = lambda v: [v] * 3
        warm = _stream(s.prompt[:513], 1)
        d.admit(warm)
        d.finish([warm])
        s.reply_prefill, s.reply_base = True, 513
    d.begin_admit(s)
    assert not isinstance(d.filling[s.sid], cofill_big.Prefill)
    assert not s.cofill_ready
    d.drop()


@pytest.mark.parametrize("fault", ["owner", "express-owner", "duplicate", "too-wide", "past-end", "stale-pos"])
def test_bad_big_frames_fail_before_another_forward(big_decoder, fault):
    _, make, forwards = big_decoder
    d = make(2, 4096)
    streams = [_stream([token] * 9000, 5) for token in (3, 7)]
    for s in streams:
        d.begin_admit(s)
    d.prefill_step()
    pieces = cofill_big.plan(d, streams[0])
    if fault == "owner":
        d.fill_owner = streams[0].sid
    elif fault == "express-owner":
        d.express_owner = streams[0].sid
    elif fault == "duplicate":
        pieces = [pieces[0], pieces[0]]
    elif fault == "too-wide":
        pieces = [(s.sid, s.fill_pos + 3000) for s in streams]
    elif fault == "past-end":
        pieces = [(streams[0].sid, 9001)]
    else:
        streams[0].st.set_pos(streams[0].st.pos + 1)
    before = len(forwards)
    with pytest.raises(RuntimeError, match="big cofill"):
        cofill_big.run(d, pieces)
    assert len(forwards) == before


def test_followers_use_rank_zero_chunks_and_cancelled_span_can_be_reused(big_decoder, monkeypatch):
    _, make, _ = big_decoder
    d = make(4, 4096)
    a, b = [_stream([token] * 10000, 15) for token in (3, 7)]
    for s in (a, b):
        d.begin_admit(s)
    d.prefill_step()
    a.cancelled = lambda: True
    d.finish(d.prefill_step())
    replacement = _stream([11] * 4500, 8, sampling=Sampling(12, .73, 20, .92))
    d.begin_admit(replacement)
    assert replacement.start == a.start
    fill(d)
    _drain(d)
    assert any(m[0] == cofill_big.FILL_BIG and m[1] == 1 for m in d.messages)
    for rank in (1, 2):
        follower = make(4, 4096)
        follower.rank = follower.owner.rank = rank
        outputs, finish = {}, follower._finish
        def capture(sids):
            outputs.update({sid: list(follower.streams[sid].out) for sid in sids})
            finish(sids)
        follower._finish = capture
        messages = iter(d.messages + [[99]])
        follower.share = lambda _: next(messages)
        monkeypatch.setattr(cofill_big, "plan", lambda *args: pytest.fail("follower planned chunks"))
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == {a.sid: [], b.sid: b.out, replacement.sid: replacement.out}


def test_checkpoint_boundaries_cancel_and_disk_resume(big_decoder, tmp_path):
    _, make, _ = big_decoder
    d = make(2, 4096)
    d.owner.drafter.ring, d.owner.drafter.window = 8192, 1024
    disk = attach(d, tmp_path, checkpoints=True, cancel=True)
    d.owner.checkpoint_after = lambda pos: (pos // 1024 + 1) * 1024
    streams = [_stream([token] * 6000, 5) for token in (3, 7)]
    for s in streams:
        d.begin_admit(s)
    d.prefill_step()
    assert [s.fill_pos for s in streams] == [1024, 1024]
    assert {len(snap.ids) for _, snap, _ in d.sessions.staged} == {1024}
    streams[0].cancelled = lambda: True
    d.finish(d.prefill_step())
    assert any(len(e.ids) == 1024 for e in disk.entries.values())
    retry = _stream(streams[0].prompt, 5)
    d.begin_admit(retry)
    assert retry.cached == 1024
    fill(d)
    _drain(d)
    ref = make(2, 4096)
    wanted = _stream(retry.prompt, 5)
    ref.admit(wanted)
    _drain(ref)
    assert retry.out == wanted.out


def test_forged_checkpoint_crossing_is_rejected_before_restore(big_decoder):
    _, make, forwards = big_decoder
    d = make(2, 4096)
    d.sessions = SimpleNamespace(store=None, saved=lambda *a, **k: None, begin=lambda *a, **k: None,
                                prefill_kwargs=lambda s: {"mark": lambda pos: 1000, "keep": lambda s: None})
    s = _stream([3] * 6000, 5)
    d.begin_admit(s)
    before = len(forwards)
    with pytest.raises(RuntimeError, match="checkpoint"):
        cofill_big.run(d, [(s.sid, 2000)])
    assert len(forwards) == before and not d.filling[s.sid].started


@pytest.mark.parametrize("fault", ["head", "position"])
def test_wrong_head_or_segment_position_breaks_alone_hash_gate(big_decoder, monkeypatch, fault):
    _, make, _ = big_decoder
    d = make(2, 4096)
    a, b = _stream([3] * 1100, 5), _stream([7] * 1200, 5)
    ref = make(2, 4096)
    expected = _stream(a.prompt, 5)
    ref.admit(expected)
    wanted = state_hash(expected), expected.out[:]
    compute = cofill_big.compute_streams
    def broken(w, segs, buf, **kwargs):
        if fault == "position":
            segs[0][0].set_pos(1)
        logits = compute(w, segs, buf, **kwargs)
        return logits[-1:].expand_as(logits).clone() if fault == "head" else logits
    monkeypatch.setattr(cofill_big, "compute_streams", broken)
    for s in (a, b):
        d.begin_admit(s)
    if fault == "position":
        with pytest.raises(RuntimeError, match="committed position"):
            d.prefill_step()
        return
    d.prefill_step()
    with pytest.raises(AssertionError):
        assert (state_hash(a), a.out) == wanted


@pytest.mark.parametrize("rows", [2048, 4096, 8192])
def test_big_kda_heads_and_hidden_bytes_are_admitted(big_decoder, monkeypatch, rows):
    text = {"hidden_size": 4096, "num_attention_heads": 66, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"], "linear_num_heads": 66,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "kv_lora_rank": 512,
            "vocab_size": 154880, "moe_intermediate_size": 2112, "num_experts_per_tok": 8}
    on = geometry.mla_geometry(text, 3, 8, sequences=8, pooled=True, latent=True, prefill_rows=rows).bytes_at(360456)
    monkeypatch.setattr(p1, "COFILL_BIG", False)
    off = geometry.mla_geometry(text, 3, 8, sequences=8, pooled=True, latent=True, prefill_rows=rows).bytes_at(360456)
    assert on - off == rows * 22 * 128 * 2 + 8 * 8 * 51627 + 2 * 8 * 4096
