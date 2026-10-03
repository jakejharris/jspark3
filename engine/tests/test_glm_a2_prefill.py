"""Tiled attention ownership and allocation gates. Arithmetic hashes live in the CUDA suite."""

import copy
import importlib
import runpy
from types import SimpleNamespace

import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from test_cuda_geometry import allocations, bytes_in
from test_glm_batched_host import setup_decoder, _stream
from test_glm_prefill_slices import sliced_decoder, drain_steps, replay
from test_glm_session_batched import finish_writers  # noqa: F401


def test_flags_independent_default_off(monkeypatch):
    for name in ("ATTENTION_TILES", "EXPRESS"):
        monkeypatch.delenv("TF_GLM_A2_" + name, raising=False)
    assert not runpy.run_path(p1.__file__)["ATTENTION_TILES"]
    assert not runpy.run_path(p1.__file__)["EXPRESS"]
    monkeypatch.setenv("TF_GLM_A2_EXPRESS", "1")
    assert runpy.run_path(p1.__file__)["EXPRESS"]
    assert not runpy.run_path(p1.__file__)["ATTENTION_TILES"]
    monkeypatch.setenv("TF_GLM_A2_ATTENTION_TILES", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        runpy.run_path(p1.__file__)


@pytest.fixture
def express_decoder(sliced_decoder, monkeypatch):
    mod, original = sliced_decoder
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    def build():
        d = original()
        d.express_buf = copy.deepcopy(d.owner.e.pbuf)
        d.express_buf.rows = 4
        return d
    return mod, build


def test_short_prefill_passes_suspended_long_with_both_followers(express_decoder):
    torch = pytest.importorskip("torch")
    mod, build = express_decoder
    d = build()
    long, short = _stream([1] * 17, 8), _stream([3, 4, 5], 8)
    d.begin_admit(long)
    d.prefill_step()
    saved = d.owner.e.pbuf.x.clone()
    assert d.fill_owner == long.sid and long.fill_layer == 1
    d.begin_admit(short)
    assert short.express and short.engine.pbuf is not long.engine.pbuf
    while not short.out:
        d.prefill_step()
        assert long.fill_layer == 1
        assert torch.equal(saved, d.owner.e.pbuf.x)
    assert d.fill_owner == long.sid and d.express_owner is None
    drain_steps(d)
    assert all(o == {long.sid: long.out, short.sid: short.out} for o in replay(mod, build, d.messages))


@pytest.mark.parametrize("lane", ["short", "long"])
@pytest.mark.parametrize("cut", [1, 3, 5])
def test_cancel_either_owner_leaves_other_buffer_and_reuses_span(express_decoder, lane, cut):
    torch = pytest.importorskip("torch")
    mod, build = express_decoder
    d = build()
    long, short = _stream([1] * 17, 4), _stream([3] * 3, 4)
    long.prefill_slice_layers = short.prefill_slice_layers = cut
    d.begin_admit(long); d.prefill_step()
    d.begin_admit(short); d.prefill_step()
    victim, survivor = (short, long) if lane == "short" else (long, short)
    saved = survivor.engine.pbuf.x.clone()
    victim.cancelled = lambda: True
    assert d.prefill_step() == [victim]
    assert torch.equal(survivor.engine.pbuf.x, saved)
    d.finish([victim])
    replacement = _stream(victim.prompt, 4)
    d.begin_admit(replacement)
    assert replacement.start == victim.start
    drain_steps(d)
    expected = {victim.sid: [], survivor.sid: survivor.out, replacement.sid: replacement.out}
    assert all(o == expected for o in replay(mod, build, d.messages))


def test_shared_buffer_negative_control_detects_corruption(express_decoder):
    _, build = express_decoder
    d = build()
    long, short = _stream([1] * 17, 4), _stream([3] * 3, 4)
    d.begin_admit(long); d.prefill_step()
    d.express_buf = d.owner.e.pbuf
    d.express_buf.rows = 4
    d.begin_admit(short)
    while not short.out:
        d.prefill_step()
    with pytest.raises(AssertionError, match="overwrote a paused chunk"):
        d.prefill_step()


@pytest.mark.parametrize("fault", [False, True], ids=["hit", "rank-miss"])
def test_r2_restore_vote_decides_express_binding(express_decoder, tmp_path, monkeypatch, fault):
    from test_glm_session_batched import attach
    _, build = express_decoder
    d = build()
    attach(d, tmp_path)
    prefix = _stream([2] * 12, 1, policy="0")
    d.begin_admit(prefix)
    drain_steps(d)
    d.sessions.store.wait_pending()  # establish durability before the forced memory eviction
    d.cache.clear()
    if fault:
        monkeypatch.setattr(d.sessions, "restore", lambda s, snap: None)
    resumed = _stream(prefix.prompt + [3, 4], 1, policy="0")
    d.begin_admit(resumed)
    assert resumed.cached_named == 12
    assert resumed.cached == (0 if fault else 12)
    assert resumed.express is (not fault)
    assert resumed.slice_layers == (1 if fault else 0)
    assert resumed.engine.pbuf is (d.owner.e.pbuf if fault else d.express_buf)
    drain_steps(d)


def test_express_checkpoint_cut_and_cancel_keep_only_its_committed_prefix(express_decoder, tmp_path):
    from test_glm_session_batched import attach
    _, build = express_decoder
    d = build()
    disk = attach(d, tmp_path, checkpoints=True, cancel=True)
    long = _stream(list(range(17)), 1, policy="0")
    short = _stream([3, 50, 5], 1, policy="0")
    short.prefill_slice_layers = 1  # Cancel a partially executed chunk, even for tiny prompts.
    d.begin_admit(long); d.prefill_step()
    d.begin_admit(short)
    while short.fill_pos < 1:
        d.prefill_step()
    assert d.fill_owner == long.sid and short.fill_pos == 1
    d.prefill_step()   # begin the next express chunk, preserving its previous checkpoint
    short.cancelled = lambda: True
    d.finish(d.prefill_step())
    assert d.fill_owner == long.sid and d.express_owner is None
    assert {tuple(e.ids) for e in disk.entries.values()} == {(3,)}
    drain_steps(d)


@pytest.mark.parametrize("explicit_slices", [False, True])
@pytest.mark.parametrize("big", [False, True])
def test_express_cofill_and_m1_keep_separate_owners(express_decoder, monkeypatch, explicit_slices, big):
    mod, build = express_decoder
    monkeypatch.setenv("TF_GLM_COFILL", "1")
    monkeypatch.setattr(p1, "COFILL_BIG", big)
    if big:
        monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    d = build()
    long = _stream([1] * 17, 8)
    short = [_stream([token] * 3, 8) for token in (3, 4)]
    for stream in (long, *short):
        stream.cofill = True
        if explicit_slices:
            stream.prefill_slice_layers = 1
    d.begin_admit(long)
    d.prefill_step()
    for stream in short:
        d.begin_admit(stream)
    assert d.fill_owner == long.sid
    assert all(not s.cofill_ready for s in (long, *short))
    assert all(s.express for s in short) and not long.express
    d.prefill_step()
    assert d.express_owner == (short[0].sid if explicit_slices else None)
    assert bool(short[0].out) is (not explicit_slices)
    assert d.fill_owner == long.sid
    assert mod.cofill.candidates(d, short[1]) == []
    # A forged cofill message must also fail on a follower before touching data.
    with pytest.raises(RuntimeError, match="suspended"):
        mod.cofill.run(d, [s.sid for s in short])
    while any(not s.out for s in short):
        d.prefill_step()
        assert long.fill_layer == 1
    drain_steps(d)
    assert not any(m[0] == mod.cofill.FILL_COFILL for m in d.messages)
    assert not any(m[0] == mod.cofill_big.FILL_BIG for m in d.messages)
    expected = {s.sid: s.out for s in (long, *short)}
    assert all(o == expected for o in replay(mod, build, d.messages))
    assert d.express_owner is d.fill_owner is None


@pytest.mark.torch
@pytest.mark.parametrize("rows", [2048, 4096, 8192])
def test_express_admission_covers_complete_buffer_set(monkeypatch, allocations, rows):
    arrays, fake = allocations
    f = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    for module in (f, f.latent, f.kda_mod, f.grouped):
        monkeypatch.setattr(module, "torch", fake)
    monkeypatch.setattr(f.latent, "ENABLED", True)
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    cfg = SimpleNamespace(heads=66, lin_heads=66, conv=4, qk_dim=256, v_dim=256, index_dim=128,
                          hidden=4096, streams=4, q_lora=1536, kv_lora=512, index_heads=32,
                          dense_width=12288, moe_width=2112, shared_width=2112, experts=288, top_k=8, quant="mlx")
    w = SimpleNamespace(cfg=cfg, device="cuda", world=3, head=SimpleNamespace(n=51627),
                        layers=[SimpleNamespace(kind="kda", kda=SimpleNamespace(proj=SimpleNamespace(n=8768)))])
    b = f.Buffers(w, 2048, 262144, prefill=True)
    b.set_taps((5, 14, 24, 33, 42), cfg.hidden)
    actual = bytes_in([a for a in arrays if a.device == "cuda"])
    text = dict(hidden_size=4096, num_attention_heads=66, linear_num_heads=66, num_hidden_layers=45,
                layer_types=["linear_attention"] * 34 + ["full_attention"] * 11, vocab_size=154880,
                qk_nope_head_dim=256, v_head_dim=256, kv_lora_rank=512, q_lora_rank=1536,
                moe_intermediate_size=2112, intermediate_size=12288, num_experts_per_tok=8)
    monkeypatch.setattr(p1, "EXPRESS", False)
    base = geometry.mla_geometry(text, 3, 8, pooled=True, sequences=8, latent=True, prefill_rows=rows).bytes_at(262144)
    monkeypatch.setattr(p1, "EXPRESS", True)
    extra = geometry.mla_geometry(text, 3, 8, pooled=True, sequences=8, latent=True, prefill_rows=rows).bytes_at(262144) - base
    assert extra >= actual
    print("express bytes: actual", actual, "admission", extra)
    arrays.clear()
    large = f.Buffers(w, rows, 262144, prefill=True)
    large.set_taps((5, 14, 24, 33, 42), cfg.hidden)
    print("main pass", rows, "buffer bytes", bytes_in([a for a in arrays if a.device == "cuda"]),
          "transient admission", geometry.mla_chunk_scratch(text, 3, 262144, latent=True, prefill_rows=rows))
