"""Pooled startup accounts for actual allocations and preserves the advertised context."""

import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda import geometry
from tensorfold.cuda.capacity import Geometry
from tensorfold.families.glm5_next.cuda.engine import GlmEngine, configured_prefill_rows, fit_shared_pool
from tests.test_cuda_geometry import allocations, bytes_in


@pytest.mark.parametrize("streams", [2, 4, 8])
@pytest.mark.parametrize("rows", [2048, 8192])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("decode_rows", [8, 16, 32])
def test_actual_pooled_constructors_fit_the_estimate(monkeypatch, allocations, streams, rows, mtp, decode_rows):
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    for name in ("forward", "kda", "latent", "attention"):
        monkeypatch.setattr(importlib.import_module("tensorfold.families.glm5_next.cuda." + name), "torch", fake)
    monkeypatch.setattr(mod.latent, "ENABLED", True)
    text = {"hidden_size": 512, "num_attention_heads": 6, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 6,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1026,
            "moe_intermediate_size": 576, "num_experts_per_tok": 2,
            "num_nextn_predict_layers": int(mtp)}
    cfg = SimpleNamespace(heads=6, lin_heads=6, conv=4, qk_dim=256, v_dim=256, index_dim=128,
                          hidden=512, streams=4, q_lora=512, kv_lora=512, index_heads=32,
                          dense_width=1152, top_k=2, moe_width=576, experts=8, quant="mlx")
    layers = [SimpleNamespace(index=i, kind="kda" if i % 2 == 0 else "dsa",
                              kda=SimpleNamespace(proj=SimpleNamespace(n=3 * 2 * 128 + 256 + 2))) for i in range(4)]
    w = SimpleNamespace(cfg=cfg, world=3, device="cpu", layers=layers, meta={"long_context": True},
                        mtp=SimpleNamespace() if mtp else None, head=SimpleNamespace(n=342))
    nominal = 262152
    arena = nominal + (streams - 1) * 64
    mod.Buffers(w, decode_rows, arena).set_taps((0, 1, 2, 3, 3), cfg.hidden)
    if mtp:
        mod.Buffers(w, decode_rows, arena)
    mod.Buffers(w, streams * decode_rows, arena).set_taps((0, 1, 2, 3, 3), cfg.hidden)
    mod.Buffers(w, rows, arena, prefill=True).set_taps((0, 1, 2, 3, 3), cfg.hidden)
    mod.State(w, arena, decode_rows)
    for _ in range(streams - 1):
        mod.State(w, 0, decode_rows)     # extra slots get views, never full KV copies
    estimate = geometry.mla_geometry(text, 3, 8, sequences=streams, latent=True, pooled=True,
                                     prefill_rows=rows, decode_rows=decode_rows).bytes_at(nominal)
    assert bytes_in(arrays) <= estimate


def test_pool_cache_slope_is_one_context_not_eight():
    text = {"hidden_size": 4096, "num_attention_heads": 66, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 66,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 153600,
            "moe_intermediate_size": 1536, "num_experts_per_tok": 8}
    for n in (2, 4, 8):
        pooled = geometry.mla_geometry(text, 3, 8, sequences=n, latent=True, pooled=True)
        serial = geometry.mla_geometry(text, 3, 8, latent=True)
        assert pooled.bytes_at(262152) - pooled.bytes_at(131080) == serial.bytes_at(262152) - serial.bytes_at(131080)
        assert pooled.reserve == serial.reserve == 8


def test_prefill_rows_is_forwarded_for_concurrency(monkeypatch):
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", "8192")
    assert configured_prefill_rows(None, context=262144, explicit=True, parallel=8) == 8192


@pytest.mark.parametrize("streams", [0, 9])
def test_invalid_stream_count_fails_before_cuda(tmp_path, streams):
    with pytest.raises(ValueError, match="between 1 and 8"):
        GlmEngine(tmp_path, rank=0, master="", port=0, parallel=streams)


def test_shared_pool_fits_rank_minimum_without_consuming_kept_store_or_safety_budget():
    target = Geometry(lambda slots: 10 * slots, reserve=8)
    draft = Geometry(lambda slots: 3 * slots, reserve=8)
    plan = dict(budget_bytes=100 + 13 * (360 + 8) + 500,
                weight_bytes_estimate=100, loading_bytes_estimate=200)
    chosen, slots, work, total = fit_shared_pool(
        plan, 260, 360, 500, target, draft, lambda row: [row, [360, 320]])
    assert (chosen, slots, work, total) == (320, 328, 13 * 328, 100 + 13 * 328)
    plan["budget_bytes"] = 100 + 13 * (260 + 8) + 500
    assert fit_shared_pool(plan, 260, 360, 500, target, draft, lambda row: [row, row])[0] == 260
    with pytest.raises(ValueError, match="must match"):
        fit_shared_pool(plan, 260, 360, 500, target, draft, lambda row: [row, [361, 260]])
