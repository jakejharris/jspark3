"""The row knob changes scratch admission, never context or KV precision. CPU only."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity, geometry
from tensorfold.families.glm5_next.cuda.engine import GlmEngine, configured_prefill_rows, configured_prefill_rows_idle
from test_cuda_capacity import checkpoint
from test_cuda_geometry import allocations  # noqa: F401


TEXT = {"max_position_embeddings": 262144, "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 6,
        "linear_num_heads": 6, "hidden_size": 512, "vocab_size": 1024,
        "qk_nope_head_dim": 256, "v_head_dim": 256, "kv_lora_rank": 512,
        "moe_intermediate_size": 768, "intermediate_size": 1536, "num_experts_per_tok": 2}


def resolve(**kw):
    return configured_prefill_rows(kw.pop("rows", None), context=kw.pop("context", 262144),
                                   explicit=kw.pop("explicit", True), parallel=kw.pop("parallel", 1))


def test_unset_preserves_default_and_constructor_override(monkeypatch):
    monkeypatch.delenv("TF_GLM_PREFILL_ROWS", raising=False)
    assert resolve(context=None, explicit=False, parallel=3) == 2048
    assert resolve(rows=16, context=None) == 16
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", "8192")
    assert resolve(rows=32) == 32


@pytest.mark.parametrize("value", ["2048", "4096", "8192"])
def test_allowed(monkeypatch, value):
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", value)
    assert resolve() == int(value)


@pytest.mark.parametrize("value", ["", "0", "-1", "2049", "16384", "4k", "4096.0", " 4096"])
def test_invalid_fails_closed(monkeypatch, value):
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", value)
    with pytest.raises(ValueError, match="must be"):
        resolve()


@pytest.mark.parametrize("options", [{"context": None}, {"context": 0}, {"explicit": False}])
def test_no_implicit_context_shrink_or_ignored_multistream_knob(monkeypatch, options):
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", "4096")
    with pytest.raises(ValueError, match="context|one GLM stream"):
        resolve(**options)


@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("slots", [32768, 131072, 262152])
def test_default_geometry_unchanged_and_rows_increase_only_scratch(latent, slots):
    default = geometry.mla_geometry(TEXT, 3, 8, latent=latent)
    values = [geometry.mla_geometry(TEXT, 3, 8, latent=latent, prefill_rows=r).bytes_at(slots)
              for r in (2048, 4096, 8192)]
    assert default.bytes_at(slots) == values[0] < values[1] < values[2]
    # Increasing capacity: identical KV slope; the only extra per-token term is pool-score scratch.
    deltas = []
    for rows in (2048, 4096, 8192):
        g = geometry.mla_geometry(TEXT, 3, 8, latent=latent, prefill_rows=rows)
        assert g.reserve == 8 and g.minimum_slots == default.minimum_slots
        deltas.append(g.bytes_at(slots + 4096) - g.bytes_at(slots) - 4096 * rows)
    assert len(set(deltas)) == 1


@pytest.mark.parametrize("rows", [2048, 4096, 8192])
@pytest.mark.parametrize("idle", [False, True])
def test_constructor_passes_knob_to_admission_before_loading_weights(tmp_path, monkeypatch, allocations, rows, idle):
    torch = pytest.importorskip("torch")
    from tensorfold.families.glm5_next.cuda import LATENT
    from tensorfold.families.glm5_next.cuda import prefill_options as p1
    from tensorfold.families.glm5_next.cuda import weights
    seen = {}
    monkeypatch.setattr(weights.Config, "read", lambda _: SimpleNamespace(dense_limit=2051))
    monkeypatch.setattr(weights, "load", lambda *a, **kw: pytest.fail("loaded before admission"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS", str(4096 if idle else rows))
    monkeypatch.setenv("TF_GLM_PREFILL_ROWS_IDLE", "8192" if idle else "0")
    monkeypatch.setattr(p1, "ATTENTION_TILES", idle)
    monkeypatch.setenv("TF_GLM_VISION", "0")
    checkpoint(tmp_path, TEXT, [])

    def admit(self, fn, model, context, explicit, torch, make_geometry, transform, **kw):
        g = make_geometry(TEXT)
        seen.update(context=context, explicit=explicit, size=g.bytes_at(262152))
        raise RuntimeError("admission reached")

    monkeypatch.setattr(GlmEngine, "_admit", admit)
    with pytest.raises(RuntimeError, match="admission reached"):
        GlmEngine(tmp_path, rank=0, master="", port=0, world=3, context=262144,
                  comm=SimpleNamespace(barrier=lambda: None))
    assert seen == {"context": 262144, "explicit": True,
                    "size": geometry.mla_geometry(TEXT, 3, 8, latent=LATENT,
                                                   prefill_rows=8192 if idle else rows).bytes_at(262152)}


@pytest.mark.parametrize("value", [None, "0", "8192", "", "4096", "16384", " 8192", "8192.0"])
def test_idle_rows_strict_opt_in(monkeypatch, value):
    from tensorfold.families.glm5_next.cuda import prefill_options as p1

    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    if value is None:
        monkeypatch.delenv("TF_GLM_PREFILL_ROWS_IDLE", raising=False)
    else:
        monkeypatch.setenv("TF_GLM_PREFILL_ROWS_IDLE", value)
    if value in (None, "0", "8192"):
        assert configured_prefill_rows_idle(4096, context=262144, explicit=True) == (8192 if value == "8192" else 0)
    else:
        with pytest.raises(ValueError, match="must be 0 or 8192"):
            configured_prefill_rows_idle(4096, context=262144, explicit=True)


@pytest.mark.parametrize("rows,tiles,context,explicit", [
    (2048, True, 262144, True), (8192, True, 262144, True), (4096, False, 262144, True),
    (4096, True, None, True), (4096, True, 0, True), (4096, True, 262144, False),
])
def test_idle_rows_refuses_an_unsafe_base_or_implicit_context(monkeypatch, rows, tiles, context, explicit):
    from tensorfold.families.glm5_next.cuda import prefill_options as p1

    monkeypatch.setenv("TF_GLM_PREFILL_ROWS_IDLE", "8192")
    monkeypatch.setattr(p1, "ATTENTION_TILES", tiles)
    with pytest.raises(ValueError, match="needs"):
        configured_prefill_rows_idle(rows, context=context, explicit=explicit)


@pytest.mark.parametrize("field", [-11, -10, -9, -8, -7, -6, -5, -4, -3],
                         ids=["whole", "tile128", "tile64", "idle", "busy", "express", "atomic", "express_cofill",
                              "exact_replay"])
def test_e1_rank_mismatch_refuses_before_weights(tmp_path, monkeypatch, allocations, field):
    import torch
    from tensorfold.families.glm5_next.cuda import prefill_options as p1, weights

    checkpoint(tmp_path, TEXT, [])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(weights.Config, "read", lambda _: SimpleNamespace(dense_limit=2051))
    monkeypatch.setattr(weights, "load", lambda *a, **kw: pytest.fail("loaded mismatched ranks"))
    for name in ("ATTENTION_TILES", "EXPERT_WHOLE_PASS", "EXPERT_PREFILL128", "EXPERT_PREFILL64"):
        monkeypatch.setattr(p1, name, True)
    for name, value in {"TF_GLM_PREFILL_ROWS": "4096", "TF_GLM_PREFILL_ROWS_IDLE": "8192",
                        "TF_GLM_VISION": "0"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TF_GLM_POOL_TOKENS", raising=False)
    monkeypatch.setattr(GlmEngine, "_admit", lambda *a, **kw: dict(
        context_window=262144, cache_slots=262152, budget_bytes=50 * 2**30, total_bytes_estimate=2**30))

    def gather(self, values):
        if len(values) == 3:  # pool setting agreement precedes the boot settings
            return [values] * 3
        assert values[-11:-6] == [1, 1, 1, 8192, 4096]
        assert values[-6:-3] == [int(p1.EXPRESS), p1.EXPRESS_ATOMIC_ROWS, int(p1.EXPRESS_COFILL)]
        assert values[-3] == int(p1.EXACT_REPLAY)
        peer = list(values)
        peer[field] ^= 1
        return [values, peer, values]

    monkeypatch.setattr(GlmEngine, "_gather_ints", gather)
    with pytest.raises(RuntimeError, match="different settings"):
        GlmEngine(tmp_path, rank=0, master="", port=0, world=3, context=262144,
                  comm=SimpleNamespace(barrier=lambda: None))


def test_larger_rows_refuse_instead_of_shrinking_same_explicit_window():
    small = geometry.mla_geometry(TEXT, 3, 8, latent=True, prefill_rows=2048)
    large = geometry.mla_geometry(TEXT, 3, 8, latent=True, prefill_rows=8192)
    # Use the real admission chooser, with its measured fitting-window result.
    plan = SimpleNamespace(settings=[262144, 262144, 1], fitting=131072,
                           requested=262144, native=262144, explicit=True)
    assert large.bytes_at(262152) > small.bytes_at(262152)
    with pytest.raises(ValueError, match="requested context 262144"):
        capacity.choose(plan)
