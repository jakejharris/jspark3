"""CPU startup refusal for the P-A latent partial index's signed-int32 limit."""
import sys
from types import SimpleNamespace

import pytest

from tensorfold.cuda.geometry import mla_geometry
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from tensorfold.families.glm5_next.cuda.split import padded_config
from test_cuda_capacity import checkpoint
from test_glm_prefill_rows import TEXT

MODEL = dict(TEXT, num_attention_heads=64, linear_num_heads=64)


@pytest.mark.parametrize('world', [1, 2, 3])
@pytest.mark.parametrize('rows', [2048, 4096, 8192])
def test_real_topologies_refuse_only_overflowing_latent_bounds(world, rows):
    text = padded_config(MODEL, world)
    kw = dict(latent=True, prefill_rows=rows)
    if rows == 8192 and world < 3:
        with pytest.raises(ValueError, match=rf'TF_GLM_PREFILL_ROWS=8192 is unsafe for TP={world}'):
            mla_geometry(text, world, 8, check_prefill_offsets=True, **kw)
    else:
        checked = mla_geometry(text, world, 8, check_prefill_offsets=True, **kw)
        unchanged = mla_geometry(text, world, 8, **kw)
        for slots in (2560, 32768, 262152):
            assert checked.bytes_at(slots) == unchanged.bytes_at(slots)


@pytest.mark.parametrize('world', [1, 2])
def test_existing_unchecked_and_nonlatent_paths_are_unchanged(world):
    text = padded_config(MODEL, world)
    mla_geometry(text, world, 8, latent=True, prefill_rows=8192)  # unset-knob path
    a = mla_geometry(text, world, 8, latent=False, prefill_rows=8192)
    b = mla_geometry(text, world, 8, latent=False, prefill_rows=8192, check_prefill_offsets=True)
    assert a.bytes_at(262152) == b.bytes_at(262152)


def test_exact_signed_offset_boundary():
    # One chunk of 512 rows: last legal offset is 2**31 - 1, not the exclusive end.
    at_limit = dict(MODEL, num_attention_heads=8192)
    mla_geometry(at_limit, 1, 8, minimum_slots=0, latent=True, prefill_rows=512,
                 check_prefill_offsets=True)
    beyond = dict(at_limit, num_attention_heads=8193)
    with pytest.raises(ValueError, match='signed int32 limit 2147483648'):
        mla_geometry(beyond, 1, 8, minimum_slots=0, latent=True, prefill_rows=512,
                     check_prefill_offsets=True)


@pytest.mark.parametrize('world', [1, 2, 3])
@pytest.mark.parametrize('setting', [None, '2048', '4096', '8192'])
def test_actual_startup_guard_precedes_weight_loading(tmp_path, monkeypatch, world, setting):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    if setting is None:
        monkeypatch.delenv('TF_GLM_PREFILL_ROWS', raising=False)
    else:
        monkeypatch.setenv('TF_GLM_PREFILL_ROWS', setting)
    monkeypatch.setenv('TF_GLM_VISION', '0')
    monkeypatch.setattr('tensorfold.families.glm5_next.cuda.LATENT', True)
    def loaded(*a, **kw):
        pytest.fail('weights must not load before the startup guard/admission')
    weights = SimpleNamespace(Config=SimpleNamespace(read=lambda _: SimpleNamespace(dense_limit=2051)), load=loaded)
    monkeypatch.setitem(sys.modules, 'tensorfold.families.glm5_next.cuda.weights', weights)
    monkeypatch.setitem(sys.modules, 'tensorfold.families.glm5_next.cuda.decode', SimpleNamespace(Engine=None))
    checkpoint(tmp_path, MODEL, [])
    def admit(self, fn, model, context, explicit, torch, make_geometry, transform, **kw):
        make_geometry(MODEL)
        raise RuntimeError('admission reached with safe offsets')
    monkeypatch.setattr(GlmEngine, '_admit', admit)
    if setting == '8192' and world < 3:
        error, message = ValueError, rf'is unsafe for TP={world}'
    else:
        error, message = RuntimeError, 'admission reached with safe offsets'
    with pytest.raises(error, match=message):
        GlmEngine(tmp_path, rank=0, master='', port=0, world=world, context=262144,
                  comm=SimpleNamespace(barrier=lambda: None))


def test_reviewed_resume_counterexample_and_tp3_bound():
    # Dense resumed suffix: c=16, r=0 reaches int32 overflow for TP2, even though the cold case can fit.
    assert (16 * 8192) * 32 * 512 == 2**31
    assert 21 * 8192 * 22 * 512 - 1 == 1937768447 < 2**31


@pytest.mark.parametrize('world', [1, 2, 3])
@pytest.mark.parametrize('nested', [False, True])
def test_local_refusal_precedes_device_and_nccl_without_supplied_comm(tmp_path, monkeypatch, world, nested):
    torch = pytest.importorskip('torch')
    from tensorfold.cuda import comm
    events = []
    monkeypatch.setenv('TF_GLM_PREFILL_ROWS', '8192')
    monkeypatch.setenv('TF_GLM_VISION', '0')
    monkeypatch.setattr('tensorfold.families.glm5_next.cuda.LATENT', True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: events.append('device'))
    def nccl(*a, **kw):
        events.append('NCCL')
        return SimpleNamespace(barrier=lambda: events.append('barrier'))
    monkeypatch.setattr(comm, 'NCCL', nccl)
    def loaded(*a, **kw):
        pytest.fail('sentinel test must not load weights')
    weights = SimpleNamespace(Config=SimpleNamespace(read=lambda _: SimpleNamespace(dense_limit=2051)), load=loaded)
    monkeypatch.setitem(sys.modules, 'tensorfold.families.glm5_next.cuda.weights', weights)
    monkeypatch.setitem(sys.modules, 'tensorfold.families.glm5_next.cuda.decode', SimpleNamespace(Engine=None))
    checkpoint(tmp_path, {'text_config': MODEL} if nested else MODEL, [])
    def admit(self, fn, model, context, explicit, torch, make_geometry, transform, **kw):
        make_geometry(MODEL)  # the later admission check stays active too
        events.append('admission')
        raise RuntimeError('safe startup reached admission')
    monkeypatch.setattr(GlmEngine, '_admit', admit)
    if world < 3:
        with pytest.raises(ValueError, match=rf'TF_GLM_PREFILL_ROWS=8192 is unsafe for TP={world}'):
            GlmEngine(tmp_path, rank=0, master='', port=0, world=world, context=262144)
        assert events == []
    else:
        with pytest.raises(RuntimeError, match='safe startup reached admission'):
            GlmEngine(tmp_path, rank=0, master='', port=0, world=world, context=262144)
        assert events == ['device', 'NCCL', 'barrier', 'admission']
