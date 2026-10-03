"""Prefill CPU control-flow and allocation gates; GPU arithmetic parity lives in cuda/test_glm_p1_prefill.py."""

import importlib
import runpy
from types import SimpleNamespace

import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from test_glm_prefill_rows import TEXT

FLAGS = ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "DEAD_WORK")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setattr(p1, flag, False)


def test_flags_default_off_and_independent(monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv("TF_GLM_P1_" + flag, raising=False)
    assert all(not runpy.run_path(p1.__file__)[f] for f in FLAGS)
    for flag in FLAGS:
        monkeypatch.setenv("TF_GLM_P1_" + flag, "1")
        values = runpy.run_path(p1.__file__)
        assert {f for f in FLAGS if values[f]} == {flag}
        monkeypatch.setenv("TF_GLM_P1_" + flag, "0")


@pytest.mark.parametrize("flag", FLAGS)
def test_invalid_flag_fails_closed(monkeypatch, flag):
    monkeypatch.setenv("TF_GLM_P1_" + flag, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        runpy.run_path(p1.__file__)


@pytest.mark.parametrize("rows", [64, 512, 2048, 8192])
@pytest.mark.parametrize("capacity", [2560, 262144, 360448])
def test_geometry_savings_are_exact_and_independent(monkeypatch, rows, capacity):
    text = dict(TEXT, num_attention_heads=66)
    def estimate():
        return geometry.mla_geometry(text, 3, 8, latent=True, prefill_rows=rows).bytes_at(capacity)
    base = estimate()
    saved_rows = max(rows - 512, 0)
    chunks = (min(capacity, 2560) + rows + 511) // 512
    dense = saved_rows * chunks * 22 * (512 + 2) * 4
    selection = saved_rows * 4 * ((capacity + 3) // 4)
    monkeypatch.setattr(p1, "PROMPT_SCRATCH", True)
    assert estimate() == base - dense
    monkeypatch.setattr(p1, "SELECT_BLOCKS", True)
    assert estimate() == base - dense - selection
    monkeypatch.setattr(p1, "PROMPT_SCRATCH", False)
    assert estimate() == base - selection


@pytest.mark.parametrize("world", [1, 2, 3])
def test_blocked_partials_make_8192_row_offsets_safe(monkeypatch, world):
    from tensorfold.families.glm5_next.cuda.split import padded_config
    monkeypatch.setattr(p1, "PROMPT_SCRATCH", True)
    geometry.mla_geometry(padded_config(dict(TEXT, num_attention_heads=64), world), world, 8,
                          latent=True, prefill_rows=8192, check_prefill_offsets=True)


@pytest.mark.torch
@pytest.mark.parametrize("rows,step,pos", [(32, 512, 0), (513, 512, 0), (1057, 512, 2033), (63, 32, 2017)])
def test_dense_blocks_keep_original_head_kernel_and_positions(monkeypatch, rows, step, pos):
    torch = pytest.importorskip("torch")
    f = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    calls = []
    def attention(q, cache, at, scratch, *, scale, nch, out, hb):
        calls.append((len(q), int(at.item()), nch, hb))
        out.copy_(q + at + torch.arange(len(q))[:, None, None])
        return out
    monkeypatch.setattr(f.latent, "attention", attention)
    q = torch.arange(rows).view(rows, 1, 1)
    out = torch.empty_like(q)
    f.dense_attention(q, None, torch.tensor([pos]), SimpleNamespace(part_rows=step),
                      scale=1, nch=9, out=out)
    assert torch.equal(out, 2 * q + pos)
    assert calls == [(min(step, rows - a), pos + a, 9, f.latent.head_block(rows)) for a in range(0, rows, step)]


@pytest.mark.torch
@pytest.mark.parametrize("pos,rows", [(0, 2048), (0, 2052), (1986, 128), (1987, 128), (1988, 128), (2049, 128), (2050, 513), (2051, 513)])
@pytest.mark.parametrize("blocked", [False, True])
def test_resumed_dense_boundary_only_overwrites_sparse_rows(monkeypatch, pos, rows, blocked):
    torch = pytest.importorskip("torch")
    f = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    calls = []
    monkeypatch.setattr(p1, "PROMPT_SCRATCH", blocked)
    monkeypatch.setattr(f, "mm", lambda *a, **kw: None)
    monkeypatch.setattr(f.glue, "router", lambda *a: None)
    monkeypatch.setattr(f.latent, "absorb_q", lambda q, a, out: out)
    monkeypatch.setattr(f.latent, "latent_write", lambda *a: None)
    monkeypatch.setattr(f.sparse, "index_update", lambda *a: None)
    monkeypatch.setattr(f.sparse, "select_tokens", lambda *a, **kw: (None, None))
    def attention(q, cache, at, s, *, scale, nch, out, hb=None):
        calls.append((len(q), int(at.item()), nch, hb if hb is not None else f.latent.head_block(len(q))))
        out.copy_(torch.arange(int(at.item()), int(at.item()) + len(q)).view(-1, 1, 1))
        return out
    def sparse_attention(q, cache, tokens, counts, out, scale):
        mask = torch.arange(pos, pos + len(q)) >= 2051
        out[mask] = -torch.arange(pos, pos + len(q), dtype=out.dtype)[mask].view(-1, 1, 1)
    monkeypatch.setattr(f.latent, "attention", attention)
    monkeypatch.setattr(f.latent, "sparse_attention", sparse_attention)
    monkeypatch.setattr(f.latent, "expand_v", lambda ol, absorb, out: ol)
    monkeypatch.setattr(f.qmm, "group_sums", lambda *a: None)
    monkeypatch.setattr(f, "out_proj", lambda w, b, o, *a: o.clone())
    z = torch.zeros((rows, 1))
    s = SimpleNamespace(qa=z.view(rows, 1, 1), ol=torch.full((rows, 1, 1), 99999.), nch=10,
                        part_rows=512 if blocked else rows)
    b = SimpleNamespace(prefill=True, lat_s=s, normed=z, xs=z, ikr=torch.zeros((rows, 2)), igr=z,
                        q=z, lat=z, qr=z, xs_qr=z, qi=z, vn=z, xs_ao=z)
    a = SimpleNamespace(heads=1, index=SimpleNamespace(kw=None, gate=None, ln_w=None, ln_b=None, ape=None, qb=None),
                        absorb=None, o=None)
    w = SimpleNamespace(cfg=SimpleNamespace(qk_dim=1, v_dim=1, dense_limit=2051, index_dim=1))
    caches = [(None, None, torch.tensor([pos]), (None, None, torch.zeros((1026, 1))), pos, 0, rows)]
    baseline = f._dsa_latent(a, w, caches, b, rows, None)
    calls.clear()
    s.ol.fill_(99999.)
    monkeypatch.setattr(p1, "DENSE_ROWS", True)
    actual = f._dsa_latent(a, w, caches, b, rows, None)
    assert torch.equal(actual, baseline)
    dense_n = min(rows, max(0, 2051 - pos))
    assert sum(c[0] for c in calls) == dense_n
    assert all(c[3] == f.latent.head_block(rows) for c in calls)
    # Negative control: corrupt the first sparse output; the full output comparison must see it.
    if dense_n < rows:
        actual[dense_n] += 1
        assert not torch.equal(actual, baseline)


@pytest.mark.torch
def test_sliced_dead_chunk_stages_once_and_updates_every_layer(monkeypatch):
    torch = pytest.importorskip("torch")
    f = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    monkeypatch.setattr(p1, "DEAD_WORK", True)
    monkeypatch.setattr(f.tp3_probe, "recorder", None)
    events = []
    def embed(ids, weights, hidden, streams, out):
        events.append("embed")
        out.zero_()
    def layer(layer, w, segs, b, rows, *args, ffn=True):
        events.append((layer.index, ffn))
        segs[0][0].written.append(layer.index)
    monkeypatch.setattr(f.glue, "embed", embed)
    monkeypatch.setattr(f, "layer_forward", layer)
    monkeypatch.setattr(f.glue, "stream_mean", lambda *a: pytest.fail("dead output was consumed"))
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=1, streams=1), embed=None,
                        layers=[SimpleNamespace(index=i) for i in range(3)])
    b = SimpleNamespace(prefill=True, ids=torch.zeros(4), x=torch.empty(4, 1), overlay=None,
                        tap_at={0: [0], 2: [1]}, taps=[torch.full((4, 1), float("nan")) for _ in range(2)])
    st = SimpleNamespace(pos=0, written=[])
    kw = dict(logits=False, host_pos=0, tap_start=4, skip_final_ffn=True)
    assert f.compute(w, st, b, 4, layer_start=0, layer_stop=1, **kw) is None
    assert st.written == [0]
    assert f.compute(w, st, b, 4, layer_start=1, layer_stop=3, **kw) is None
    assert st.written == [0, 1, 2]
    assert events == ["embed", (0, True), (1, True), (2, False)]
