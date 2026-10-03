"""Sparse-trim boot flag and the prefill/decode dispatch boundary."""

import runpy

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import latent, prefill_options


def test_sparse_trim_flag_is_strict_default_off(monkeypatch):
    monkeypatch.delenv("TF_GLM_SPARSE_TRIM", raising=False)
    assert not runpy.run_path(prefill_options.__file__)["SPARSE_TRIM"]
    monkeypatch.setenv("TF_GLM_SPARSE_TRIM", "1")
    assert runpy.run_path(prefill_options.__file__)["SPARSE_TRIM"]
    monkeypatch.setenv("TF_GLM_SPARSE_TRIM", "yes")
    with pytest.raises(ValueError, match="TF_GLM_SPARSE_TRIM must be 0 or 1"):
        runpy.run_path(prefill_options.__file__)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 63, 64, 65, 512])
@pytest.mark.parametrize("hb", [None, 16, 32])
def test_sparse_trim_preserves_small_rows_and_explicit_head_tiles(monkeypatch, enabled, rows, hb):
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kw: calls.append((self.name, grid, kw))

    for name in ("_sparse_chunks", "_sparse_chunks_trim", "_merge"):
        monkeypatch.setattr(latent, name, Kernel(name))
    monkeypatch.setattr(prefill_options, "SPARSE_TRIM", enabled)
    qa = torch.empty((rows, 22, 512), dtype=torch.bfloat16)
    latent.sparse_attention(qa, torch.empty((1, 512)), torch.empty((rows, 2051)),
                            torch.empty(rows), torch.empty_like(qa), .0625, hb=hb)
    name, grid, kw = calls[0]
    selected = enabled and rows >= 64
    heads = hb if hb is not None else 32 if rows >= 64 else 16
    assert name == ("_sparse_chunks_trim" if selected else "_sparse_chunks")
    assert grid == ((rows * ((22 + heads - 1) // heads) * 5,) if selected
                    else (rows, (22 + heads - 1) // heads, 5))
    assert (kw["HBT"], kw["KTT"], kw["CH"], kw["num_warps"], kw["num_stages"]) == (
        heads, 32, 512, 8, 3 if selected else 1)
    assert calls[1][0:2] == ("_merge", (rows, 22))
