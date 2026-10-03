"""Boot flag and dispatch: the experimental KDA layout cannot change decode."""

import runpy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import kda, kda_tiles, prefill_options


def test_kda_layout_flag_is_strict_default_off(monkeypatch):
    monkeypatch.delenv("TF_GLM_KDA_VALUE_TILES", raising=False)
    assert not runpy.run_path(prefill_options.__file__)["KDA_VALUE_TILES"]
    monkeypatch.setenv("TF_GLM_KDA_VALUE_TILES", "1")
    assert runpy.run_path(prefill_options.__file__)["KDA_VALUE_TILES"]
    monkeypatch.setenv("TF_GLM_KDA_VALUE_TILES", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        runpy.run_path(prefill_options.__file__)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows,wide", [(1, None), (8, None), (32, True), (63, None), (64, None), (1000, None)])
def test_only_wide_prefill_dispatch_changes(monkeypatch, enabled, rows, wide):
    calls = []
    stock = SimpleNamespace(chain=lambda *a: calls.append("chain"),
                            chain_wide=lambda *a: calls.append("wide"))
    candidate = SimpleNamespace(chain_wide=lambda *a: calls.append(("tiles", a[-1])))
    monkeypatch.setattr(kda, "_ext", lambda: stock)
    monkeypatch.setattr(kda_tiles, "_ext", lambda: candidate)
    monkeypatch.setattr(prefill_options, "KDA_VALUE_TILES", enabled)
    monkeypatch.setattr(kda, "_wide_scratch", lambda *a: (None, None))
    data = torch.empty((rows, 1))
    scratch = SimpleNamespace(out=data, k=None, v=None, g=None, b=None)
    out = kda.chain(data, 0, data, data, None, None, None, torch.empty(1), None, None,
                    1e-5, -5, rows, scratch, None, wide=wide)
    kind = ("tiles", kda_tiles.VARIANT) if enabled and rows >= 64 else "wide" if wide or rows >= 64 else "chain"
    assert calls == [kind]
    assert out.data_ptr() == data.data_ptr()
