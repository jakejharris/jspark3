"""The native exact expert tile stays opt in, with strict per-boot flag parsing."""
import runpy

import pytest

from tensorfold.families.glm5_next.cuda import prefill_options


@pytest.mark.parametrize("suffix", ["128", "64"])
def test_expert_prefill_flag(monkeypatch, suffix):
    name = f"TF_GLM_EXPERT_PREFILL{suffix}"
    key = f"EXPERT_PREFILL{suffix}"
    monkeypatch.delenv(name, raising=False)
    assert not runpy.run_path(prefill_options.__file__)[key]
    monkeypatch.setenv(name, "1")
    assert runpy.run_path(prefill_options.__file__)[key]
    monkeypatch.setenv(name, "0")
    assert not runpy.run_path(prefill_options.__file__)[key]
    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError, match=name):
        runpy.run_path(prefill_options.__file__)
