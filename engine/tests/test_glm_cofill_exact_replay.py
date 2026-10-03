"""Co-filled prompts keep their head row, so their exact repeats replay without prefill."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _drain, _stream  # noqa: F401
from test_glm_cofill import cofill_decoder  # noqa: F401


@pytest.mark.parametrize("sampled", [False, True])
def test_cofilled_prompts_replay_exactly(cofill_decoder, monkeypatch, sampled):
    mod, make, forwards = cofill_decoder
    monkeypatch.setattr(p1, "EXACT_REPLAY", True)
    d = make(4)
    prompts = [[3 + i] * (8 + i) + [11, 12] for i in range(3)]
    sampling = [Sampling(2**64 - i - 1, .73, 20, .92) if sampled else None for i in range(3)]
    cold = []
    for prompt, samp in zip(prompts, sampling):
        s = _stream(prompt, 9, sampling=samp)
        d.begin_admit(s)
        cold.append(s)
    before = len(forwards)
    d.prefill_step()
    assert len(forwards) == before + 1 and len(forwards[-1]) == 3       # one co-filled pass
    _drain(d)
    kept = {tuple(c.ids): c for c in d.cache}
    assert all(kept[tuple(p)].last_logits is not None and kept[tuple(p)].head_cols == 64 for p in prompts)
    for prompt, samp, ref in zip(prompts, sampling, cold):
        before = len(forwards)
        warm = _stream(prompt, 9, sampling=samp)
        d.begin_admit(warm)
        assert warm.exact_replay and warm.cached == len(prompt)
        d.prefill_step()
        assert len(forwards) == before                                  # no forward for the replay
        _drain(d)
        assert warm.out == ref.out


def test_cofill_keeps_no_head_row_with_the_flag_off(cofill_decoder, monkeypatch):
    mod, make, _ = cofill_decoder
    monkeypatch.setattr(p1, "EXACT_REPLAY", False)
    d = make(4)
    for i in range(2):
        d.begin_admit(_stream([5 + i] * 9, 3))
    d.prefill_step()
    _drain(d)
    assert d.cache and all(c.last_logits is None and c.last_hidden is None for c in d.cache)
