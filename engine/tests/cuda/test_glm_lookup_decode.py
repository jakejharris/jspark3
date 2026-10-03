"""Copied drafts on the tiny synthetic checkpoint of test_glm_engine.py: rounds that verify proposed tokens (arm "p",
right in some rounds and wrong in others) between MTP rounds leave the reply serial decoding gives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import lookup  # noqa: E402

from test_glm_engine import _forget, _generate, engine  # noqa: E402,F401


class _Oracle:
    """Proposes the serial reply's next tokens every other round, wrong ones every third, nothing otherwise."""

    reply: list[int] = []

    def __init__(self, prompt) -> None:
        self.n, self.done, self.rounds = len(prompt), 0, 0

    def extend(self, tokens) -> None:
        self.done += len(tokens)

    def propose(self, room: int) -> list[int]:
        self.rounds += 1
        nxt = self.reply[self.done:self.done + min(room, 5)]
        if self.rounds % 3 == 0:
            return [(t + 1) % 990 for t in nxt][:2]
        return nxt if self.rounds % 2 == 0 else []


@pytest.mark.parametrize("sampling", [Sampling(4321, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_copied_drafts_equal_serial(engine, sampling, monkeypatch):  # noqa: F811
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, 990, size=90)]
    _forget(engine)
    serial, _ = _generate(engine, prompt, sampling, draft=False, tokens=60)
    _Oracle.reply = serial
    monkeypatch.setattr(lookup, "PromptLookup", _Oracle)
    _forget(engine)
    drafted, stats = _generate(engine, prompt, sampling, tokens=60)
    assert drafted == serial
    arms = stats.get("drafters", "")
    assert "p" in arms and "m" in arms
