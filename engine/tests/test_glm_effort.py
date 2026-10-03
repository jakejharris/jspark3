"""GLM-5.3-Flash on CUDA reads the OpenAI reasoning_effort: its chat template knows the levels low and high (any
other value renders as max), and "none" turns thinking off."""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda.app import with_effort


@pytest.mark.parametrize("effort, want", [("minimal", "low"), ("low", "low"), ("medium", "high"), ("High", "high"),
                                          ("xhigh", "max"), ("max", "max")])
def test_levels_become_the_templates_levels(effort, want):
    assert with_effort({"reasoning_effort": effort})["chat_template_kwargs"] == {"reasoning_effort": want}


def test_none_turns_thinking_off():
    assert with_effort({"reasoning_effort": "none"})["chat_template_kwargs"] == {"enable_thinking": False}


def test_explicit_template_kwargs_win_and_unknown_values_pass():
    body = {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "max", "x": 1}}
    assert with_effort(body)["chat_template_kwargs"] == {"reasoning_effort": "max", "x": 1}
    for body in ({}, {"reasoning_effort": "ultra"}, {"reasoning_effort": 3}):
        assert with_effort(body) is body
