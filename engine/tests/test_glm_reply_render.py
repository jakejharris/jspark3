"""The candidate uses the API's normalization, and rejects changed prefix bytes."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.server import ChatTemplate
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tensorfold.families.glm5_next.cuda.reply_prefill import ReplyConfig, candidate, enqueue
from test_cuda_request_policy import TextTokenizer


@pytest.fixture
def renderer(tmp_path):
    # A small template with the same important inputs; the real checkpoint is
    # separately exercised with tools/check_glm_reply_prefill.py and captured pairs.
    template = ("{{ reasoning_effort|default('max') }}:{{ tools|tojson }};"
                "{% for m in messages %}{{ m.role }}:"
                "{% if m.role == 'assistant' %}<think>{{ m.reasoning_content|default('') }}</think>{% endif %}"
                "{{ m.content }}{% if m.tool_calls %}{{ m.tool_calls|tojson }}{% endif %}"
                "{% endfor %}{% if add_generation_prompt %}assistant:<think>{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = GlmApp.__new__(GlmApp)
    app.template, app.tok = ChatTemplate(tmp_path), TextTokenizer()
    app.default_thinking, app.max_tokens = True, 1
    app.engine = SimpleNamespace(limit=262144, request=SimpleNamespace(reply_prefill_epoch=1))
    body = {"messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "medium"}
    result = {"content": "World", "reasoning": "Consider it.", "finish": "stop", "calls": None,
              "stats": {"drafts": True, "policy": "fc7:0.3"}}
    return app, body, result


def test_normalized_assistant_is_a_strict_next_prompt_prefix(renderer):
    app, body, result = renderer
    prepared = app._prepare(body, True)
    ids, reason = candidate(app, body, prepared, result)
    assert reason == "candidate" and ids[:len(prepared.prompt)] == prepared.prompt
    actual = {**body, "messages": body["messages"] + [
        {"role": "assistant", "content": result["content"], "reasoning_content": result["reasoning"]},
        {"role": "user", "content": "Continue"}]}
    assert ids == app._prepare(actual, True).prompt[:len(ids)]
    changed = copy.deepcopy(actual)
    changed["messages"][1].pop("reasoning_content")
    assert ids != app._prepare(changed, True).prompt[:len(ids)]


@pytest.mark.parametrize("change", ["effort", "tools", "body", "limit", "image", "length", "no-cache"])
def test_candidate_negative_controls(renderer, change):
    app, body, result = renderer
    prepared = app._prepare(body, True)
    if change == "effort":
        body["reasoning_effort"] = "low"
    elif change == "tools":
        prepared.tools = [{"type": "function", "function": {"name": "changed", "parameters": {}}}]
    elif change == "body":
        body["messages"][0]["content"] = "Forked"
    elif change == "limit":
        app.engine.limit = len(prepared.prompt) + 1
    elif change == "image":
        prepared.images = [object()]
    elif change == "length":
        result["finish"] = "length"
    else:
        result["stats"]["drafts"] = False
    assert candidate(app, body, prepared, result)[0] is None


@pytest.mark.parametrize("flag", ["yes", "2", ""])
def test_config_rejects_bad_flag(monkeypatch, flag):
    monkeypatch.setenv("TF_GLM_REPLY_PREFILL", flag)
    with pytest.raises(ValueError, match="0 or 1"):
        ReplyConfig.from_env(8)


def test_config_default_off_and_memory_reserve(monkeypatch):
    monkeypatch.delenv("TF_GLM_REPLY_PREFILL", raising=False)
    monkeypatch.setenv("TF_GLM_REPLY_PREFILL_ROWS", "invalid-ignored-when-off")
    assert ReplyConfig.from_env(1).reserve == 0
    monkeypatch.setenv("TF_GLM_REPLY_PREFILL", "1")
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "0")
    with pytest.raises(ValueError, match="pooled serving"):
        ReplyConfig.from_env(8)
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    with pytest.raises(ValueError, match="pooled serving"):
        ReplyConfig.from_env(1)
    for rows in ("0", "257"):
        monkeypatch.setenv("TF_GLM_REPLY_PREFILL_ROWS", rows)
        with pytest.raises(ValueError, match="between 1 and 256"):
            ReplyConfig.from_env(8)
    monkeypatch.setenv("TF_GLM_REPLY_PREFILL_ROWS", "128")
    config = ReplyConfig.from_env(8)
    assert config.rows == 128 and config.reserve == 64 * 2**20


def test_request_opt_out_and_startup_capability(renderer):
    app, body, _ = renderer
    assert app._check_fields({**body, "tf_reply_prefill": "false"}) == "tf_reply_prefill must be a boolean"
    assert "startup" in app._check_fields({**body, "tf_reply_prefill": True})
    app.engine.reply_prefill = True
    assert app._check_fields({**body, "tf_reply_prefill": True}) is None
    assert app._check_fields({**body, "tf_reply_prefill": False}) is None


@pytest.mark.parametrize("mode", ["queued", "busy", "stale", "race", "error"])
def test_enqueue_is_bounded_stale_safe_and_does_not_leak_private_errors(renderer, mode):
    app, body, result = renderer
    calls = []
    sched = SimpleNamespace(reply_render_lock=threading.Lock(), state_lock=threading.Lock(),
                            reply_epoch=1, failed=None)
    def submit(ids, **kwargs):
        calls.append((ids, kwargs))
        return mode != "race"
    sched.submit_reply_prefill = submit
    app.engine.scheduler = sched
    prepared = app._prepare(body, True)
    if mode == "busy":
        sched.reply_render_lock.acquire()
    elif mode == "stale":
        sched.reply_epoch = 2
    elif mode == "error":
        def fail(*a, **k):
            raise ValueError("private conversation content")
        app.template.render = fail
    enqueue(app, body, prepared, result)
    status = {"queued": "queued", "busy": "renderer-busy", "stale": "foreground-arrived",
              "race": "foreground-arrived", "error": "render-error"}[mode]
    assert result["stats"]["reply_prefill"]["status"] == status
    assert "private" not in str(result["stats"])
    assert bool(calls) == (mode in ("queued", "race"))
    if mode != "busy":
        assert sched.reply_render_lock.acquire(blocking=False)
