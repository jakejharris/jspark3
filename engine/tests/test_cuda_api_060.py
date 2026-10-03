"""CPU checks for the Pi-facing CUDA response changes."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from tensorfold.cuda import server
from tensorfold.cuda.reply_text import parse_tool_calls
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tests.test_cuda_admission import Engine, http_server, post, request


@pytest.fixture
def model_dir(tmp_path):
    words = ["[UNK]", "answer", "header", "thinking", "tools"] + [f"w{i}" for i in range(40)]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}header"
                         "{% if enable_thinking %} thinking{% endif %}{% if tools %} tools{% endif %}"}))
    return tmp_path


TOOLS = [{"type": "function", "function": {"name": "question", "parameters": {
    "type": "object", "properties": {"questions": {"type": "array"}, "count": {"type": "integer"}}}}}]
GLM = ('<tool_call>question<arg_key>questions</arg_key><arg_value>[{"label":"Blue"}]</arg_value>'
       '<arg_key>count</arg_key><arg_value>2</arg_value></tool_call>')


@pytest.mark.parametrize("step", [1, 7, 1000])
def test_glm_fragments_match_complete_calls(step):
    text = GLM + GLM
    streamer = ToolCallStreamer(TOOLS)
    deltas = []
    for n in range(1, len(text) + 1, step):
        deltas.extend(streamer.feed(text[:n]))
    deltas.extend(streamer.feed(text))
    _, calls = parse_tool_calls(text, TOOLS)
    assert len(streamer.ids) == len(calls) == 2
    assert streamer.ids[0] != streamer.ids[1]
    for index, call in enumerate(calls):
        fragments = [d["tool_calls"][0] for d in deltas if d["tool_calls"][0]["index"] == index]
        assert fragments[0]["id"] == streamer.ids[index]
        args = "".join(d["function"].get("arguments", "") for d in fragments)
        assert json.loads(args) == json.loads(call["function"]["arguments"]) == {
            "questions": [{"label": "Blue"}], "count": 2}


def test_glm_python_spelled_typed_value():
    tools = [{"type": "function", "function": {"name": "switch", "parameters": {
        "type": "object", "properties": {"enabled": {"type": "boolean"}}}}}]
    text = "<tool_call>switch<arg_key>enabled</arg_key><arg_value>False</arg_value></tool_call>"
    _, calls = parse_tool_calls(text, tools)
    deltas = ToolCallStreamer(tools).feed(text)
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == json.loads(calls[0]["function"]["arguments"]) == {"enabled": False}


def test_glm_call_reaches_client_before_generation_ends(model_dir):
    class ProducingEngine(Engine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            for part in ("reasoning <tool_call>question</tool_call></think>" + GLM[:60],
                         GLM[60:] + "<tool_call>question</tool_call>"):
                on_tokens([ord(ch) for ch in part])
            on_tokens([0])
            return {}

    class CharacterTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    app = server.App(ProducingEngine(), model_dir, "test", context_window=0)
    body = {"messages": [{"role": "user", "content": "answer"}], "tools": TOOLS,
            "temperature": 0, "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": True}}
    prepared = app.prepare(body, True)
    app.tok = CharacterTokenizer()
    deltas = []

    def emit(delta):
        deltas.append(delta)
        return True

    result = app.run(body, True, emit, prepared=prepared)
    assert result["finish"] == "tool_calls"
    assert result["reasoning"].startswith("reasoning <tool_call>")
    assert result["calls_streamed"] == 2
    first = next(d for d in deltas if "tool_calls" in d)["tool_calls"][0]
    assert first["id"] == result["calls"][0]["id"]
    assert [call["id"] for call in result["calls"]] == [
        d["tool_calls"][0]["id"] for d in deltas if "id" in d.get("tool_calls", [{}])[0]]


def test_context_refusal_has_openai_code_before_stream_headers(model_dir):
    app = server.App(Engine(), model_dir, "test", context_window=12)
    with http_server(app) as port:
        for stream in (False, True):
            status, payload = post(port, request(12, True, max_tokens=1, stream=stream), True)
            assert status == 400
            error = json.loads(payload)["error"]
            assert error["code"] == "context_length_exceeded"
            assert error["message"].startswith("This server's maximum context length is 12 tokens")


@pytest.mark.parametrize("parity", [False, True])
def test_glm_context_refusal_has_openai_code(model_dir, monkeypatch, parity):
    monkeypatch.setattr(server, "V2_PARITY", parity)
    engine = Engine()
    engine.limit = 12
    app = GlmApp(engine, model_dir, "test", context_window=12)
    with http_server(app) as port:
        status, payload = post(port, request(12, True, max_tokens=1, stream=True), True)
        assert status == 400
        error = json.loads(payload)["error"]
        assert error["code"] == "context_length_exceeded"
