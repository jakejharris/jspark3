"""OpenAI server for the CUDA engines: a family's ``cuda_engine`` gives ``eos``, ``generate`` and ``follow``."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server.errors import ContextLengthError, RequestError
from tensorfold.server.http import Server
from tensorfold.server.messages import (_normalize_tool_call_arguments, late_system_role, normalize_messages,
                                        validate_modalities)
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.engine.call_gate import CallGate, call_format, generate_gated
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

from tensorfold.cuda.reply_text import StreamDecoder, hide_tool_calls, parse_tool_calls
from tensorfold.server.text import split_thinking


MAX_BODY_BYTES = 192 * 2 ** 20
BODY_BUDGET_BYTES = 2048 * 2 ** 20              # conservative JSON/body charge, held through the reply
BODY_MIN_CHARGE = 32 * 2 ** 20                  # at most 64 retained request bodies, including small ones
SOCKET_IO_SECONDS = 10.0                       # per socket operation, never a generation deadline
BODY_READ_GRACE_SECONDS = 10.0
BODY_READ_BYTES_PER_SECOND = 2 ** 20            # largest body: 192 s transfer allowance + 10 s grace
_JSON_MARK = re.compile(r'["{}\[\],:]')
_body_lock = threading.Lock()
_body_bytes = 0


class BodyTooLarge(RequestError):
    status = 413


class BodyBusy(RequestError):
    status = 503


class BodyReadTimeout(RequestError):
    status = 408


def _json_structure(raw):
    """Count structure outside strings before JSON allocates containers; skip long strings with str.find."""
    if isinstance(raw, bytes):
        raw = raw.decode(json.detect_encoding(raw), errors="surrogatepass")
    pos = count = depth = 0
    while mark := _JSON_MARK.search(raw, pos):
        pos = mark.end()
        char = raw[mark.start()]
        if char == '"':
            while True:
                end = raw.find('"', pos)
                if end < 0:
                    return count               # json.loads supplies the syntax error
                back = end - 1
                while back >= pos and raw[back] == "\\":
                    back -= 1
                pos = end + 1
                if (end - 1 - back) % 2 == 0:
                    break
        else:
            count += 1
            depth += (1 if char in "{[" else -1 if char in "}]" else 0)
            if count > 1_048_576 or depth > 64:
                raise BodyTooLarge("the request body exceeds JSON structure limits (1048576 markers, 64 levels)")
    return count


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg["chat_template"]

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        self.late_system = late_system_role(
            lambda messages: self.template.render(**self.specials, messages=messages, add_generation_prompt=False))

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, media: bool = False) -> str:
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system,
                                                                     media=media))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)


# -- HTTP ------------------------------------------------------------------------------------

@dataclass(slots=True)
class PreparedRequest:
    prompt: list[int]
    max_tokens: int
    tools: list[dict[str, Any]]
    thinking: bool
    images: list | None = None      # prepared images in prompt order (GLM: ``vision.Image``), else None

    def close(self):
        images, self.images = self.images, None
        close = getattr(images, "close", None)
        if close is not None:
            close()


def _native_context(model_dir: Path) -> int:
    path = model_dir / "config.json"
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


import os as _os

# JSpark3 measurement adapter: TF_V2_PARITY=1 adds /tokenize, /metrics, token-id prompts,
# and always-parsed reasoning, so mmastrac's gate scripts and RigMark run unmodified against this server.
V2_PARITY = _os.environ.get("TF_V2_PARITY", "0") == "1"
V2_COUNTERS = {"requests_total": 0, "running": 0, "prompt_tokens_total": 0, "generation_tokens_total": 0,
               "prefix_cache_queries_total": 0, "prefix_cache_hits_total": 0}
V2_START = __import__("time").time()


class App:
    """Serve one engine with sampling and reply-length defaults for requests that omit them."""

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096,
                 context_window: int | None = None):
        from tokenizers import Tokenizer

        self.engine = engine
        self.served = served
        self.root = str(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.native_context_window = _native_context(model_dir)
        self.context_window = self.native_context_window if context_window is None else int(context_window)
        if self.context_window < 0:
            raise ValueError("context_window must be 0 or a positive token count")
        self.lock = threading.Lock()

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        import inspect

        if not isinstance(body, dict):
            return "the request body must be a JSON object"
        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        return None

    def _engine_capacity(self) -> int | None:
        capacities = []
        for name in ("context_window", "limit"):
            limit = getattr(self.engine, name, None)
            if isinstance(limit, int):
                capacities.append(max(0, limit))
        return min(capacities) if capacities else None

    def _restart(self, need: int, ranks: str = "") -> str:
        """A larger ``--context`` to restart with, only where the startup admission would accept it."""

        largest = (getattr(self.engine, "capacity_plan", None) or {}).get("largest_window")
        if largest is None or need > largest:
            return ""
        return f", or restart{ranks} with --context {need} or more (this memory admits up to {largest})"

    def _context_limit(self) -> int | None:
        limits = [self.context_window] if self.context_window > 0 else []
        capacity = self._engine_capacity()
        if capacity is not None:
            limits.append(capacity)
        return min(limits) if limits else None

    @property
    def effective_context_window(self) -> int | None:
        """Safe prompt-plus-reply capacity; None is unlimited, while zero refuses every prompt."""

        return self._context_limit()

    def _requested_tokens(self, body: dict[str, Any]) -> int:
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is not None:
                try:
                    int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RequestError(f"{name} must be an integer token count") from exc
        return max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens))

    def _prepare(self, body: dict[str, Any], chat: bool, media: bool = False) -> PreparedRequest:
        validate_modalities(body)
        ToolCallPolicy(body)
        max_tokens = self._requested_tokens(body)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
        if chat:
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            text = self.template.render(body["messages"], tools=tools, enable_thinking=thinking, extra=kwargs,
                                        **({"media": True} if media else {}))
            if V2_PARITY:
                thinking = True     # the v2 template always opens <think>; parse it as vLLM's patched glm45 parser does
            prompt = self.tok.encode(text, add_special_tokens=False).ids
        else:
            text = body.get("prompt")
            if V2_PARITY and isinstance(text, list) and text and all(type(t) is int for t in text):
                prompt = list(text)     # token ids, as vLLM's /v1/completions takes them (RigMark's prefill rows)
            elif not isinstance(text, str):
                raise RequestError("prompt must be a string")
            else:
                prompt = self.tok.encode(text, add_special_tokens=False).ids
        if not prompt:
            raise RequestError("rendered prompt is empty")
        return PreparedRequest(prompt, max_tokens, tools, thinking)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Why the request cannot run, or None; rendered before a stream's headers are sent."""

        problem = self._check_fields(body)
        if problem:
            return problem
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        limit = self._context_limit()
        if limit is not None and len(prepared.prompt) >= limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            native = f" (model window: {self.native_context_window} tokens)" if self.native_context_window else ""
            return (f"This server's maximum context length is {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and leaves no room for a reply in "
                    f"the server's {limit}-token {kind}{native}; shorten the prompt"
                    f"{self._restart(len(prepared.prompt) + 1)}")
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"This server's maximum context length is {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and requests {prepared.max_tokens} "
                    f"reply tokens, exceeding the server's {limit}-token {kind}; reduce the prompt or reply "
                    f"length{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
        return None

    def prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        problem = self._check_fields(body)
        if problem:
            raise RequestError(problem)
        prepared = self._prepare(body, chat)
        try:
            problem = self.check(body, prepared=prepared)
            if problem:
                context_problem = problem.startswith(("This server's maximum context length is",
                                                     "This model's maximum context length is",
                                                     "this request needs a "))
                raise (ContextLengthError if context_problem else RequestError)(problem)
            limit = self._context_limit()
            if limit is not None:
                prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))
            return prepared
        except BaseException:
            prepared.close()
            raise

    @contextmanager
    def _serving_lock(self, cancelled, images):
        if getattr(self.engine, "concurrent", False):
            yield
        elif images and cancelled is not None:
            # A disconnected image waiter must release its tensors without waiting for the active generation.
            while True:
                if cancelled():
                    raise RequestCancelled("the client left before the request started")
                if self.lock.acquire(timeout=0.05):
                    break
            try:
                yield
            finally:
                self.lock.release()
        else:
            with self.lock:
                yield

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy decoding."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        temp = float(body["temperature"] if body.get("temperature") is not None else self.sampling["temperature"])
        if temp <= 0:
            return None
        seed = body.get("seed")
        top_k = body["top_k"] if body.get("top_k") is not None else self.sampling["top_k"]
        top_p = body["top_p"] if body.get("top_p") is not None else self.sampling["top_p"]
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``."""

        prepared = prepared if prepared is not None else self.prepare(body, chat)
        prompt, max_tokens = prepared.prompt, prepared.max_tokens
        tools, thinking = prepared.tools, prepared.thinking
        policy = ToolCallPolicy(body)
        sampling = self.sampling_for(body, prompt)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False}
        failed: list[Exception] = []
        stream = StreamDecoder(self.tok, tuple(self.engine.eos))
        calls_stream = ToolCallStreamer(tools) if chat and tools and not policy.single else None
        answer_raw = [""]

        def visible(finished: bool) -> tuple[str, str]:
            raw = stream.final() if finished else stream.text
            if chat and thinking:
                reasoning, answer = split_thinking(raw, finished=finished)
            else:
                reasoning, answer = "", raw
            answer_raw[0] = answer
            if tools:
                answer = (policy.content(answer, finished=finished) if policy.single
                          else hide_tool_calls(answer, finished=finished))
            return reasoning, answer

        def on_tokens(new: list[int]) -> bool:
            # True stops the engine after this round; engines that finish on both ranks keep calling and get True
            if stopped["client"] or failed:
                return True
            try:
                out.extend(new)
                stream.add(new)
                reasoning, answer = visible(False)
                delta: dict[str, Any] = {}
                if len(reasoning) > sent["reasoning"]:
                    delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                    sent["reasoning"] = len(reasoning)
                if len(answer) > sent["content"]:
                    delta["content"] = answer[sent["content"]:]
                    sent["content"] = len(answer)
                if delta and not emit(delta):
                    stopped["client"] = True
                if calls_stream is not None and not stopped["client"]:
                    for call_delta in calls_stream.feed(answer_raw[0]):
                        if not emit(call_delta):
                            stopped["client"] = True
                            break
                if not stopped["client"] and cancelled is not None and cancelled():     # every round, with or without new text
                    stopped["client"] = True
            except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
                failed.append(exc)
                return True
            return stopped["client"]

        draft = body.get("draft", True) is not False
        gate = self._call_gate(prompt, tools) if tools and tool_choice_requires_call(body.get("tool_choice")) else None

        def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
            return self.engine.generate(ids, count, sampling, feed, **({} if draft else {"draft": False}))

        # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
        with self._serving_lock(cancelled, bool(prepared.images)):
            if cancelled is not None and cancelled():                # the client left while this request waited
                raise RequestCancelled("the client left before the request started")
            V2_COUNTERS["running"] += 1
            try:
                stats = generate_gated(generate, prompt, max_tokens, gate, on_tokens)
            finally:
                V2_COUNTERS["running"] -= 1
                V2_COUNTERS["requests_total"] += 1
                V2_COUNTERS["prompt_tokens_total"] += len(prompt)
                V2_COUNTERS["generation_tokens_total"] += len(out)
        V2_COUNTERS["prefix_cache_queries_total"] += len(prompt)
        V2_COUNTERS["prefix_cache_hits_total"] += int((stats or {}).get("cached") or 0)
        if failed:
            raise failed[0]
        if stopped["client"]:                                        # as the Mac server: nothing more is written
            raise RequestCancelled("the client left during the reply")
        V2_COUNTERS["request_success_total"] = V2_COUNTERS.get("request_success_total", 0) + 1
        st = stats or {}
        if st.get("drafts") and isinstance(st.get("rounds"), int) and len(out) > 1:
            # a round commits one token plus its accepted drafts; the first token comes from prefill
            V2_COUNTERS["spec_decode_num_drafts_total"] = V2_COUNTERS.get("spec_decode_num_drafts_total", 0) + st["rounds"]
            V2_COUNTERS["spec_decode_num_accepted_tokens_total"] = (
                V2_COUNTERS.get("spec_decode_num_accepted_tokens_total", 0) + max(0, len(out) - 1 - st["rounds"]))
        stats = {**(stats or {}), "token_sha": token_sha(out)}
        reasoning, answer = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final["reasoning_content"] = reasoning[sent["reasoning"]:]
        raw_answer = split_thinking(self.tok.decode([t for t in out if t not in self.engine.eos],
                                                    skip_special_tokens=False), finished=True)[1] \
            if chat and thinking else self.tok.decode([t for t in out if t not in self.engine.eos],
                                                      skip_special_tokens=False)
        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
        streamed = min(len(calls or []), len(calls_stream.ids)) if calls_stream is not None else 0
        for i in range(streamed):
            calls[i]["id"] = calls_stream.ids[i]
        content = policy.content(content) if tools else content
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        finish = "tool_calls" if calls else ("stop" if out and out[-1] in self.engine.eos else "length")
        s = stats or {}
        prefill_s, decode_s = float(s.get("prefill_s") or 0.0), float(s.get("decode_s") or 0.0)
        rate = f"{(len(out) - 1) / decode_s:.1f} tok/s" if decode_s > 0 and len(out) > 1 else "-"
        seen = f", {len(prepared.images)} images" if prepared.images else ""
        print(f"[tensorfold] request: {len(prompt)} prompt tokens ({int(s.get('cached') or 0)} resumed{seen}), prefill "
              f"{prefill_s:.1f}s; {len(out)} reply tokens in {decode_s:.1f}s ({rate}), {len(reasoning)} chars of "
              f"thinking, finish {finish}{', tools ' + ','.join(c['function']['name'] for c in calls) if calls else ''}",
              flush=True)
        if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
            stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
        return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "stats": stats,
                "calls_streamed": streamed}

    def _call_gate(self, prompt: list[int], tools: list[dict[str, Any]]) -> CallGate:
        """The gate a required tool call needs, from this template's call markup and the rendered prompt."""

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            try:
                text = self.template.render(probe, tools=None, enable_thinking=False)
            except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                text = ""
            openers = [o for o in ("<tool_call>", "<|tool_call>") if self.tok.token_to_id(o) is not None]
            self._form = call_format(text, "tfprobe_fn", openers) or ((openers[0], None, None) if openers else None)
        if self._form is None:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opener, lead, tail = self._form
        eos = set(self.engine.eos)

        def text(token: int) -> str:
            return self.tok.decode([token], skip_special_tokens=False)

        def blank(token: int) -> bool:
            return token not in eos and not text(token).strip()

        think = [-1 if self.tok.token_to_id(t) is None else self.tok.token_to_id(t) for t in ("<think>", "</think>")]
        names = [str((t.get("function") or t).get("name") or "") for t in tools] if lead is not None else []
        return CallGate.after_prompt(prompt, self.tok.token_to_id(opener), blank, think_open=think[0],
                                     think_end=think[1], text=text, lead=lead or "", names=names, tail=tail or "",
                                     encode=lambda t: list(self.tok.encode(t, add_special_tokens=False).ids))


def token_sha(tokens: list[int]) -> str:
    """A reply's token ids, hashed as the Mac server does: drafted and ``"draft": false`` replies must match."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = SOCKET_IO_SECONDS            # StreamRequestHandler applies this before reading headers

        def _io_timeout(self, seconds):
            # Memory-stream host fixtures have no socket; production connections always provide settimeout.
            setter = getattr(self.connection, "settimeout", None)
            if setter is not None:
                setter(seconds)

        def _read_body(self, size):
            deadline = time.monotonic() + BODY_READ_GRACE_SECONDS + size / BODY_READ_BYTES_PER_SECOND
            data = bytearray()
            try:
                while len(data) < size:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise BodyReadTimeout("the request body upload timed out")
                    self._io_timeout(min(SOCKET_IO_SECONDS, left))
                    try:
                        # read1 performs at most one raw read. BufferedReader.read could keep filling itself
                        # while a trickle resets the socket's idle timeout, hiding the absolute deadline.
                        chunk = self.rfile.read1(min(65536, size - len(data)))
                    except TimeoutError:
                        raise BodyReadTimeout("the request body upload timed out") from None
                    if time.monotonic() >= deadline:
                        raise BodyReadTimeout("the request body upload timed out")
                    if not chunk:
                        raise RequestError("the request body ended before Content-Length")
                    data.extend(chunk)
                return bytes(data)
            finally:
                self._io_timeout(SOCKET_IO_SECONDS)  # independent read/write waits after upload, not elapsed compute

        def log_message(self, fmt, *args):  # quiet
            pass

        def end_headers(self):
            super().end_headers()
            self._headers_sent = True

        def _error(self, message: str, status: int = 400, code: str | None = None):
            error = {"error": {"message": message, "type": "server_error" if status >= 500 else "invalid_request_error"}}
            if code:
                error["error"]["code"] = code
            if getattr(self, "_stream_started", False):
                try:
                    self.wfile.write(f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True
            elif not getattr(self, "_headers_sent", False):
                self.close_connection = True
                self._json(status, error)
            else:
                self.close_connection = True

        def _guard(self, operation):
            global _body_bytes
            self._stream_started = False
            self._headers_sent = False
            self._prepared = None
            self._body_charge = 0
            try:
                self._io_timeout(SOCKET_IO_SECONDS)
                return operation()
            except (RequestCancelled, BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except RequestError as exc:
                self._error(str(exc), getattr(exc, "status", 400), getattr(exc, "code", None))
            except Exception as exc:  # noqa: BLE001 - keep internal failures out of client error bodies
                print(f"[tensorfold] HTTP handler failed: {type(exc).__name__}", file=sys.stderr, flush=True)
                self._error("internal server error", 500)
            finally:
                close = getattr(self._prepared, "close", None)
                if close is not None:
                    close()
                self._prepared = None
                with _body_lock:
                    _body_bytes -= self._body_charge
                self._body_charge = 0

        def _body(self):
            global _body_bytes
            try:
                size = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise RequestError("Content-Length must be a non-negative integer") from None
            if size < 0:
                raise RequestError("Content-Length must be a non-negative integer")
            if self.headers.get("Transfer-Encoding"):
                raise RequestError("Transfer-Encoding is not supported; send Content-Length")
            if size > MAX_BODY_BYTES:
                raise BodyTooLarge("the request body exceeds 192 MiB")
            # Raw bytes + worst-width decoded input + retained strings, then charge containers separately.
            charge = max(BODY_MIN_CHARGE, 9 * size)
            with _body_lock:
                if _body_bytes + charge > BODY_BUDGET_BYTES:
                    raise BodyBusy("request body budget is busy (2048 MiB); retry later")
                _body_bytes += charge
                self._body_charge += charge
            raw = self._read_body(size)
            try:
                text = raw.decode(json.detect_encoding(raw), errors="surrogatepass")
            except UnicodeDecodeError:
                raise RequestError("the request body is not JSON") from None
            charge = max(BODY_MIN_CHARGE, 9 * size + 128 * _json_structure(text)) - self._body_charge
            with _body_lock:
                if _body_bytes + charge > BODY_BUDGET_BYTES:
                    raise BodyBusy("request body budget is busy (2048 MiB); retry later")
                _body_bytes += charge
                self._body_charge += charge
            try:
                body = json.loads(text or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RequestError("the request body is not JSON") from None
            if not isinstance(body, dict):
                raise RequestError("the request body must be a JSON object")
            return body

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                # Portions adapted from TensorFold 50dfe38 (Apache-2.0); modified by JSpark3.
                # Moved from cuda/http.py and reduced to these header lines; see NOTICE.
                if self.close_connection:
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except OSError:          # the client has gone
                self.close_connection = True

        def do_GET(self):
            return self._guard(self._get)

        def _get(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                model = {"id": app.served, "object": "model", "owned_by": "tensorfold"}
                window = app.effective_context_window
                if window:                   # the served window, as vLLM reports it, so clients size their history
                    model["max_model_len"] = model["context_window"] = int(window)
                if V2_PARITY:                # vLLM's root: the path the engine loaded the checkpoint from
                    model["root"] = app.root
                self._json(200, {"object": "list", "data": [model]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._json(200, {"ok": True})
            elif V2_PARITY and self.path.rstrip("/") == "/metrics":
                c = dict(V2_COUNTERS)
                active = getattr(app.engine, "running_requests", None)
                if getattr(app.engine, "concurrent", False) and active is not None:
                    c["running"] = active()     # admitted slots, including their in-progress prefill
                run, inflight = c["running"], c.get("inflight", 0)
                text = ("".join(f"tensorfold:{k} {v}\n" for k, v in c.items())
                        # vLLM-named aliases carrying TensorFold's own values, so vLLM-shaped quiet checks read them.
                        # TensorFold has no preemption: its count is 0 by construction.
                        + f"vllm:num_requests_running {run}\nvllm:num_requests_waiting {max(0, inflight - run)}\n"
                        + "vllm:num_preemptions_total 0\n"
                        + f"vllm:request_success_total {c.get('request_success_total', 0)}\n"
                        # drafted requests only: verify rounds, and tokens committed beyond one a round.
                        # Proposed-draft counts are not in TensorFold's stats, so no num_draft_tokens is exported.
                        + f"vllm:spec_decode_num_drafts_total {c.get('spec_decode_num_drafts_total', 0)}\n"
                        + f"vllm:spec_decode_num_accepted_tokens_total {c.get('spec_decode_num_accepted_tokens_total', 0)}\n"
                        + f"vllm:prefix_cache_queries_total {c['prefix_cache_queries_total']}\n"
                        + f"vllm:prefix_cache_hits_total {c['prefix_cache_hits_total']}\n"
                        + f"vllm:prompt_tokens_total {c['prompt_tokens_total']}\n"
                        + f"vllm:generation_tokens_total {c['generation_tokens_total']}\n"
                        + f"process_start_time_seconds {V2_START:.2f}\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)
            else:
                self._error("not found", 404)

        def _tokenize(self):
            body = self._body()
            if isinstance(body.get("messages"), list):
                kwargs = dict(body.get("chat_template_kwargs") or {})
                thinking = bool(kwargs.pop("enable_thinking", app.default_thinking))
                text = app.template.render(body["messages"], tools=body.get("tools"), enable_thinking=thinking,
                                           extra=kwargs)
            else:
                text = body.get("prompt")
            if not isinstance(text, str):
                return self._error("prompt must be a string")
            ids = app.tok.encode(text, add_special_tokens=bool(body.get("add_special_tokens", False))).ids
            window = app.effective_context_window
            self._json(200, {"count": len(ids), "max_model_len": int(window) if window else None, "tokens": ids})

        def _quality_score(self):
            import ipaddress
            from tensorfold.families.glm5_next.cuda.quality_score import validate_request

            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                return self._error("quality scoring is local-only", 403)
            body = self._body()
            try:
                vocab = min(app.tok.get_vocab_size(with_added_tokens=True), app.engine.w.cfg.vocab)
                ids, start = validate_request(body, app.served, vocab, app.engine.limit)
            except ValueError as exc:
                raise RequestError(str(exc)) from None
            with app._serving_lock(None, False):
                nll = app.engine.score(ids, start)
            self._json(200, {"token_ids": ids, "score_start": start, "nll": nll})

        def do_POST(self):
            return self._guard(self._dispatch_post)

        def _dispatch_post(self):
            if self.path.rstrip("/") == "/v1/quality/score":
                if not getattr(app.engine, "quality_score_enabled", False):
                    return self._error("quality scoring is off", 404)
                return self._quality_score()
            if V2_PARITY and self.path.rstrip("/") == "/tokenize":
                return self._tokenize()
            V2_COUNTERS["inflight"] = V2_COUNTERS.get("inflight", 0) + 1
            try:
                return self._post()
            finally:
                V2_COUNTERS["inflight"] -= 1

        def _post(self):
            chat = self.path.rstrip("/").endswith("/chat/completions")
            if not chat and not self.path.rstrip("/").endswith("/completions"):
                return self._error("not found", 404)
            body = self._body()
            prepared = app.prepare(body, chat)
            self._prepared = prepared
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            stream = bool(body.get("stream"))
            kind = "chat.completion.chunk" if chat else "text_completion"
            gone = socket_cancellation(self.connection)          # the Mac server's check: the client has closed
            write_failed = False
            cancelled = lambda: write_failed or gone.cancelled  # noqa: E731

            def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                if chat:
                    return {"id": rid, "object": kind, "created": created, "model": app.served,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                return {"id": rid, "object": kind, "created": created, "model": app.served,
                        "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self._stream_started = True

                def emit(delta: dict[str, Any]) -> bool:
                    nonlocal write_failed
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk(delta))}\n\n".encode())
                        self.wfile.flush()
                        return True
                    except OSError:             # reset, broken pipe, timed out, host unreachable: the client has gone
                        write_failed = True
                        self.close_connection = True
                        return False

                if chat and not emit({"role": "assistant"}):
                    raise RequestCancelled("the client stopped reading before generation")
                try:
                    result = app.run(body, chat, emit, prepared=prepared, cancelled=cancelled)
                except RequestCancelled:
                    self.close_connection = True
                    return
                if result["final"] and not emit(result["final"]):
                    return
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        if i < result["calls_streamed"]:
                            continue
                        if not emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                                     "function": {"name": call["function"]["name"],
                                                                  "arguments": call["function"]["arguments"]}}]}):
                            return
                end = chunk({}, result["finish"])
                end["tensorfold"] = result["stats"]
                if "pass_economics" in result["stats"]:
                    end["pass_economics"] = result["stats"]["pass_economics"]
                usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                         "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}
                if (body.get("stream_options") or {}).get("include_usage"):
                    end["usage"] = usage
                try:
                    self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True
                return
            try:
                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)
            except RequestCancelled:
                self.close_connection = True
                return
            usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                     "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}
            if chat:
                message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
                if result["reasoning"]:
                    message["reasoning_content"] = result["reasoning"]
                if result["calls"]:
                    message["tool_calls"] = result["calls"]
                payload = {"id": rid, "object": "chat.completion", "created": created, "model": app.served,
                           "choices": [{"index": 0, "message": message, "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
                if "pass_economics" in result["stats"]:
                    payload["pass_economics"] = result["stats"]["pass_economics"]
            else:
                payload = {"id": rid, "object": "text_completion", "created": created, "model": app.served,
                           "choices": [{"index": 0, "text": result["content"], "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
            self._json(200, payload)

    return Handler


def serve(app: App, host: str, port: int) -> None:
    """Serve until interrupted (SIGTERM included)."""

    import signal

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    server = Server((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
