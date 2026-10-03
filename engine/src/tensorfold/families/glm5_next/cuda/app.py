"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.server.messages import image_url, message_images, normalize_messages


# OpenAI reasoning_effort -> GLM-5.3's template levels (it knows low and high; anything else renders as max)
EFFORTS = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


def with_effort(body: dict[str, Any]) -> dict[str, Any]:
    """The request with its ``reasoning_effort`` as template kwargs: "none" turns thinking off, a level picks the
    template's Reasoning Effort line; explicit chat_template_kwargs win. Without it the template thinks at max."""

    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        return body
    effort = effort.strip().lower()
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if effort == "none":
        kwargs.setdefault("enable_thinking", False)
    elif effort in EFFORTS:
        kwargs.setdefault("reasoning_effort", EFFORTS[effort])
    else:
        return body
    return {**body, "chat_template_kwargs": kwargs}


class ThinkingOffTemplate:
    """The checkpoint's chat template, rendered as GLM-5.3's thinking-off template renders it when thinking is off."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def render(self, messages, *, tools, enable_thinking, extra=None, media=False) -> str:
        text = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra, media=media)
        if not enable_thinking:
            text = text.replace("<|system|>Reasoning Effort: Max", "", 1)
            if text.endswith("<|assistant|><think>"):
                text += "</think>"
        return text


class GlmApp(App):
    def __init__(self, engine, model_dir, served: str, *, image_urls: bool = False, **kwargs: Any) -> None:
        self.image_urls = image_urls
        super().__init__(engine, model_dir, served, **kwargs)
        from tensorfold.cuda.server import V2_PARITY
        if not V2_PARITY:           # v2 parity: the v2 template maps thinking off to Low effort itself
            self.template = ThinkingOffTemplate(self.template)

    def prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        sessions = getattr(self.engine, "session_cache", None)
        if sessions is not None and sessions.store is not None:
            # Pause optional writer copies before rendering/tokenizing a large
            # warm request, not only when its GPU admission eventually starts.
            with sessions.store.write_gate.prepare():
                return super().prepare(body, chat)
        return super().prepare(body, chat)

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        problem = super()._check_fields(body)
        if problem:
            return problem
        if body.get("priority") not in (None, "interactive", "background"):
            return "priority must be interactive or background"
        if "tf_reply_prefill" in body and type(body["tf_reply_prefill"]) is not bool:
            return "tf_reply_prefill must be a boolean"
        if body.get("tf_reply_prefill") and not getattr(self.engine, "reply_prefill", False):
            return "tf_reply_prefill requires TF_GLM_REPLY_PREFILL=1 at startup"
        layers = body.get("prefill_slice_layers")
        if layers is not None and (type(layers) is not int or layers < 0):
            return "prefill_slice_layers must be a nonnegative integer"
        return None

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        """A chat with image parts: each image's placeholder becomes its token run, keyed by the image's hash."""

        body = with_effort(body)
        messages = body.get("messages") if chat else None
        if not isinstance(messages, list) or not any(
                isinstance(m, dict) and isinstance(m.get("content"), list) and
                any(isinstance(p, dict) and image_url(p) for p in m["content"]) for m in messages):
            return super()._prepare(body, chat)
        if getattr(self.engine, "tower", None) is None:
            raise RequestError("this server reads text only: start it with the vision tower (see the startup log) "
                               "for image input")
        from . import vision

        urls = message_images(normalize_messages(messages, media=True))
        try:
            images = vision.prepare_images(urls, allow_remote=getattr(self, "image_urls", False))
        except vision.ImageBusy:
            raise
        except (ValueError, OSError) as exc:
            raise RequestError(str(exc)) from None
        try:
            prepared = super()._prepare(body, chat, media=True)
            token = self.engine.w.cfg.image_token
            prompt, k = [], 0
            for t in prepared.prompt:
                if t == token and k < len(images):
                    prompt.extend([images[k].key] * images[k].tokens)
                    k += 1
                else:
                    prompt.append(t)
            if k != len(images) or token in prompt:
                raise RequestError(f"the chat template placed {sum(t == token for t in prepared.prompt)} images "
                                   f"for {len(images)} image parts")
            prepared.prompt, prepared.images = prompt, images
            return prepared
        except BaseException:
            images.close()
            raise

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body)
        if "tf_copy_drafts" in body and type(body["tf_copy_drafts"]) is not bool:
            return "tf_copy_drafts must be a boolean"
        if "tf_cofill" in body and type(body["tf_cofill"]) is not bool:
            return "tf_cofill must be a boolean"
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        problem = (f"this request needs a {need}-token context ({detail}), and this server was started for {limit}: "
                   f"shorten the prompt or reply{self._restart(need, ' both ranks')}")
        from tensorfold.cuda.server import V2_PARITY
        if V2_PARITY:               # v2 parity: vLLM's wording, which clients and mmastrac's smoketest match
            problem = f"This model's maximum context length is {limit} tokens; {problem}"
        return problem

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        prepared = prepared if prepared is not None else self.prepare(body, chat)
        try:
            model = str(body.get("model") or "")
            self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
            self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
            self.engine.request.images = prepared.images
            self.engine.request.cancelled = cancelled
            self.engine.request.priority = body.get("priority")
            self.engine.request.prefill_slice_layers = body.get("prefill_slice_layers")
            event = body.get("tf_session_event")
            self.engine.request.session_event = event if event in ("boot", "compaction", "fork") else None
            self.engine.request.copy_drafts = body.get("tf_copy_drafts")
            self.engine.request.cofill = body.get("tf_cofill")
            result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
            if chat and getattr(self.engine, "reply_prefill", False) and body.get("tf_reply_prefill", True):
                from .reply_prefill import enqueue
                enqueue(self, body, prepared, result)
            return result
        finally:
            if hasattr(self.engine, "request"):
                self.engine.request.images = None
                self.engine.request.cancelled = None
                self.engine.request.priority = None
                self.engine.request.prefill_slice_layers = None
                self.engine.request.session_event = None
                self.engine.request.copy_drafts = None
                self.engine.request.cofill = None
                self.engine.request.reply_prefill_epoch = None
            prepared.close()
