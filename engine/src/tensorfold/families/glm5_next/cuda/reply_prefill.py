"""Opt-in reply rendering for idle, prefill-only work. No decoded KV is reused."""

from __future__ import annotations

import os
import time
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyConfig:
    enabled: bool = False
    rows: int = 256

    @classmethod
    def from_env(cls, parallel: int):
        flag = os.environ.get("TF_GLM_REPLY_PREFILL", "0")
        if flag not in ("0", "1"):
            raise ValueError("TF_GLM_REPLY_PREFILL must be 0 or 1")
        if flag == "0":
            return cls()
        if parallel < 2 or os.environ.get("TF_GLM_FAIR_SCHED", "0") != "1":
            raise ValueError("reply prefill requires pooled serving and TF_GLM_FAIR_SCHED=1")
        rows = int(os.environ.get("TF_GLM_REPLY_PREFILL_ROWS", "256"))
        if not 1 <= rows <= 256:
            raise ValueError("TF_GLM_REPLY_PREFILL_ROWS must be between 1 and 256")
        return cls(True, rows)

    @property
    def reserve(self):
        # One active and one pending token list plus bounded rendering/queue overhead.
        # GPU work reuses the admitted pool and existing prefill buffers/store budget.
        return 64 * 2**20 if self.enabled else 0


def candidate(app, body, prepared, result):
    """Return a strict extension of the original rendered prompt, or a harmless miss.

    Render the API's normalized assistant message, not the raw generation IDs.
    Image requests and truncated replies are deliberately outside this first version.
    """
    from .app import with_effort

    if prepared.images or result["finish"] not in ("stop", "tool_calls"):
        return None, "image-or-incomplete"
    if not result.get("stats", {}).get("drafts"):
        return None, "cache-disabled"
    message = {"role": "assistant", "content": result["content"],
               "reasoning_content": result["reasoning"]}
    if result.get("calls"):
        message["tool_calls"] = result["calls"]
    kwargs = dict(with_effort(body).get("chat_template_kwargs") or {})
    thinking = bool(kwargs.pop("enable_thinking", app.default_thinking))
    kwargs["add_generation_prompt"] = False
    text = app.template.render([*body["messages"], message], tools=prepared.tools,
                               enable_thinking=thinking, extra=kwargs)
    ids = app.tok.encode(text, add_special_tokens=False).ids
    base = len(prepared.prompt)
    if not base < len(ids) < app.engine.limit or ids[:base] != prepared.prompt:
        return None, "render-prefix-miss"
    return ids, "candidate"


def enqueue(app, body, prepared, result):
    """A failed optional prediction must not fail an already completed chat reply."""
    start = time.perf_counter()
    scheduler = app.engine.scheduler
    # Concurrent HTTP completions must not each allocate a full rendered context.
    if not scheduler.reply_render_lock.acquire(blocking=False):
        result["stats"]["reply_prefill"] = {"status": "renderer-busy"}
        return
    try:
        epoch = getattr(app.engine.request, "reply_prefill_epoch", None)
        with scheduler.state_lock:
            current = epoch is not None and epoch == scheduler.reply_epoch and scheduler.failed is None
        if not current:
            result["stats"]["reply_prefill"] = {"status": "foreground-arrived"}
            return
        ids, reason = candidate(app, body, prepared, result)
        fingerprint = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() if ids else ""
        queued = ids is not None and scheduler.submit_reply_prefill(
            ids, base=len(prepared.prompt), policy=result["stats"]["policy"],
            epoch=epoch, fingerprint=fingerprint)
        if ids is not None:
            reason = "queued" if queued else "foreground-arrived"
        result["stats"]["reply_prefill"] = {
            "status": reason, "candidate_tokens": len(ids) if ids is not None else 0,
            "reply_tokens": len(ids) - len(prepared.prompt) if ids is not None else 0,
            "render_s": round(time.perf_counter() - start, 6),
        }
        if ids is not None:
            result["stats"]["reply_prefill"]["candidate_sha256"] = fingerprint
    except Exception:
        # Includes tokenizer/template failures. Never serialize the exception:
        # a template exception may include private message text.
        result["stats"]["reply_prefill"] = {"status": "render-error"}
    finally:
        scheduler.reply_render_lock.release()
