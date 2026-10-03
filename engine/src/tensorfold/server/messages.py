"""Text chat messages shared by the HTTP and template paths."""

import json
from typing import Any, Callable

from tensorfold.server.errors import RequestError

_MEDIA = ("image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url")
_IMAGE_PARTS = ("image_url", "image", "input_image")


def image_url(part: dict[str, Any]) -> str | None:
    """An image content part's URL (OpenAI ``image_url``, or ``image``/``input_image`` with a URL), else None."""

    if part.get("type") not in _IMAGE_PARTS:
        return None
    value = part.get(part["type"]) if part["type"] != "input_image" else part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return value if isinstance(value, str) and value else None


def validate_modalities(body: dict[str, Any]) -> None:
    if any(body.get(k) for k in _MEDIA) or body.get("modalities") not in (None, ["text"]):
        raise RequestError("this server accepts and produces text only; image, audio and video are unsupported")


_PROBE = "tensorfold-late-system-probe"


def late_system_role(render: Callable[[list[dict[str, Any]]], Any]) -> str:
    """``system`` when ``render`` (a chat template, as text) keeps a later system message in place, else ``user``."""

    probe = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
             {"role": "assistant", "content": "a"}, {"role": "system", "content": _PROBE},
             {"role": "user", "content": "v"}]
    try:
        return "system" if _PROBE in str(render(probe)) else "user"
    except Exception:  # noqa: BLE001 - a template that rejects the probe cannot render later system messages
        return "user"


def normalize_messages(messages: list[dict[str, Any]], *, late_system: str = "system",
                       media: bool = False) -> list[dict[str, Any]]:
    """Merge leading instructions as system text and retain later instructions as ``late_system`` so earlier conversation tokens stay unchanged.

    ``media``: user and tool messages may hold image parts, kept in place as ``{"type": "image", "url": ...}``.
    """

    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out, instructions = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError("message role must be system, developer, user, assistant or tool")
        if any(message.get(k) for k in _MEDIA):
            raise RequestError("this server accepts text only; image, audio and video inputs are unsupported")
        content = message.get("content")
        if isinstance(content, list):
            text, parts = [], []
            for part in content:
                url = image_url(part) if isinstance(part, dict) and media and role in ("user", "tool") else None
                if url is not None:
                    if text:
                        parts.append({"type": "text", "text": "".join(text)})
                        text = []
                    parts.append({"type": "image", "url": url})
                    continue
                if (not isinstance(part, dict) or part.get("type") != "text"
                        or any(part.get(k) for k in _MEDIA)):
                    raise RequestError("this server accepts text parts only; image, audio and video inputs are unsupported"
                                       if not media else "images are accepted in user and tool messages only, as "
                                       "image_url parts; audio and video are unsupported")
                if not isinstance(part.get("text"), str):
                    raise RequestError("a text content part must contain a text string")
                text.append(part["text"])
            if parts:
                content = parts + ([{"type": "text", "text": "".join(text)}] if text else [])
            else:
                content = "".join(text)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            raise RequestError("message content must be text or an array of text parts")
        item = message if content is message.get("content") else {**message, "content": content}
        if role in ("system", "developer"):
            if not out:
                instructions.append(item)
                continue
            if role != late_system:
                item = {**item, "role": late_system}
        out.append(item)
    if instructions:
        out.insert(0, {**instructions[0], "role": "system",
                       "content": "\n\n".join(m["content"] for m in instructions)})
    return messages if out == messages else out


def message_images(messages: list[dict[str, Any]]) -> list[str]:
    """The image URLs of normalized messages, in the order a chat template renders them."""

    return [part["url"] for m in messages if isinstance(m.get("content"), list)
            for part in m["content"] if part.get("type") == "image"]


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy assistant argument strings into mappings for templates, preserving caller messages."""

    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not calls:
            out.append(message)
            continue
        new_calls = []
        touched = False
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    call = {**call, "function": {**fn, "arguments": parsed}}
                    touched = True
            new_calls.append(call)
        if touched:
            out.append({**message, "tool_calls": new_calls})
            changed = True
        else:
            out.append(message)
    return out if changed else messages
