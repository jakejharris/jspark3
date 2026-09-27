#!/usr/bin/env python3
"""Small OpenAI-compatible API smoke test; no endpoint or credential is embedded."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


class Refusal(RuntimeError):
    pass


def request(url: str, body: bytes | None, api_key: str, timeout: int = 60):
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=timeout)


def json_call(url: str, payload: dict | None, api_key: str) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with request(url, body, api_key) as response:
        return response.status, json.loads(response.read())


def stream_call(url: str, payload: dict, api_key: str) -> dict:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    started = time.monotonic_ns()
    first = last = None
    done = False
    usage = None
    visible = []
    with request(url, body, api_key, timeout=120) as response:
        status = response.status
        for wire in response:
            line = wire.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    now = time.monotonic_ns()
                    first = now if first is None else first
                    last = now
                    visible.append(text)
    if status != 200 or not done or usage is None or first is None or last is None:
        raise Refusal("stream did not satisfy HTTP/SSE/usage/timing gates")
    completion = int(usage["completion_tokens"])
    duration = max(0, last - first)
    rate = None if completion < 2 or duration == 0 else (completion - 1) / (duration / 1e9)
    return {"http": status, "sse_done": done, "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": completion, "ttft_seconds": (first - started) / 1e9,
            "decode_tok_s": rate, "visible_characters": len("".join(visible))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="for example http://controller-address:8000")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    key = os.environ.get(args.api_key_env, "")
    try:
        with request(base + "/health", None, key, timeout=10) as response:
            if response.status != 200:
                raise Refusal("health endpoint is not HTTP 200")
        code, models = json_call(base + "/v1/models", None, key)
        identifiers = [row.get("id") for row in models.get("data") or []]
        if code != 200 or "glm-5.3-flash" not in identifiers:
            raise Refusal("served model is absent")
        arithmetic = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Return only the integer: 117 + 206"}],
                      "temperature": 0, "max_tokens": 8, "stream": False,
                      "chat_template_kwargs": {"enable_thinking": False}}
        code, result = json_call(base + "/v1/chat/completions", arithmetic, key)
        answer = result["choices"][0]["message"]["content"].strip()
        if code != 200 or answer != "323":
            raise Refusal("arithmetic answer is not exactly 323")
        streamed = stream_call(base + "/v1/chat/completions", {
            "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Reply with one short sentence saying hello."}],
            "temperature": 0, "top_p": 1, "max_tokens": 64, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
        }, key)
        print(json.dumps({"status": "PASS", "served_model": "glm-5.3-flash", "arithmetic": 323,
                          "short_stream": streamed, "performance_claim": None}, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError,
            urllib.error.URLError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
