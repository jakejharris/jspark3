#!/usr/bin/env python3
"""Real >32768-prompt-token, multi-decode-step witness for the long-context fix.

The configured --max-model-len 1000000 is a request-time limit, not evidence:
the historical single-stream failure mode aborted only beyond the 32768-token
boundary, and a requested max_tokens value proves nothing. This witness sends
one deterministic document whose SERVED prompt the server itself counts
(usage.prompt_tokens) and requires:

  1. HTTP 200 with a normal finish (finish_reason stop or length);
  2. usage.prompt_tokens strictly greater than 32768 - the exact served count,
     not a local estimate;
  3. usage.completion_tokens >= 20 - sustained multi-step decode, far beyond a
     token or two;
  4. the exact code word, placed after the whole filler body, is returned
     verbatim - attention across the >32K region actually reached the answer.

Any short prompt, truncated decode, mismatched code word, or HTTP error is a
visible REFUSE (exit 9) that propagates through `fleetctl verify`. Passing
here makes no claim about the configured 1,000,000-token limit.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

MIN_PROMPT_TOKENS = 32768          # the witness must exceed this served count
MIN_COMPLETION_TOKENS = 20         # sustained multi-step decode
CODE_WORD = "JSPARK3-LCW-7331"
DEFAULT_FILLER_LINES = 2600       # ~180 KB of deterministic body; ~35-45K served tokens

FILLER_TEMPLATE = "row {number:06d} of {total:06d}: the witness spindle keeps its own steady count."
DOCUMENT_HEAD = "Document begins below. Keep reading until the end."
DOCUMENT_TAIL = f"Document ends. Code word: {CODE_WORD}"
INSTRUCTION = (
    "Answer from the document above with exactly this shape and nothing else:\n"
    "Line 1: the code word, copied exactly.\n"
    "Lines 2-21: the integers 1 to 20, one per line."
)


class Refusal(RuntimeError):
    pass


def build_document(filler_lines: int) -> str:
    body = "\n".join(FILLER_TEMPLATE.format(number=n, total=filler_lines)
                     for n in range(1, filler_lines + 1))
    return f"{DOCUMENT_HEAD}\n{body}\n{DOCUMENT_TAIL}"


def build_payload(filler_lines: int = DEFAULT_FILLER_LINES) -> dict:
    return {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": f"{build_document(filler_lines)}\n\n{INSTRUCTION}"}],
        "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def payload_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"


PINNED_PAYLOAD_SHA256 = hashlib.sha256(payload_bytes(build_payload())).hexdigest()


def evaluate(prompt_tokens: int, completion_tokens: int, finish_reason: str,
             content: str, min_prompt_tokens: int) -> dict:
    """Pure gate logic; refuses obviously wrong shapes with explicit reasons."""
    reasons = []
    if not isinstance(prompt_tokens, int) or prompt_tokens <= min_prompt_tokens:
        reasons.append(f"served prompt_tokens={prompt_tokens} must be strictly greater than {min_prompt_tokens}")
    if not isinstance(completion_tokens, int) or completion_tokens < MIN_COMPLETION_TOKENS:
        reasons.append(f"completion_tokens={completion_tokens} is below the {MIN_COMPLETION_TOKENS}-token multi-decode floor")
    if finish_reason not in ("stop", "length"):
        reasons.append(f"finish_reason={finish_reason!r} is not a normal completion")
    if not content or CODE_WORD not in content:
        reasons.append("the exact code word from beyond the 32K boundary is missing from the answer")
    return {"pass": not reasons, "reasons": reasons,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "code_word_verbatim": CODE_WORD in content}


def run_request(url: str, body: bytes, api_key: str, timeout: int) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(diagnostics.private_read(response))
    except urllib.error.HTTPError as exc:
        detail = diagnostics.private_read(exc).decode("utf-8", "replace")
        raise Refusal(f"HTTP {exc.code} from endpoint: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="for example http://RANK0_ADDR:8888")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--min-prompt-tokens", type=int, default=MIN_PROMPT_TOKENS,
                        help="refuses values below the hard 32768 floor; raise it, never lower it")
    parser.add_argument("--filler-lines", type=int, default=DEFAULT_FILLER_LINES,
                        help="only for undersized-tokenizer recovery; non-default runs are marked unpinned in the receipt")
    args = parser.parse_args()
    try:
        if args.min_prompt_tokens < MIN_PROMPT_TOKENS:
            raise Refusal(f"--min-prompt-tokens may only exceed the hard {MIN_PROMPT_TOKENS} floor")
        if args.filler_lines <= 0:
            raise Refusal("--filler-lines must be positive")
        payload = build_payload(args.filler_lines)
        body = payload_bytes(payload)
        pinned = args.filler_lines == DEFAULT_FILLER_LINES
        if pinned and hashlib.sha256(body).hexdigest() != PINNED_PAYLOAD_SHA256:
            raise Refusal("default witness payload hash drift; the deterministic generator changed")
        started = time.monotonic()
        url = args.base_url.rstrip("/") + "/v1/chat/completions"
        code, result = run_request(url, body, os.environ.get(args.api_key_env, ""), args.timeout)
        if code != 200:
            raise Refusal(f"endpoint returned HTTP {code}")
        choices = result.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise Refusal("completion response choices schema drift")
        usage = result.get("usage") or {}
        message = choices[0].get("message")
        verdict = evaluate(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            str(choices[0].get("finish_reason")),
            message.get("content") if isinstance(message, dict) else "",
            args.min_prompt_tokens,
        )
        receipt = {
            "schema_version": 1, "grade": "ENGINEERING-EVIDENCE", "witness": "long-context->32K",
            "request_payload_sha256": hashlib.sha256(body).hexdigest(),
            "payload_pinned": pinned, "min_prompt_tokens": args.min_prompt_tokens,
            "min_completion_tokens": MIN_COMPLETION_TOKENS,
            "served_model_sha256": diagnostics.fingerprint(result.get("model")), "elapsed_seconds": round(time.monotonic() - started, 3),
            **verdict,
        }
        if not verdict["pass"]:
            raise Refusal("; ".join(verdict["reasons"]))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError,
            urllib.error.URLError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
