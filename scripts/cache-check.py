#!/usr/bin/env python3
"""Check cold isolation, next-turn prefix reuse and a different-nonce control.

    python3 scripts/cache-check.py [URL]

After scripts/wait-ready.sh, run on an otherwise idle server. Default endpoint:
API_HOST and API_PORT in cluster.env, else loopback:8002; model glm53, like
scripts/smoke.sh. An optional URL must use HTTP(S) without credentials, a query
or a fragment. Requests use no proxies and do not follow redirects.
CACHE PASS 15/15 includes plain reuse and cache_salt isolation/validation when
config/serve.env sets TF_GLM_CACHE_SALT=1. When absent or any other value,
cache_salt checks are not run because that configuration does not enable them:
the script prints SKIP, then CACHE PASS 3/3 covers only the plain checks.
Non-empty UTF-8 strings have no upper length limit.
Empty/non-null non-string salts and lone surrogates must return clean HTTP 400
without echoing the value; null is unsalted. Exit 0: PASS; 1: FAIL;
2: unreachable or timed out. This does not test exact-repeat replay,
session-store restoration from disk or logprob parity.
Plain checks have a 180-second wall cap. The enabled salt phase has its own
600-second cap: at an assumed 20 output tokens/s, its maximum 7,680 output
tokens take 384 seconds, leaving 216 seconds for prefill/probes/overhead.
That rate is a budget estimate, not a throughput measurement without a draft
model. Both phases allow at most 780 seconds combined; when salt checks are
skipped, only the first cap applies.

Tolerance is ZERO tokens: the server saves the entire prompt token sequence
and matches that full prefix, with no block rounding. The installed template
(template/chat-template.jinja) preserves the assistant <think> prefix. Keep
returned reasoning in the follow-up. No tools, forced calls or token-id flags.
Salt coverage comes only from config/serve.env; no command-line option, process
environment override or server probe selects it. Salt checks require server
support for cache_salt with pooled prefix caching.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import secrets
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _api_check import ERRORS, OPENER, Failure, Unreachable, body, cached, deadline, endpoint, failure_exit, stream

SALT_SWITCH = "TF_GLM_CACHE_SALT"


def correct_answer(content):
    """Keep the arithmetic canary independent of equation/prose/Markdown format."""
    numbers = {Decimal(number) for number in re.findall(r"[+-]?\d+(?:\.\d+)?", content.replace(",", ""))}
    return 391 in numbers and numbers <= {17, 23, 391}


def invalid_salt(url, value):
    """A rejected salt must produce a clean JSON 400, not an engine traceback."""
    label = "empty" if value == "" else "surrogate" if isinstance(value, str) else "non-string"
    payload = body([{"role": "user", "content": secrets.token_hex(24) + "\nSay hello."}], max_tokens=1)
    payload["cache_salt"] = value
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with OPENER.open(request, timeout=30) as response:
            if response.status == 200:
                raise Failure("server ignores or accepts %s cache_salt" % label)
            raise Failure("%s cache_salt did not return HTTP 400" % label)
    except urllib.error.HTTPError as exc:
        with exc:
            if exc.code != 400:
                raise Failure("%s cache_salt returned HTTP %d, expected 400" % (label, exc.code)) from None
            raw = exc.read(65537)
            try:
                error = json.loads(raw)
            except (ValueError, UnicodeError):
                raise Failure("%s cache_salt did not return a clean JSON error" % label) from None
            # The empty string has no identifiable bytes to search for. For the
            # other probes, inspect decoded JSON too so Unicode escapes cannot
            # hide an echo. Never include the value or body in a refusal message.
            if value != "" and str(value) in json.dumps(error, ensure_ascii=False):
                raise Failure("%s cache_salt error body echoes the supplied salt" % label)
            message = error.get("error") if isinstance(error, dict) else None
            if isinstance(message, dict):
                message = message.get("message")
            if (len(raw) > 65536 or exc.headers.get_content_type() != "application/json"
                    or not isinstance(message, str) or "cache_salt" not in message
                    or "traceback" in raw.decode("utf-8", errors="replace").lower()):
                raise Failure("%s cache_salt did not return a clean cache_salt error" % label)
    except (urllib.error.URLError, OSError):
        raise Unreachable("cache_salt validation: server unreachable or timed out") from None
    print("PASS  cache_salt-%s HTTP 400 with clean error" % label, flush=True)


def salt_cases(url):
    salt_a, salt_b = secrets.token_hex(24), secrets.token_hex(24)

    def fresh_prompt():
        return [{"role": "user", "content": secrets.token_hex(24) + "\n"
                 + "A ledger records items on shelves.\n" * 128 + "\nReply with the word READY."}]

    def ask(label, messages, salt, minimum=0):
        payload = body(messages, max_tokens=1024 if len(messages) > 1 else 512)
        if salt is not None:
            payload["cache_salt"] = salt
        try:
            reply = stream(url, payload)
            hit = cached(reply)
            if minimum:
                if reply["usage"]["prompt_tokens"] <= minimum or hit < minimum:
                    raise Failure("same-salt next turn did not reuse the entire earlier prompt")
            elif hit != 0:
                raise Failure("isolated cache_salt request reused cached tokens")
            if not reply["content"].strip():
                raise Failure("cache_salt reply has no visible content")
        except Failure as exc:
            detail = str(exc)
            if salt is not None and detail == "server returned HTTP 400":
                detail += "; server does not support cache_salt in this serving mode"
            raise type(exc)(label + ": " + detail) from None
        print("PASS  %s cached=%d minimum=%d" % (label, hit, minimum), flush=True)
        return reply

    def extend(messages, reply):
        return messages + [{"role": "assistant", "content": reply["content"],
                            "reasoning_content": reply["reasoning"]},
                           {"role": "user", "content": "What is 17*23? Reply with the number only."}]

    first = fresh_prompt()
    cold = ask("cache_salt-cold-A", first, salt_a)
    followup = extend(first, cold)
    ask("cache_salt-warm-A", followup, salt_a, cold["usage"]["prompt_tokens"])
    ask("cache_salt-cross-B", followup, salt_b)
    ask("cache_salt-unsalted-sees-salted", followup, None)
    # Own fresh nonce: never use an exact repeat or depend on earlier snapshots.
    other = fresh_prompt()
    unsalted = ask("cache_salt-unsalted-cold", other, None)
    ask("cache_salt-salted-sees-unsalted", extend(other, unsalted), salt_a)
    long_salt = secrets.token_hex(600)
    ask("cache_salt-long-sees-unsalted", extend(other, unsalted), long_salt)
    long_first = fresh_prompt()
    long_cold = ask("cache_salt-long-cold", long_first, long_salt)
    ask("cache_salt-unsalted-sees-long", extend(long_first, long_cold), None)
    invalid_salt(url, "")
    invalid_salt(url, 123)
    invalid_salt(url, "\ud800")  # JSON can carry this escape, but it is not valid UTF-8 text.
    return 12


def salt_policy():
    """Only the shipped recipe chooses coverage; neither env nor API can skip it."""
    path = Path(__file__).resolve().parent.parent / "config" / "serve.env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        raise Failure("shipped config/serve.env is missing; use a complete staged recipe") from None
    value = None
    for line in lines:
        key, separator, setting = line.split("#", 1)[0].strip().partition("=")
        if key.strip() == SALT_SWITCH and separator:
            value = setting.strip()
    return "honored" if value == "1" else "ignored"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?")
    args = parser.parse_args()
    step = "release-metadata"
    try:
        policy = salt_policy()
        step = "setup"
        url = endpoint(args.url)
        deadline(180)
        # The template's fixed header precedes this first-user-message nonce.
        # Full-prefix matching prevents a hit on another conversation's saved prompt.
        text = secrets.token_hex(24) + "\n" + ("A ledger records items on shelves.\n" * 128)
        text += "\nReply with the word READY."
        first = [{"role": "user", "content": text}]
        step = "cold"
        cold = stream(url, body(first))
        if cached(cold) != 0:
            raise Failure("cold prompt reused cached tokens")
        if not cold["content"].strip():
            raise Failure("cold reply has no visible content")
        n = cold["usage"]["prompt_tokens"]
        print("PASS  cold cached=0 prompt_tokens=%d" % n, flush=True)
        assistant = {"role": "assistant", "content": cold["content"], "reasoning_content": cold["reasoning"]}
        followup = {"role": "user", "content": "What is 17*23? Reply with the number only."}
        step = "next-turn"
        warm = stream(url, body(first + [assistant, followup], max_tokens=1024))
        hit = cached(warm)
        if warm["usage"]["prompt_tokens"] <= n or hit < n:
            raise Failure("next turn did not reuse the entire earlier prompt")
        if not correct_answer(warm["content"]):
            raise Failure("next-turn arithmetic answer is incorrect")
        print("PASS  next-turn cached=%d minimum=%d (zero-token tolerance)" % (hit, n), flush=True)
        step = "negative-control"
        other = secrets.token_hex(24) + text[48:]
        control = stream(url, body([{"role": "user", "content": other}, assistant, followup], max_tokens=1024))
        if cached(control) != 0:
            raise Failure("different-nonce control reused cached tokens")
        if not correct_answer(control["content"]):
            raise Failure("cold control and warm arithmetic answers do not agree")
        print("PASS  negative-control cached=0; warm and cold answers agree", flush=True)
        step = "cache_salt"
        count = 3
        if policy == "honored":
            deadline(600)
            count += salt_cases(url)
        else:
            print("SKIP  cache_salt: config/serve.env does not set " + SALT_SWITCH
                  + "=1; salt checks were not run", flush=True)
        print("CACHE PASS %d/%d" % (count, count))
        return 0
    except ERRORS as exc:
        return failure_exit("CACHE", step, exc)
    finally:
        import signal
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
