"""Small stdlib HTTP/SSE helpers for the shipped API checks (no proxies or redirects)."""
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent.parent
MODEL = "glm53"  # config/serve.conf, also used by smoke.sh and wait-ready.sh
TEMPLATE = {"reasoning_effort": "low"}


class Failure(Exception):
    """A check failed; messages contain no server response, address or local path."""


class Unreachable(Failure):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Failure("server redirected the request; use the configured serving endpoint")


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def endpoint(url=None):
    if url is None:
        defaults = {}
        for line in (HERE / "config/serve.conf").read_text().splitlines():
            match = re.fullmatch(r"SERVE_(API_HOST|API_PORT)=([A-Za-z0-9_.:\[\]-]+)", line)
            if match:
                defaults[match[1]] = match[2]
        if defaults.keys() != {"API_HOST", "API_PORT"}:
            raise Failure("missing API defaults in shipped config/serve.conf")
        values = defaults.copy()
        path = Path(os.environ.get("CLUSTER_ENV", HERE / "cluster.env"))
        if path.exists():
            for line in path.read_text().splitlines():
                match = re.fullmatch(r"\s*(API_HOST|API_PORT)=([A-Za-z0-9_.:\[\]-]*)\s*(?:#.*)?", line)
                if match:
                    values[match[1]] = match[2] or defaults[match[1]]
                elif re.match(r"\s*(API_HOST|API_PORT)=", line):
                    raise Failure("invalid API_HOST or API_PORT in cluster.env")
        host = values["API_HOST"]
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = "[" + host + "]"
        url = "http://" + host + ":" + values["API_PORT"]
    try:
        parts = urllib.parse.urlsplit(url)
        if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or parts.port == 0):
            raise ValueError
        parts.port  # validate the port without printing it
    except ValueError:
        raise Failure("invalid serving URL; use an http(s) endpoint without credentials") from None
    return url.rstrip("/")


def open_response(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    try:
        response = OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise Failure("server returned HTTP %d" % exc.code) from None
    except (urllib.error.URLError, OSError):
        raise Unreachable("server unreachable or timed out; check readiness and the serving endpoint") from None
    if response.status != 200:
        response.close()
        raise Failure("server did not return HTTP 200")
    return response


def json_call(url, payload=None, timeout=30):
    with open_response(url, payload, timeout) as response:
        try:
            result = json.load(response)
            if not isinstance(result, dict):
                raise Failure("server JSON response is not an object")
            return result
        except (ValueError, UnicodeError):
            raise Failure("server returned invalid JSON") from None


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise Failure(name + " is missing or is not a valid integer")
    return value


def seconds(value, name, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise Failure(name + " is missing or is not a finite valid time")
    return value


def body(messages, max_tokens=512):
    return {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": TEMPLATE, "stream": True, "stream_options": {"include_usage": True}}


def stream(url, payload, timeout=90):
    """Require the server's one-data-line frames, a final usage chunk, then DONE.

    Keep the final tensorfold object, never an earlier chunk's stats. Record both
    the first generated token (possibly reasoning) and the first visible content.
    """
    start = time.monotonic()
    first = visible_first = visible_last = None
    content, reasoning = [], []
    final = None
    pending = None
    done = False
    with open_response(url + "/v1/chat/completions", payload, timeout) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise Failure("stream Content-Type is not text/event-stream")
        while True:
            raw = response.readline(1048577)
            if not raw:
                break
            if len(raw) > 1048576 or not raw.endswith(b"\n"):
                raise Failure("broken SSE framing: oversized or unterminated line")
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeError:
                raise Failure("broken SSE framing: invalid UTF-8") from None
            if time.monotonic() - start > timeout:
                raise Failure("stream exceeded its request time limit")
            if line:
                if done or pending is not None or not line.startswith("data:"):
                    raise Failure("broken SSE framing: expected data line and blank separator")
                pending = line[5:].strip()
                continue
            if pending is None:
                continue
            data, pending = pending, None
            if data == "[DONE]":
                done = True
                continue
            try:
                event = json.loads(data)
            except ValueError:
                raise Failure("broken SSE framing: invalid data JSON") from None
            if not isinstance(event, dict) or not isinstance(event.get("choices"), list):
                raise Failure("stream chunk has invalid choices")
            final = event
            for choice in event["choices"]:
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    raise Failure("stream chunk has invalid delta")
                delta = choice["delta"]
                for key, target in (("content", content), ("reasoning_content", reasoning)):
                    text = delta.get(key)
                    if text is not None and not isinstance(text, str):
                        raise Failure("stream delta text is not a string")
                    if text:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        if key == "content":
                            if visible_first is None:
                                visible_first = now
                            visible_last = now
                        target.append(text)
    if not done or pending is not None:
        raise Failure("broken SSE framing: missing final [DONE] frame")
    usage = (final or {}).get("usage")
    if not isinstance(usage, dict):
        raise Failure("final stream chunk is missing usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        integer(usage.get(key), "usage." + key, 1)
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise Failure("usage total does not equal prompt plus completion tokens")
    if first is None:
        raise Failure("stream has no generated text")
    return {"usage": usage, "stats": final.get("tensorfold"), "content": "".join(content),
            "reasoning": "".join(reasoning), "ttft_s": first - start,
            "visible_ttft_s": None if visible_first is None else visible_first - start,
            "visible_duration_s": None if visible_first is None else visible_last - visible_first,
            "elapsed_s": time.monotonic() - start}


def cached(reply):
    stats = reply["stats"]
    count = integer(stats.get("cached") if isinstance(stats, dict) else None, "tensorfold.cached")
    if count > reply["usage"]["prompt_tokens"]:
        raise Failure("tensorfold.cached exceeds the prompt length")
    details = reply["usage"].get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise Failure("usage.prompt_tokens_details is not an object")
        if "cached_tokens" in details:
            other = integer(details["cached_tokens"], "usage.prompt_tokens_details.cached_tokens")
            if other != count:
                raise Failure("usage cached_tokens disagrees with tensorfold.cached")
    return count


def timing(reply):
    stats = reply["stats"]
    if not isinstance(stats, dict):
        raise Failure("final stream chunk is missing tensorfold timings")
    return (seconds(stats.get("prefill_s"), "tensorfold.prefill_s", positive=True),
            seconds(stats.get("decode_s"), "tensorfold.decode_s"))


def deadline(seconds_):
    """Wall-clock cap also stops a server trickling bytes past socket timeouts."""
    def expired(signum, frame):
        raise Failure("run exceeded its wall-clock time limit")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds_)


def failure_exit(label, step, exc):
    # Never echo urllib errors or response text (can contain addresses or prompts).
    detail = str(exc) if isinstance(exc, Failure) else "invalid or interrupted server response"
    print("%s FAIL %s: %s" % (label, step, detail), flush=True)
    return 2 if isinstance(exc, Unreachable) else 1


ERRORS = (Failure, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, http.client.HTTPException)
