#!/usr/bin/env bash
# Quick checks against the API that can fail: it lists the model, an arithmetic answer, a JSON answer, counting to
# 200, a data-URL image (not a measured case), and a streamed reply. Run on rank 0, or anywhere through your ssh tunnel.
#
#   scripts/smoke.sh [URL]      default API_HOST/API_PORT from cluster.env, else http://127.0.0.1:8002
#
# Prints PASS or FAIL per check and SMOKE PASS n/n at the end; exit 0 only when every check passed.
set -uo pipefail
case ${1:-} in
  -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  --dry-run) echo "smoke.sh: dry run: would send six requests to the API"; exit 0 ;;
esac
python3 - "$(dirname "$0")" "${1:-}" <<'PYCODE'
import base64, binascii, json, struct, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from _api_check import (ERRORS, Failure, body, deadline, endpoint, failure_exit,
                        json_call, stream, timing)


def ask(q, n=512):
    """Greedy chat request; returns the stripped reply content."""
    body = {"model": "glm53", "messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    return (json_call(url + "/v1/chat/completions", body, timeout=900)["choices"][0]["message"].get("content") or "").strip()


def parse_json(s):
    """Parse a reply as JSON, tolerating a ```json fence; None if it is not JSON."""
    try:
        return json.loads(s.strip("`").removeprefix("json").strip())
    except ValueError:
        return None


def red_png():
    """A 32x32 solid red PNG built here, with no external image asset."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 32) * 32)) + chunk(b"IEND", b""))


checks = []  # (name, ok, what)
try:
    url = endpoint(sys.argv[2] or None)
    models = json_call(url + "/v1/models", timeout=10)
    checks.append(("models", any(m.get("id") == "glm53" for m in models.get("data", [])), "glm53 listed"))
    checks.append(("arithmetic", ask("What is 17*23? Reply with the number only.").rstrip(".") == "391", "17*23 = 391"))
    checks.append(("json", parse_json(ask('Reply with exactly this JSON and nothing else: {"a": 1, "b": [2, 3]}')) == {"a": 1, "b": [2, 3]}, "exact JSON"))
    png = red_png()
    answer = ask([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                  {"type": "text", "text": "What color fills this image? Answer with one word."}], 64)
    checks.append(("image", answer.lower().strip(". ") == "red", "solid red image; not the measured case"))
    got = ask("Count from 1 to 200, separated by single spaces. Output only the numbers.", 2048).split()
    checks.append(("count", got == [str(i) for i in range(1, 201)], "1..200, nothing missing or corrupt"))
    # A larger cap than 64 lets Low reasoning finish before the visible answer.
    deadline(120)
    reply = stream(url, body([{"role": "user", "content": "Say hello in one short sentence."}]), timeout=120)
    prefill, decode = timing(reply)
    if not reply["content"].strip():
        raise Failure("stream has no visible content")
    if reply["ttft_s"] >= 60 or reply["visible_ttft_s"] >= 90:
        raise Failure("stream first-token time exceeded its generous bound")
    # v1.8.4 measured TTFT and visible decode timing at the client. Keep both,
    # alongside this engine's final-chunk prefill_s/decode_s counterparts.
    duration = reply["visible_duration_s"]
    count = reply["usage"]["completion_tokens"]
    rate = "n/a" if count < 2 or duration <= 0 else "%.1f" % ((count - 1) / duration)
    checks.append(("streaming", True, "HTTP/SSE/usage/timing; TTFT=%.3fs visible=%.3fs prefill=%.3fs decode=%.3fs "
                   "client_decode=%s tok/s (includes reasoning tokens)" %
                   (reply["ttft_s"], reply["visible_ttft_s"], prefill, decode, rate)))
except ERRORS as e:
    failure_exit("SMOKE", "request/streaming", e)
    checks.append(("request", False, "request or streaming check failed"))
finally:
    import signal
    signal.alarm(0)

for name, ok, what in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name:10s} {what}")
bad = sum(not ok for _, ok, _ in checks)
print(f"SMOKE {'PASS' if not bad else 'FAIL'} {len(checks) - bad}/{len(checks)}")
sys.exit(1 if bad else 0)
PYCODE
