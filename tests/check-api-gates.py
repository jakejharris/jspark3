#!/usr/bin/env python3
"""Offline HTTP fixtures for cache, prefill and smoke gates; no models or containers."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import _api_check as api

failures = []


def check(ok, name):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failures.append(name)


def prompt_count(messages):
    # Deliberately independent of production sizing: stand-in tokenizer counts words.
    return sum(len(m["content"].split()) + 8 for m in messages) + 3


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.gets.append(self.path)
        self.send(b'{"data":["private detail"]}' if self.server.mode == "bad-model-entry"
                  else b'{"data":[{"id":"glm53"}]}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, request))
        if self.server.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", self.server.redirect_to + "/escaped")
            self.end_headers()
            return
        if self.path == "/tokenize":
            count = prompt_count(request["messages"])
            if self.server.mode == "bad-tokenize":
                count = None
            self.send(json.dumps({"count": count}).encode())
            return
        if self.server.kind == "cache" and request.get("cache_salt") is not None:
            salt, mode = request["cache_salt"], self.server.mode
            surrogate = isinstance(salt, str) and any(0xd800 <= ord(char) <= 0xdfff for char in salt)
            invalid = not isinstance(salt, str) or not salt or surrogate
            if invalid or mode == "salt-field-rejected" or (mode == "long-salt-rejected" and len(salt) > 1000):
                if ((salt == "" and mode == "empty-salt-accepted")
                        or (salt == 123 and mode == "non-string-salt-accepted")
                        or (surrogate and mode == "surrogate-salt-accepted")):
                    self.send(b'{"accepted":true}')
                elif mode == "salt-error-echo" and salt == 123:
                    self.send(json.dumps({"error": {"message": "cache_salt rejected", "value": salt}}).encode(), status=400)
                elif mode == "salt-error-escaped-echo" and salt == 123:
                    error = '{"error":{"message":"cache_salt rejected: \u0031\u0032\u0033"}}'
                    # Send JSON Unicode escapes on the wire, not just decoded digits.
                    error = error.replace("123", "".join("\\u%04x" % ord(char) for char in "123"))
                    self.send(error.encode(), status=400)
                elif mode == "surrogate-error-echo" and surrogate:
                    self.send(json.dumps({"error": {"message": "cache_salt rejected", "value": salt}}).encode(), status=400)
                elif mode == "surrogate-error-traceback" and surrogate:
                    self.send(b'{"error":{"message":"cache_salt Traceback: private detail"}}', status=400)
                elif mode == "salt-error-traceback":
                    self.send(b'{"error":{"message":"cache_salt Traceback (most recent call last): private detail"}}', status=400)
                elif mode == "salt-error-not-json":
                    self.send(b'Traceback: private detail', content_type="text/plain", status=400)
                elif mode == "salt-error-missing-message":
                    self.send(b'{"error":{}}', status=400)
                else:
                    message = "cache_salt must be a non-empty string"
                    self.send(json.dumps({"error": {"message": message}}).encode(),
                              status=422 if mode == "salt-error-status" else 400)
                return
        if not request.get("stream"):
            if self.server.mode == "empty-chat-choices":
                self.send(b'{"choices":[]}')
                return
            content = request["messages"][-1]["content"]
            if isinstance(content, list):
                answer = "red"
            elif "17*23" in content:
                answer = "391"
            elif "exactly this JSON" in content:
                answer = '{"a": 1, "b": [2, 3]}'
            else:
                answer = " ".join(str(i) for i in range(1, 201))
            self.send(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())
            return
        self.server.streams.append(request)
        index = len(self.server.streams)
        self.server.contract_ok &= (request.get("stream_options") == {"include_usage": True}
                                   and request.get("model") == "glm53"
                                   and "tool_choice" not in request and "return_token_ids" not in request)
        mode, kind = self.server.mode, self.server.kind
        messages = request["messages"]
        n = prompt_count(messages)
        hit = prompt_count(messages[:1]) if kind == "cache" and index in (2, 5) else 0
        text = "READY" if kind == "cache" and index in (1, 4, 8, 11) else "391" if kind == "cache" else "Hello."
        if kind == "cache" and index in (2, 3):
            if mode == "formatted-answer":
                text = "17 × 23 = 391" if index == 2 else "The answer is **391**."
            elif mode == "no-answer-value":
                text = "The product is unknown."
            text = getattr(self.server, "answers", {}).get(index, text)
        if (kind == "cache" and mode == "needs-answer-headroom" and len(messages) > 1
                and request["max_tokens"] < 1024):
            text = ""  # Low reasoning exhausted the smaller output budget.
        if kind == "cache":
            if (mode == "cross-salt-hit" and index == 6 or mode == "unsalted-sees-salted" and index == 7
                    or mode == "salted-sees-unsalted" and index == 9 or mode == "salt-cold-hit" and index == 4
                    or mode == "long-sees-unsalted" and index == 10 or mode == "unsalted-sees-long" and index == 12
                    or mode == "long-cold-hit" and index == 11):
                hit = 8
            elif mode == "salt-warm-miss" and index == 5:
                hit -= 1
        stat = {"cached": hit, "prefill_s": n / 2100, "decode_s": 0.1}
        if kind == "prefill" and mode == "slow-warmup" and index <= 2:
            stat["prefill_s"] = 110  # startup work must not be measured as a cold gate
        usage = {"prompt_tokens": n, "completion_tokens": 3, "total_tokens": n + 3}
        if mode != "legacy":
            usage["prompt_tokens_details"] = {"cached_tokens": hit}
        # Only corrupt the relevant measured step, so warmup cannot hide a failure.
        active = kind != "prefill" or index > 2
        if active:
            if mode == "missing-cache":
                stat.pop("cached")
            elif mode == "bool-cache":
                stat["cached"] = False
            elif mode == "string-cache":
                stat["cached"] = "0"
            elif mode == "negative-cache":
                stat["cached"] = -1
            elif mode == "over-cache":
                stat["cached"] = n + 1
            elif (mode == "cold-hit" and index == 1) or mode == "prefill-hit":
                stat["cached"] = 8
            elif mode == "warm-miss" and index == 2:
                stat["cached"] = hit - 1
            elif mode == "control-hit" and index == 3:
                stat["cached"] = 8
            elif mode == "parity" and index == 3:
                text = "392"
            elif mode == "warm-answer" and index == 2:
                text = "392"
            elif mode == "below-floor":
                stat["prefill_s"] = n / 1000
            elif mode == "outlier":
                stat["decode_s"] = 100
            elif mode == "usage-length":
                usage["prompt_tokens"] += 100
                usage["total_tokens"] += 100
            elif mode == "missing-prefill":
                stat.pop("prefill_s")
            elif mode == "nan-prefill":
                stat["prefill_s"] = float("nan")
            elif mode == "zero-prefill":
                stat["prefill_s"] = 0
            elif mode == "missing-decode":
                stat.pop("decode_s")
            elif mode == "negative-decode":
                stat["decode_s"] = -1
            if "prompt_tokens_details" in usage and "cached" in stat:
                usage["prompt_tokens_details"]["cached_tokens"] = stat["cached"]
            if mode == "mismatch-cache":
                usage["prompt_tokens_details"]["cached_tokens"] = hit + 1
            elif mode == "bad-details":
                usage["prompt_tokens_details"]["cached_tokens"] = "0"
            if mode == "empty-content" or mode == "cold-empty" and index == 1:
                text = ""
            if mode == "missing-prompt":
                usage.pop("prompt_tokens")
            elif mode == "float-completion":
                usage["completion_tokens"] = 3.0
            elif mode == "bool-total":
                usage["total_tokens"] = True
            elif mode == "wrong-total":
                usage["total_tokens"] += 1
        chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"reasoning_content": "Let me check."}}]},
                  {"choices": [{"delta": {"content": text}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage, "tensorfold": stat}]
        if active and mode == "no-text":
            chunks[1]["choices"][0]["delta"] = {}
            chunks[2]["choices"][0]["delta"] = {}
        if active and mode == "missing-usage":
            chunks[-1].pop("usage")
        if active and mode == "early-usage":
            chunks[0]["usage"] = chunks[-1].pop("usage")
        if kind == "cache" and index == 5:
            if mode == "salt-missing-cache":
                chunks[-1]["tensorfold"].pop("cached")
            elif mode == "salt-mismatch-cache":
                chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] += 1
        if active and mode == "early-cache":
            chunks[0]["tensorfold"] = chunks[-1].pop("tensorfold")
        if active and mode == "empty-choices":
            chunks[-1]["choices"] = []  # also accept a separate final usage chunk
        data = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"
        if active:
            if mode == "no-done":
                data = data.replace(b"data: [DONE]\n\n", b"")
            elif mode == "bad-separator":
                data = data.replace(b"\n\n", b"\n", 1)
            elif mode == "bad-prefix":
                data = data.replace(b"data:", b"datum:", 1)
            elif mode == "bad-json":
                data = data.replace(b"data: {", b"data: !", 1)
            elif mode == "after-done":
                data += b"data: {}\n\n"
            elif mode == "truncated-done":
                data = data[:-1]
            elif mode == "bad-utf8":
                data = b"data: \xff\n\n" + data
        self.send(data, "application/json" if mode == "bad-type" else "text/event-stream; charset=utf-8",
                  503 if mode == "http-error" else 201 if mode == "http-created" else 200)


@contextlib.contextmanager
def fixture(kind, mode="pass"):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.kind, server.mode = kind, mode
    server.requests, server.streams, server.gets = [], [], []
    server.contract_ok = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, "http://127.0.0.1:%d" % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run(kind, url, *args, env=None, policy="1"):
    # Exercise shipped metadata using an isolated recipe copy, never an override
    # in the production CLI/environment and never a mutation of this worktree.
    with tempfile.TemporaryDirectory() as temp:
        root = HERE
        if kind == "cache":
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "config").mkdir()
            for name in ("cache-check.py", "_api_check.py"):
                shutil.copyfile(HERE / "scripts" / name, root / "scripts" / name)
            shutil.copyfile(HERE / "config/serve.conf", root / "config/serve.conf")
            (root / "config/serve.env").write_text("# shipped settings\n" if policy is None
                                                 else "TF_GLM_CACHE_SALT=" + policy + "\n")
        command = (["bash", str(root / "scripts/smoke.sh")] if kind == "smoke" else
                   [sys.executable, str(root / ("scripts/%s-check.py" % kind))])
        return subprocess.run(command + ([url] if url else []) + list(args), capture_output=True, text=True,
                              timeout=15, env=env)


def cases():
    for kind, summary in (("cache", "CACHE PASS 15/15"), ("prefill", "PREFILL PASS 2/2"), ("smoke", "SMOKE PASS 6/6")):
        for mode in ("pass", "legacy", "empty-choices"):
            with fixture(kind, mode) as (server, url):
                result = run(kind, url)
                check(result.returncode == 0 and summary in result.stdout.splitlines() and server.contract_ok,
                      "%s %s: final SSE usage/stats, include_usage, shipped request fields" % (kind, mode))
                if result.returncode:
                    print(result.stdout, result.stderr)
                if kind == "cache":
                    sent = server.streams
                    check(len(sent) == 12 and sent[0]["messages"] == sent[1]["messages"][:1]
                          and sent[1]["messages"][1]["reasoning_content"] == "Let me check."
                          and sent[1]["messages"][1:] == sent[2]["messages"][1:]
                          and sent[0]["messages"][0]["content"][:48] != sent[2]["messages"][0]["content"][:48],
                          "cache conversation extension and nonce-first control")
                    check(len(sent) == 12
                          and sent[3]["cache_salt"] == sent[4]["cache_salt"] == sent[8]["cache_salt"]
                          and sent[3]["cache_salt"] != sent[5]["cache_salt"]
                          and all(isinstance(sent[i]["cache_salt"], str) and sent[i]["cache_salt"] for i in (3, 4, 5, 8))
                          and sent[3]["messages"] == sent[4]["messages"][:1]
                          and sent[4]["messages"] == sent[5]["messages"] == sent[6]["messages"]
                          and "cache_salt" not in sent[6] and "cache_salt" not in sent[7]
                          and sent[7]["messages"] == sent[8]["messages"][:1]
                          and len({sent[i]["messages"][0]["content"][:48] for i in (0, 3, 7)}) == 3
                          and [req["cache_salt"] for _, req in server.requests[-3:]] == ["", 123, "\ud800"]
                          and len(sent[9]["cache_salt"]) == 1200
                          and sent[9]["cache_salt"] == sent[10]["cache_salt"]
                          and sent[7]["messages"] == sent[9]["messages"][:1]
                          and sent[10]["messages"] == sent[11]["messages"][:1]
                          and "cache_salt" not in sent[11]
                          and sent[10]["messages"][0]["content"][:48] != sent[7]["messages"][0]["content"][:48],
                          "salt A reuse, salt B isolation, both unsalted boundaries, fresh nonces, invalid salts")
                    check(all(req["max_tokens"] == (1024 if len(req["messages"]) > 1 else 512)
                              for req in sent[3:]) and sum(req["max_tokens"] for req in sent[3:]) == 7680,
                          "salt follow-ups have 1024 tokens; producer and total output budgets stay bounded")
                if kind == "prefill":
                    texts = [req["messages"][0]["content"] for req in server.streams]
                    check(len(texts) == 4 and len({t[:48] for t in texts}) == 4
                          and all(re.match(r"[0-9a-f]{48}\n", t) for t in texts),
                          "prefill warms both lengths and uses four distinct initial nonces")
        bad = ["missing-usage", "early-usage", "missing-prompt", "float-completion", "bool-total", "wrong-total",
               "no-done", "bad-separator", "bad-prefix", "bad-json", "after-done", "truncated-done", "bad-utf8",
               "bad-type", "http-error", "http-created", "no-text"]
        if kind in ("cache", "prefill"):
            bad += ["missing-cache", "bool-cache", "string-cache", "negative-cache", "over-cache",
                    "mismatch-cache", "bad-details", "early-cache"]
        if kind == "cache":
            bad += ["cold-hit", "warm-miss", "control-hit", "parity", "warm-answer", "cold-empty", "no-answer-value"]
        if kind in ("prefill", "smoke"):
            bad += ["missing-prefill", "nan-prefill", "zero-prefill", "missing-decode", "negative-decode"]
        if kind == "prefill":
            bad += ["bad-tokenize", "below-floor", "outlier", "prefill-hit", "usage-length"]
        if kind == "smoke":
            bad += ["empty-content", "bad-model-entry", "empty-chat-choices"]
        for mode in bad:
            with fixture(kind, mode) as (_, url):
                result = run(kind, url)
                check(result.returncode == 1 and "FAIL" in result.stdout and "Traceback" not in result.stderr,
                      "%s rejects %s" % (kind, mode))
                if result.returncode != 1:
                    print(result.stdout, result.stderr)
                if mode in ("bad-model-entry", "empty-chat-choices"):
                    check("SMOKE FAIL" in result.stdout and "FAIL  request" in result.stdout
                          and "private detail" not in result.stdout + result.stderr,
                          "malformed smoke JSON retains redacted FAIL row and summary")
    with fixture("cache", "formatted-answer") as (_, url):
        result = run("cache", url)
        check(result.returncode == 0 and "CACHE PASS 15/15" in result.stdout,
              "cache accepts 17 × 23 = 391 and a differently formatted control answer")
    for mode in ("cross-salt-hit", "unsalted-sees-salted", "salted-sees-unsalted", "salt-cold-hit", "salt-warm-miss",
                 "salt-field-rejected", "empty-salt-accepted", "non-string-salt-accepted", "salt-missing-cache",
                 "salt-mismatch-cache", "salt-error-traceback", "salt-error-not-json", "salt-error-missing-message",
                 "salt-error-status", "long-salt-rejected", "long-sees-unsalted", "unsalted-sees-long", "long-cold-hit",
                 "salt-error-echo", "salt-error-escaped-echo", "surrogate-salt-accepted",
                 "surrogate-error-echo", "surrogate-error-traceback"):
        with fixture("cache", mode) as (_, url):
            result = run("cache", url)
            check(result.returncode == 1 and "cache_salt" in result.stdout and "FAIL" in result.stdout
                  and "Traceback" not in result.stdout + result.stderr and "private detail" not in result.stdout,
                  "cache rejects %s with clean cache_salt failure" % mode)
            if mode == "salt-field-rejected":
                check(result.stdout.splitlines()[-1] ==
                      "CACHE FAIL cache_salt: cache_salt-cold-A: server returned HTTP 400; "
                      "server does not support cache_salt in this serving mode",
                      "valid salt 400 explains the missing server support")
            if mode.startswith("surrogate-"):
                check("surrogate cache_salt" in result.stdout and "\\ud800" not in result.stdout + result.stderr,
                      "surrogate rejection fault is named without echoing the escaped value")
            if result.returncode != 1:
                print(result.stdout, result.stderr)
    for scale in ("0", "-1", "nan", "inf"):
        result = run("prefill", "http://127.0.0.1:1", "--floor-scale", scale)
        check(result.returncode == 1 and "--floor-scale" in result.stdout, "prefill refuses invalid floor scale " + scale)
    with fixture("prefill", "below-floor") as (_, url):
        result = run("prefill", url, "--floor-scale", "0.5")
        check(result.returncode == 0 and "PREFILL PASS 2/2" in result.stdout, "floor scale adjusts both floors")
    with fixture("cache") as (server, url):
        first = run("cache", url)
        first_nonce = server.streams[0]["messages"][0]["content"][:48]
        first_salt = server.streams[3]["cache_salt"]
        server.streams.clear()
        second = run("cache", url)
        check(first.returncode == second.returncode == 0
              and first_nonce != server.streams[0]["messages"][0]["content"][:48]
              and first_salt != server.streams[3]["cache_salt"],
              "cache re-runs start with independent random nonces and salts")
    # A reserved but non-listening socket cannot accidentally contact a live server.
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        url = "http://127.0.0.1:%d" % closed.getsockname()[1]
        for kind in ("cache", "prefill"):
            result = run(kind, url)
            check(result.returncode == 2 and "unreachable" in result.stdout and url not in result.stdout,
                  kind + " distinguishes unreachable from a failed assertion without leaking the endpoint")
    with fixture("cache", "redirect") as (redirect, url), fixture("cache") as (sink, sink_url):
        redirect.redirect_to = sink_url
        for kind in ("cache", "prefill", "smoke"):
            result = run(kind, url)
            check(result.returncode == 1 and "redirected" in result.stdout and not sink.requests and not sink.gets,
                  kind + " never follows a redirect to another endpoint")


def salt_coverage():
    skip_line = ("SKIP  cache_salt: config/serve.env does not set TF_GLM_CACHE_SALT=1; "
                 "salt checks were not run")
    for label, policy in (("disabled", "0"), ("absent", None), ("comment only", "# unset"),
                          ("disabled with comment", "0 # disabled"), ("disabled with whitespace", " 0 ")):
        with fixture("cache", "salt-field-rejected") as (server, url):
            result = run("cache", url, policy=policy)
            check(result.returncode == 0 and "CACHE PASS 3/3" in result.stdout.splitlines()
                  and skip_line in result.stdout.splitlines()
                  and "PASS  cache_salt" not in result.stdout and len(server.streams) == 3
                  and all("cache_salt" not in request for _, request in server.requests),
                  label + ": shipped switch skips every salt request and explains why checks were not run")
    for policy in ("", "auto", "true", "01", "honored", "1\nTF_GLM_CACHE_SALT=0"):
        with fixture("cache") as (server, url):
            result = run("cache", url, policy=policy)
            check(result.returncode == 0 and skip_line in result.stdout.splitlines()
                  and "CACHE PASS 3/3" in result.stdout.splitlines() and len(server.streams) == 3,
                  "shipped switch other than exactly 1 skips salts: " + repr(policy))
    for declared, supplied, mode, expected in (("1", "0", "cross-salt-hit", 1),
                                               ("0", "1", "salt-field-rejected", 0)):
        with fixture("cache", mode) as (server, url):
            env = dict(os.environ, TF_GLM_CACHE_SALT=supplied)
            result = run("cache", url, policy=declared, env=env)
            check(result.returncode == expected and ("SKIP  cache_salt" in result.stdout) == (declared == "0"),
                  "shell environment cannot override shipped salt coverage: " + declared)


def configuration():
    with tempfile.TemporaryDirectory() as temp, fixture("cache") as (server, url), fixture("cache") as (proxy, proxy_url):
        cluster = Path(temp) / "cluster.env"
        cluster.write_text("API_HOST=0.0.0.0\nAPI_PORT=%d\n" % server.server_port)
        env = dict(os.environ, CLUSTER_ENV=str(cluster), http_proxy=proxy_url, HTTP_PROXY=proxy_url,
                   ALL_PROXY=proxy_url, all_proxy=proxy_url, NO_PROXY="", no_proxy="")
        result = run("cache", None, env=env)
        check(result.returncode == 0 and not proxy.requests and not proxy.gets,
              "cluster API_HOST/API_PORT resolved locally, wildcard mapped to loopback, proxies ignored")
    with patch.dict(os.environ, {"CLUSTER_ENV": str(HERE / "missing-cluster.env")}):
        check(api.endpoint() == "http://127.0.0.1:8002", "missing cluster.env uses the shipped loopback default")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "config").mkdir()
        (root / "config/serve.conf").write_text("SERVE_API_HOST=127.0.0.2\nSERVE_API_PORT=8765\n")
        cluster = root / "cluster.env"
        for assignments in ("API_HOST=\nAPI_PORT=\n", "API_HOST=   # default\nAPI_PORT=   # default\n",
                            "API_HOST=127.0.0.3\nAPI_HOST=\nAPI_PORT=9876\nAPI_PORT=\n"):
            cluster.write_text(assignments)
            with patch.object(api, "HERE", root), patch.dict(os.environ, {"CLUSTER_ENV": str(cluster)}):
                check(api.endpoint() == "http://127.0.0.2:8765",
                      "blank API fields use shipped serve.conf defaults, including comments and reassignment")
    for url in ("file:///etc/passwd", "http://user:secret@localhost", "http://localhost:99999", "http://localhost/?secret=1"):
        try:
            api.endpoint(url)
        except api.Failure:
            check(True, "invalid or credentialed serving URL refused")
        else:
            check(False, "invalid or credentialed serving URL refused")


def timing_bounds():
    # Exercise the real smoke entry point and HTTP fixture; advance only its clock,
    # avoiding a minute-long sleep in an offline test.
    source = (HERE / "scripts/smoke.sh").read_text().split("<<'PYCODE'\n", 1)[1].rsplit("\nPYCODE", 1)[0]
    for elapsed, change_at in ((61, 2), (91, 9)):
        with fixture("smoke") as (_, url):
            calls = [0]
            def clock():
                calls[0] += 1
                return 0 if calls[0] < change_at else elapsed
            output = io.StringIO()
            with patch.object(sys, "argv", ["-", str(HERE / "scripts"), url]), patch.object(api.time, "monotonic", clock), contextlib.redirect_stdout(output):
                try:
                    exec(compile(source, "smoke.sh", "exec"), {"__name__": "__main__"})
                except SystemExit as exc:
                    code = exc.code
            check(code == 1 and "first-token time" in output.getvalue(), "smoke rejects late first token (%ds)" % elapsed)


def prefill_warmup():
    spec = importlib.util.spec_from_file_location("prefill_check", HERE / "scripts/prefill-check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check([(row["tokens"], row["floor"]) for row in module.FLOOR_TABLE] == [(8000, 1397), (32000, 1482)]
          and all(row["provisional"] is False and row["margin"] == 0.25 for row in module.FLOOR_TABLE),
          "prefill calibrated floors are 8K 1397 / 32K 1482, not provisional, with 25% margin")
    with fixture("prefill", "slow-warmup") as (_, url):
        output = io.StringIO()
        with patch.object(sys, "argv", ["prefill-check.py", url]), patch.object(module, "deadline") as cap, \
                patch.object(module, "stream", wraps=module.stream) as request, contextlib.redirect_stdout(output):
            result = module.main()
        check(result == 0 and [call.args[0] for call in cap.call_args_list] == [230, 300, 300, 230]
              and [call.kwargs["timeout"] for call in request.call_args_list] == [300, 300, 90, 90],
              "prefill gives each warmup 300s and starts a separate 230s measured cap")
        check("PROVISIONAL" not in output.getvalue()
              and re.findall(r" floor=([\d.]+)", output.getvalue()) == ["1397.0", "1482.0"],
              "prefill displays calibrated floors without the provisional banner")
        rates = re.findall(r"target_tokens=(\d+) prompt_tokens=(\d+) prefill_s=([\d.]+) rate=([\d.]+)", output.getvalue())
        check(len(rates) == 2 and {int(row[0]) for row in rates} == {8000, 32000}
              and all(abs(int(tokens) / float(seconds) - float(rate)) < 0.01 for _, tokens, seconds, rate in rates),
              "prefill receipt lines identify both targets and preserve enough rate/timing precision")
    for flags in ((True, True), (False, True), (False, False)):
        rows = tuple(dict(row, floor=2000, source="fixture-calibration", provisional=flag)
                     for row, flag in zip(module.FLOOR_TABLE, flags))
        with fixture("prefill") as (_, url):
            output = io.StringIO()
            with patch.object(sys, "argv", ["prefill-check.py", url]), patch.object(module, "FLOOR_TABLE", rows), \
                    contextlib.redirect_stdout(output):
                result = module.main()
            check(result == 0 and ("PROVISIONAL" in output.getvalue()) == any(flags)
                  and output.getvalue().count("floor=2000.0") == 2
                  and not any(row["source"] in output.getvalue() for row in module.FLOOR_TABLE),
                  "one table controls both displayed floors and calibration banner: " + repr(flags))
    rows = tuple(dict(row, floor=2500, source="fixture-calibration", provisional=False) for row in module.FLOOR_TABLE)
    with fixture("prefill") as (_, url):
        output = io.StringIO()
        with patch.object(sys, "argv", ["prefill-check.py", url]), patch.object(module, "FLOOR_TABLE", rows), \
                contextlib.redirect_stdout(output):
            result = module.main()
        check(result == 1 and "below its floor" in output.getvalue() and "floor=2500.0" in output.getvalue()
              and "PROVISIONAL" not in output.getvalue(), "calibrated table edits also control the actual rate gate")
    output = io.StringIO()
    with patch.object(sys, "argv", ["prefill-check.py", "--help"]), patch.object(module, "FLOOR_TABLE", rows), \
            contextlib.redirect_stdout(output):
        try:
            module.main()
        except SystemExit as exc:
            result = exc.code
    check(result == 0 and "FLOOR_TABLE" in output.getvalue()
          and not any(old in output.getvalue() for old in ("1397", "1482", "fixture-calibration", "PROVISIONAL")),
          "prefill help has no duplicated floor values, sources or calibration claim")
    # Exercise both timeout locations without waiting for real socket deadlines.
    for error, expected in ((api.Unreachable("server unreachable or timed out"), 2), (socket.timeout(), 1)):
        output = io.StringIO()
        with patch.object(sys, "argv", ["prefill-check.py", "http://127.0.0.1:1"]), \
                patch.object(module, "deadline"), patch.object(module, "sized_prompt", return_value=([], 8000)), \
                patch.object(module, "stream", side_effect=error), contextlib.redirect_stdout(output):
            result = module.main()
        check(result == expected and "warmup-8000" in output.getvalue() and "fresh boot" in output.getvalue()
              and "retry once" in output.getvalue(), "warmup timeout retains exit code and names fresh-boot retry")


def answer_shapes():
    good = ("391", "391.", "17 × 23 = 391", "The answer is 391", "**391**", "The answer is **391**.",
            "17*23 = 391", "391 = 17 × 23", "391 (17 × 23)", "Answer: 391 ✓ (computed 17*23)",
            "17 × 23 = 391\n\nCheck: 391 / 23 = 17", "391.0")
    bad = ("not 391, it's 390", "390", "392", "391 is wrong; it's 390.", "The product is unknown.",
           "", "391,000", "١٢٣", "-391", "391.00000000000000001")
    for expected, shapes in ((0, good), (1, bad)):
        for answer in shapes:
            for index, step in ((2, "next-turn"), (3, "negative-control")):
                with fixture("cache") as (server, url):
                    server.answers = {index: answer}
                    result = run("cache", url, policy="0")
                    check(result.returncode == expected
                          and ("CACHE PASS 3/3" in result.stdout if expected == 0 else "CACHE FAIL " + step in result.stdout),
                          "%s answer %r: %s" % (step, answer, "PASS" if expected == 0 else "FAIL"))


def cache_budgets():
    spec = importlib.util.spec_from_file_location("cache_check", HERE / "scripts/cache-check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for policy, salt_duration, expected in (("honored", 50, 0), ("honored", 90, 1), ("ignored", 50, 0)):
        # Real HTTP replies with a virtual elapsed clock; no multi-minute sleeps.
        # Plain takes 150s; healthy salt takes 450s, and over-budget salt exceeds 600s.
        now, expires, calls = [0], [0], [0]
        caps = []
        def cap(seconds):
            caps.append(seconds)
            expires[0] = now[0] + seconds
        def request(*args, **kwargs):
            reply = api.stream(*args, **kwargs)
            calls[0] += 1
            now[0] += 50 if calls[0] <= 3 else salt_duration
            if now[0] > expires[0]:
                raise api.Failure("run exceeded its wall-clock time limit")
            return reply
        with fixture("cache", "needs-answer-headroom") as (_, url):
            output = io.StringIO()
            with patch.object(sys, "argv", ["cache-check.py", url]), patch.object(module, "salt_policy", return_value=policy), \
                    patch.object(module, "deadline", cap), patch.object(module, "stream", request), \
                    contextlib.redirect_stdout(output):
                result = module.main()
            check(result == expected and caps == ([180, 600] if policy == "honored" else [180])
                  and ("wall-clock time limit" in output.getvalue() if expected else "CACHE PASS" in output.getvalue()),
                  "cache independent bounded phases with %s policy and %ds salt replies" % (policy, salt_duration))


def main():
    cases()
    configuration()
    salt_coverage()
    timing_bounds()
    prefill_warmup()
    answer_shapes()
    cache_budgets()
    print("check-api-gates: %d failed" % len(failures))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
