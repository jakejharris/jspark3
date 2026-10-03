"""Deterministic slow-client regressions: real handlers, fake sockets/clock, no network."""
import io
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda import vision
from test_cuda_http_hardening import app_stub, error, no_network, request
from test_glm_image_lifetime import image_body, serving_app


class Clock:
    def __init__(self):
        self.now = 0.0
        self.changed = threading.Condition()
        self.entered = 0

    def monotonic(self):
        with self.changed:
            return self.now

    def advance(self, seconds):
        with self.changed:
            self.now += seconds
            self.changed.notify_all()


class Connection:
    def __init__(self):
        self.timeout = None
        self.timeouts = []

    def settimeout(self, seconds):
        self.timeout = seconds
        self.timeouts.append(seconds)


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    # Replace only this module's time reference; thread/future guards keep their real clock.
    monkeypatch.setattr(server, "time", NS(monotonic=value.monotonic, time=server.time.time))
    assert server._body_bytes == vision._image_count == vision._image_bytes == 0
    yield value
    assert server._body_bytes == vision._image_count == vision._image_bytes == 0


def handler(app, data=b"{}", *, size=None, source=None, sink=None, conn=None, path="/v1/chat/completions"):
    h = object.__new__(server.make_handler(app))
    h.command, h.path = "POST", path
    h.request_version, h.requestline = "HTTP/1.1", f"POST {path} HTTP/1.1"
    h.date_time_string = lambda: "Thu, 01 Jan 1970 00:00:00 GMT"
    h.headers = {"Content-Length": str(len(data) if size is None else size)}
    h.connection = conn if conn is not None else Connection()
    h.rfile = source if source is not None else io.BytesIO(data)
    h.wfile = sink if sink is not None else io.BytesIO()
    h.close_connection = False
    return h


def run(h):
    h.do_POST()
    return h.wfile.getvalue()


class Stalled:
    def __init__(self, clock, conn):
        self.clock, self.conn = clock, conn

    def read1(self, count):
        with self.clock.changed:
            end = self.clock.now + self.conn.timeout
            self.clock.entered += 1
            self.clock.changed.notify_all()
            assert self.clock.changed.wait_for(lambda: self.clock.now >= end, timeout=5)
        raise TimeoutError()


@pytest.mark.parametrize("count", [1, 64])
def test_stalled_uploads_release_and_next_chat_and_tokenize_succeed(clock, monkeypatch, count):
    monkeypatch.setattr(server, "V2_PARITY", True)
    app = app_stub(tok=NS(encode=lambda *a, **k: NS(ids=[1])))
    handlers = []
    for _ in range(count):
        conn = Connection()
        handlers.append(handler(app, source=Stalled(clock, conn), conn=conn))
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(run, h) for h in handlers]
        try:
            with clock.changed:
                assert clock.changed.wait_for(lambda: clock.entered == count, timeout=5)
            assert server._body_bytes == count * server.BODY_MIN_CHARGE
            assert vision._image_count == vision._image_bytes == 0
            assert vision._slots.acquire(blocking=False)
            assert vision._slots.acquire(blocking=False)
            vision._slots.release()
            vision._slots.release()
            if count == 64:
                error(request(app, {}), 503, "request body budget is busy (2048 MiB); retry later", "server_error")
            clock.advance(10)
            for h, f in zip(handlers, futures):
                error(f.result(timeout=3), 408, "the request body upload timed out", "invalid_request_error")
                assert h.close_connection
        finally:
            clock.advance(1000)  # also clean up deterministically if an assertion failed
    assert server._body_bytes == 0
    assert request(app, {}).startswith(b"HTTP/1.1 200 ")
    assert request(app, {"prompt": "ok"}, path="/tokenize").startswith(b"HTTP/1.1 200 ")
    print(json.dumps(dict(stalled_uploads=count, payload_bytes=0, timeout_status=408,
                          admission_after_timeout=server._body_bytes, next_chat=200, next_tokenize=200)))


def test_trickle_cannot_reset_absolute_deadline(clock):
    conn = Connection()
    class Trickle:
        calls = 0
        def read1(self, count):
            self.calls += 1
            if conn.timeout < 9:
                clock.advance(conn.timeout)
                raise TimeoutError()
            clock.advance(9)
            return b" "
    source = Trickle()
    h = handler(app_stub(), size=4, conn=conn, source=source)
    error(run(h), 408, "the request body upload timed out", "invalid_request_error")
    assert clock.now == pytest.approx(10 + 4 / 2**20)
    assert source.calls == 2 and min(conn.timeouts) < 2
    assert h.close_connection


def test_late_final_chunk_is_refused(clock):
    class Late:
        def read1(self, count):
            clock.advance(11)
            return b"{}"
    error(run(handler(app_stub(), source=Late())), 408, "the request body upload timed out", "invalid_request_error")


def test_premature_eof_closes_and_releases(clock):
    h = handler(app_stub(), data=b"{", size=2)
    error(run(h), 400, "the request body ended before Content-Length", "invalid_request_error")
    assert h.close_connection


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/tokenize"])
def test_cap_refuses_before_read1(clock, monkeypatch, path):
    monkeypatch.setattr(server, "V2_PARITY", True)
    class NeverRead:
        def read1(self, count):
            pytest.fail("oversize body was read")
    h = handler(app_stub(), size=server.MAX_BODY_BYTES + 1, source=NeverRead(), path=path)
    error(run(h), 413, "the request body exceeds 192 MiB", "invalid_request_error")


@pytest.mark.parametrize("rate,accepted", [(1.0, True), (0.96, True), (0.94, False)])
def test_full_192_mib_body_rate_boundary(clock, rate, accepted):
    class Upload:
        sent = 0
        def read1(self, count):
            prefix = b"{}" if not self.sent else b""
            self.sent += count
            clock.advance(count / (rate * 2**20))
            return prefix + b" " * (count - len(prefix))
    h = handler(app_stub(), size=server.MAX_BODY_BYTES, source=Upload())
    wire = run(h)
    if accepted:
        assert wire.startswith(b"HTTP/1.1 200 ")
        assert clock.now == pytest.approx(192 / rate)
    else:
        error(wire, 408, "the request body upload timed out", "invalid_request_error")
        assert 202 <= clock.now < 203
    assert h.connection.timeout == 10


@pytest.mark.parametrize("stream", [False, True])
def test_long_generation_has_no_elapsed_socket_deadline(clock, stream):
    app = app_stub()
    original = app.run
    def compute(*a, **k):
        clock.advance(10000)
        return original(*a, **k)
    app.run = compute
    wire = run(handler(app, data=json.dumps({"stream": stream}).encode()))
    assert wire.startswith(b"HTTP/1.1 200 ")
    assert b"[DONE]" in wire if stream else b'"content": "ok"' in wire


class BlockedWriter(io.BytesIO):
    def __init__(self, clock, conn, fail_at):
        super().__init__()
        self.clock, self.conn, self.fail_at = clock, conn, fail_at
        self.calls = 0

    def write(self, data):
        self.calls += 1
        if self.calls >= self.fail_at:
            self.clock.advance(self.conn.timeout)
            raise TimeoutError()
        return super().write(data)


@pytest.mark.parametrize("stream,fail_at", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_blocked_image_response_releases_all_ownership(clock, stream, fail_at):
    app = serving_app()
    refs, runs = [], []
    original = app.prepare
    def prepare(*a):
        result = original(*a)
        refs.extend(weakref.ref(im.patches) for im in result.images)
        return result
    app.prepare = prepare
    app.engine.generate = lambda *a, **k: runs.append(True) or {}
    conn = Connection()
    sink = BlockedWriter(clock, conn, fail_at)
    h = handler(app, data=json.dumps(image_body(stream=stream)).encode(), conn=conn, sink=sink)
    run(h)
    assert clock.now == 10 and h.close_connection
    assert len(runs) == (0 if stream else 1)
    assert all(ref() is None for ref in refs)
    assert getattr(app.engine.request, "images", None) is None


def test_failed_stream_emit_cancels_running_image_generation(clock):
    app = serving_app()
    app.tok.decode = lambda ids, **k: "x" * len(ids)
    feeds = []
    def generate(prompt, count, sampling, feed, **kwargs):
        for _ in range(20):
            stop = feed([1])
            feeds.append(stop)
            if stop:
                return {}
        pytest.fail("timed-out output did not cancel generation")
    app.engine.generate = generate
    conn = Connection()
    sink = BlockedWriter(clock, conn, 3)  # headers + role succeed; generated output stalls
    h = handler(app, data=json.dumps(image_body(stream=True)).encode(), conn=conn, sink=sink)
    run(h)
    assert feeds[-1] and clock.now == 10 and h.close_connection
    assert getattr(app.engine.request, "images", None) is None


def test_timeout_reply_nonreader_is_bounded_too(clock):
    conn = Connection()
    class Idle:
        def read1(self, count):
            clock.advance(conn.timeout)
            raise TimeoutError()
    sink = BlockedWriter(clock, conn, 2)  # 408 headers succeed, 408 JSON write times out
    h = handler(app_stub(), source=Idle(), conn=conn, sink=sink)
    run(h)
    assert h.close_connection and clock.now == 20 and sink.calls == 2
    assert server._body_bytes == 0


def test_idle_partial_headers_have_socket_timeout_before_body_admission(clock):
    class Headers:
        lines = iter([b"POST /v1/chat/completions HTTP/1.1\r\n", b"Host: example.invalid\r\n"])
        closed = False
        def readline(self, count):
            value = next(self.lines, None)
            if value is not None:
                return value
            clock.advance(conn.timeout)
            raise TimeoutError()
        def close(self):
            self.closed = True
    conn = Connection()
    source, sink = Headers(), io.BytesIO()
    conn.makefile = lambda *a: source
    conn.sendall = sink.write
    conn.fileno = lambda: -1
    h = server.make_handler(app_stub())(conn, ("fixture", 0), NS())
    assert h.close_connection and source.closed and clock.now == 10
    assert sink.getvalue() == b"" and server._body_bytes == 0
