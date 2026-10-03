"""Exercise the actual HTTP handler on memory streams; never listen or connect."""

import io
import json
import signal
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli
from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda import vision
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tensorfold.server.errors import RequestError


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("real socket in host test"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("real DNS in host test"))
    monkeypatch.setattr(server, "socket_cancellation", lambda conn: NS(cancelled=False))


def request(app, body, *, path="/v1/chat/completions", method="POST", headers=None, module=server):
    handler = module.make_handler(app)
    h = object.__new__(handler)
    h.command, h.path = method, path
    h.request_version, h.requestline = "HTTP/1.1", f"{method} {path} HTTP/1.1"
    h.date_time_string = lambda: "Thu, 01 Jan 1970 00:00:00 GMT"
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    h.headers = {"Content-Length": str(len(data)), **(headers or {})}
    h.rfile, h.wfile, h.connection = io.BytesIO(data), io.BytesIO(), object()
    h.close_connection = False
    getattr(h, "do_" + method)()
    return h.wfile.getvalue()


def fail(exc):
    def call(*a, **k):
        raise exc
    return call


def app_stub(**kwargs):
    result = dict(final={}, calls=[], finish="stop", stats={}, prompt_tokens=2, completion_tokens=1,
                  content="ok", reasoning="")
    return NS(served="stub", prepare=lambda *a: object(), run=lambda *a, **k: result,
              effective_context_window=100, **kwargs)


def error(wire, status, message, kind):
    head, body = wire.split(b"\r\n\r\n", 1)
    assert head.startswith(f"HTTP/1.1 {status} ".encode())
    assert json.loads(body) == {"error": {"message": message, "type": kind}}


@pytest.mark.parametrize("body, headers, message", [
    (b"{", {}, "the request body is not JSON"),
    (b"\xff", {}, "the request body is not JSON"),
    (b"[]", {}, "the request body must be a JSON object"),
    (b"null", {}, "the request body must be a JSON object"),
    (b"{}", {"Content-Length": "-1"}, "Content-Length must be a non-negative integer"),
    (b"{}", {"Content-Length": "oops"}, "Content-Length must be a non-negative integer"),
    (b"{}", {"Transfer-Encoding": "chunked"}, "Transfer-Encoding is not supported; send Content-Length"),
])
def test_bad_bodies_are_openai_shaped_400(body, headers, message):
    error(request(app_stub(), body, headers=headers), 400, message, "invalid_request_error")


@pytest.mark.parametrize("stream", [False, True])
def test_saturated_image_admission_answers_503_before_stream_headers(stream):
    app = app_stub()
    app.prepare = fail(vision.ImageBusy("image preparation is busy (2 slots); retry later"))
    error(request(app, {"stream": stream}), 503, "image preparation is busy (2 slots); retry later", "server_error")


def test_actual_image_default_refusal_is_openai_shaped_400():
    from test_glm_vision_inputs import _app
    app = _app()
    app.prepare = app._prepare
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": "https://images.example/p.png"}]}], "stream": True}
    error(request(app, body), 400,
          "image 1: remote image URLs are disabled; send a data: URI or start with --image-urls", "invalid_request_error")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["remote", "bytes", "pixels", "busy"])
def test_image_refusal_finishes_while_generation_lock_is_held(monkeypatch, stream, case):
    from test_glm_vision_inputs import _app, _png, _url
    app = _app()
    app.lock = threading.Lock()
    url = "https://images.example/p.png"
    message = "image 1: remote image URLs are disabled; send a data: URI or start with --image-urls"
    status, kind = 400, "invalid_request_error"
    if case == "bytes":
        monkeypatch.setattr(vision, "MAX_BYTES", 64)
        url, message = "data:image/png," + "x" * 65, "image 1: an image is larger than 32 MB"
    elif case == "pixels":
        monkeypatch.setattr(vision, "MAX_PIXELS", 64)
        url, message = _url(_png(9, 8)), "image 1: an image exceeds 32000000 decoded pixels"
    elif case == "busy":
        monkeypatch.setattr(vision, "_slots", threading.BoundedSemaphore(0))
        status, kind = 503, "server_error"
        message = "image preparation is busy (2 slots); retry later"
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": url}]}], "stream": stream}
    # Use the real public App.prepare path. The lock stays owned by this thread
    # while a different request thread must finish its refusal, including SSE requests.
    with ThreadPoolExecutor(max_workers=1) as pool:
        app.lock.acquire()
        try:
            wire = pool.submit(request, app, body).result(timeout=2)
            assert app.lock.locked()
            error(wire, status, message, kind)
        finally:
            app.lock.release()


@pytest.mark.parametrize("phase", ["prepare", "run"])
def test_unexpected_exceptions_get_json_500_without_internal_details(phase):
    app = app_stub()
    setattr(app, phase, fail(RuntimeError("secret input /internal/path")))
    before = server.V2_COUNTERS.get("inflight", 0)
    error(request(app, {}), 500, "internal server error", "server_error")
    assert server.V2_COUNTERS["inflight"] == before


@pytest.mark.parametrize("exc, message, kind", [
    (RuntimeError("private details"), "internal server error", "server_error"),
    (RequestError("bad request"), "bad request", "invalid_request_error"),
])
def test_stream_errors_are_one_event_then_done(exc, message, kind):
    app = app_stub()
    app.run = fail(exc)
    wire = request(app, {"stream": True})
    head, body = wire.split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 ") and b"text/event-stream" in head
    frames = body.decode().strip().split("\n\n")
    assert frames[-1] == "data: [DONE]" and body.count(b"[DONE]") == 1
    assert json.loads(frames[-2][6:]) == {"error": dict(message=message, type=kind)}
    assert b"private details" not in wire and b"HTTP/1.1 500" not in wire


def test_get_and_tokenize_are_guarded(monkeypatch):
    class Broken:
        served = "stub"
        @property
        def effective_context_window(self):
            raise RuntimeError("broken")
    error(request(Broken(), {}, path="/v1/models", method="GET"), 500, "internal server error", "server_error")
    monkeypatch.setattr(server, "V2_PARITY", True)
    error(request(app_stub(), b"[]", path="/tokenize"), 400,
          "the request body must be a JSON object", "invalid_request_error")


def test_image_url_flag_is_off_by_default():
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "fixture"]).image_urls is False
    assert parser.parse_args(["serve", "fixture", "--image-urls"]).image_urls is True


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_sigusr1_is_registered_before_loading_every_rank(tmp_path, monkeypatch, rank):
    import faulthandler
    events = []
    monkeypatch.setattr(faulthandler, "register", lambda *a, **k: events.append((a, k)))
    def load(*a, **k):
        events.append("load")
        raise RuntimeError("stop before GPU")
    family = NS(title="stub", model_type="stub", package=NS(cuda_engine=load, CUDA_TP=(1, 2, 3)))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--tp", "3", "--rank", str(rank),
                                           "--master", "fixture"])
    with pytest.raises(RuntimeError, match="stop before GPU"):
        cli._serve_cuda(args, family, tmp_path)
    assert events == [((signal.SIGUSR1,), dict(file=2, all_threads=True)), "load"]


def test_opt_in_reaches_glm_app_without_becoming_an_engine_option(tmp_path, monkeypatch):
    import faulthandler
    monkeypatch.setattr(faulthandler, "register", lambda *a, **k: None)
    seen = []
    class App(GlmApp):
        effective_context_window = 100

        def __init__(self, *a, **k):
            seen.append(k)
    loaded = []
    family = NS(title="stub", model_type="stub", package=NS(CUDA_APP=App,
               cuda_engine=lambda *a, **k: loaded.append(k) or object()))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--image-urls"])
    assert cli._serve_cuda(args, family, tmp_path) == 0
    assert seen[0]["image_urls"] is True and "image_urls" not in loaded[0]


def test_unsupported_family_refuses_opt_in_before_engine_load(tmp_path, monkeypatch):
    import faulthandler
    monkeypatch.setattr(faulthandler, "register", lambda *a, **k: None)
    family = NS(title="stub", package=NS(cuda_engine=lambda *a, **k: pytest.fail("engine loaded")))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--image-urls"])
    with pytest.raises(ValueError, match="supported only by GLM on CUDA"):
        cli._serve_cuda(args, family, tmp_path)
