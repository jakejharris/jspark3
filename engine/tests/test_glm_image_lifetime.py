"""Real HTTP/prepare/run ownership with CPU images and memory streams; no listeners or network."""
import io
import json
import queue
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda import vision
from test_cuda_http_hardening import request, error, app_stub, no_network
from test_glm_vision_inputs import _app, _png, _url


def image_body(*, stream=False, urls=None):
    return {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": u} for u in (urls or [_url(_png(30, 30))])]}],
        "temperature": 0, "max_tokens": 1, "stream": stream}


def serving_app():
    app = _app()
    app.served, app.native_context_window = "stub", 0
    app.lock = threading.Lock()
    app.sampling = {"temperature": 0, "top_k": 20, "top_p": .95}
    app.tok.decode = lambda *a, **k: ""
    app.engine.request, app.engine.eos = threading.local(), ()
    app.engine.generate = lambda *a, **k: {}
    return app


def eventually(predicate):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


@pytest.fixture(autouse=True)
def resources_return_to_zero():
    assert vision._image_count == vision._image_bytes == server._body_bytes == 0
    yield
    eventually(lambda: vision._image_count == vision._image_bytes == server._body_bytes == 0)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", ["count", "bytes"])
def test_33_real_handlers_are_bounded_and_disconnects_free_waiters(monkeypatch, stream, budget):
    app = serving_app()
    cancelled = threading.Event()
    class Cancel:
        @property
        def cancelled(self): return cancelled.is_set()
    monkeypatch.setattr(server, "socket_cancellation", lambda _: Cancel())
    prepared = queue.Queue()
    refs = []
    original = app.prepare
    def measured(*a):
        p = original(*a)
        refs.extend(weakref.ref(im.patches) for im in p.images)
        prepared.put(True)
        return p
    app.prepare = measured
    limit = 4
    if budget == "bytes":
        im = vision.prepare(_png(30, 30))
        monkeypatch.setattr(vision, "PREPARED_BYTES", im.patches.numel() * im.patches.element_size())
        del im
        limit = 1
    accepted = []
    app.lock.acquire()
    with ThreadPoolExecutor(max_workers=33) as pool:
        try:
            for _ in range(limit):
                accepted.append(pool.submit(request, app, image_body(stream=stream)))
                prepared.get(timeout=3)
            expected = ("image requests are busy (4 active or waiting); retry later" if budget == "count" else
                        "prepared image budget is busy (1024 MiB); retry with fewer or smaller images")
            for _ in range(33 - limit):
                wire = pool.submit(request, app, image_body(stream=stream)).result(timeout=3)
                error(wire, 503, expected, "server_error")
            assert vision._image_count == limit
            assert vision._image_bytes <= vision.PREPARED_BYTES
            assert sum(r() is not None for r in refs) == limit
            assert all(not f.done() for f in accepted)
            # CPU preparation has ended, but image request ownership and bytes stay charged.
            assert vision._slots.acquire(timeout=1)
            assert vision._slots.acquire(timeout=1)
            vision._slots.release(); vision._slots.release()
            print(json.dumps(dict(submissions=33, retained=limit, refused=33-limit,
                                  prepared_bytes=vision._image_bytes, budget=budget, stream=stream)))
            cancelled.set()
            for f in accepted:
                f.result(timeout=3)              # must finish BEFORE the generation lock is released
            assert app.lock.locked()
            eventually(lambda: all(r() is None for r in refs))
            assert vision._image_count == vision._image_bytes == 0
        finally:
            cancelled.set()
            app.lock.release()


@pytest.mark.parametrize("phase", ["success", "context", "generate", "sampling", "headers", "cancel"])
@pytest.mark.parametrize("stream", [False, True])
def test_success_and_every_error_release_image_storage(monkeypatch, phase, stream):
    app = serving_app()
    refs = []
    prepare = app._prepare
    def measured(*a, **k):
        p = prepare(*a, **k)
        refs.extend(weakref.ref(im.patches) for im in p.images)
        return p
    app._prepare = measured
    if phase == "context":
        app.check = lambda *a, **k: "bad context"
    elif phase == "generate":
        def fail(*a, **k): raise RuntimeError("private generation detail")
        app.engine.generate = fail
    elif phase == "sampling":
        def fail(*a, **k): raise RuntimeError("private sampling detail")
        app.sampling_for = fail
    elif phase == "headers":
        original = server.make_handler
        def handler(app):
            h = original(app)
            def end_headers(self): raise BrokenPipeError()
            h.end_headers = end_headers
            return h
        monkeypatch.setattr(server, "make_handler", handler)
    elif phase == "cancel":
        gone = NS(cancelled=False)
        monkeypatch.setattr(server, "socket_cancellation", lambda _: gone)
        def cancel(prompt, count, sampling, feed, **kwargs):
            gone.cancelled = True
            assert feed([1])
            return {}
        app.engine.generate = cancel
    wire = request(app, image_body(stream=stream))
    if phase == "success":
        assert wire.startswith(b"HTTP/1.1 200")
    elif phase == "context":
        error(wire, 400, "bad context", "invalid_request_error")
    elif phase in ("generate", "sampling"):
        assert b"internal server error" in wire and b"private" not in wire
    assert getattr(app.engine.request, "images", None) is None
    assert all(r() is None for r in refs)
    assert vision._image_count == vision._image_bytes == 0


def test_multi_image_decode_failure_releases_already_prepared_storage():
    app = serving_app()
    wire = request(app, image_body(urls=[_url(_png(30, 30)), "data:image/png;base64,YmFk"]))
    error(wire, 400, "image 2: the image could not be decoded", "invalid_request_error")
    assert vision._image_count == vision._image_bytes == 0


def test_native_timeout_keeps_count_and_bytes_until_worker_exits(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(vision, "PREP_SECONDS", .03)
    def blocked(data, *, reserve):
        reserve(100)
        entered.set()
        assert release.wait(3)
        return object()
    monkeypatch.setattr(vision, "prepare", blocked)
    try:
        wire = request(serving_app(), image_body())
        error(wire, 400, "image preparation exceeded 10 seconds", "invalid_request_error")
        assert entered.is_set() and vision._image_count == 1 and vision._image_bytes == 100
        assert server._body_bytes == 0
    finally:
        release.set()
    eventually(lambda: vision._image_count == vision._image_bytes == 0)


def test_prepared_byte_budget_refuses_before_pixel_decode(monkeypatch):
    monkeypatch.setattr(vision, "PREPARED_BYTES", 1)
    monkeypatch.setattr(Image.Image, "load", lambda *a: pytest.fail("over-budget image decoded"))
    # Construct before patching decoder load: save() may load its source image too.
    data = b'\x89PNG\r\n\x1a\n'
    import struct, zlib
    def chunk(kind, value):
        return struct.pack('!I', len(value)) + kind + value + struct.pack('!I', zlib.crc32(kind + value))
    data += chunk(b'IHDR', struct.pack('!IIBBBBB', 30, 30, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(b'\0' * (30 * 3 + 1) * 30)) + chunk(b'IEND', b'')
    error(request(serving_app(), image_body(urls=[_url(data)])), 503,
          "prepared image budget is busy (1024 MiB); retry with fewer or smaller images", "server_error")


@pytest.mark.parametrize("fmt", ["TIFF", "BMP", "ICO", "TGA", "PPM", "PCX", "SGI", "DDS", "IM", "SPIDER"])
def test_data_uri_cannot_invoke_other_image_decoders(fmt):
    data = io.BytesIO()
    im = Image.new("F" if fmt == "SPIDER" else "RGB", (32, 32))
    im.save(data, fmt)
    wire = request(serving_app(), image_body(urls=[_url(data.getvalue())]))
    error(wire, 400, "image 1: the image could not be decoded", "invalid_request_error")


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "GIF"])
def test_allowed_data_formats_still_succeed(fmt):
    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, fmt)
    assert request(serving_app(), image_body(urls=[_url(data.getvalue())])).startswith(b"HTTP/1.1 200")


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/completions", "/tokenize"])
def test_body_cap_refuses_before_reading(monkeypatch, route):
    monkeypatch.setattr(server, "V2_PARITY", True)
    class NeverRead(io.BytesIO):
        def read(self, *a): pytest.fail("over-limit body was read")
    monkeypatch.setattr(io, "BytesIO", NeverRead)
    wire = request(app_stub(), b"{}", path=route, headers={"Content-Length": str(server.MAX_BODY_BYTES + 1)})
    error(wire, 413, "the request body exceeds 192 MiB", "invalid_request_error")


def test_body_cap_boundary_and_utf8(monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 2)
    assert request(app_stub(), b"{}").startswith(b"HTTP/1.1 200")
    error(request(app_stub(), b"{} "), 413, "the request body exceeds 192 MiB", "invalid_request_error")
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 100)
    for encoding in ("utf-8-sig", "utf-16", "utf-16-be", "utf-32", "utf-32-be"):
        assert request(app_stub(), "{}".encode(encoding)).startswith(b"HTTP/1.1 200")


def test_body_admission_retained_while_request_waits(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    app = app_stub()
    run = app.run
    def blocked(*a, **k):
        entered.set()
        assert release.wait(3)
        return run(*a, **k)
    app.run = blocked
    monkeypatch.setattr(server, "BODY_BUDGET_BYTES", server.BODY_MIN_CHARGE)
    with ThreadPoolExecutor(max_workers=1) as pool:
        f = pool.submit(request, app, {})
        try:
            assert entered.wait(2)
            error(request(app, {}), 503, "request body budget is busy (2048 MiB); retry later", "server_error")
            assert server._body_bytes == server.BODY_MIN_CHARGE
        finally:
            release.set()
        assert f.result(timeout=2).startswith(b"HTTP/1.1 200")
    assert server._body_bytes == 0


def test_json_container_charge_is_reserved_before_parsing(monkeypatch):
    raw = json.dumps({"x": [1, 2, 3]}).encode()
    monkeypatch.setattr(server, "BODY_MIN_CHARGE", 0)
    monkeypatch.setattr(server, "BODY_BUDGET_BYTES", 9 * len(raw))
    app = app_stub()
    app.prepare = lambda *a: pytest.fail("over-budget body reached prepare")
    error(request(app, raw), 503, "request body budget is busy (2048 MiB); retry later", "server_error")
    assert server._body_bytes == 0


def test_direct_app_run_also_releases_ownership():
    app = serving_app()
    app.run(image_body(), True, lambda delta: True)
    assert app.engine.request.images is None
    assert vision._image_count == vision._image_bytes == 0


@pytest.mark.parametrize("w,h", [(30, 30), (60, 40), (1000, 700)])
def test_reservation_equals_actual_tensor_storage(w, h):
    images = vision.prepare_images([_url(_png(w, h))])
    try:
        assert images.reserved == sum(im.patches.untyped_storage().nbytes() for im in images)
        assert vision._image_count == 1 and vision._image_bytes == images.reserved
    finally:
        images.close()
        images.close()                         # HTTP and run cleanup can both occur


@pytest.mark.parametrize("body", [b'{', b'[]', b'\xff', b'{"x":' + b'['*65 + b']'*65 + b'}'])
def test_parse_errors_release_body_charge(body):
    wire = request(app_stub(), body)
    assert wire.startswith((b"HTTP/1.1 400", b"HTTP/1.1 413"))
    assert server._body_bytes == 0


def test_json_structure_scanner_ignores_escaped_strings_and_counts_containers():
    body = {"prompt": 'a"[\\\\]b\\"{c' * 10000, "x": [[{}, {"z": 1}]]}
    raw = json.dumps(body).encode()
    assert server._json_structure(raw) == server._json_structure(json.dumps({**body, "prompt": ""}).encode())
    assert request(app_stub(), raw).startswith(b"HTTP/1.1 200")
    error(request(app_stub(), {"x": [[]] * 400000}), 413,
          "the request body exceeds JSON structure limits (1048576 markers, 64 levels)", "invalid_request_error")
