"""Bounded image input. All DNS, sockets and TLS below are in-process stubs; no network."""

import base64
import io
import ipaddress
import socket
import threading
import time
from types import SimpleNamespace as NS
from urllib.parse import quote_from_bytes

import pytest

from tensorfold.families.glm5_next.cuda import vision
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tensorfold.server.errors import RequestError
from test_glm_vision_inputs import _app, _png, _url


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("test attempted real network access")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


class StubSocket:
    def __init__(self, wire):
        self.wire, self.sent, self.address = wire, [], None
        self.closed = threading.Event()
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, address):
        self.address = address

    def do_handshake(self):
        pass

    def sendall(self, data):
        self.sent.append(bytes(data))

    def makefile(self, *args):
        return io.BytesIO(self.wire)

    def shutdown(self, how):
        self.closed.set()

    def close(self):
        self.closed.set()


@pytest.fixture
def remote(monkeypatch):
    data = _png(30, 30)
    state = NS(data=data, status=200, headers={"Content-Type": "image/png"}, addresses=["8.8.8.8"],
               sockets=[], lookups=[], tls=[])

    def resolve(host, port, **kwargs):
        state.lookups.append((host, port))
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 "", (ip, port)) for ip in state.addresses]

    def connect(*args):
        head = f"HTTP/1.1 {state.status} Stub\r\n"
        head += "".join(f"{k}: {v}\r\n" for k, v in state.headers.items())
        sock = StubSocket(head.encode() + b"\r\n" + state.data)
        state.sockets.append(sock)
        return sock

    def wrap(sock, **kwargs):
        state.tls.append(kwargs)
        return sock

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", connect)
    monkeypatch.setattr(vision.ssl, "create_default_context", lambda: NS(wrap_socket=wrap))
    return state


def test_data_uris_still_prepare_identical_pixels():
    data = _png(60, 40)
    assert vision.read(_url(data)) == data
    assert vision.read("data:image/png," + quote_from_bytes(data)) == data
    direct = vision.prepare(data)
    got = vision.prepare_images([_url(data)])[0]
    assert got.grid == direct.grid and got.digest == direct.digest
    assert got.patches.equal(direct.patches)


@pytest.mark.parametrize("url", ["http://images.example/p.png", "https://images.example/p.png"])
def test_remote_is_refused_before_dns_by_default(url):
    with pytest.raises(ValueError, match="remote image URLs are disabled"):
        vision.read(url)


def test_opt_in_uses_numeric_checked_address_and_original_tls_name(remote):
    assert vision.read("https://images.example:443/a.png?q=1", allow_remote=True) == remote.data
    assert remote.lookups == [("images.example", 443)]
    sock, = remote.sockets
    assert sock.address == ("8.8.8.8", 443)
    assert remote.tls == [dict(server_hostname="images.example", do_handshake_on_connect=False)]
    assert b"Host: images.example\r\n" in b"".join(sock.sent)
    assert b"GET /a.png?q=1 HTTP/1.1\r\n" in b"".join(sock.sent)
    assert sock.closed.is_set() and all(0 < t <= 10 for t in sock.timeouts)


def test_dns_rebind_cannot_cause_a_second_hostname_connection(remote, monkeypatch):
    checked = socket.getaddrinfo
    def rebind(*args, **kwargs):
        if remote.lookups:
            pytest.fail("a second DNS lookup would resolve to loopback")
        return checked(*args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", rebind)
    vision.read("https://images.example/p.png", allow_remote=True)
    assert remote.sockets[0].address == ("8.8.8.8", 443)


@pytest.mark.parametrize("ip", ["127.0.0.1", str(ipaddress.IPv4Address(0x0a000001)), str(ipaddress.IPv4Address(0xa9fea9fe)), str(ipaddress.IPv4Address(0x64400001)), "224.0.0.1",
    "0.0.0.0", "240.0.0.1", "::1", "::", str(ipaddress.IPv6Address((0xfe80 << 112) | 1)), str(ipaddress.IPv6Address((0xfec0 << 112) | 1)), str(ipaddress.IPv6Address((0xfc00 << 112) | 1)), "ff02::1", "2001:db8::1",
    "::ffff:127.0.0.1", str(ipaddress.IPv6Address((0xffff << 32) | 0x64400001)), "::ffff:224.0.0.1", "2002:7f00:1::", "2001::1"])
def test_every_dns_address_must_be_global_unicast(remote, ip):
    remote.addresses.append(ip)  # a public first answer does not excuse a private second answer
    with pytest.raises(ValueError, match="global unicast"):
        vision.read("https://images.example/p.png", allow_remote=True)
    assert not remote.sockets


@pytest.mark.parametrize("url", ["http://images.example/p.png", "https://images.example:444/p.png",
    "https://u:p" + "@" + "images.example/p.png", "https://images.example/p.png#fragment", f"https://[{ipaddress.IPv6Address((0xfe80 << 112) | 1)}%25eth0]/p.png",
    "https://images.example:bad/p.png", "https:///p.png"])
def test_opt_in_url_rules_run_before_dns(url):
    with pytest.raises(ValueError, match="require HTTPS on port 443"):
        vision.read(url, allow_remote=True)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_never_follow_location(remote, status):
    remote.status = status
    remote.headers["Location"] = "https://127.0.0.1/private"
    with pytest.raises(ValueError, match="redirects are not allowed"):
        vision.read("https://images.example/p.png", allow_remote=True)
    assert len(remote.lookups) == len(remote.sockets) == 1


@pytest.mark.parametrize("header, value, message", [
    ("Content-Type", "text/html", "Content-Type must"),
    ("Content-Type", "image/jpeg", "bytes do not match"),
    ("Content-Encoding", "gzip", "Content-Encoding must"),
    ("Content-Length", "-1", "Content-Length is invalid"),
    ("Content-Length", "abc", "Content-Length is invalid"),
    ("Content-Length", "999999999", "larger than 32 MB"),
    ("Content-Length", "1000", "does not match Content-Length"),
])
def test_remote_header_magic_and_size_rules(remote, header, value, message):
    remote.headers[header] = value
    with pytest.raises(ValueError, match=message):
        vision.read("https://images.example/p.png", allow_remote=True)
    assert remote.sockets[0].closed.is_set()


@pytest.mark.parametrize("kind, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_three_remote_image_formats_by_header_and_magic(remote, kind, mime):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, kind)
    remote.data = buf.getvalue()
    remote.headers = {"Content-Type": mime + "; charset=binary", "Content-Length": str(len(remote.data))}
    assert vision.read("https://images.example/p", allow_remote=True) == remote.data


def test_remote_stream_cannot_exceed_byte_bound_without_a_length(remote, monkeypatch):
    monkeypatch.setattr(vision, "MAX_BYTES", len(remote.data) - 1)
    with pytest.raises(ValueError, match="larger than 32 MB"):
        vision.read("https://images.example/p.png", allow_remote=True)


def test_tls_failure_is_sanitized_and_closes_the_socket(remote, monkeypatch):
    def handshake(self):
        raise vision.ssl.SSLError("private certificate diagnostic")
    monkeypatch.setattr(StubSocket, "do_handshake", handshake)
    with pytest.raises(ValueError, match="^remote image fetch failed$"):
        vision.read("https://images.example/p.png", allow_remote=True)
    assert remote.sockets[0].closed.is_set()


def test_non_200_is_a_request_error(remote):
    remote.status = 500
    with pytest.raises(ValueError, match="must return HTTP 200"):
        vision.read("https://images.example/p.png", allow_remote=True)


def test_data_encoding_limits_before_and_after_decode(monkeypatch):
    monkeypatch.setattr(vision, "MAX_BYTES", 6)
    with pytest.raises(ValueError, match="encoding limit"):
        vision.read("data:image/png;base64," + "A" * 12)
    with pytest.raises(ValueError, match="encoding limit"):
        vision.read("data:," + "%41" * 100)
    with pytest.raises(ValueError, match="larger than 32 MB"):
        vision.read("data:,1234567")
    with pytest.raises(ValueError, match="does not decode"):
        vision.read("data:;base64,???")
    with pytest.raises(ValueError, match="does not decode"):
        vision.read("data:,é")
    with pytest.raises(ValueError, match="does not decode"):
        vision.read("data:image/png")
    with pytest.raises(ValueError, match="larger than 32 MB"):
        vision.prepare(b"1234567")


def test_decoded_pixel_cap_is_checked_before_load(monkeypatch):
    from PIL import Image
    closed = []
    monkeypatch.setattr(Image, "open", lambda *a, **k: NS(width=8000, height=4001,
                        load=lambda: pytest.fail("oversized pixels were decoded"), close=lambda: closed.append(True)))
    with pytest.raises(ValueError, match="32000000 decoded pixels"):
        vision.prepare(b"stub")
    assert closed == [True]


def test_request_image_count_and_aggregate_bytes_are_bounded(monkeypatch):
    with pytest.raises(ValueError, match="at most 16 images"):
        vision.prepare_images(["data:,"] * 17)
    monkeypatch.setattr(vision, "MAX_BYTES", 5)
    monkeypatch.setattr(vision, "prepare", lambda data, **k: data)
    with pytest.raises(ValueError, match="image 2: images in a request exceed 32 MB in total"):
        vision.prepare_images(["data:,abc", "data:,def"])


def test_app_opt_in_is_forwarded_and_errors_keep_the_image_number(remote):
    app = _app()
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url":
                        "https://images.example/p.png"}]}]}
    with pytest.raises(RequestError, match="image 1: remote image URLs are disabled"):
        app._prepare(body, True)
    app.image_urls = True
    assert app._prepare(body, True).images[0].tokens > 0


def test_wall_deadline_includes_dns_and_retains_the_slot_until_worker_exits(monkeypatch):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(vision, "_slots", slots)
    monkeypatch.setattr(vision, "PREP_SECONDS", 0.05)
    def stalled(*a, **k):
        entered.set()
        assert release.wait(2)
        done.set()
        return []
    monkeypatch.setattr(socket, "getaddrinfo", stalled)
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="exceeded 10 seconds"):
            vision.read("https://images.example/p.png", allow_remote=True)
        assert entered.is_set() and time.monotonic() - started < 1
        with pytest.raises(vision.ImageBusy):
            vision.prepare_images(["data:,x"])
    finally:
        release.set()
    assert done.wait(2)
    assert slots.acquire(timeout=2)  # timeout must not release before the resolver completes
    slots.release()


def test_completed_work_cannot_return_success_past_the_deadline():
    def late(job):
        job.deadline = time.monotonic() - 1
        return b"too late"
    with pytest.raises(ValueError, match="^image preparation exceeded 10 seconds$"):
        vision._bounded(late)


def test_wall_deadline_aborts_slow_headers(remote, monkeypatch):
    monkeypatch.setattr(vision, "PREP_SECONDS", 0.05)
    def makefile(self, *args):
        class Stalled(io.BytesIO):
            def readline(inner, *args):
                assert self.closed.wait(2)
                return b""
        return Stalled()
    monkeypatch.setattr(StubSocket, "makefile", makefile)
    with pytest.raises(ValueError, match="exceeded 10 seconds"):
        vision.read("https://images.example/p.png", allow_remote=True)
    assert remote.sockets[0].closed.is_set()


def test_two_slots_also_cover_data_uri_preparation(monkeypatch):
    held = []
    try:
        for _ in range(2):
            assert vision._slots.acquire(timeout=2)
            held.append(True)
        with pytest.raises(vision.ImageBusy, match=r"busy \(2 slots\)") as exc:
            vision.prepare_images([_url(_png(2, 2))])
        assert exc.value.status == 503
    finally:
        for _ in held:
            vision._slots.release()
    assert vision.prepare_images([_url(_png(2, 2))])


def test_slots_are_released_after_decode_errors():
    for _ in range(20):
        with pytest.raises(ValueError, match="could not be decoded"):
            vision.prepare_images(["data:,broken"])
    assert vision.prepare_images([_url(_png(2, 2))])
