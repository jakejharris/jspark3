"""GLM-5.3-Flash's image input: the checkpoint processor's canvas and patches, and its vision tower (bf16, rank 0).

A prompt carries an image as a run of placeholder tokens; the tower's rows replace their embeddings. In the engine
those tokens are keyed by the image's hash (``key``: a negative id), so a cached conversation resumes only when its
images are the same ones; staging maps keyed ids back to the placeholder token.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import io
import ipaddress
import json
import math
import socket
import ssl
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.server.errors import RequestError

PATCH, TEMPORAL, MERGE = 14, 2, 2
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
MIN_TOKENS, MAX_TOKENS = 16, 8000                 # processor_config.json's image token budget
MAX_BYTES = 32 * 2 ** 20                          # an encoded image, as sent or fetched
MAX_PIXELS = 32_000_000                         # checked before the decoder allocates pixels
MAX_IMAGES = 16                                # one request cannot retain an unbounded image list
PREP_SECONDS = 10.0                            # includes DNS, TLS, headers, body and CPU preparation
PREP_SLOTS = 2
IMAGE_REQUESTS = 4                             # includes preparation, queueing and generation
PREPARED_BYTES = 1024 * 2 ** 20                 # aggregate reserved patch storage, including active preparation
_slots = threading.BoundedSemaphore(PREP_SLOTS)
_workers = ThreadPoolExecutor(max_workers=PREP_SLOTS, thread_name_prefix="image-prepare")
_image_lock = threading.Lock()
_image_count = _image_bytes = 0
VISION_FILE = "vision.safetensors"                # the tower beside a rank folder's weights (``split.py``)
PREFIX = "model.visual."


@dataclass(slots=True)
class Image:
    """One prepared image: its patches [t*h*w, 3*2*14*14] fp32, grid (t, h, w) and content hash."""

    patches: torch.Tensor
    grid: tuple[int, int, int]
    digest: bytes

    @property
    def tokens(self) -> int:
        t, h, w = self.grid
        return t * h * w // (MERGE * MERGE)

    @property
    def key(self) -> int:
        """The negative id its placeholder tokens carry in the engine (31 bits of the hash)."""

        return -(1 + int.from_bytes(self.digest[:4], "little") % (2 ** 31 - 2))


# -- the processor (Glm5NextImageProcessor, resize_mode "pad") ---------------------------------------------------
class ImageBusy(RequestError):
    """Admission is full; never enqueue more image work than there are workers."""

    status = 503


class PreparedImages(list):
    """Own admission and patch-byte reservations until the request releases its images."""

    def __init__(self):
        global _image_count
        super().__init__()
        self.closed, self.reserved = True, 0
        with _image_lock:
            if _image_count >= IMAGE_REQUESTS:
                raise ImageBusy("image requests are busy (4 active or waiting); retry later")
            _image_count += 1
            self.closed = False

    def reserve(self, size):
        global _image_bytes
        with _image_lock:
            if self.closed or _image_bytes + size > PREPARED_BYTES:
                raise ImageBusy("prepared image budget is busy (1024 MiB); retry with fewer or smaller images")
            _image_bytes += size
            self.reserved += size

    def close(self):
        global _image_count, _image_bytes
        with _image_lock:
            if not self.closed:
                self.clear()                   # free tensors before returning their memory reservation
                _image_bytes -= self.reserved
                _image_count -= 1
                self.reserved, self.closed = 0, True

    def __del__(self):
        self.close()                           # direct prepare callers; HTTP/run also close deterministically


class _Preparation:
    def __init__(self):
        self.deadline = time.monotonic() + PREP_SECONDS
        self.lock = threading.Lock()
        self.sock = None
        self.cancelled = False

    def remaining(self):
        left = self.deadline - time.monotonic()
        if self.cancelled or left <= 0:
            raise ValueError("image preparation exceeded 10 seconds")
        return left

    def attach(self, sock):
        with self.lock:
            if self.cancelled:
                sock.close()
                raise ValueError("image preparation exceeded 10 seconds")
            self.sock = sock

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.sock.close()


def _bounded(work, cleanup=None):
    slots = _slots
    if not slots.acquire(blocking=False):
        if cleanup is not None:
            cleanup()
        raise ImageBusy("image preparation is busy (2 slots); retry later")
    job = _Preparation()
    def execute():
        try:
            return work(job)
        except BaseException as exc:
            # Do not let a completed Future retain decoder temporaries after its CPU slot becomes free.
            traceback.clear_frames(exc.__traceback__)
            raise
    try:
        future = _workers.submit(execute)
    except BaseException:
        slots.release()
        if cleanup is not None:
            cleanup()
        raise
    # A resolver/decoder that cannot be interrupted keeps its slot until it returns.
    # Timed-out callers therefore cannot grow an unbounded executor queue.
    future.add_done_callback(lambda _: slots.release())
    delivered = False
    try:
        result = future.result(timeout=job.remaining())
        job.remaining()
        delivered = True
        return result
    except FutureTimeout:
        raise ValueError("image preparation exceeded 10 seconds") from None
    except ValueError:
        if time.monotonic() >= job.deadline:
            raise ValueError("image preparation exceeded 10 seconds") from None
        raise
    finally:
        job.cancel()
        if not delivered and cleanup is not None:
            def dispose(done):
                exc = done.exception()
                if exc is not None:
                    traceback.clear_frames(exc.__traceback__)
                cleanup()
            future.add_done_callback(dispose)    # native work must exit before its storage/admission is released


def _remote(url: str, job: _Preparation) -> bytes:
    try:
        u = urllib.parse.urlsplit(url)
        host = u.hostname
        valid = (u.scheme == "https" and host and u.port in (None, 443) and
                 u.username is None and u.password is None and not u.fragment and "%" not in host)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("remote image URLs require HTTPS on port 443 without credentials or fragments")
    host = host.encode("idna").decode("ascii")
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        job.remaining()
        if not addresses:
            raise OSError("no addresses")
        for _, _, _, _, addr in addresses:
            ip = ipaddress.ip_address(addr[0])
            if isinstance(ip, ipaddress.IPv6Address) and (ip.sixtofour or ip.teredo):
                raise ValueError("remote image URL must resolve only to global unicast addresses")
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if (not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified or
                    getattr(ip, "is_site_local", False)):
                raise ValueError("remote image URL must resolve only to global unicast addresses")
        family, kind, proto, _, address = addresses[0]
        raw = socket.socket(family, kind, proto)
        job.attach(raw)
        raw.settimeout(job.remaining())
        raw.connect(address)                    # numeric sockaddr from this lookup: no second DNS resolution
        context = ssl.create_default_context()
        # Attach before the handshake so the wall-clock timeout can interrupt TLS too.
        tls = context.wrap_socket(raw, server_hostname=host, do_handshake_on_connect=False)
        job.attach(tls)
        tls.settimeout(job.remaining())
        tls.do_handshake()
        conn = http.client.HTTPSConnection(host, 443, context=context)
        conn.auto_open = 0                       # never reconnect through HTTPConnection's hostname lookup
        conn.sock = tls
        response = None
        try:
            tls.settimeout(job.remaining())
            conn.request("GET", urllib.parse.urlunsplit(("", "", u.path or "/", u.query, "")),
                         headers={"Accept": "image/jpeg, image/png, image/webp", "Accept-Encoding": "identity"})
            response = conn.getresponse()
            if 300 <= response.status < 400:
                raise ValueError("remote image redirects are not allowed")
            if response.status != 200:
                raise ValueError("remote image server must return HTTP 200")
            mime = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                raise ValueError("remote image Content-Type must be image/jpeg, image/png or image/webp")
            if response.getheader("Content-Encoding", "identity").lower() not in ("", "identity"):
                raise ValueError("remote image Content-Encoding must be identity")
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    size = int(length)
                except ValueError:
                    raise ValueError("remote image Content-Length is invalid") from None
                if size < 0:
                    raise ValueError("remote image Content-Length is invalid")
                if size > MAX_BYTES:
                    raise ValueError("an image is larger than 32 MB")
            data = bytearray()
            while True:
                tls.settimeout(job.remaining())
                part = response.read1(min(65536, MAX_BYTES + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
                if len(data) > MAX_BYTES:
                    raise ValueError("an image is larger than 32 MB")
            if length is not None and len(data) != size:
                raise ValueError("remote image body does not match Content-Length")
            magic = ("image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else
                     "image/jpeg" if data.startswith(b"\xff\xd8\xff") else
                     "image/webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else None)
            if magic != mime:
                raise ValueError("remote image bytes do not match Content-Type")
            job.remaining()
            return bytes(data)
        finally:
            if response is not None:
                response.close()
            conn.close()
    except (OSError, http.client.HTTPException):
        job.remaining()
        raise ValueError("remote image fetch failed") from None


def _read(url: str, allow_remote: bool, job: _Preparation) -> bytes:

    if not isinstance(url, str) or not url:
        raise ValueError("an image part needs a URL")
    if url.startswith("data:"):
        # Bound the URI before partitioning or decoding (percent escapes use at most 3 chars per byte).
        if len(url) > 3 * MAX_BYTES + 256:
            raise ValueError("image data URL exceeds the 32 MB encoding limit")
        if not url.isascii() or "," not in url:
            raise ValueError("the image data URL does not decode")
        head, _, body = url.partition(",")
        if len(head) > 256 or (head.endswith(";base64") and len(body) > 4 * ((MAX_BYTES + 2) // 3)):
            raise ValueError("image data URL exceeds the 32 MB encoding limit")
        try:
            data = base64.b64decode(body, validate=True) if head.endswith(";base64") else \
                urllib.parse.unquote_to_bytes(body)
        except (binascii.Error, ValueError):
            raise ValueError("the image data URL does not decode") from None
    elif url.lower().startswith(("http://", "https://")):
        if not allow_remote:
            raise ValueError("remote image URLs are disabled; send a data: URI or start with --image-urls")
        data = _remote(url, job)
    else:
        raise ValueError("image URLs must be data: URIs (or HTTPS with --image-urls)")
    if len(data) > MAX_BYTES:
        raise ValueError("an image is larger than 32 MB")
    return data


def read(url: str, *, allow_remote: bool = False) -> bytes:
    """Read an image with bounded admission/time; remote fetch requires explicit opt-in."""

    return _bounded(lambda job: _read(url, allow_remote, job))


def prepare_images(urls: list[str], *, allow_remote: bool = False) -> list[Image]:
    if len(urls) > MAX_IMAGES:
        raise ValueError("a request may contain at most 16 images")

    images = PreparedImages()

    def work(job):
        size = 0
        for i, url in enumerate(urls, 1):
            job.remaining()
            try:
                data = _read(url, allow_remote, job)
                size += len(data)
                if size > MAX_BYTES:
                    raise ValueError("images in a request exceed 32 MB in total")
                images.append(prepare(data, reserve=images.reserve))
            except ImageBusy:
                raise
            except (ValueError, OSError) as exc:
                raise ValueError(f"image {i}: {exc}") from None
        job.remaining()
        return images

    return _bounded(work, cleanup=images.close)


def _ceil(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _fit(t: int, h: int, w: int, factor: int, max_pixels: int) -> tuple[int, int]:
    """The largest proportional canvas whose upward-aligned size fits the budget (binary search on the height)."""

    low, high = 1, h
    best = factor, factor
    while low <= high:
        ch = (low + high) // 2
        cw = max(1, math.floor(w * ch / h))
        ah, aw = _ceil(ch, factor), _ceil(cw, factor)
        if t * ah * aw <= max_pixels:
            best = ah, aw
            low = ch + 1
        else:
            high = ch - 1
    return best


def canvas(h: int, w: int, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """The processor's ``smart_resize``: the height and width rounded up to 28, refit into the token budget."""

    factor = PATCH * MERGE
    per = TEMPORAL * factor * factor
    min_pixels, max_pixels = min_tokens * per, max_tokens * per
    t = TEMPORAL
    hb, wb = _ceil(h, factor), _ceil(w, factor)
    if t * hb * wb > max_pixels:
        return _fit(t, h, w, factor, max_pixels)
    if t * hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        hb, wb = _ceil(max(1, math.ceil(h * beta)), factor), _ceil(max(1, math.ceil(w * beta)), factor)
        if t * hb * wb > max_pixels:
            return _fit(t, h, w, factor, max_pixels)
    return hb, wb


def prepare(data: bytes, *, max_tokens: int = MAX_TOKENS, reserve=None) -> Image:
    """Decode, fit onto the canvas keeping the aspect ratio (bicubic, zero padding right and bottom), normalize, patch."""

    from PIL import Image as PILImage

    if len(data) > MAX_BYTES:
        raise ValueError("an image is larger than 32 MB")
    digest = hashlib.sha256(data).digest()
    try:
        img = PILImage.open(io.BytesIO(data), formats=("JPEG", "PNG", "WEBP"))
    except PILImage.DecompressionBombError:
        raise ValueError("an image exceeds 32000000 decoded pixels") from None
    except Exception:  # noqa: BLE001 - do not echo decoder internals or input bytes
        raise ValueError("the image could not be decoded") from None
    if img.width * img.height > MAX_PIXELS:
        img.close()
        raise ValueError("an image exceeds 32000000 decoded pixels")
    if reserve is not None:
        ch, cw = canvas(img.height, img.width, max_tokens=max_tokens)
        try:
            reserve((ch // PATCH) * (cw // PATCH) * (3 * TEMPORAL * PATCH * PATCH) * 4)
        except BaseException:
            img.close()
            raise
    try:
        img.load()
    except Exception:  # noqa: BLE001
        img.close()
        raise ValueError("the image could not be decoded") from None
    if img.mode != "RGB":                         # transformers' convert_to_rgb: transparency over white
        rgba = img.convert("RGBA")
        white = PILImage.new("RGBA", rgba.size, (255, 255, 255))
        img = PILImage.alpha_composite(white, rgba).convert("RGB")
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)      # [3, H, W] uint8
    H, W = x.shape[1:]
    ch, cw = canvas(H, W, max_tokens=max_tokens)
    scale = min(ch / H, cw / W)
    if TEMPORAL * H * W >= MIN_TOKENS * TEMPORAL * (PATCH * MERGE) ** 2:
        scale = min(1.0, scale)                   # small images are enlarged only below the minimum budget
    th, tw = max(1, min(ch, math.floor(H * scale))), max(1, min(cw, math.floor(W * scale)))
    if (th, tw) != (H, W):                        # torchvision's tensor resize: float bicubic, antialiased, rounded
        x = F.interpolate(x[None].float(), size=(th, tw), mode="bicubic", align_corners=False, antialias=True)[0]
        x = x.round_().clamp_(0, 255).to(torch.uint8)
    x = F.pad(x, (0, cw - tw, 0, ch - th), value=0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    x = (x.float() * (1 / 255.0) - mean) / std
    gh, gw = ch // PATCH, cw // PATCH
    frames = x[None].expand(TEMPORAL, 3, ch, cw)                   # a still image repeats its frame
    p = frames.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                       # (t, gh, gw, mh, mw, C, tp, ph, pw)
    return Image(p.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH).contiguous(), (1, gh, gw), digest)


# -- the vision tower ---------------------------------------------------------------------------------------------
def _tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """The ``model.visual.*`` tensors: a rank folder's vision file, else the checkpoint's own files."""

    from safetensors import safe_open

    path = model_dir / VISION_FILE
    if path.exists():
        files = {path: None}
    else:
        index = model_dir / "model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(f"{model_dir}: no {VISION_FILE} and no checkpoint index to find the vision tower")
        names = json.loads(index.read_text())["weight_map"]
        files = {model_dir / f: None for n, f in names.items() if n.startswith(PREFIX)}
        if not files:
            raise FileNotFoundError(f"{model_dir}: this checkpoint has no vision tower")
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for n in h.keys():
                if n.startswith(PREFIX):
                    out[n[len(PREFIX):]] = h.get_tensor(n)
    return out


def available(model_dir: Path) -> bool:
    if (model_dir / VISION_FILE).exists():
        return True
    index = model_dir / "model.safetensors.index.json"
    return index.exists() and any(n.startswith(PREFIX) for n in json.loads(index.read_text())["weight_map"])


def tower_bytes(model_dir: Path) -> int:
    """The tower's weights as loaded (bf16), for the startup memory estimate."""

    from .split import read_header

    path = model_dir / VISION_FILE
    if path.exists():
        header = read_header(path)[0]
    else:
        index = model_dir / "model.safetensors.index.json"
        if not index.exists():
            return 0
        names = json.loads(index.read_text())["weight_map"]
        header = {}
        for f in sorted({f for n, f in names.items() if n.startswith(PREFIX)}):
            header.update(read_header(model_dir / f)[0])
    return sum(math.prod(v["shape"]) * 2 for k, v in header.items() if k.startswith(PREFIX))


# rows of the tower's MLP, of its merger and queries of its attention at a time: the scratch of an 8,000-token
# image (32,000 patches, fp32) peaks at 1.18 GiB, in its attention
ROWS = 4096
MERGE_ROWS = 2048
QUERIES = 4096
WORKSPACE = 5 * 2 ** 28                           # 1.25 GiB for the startup estimate


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _tf32(x: torch.Tensor) -> torch.Tensor:
    """x's leading 11 significant bits (TF32 keeps them exactly); x - _tf32(x) holds the rest."""

    return (x.contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _attention(device: torch.device):
    """The memory-efficient attention kernel on CUDA (fp32 products at fp32 accuracy); any kernel elsewhere."""

    from contextlib import nullcontext

    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


class Tower:
    """The vision transformer (24 blocks, 2D rotary, q/k norms), the 2x2 downsample and the SwiGLU merger.

    Its weights stay bf16 as stored; every product is fp32. The tower amplifies rounding: run all in bf16 (as
    transformers and vLLM run it) its rows for a photo are 7% off the exact ones (some below cosine 0.84), with
    TF32 products 1.7%. NVIDIA's containers force TF32 on cuBLAS, so a product is two TF32 products (an input's
    leading bits, then the rest; bf16 weights are exact in TF32), and attention runs the memory-efficient
    kernel, whose fp32 products are split alike.
    """

    def __init__(self, model_dir: Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        v = cfg["vision_config"]
        self.depth, self.dim, self.heads = int(v["depth"]), int(v["hidden_size"]), int(v["num_heads"])
        self.out = int(v["out_hidden_size"])
        self.eps = float(v.get("rms_norm_eps", 1e-5))
        self.limit = float(v.get("swiglu_limit") or 10.0)
        if (int(v["patch_size"]), int(v["temporal_patch_size"]), int(v["spatial_merge_size"])) != (PATCH, TEMPORAL,
                                                                                                    MERGE):
            raise ValueError("this vision tower's patch geometry differs from GLM-5.3-Flash's")
        # matrices in ``dtype``; norms and biases fp32
        t = {k: x.to(device=device, dtype=dtype if x.dim() > 1 else torch.float32)
             for k, x in _tensors(Path(model_dir)).items()}
        t["patch_embed.proj.weight"] = t["patch_embed.proj.weight"].reshape(self.dim, -1)
        t["downsample.weight"] = t["downsample.weight"].reshape(self.out, -1)
        self.t = t
        hd = self.dim // self.heads
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd // 2, 2, dtype=torch.float32, device=device)
                                           / (hd // 2)))
        self.device = device

    def _lin(self, x: torch.Tensor, name: str, bias: bool = True, part: slice = slice(None)) -> torch.Tensor:
        """x @ W[part].T (+ b[part]), fp32."""

        w = self.t[name + ".weight"][part].float()
        hi = _tf32(x)
        y = F.linear(hi, w)
        y += F.linear(torch.sub(x, hi, out=hi), w)
        return y.add_(self.t[name + ".bias"][part]) if bias else y

    def _rotary(self, grid: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        t, h, w = grid
        hp = torch.arange(h, device=self.device)[:, None].expand(h, w)
        wp = torch.arange(w, device=self.device)[None, :].expand(h, w)

        def order(p):
            return p.reshape(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).reshape(-1)

        pos = torch.stack([order(hp), order(wp)], dim=-1).repeat(t, 1).float()    # [N, 2]
        freqs = (pos[:, :, None] * self.inv_freq[None, None, :]).reshape(pos.shape[0], -1)   # [N, hd/2]
        return freqs.cos(), freqs.sin()

    def _block(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        t, p = self.t, f"blocks.{i}."
        N, H = x.shape[0], self.heads
        hd = self.dim // H
        h = _rms(x, t[p + "norm1.weight"], self.eps)
        c, s = cos[:, None, :], sin[:, None, :]

        def rope(z):                               # neox halves over the whole head: [h freqs | w freqs] twice
            a, b = z[..., :hd // 2], z[..., hd // 2:]
            return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

        def head(i, norm=None):                    # q, k or v: [1, H, N, hd] (one at a time: less scratch)
            z = self._lin(h, p + "attn.qkv", part=slice(i * self.dim, (i + 1) * self.dim)).view(N, H, hd)
            if norm is not None:
                z = rope(_rms(z, t[p + norm], 1e-5))
            return z.transpose(0, 1).contiguous()[None]

        q, k, v = head(0, "attn.q_norm.weight"), head(1, "attn.k_norm.weight"), head(2)
        del h
        att = torch.empty_like(q)
        with _attention(q.device):
            for a in range(0, N, QUERIES):
                att[:, :, a:a + QUERIES] = F.scaled_dot_product_attention(q[:, :, a:a + QUERIES], k, v)
        x = x + self._lin(att[0].transpose(0, 1).reshape(N, self.dim), p + "attn.proj")
        for a in range(0, N, ROWS):
            h = _rms(x[a:a + ROWS], t[p + "norm2.weight"], self.eps)
            act = _swiglu(self._lin(h, p + "mlp.gate_proj"), self._lin(h, p + "mlp.up_proj"), self.limit)
            x[a:a + ROWS] += self._lin(act, p + "mlp.down_proj")
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t
        x = self._lin(x, "merger.proj", bias=False)
        x = F.gelu(F.layer_norm(x, (self.out,), t["merger.post_projection_norm.weight"],
                                t["merger.post_projection_norm.bias"], 1e-5))
        act = _swiglu(self._lin(x, "merger.gate_proj", False), self._lin(x, "merger.up_proj", False), self.limit)
        return self._lin(act, "merger.down_proj", bias=False)

    @torch.no_grad()
    def encode(self, image: Image) -> torch.Tensor:
        """An image's rows for the language model: [tokens, out_hidden] bf16."""

        try:
            return self._encode(image)
        finally:
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()            # the scratch back for the prefill

    def _encode(self, image: Image) -> torch.Tensor:
        x = self._lin(image.patches.to(self.device), "patch_embed.proj")
        cos, sin = self._rotary(image.grid)
        for i in range(self.depth):
            x = self._block(i, x, cos, sin)
        x = _rms(x, self.t["post_layernorm.weight"], self.eps)
        m = x.shape[0] // (MERGE * MERGE)
        x = x.view(m, MERGE, MERGE, self.dim).permute(0, 3, 1, 2).reshape(m, -1)    # (C, kh, kw) like the conv
        x = self._lin(x, "downsample")
        return torch.cat([self._merge(x[a:a + MERGE_ROWS]).to(torch.bfloat16) for a in range(0, m, MERGE_ROWS)])


def write_vision(src: Path, out: Path) -> int:
    """Copy the checkpoint's ``model.visual.*`` tensors into OUT/vision.safetensors (a rank 0 folder); their count."""

    from .split import write, read_header

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for f in sorted({f for n, f in index.items() if n.startswith(PREFIX)}):
        header, base = read_header(src / f)
        mm = np.memmap(src / f, dtype=np.uint8, mode="r")
        for name in sorted(header):
            if name.startswith(PREFIX):
                a, b = header[name]["data_offsets"]
                tensors.append((name, header[name]["dtype"], header[name]["shape"], np.array(mm[base + a:base + b])))
    if tensors:
        write(str(out / VISION_FILE), tensors, None)
    return len(tensors)
