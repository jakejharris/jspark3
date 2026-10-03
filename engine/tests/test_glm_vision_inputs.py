"""GLM-5.3-Flash's image inputs before the GPU: the checkpoint processor's canvas and patches, image parts kept in
chat messages, and each image's placeholder expanded into its run of keyed tokens."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PIL = pytest.importorskip("PIL.Image")

from tensorfold.families.glm5_next.cuda import vision  # noqa: E402
from tensorfold.server.errors import RequestError  # noqa: E402
from tensorfold.server.messages import image_url, message_images, normalize_messages  # noqa: E402


def _png(w: int, h: int, color=(200, 30, 30), mode: str = "RGB") -> bytes:
    img = PIL.new(mode, (w, h), color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


@pytest.mark.parametrize("h, w, want", [
    (1080, 1920, (1092, 1932)),       # a screenshot: rounded up to 28, padded
    (700, 1000, (700, 1008)),
    (40, 60, (112, 140)),             # below 16 tokens: enlarged
    (2600, 3900, (2044, 3052)),       # above 8,000 tokens: refit
])
def test_canvas_matches_the_processor(h, w, want):
    assert vision.canvas(h, w) == want
    ch, cw = want
    assert 16 <= ch * cw // 28 ** 2 <= 8000


def test_prepare_pads_right_and_bottom_and_groups_2x2():
    img = vision.prepare(_png(1000, 700, (255, 0, 0)))
    assert img.grid == (1, 50, 72) and img.tokens == 900
    assert img.patches.shape == (50 * 72, 3 * 2 * 14 * 14)
    p = img.patches.view(25, 36, 2, 2, 3, 2, 14, 14)            # (gh/2, gw/2, mh, mw, C, t, ph, pw)
    red = (1.0 - vision.MEAN[0]) / vision.STD[0]
    pad = -vision.MEAN[0] / vision.STD[0]
    torch.testing.assert_close(p[0, 0, 0, 0, 0], torch.full((2, 14, 14), red), rtol=0, atol=1e-5)
    # the canvas is 1008 wide: the last patch column holds 6 image pixels, then 8 of padding
    last = p[0, 35, 0, 1, 0, 0]
    torch.testing.assert_close(last[:, :6], torch.full((14, 6), red), rtol=0, atol=1e-5)
    torch.testing.assert_close(last[:, 6:], torch.full((14, 8), pad), rtol=0, atol=1e-5)
    assert torch.equal(p[..., 0, :, :], p[..., 1, :, :])        # a still image repeats its frame


def test_transparency_is_composited_over_white():
    clear = vision.prepare(_png(56, 56, (0, 0, 0, 0), "RGBA"))
    white = vision.prepare(_png(56, 56, (255, 255, 255)))
    assert torch.equal(clear.patches, white.patches) and clear.digest != white.digest
    assert clear.key < 0 and vision.prepare(_png(56, 56, (0, 0, 0, 0), "RGBA")).key == clear.key


def test_read_takes_data_urls_only_or_http():
    data = _png(30, 30)
    assert vision.read(_url(data)) == data
    with pytest.raises(ValueError):
        vision.read("file:///etc/passwd")
    with pytest.raises(ValueError):
        vision.prepare(b"not an image")


def test_image_parts_stay_in_place_with_media():
    a, b = _url(_png(30, 30)), _url(_png(40, 40))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "look "}, {"type": "image_url", "image_url": {"url": a}},
                                     {"type": "text", "text": "and"}, {"type": "text", "text": " say"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "input_image", "image_url": b}]},
    ]
    out = normalize_messages(messages, media=True)
    assert out[1]["content"] == [{"type": "text", "text": "look "}, {"type": "image", "url": a},
                                 {"type": "text", "text": "and say"}]
    assert message_images(out) == [a, b]
    with pytest.raises(RequestError):
        normalize_messages(messages)                            # text-only servers still refuse images
    with pytest.raises(RequestError):
        normalize_messages([{"role": "system", "content": [{"type": "image_url", "image_url": {"url": a}}]}],
                           media=True)
    assert image_url({"type": "image_url", "image_url": a}) == a and image_url({"type": "text", "text": a}) is None


class _Template:
    def render(self, messages, *, tools, enable_thinking, extra=None, media=False):
        parts = []
        for m in normalize_messages(messages, media=media):
            content = m["content"]
            for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                parts.append("<|image|>" if part["type"] == "image" else part["text"])
        return "".join(parts)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, piece in enumerate(text.split("<|image|>")):
            if i:
                ids.append(999)
            ids.extend(ord(c) % 500 for c in piece)
        return SimpleNamespace(ids=ids)


def _app():
    from tensorfold.families.glm5_next.cuda.app import GlmApp

    app = object.__new__(GlmApp)
    app.engine = SimpleNamespace(tower=object(), w=SimpleNamespace(cfg=SimpleNamespace(image_token=999)), limit=10 ** 6)
    app.template, app.tok = _Template(), _Tokenizer()
    app.default_thinking, app.max_tokens, app.context_window = False, 64, 0
    return app


def test_each_placeholder_becomes_its_images_keyed_run():
    a, b = _png(60, 40), _png(1000, 700)
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "ab"}, {"type": "image_url", "image_url": {"url": _url(a)}},
        {"type": "text", "text": "c"}, {"type": "image_url", "image_url": {"url": _url(b)}}]}]}
    prepared = _app()._prepare(body, True)
    ia, ib = prepared.images
    assert (ia.tokens, ib.tokens) == (20, 900)
    want = [ord("a"), ord("b")] + [ia.key] * 20 + [ord("c")] + [ib.key] * 900
    assert prepared.prompt == want and 999 not in prepared.prompt
    text_only = _app()._prepare({"messages": [{"role": "user", "content": "hi"}]}, True)
    assert text_only.images is None and min(text_only.prompt) >= 0


def test_images_need_the_tower():
    app = _app()
    app.engine.tower = None
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _url(_png(9, 9))}}]}]}
    with pytest.raises(RequestError, match="text only"):
        app._prepare(body, True)


def test_image_runs_must_match_the_request():
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    img = vision.prepare(_png(60, 40))
    prompt = [1, 2] + [img.key] * img.tokens + [3]
    assert GlmEngine._image_starts(prompt, [img]) == [2]
    with pytest.raises(ValueError):
        GlmEngine._image_starts(prompt[:-3], [img])            # a cut run
    with pytest.raises(ValueError):
        GlmEngine._image_starts(prompt, [img, img])
    assert GlmEngine._image_starts([1, 2, 3], None) == []
    assert np.asarray(prompt).min() == img.key
