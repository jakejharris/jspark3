"""Animated image inputs use frame zero, including repeated tool-result history."""

import base64
import io

import pytest
import torch
from PIL import Image

from tensorfold.families.glm5_next.cuda import vision
from tensorfold.server.errors import RequestError
from test_glm_vision_inputs import _app, _png, _url
from test_glm_image_hardening import remote, no_network  # noqa: F401


def gif():
    first = Image.new("P", (56, 56), 0)
    first.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
    second = first.copy()
    second.paste(1, (0, 0, 56, 56))
    buf = io.BytesIO()
    first.save(buf, "GIF", save_all=True, append_images=[second], duration=100, loop=0)
    return buf.getvalue()


def test_animated_palette_gif_uses_only_first_frame(monkeypatch):
    data = gif()
    from PIL.GifImagePlugin import GifImageFile

    original = GifImageFile.seek

    def first_only(self, frame):
        assert frame == 0, "image input must not decode later animation frames"
        return original(self, frame)

    monkeypatch.setattr(GifImageFile, "seek", first_only)
    image = vision.prepare(data)
    reference = vision.prepare(_png(56, 56, (255, 0, 0)))
    assert image.grid == reference.grid
    assert torch.equal(image.patches, reference.patches)
    assert image.digest == vision.prepare(data).digest


@pytest.mark.parametrize("older_images", [7, 6])
def test_gif_in_replayed_and_compacted_history_does_not_wedge_request(older_images):
    url = "data:image/gif;base64," + base64.b64encode(gif()).decode()
    parts = [{"type": "image_url", "image_url": {"url": _url(_png(28, 28))}} for _ in range(older_images)]
    parts += [{"type": "image_url", "image_url": {"url": url}}]
    body = {"messages": [{"role": "user", "content": parts}]}
    for _ in range(2):
        prepared = _app()._prepare(body, True)
        try:
            assert len(prepared.images) == older_images + 1
            assert prepared.images[-1].tokens > 0
        finally:
            prepared.images.close()


def test_remote_gif_header_and_magic_agree(remote):
    remote.data = gif()
    remote.headers = {"Content-Type": "image/gif"}
    data = vision.read("https://images.example/animation.gif", allow_remote=True)
    assert vision.prepare(data).tokens > 0


def test_transparent_gif_frame_is_composited_over_white():
    img = Image.new("P", (56, 56), 0)
    img.putpalette([255, 0, 0] + [0] * 765)
    buf = io.BytesIO()
    img.save(buf, "GIF", transparency=0)
    prepared = vision.prepare(buf.getvalue())
    assert torch.equal(prepared.patches, vision.prepare(_png(56, 56, (255, 255, 255))).patches)


def test_corrupt_gif_still_returns_numbered_request_error():
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lGODlh"}}]}]}
    with pytest.raises(RequestError, match="image 1: the image could not be decoded"):
        _app()._prepare(body, True)
