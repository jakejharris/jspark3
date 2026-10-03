"""GLM-5.3-Flash's image input on the GPU, on the tiny synthetic checkpoint of test_glm_engine.py with a tiny vision
tower: the tower equals a convolution-and-softmax reference in float64, image rows replace their placeholders'
embeddings in every prompt chunking, a conversation resumes only with the same images, and drafted replies with
images equal serial ones."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
PIL = pytest.importorskip("PIL.Image")

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import split, vision  # noqa: E402

from test_glm_engine import D, _checkpoint, _forget, _generate, _state, _TwoCopies  # noqa: E402

IMAGE, BEGIN, END = 999, 997, 998
VC = {"depth": 2, "hidden_size": 64, "num_heads": 4, "intermediate_size": 128, "patch_size": 14,
      "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": D, "projection_intermediate_size": 256,
      "rms_norm_eps": 1e-5, "swiglu_limit": 10.0, "in_channels": 3}


def _tower(path) -> None:
    """A random vision tower beside the checkpoint (``vision.safetensors``) and its config."""

    rng = np.random.default_rng(7)
    tensors = []
    H, I, O, P = VC["hidden_size"], VC["intermediate_size"], VC["out_hidden_size"], VC["projection_intermediate_size"]

    def bf16(name, shape, scale=0.05, offset=0.0):
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append(("model.visual." + name, "BF16", list(shape), x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    bf16("patch_embed.proj.weight", [H, 3, 2, 14, 14], 0.03)
    bf16("patch_embed.proj.bias", [H], 0.02)
    for i in range(VC["depth"]):
        p = f"blocks.{i}."
        bf16(p + "norm1.weight", [H], 0.05, 1.0)
        bf16(p + "norm2.weight", [H], 0.05, 1.0)
        bf16(p + "attn.qkv.weight", [3 * H, H], 0.12)
        bf16(p + "attn.qkv.bias", [3 * H], 0.02)
        bf16(p + "attn.q_norm.weight", [H // VC["num_heads"]], 0.05, 1.0)
        bf16(p + "attn.k_norm.weight", [H // VC["num_heads"]], 0.05, 1.0)
        bf16(p + "attn.proj.weight", [H, H], 0.12)
        bf16(p + "attn.proj.bias", [H], 0.02)
        for name, shape in (("gate_proj", [I, H]), ("up_proj", [I, H]), ("down_proj", [H, I])):
            bf16(p + f"mlp.{name}.weight", shape, 0.12)
            bf16(p + f"mlp.{name}.bias", [shape[0]], 0.02)
    bf16("post_layernorm.weight", [H], 0.05, 1.0)
    bf16("downsample.weight", [O, H, 2, 2], 0.06)
    bf16("downsample.bias", [O], 0.02)
    bf16("merger.proj.weight", [O, O], 0.04)
    bf16("merger.post_projection_norm.weight", [O], 0.05, 1.0)
    bf16("merger.post_projection_norm.bias", [O], 0.02)
    bf16("merger.gate_proj.weight", [P, O], 0.04)
    bf16("merger.up_proj.weight", [P, O], 0.04)
    bf16("merger.down_proj.weight", [O, P], 0.04)
    split.write(str(path / vision.VISION_FILE), tensors, None)
    config = json.loads((path / "config.json").read_text())
    config.update(vision_config=VC, image_token_id=IMAGE)
    (path / "config.json").write_text(json.dumps(config))


def _reference(path, image: vision.Image) -> torch.Tensor:
    """The tower as GLM-OCR's modules compute it (Conv3d, rotate_half rotary, softmax attention), in float64."""

    from safetensors.torch import load_file

    t = {k[len("model.visual."):]: v.double() for k, v in load_file(str(path / vision.VISION_FILE)).items()}
    H, heads = VC["hidden_size"], VC["num_heads"]
    hd = H // heads
    eps, L = VC["rms_norm_eps"], VC["swiglu_limit"]

    def rms(x, w, e=eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e) * w

    def mlp(x, gate, up, down, gb=None, ub=None, db=None):
        g, u = F.linear(x, gate, gb).clamp(max=L), F.linear(x, up, ub).clamp(-L, L)
        return F.linear(F.silu(g) * u, down, db)

    x = F.conv3d(image.patches.double().view(-1, 3, 2, 14, 14), t["patch_embed.proj.weight"],
                 t["patch_embed.proj.bias"], stride=(2, 14, 14)).view(-1, H)
    _, h, w = image.grid
    hp = torch.arange(h)[:, None].expand(h, w).reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()
    wp = torch.arange(w)[None, :].expand(h, w).reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd // 2, 2, dtype=torch.float64) / (hd // 2)))
    table = torch.outer(torch.arange(max(h, w), dtype=torch.float64), inv)
    emb = table[torch.stack([hp, wp], -1)].flatten(1)
    emb = torch.cat([emb, emb], -1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]

    def rotate(z):
        return torch.cat([-z[..., hd // 2:], z[..., :hd // 2]], -1)

    for i in range(VC["depth"]):
        p = f"blocks.{i}."
        q, k, v = F.linear(rms(x, t[p + "norm1.weight"]), t[p + "attn.qkv.weight"],
                           t[p + "attn.qkv.bias"]).view(-1, 3, heads, hd).unbind(1)
        q, k = rms(q, t[p + "attn.q_norm.weight"], 1e-5), rms(k, t[p + "attn.k_norm.weight"], 1e-5)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        a = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd), -1)
        x = x + F.linear(torch.einsum("hqk,khd->qhd", a, v).reshape(-1, H), t[p + "attn.proj.weight"],
                         t[p + "attn.proj.bias"])
        x = x + mlp(rms(x, t[p + "norm2.weight"]), t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"],
                    t[p + "mlp.down_proj.weight"], t[p + "mlp.gate_proj.bias"], t[p + "mlp.up_proj.bias"],
                    t[p + "mlp.down_proj.bias"])
    x = rms(x, t["post_layernorm.weight"]).view(-1, 2, 2, H).permute(0, 3, 1, 2)
    x = F.conv2d(x, t["downsample.weight"], t["downsample.bias"], stride=2).view(-1, VC["out_hidden_size"])
    x = F.linear(x, t["merger.proj.weight"])
    x = F.gelu(F.layer_norm(x, (x.shape[-1],), t["merger.post_projection_norm.weight"],
                            t["merger.post_projection_norm.bias"], 1e-5))
    return mlp(x, t["merger.gate_proj.weight"], t["merger.up_proj.weight"], t["merger.down_proj.weight"])


def _image(seed: int, w: int = 60, h: int = 40) -> vision.Image:
    rng = np.random.default_rng(seed)
    pixels = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PIL.fromarray(pixels).save(buf, "PNG")
    return vision.prepare(buf.getvalue())


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision")
    _checkpoint(path)
    _tower(path)
    e = GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), prefill_rows=32)    # images straddle chunks
    e.path = path
    return e


def _prompt(seed: int, images, text: int = 30) -> list[int]:
    rng = np.random.default_rng(seed)
    out = [int(t) for t in rng.integers(0, 990, size=text)]
    for img in images:
        out += [BEGIN] + [img.key] * img.tokens + [END] + [int(t) for t in rng.integers(0, 990, size=7)]
    return out


def _run(engine, prompt, images, sampling, *, draft=True, tokens=16):
    engine.request.images = images
    try:
        return _generate(engine, prompt, sampling, draft=draft, tokens=tokens)
    finally:
        engine.request.images = None


@pytest.mark.parametrize("size", [(60, 40), (300, 200), (57, 443)])
def test_tower_equals_the_convolution_reference(engine, size):
    image = _image(1, *size)
    got = engine.tower.encode(image).double().cpu()
    want = _reference(engine.path, image)
    assert got.shape == (image.tokens, VC["out_hidden_size"])
    assert float((got - want).norm() / want.norm()) < 4e-3        # the bf16 rows' rounding
    assert float(F.cosine_similarity(got, want, dim=-1).min()) > 0.9999


def test_image_rows_replace_the_placeholders_in_every_chunking(engine):
    """Prefill chunks cut images anywhere: every chunking leaves the same state and first token as one chunk."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    images = [_image(2), _image(3, 90, 70)]
    prompt = _prompt(4, images)
    rows = engine._image_rows(prompt, 0, images)
    assert rows[0].size == sum(i.tokens for i in images)
    want = None
    for n in (len(prompt), 7, 16, 32):
        e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=n)
        e.images = rows
        first = prefill(e, prompt, None)
        state = [t.clone() for t in _state(e)]
        if want is None:
            want = first, state
            continue
        assert first == want[0], n
        assert all(torch.equal(a, b) for a, b in zip(state, want[1])), n
    e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=16)
    prefill(e, prompt, None)                                     # the placeholder's own embedding instead
    assert not all(torch.equal(a, b) for a, b in zip(_state(e), want[1]))


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_with_images_equal_serial(engine, sampling):
    images = [_image(5)]
    prompt = _prompt(6, images)
    _forget(engine)
    drafted, _ = _run(engine, prompt, images, sampling)
    serial, _ = _run(engine, prompt, images, sampling, draft=False)
    assert drafted == serial


def test_conversations_resume_only_with_the_same_images(engine):
    sampling = Sampling(77, 1.0, 20, 0.95)
    a, b = _image(8), _image(9)
    _forget(engine)
    first = _prompt(10, [a])
    reply, _ = _run(engine, first, [a], sampling)
    later_image = _image(11, 80, 50)
    after = first + reply + [5, 6] + [BEGIN] + [later_image.key] * later_image.tokens + [END, 7]
    warm, stats = _run(engine, after, [a, later_image], sampling)
    assert stats["cached"] == len(first)                        # the earlier image is not encoded again
    _forget(engine)
    cold, stats = _run(engine, after, [a, later_image], sampling)
    assert stats["cached"] == 0 and warm == cold
    # the same text and token count with another image: nothing past the image resumes
    _run(engine, first, [a], sampling)
    other = [b.key if t == a.key else t for t in first]
    got, stats = _run(engine, other + [3], [b], sampling)
    assert stats["cached"] < first.index(a.key)
    _forget(engine)
    fresh, _ = _run(engine, other + [3], [b], sampling)
    assert got == fresh


def test_a_request_whose_images_do_not_match_is_refused_before_rank_1(engine):
    img = _image(12)
    prompt = _prompt(13, [img])
    with pytest.raises(ValueError):
        _run(engine, prompt[:-12], [img], None)                 # a cut image run
    with pytest.raises(ValueError):
        _run(engine, prompt, [img, img], None)
