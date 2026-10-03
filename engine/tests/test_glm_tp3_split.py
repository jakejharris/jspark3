"""GLM-5.3-Flash over three ranks, host side (no GPU): the dims that do not divide by three are zero-padded at their
end, every rank's share is whole heads and whole 64-input groups, the vocabulary head splits unevenly without
padding, and the padding leaves every projection's output exactly as it was."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm5_next.cuda import split  # noqa: E402
from tensorfold.families.glm5_next.cuda.dflash2 import _part, rank_heads  # noqa: E402
from tensorfold.families.glm5_next.cuda.weights import load, vocab_share  # noqa: E402
from tests import glm5_tp3_fakes as fakes  # noqa: E402

# GLM-5.3-Flash's own sizes (Vontra/GLM-5.3-Flash-MLX-4bit-MTP config.json)
GLM53 = {"text_config": {"num_attention_heads": 64, "linear_attn_config": {"num_heads": 64},
                         "moe_intermediate_size": 2048, "intermediate_size": 12288, "vocab_size": 154880}}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return fakes.write(tmp_path_factory.mktemp("tp3") / "ckpt")


def _bytes(t: torch.Tensor) -> np.ndarray:
    """A tensor's bytes as rows of its leading dim: [shape[0], rest]."""

    return t.contiguous().view(torch.uint8).numpy().reshape(t.shape[0], -1)


def test_glm53_pads_heads_to_66_and_experts_to_2112_over_three_ranks():
    assert split.padded_dims(GLM53, 3) == {"heads": (64, 66), "lin": (64, 66), "moe": (2048, 2112),
                                           "dense": (12288, 12288)}
    assert all(real == padded for real, padded in split.padded_dims(GLM53, 2).values())
    assert [vocab_share(154880, r, 3) for r in range(3)] == [(0, 51648), (51648, 103296), (103296, 154880)]
    assert [vocab_share(154880, r, 2) for r in range(2)] == [(0, 77440), (77440, 154880)]
    assert rank_heads(32, 8, 3) == (12, 3) and rank_heads(32, 8, 2) == (16, 4)       # the DFlash2 drafter


def test_three_shares_are_the_tensor_then_zeros(ckpt):
    config = json.loads((ckpt / "config.json").read_text())
    source = fakes.tensors()
    readers = [split.RankReader(ckpt, r, 3) for r in range(3)]
    pads = split.padded_dims(config, 3)
    seen = set()
    for name, (dtype, shape, values) in source.items():
        full = fakes.stored(dtype, values).reshape(shape[0], -1)
        parts = [_bytes(rd.get(name)) for rd in readers]
        kind = split.rule(name)
        if kind == "rep":
            assert all(np.array_equal(p, full) for p in parts), name
            continue
        assert len({p.shape for p in parts}) == 1, name                       # every rank holds the same shape
        axis = 0 if kind == "row" else 1
        joined = np.concatenate(parts, axis=axis)
        real, padded = pads[split.dim_key(name, config)]
        assert joined.shape[axis] * real == full.shape[axis] * padded, name
        n = full.shape[axis]
        kept = joined[:n] if axis == 0 else joined[:, :n]
        extra = joined[n:] if axis == 0 else joined[:, n:]
        assert np.array_equal(kept, full) and not extra.any(), name          # the tensor, then zeros
        seen.add(split.dim_key(name, config))
    assert seen == {"heads", "lin", "moe", "dense"}


def test_presplit_folders_read_like_the_checkpoint(ckpt, tmp_path):
    for rank in range(3):
        out = tmp_path / f"rank{rank}"
        split.main([str(ckpt), "--rank", str(rank), "--world", "3", str(out)])
        assert split.rank_files(out, rank, 3) and not split.rank_files(out, rank, 2)
        whole, folder = split.RankReader(ckpt, rank, 3), split.RankReader(out, rank, 3)
        assert folder.split and not whole.split
        for name in fakes.tensors():
            assert torch.equal(_as_u8(whole.get(name)), _as_u8(folder.get(name))), name
    with pytest.raises(ValueError):
        split.RankReader(tmp_path / "rank0", 1, 3)       # rank 0's folder given to rank 1
    with pytest.raises(ValueError):
        split.RankReader(tmp_path / "rank0", 0, 2)       # a three-rank folder given to a two-rank engine


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def test_every_rank_loads_whole_heads_and_groups(ckpt):
    ws = [load(ckpt, rank=r, world=3, device="cpu") for r in range(3)]
    for w in ws:
        kda_layer, mla_layer = w.layers
        assert w.cfg.heads == w.cfg.lin_heads == 6 and w.cfg.moe_width == w.cfg.shared_width == 192
        assert kda_layer.kda.heads == 2 and mla_layer.dsa.heads == w.mtp.layer.dsa.heads == 2
        assert kda_layer.kda.proj.n % 64 == 0 and kda_layer.kda.proj.n >= 3 * 2 * 128 + 256 + 2
        assert mla_layer.dsa.q_b.n == 2 * 256 and mla_layer.dsa.o.k == 2 * 256 and kda_layer.kda.o.k == 2 * 128
        assert mla_layer.moe.experts.width == 64 and kda_layer.mlp.width == fakes.DENSE // 3
    assert [w.head.n for w in ws] == [384, 384, 232]                # the vocabulary: 64-row shares, the last shorter
    assert [w.vocab_offset for w in ws] == [0, 384, 768]
    two = load(ckpt, rank=0, world=2, device="cpu")                 # two ranks: halves, nothing padded
    assert two.layers[1].dsa.heads == 2 and two.layers[1].moe.experts.width == 64 and two.head.n == 500
    assert two.layers[0].kda.proj.n == 3 * 2 * 128 + 256 + 2


def _dequant(words: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX affine 4-bit rows (words [n, k / 8], scales and biases [n, k / 64]) -> float64 [n, k]."""

    q = (words.astype(np.uint64)[:, :, None] >> (4 * np.arange(8, dtype=np.uint64))) & 0xF
    q = q.reshape(words.shape[0], -1).astype(np.float64)
    return q * np.repeat(scales.astype(np.float64), 64, axis=1) + np.repeat(biases.astype(np.float64), 64, axis=1)


def _matrix(rd: split.RankReader, name: str) -> np.ndarray:
    words = rd.get(name + ".weight").view(torch.int32).numpy().view(np.uint32)
    return _dequant(words, rd.get(name + ".scales").float().numpy(), rd.get(name + ".biases").float().numpy())


def _source(name: str) -> np.ndarray:
    """The checkpoint's matrix, from the values the fake wrote (BF16 metadata truncated as stored)."""

    t = fakes.tensors()
    bf16 = lambda v: ((v.astype(np.float32).view(np.uint32) >> 16) << 16).view(np.float32)   # noqa: E731
    return _dequant(t[name + ".weight"][2], bf16(t[name + ".scales"][2]), bf16(t[name + ".biases"][2]))


def test_zero_padding_adds_nothing_to_any_projection(ckpt):
    """Column-parallel rows past the checkpoint compute 0; row-parallel inputs past it meet zero columns, so the
    ranks' partials add up (in rank order) to the unpadded product."""

    rng = np.random.default_rng(1)
    readers = [split.RankReader(ckpt, r, 3) for r in range(3)]
    pre = "model.language_model.layers"
    # column-parallel (rows split): MLA queries, KDA values, an expert's gate, the MTP head's shared expert's up
    for name in (f"{pre}.1.self_attn.q_b_proj", f"{pre}.0.self_attn.v_proj", f"{pre}.1.mlp.experts.3.gate_proj",
                 f"{pre}.2.mlp.shared_experts.up_proj"):
        w = _source(name)
        x = rng.standard_normal(w.shape[1])
        out = np.concatenate([_matrix(rd, name) @ x for rd in readers])
        assert np.allclose(out[:w.shape[0]], w @ x, rtol=0, atol=1e-12) and not out[w.shape[0]:].any(), name
    # row-parallel (columns split): MLA and KDA outputs, routed and shared experts' down projections, the MTP head's
    for name in (f"{pre}.1.self_attn.o_proj", f"{pre}.0.self_attn.o_proj", f"{pre}.1.mlp.experts.2.down_proj",
                 f"{pre}.1.mlp.shared_experts.down_proj", f"{pre}.2.self_attn.o_proj", f"{pre}.0.mlp.down_proj"):
        w = _source(name)
        k = w.shape[1]
        parts = [_matrix(rd, name) for rd in readers]
        per = parts[0].shape[1]
        x = rng.standard_normal(per * 3)                 # whatever a padded input holds, zero columns ignore it
        got = sum(p @ x[r * per:(r + 1) * per] for r, p in enumerate(parts))
        assert np.allclose(got, w @ x[:k], rtol=0, atol=1e-9), name
        assert not np.concatenate(parts, axis=1)[:, k:].any(), name


def test_drafter_parts_pad_heads_at_the_end():
    t = torch.arange(32 * 4, dtype=torch.float32).view(32, 4)         # 8 kv heads of 4 rows: rank r of 3 takes 12
    parts = [_part(t, 0, r * 12, 12) for r in range(3)]
    assert torch.equal(torch.cat(parts)[:32], t) and not torch.cat(parts)[32:].any()
    assert _part(t, 1, 2, 2).data_ptr() == t[:, 2:4].data_ptr()       # a whole slice is a view, as before
    assert torch.equal(_part(t, 1, 3, 2), torch.cat([t[:, 3:4], torch.zeros(32, 1)], dim=1))


def test_the_memory_estimate_sizes_padded_shares_from_either_source(ckpt, tmp_path):
    """Admission's per-rank weight estimate: the full checkpoint through the padding transform and the rank's
    pre-split folder (read as it is) agree; the padded ranks estimate more than an unpadded third."""

    from tensorfold.cuda.capacity import estimate_weights
    from tensorfold.cuda.geometry import split_weights

    config = json.loads((ckpt / "config.json").read_text())
    transform = split_weights(split.rule, 3, lambda name, kind: split.split_pad(name, kind, config, 3))
    for rank in range(3):
        out = tmp_path / f"rank{rank}"
        split.main([str(ckpt), "--rank", str(rank), "--world", "3", str(out)])
        whole = estimate_weights(ckpt, transform, rank=rank)
        folder = estimate_weights(out, transform, rank=rank, files=split.rank_files(out, rank, 3))
        assert whole.resident == folder.resident, rank
    halves = estimate_weights(ckpt, split_weights(split.rule, 2), rank=0)
    assert whole.resident > halves.resident * 2 // 3


def test_estimates_see_the_padded_sizes():
    t = dict(GLM53["text_config"])
    assert split.padded_config(t, 2) is t
    p = split.padded_config(t, 3)
    assert (p["num_attention_heads"], p["linear_attn_config"]["num_heads"], p["moe_intermediate_size"],
            p["intermediate_size"]) == (66, 66, 2112, 12288)
    assert t["num_attention_heads"] == 64                      # the config itself is left as it was
