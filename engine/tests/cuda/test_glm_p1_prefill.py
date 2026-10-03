"""Bounded prefill bit gates, adapted from upstream #128/#140 for TP3's 22 heads and independent flags."""

import hashlib

import pytest
import torch

from tensorfold.families.glm5_next.cuda import forward, latent, prefill_options as p1, sparse


def bits(t):
    return t.contiguous().view(torch.uint8)


def same(got, want):
    assert torch.equal(bits(got), bits(want))
    # The gate must catch a one-bit error, including in a padded/dense-boundary row.
    broken = bits(got).clone()
    broken.reshape(-1)[-1] ^= 1
    assert not torch.equal(broken, bits(want))
    print("sha256", hashlib.sha256(bits(got).cpu().numpy().tobytes()).hexdigest())


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for flag in ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "DEAD_WORK"):
        monkeypatch.setattr(p1, flag, False)


@pytest.mark.parametrize("pos,rows", [(0, n) for n in range(1, 33)] + [(0, 64), (0, 1057), (511, 513)] + [(2051 - n, 128) for n in (1, 2, 63, 64, 65)])
def test_dense_blocking_and_boundary_prefix_keep_bits(pos, rows):
    gen = torch.Generator(device="cuda").manual_seed(pos + rows)
    heads, width, nch = 22, 512, latent.chunks_for(pos + rows)
    cache = torch.randn((pos + rows + 1, width), dtype=torch.bfloat16, device="cuda", generator=gen)
    qa = torch.randn((rows, heads, width), dtype=torch.bfloat16, device="cuda", generator=gen) * 0.05
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    whole = latent.LatentScratch(rows, heads, nch, "cuda")
    want = latent.attention(qa, cache, at, whole, scale=0.0625, nch=nch, out=torch.empty_like(qa)).clone()
    del whole
    s = latent.LatentScratch(rows, heads, nch, "cuda", part_rows=512)
    for partial in (s.po, s.pm, s.pl):
        partial.fill_(float("nan"))
    got = torch.full_like(qa, float("nan"))
    forward.dense_attention(qa, cache, at, s, scale=0.0625, nch=nch, out=got)
    same(got, want)
    dense_n = min(rows, max(0, 2051 - pos))
    got.fill_(float("nan"))
    forward.dense_attention(qa[:dense_n], cache, at, s, scale=0.0625, nch=nch, out=got[:dense_n],
                            hb=latent.head_block(rows))
    same(got[:dense_n], want[:dense_n])
    assert got[dense_n:].isnan().all()
    shifted = torch.empty_like(got[:dense_n])
    forward.dense_attention(qa[:dense_n], cache, at + 1, s, scale=0.0625, nch=nch, out=shifted,
                            hb=latent.head_block(rows))
    assert not torch.equal(bits(shifted), bits(want[:dense_n])), "offset negative control did not fail"
    if rows > 512:
        with pytest.raises(ValueError, match="past the scratch"):
            latent.attention(qa, cache, at, s, scale=0.0625, nch=nch, out=got)


@pytest.mark.parametrize("pos,rows", [(0, 32), (2047, 65), (8191, 513), (130000, 513), (262144 - 33, 33)])
def test_selection_flags_keep_tokens_and_counts(monkeypatch, pos, rows):
    gen = torch.Generator(device="cuda").manual_seed(pos + rows)
    heads, dim, capacity = 32, 128, 262144
    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=gen)
    pk, qi, wts = rand(capacity // 4 + 2, dim), rand(rows, heads * dim), rand(rows, heads)
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    want = sparse.select_tokens(qi, wts, pk, pos, rows, capacity // 4, at)
    for blocks, visible in ((True, False), (False, True), (True, True)):
        monkeypatch.setattr(p1, "SELECT_BLOCKS", blocks)
        monkeypatch.setattr(p1, "VISIBLE_POOLS", visible)
        got = sparse.select_tokens(qi, wts, pk, pos, rows, capacity // 4, at)
        same(got[0], want[0])
        same(got[1], want[1])


@pytest.mark.parametrize("pos", [0, 2048, 2051, 4095, 205000])
@pytest.mark.parametrize("ties", [False, True])
def test_visible_radix_ties_and_graphs(pos, ties):
    torch.manual_seed(127)
    rows, width = 6, 65536
    scores = torch.zeros((rows, width), device="cuda") if ties else torch.rand((rows, width), device="cuda") - 0.5
    # Signed zero and -inf ties must still choose the lowest pool index.
    scores[:, 0] = -0.0
    visible = (pos + torch.arange(rows, device="cuda") + 1) // 4
    scores.masked_fill_(torch.arange(width, device="cuda")[None, :] >= visible[:, None], float("-inf"))
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    want = sparse._top_pools(scores, 512)
    same(sparse.top_pools(scores, 512, at), want)
    for _ in range(3):
        sparse.top_pools(scores, 512, at)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = sparse.top_pools(scores, 512, at)
    graph.replay()
    torch.cuda.synchronize()
    same(replayed, want)


def test_selection_graph_reads_device_position(monkeypatch):
    monkeypatch.setattr(p1, "SELECT_BLOCKS", True)
    monkeypatch.setattr(p1, "VISIBLE_POOLS", True)
    torch.manual_seed(3)
    rows, width, capacity = 8, 128, 32768
    qi = torch.randn((rows, 32 * width), dtype=torch.bfloat16, device="cuda")
    wts = torch.randn((rows, 32), dtype=torch.bfloat16, device="cuda")
    pk = torch.randn((capacity // 4 + 2, width), dtype=torch.bfloat16, device="cuda")
    at = torch.tensor([2051], dtype=torch.int32, device="cuda")
    def run():
        return sparse.select_tokens(qi, wts, pk, None, rows, capacity // 4, at, bucket=8192)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run()
    for pos in (2051, 4095, 8191, 16380):
        at.fill_(pos)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setattr(p1, "SELECT_BLOCKS", False)
        monkeypatch.setattr(p1, "VISIBLE_POOLS", False)
        want = run()
        same(got[0], want[0])
        same(got[1], want[1])
        monkeypatch.setattr(p1, "SELECT_BLOCKS", True)
        monkeypatch.setattr(p1, "VISIBLE_POOLS", True)
