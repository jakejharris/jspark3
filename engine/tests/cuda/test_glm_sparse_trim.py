"""Native byte gates for sparse trimming, including partials, dispatch and graph replay."""

import hashlib

import numpy as np
import pytest
import torch
import triton

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import latent, prefill_options

COUNTS = [0, 1, 2, 31, 32, 33, 511, 512, 513, 2048, 2049, 2051]


def inputs(rows, context=32768, heads=22, width=512):
    rng = np.random.default_rng(917 + rows + context)
    tokens = np.full((rows, 2051), -1, dtype=np.int32)
    counts = np.empty(rows, dtype=np.int32)
    for r in range(rows):
        pools = np.sort(rng.choice((context - 4) // 4, 512, replace=False))
        tokens[r, :2048] = (pools[:, None] * 4 + np.arange(4)).reshape(-1)
        tail = r % 4
        counts[r] = 2048 + tail
        tokens[r, 2048:2048 + tail] = np.arange(context - 4, context - 4 + tail)
    gen = torch.Generator(device="cuda").manual_seed(827)
    qa = (torch.randn((rows, heads, width), generator=gen, device="cuda") * .04).bfloat16()
    cache = torch.randn((context, width), generator=gen, device="cuda").bfloat16()
    return qa, cache, torch.from_numpy(tokens).cuda(), torch.from_numpy(counts).cuda()


def buffers(x):
    rows, heads, width = x[0].shape
    n = triton.cdiv(x[2].shape[1], 512) * rows * heads
    return (torch.empty(n * width, device="cuda"), torch.empty(n, device="cuda"),
            torch.empty(n, device="cuda"), torch.full_like(x[0], 17))


def byte_equal(got, want):
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))


def run_chunks(x, b, trimmed, hb=None):
    qa, cache, tokens, counts = x
    rows, heads, width = qa.shape
    hb = latent.head_block(rows) if hb is None else hb
    nch = triton.cdiv(tokens.shape[1], 512)
    kernel = latent._sparse_chunks_trim if trimmed else latent._sparse_chunks
    grid = (rows * triton.cdiv(heads, hb) * nch,) if trimmed else (rows, triton.cdiv(heads, hb), nch)
    kernel[grid](qa, cache, tokens, counts, *b[:3], rows, W=tokens.shape[1], H=heads, LW=width,
                 CH=512, SCALE=.0625, HBT=hb, KTT=32, num_warps=8, num_stages=3 if trimmed else 1)
    latent._merge[(rows, heads)](*b[:3], b[3], counts, rows, H=heads, LW=width,
                                 NCH=nch, SPARSE=True, num_warps=4)


@pytest.mark.parametrize("rows", [1, 12, 63, 64, 65])
@pytest.mark.parametrize("values", ["normal", "large_query", "zero_query", "signed_zero", "query_nan", "cache_inf"])
def test_partial_and_final_bytes_at_selection_boundaries(rows, values):
    x = inputs(rows)
    qa, cache, tokens, counts = x
    if values == "large_query":
        qa.mul_(400)
    elif values == "zero_query":
        qa.zero_()
    elif values == "signed_zero":
        qa.zero_(); qa[..., ::2] = -0.0
        cache.zero_(); cache[:, ::2] = -0.0
    elif values == "query_nan":
        qa[:, 0, 0] = float("nan")
    elif values == "cache_inf":
        cache[tokens[:, 0].long(), 0] = float("inf")
        cache[tokens[:, 1].long(), 0] = float("-inf")
    want, got = buffers(x), buffers(x)
    # A one-row launch covers every boundary too, including a fully empty row.
    for offset in range(len(COUNTS) if rows == 1 else 1):
        counts.copy_(torch.tensor([COUNTS[(r + offset) % len(COUNTS)] for r in range(rows)], device="cuda"))
        # Give every boundary count a valid tail; remaining padding is masked.
        for r in range(rows):
            tokens[r, 2048:] = torch.arange(32764, 32767, device="cuda", dtype=torch.int32)
        run_chunks(x, want, False)
        run_chunks(x, got, True)
        for a, b in zip(got, want):
            byte_equal(a, b)
        if bool((counts == 0).any()):
            byte_equal(got[3][counts == 0], torch.full_like(got[3][counts == 0], 17))


@pytest.mark.parametrize("rows,context,hb,width", [
    (64, 32768, 16, 512), (65, 32768, 32, 512), (512, 32768, None, 512),
    (2048, 32768, None, 512), (2048, 131072, None, 512), (65, 32768, 16, 128),
])
def test_native_partials_dispatch_and_negative_control(monkeypatch, rows, context, hb, width):
    x = inputs(rows, context, width=width)
    want, got = buffers(x), buffers(x)
    run_chunks(x, want, False, hb)
    run_chunks(x, got, True, hb)
    for a, b in zip(got, want):
        byte_equal(a, b)
    out = torch.full_like(x[0], 17)
    for enabled in (False, True):
        monkeypatch.setattr(prefill_options, "SPARSE_TRIM", enabled)
        latent.sparse_attention(*x, out, .0625, hb=hb)
        byte_equal(out, want[3])
    print("rows", rows, "context", context, "hb", hb, "width", width, "output_sha256",
          hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
    for a, b in zip(got, want):
        a.view(torch.uint8).flatten()[0] ^= 1
        with pytest.raises(AssertionError):
            byte_equal(a, b)


def test_cuda_graph_replay_reads_changed_counts_and_preserves_empty_rows(monkeypatch):
    x = inputs(65)
    out = torch.full_like(x[0], 17)
    monkeypatch.setattr(prefill_options, "SPARSE_TRIM", True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        latent.sparse_attention(*x, out, .0625)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent.sparse_attention(*x, out, .0625)
    x[3][:] = 33
    x[3][0] = 0
    out.fill_(17)
    graph.replay()
    want = torch.full_like(out, 17)
    monkeypatch.setattr(prefill_options, "SPARSE_TRIM", False)
    latent.sparse_attention(*x, want, .0625)
    byte_equal(out, want)


@pytest.mark.parametrize("tiled", [False, True])
def test_prefill_resume_and_continuation_have_stock_bytes(model, monkeypatch, tiled):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda import decode, seqpar
    from test_glm_p1_state import engine, same, state

    # _Copies replicates all-gather partials; it cannot emulate point-to-point SP.
    monkeypatch.setattr(seqpar, "ENABLED", False)
    monkeypatch.setattr(seqpar, "OVERLAP", False)
    w, _ = model
    prompt = list(np.random.default_rng(171).integers(0, 1000, 2181))
    sampling = Sampling(721, .7, 20, .95)
    for flag in ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "ATTENTION_TILES"):
        monkeypatch.setattr(prefill_options, flag, tiled)
    reference = None
    for enabled in (False, True):
        monkeypatch.setattr(prefill_options, "SPARSE_TRIM", enabled)
        e = engine(w, 513, span=64, images=True)
        snapshots = []

        def keep(snap):
            decode.save_rows(e, snap)
            snapshots.append(snap)

        first = decode.prefill(e, prompt, sampling, mtp=True,
                               mark=lambda n: 2051 if n < 2051 else None, keep=keep)
        cold, logits = state(e), e.pbuf.logits[:1].clone()
        decode.load_rows(e, snapshots[0])
        assert decode.prefill(e, prompt, sampling, mtp=True, resume=snapshots[0]) == first
        same(state(e), cold); byte_equal(e.pbuf.logits[:1], logits)
        tokens = decode.serial_decode(e, first, 16, sampling).tokens
        if reference is None:
            reference = first, cold, logits, tokens
        else:
            assert first == reference[0] and tokens == reference[3]
            same(cold, reference[1]); byte_equal(logits, reference[2])


# Tiny random-weight TP3 fixture; real TP3 hashes require a separate serving check.
from test_glm_p1_state import model  # noqa: E402,F401
