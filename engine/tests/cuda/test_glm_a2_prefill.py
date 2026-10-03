"""Attention tile, new chunk-size and express-lane byte gates on a bounded synthetic TP3 rank."""

import copy

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode, forward, latent, prefill_options as p1, sparse
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from test_glm_p1_prefill import same as same_tensor
from test_glm_p1_state import model, state, same
from test_glm_batched import _stream


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for flag in ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "DEAD_WORK",
                 "ATTENTION_TILES", "EXPRESS"):
        monkeypatch.setattr(p1, flag, False)


@pytest.mark.parametrize("rows", list(range(1, 33)) + [64, 513, 1025, 8192])
def test_sparse_attention_tile_hashes(rows, monkeypatch):
    # R8192's untiled sparse partials fit below 3.5 GiB, unlike its dense partials.
    torch.manual_seed(904 + rows)
    pos, capacity, heads = 2047, 16384, 22
    qa = torch.randn((rows, heads, 512), dtype=torch.bfloat16, device="cuda") * .04
    cache = torch.randn((capacity, 512), dtype=torch.bfloat16, device="cuda")
    qi = torch.randn((rows, 4096), dtype=torch.bfloat16, device="cuda")
    wts = torch.randn((rows, 32), dtype=torch.bfloat16, device="cuda")
    pk = torch.randn((capacity // 4 + 2, 128), dtype=torch.bfloat16, device="cuda")
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    tokens, counts = sparse.select_tokens(qi, wts, pk, pos, rows, capacity // 4, at)
    want = torch.full_like(qa, 17)
    latent.sparse_attention(qa, cache, tokens, counts, want, .0625)
    del tokens, counts
    got = torch.full_like(qa, 17)
    forward.sparse_attention_tiles(qa, cache, qi, wts, pk, pos, at, got, .0625)
    same_tensor(got, want)
    if rows <= 32:
        # Force a tail under 64 while retaining the original segment's kernel.
        monkeypatch.setattr(p1, "ATTENTION_ROWS", 7)
        forward.sparse_attention_tiles(qa, cache, qi, wts, pk, pos, at, got, .0625)
        same_tensor(got, want)
    # Deliberately use the wrong absolute position; sparse rows must change.
    if rows >= 64:
        forward.sparse_attention_tiles(qa, cache, qi, wts, pk, pos + 4, at + 4, got, .0625)
        assert not torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@pytest.mark.parametrize("rows", [4096, 8192])
def test_new_pass_sizes_equal_2048_and_resumed_state(model, monkeypatch, rows):
    w, _ = model
    prompt = list(np.random.default_rng(901).integers(0, 1000, rows + 65))
    sample = Sampling(79, .7, 20, .95)
    baseline = decode.Engine(w, capacity=12288, prefill_rows=2048, long_context=True)
    first = decode.prefill(baseline, prompt, sample, mtp=True)
    want, logits = state(baseline), baseline.pbuf.logits.clone()
    del baseline
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    candidate = decode.Engine(w, capacity=12288, prefill_rows=rows, long_context=True)
    assert decode.prefill(candidate, prompt, sample, mtp=True) == first
    same(state(candidate), want); same_tensor(candidate.pbuf.logits, logits)
    cut = rows - 63
    kept = []
    def keep(snap):
        decode.save_rows(candidate, snap)
        kept.append(snap)
    assert decode.prefill(candidate, prompt, sample, mtp=True,
                          mark=lambda n: cut if n < cut else None, keep=keep) == first
    same(state(candidate), want)
    decode.load_rows(candidate, kept[0])
    assert decode.prefill(candidate, prompt, sample, mtp=True, resume=kept[0]) == first
    same(state(candidate), want); same_tensor(candidate.pbuf.logits, logits)


@pytest.mark.parametrize("sampling", [None, Sampling(99, .71, 20, .95)])
@pytest.mark.parametrize("cancel", [None, "long", "short"])
def test_express_while_long_suspended_matches_alone(model, monkeypatch, sampling, cancel):
    w, path = model
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    monkeypatch.setattr(p1, "EXPRESS", True)
    monkeypatch.setattr(p1, "EXPRESS_ROWS", 32)
    owner = object.__new__(GlmEngine)
    owner.torch, owner.rank, owner.world, owner.w = torch, 0, 3, w
    owner.comm, owner.limit = w.comm, 4096
    owner.drafter = Drafter(path, w, capacity=4608, streams=4)
    owner.e = decode.Engine(w, capacity=4608, prefill_rows=129, long_context=True,
                            taps=owner.drafter.tap_layers)
    owner.policy, owner.cache_bytes, owner.cache_entries = "fc7:0.3", 32 * 2**20, 8
    owner._share = lambda values: values
    prompts = [list(np.random.default_rng(seed).integers(0, 1000, n)) for seed, n in ((31, 257), (32, 19))]
    expected, expected_state = [], []
    for prompt in prompts:
        # Use fresh cache storage for both arms. Serial continuations write past
        # the prompt into pool fence rows that reset() intentionally retains.
        owner.e = decode.Engine(w, capacity=4608, prefill_rows=129, long_context=True,
                                taps=owner.drafter.tap_layers)
        first = decode.prefill(owner.e, prompt, sampling, mtp=False)
        expected_state.append(state(owner.e))
        expected.append(decode.serial_decode(owner.e, first, 12, sampling).tokens)
    owner.e = decode.Engine(w, capacity=4608, prefill_rows=129, long_context=True,
                            taps=owner.drafter.tap_layers)
    d = BatchedDecoder(owner, 4)
    long, short = [_stream(p, 12, sampling=sampling) for p in prompts]
    if cancel == "short":
        short.prefill_slice_layers = 1  # Exercise cancellation while the express chunk is suspended.
    d.begin_admit(long); d.prefill_step()
    paused = long.engine.pbuf.x.clone()
    d.begin_admit(short); d.prefill_step()
    same_tensor(long.engine.pbuf.x, paused)
    if cancel:
        victim = long if cancel == "long" else short
        victim.cancelled = lambda: True
        d.finish(d.prefill_step())
    for _ in range(100):
        if not d.filling: break
        d.finish(d.prefill_step())
    assert not d.filling
    for i, stream in enumerate((long, short)):
        if cancel == ("long", "short")[i]: continue
        same(state(stream.engine), expected_state[i])
    while d.live():
        d.finish(d.round())
    for i, stream in enumerate((long, short)):
        assert stream.out == ([] if cancel == ("long", "short")[i] else expected[i])
    assert d.fill_owner is d.express_owner is None
