"""Co-prefill large-pass byte gates with real EXL3 kernels and replicated TP2 partials.

This bounded single-GPU fixture does not replace the full-weight TP3 boot gate.
Prefill is compared with prefill: every hidden/tap row, then kept state and reply.
"""

import gc
import hashlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import cofill_big, decode, forward, prefill_options as p1
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from tensorfold.families.glm5_next.cuda.weights import load
from test_glm_batched import _stream, _prompt, _drain
from test_glm_engine import _checkpoint, _drafter, _TwoCopies
from test_glm_p1_state import state


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("m8_big_exl3")
    _checkpoint(path / "target", exl3=True)
    _drafter(path / "draft")
    w = load(path / "target", rank=0, world=2)
    w.comm = _TwoCopies()
    return w, path / "draft"


@pytest.fixture
def make(model, monkeypatch):
    w, path = model
    for flag in ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "DEAD_WORK", "EXPRESS"):
        monkeypatch.setattr(p1, flag, False)
    monkeypatch.setattr(p1, "ATTENTION_TILES", True)
    monkeypatch.setattr(p1, "COFILL_BIG", True)
    def build(rows, slots):
        owner = object.__new__(GlmEngine)
        owner.torch, owner.rank, owner.world, owner.w = torch, 0, 2, w
        owner.comm, owner.limit, owner.pool_limit = w.comm, 12288, 65536
        capacity = owner.pool_limit + 8 + (slots - 1) * 64
        owner.drafter = Drafter(path, w, capacity=capacity, streams=slots)
        owner.e = decode.Engine(w, capacity=capacity, prefill_rows=rows, long_context=True,
                                taps=owner.drafter.tap_layers)
        owner.policy, owner.cache_bytes, owner.cache_entries = "f7", 128 * 2**20, 8
        owner._share = lambda values: values
        return BatchedDecoder(owner, slots)
    yield build
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("slots,rows", [(2, 4096), (4, 8192), (8, 8192)])
@pytest.mark.parametrize("cached", [0, 2047, 2051])
@pytest.mark.parametrize("sampled", [False, True])
def test_each_prefill_row_and_state_hash_equal_alone(make, monkeypatch, slots, rows, cached, sampled):
    prompts = [_prompt(670 + i, 4097 + 29 * i) for i in range(slots)]
    samplers = [Sampling(20261001 + i, .7, 20, .95) if sampled else None for i in range(slots)]
    original, collected = forward.compute_streams, {}
    def observe(w, segs, b, **kwargs):
        logits = original(w, segs, b, **kwargs)
        if b.prefill:
            for st, a0, a1 in segs:
                # Every row is compared independently, irrespective of chunking.
                raw = torch.cat([b.fnormed[a0:a1], *[tap[a0:a1] for tap in b.taps]], dim=1)
                raw = raw.contiguous().view(torch.uint8).cpu().numpy()
                collected.setdefault(id(st), []).extend(
                    hashlib.sha256(row.tobytes()).hexdigest() for row in raw)
        return logits
    monkeypatch.setattr(forward, "compute_streams", observe)
    monkeypatch.setattr(cofill_big, "compute_streams", observe)
    expected = []
    for prompt, sampler in zip(prompts, samplers):
        d = make(2048, slots)
        if cached:
            warm = _stream(prompt[:cached], 1)
            d.admit(warm); d.finish([warm])
            del warm
        collected.clear()
        s = _stream(prompt, 12, sampling=sampler)
        s.cofill = False
        d.admit(s)
        expected.append((list(collected[id(s.st)]), digest(state(s.engine, s.drafter))))
        _drain(d)
        expected[-1] += (s.out[:],)
        d.drop()
        del d, s
        gc.collect()
        torch.cuda.empty_cache()
    d = make(rows, slots)
    if cached:
        for prompt in prompts:
            warm = _stream(prompt[:cached], 1)
            d.admit(warm); d.finish([warm])
    collected.clear()
    streams = [_stream(p, 12, sampling=s) for p, s in zip(prompts, samplers)]
    for s in streams:
        d.begin_admit(s)
        assert isinstance(d.filling[s.sid], cofill_big.Prefill)
    for _ in range(100):
        if not d.filling:
            break
        d.finish(d.prefill_step())
    assert not d.filling
    for i, s in enumerate(streams):
        actual = collected[id(s.st)]
        assert len(actual) == len(s.prompt) - cached
        assert actual == expected[i][0], ("hidden/tap row", i)
        assert digest(state(s.engine, s.drafter)) == expected[i][1], ("kept state", i)
        assert s.cofill_stats["max_total_rows"] == rows
        assert s.cofill_stats["cofilled_passes"] > 0
    _drain(d)
    assert [s.out for s in streams] == [e[2] for e in expected]
    # Negative control: the same hash gate must reject one changed row.
    row = streams[-1].st.kc[0][:1].clone()
    before = digest([row])
    row.view(torch.uint8).flatten()[0] ^= 1
    with pytest.raises(AssertionError):
        assert digest([row]) == before
    d.drop()
