"""KDA value layouts must preserve output, FP32 state, and replay bytes."""

import hashlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import kda, kda_tiles, prefill_options


def inputs(rows, heads=22, seed=521):
    g = torch.Generator(device="cuda").manual_seed(seed)
    c = 3 * heads * 128
    width = ((c + 256 + heads + 63) // 64) * 64
    rand = lambda shape: torch.randn(shape, generator=g, device="cuda")
    return dict(p=(rand((rows, width)) * .5).bfloat16(), a=rand((rows, heads * 128)).bfloat16(),
                gate=rand((rows, heads * 128)).bfloat16(), cs=(rand((3, c)) * .5).bfloat16(),
                cw=(rand((c, 4)) * .3).bfloat16(), state=rand((heads, 128, 128)) * .05,
                a_log=rand((heads,)), dt=rand((heads * 128,)), norm=(1 + rand((128,)) * .1).bfloat16(),
                b_off=c + 256)


def buffers(x, rows):
    h = x["state"].shape[0]
    sc = kda.KDAScratch(rows, h, "cuda")
    next_state = torch.empty_like(x["state"])
    q = torch.empty_like(sc.k)
    y = torch.empty_like(sc.v)
    return sc, next_state, q, y


def arguments(x, rows, buf):
    sc, next_state, q, y = buf
    return (x["p"], x["p"].stride(0), x["b_off"], x["a"], x["a"].stride(0), x["gate"], x["gate"].stride(0),
            x["cs"], x["cw"], x["state"], x["a_log"], x["dt"], x["norm"], 1e-5, -5., rows,
            sc.out, next_state, sc.k, sc.v, sc.g, sc.b, q, y)


def byte_equal(got, want):
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))


def check_buffers(got, want):
    for attr in ("out", "k", "v", "g", "b"):
        byte_equal(getattr(got[0], attr), getattr(want[0], attr))
    for i in range(1, 4):
        byte_equal(got[i], want[i])


@pytest.mark.parametrize("rows", list(range(1, 33)) + [63, 64, 65, 129, 1000, 2048, 8192])
def test_all_layouts_have_stock_bytes(rows):
    x = inputs(rows)
    want, got = buffers(x, rows), buffers(x, rows)
    kda._ext().chain_wide(*arguments(x, rows, want))
    for variant in range(7):
        kda_tiles._ext().chain_wide(*arguments(x, rows, got), variant)
        check_buffers(got, want)
    digest = hashlib.sha256(got[1].cpu().numpy().tobytes()).hexdigest()
    print("rows", rows, "state_sha256", digest)
    # A one-bit state error cannot pass a norm/tolerance shortcut.
    got[1].view(torch.int32).flatten()[0] ^= 1
    with pytest.raises(AssertionError):
        byte_equal(got[1], want[1])


@pytest.mark.parametrize("cut", [1, 7, 16, 31, 63, 64, 65, 127])
def test_resume_and_replay_keep_the_same_recurrence(cut):
    rows = 193
    x = inputs(rows, heads=2, seed=823)
    full = buffers(x, rows)
    kda_tiles._ext().chain_wide(*arguments(x, rows, full), kda_tiles.VARIANT)
    before = {**x, "p": x["p"][:cut], "a": x["a"][:cut], "gate": x["gate"][:cut]}
    first = buffers(before, cut)
    kda_tiles._ext().chain_wide(*arguments(before, cut, first), kda_tiles.VARIANT)
    replayed = torch.empty_like(x["state"])
    kda.replay(x["state"], full[0], cut, replayed)
    byte_equal(first[1], replayed)
    c = x["cs"].shape[1]
    after = {**x, "p": x["p"][cut:], "a": x["a"][cut:], "gate": x["gate"][cut:],
             "state": first[1], "cs": torch.cat([x["cs"], x["p"][:cut, :c]])[-3:].contiguous()}
    second = buffers(after, rows - cut)
    kda_tiles._ext().chain_wide(*arguments(after, rows - cut, second), kda_tiles.VARIANT)
    byte_equal(second[1], full[1])
    byte_equal(torch.cat([first[0].out, second[0].out]), full[0].out)


def test_gate_handles_padded_projection_and_gate_row_strides(monkeypatch):
    x = inputs(258, heads=3)
    x = {k: v[::2] if k in ("p", "a", "gate") else v for k, v in x.items()}
    want, got = buffers(x, 129), buffers(x, 129)
    kda._ext().chain_wide(*arguments(x, 129, want))
    monkeypatch.setattr(prefill_options, "KDA_VALUE_TILES", True)
    sc, nxt, _, _ = got
    result = kda.chain(x["p"], x["b_off"], x["a"], x["gate"], x["cs"], x["cw"], x["state"], x["a_log"],
                       x["dt"], x["norm"], 1e-5, -5., 129, sc, nxt)
    byte_equal(result, want[0].out)
    byte_equal(nxt, want[1])


@pytest.mark.parametrize("sampled", [False, True])
def test_complete_prefill_resume_and_continuation_match(model, monkeypatch, sampled):
    import numpy as np
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda import decode
    from test_glm_p1_state import engine, same, state

    w, _ = model
    prompt = list(np.random.default_rng(171).integers(0, 1000, 1217))
    sampling = Sampling(721, .7, 20, .95) if sampled else None
    expected = None
    for enabled in (False, True):
        monkeypatch.setattr(prefill_options, "KDA_VALUE_TILES", enabled)
        e = engine(w, 513, span=64, images=True)
        snapshots = []
        def keep(snap):
            decode.save_rows(e, snap)
            snapshots.append(snap)
        first = decode.prefill(e, prompt, sampling, mtp=True,
                               mark=lambda n: 607 if n < 607 else None, keep=keep)
        current = state(e), e.pbuf.logits[:1].clone()
        if expected is None:
            expected = current
        else:
            same(current[0], expected[0]); byte_equal(current[1], expected[1])
        decode.load_rows(e, snapshots[0])
        assert decode.prefill(e, prompt, sampling, mtp=True, resume=snapshots[0]) == first
        same(state(e), current[0]); byte_equal(e.pbuf.logits[:1], current[1])
        tokens = decode.serial_decode(e, first, 16, sampling).tokens
        if not enabled:
            expected_tokens = tokens
        else:
            assert tokens == expected_tokens


# Reuse the tiny TP3 random-weight fixture; it replicates rank partials on one GPU.
from test_glm_p1_state import model  # noqa: E402,F401
