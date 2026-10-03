"""Prefill semantic state/output gates on the TP3 synthetic model; a single GPU replicates rank partials.

These complement the 22-head kernel gate; real TP3 hashes remain a boot gate.
"""

import json

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode, forward, prefill_options as p1, sparse
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
from tensorfold.families.glm5_next.cuda.weights import load
from test_glm_tp3 import _checkpoint, _Copies, _drafter

FLAGS = ("PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "DEAD_WORK")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for flag in FLAGS: monkeypatch.setattr(p1, flag, False)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("p1_state")
    _checkpoint(path / "model")
    _drafter(path / "draft")
    config = json.loads((path / "draft/config.json").read_text())
    config["sliding_window"] = 66       # wrap several times in a short, bounded fixture
    (path / "draft/config.json").write_text(json.dumps(config))
    w = load(path / "model", rank=0, world=3)
    w.comm = _Copies(0)
    return w, path / "draft"


def engine(w, rows=513, *, span=0, taps=(), images=False):
    e = decode.Engine(w, capacity=4352, max_rows=8, prefill_rows=rows, long_context=True, taps=taps)
    if span:
        # The same view offsets as pooled admission; index pool starts at span/4.
        e.st.kc = [torch.zeros((4352 + span, *t.shape[1:]), dtype=t.dtype, device=t.device)[span:] for t in e.st.kc]
        e.st.index = [tuple(torch.zeros((t.shape[0] + offset, *t.shape[1:]), dtype=t.dtype,
                                       device=t.device)[offset:] for t, offset in zip(group, (span, span, span // 4)))
                      for group in e.st.index]
    if images:
        positions = np.array([11, 97, 1988, 2050], dtype=np.int64)
        e.images = (positions, torch.arange(4 * w.cfg.hidden, device="cuda").view(4, -1).to(torch.bfloat16) * 0.001)
    return e


def state(e, dr=None):
    st = e.st
    values = [st.rec[i, li] for li, i in enumerate(st.cur)] + [st.conv]
    values += decode._row_views(st, st.pos, st.mtp_len)
    if dr is not None:
        assert dr.context_end == st.pos == int(dr.pos_dev.item())
        values += decode._ring_window(dr, dr.context_end)
    return [v.clone() for v in values]


def same(a, b):
    assert len(a) == len(b)
    assert all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x, y in zip(a, b))


@pytest.mark.parametrize("dense_left", [1, 2, 63, 64, 65])
@pytest.mark.parametrize("span,images", [(0, False), (64, True)])
def test_prefill_ports_crossing_resume_state_and_output(model, monkeypatch, dense_left, span, images):
    w, _ = model
    prompt = list(np.random.default_rng(71).integers(0, 1000, 2181))
    cut = 2051 - dense_left
    sampling = Sampling(89, 1.0, 20, 0.95)
    reference = None
    for flag in (None, "PROMPT_SCRATCH", "SELECT_BLOCKS", "VISIBLE_POOLS", "DENSE_ROWS", "all"):
        for f in FLAGS: monkeypatch.setattr(p1, f, flag == f or flag == "all" and f != "DEAD_WORK")
        e = engine(w, span=span, images=images)
        kept = []
        def keep(snap):
            decode.save_rows(e, snap)
            kept.append(snap)
        mark = lambda n: cut if n < cut else None
        first = decode.prefill(e, prompt, sampling, mtp=True, mark=mark, keep=keep)
        cold, logits = state(e), e.pbuf.logits[:1].clone()
        if reference is None:
            reference = first, cold, logits
        else:
            assert first == reference[0]
            same(cold, reference[1]); same([logits], [reference[2]])
        e.reset()
        decode.load_rows(e, kept[0])
        assert decode.prefill(e, prompt, sampling, mtp=True, resume=kept[0]) == first
        same(state(e), cold); same([e.pbuf.logits[:1]], [logits])


@pytest.mark.parametrize("mtp,images", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("rows", [65, 129])
def test_dead_work_ring_wrap_checkpoints_poison_and_continuation(model, monkeypatch, mtp, images, rows):
    w, path = model
    prompt = list(np.random.default_rng(72).integers(0, 1000, 521))
    sampling = Sampling(90, 1.0, 20, 0.95)
    reference = None
    for enabled in (False, True):
        monkeypatch.setattr(p1, "DEAD_WORK", enabled)
        dr = Drafter(path, w, capacity=4352)
        for cache in dr.kc + dr.vc: cache.fill_(float("nan"))
        e = engine(w, rows, span=64, taps=dr.tap_layers, images=images)
        kept, states = [], []
        def keep(snap):
            decode.save_rows(e, snap, dr)
            kept.append(snap); states.append(state(e, dr))
        mark = lambda n: next((cut for cut in (131, 337) if n < cut), None)
        first = decode.prefill(e, prompt, sampling, mtp=mtp, drafter=dr, mark=mark, keep=keep)
        cold, logits = state(e, dr), e.pbuf.logits[:1].clone()
        # Restore the window after poisoning every ring cell, including the masked leading tile.
        for cache in dr.kc + dr.vc: cache.fill_(float("nan"))
        decode.load_rows(e, kept[-1], dr)
        resumed = decode.prefill(e, prompt, sampling, mtp=mtp, drafter=dr, resume=kept[-1])
        assert resumed == first
        same(state(e, dr), cold); same([e.pbuf.logits[:1]], [logits])
        reply = decode.dflash_decode(e, dr, first, 12, sampling).tokens
        if reference is None:
            reference = first, cold, logits, states, reply
        else:
            assert first == reference[0] and reply == reference[4]
            same(cold, reference[1]); same([logits], [reference[2]])
            for a, b in zip(states, reference[3]): same(a, b)


def test_omitted_dense_prefix_index_write_is_detected(model, monkeypatch):
    w, _ = model
    prompt = list(np.random.default_rng(73).integers(0, 1000, 2181))
    monkeypatch.setattr(p1, "DENSE_ROWS", True)
    e = engine(w)
    decode.prefill(e, prompt, None, mtp=False)
    want = state(e)
    original = sparse.index_update
    def omit_dense(*args):
        if int(args[-1].item()) >= w.cfg.dense_limit:
            original(*args)
    monkeypatch.setattr(sparse, "index_update", omit_dense)
    bad = engine(w)
    decode.prefill(bad, prompt, None, mtp=False)
    with pytest.raises(AssertionError):
        same(state(bad), want)


def test_poisoning_masked_leading_ring_tile_breaks_real_drafter(model, monkeypatch):
    w, path = model
    monkeypatch.setattr(p1, "DEAD_WORK", True)
    dr = Drafter(path, w, capacity=4352)
    e = engine(w, 129, taps=dr.tap_layers)
    first = decode.prefill(e, list(np.random.default_rng(74).integers(0, 1000, 521)), None, mtp=False, drafter=dr)
    _, values, projected = dr.candidates(first, 7)
    assert np.isfinite(values).all() and np.isfinite(projected).all()
    lo = max(0, dr.context_end - dr.window) // 64 * 64
    assert lo < dr.context_end - dr.window
    dr.vc[0][:, lo % dr.ring] = float("nan")
    _, values, projected = dr.candidates(first, 7)
    assert not np.isfinite(values).all() or not np.isfinite(projected).all()


def test_p1_and_layer_slices_keep_state_and_drafter(model, monkeypatch):
    w, path = model
    for flag in FLAGS: monkeypatch.setattr(p1, flag, True)
    prompt = list(np.random.default_rng(75).integers(0, 1000, 2181))
    dr = Drafter(path, w, capacity=4352)
    plain = engine(w, 513, span=64, taps=dr.tap_layers, images=True)
    first = decode.prefill(plain, prompt, None, mtp=False, drafter=dr)
    want, logits = state(plain, dr), plain.pbuf.logits[:1].clone()
    dr = Drafter(path, w, capacity=4352)
    sliced = engine(w, 513, span=64, taps=dr.tap_layers, images=True)
    stop = 1
    steps = decode.prefill_steps(sliced, prompt, None, mtp=False, drafter=dr, slice_end=lambda: stop)
    while True:
        try:
            progress = next(steps)
        except StopIteration as done:
            assert done.value == first
            break
        stop = len(w.layers) if isinstance(progress, tuple) else 1
    same(state(sliced, dr), want)
    same([sliced.pbuf.logits[:1]], [logits])
