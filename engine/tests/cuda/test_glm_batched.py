"""Tiny-checkpoint gates for the pooled GLM target, DFlash2 and graph paths.

The two- and three-rank cases use identical stand-in rank partials on one GPU;
they test supported serving shapes and ordered gathers, not multi-host transport.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder  # noqa: E402
from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter  # noqa: E402
from tensorfold.families.glm5_next.cuda.engine import GlmEngine  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import commit, compute_streams, stage_streams  # noqa: E402
from tensorfold.families.glm5_next.cuda.weights import load  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter  # noqa: E402
from test_glm_tp3 import _Copies, _checkpoint as _checkpoint_tp3  # noqa: E402


@pytest.fixture(scope="module", params=["q4", "q4_tp3", "exl3"])
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_batched_" + request.param)
    world = {"q4": 1, "q4_tp3": 3, "exl3": 2}[request.param]
    if world == 3:
        _checkpoint_tp3(path / "target")
    else:
        _checkpoint(path / "target", exl3=request.param == "exl3")
    _drafter(path / "draft")
    return path, world


@pytest.fixture
def owner(checkpoint, request):
    path, world = checkpoint
    owner = object.__new__(GlmEngine)
    owner.torch, owner.rank, owner.world = torch, 0, world
    owner.w = load(path / "target", rank=0, world=world)
    owner.comm = _Copies(0) if world == 3 else _TwoCopies() if world == 2 else None
    owner.w.comm = owner.comm
    owner.limit = 4096
    owner.pool_limit = owner.limit * (8 if getattr(request, "param", 8) > 8 else 1)
    capacity = owner.pool_limit + 8 + 7 * 64
    owner.drafter = Drafter(path / "draft", owner.w, capacity=capacity, streams=8)
    owner.e = Engine(owner.w, capacity=capacity, max_rows=getattr(request, "param", 8), prefill_rows=64, long_context=True,
                     taps=owner.drafter.tap_layers)
    owner.policy = "fc7:0.3"
    owner.cache_bytes, owner.cache_entries = 32 * 2**20, 8
    owner._share = lambda values: values
    yield owner
    del owner
    torch.cuda.empty_cache()


def _prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, n)]


def _stream(prompt, count=20, *, policy="fc7:0.3", sampling=None, draft=True):
    s = Stream(list(prompt), count, sampling, draft=draft, stop_eos=False)
    s.policy = policy
    return s


def _serial(owner, s):
    first = prefill(owner.e, s.prompt, s.sampling, mtp=False)
    return serial_decode(owner.e, first, s.count, s.sampling).tokens


def _drain(decoder):
    decoder.finish([s for s in decoder.streams.values() if s.done])
    for _ in range(200):
        if not decoder.live():
            return
        decoder.finish(decoder.round())
    pytest.fail("pooled decoder did not finish")


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_mixed_widths_and_policies_are_exact_per_request(owner, slots):
    policies = ["f7", "fc7:0.3", "f2", "0", "f7", "fc7:0.0", "2", "a:0.6:0.85"]
    streams = [_stream(_prompt(70 + i, 17 + i * 3), 15 + i, policy=policies[i], draft=i != 3,
                       sampling=Sampling(2**63 + i, .712345678901, 20 + i, .934567890123) if i % 2 else None)
               for i in range(slots)]
    expected = [_serial(owner, s) for s in streams]
    decoder = BatchedDecoder(owner, slots)
    for s in streams:
        decoder.admit(s)
    assert streams[0].drafts and len(streams[0].drafts) == 7
    _drain(decoder)
    assert [s.out for s in streams] == expected
    for s in streams:
        assert s.drafted_tokens == sum(s.depths)
        assert s.accepted_draft_tokens == sum(k - 1 for k in s.keeps)
        assert s.st.pos == len(s.prompt) + s.count - 1
    assert decoder.pool.available == decoder.capacity


def test_c8_full_64_rows_match_individual_logits_taps_and_committed_state(owner):
    decoder = BatchedDecoder(owner, 8)
    streams = [_stream(_prompt(130 + i, 13 + i * 4), 24, policy="f7") for i in range(8)]
    for s in streams:
        decoder.admit(s)
    windows = [(s.st, [s.out[-1], *s.drafts]) for s in streams]
    assert sum(len(tokens) for _, tokens in windows) == 64
    references = [s.st.clone() for s in streams]
    segs = stage_streams(owner.w, decoder.buf, windows)
    together = compute_streams(owner.w, segs, decoder.buf, eager=True).clone()
    taps = [t[:64].clone() for t in decoder.buf.taps]
    keeps = [1 + i for i in range(8)]
    for s, keep in zip(streams, keeps):
        commit(owner.w, s.st, decoder.buf, 8, keep)
    for i, (s, ref, (_, tokens), keep) in enumerate(zip(streams, references, windows, keeps)):
        one = stage_streams(owner.w, decoder.buf, [(ref, tokens)])
        alone = compute_streams(owner.w, one, decoder.buf, eager=True)
        assert torch.equal(alone, together[i * 8:(i + 1) * 8]), ("logits", i)
        for tap, batched in zip(decoder.buf.taps, taps):
            assert torch.equal(tap[:8], batched[i * 8:(i + 1) * 8]), ("taps", i)
        commit(owner.w, ref, decoder.buf, 8, keep)
        assert torch.equal(s.st.rec[s.st.cur[0]], ref.rec[ref.cur[0]]), ("KDA", i)
        assert torch.equal(s.st.conv, ref.conv), ("conv", i)
        assert s.st.pos == ref.pos
        for actual, expected in zip(s.st.kc, ref.kc):
            assert torch.equal(actual[:ref.pos], expected[:ref.pos]), ("target rows", i)
        for actual, expected in zip(s.st.index, ref.index):
            for j, (a, b) in enumerate(zip(actual, expected)):
                n = ref.pos // 4 if j == 2 else ref.pos
                assert torch.equal(a[:n], b[:n]), ("index rows", i, j)
    decoder.drop()


@pytest.mark.parametrize("owner", [16, 32], indirect=True)
def test_wide_c8_logits_taps_and_kept_state_equal_serial_rows(owner):
    """Real kernels through 256 aggregate rows, both sides of the sparse boundary."""
    rows = owner.e.rows
    decoder = BatchedDecoder(owner, 8)
    boundary = owner.w.cfg.dense_limit
    lengths = [13, 17, boundary - 17, boundary - 8, boundary - 1, boundary, boundary + 1, boundary + 15]
    streams = [_stream(_prompt(430 + i, n), rows + 8, policy="f7") for i, n in enumerate(lengths)]
    for s in streams:
        decoder.admit(s)
    windows = [(s.st, [s.out[-1], *_prompt(510 + i, rows - 1)]) for i, s in enumerate(streams)]
    refs = [s.st.clone() for s in streams]
    segs = stage_streams(owner.w, decoder.buf, windows)
    together = compute_streams(owner.w, segs, decoder.buf, eager=True).clone()
    taps = [t[:rows * 8].clone() for t in decoder.buf.taps]
    keeps = [min(rows, n) for n in (1, 2, 7, 8, 9, 15, 31, 32)]
    for s, keep in zip(streams, keeps):
        commit(owner.w, s.st, decoder.buf, rows, keep)
    for i, (s, ref, (_, tokens), keep) in enumerate(zip(streams, refs, windows, keeps)):
        for j, token in enumerate(tokens):
            one = stage_streams(owner.w, decoder.buf, [(ref, [token])])
            logits = compute_streams(owner.w, one, decoder.buf, eager=True)
            assert torch.equal(logits[0], together[i * rows + j]), ("logits", rows, i, j)
            for current, saved in zip(decoder.buf.taps, taps):
                assert torch.equal(current[0], saved[i * rows + j]), ("taps", rows, i, j)
            commit(owner.w, ref, decoder.buf, 1, 1)
            if j + 1 == keep:
                assert torch.equal(s.st.conv, ref.conv), ("conv", rows, i, keep)
                for layer, cur in enumerate(ref.cur):
                    assert torch.equal(s.st.rec[s.st.cur[layer], layer], ref.rec[cur, layer]), ("KDA", rows, i, layer)
                assert s.st.pos == ref.pos
                for a, b in zip(s.st.kc, ref.kc):
                    assert torch.equal(a[:ref.pos], b[:ref.pos]), ("target rows", rows, i)
                for actual, expected in zip(s.st.index, ref.index):
                    for k, (a, b) in enumerate(zip(actual, expected)):
                        n = ref.pos // 4 if k == 2 else ref.pos
                        assert torch.equal(a[:n], b[:n]), ("index rows", rows, i, k)
    decoder.drop()


def test_c1_original_graphs_equal_eager_and_fresh_serial(owner):
    from tensorfold.families.glm5_next.cuda.graphs import Graphs

    # Capture only after taps are installed, exactly as the serving initializer does.
    owner.e.graphs = Graphs(owner.e, tuple(range(1, 9)), tuple(range(1, 9)))
    owner.e.reset()
    owner.drafter.capture()
    source = _stream(_prompt(170, 37), 30, policy="f7", sampling=Sampling(9, .7234567891, 20, .9456789123))
    serial = _serial(owner, source)
    before = sum(owner.e.replays[k] for k in ("main", "sparse"))
    decoder = BatchedDecoder(owner, 2)
    graphed = _stream(source.prompt, source.count, policy=source.policy, sampling=source.sampling)
    decoder.admit(graphed)
    _drain(decoder)
    assert sum(owner.e.replays[k] for k in ("main", "sparse")) > before
    owner.e.graphs = None
    owner.drafter.block_graph, owner.drafter.tap_graphs = None, {}
    decoder = BatchedDecoder(owner, 2)
    eager = _stream(source.prompt, source.count, policy=source.policy, sampling=source.sampling)
    decoder.admit(eager)
    _drain(decoder)
    assert graphed.out == eager.out == serial


def test_saved_conversations_restore_target_and_full_drafter_prefix(owner):
    prompts = [_prompt(210, 64), _prompt(211, 64)]
    followups = [_stream(p + _prompt(212 + i, 17), 16, policy="f7") for i, p in enumerate(prompts)]
    expected = [_serial(owner, s) for s in followups]
    decoder = BatchedDecoder(owner, 2)
    first = [_stream(p, 8, policy="f7") for p in prompts]
    for s in first:
        decoder.admit(s)
    _drain(decoder)
    assert len(decoder.cache) == 2
    # Old cache memory is deliberately hostile: every resumed prefix row must be restored.
    for c in owner.drafter.kc + owner.drafter.vc:
        c.fill_(float("nan"))
    for c in owner.e.st.kc:
        c.fill_(float("nan"))
    for s in followups:
        decoder.admit(s)
    assert [s.cached for s in followups] == [64, 64]
    _drain(decoder)
    assert [s.out for s in followups] == expected
    assert decoder.held_bytes() <= owner.cache_bytes


def test_drafter_view_keeps_parent_head_stride_and_matches_contiguous_cache(owner):
    decoder = BatchedDecoder(owner, 2)
    a, b = _stream(_prompt(240, 17), 18), _stream(_prompt(241, 21), 18, policy="f7")
    decoder.admit(a)
    decoder.admit(b)
    view = b.drafter
    assert view.kc[0].stride(0) == owner.drafter.kc[0].stride(0)
    assert view.cap == owner.drafter.cap and view.pos_dev is not owner.drafter.pos_dev
    contiguous = copy.copy(view)
    contiguous.kc = [c.clone() for c in view.kc]
    contiguous.vc = [c.clone() for c in view.vc]
    contiguous.cap = contiguous.kc[0].shape[1]
    contiguous.pos_dev = view.pos_dev.clone()
    contiguous.block_graph, contiguous.tap_graphs = None, {}
    sliced = view.candidates(b.out[-1], 7)
    compact = contiguous.candidates(b.out[-1], 7)
    for actual, expected in zip(sliced, compact):
        assert np.array_equal(actual, expected)
    decoder.drop()


def test_nonzero_pool_offset_crosses_dense_to_sparse_exactly(owner):
    short = _stream(_prompt(260, 29), 12, policy="f7")
    long = _stream(_prompt(261, owner.w.cfg.dense_limit - 3), 12, policy="f7",
                   sampling=Sampling(13, .7654321987, 20, .9123456789))
    expected = [_serial(owner, s) for s in (short, long)]
    decoder = BatchedDecoder(owner, 2)
    decoder.admit(short)
    decoder.admit(long)
    assert long.start > 0
    _drain(decoder)
    assert [short.out, long.out] == expected
    assert long.st.pos > owner.w.cfg.dense_limit


def test_cancel_and_reused_span_leave_peer_equal_to_serial(owner):
    peer = _stream(_prompt(270, 43), 20, policy="f7", sampling=Sampling(3, .7, 20, .95))
    replacement = _stream(_prompt(271, 57), 12, policy="f7")
    expected = [_serial(owner, s) for s in (peer, replacement)]
    decoder = BatchedDecoder(owner, 2)
    cancelled = _stream(_prompt(272, 64), 25, policy="f7")
    decoder.admit(cancelled)
    decoder.admit(peer)
    cancelled.cancelled = lambda: True
    done = decoder.round()
    assert cancelled in done and cancelled.rounds == 0
    decoder.finish(done)
    decoder.admit(replacement)
    assert replacement.start == cancelled.start
    _drain(decoder)
    assert [peer.out, replacement.out] == expected
