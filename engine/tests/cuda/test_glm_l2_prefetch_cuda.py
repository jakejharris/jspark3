"""L2 prefetch read-only hints: all 1–32 row kernels, graph replay, pooled state and replies.

The tiny model's TP3 partials are stand-ins on one GPU; real NCCL serving checks
remain mandatory. This file can run only when the shared MEASURING lock is free.
"""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.glm5_next.cuda import decode, forward, l2_prefetch as l2, qmm
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from tensorfold.families.glm5_next.cuda.graphs import Graphs
from tensorfold.engine.exact_sampling import Sampling
from test_glm_batched import checkpoint, owner, _prompt, _stream, _serial
from test_glm_sliced_cuda import drain


def bits(x):
    return x.contiguous().view(torch.uint8)


@pytest.fixture(scope="module")
def matrix():
    gen = torch.Generator(device="cuda").manual_seed(541)
    q = qmm.quantize4(torch.randn((512, 512), generator=gen, device="cuda", dtype=torch.bfloat16) * .03)
    return q


@pytest.mark.parametrize("rows", list(range(1, 33)) + [63, 64])
def test_projection_bits_and_weight_bytes_survive_hints(matrix, rows):
    torch.manual_seed(rows)
    x = torch.randn((rows, matrix.k), device="cuda", dtype=torch.bfloat16)
    reference = qmm.matmul(x, matrix).clone()
    weights = [t.clone() for t in (matrix.weight, matrix.scales, matrix.biases)]
    hint = l2.Prefetch(NS(device=x.device, layers=[]))
    hint.pending = matrix.weight
    parent = hint.start()
    torch.cuda._sleep(10000)  # stand-in gather delay; no model output depends on it
    hint.finish(parent)
    actual = qmm.matmul(x, matrix)
    assert torch.equal(bits(actual), bits(reference))
    assert all(torch.equal(bits(a), bits(b)) for a, b in zip(weights, (matrix.weight, matrix.scales, matrix.biases)))
    damaged = bits(actual).clone()
    damaged.flatten()[0] ^= 1
    assert not torch.equal(damaged, bits(reference)), "output checker failed its corruption control"


@pytest.mark.parametrize("size", [1, 31, 32, 33, 4095, 4096, 4097])
def test_hint_tail_addresses_and_no_write(size):
    weight = torch.arange(size, device="cuda", dtype=torch.int32).to(torch.uint8)
    saved = weight.clone()
    program = l2.tensor(weight)
    torch.cuda.synchronize()
    assert "prefetch.global.L2" in program.asm["ptx"]
    assert torch.equal(weight, saved)
    with pytest.raises(ValueError, match="contiguous CUDA weight"):
        l2.tensor(torch.ones((5, 7), device="cuda").T)


def test_multistream_graphs_replay_new_inputs_without_writing_weights(matrix):
    x = torch.zeros((8, matrix.k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((8, matrix.n), device="cuda", dtype=torch.bfloat16)
    part = torch.empty((8 * 8 * 16384,), device="cuda", dtype=torch.float32)
    hint = l2.Prefetch(NS(device=x.device, layers=[]))
    def run():
        hint.pending = matrix.weight
        parent = hint.start()
        torch.cuda._sleep(10000)
        hint.finish(parent)
        qmm.matmul(x, matrix, out=out, part=part)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph in graphs:
        with torch.cuda.graph(graph):
            run()
    for seed, graph in enumerate(graphs * 3):
        torch.manual_seed(seed)
        x.normal_()
        reference = qmm.matmul(x, matrix).clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), bits(reference))


@pytest.mark.parametrize("policy", ["f7", "3"])
def test_nonzero_slot_slicing_and_priorities_keep_serial_replies(owner, monkeypatch, policy):
    sampled = Sampling(2**63 + 7, .713456789012, 20, .934567890123)
    sources = [_stream(_prompt(510 + i, 41 + 67 * i), 24, policy=policy, sampling=sampled) for i in range(2)]
    reference = [_serial(owner, stream) for stream in sources]
    monkeypatch.setattr(l2, "ENABLED", True)
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    decoder = BatchedDecoder(owner, 2)
    owner.e.buf.l2_prefetch = l2.make(owner.w, prefill=False)
    for i, stream in enumerate(sources):
        stream.priority = "background" if i else "interactive"
        stream.prefill_slice_layers = 1
        decoder.begin_admit(stream)
    assert sources[1].start > 0
    drain(decoder)
    assert [stream.out for stream in sources] == reference
    assert decoder.pool.available == decoder.capacity


def test_original_graph_path_matches_plain_serial(owner):
    stream = _stream(_prompt(520, 65), 24, policy="f7", sampling=Sampling(19, .71, 20, .95))
    reference = _serial(owner, stream)
    owner.e.buf.l2_prefetch = l2.Prefetch(owner.w) if owner.w.world > 1 else None
    owner.e.graphs = Graphs(owner.e, tuple(range(1, 9)), (1, 3))
    owner.e.reset()
    decoder = BatchedDecoder(owner, 2)
    decoder.admit(stream)
    drain(decoder)
    assert stream.out == reference
    assert owner.e.replays["main"] > 0


def test_c8_64_rows_logits_taps_and_committed_state_equal_flag_off(owner):
    decoder = BatchedDecoder(owner, 8)
    streams = [_stream(_prompt(550 + i, 13 + i * 4), 24, policy="f7") for i in range(8)]
    for stream in streams:
        decoder.admit(stream)
    windows = [(stream.st, [stream.out[-1], *stream.drafts]) for stream in streams]
    assert sum(len(tokens) for _, tokens in windows) == 64
    copies = [stream.st.clone() for stream in streams]
    b = decoder.buf
    b.l2_prefetch = None
    segments = forward.stage_streams(owner.w, b, windows)
    reference = forward.compute_streams(owner.w, segments, b, eager=True).clone()
    taps = [tap[:64].clone() for tap in b.taps]
    for stream, keep in zip(streams, range(1, 9)):
        forward.commit(owner.w, stream.st, b, 8, keep)
    b.l2_prefetch = l2.Prefetch(owner.w) if owner.w.world > 1 else None
    segments = forward.stage_streams(owner.w, b, [(copy, tokens) for copy, (_, tokens) in zip(copies, windows)])
    actual = forward.compute_streams(owner.w, segments, b, eager=True)
    assert torch.equal(bits(actual), bits(reference))
    assert all(torch.equal(bits(tap[:64]), bits(saved)) for tap, saved in zip(b.taps, taps))
    for copy, stream, keep in zip(copies, streams, range(1, 9)):
        forward.commit(owner.w, copy, b, 8, keep)
        def state(st):
            return [st.rec[cur, li] for li, cur in enumerate(st.cur)] + [st.conv] + decode._row_views(st, st.pos, 0)
        assert copy.pos == stream.st.pos
        assert all(torch.equal(bits(a), bits(b)) for a, b in zip(state(copy), state(stream.st), strict=True))
    decoder.drop()
