"""Real DFlash graphs: slot outputs, ring wrap, cached reuse and c8 verification.

The TP2/TP3 cases repeat rank partials on one GPU, not multi-host transport.
"""

import copy

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import draft_graphs  # noqa: E402
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder  # noqa: E402
from test_glm_batched import owner, checkpoint, _stream, _prompt, _drain  # noqa: E402,F401


def test_slot_graph_bank_matches_eager_and_resumed_c8(owner):
    parent = owner.drafter
    parent.capture()
    # These fixtures use a local gather stand-in; startup agreement is separate.
    owner._gather_ints = lambda v: [v] * owner.world
    owner.draft_graph_bank = draft_graphs.capture(owner, 8)
    decoder = BatchedDecoder(owner, 8)
    gen = torch.Generator(device="cuda").manual_seed(812)
    taps = torch.randn((parent.ring + 3, len(parent.tap_layers) * parent.D),
                       dtype=torch.bfloat16, device="cuda", generator=gen)
    for slot in range(8):
        graph = decoder._drafter(slot, 1024, 128, eager=True)
        eager = copy.copy(graph)
        eager.kc = [c.clone() for c in graph.kc]
        eager.vc = [c.clone() for c in graph.vc]
        eager.cap = eager.kc[0].shape[1]
        eager.pos_dev = torch.zeros_like(graph.pos_dev)
        eager.block_graph, eager.tap_graphs = None, {}
        graph.reset()
        eager.reset()
        # Seeded and greedy outputs, graph tap sizes 1..8, ring wrap and eager
        # prefill-size taps. Compare all candidate values, not only the winner.
        for n in [1, 2, 3, 4, 5, 6, 7, 8, parent.ring + 3]:
            for d in (graph, eager):
                d.add_taps(taps[:n])
            for depth in (1, 4, 7):
                a, b = graph.candidates(11, depth), eager.candidates(11, depth)
                assert all(np.array_equal(x, y) for x, y in zip(a, b))
                for sampling in (None, Sampling(177, .7, 20, .95)):
                    assert graph.chain(*a, 11, graph.context_end + 1, sampling, .3) == \
                           eager.chain(*b, 11, eager.context_end + 1, sampling, .3)
            assert graph.context_end == eager.context_end
            assert torch.equal(graph.pos_dev, eager.pos_dev)
            for a, b in zip(graph.kc + graph.vc, eager.kc + eager.vc):
                # Initial unused rows differ; after the ring-length update all
                # rows, including speculative rows, must be identical.
                if n > parent.ring:
                    assert torch.equal(a[:, :parent.ring], b[:, :parent.ring])
        # A wrong-position control must fail the same state gate.
        eager.pos_dev.add_(1)
        assert not torch.equal(graph.pos_dev, eager.pos_dev)

    results = []
    bank = owner.draft_graph_bank
    for enabled in (False, True):
        owner.draft_graph_bank = bank if enabled else None
        decoder = BatchedDecoder(owner, 8)
        streams = [_stream(_prompt(i, 15 + i), count=24,
                           sampling=Sampling(11 + i, .7, 20, .95) if i % 2 else None)
                   for i in range(8)]
        for s in streams:
            decoder.admit(s)
        decoder.finish([streams[3]])
        replacement = _stream(_prompt(99, 11), count=16)
        decoder.admit(replacement)
        assert replacement.slot == streams[3].slot
        _drain(decoder)
        resumed = _stream(streams[7].prompt + [5, 7], count=16, sampling=streams[7].sampling)
        decoder.admit(resumed)
        assert resumed.cached == len(streams[7].prompt)
        _drain(decoder)
        results.append([s.out for s in streams + [replacement, resumed]])
    assert results[0] == results[1]
