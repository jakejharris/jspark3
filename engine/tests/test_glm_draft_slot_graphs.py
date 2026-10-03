"""Slot graph ownership, startup budget, reuse and default-off controls."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import draft_graphs
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream, _drain  # noqa: F401


def test_flag_defaults_off_and_rejects_invalid_startup(monkeypatch):
    monkeypatch.delenv("TF_GLM_DRAFT_SLOT_GRAPHS", raising=False)
    assert not draft_graphs.configured(8, "draft")
    for value in ("yes", "2", ""):
        monkeypatch.setenv("TF_GLM_DRAFT_SLOT_GRAPHS", value)
        with pytest.raises(ValueError, match="0 or 1"):
            draft_graphs.configured(8, "draft")
    monkeypatch.setenv("TF_GLM_DRAFT_SLOT_GRAPHS", "1")
    assert draft_graphs.configured(8, "draft")
    for slots, drafter in ((1, "draft"), (8, None)):
        with pytest.raises(ValueError, match="requires"):
            draft_graphs.configured(slots, drafter)


@pytest.fixture
def graph_decoder(setup_decoder, monkeypatch):
    _, make = setup_decoder
    captures = []

    def build(slots=8, enabled=True):
        d = make(slots, pool_limit=4096)
        d.owner.cache_entries = 16
        parent = d.owner.drafter
        parent.ring = 256
        parent.window = 127

        def capture(self):
            captures.append(self.kc[0].data_ptr())
            self.reset()
            self.block_graph = object()
            self.tap_graphs = {n: object() for n in range(1, self.block + 1)}
            self.packed, self.proj = torch.empty(1), torch.empty(1)

        monkeypatch.setattr(type(parent), "capture", capture, raising=False)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
        d.owner._gather_ints = lambda vote: [vote] * 3
        if enabled:
            d.owner.draft_graph_bank = draft_graphs.capture(d.owner, slots)
        return d

    return build, captures


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_captures_only_at_startup_and_keeps_slot_pointers(graph_decoder, slots):
    make, captures = graph_decoder
    d = make(slots)
    assert len(captures) == slots - 1
    parent = d.owner.drafter
    positions, graphs = [], []
    for slot in range(slots):
        # Reply admission LOAD may change the target pool offset and request an eager target.
        a = d._drafter(slot, 1200, 100, eager=True)
        a.pos_dev.fill_(37)
        b = d._drafter(slot, 2000, 200, eager=True)
        assert a is not b and b.context_end == 0
        assert b.pos_dev.item() == 37  # constructing a view never mutates a live pointer
        assert a.pos_dev is b.pos_dev
        assert a.block_graph is b.block_graph
        assert a.kc[0].data_ptr() == parent.kc[0][:, slot * parent.ring:].data_ptr()
        assert a.cap == parent.cap and a.kc[0].stride(0) == parent.kc[0].stride(0)
        positions.append(a.pos_dev.data_ptr())
        graphs.append(a.block_graph)
    assert len(set(positions)) == slots and len(set(graphs)) == slots
    assert len(captures) == slots - 1


@pytest.mark.parametrize("seeded", [False, True])
def test_c8_outputs_resumption_cancel_and_slot_reuse_equal_off(graph_decoder, seeded):
    from tensorfold.engine.exact_sampling import Sampling

    make, _ = graph_decoder
    results = []
    for enabled in (False, True):
        d = make(enabled=enabled)
        streams = [_stream([3 + i, 11, 5], 15,
                           sampling=Sampling(i + 40, .7, 20, .95) if seeded else None)
                   for i in range(8)]
        for s in streams:
            d.admit(s)
        cancelled = streams[3]
        d.finish([cancelled])
        replacement = _stream([7, 4, 9], 12)
        d.admit(replacement)
        assert replacement.slot == cancelled.slot
        _drain(d)
        resumed = _stream(streams[0].prompt + [6, 8], 12, sampling=streams[0].sampling)
        d.admit(resumed)
        assert resumed.cached == 3
        _drain(d)
        results.append([s.out for s in streams + [replacement, resumed]])
    assert results[0] == results[1]


def test_budget_failure_is_collective_and_stops_before_next_slot(graph_decoder, monkeypatch):
    make, captures = graph_decoder
    d = make(enabled=False)
    reads = iter([0, draft_graphs.SLOT_BUDGET + 1])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: next(reads))
    with pytest.raises(RuntimeError, match="budget"):
        draft_graphs.capture(d.owner, 8)
    assert len(captures) == 1
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    d.owner._gather_ints = lambda _: [[1], [0], [1]]
    with pytest.raises(RuntimeError, match="budget"):
        draft_graphs.capture(d.owner, 8)
    assert len(captures) == 2


def test_unbounded_ring_refused_before_capture():
    with pytest.raises(ValueError, match="bounded"):
        draft_graphs.capture(SimpleNamespace(drafter=SimpleNamespace(ring=0)), 8)
