"""Sliced prefill one-token turn gates on tiny native checkpoints, including TP3 partial stand-ins."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from test_glm_batched import checkpoint, owner, _prompt, _stream, _serial  # noqa: F401
from test_glm_sliced_cuda import drain


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("sampling", [None, Sampling(2**64 - 3, .713456789012, 20, .934567890123)])
@pytest.mark.parametrize("slot_zero", [False, True])
def test_quantum_keeps_outputs_taps_and_resumed_drafting(owner, monkeypatch, stride, sampling, slot_zero):
    monkeypatch.setattr(p1, "PREFILL_DECODE_QUANTUM", stride)
    if slot_zero:
        from tensorfold.families.glm5_next.cuda.graphs import Graphs
        owner.e.graphs = Graphs(owner.e, tuple(range(1, 9)), tuple(range(1, 9)))
        owner.e.reset()
        owner.drafter.capture()
    streams = [_stream(_prompt(701, 19), 60, policy="f7", sampling=sampling),
               _stream(_prompt(702, 257), 20, policy="f7", sampling=sampling)]
    expected = [_serial(owner, s) for s in streams]
    d = BatchedDecoder(owner, 3)
    if not slot_zero:
        # Retain slot zero until the incumbent binds to the eager nonzero path.
        prefix = _stream(_prompt(700, 13), 1, policy="0")
        d.admit(prefix)
    incumbent, newcomer = streams
    d.admit(incumbent)
    if not slot_zero:
        d.finish([prefix])
    newcomer.prefill_slice_layers = 1
    d.begin_admit(newcomer)
    while newcomer.sid in d.filling:
        done = d.prefill_step()
        done += d.round()
        if not incumbent.done:
            assert incumbent.drafter.context_end == incumbent.st.pos
        d.finish(done)
    before = len(incumbent.depths)
    drain(d)
    assert [s.out for s in streams] == expected
    assert incumbent.prefill_quantum_rounds > 0
    assert any(depth > 0 for depth in incumbent.depths[before:])
    assert not d.streams and d.pool.available == d.capacity
