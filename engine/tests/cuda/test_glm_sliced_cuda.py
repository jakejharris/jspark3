"""Sliced/background prefill device gates on existing tiny checkpoints, including padded TP3 shapes.

Run only in an authorized GPU window. The shared fixture uses rank-partial copies
on one GPU, not multi-host transport; the serving TP3 replay is still mandatory.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from test_glm_batched import checkpoint, owner, _prompt, _stream, _serial


def drain(d):
    for _ in range(600):
        if not d.live():
            return
        done = d.prefill_step()
        done += d.round()
        d.finish(done)
    pytest.fail("sliced prefill/decode did not finish")


@pytest.mark.parametrize("sampling", [None, Sampling(2**64 - 3, .713456789012, 20, .934567890123)])
@pytest.mark.parametrize("policy", ["f7", "3"])
def test_sliced_nonzero_slot_and_coserved_reply_equal_serial(owner, sampling, policy):
    peer = _stream(_prompt(310, 19), 20, sampling=sampling)
    filling = _stream(_prompt(311, 131), 24, policy=policy, sampling=sampling)
    expected = [_serial(owner, s) for s in (peer, filling)]
    d = BatchedDecoder(owner, 2)
    d.admit(peer)
    filling.prefill_slice_layers = 1
    d.begin_admit(filling)
    assert filling.slot == 1 and filling.start > 0
    drain(d)
    assert [peer.out, filling.out] == expected
    assert d.pool.available == d.capacity and d.fill_owner is None


def test_sliced_resume_equals_fresh_snapshot_and_continuation(owner):
    prompt = _prompt(320, 137)
    d = BatchedDecoder(owner, 2)
    prefix = _stream(prompt[:67], 1, policy="f7")
    prefix.prefill_slice_layers = 1
    d.begin_admit(prefix)
    drain(d)
    resumed = _stream(prompt, 20, policy="f7")
    resumed.prefill_slice_layers = 1
    d.begin_admit(resumed)
    assert resumed.cached == 67
    drain(d)
    saved = d.cache[-1]
    fresh_decoder = BatchedDecoder(owner, 2)
    fresh = _stream(prompt, 20, policy="f7")
    fresh.prefill_slice_layers = 1
    fresh_decoder.begin_admit(fresh)
    drain(fresh_decoder)
    reference = fresh_decoder.cache[-1]
    assert resumed.out == fresh.out
    assert saved.mtp_len == reference.mtp_len and saved.drafter_end == reference.drafter_end
    def state(snap):
        rows = [torch.cat(r, dim=0) if isinstance(r, tuple) else r for r in snap.rows]
        # Only complete groups of four have a committed pool key. Snapshot
        # storage includes one future fence; prior decoding can leave any bits
        # there, but _pool_keys overwrites it before it becomes visible.
        first_index = len(owner.e.st.kc) + sum(v is not None for v in owner.e.st.vc)
        for layer in range(min(len(owner.e.st.index or []), len(owner.e.st.kc))):
            pool_row = first_index + layer * 3 + 2
            rows[pool_row] = rows[pool_row][:len(snap.ids) // 4]
        return [snap.rec, snap.conv, *rows, *(snap.drafter_rows or [])]

    actual_state, expected_state = state(saved), state(reference)
    assert len(actual_state) == len(expected_state)
    for i, (actual, expected) in enumerate(zip(actual_state, expected_state)):
        assert torch.equal(actual, expected), {
            "state_tensor": i, "shape": tuple(actual.shape),
            "different_rows": (actual != expected).reshape(actual.shape[0], -1).any(1).nonzero().flatten().tolist(),
        }
    # The former fence at pool16 is now committed. This catches the stale
    # transferred-prefix row found by the original strict native gate.
    pool_state = 2 + len(owner.e.st.kc) + sum(v is not None for v in owner.e.st.vc) + 2
    damaged_pool = actual_state[pool_state].clone()
    damaged_pool[67 // 4].view(torch.uint8).flatten()[0].bitwise_xor_(1)
    assert not torch.equal(damaged_pool, expected_state[pool_state])
    # Negative control: the state equality check detects a one-element corruption.
    damaged = saved.rec.clone()
    damaged.view(torch.uint8).flatten()[0].bitwise_xor_(1)
    assert not torch.equal(damaged, reference.rec)


def test_cancel_after_kda_tap_discards_partial_state_and_reuses_span(owner):
    d = BatchedDecoder(owner, 2)
    cancelled = _stream(_prompt(330, 71), 20)
    cancelled.prefill_slice_layers = 1
    d.begin_admit(cancelled)
    d.prefill_step()
    assert cancelled.fill_layer == 1 and cancelled.st.pos == 0 and not d.cache
    cancelled.cancelled = lambda: True
    d.finish(d.prefill_step())
    assert d.fill_owner is None and owner.e.pbuf.overlay is None
    replacement = _stream(_prompt(331, 71), 20)
    replacement.prefill_slice_layers = 1
    d.begin_admit(replacement)
    assert replacement.start == cancelled.start
    drain(d)
    expected = _serial(owner, replacement)
    assert replacement.out == expected


def test_background_policy_cap_keeps_heterogeneous_outputs_exact(owner, monkeypatch):
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1")
    streams = [_stream(_prompt(340 + i, 17 + i), 20, policy="f7",
                       sampling=Sampling(12 + i, .7, 20, .95)) for i in range(4)]
    expected = [_serial(owner, s) for s in streams]
    d = BatchedDecoder(owner, 4)
    for i, stream in enumerate(streams):
        stream.priority = "background" if i % 2 else "interactive"
        stream.prefill_slice_layers = 1
        d.begin_admit(stream)
    drain(d)
    assert [s.out for s in streams] == expected
    assert all(max(s.depths) <= 1 for s in streams[1::2])
