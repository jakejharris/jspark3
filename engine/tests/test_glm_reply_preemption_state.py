"""Reply prefill cancellation must restore every KDA layer's active recurrent bank.

The ordinary host prefill fake has one KDA layer and overwrites its state at
commit, so it cannot expose a mixed-parity restore after a partial layer pass.
This fake keeps three independent recurrent chains, as kda_block does, and
makes the next sampled token depend on their restored contents.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from test_glm_a2_prefill import express_decoder  # noqa: F401
from test_glm_reply_prefill import background, enable, warm
from test_glm_session_batched import attach, finish_writers  # noqa: F401


@pytest.fixture
def recurrent_decoder(express_decoder, monkeypatch):
    mod, make = express_decoder
    from tensorfold.families.glm5_next.cuda import decode, forward

    def compute(w, st, b, n, *, layer_start=0, layer_stop=None, **kwargs):
        stop = len(w.layers) if layer_stop is None else layer_stop
        data = torch.tensor([[token, st.pos + i] for i, token in enumerate(b.tokens)], dtype=torch.float32)
        for layer in range(layer_start, stop):
            if layer % 2 == 0:
                li = layer // 2
                cur = st.cur[li]
                # forward.kda_block commits each layer immediately in prefill.
                st.rec[1 - cur, li].copy_(st.rec[cur, li] + data[:, 0].sum() * (li + 1))
                st.cur[li] = 1 - cur
                st.conv[li].fill_(st.pos + n)
        if stop < len(w.layers):
            return None
        b.fnormed[:n].copy_(data)
        for tap in b.taps:
            tap[:n].copy_(data)
        st.kc[0][st.pos:st.pos + n].copy_(data)
        for ik, ig, pk in st.index:
            ik[st.pos:st.pos + n].copy_(data)
            ig[st.pos:st.pos + n].copy_(data)
            for at in range(st.pos // 4, (st.pos + n) // 4):
                pk[at].copy_(ik[4 * at:4 * at + 4].sum(0))
        state = sum(int(st.rec[cur, li, 0, 0]) for li, cur in enumerate(st.cur))
        logits = torch.full((1, 64), -100.0)
        logits[0, state % 63] = 100.0
        return logits

    monkeypatch.setattr(decode, "compute", compute)
    # Production prefill commit only advances position: kda_block already wrote
    # the recurrent state and selected its bank, independently for each layer.
    monkeypatch.setattr(decode, "commit", lambda w, st, b, n, keep: st.set_pos(st.pos + keep))

    def build():
        d = enable(make)
        parent = d.owner.drafter
        parent.ring, parent.window = parent.cap, parent.cap - 64
        original = d._engine

        def engine(*args):
            e = original(*args)
            if len(e.st.cur) != 3:
                e.st.cur = [0] * 3
                e.st.rec = torch.zeros(2, 3, 2, 2)
                e.st.conv = torch.zeros(3, 3, 2)
            e.reset = lambda: forward.State.reset(e.st)
            return e

        d._engine = engine
        return d

    return mod, build


@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5, 6], ids=lambda n: f"after-layer-{n}")
@pytest.mark.parametrize("disk_on", [False, True], ids=["reply-prefill", "disk-reply-prefill"])
def test_partial_reply_then_memory_return_matches_fresh(recurrent_decoder, tmp_path, cut, disk_on):
    mod, build = recurrent_decoder
    from tensorfold.cuda.scheduler import Scheduler

    d = build()
    if disk_on:
        disk = attach(d, tmp_path)
        disk.write_gate.quiet_s = 3600  # pending writer cannot mask the state bug
    prefix = [2, 3, 4, 5, 6, 7]  # 2 mod 4, like the failed native 14378-token prefix
    warm(d, prefix)
    if disk_on:
        assert disk.pending is not None and not disk.pending.done.is_set()
    saved = d.cache[-1]
    before = saved.rec.clone()
    job = background(prefix + [8, 9], len(prefix))
    d.begin_admit(job)
    for _ in range(cut):
        d.prefill_step()
    partial = list(job.st.cur)
    assert (len(set(partial)) > 1) == (0 < cut < 5)
    # Real scheduler preemption closes the generator and releases the span.
    job.cancelled = lambda: True
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.decoder, scheduler.reply_active = d, job
    scheduler._preempt_reply_prefill()
    assert not d.filling and d.express_owner is d.fill_owner is None
    assert torch.equal(saved.rec, before)  # immutable snapshot itself was not corrupted
    actual = _stream(prefix + [8, 9, 10, 11, 12, 13, 14, 15], 1, policy="fc7:0.3")
    d.begin_admit(actual)
    assert actual.cached == len(prefix) + (2 if cut == 6 else 0)
    assert actual.reply_prefill_hit_tokens == (2 if cut == 6 else 0)
    if disk_on:
        assert actual.session_cache_source == "memory"
    drain_steps(d)
    fresh = build()
    reference = _stream(actual.prompt, 1, policy="fc7:0.3")
    fresh.begin_admit(reference)
    drain_steps(fresh)
    assert actual.out == reference.out
    got, want = d.cache[-1], fresh.cache[-1]
    assert torch.equal(got.rec, want.rec), "resumed recurrent state differs from fresh"
    assert len(set(actual.st.cur)) == 1
    assert torch.equal(got.conv, want.conv)
    def tensor(row):
        return torch.cat(row) if isinstance(row, tuple) else row
    assert all(torch.equal(tensor(a), tensor(b)) for a, b in zip(got.rows, want.rows))
    if not disk_on:
        from test_glm_prefill_slices import replay
        followers = []
        def follower():
            decoder = build()
            followers.append(decoder)
            return decoder
        observed = replay(mod, follower, d.messages)
        assert all(o[actual.sid] == actual.out for o in observed)
        assert all(torch.equal(f.cache[-1].rec, want.rec) for f in followers)


def test_mixed_bank_negative_control_fails_the_same_state_gate(recurrent_decoder, monkeypatch, tmp_path):
    from tensorfold.families.glm5_next.cuda import decode
    restore = decode.restore
    def old_restore(e, snap, drafter=None):
        selectors = list(e.st.cur)
        restore(e, snap, drafter)
        e.st.cur = selectors  # reintroduce the old missing-normalization bug
    monkeypatch.setattr(decode, "restore", old_restore)
    with pytest.raises(AssertionError, match="resumed recurrent state differs from fresh"):
        test_partial_reply_then_memory_return_matches_fresh(recurrent_decoder, tmp_path, 1, False)
