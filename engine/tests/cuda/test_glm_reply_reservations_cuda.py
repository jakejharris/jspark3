"""Reply reservation tiny-device gates. TP3 stand-in partials here do not certify multi-host serving."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder
from tensorfold.families.glm5_next.cuda.live_store import tensor_views
from tensorfold.families.glm5_next.cuda.pool import TokenPool
from tensorfold.families.glm5_next.cuda.reply_reservations import ReplyConfig, ReplyReservations
from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig
from tensorfold.families.glm5_next.cuda.session_disk import SessionStore
from test_glm_batched import checkpoint, owner, _prompt, _stream, _serial  # noqa: F401


def growing(owner, path, *, slots=4, limit=240):
    owner.pool_limit = limit
    owner.cache_entries = 0
    d = BatchedDecoder(owner, slots)
    d.pool = TokenPool(limit + d.guard, guard=d.guard)
    # The fixture's three-rank path supplies identical rank partials. Storage
    # votes in this single-GPU test use that same explicit stand-in convention.
    owner._gather_ints = lambda values: [list(values)] * owner.world
    disk = SessionStore(path, 2**30, "tiny-exact-checkpoint", min_free=0)
    d.sessions = SessionCache(owner, SessionConfig(disk=True), disk)
    d.growth = ReplyReservations(d, ReplyConfig(True, 64))
    return d


def drain(d):
    for _ in range(1600):
        if not d.live():
            return
        done = d.prefill_step()
        done += d.round()
        d.finish(done)
    pytest.fail("finite response failed to drain")


@pytest.mark.parametrize("policy", ["0", "3", "f7"])
@pytest.mark.parametrize("sampling", [None, Sampling(2**63 + 3, .7134567890123, 20, .934567890123)])
def test_fragmented_pressure_retains_full_count_and_exact_tokens(owner, tmp_path, policy, sampling):
    streams = [_stream(_prompt(601, 17), 1, policy=policy, sampling=sampling),
               _stream(_prompt(602, 17), 200, policy=policy, sampling=sampling),
               _stream(_prompt(603, 17), 180, policy=policy, sampling=sampling)]
    expected = [_serial(owner, s) for s in streams]
    d = growing(owner, tmp_path)
    emitted = {id(s): [] for s in streams}
    for s in streams:
        s.emit = lambda new, s=s: emitted[id(s)].extend(new)
        d.admit(s)
    d.finish([streams[0]])
    drain(d)
    assert [s.out for s in streams] == expected
    assert all(emitted[id(s)] == s.out and len(s.out) == s.count for s in streams)
    assert streams[1].reply_parks and streams[2].reply_parks
    assert streams[1].start == 0 and streams[1].engine is not owner.e
    assert streams[1].engine.graphs is None
    assert not d.growth.store.records and not d.pool.spans


@pytest.mark.parametrize("policy", ["3", "f7"])
@pytest.mark.parametrize("graphs", [False, True])
def test_live_bytes_and_continuation_survive_poisoned_released_span(owner, tmp_path, policy, graphs):
    if graphs:
        from tensorfold.families.glm5_next.cuda.graphs import Graphs
        owner.e.graphs = Graphs(owner.e, tuple(range(1, 9)), tuple(range(1, 9)))
        owner.e.reset()
        owner.drafter.capture()
    s = _stream(_prompt(610, 37), 110, policy=policy,
                sampling=Sampling(11, .72, 20, .94))
    expected = _serial(owner, s)
    d = growing(owner, tmp_path)
    d.admit(s)
    for _ in range(5):
        assert not d.round()
    old_rows, old_pools = s.span - 8, s.span // 4
    draft = s.drafter if s.use_dflash else None
    bits = {name: t.view(torch.uint8).clone() for name, t in tensor_views(s.engine, draft, old_rows, old_pools)}
    meta = (list(s.st.cur), s.st.pos, s.st.mtp_len, s.st.mtp_drafted, list(s.drafts), list(s.out))
    assert d.growth.park(s)
    for _, t in tensor_views(owner.e, owner.drafter if s.use_dflash else None, old_rows, old_pools):
        t.fill_(17)
    assert d.growth.resume(s)
    now = dict(tensor_views(s.engine, s.drafter if s.use_dflash else None, old_rows, old_pools))
    assert all(torch.equal(now[name].view(torch.uint8), raw) for name, raw in bits.items())
    assert (list(s.st.cur), s.st.pos, s.st.mtp_len, s.st.mtp_drafted, s.drafts, s.out) == meta
    # The byte gate must catch a leading ring tile / recurrence corruption,
    # including regions a reusable prompt checkpoint would have omitted.
    name = "draft0h0" if s.use_dflash else "rec"
    damaged = now[name].view(torch.uint8)
    damaged[0].bitwise_xor_(1)
    assert not torch.equal(damaged, bits[name])
    damaged[0].bitwise_xor_(1)
    drain(d)
    assert s.out == expected and len(s.out) == 110


@pytest.mark.parametrize("policy", ["3", "f7"])
def test_complete_sliced_prefill_resumes_without_double_mtp_absorption(owner, tmp_path, policy):
    s = _stream(_prompt(620, 131), 100, policy=policy)
    expected = _serial(owner, s)
    d = growing(owner, tmp_path, limit=300)
    s.prefill_slice_layers = 1
    d.begin_admit(s)
    d.prefill_step()
    assert d.fill_owner == s.sid
    assert d.growth.park(s)  # completes this slice's chunk before writing
    assert s.fill_pos == 64 and not d.filling and owner.e.pbuf.overlay is None
    assert d.growth.resume(s)
    drain(d)
    assert s.out == expected and len(s.out) == s.count


def test_c8_full_64_rows_preserve_adjacent_prefixes_and_unused_pool_canary(owner, tmp_path):
    streams = [_stream(_prompt(640 + i, 13), 160, policy="f7") for i in range(8)]
    expected = [_serial(owner, s) for s in streams]
    d = growing(owner, tmp_path, slots=8, limit=800)
    for s in streams:
        d.admit(s)
    assert sum(1 + len(s.drafts) for s in streams) == 64
    end = max(start + size for start, size in d.pool.spans.values())
    tails = [t[end:] for t in owner.e.st.kc]
    for ik, ig, pk in owner.e.st.index:
        tails.extend((ik[end:], ig[end:], pk[end // 4:]))
    for t in tails:
        t.view(torch.uint8).fill_(0xA5)
    before = [[c[:s.st.pos].clone() for c in s.st.kc] for s in streams]
    assert not d.round()
    for s, prefix in zip(streams, before):
        assert all(torch.equal(c[:13], saved) for c, saved in zip(s.st.kc, prefix))
    drain(d)
    assert [s.out for s in streams] == expected and any(s.reply_parks for s in streams)
    assert all(torch.all(t.view(torch.uint8) == 0xA5) for t in tails)


@pytest.mark.parametrize("width", [16, 32])
def test_wide_verify_grows_and_relocates_nonzero_span_without_changing_tokens(owner, tmp_path, width):
    from tensorfold.families.glm5_next.cuda.decode import Engine

    owner.e = Engine(owner.w, capacity=owner.e.st.capacity, max_rows=width, prefill_rows=64,
                     long_context=True, taps=owner.drafter.tap_layers)
    source = _stream(_prompt(670, 17), 430, policy="0")
    expected = _serial(owner, source)
    d = growing(owner, tmp_path, slots=2, limit=500)
    prefix = _stream(_prompt(671, 300), 1, policy="0")
    d.admit(prefix)
    d.admit(source)
    assert source.start > 0
    d.finish([prefix])
    for _ in range(100):
        if source.done:
            break
        # Supply known-correct copies to isolate wide address bounds from the
        # price table. Host tests separately run the production ROUND2 selector.
        room = min(width - 1, source.count - len(source.out) - 1)
        source.drafts = expected[len(source.out):len(source.out) + room]
        d.finish(d.round())
    assert source.done and source.out == expected
    assert source.reply_parks and source.start == 0 and source.engine is not owner.e
