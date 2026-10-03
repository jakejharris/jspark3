"""Exact replay: a prompt equal to a stored prompt-end snapshot samples its first token without prefill."""

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import prefill_options as p1
from test_cuda_geometry import allocations  # noqa: F401  (fake Triton for CPU-only installations)
from test_glm_batched_host import _drain, _logits, _stream, setup_decoder  # noqa: F401


@pytest.fixture
def replay(setup_decoder, monkeypatch):
    """The host decoder with the flag on; its fake prefill keeps the head row it sampled from, as prefill_steps does."""
    mod, make = setup_decoder
    monkeypatch.setattr(p1, "EXACT_REPLAY", True)
    base, calls = mod.prefill, []

    def prefill(e, prompt, sampling, mtp, drafter, resume, **kw):
        calls.append((list(prompt), len(resume.ids) if resume is not None else 0))
        first = base(e, prompt, sampling, mtp, drafter, resume)
        if p1.EXACT_REPLAY:
            e.last_logits = _logits(prompt[-1:], len(prompt) - 1)
        return first

    monkeypatch.setattr(mod, "prefill", prefill)
    # fq7 (the shipped policy) prices drafts from the drafter's claims.
    drafter_type = type(make(2).owner.drafter)
    original_propose = drafter_type.propose

    def observed_propose(self, pending, depth, sampling, confidence):
        output = original_propose(self, pending, depth, sampling, confidence)
        if getattr(self, "observe", False):
            self.last_observation = {"claims": [.8] * len(output),
                                     "candidates": [list(range(64)) for _ in output]}
        return output

    monkeypatch.setattr(drafter_type, "propose", observed_propose)
    return mod, make, calls


def _snap(mod, ids, *, head=True, drafter_end=None, mtp_len=-1):
    snap = mod.Snapshot(list(ids), torch.zeros(1), torch.zeros(1), None, mtp_len,
                        len(ids) if drafter_end is None else drafter_end)
    if head:
        snap.last_logits, snap.head_cols, snap.last_hidden = torch.zeros(1, 64), 64, torch.zeros(1, 2)
    return snap


# -- lookup ------------------------------------------------------------------------------------------------------
def test_lookup_takes_equal_entry_only_with_head_row(replay):
    mod, make, _ = replay
    d = make(2)
    code = mod.encode_policy("fq7:0.3")
    shorter, equal = _snap(mod, [1, 2]), _snap(mod, [1, 2, 3])
    d.cache = [shorter, equal]
    assert d._saved([1, 2, 3], code) is equal
    assert d._saved([1, 2, 3], code, cached=3) is equal and d._saved([1, 2, 3], code, cached=2) is shorter
    assert d._saved([1, 2, 3, 4], code) is equal                    # strict prefixes are unchanged
    assert d._saved([1, 2, 4], code) is shorter                     # same length, different ids
    assert d._saved([1, 2, 3], code, memory_only=True) is shorter   # reply prefill keeps strict prefixes
    equal.last_logits = None
    assert d._saved([1, 2, 3], code) is shorter                     # no head row: the old rule
    equal.last_logits, equal.drafter_end = torch.zeros(1, 64), 2
    assert d._saved([1, 2, 3], code) is shorter                     # drafter not at the prompt end


def test_lookup_never_replays_mtp_or_mismatched_images(replay):
    mod, make, _ = replay
    d = make(2)
    mtp = [1, 3, 0, 0]
    shorter, equal = _snap(mod, [1, 2], mtp_len=1), _snap(mod, [1, 2, 3], mtp_len=2)
    d.cache = [shorter, equal]
    assert d._saved([1, 2, 3], mtp) is shorter
    image = _snap(mod, [1, -7, -7])
    image.image_digests = (b"a" * 32,)
    d.cache = [image]
    code = mod.encode_policy("fq7:0.3")
    assert d._saved([1, -7, -7], code, images=[SimpleNamespace(digest=b"a" * 32)]) is image
    assert d._saved([1, -7, -7], code, images=[SimpleNamespace(digest=b"b" * 32)]) is None


def test_flag_off_is_the_strict_prefix_rule(replay, monkeypatch):
    mod, make, calls = replay
    monkeypatch.setattr(p1, "EXACT_REPLAY", False)
    d = make(2)
    d.cache = [_snap(mod, [1, 2, 3])]
    assert d._saved([1, 2, 3], mod.encode_policy("fq7:0.3")) is None
    d.cache = []
    first = _stream([1, 2, 3, 4], 5)
    d.admit(first)
    _drain(d)
    assert d.cache[0].last_logits is None and d.cache[0].last_hidden is None
    again = _stream([1, 2, 3, 4], 5)
    d.admit(again)
    _drain(d)
    assert again.cached == 0 and not again.exact_replay and again.out == first.out
    assert calls == [([1, 2, 3, 4], 0), ([1, 2, 3, 4], 0)]


def test_head_row_width_is_rank_invariant():
    from tensorfold.families.glm5_next.cuda.decode import keep_head_row
    from tensorfold.families.glm5_next.cuda.weights import vocab_share

    widths = set()
    for rank in range(3):
        lo, hi = vocab_share(154880, rank, 3)
        logits = torch.arange(3 * (hi - lo), dtype=torch.float32).view(3, hi - lo)
        row, cols = keep_head_row(SimpleNamespace(cfg=SimpleNamespace(vocab=154880), world=3), logits)
        assert cols == hi - lo and torch.equal(row[:, :cols], logits[-1:])
        widths.add(row.shape[1])
    assert widths == {51648}


def test_snapshot_bytes_count_the_head_row(replay):
    mod, _, _ = replay
    snap = _snap(mod, [1, 2, 3], head=False)
    plain = mod.snapshot_bytes(snap)
    snap.last_logits, snap.last_hidden = torch.zeros(1, 64, dtype=torch.bfloat16), torch.zeros(1, 2,
                                                                                              dtype=torch.bfloat16)
    assert mod.snapshot_bytes(snap) == plain + 128 + 4


# -- admission ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ring", [False, True], ids=["full-draft-cache", "draft-ring"])
@pytest.mark.parametrize("sampling", [None, Sampling(2**64 - 3, .71234567891, 16, .92345678911)],
                         ids=["greedy", "seeded"])
def test_replay_equals_cold_without_prefill(replay, ring, sampling):
    mod, make, calls = replay
    d = make(2)
    if ring:
        d.owner.drafter.ring, d.owner.drafter.window = 64, 3
    prompt = [5, 9, 2, 7, 11]
    cold = _stream(prompt, 9, sampling=sampling, policy="fq7:0.3")
    d.admit(cold)
    _drain(d)
    entry = d.cache[-1]
    assert entry.last_logits is not None and entry.head_cols == 64
    # Poison every live cache: a replay must restore all of it.
    d.owner.e.st.kc[0].fill_(float("nan"))
    d.owner.e.st.rec.fill_(-1)
    d.owner.e.st.conv.fill_(-1)
    d.owner.drafter.kc[0].fill_(float("nan"))
    d.owner.drafter.context_end = 0
    warm = _stream(prompt, 9, sampling=sampling, policy="fq7:0.3")
    d.admit(warm)
    assert warm.cached == len(prompt) and warm.exact_replay
    assert calls == [(prompt, 0)]                                   # no prefill for the replay
    _drain(d)
    assert warm.out == cold.out
    assert warm.drafted_tokens == cold.drafted_tokens and warm.accepted_draft_tokens == cold.accepted_draft_tokens
    assert d.cache == [entry]                                       # kept, not copied again
    third = _stream(prompt, 9, sampling=sampling, policy="fq7:0.3")
    d.admit(third)
    _drain(d)
    assert third.exact_replay and third.out == cold.out


def test_continuation_after_replay_still_resumes(replay):
    mod, make, calls = replay
    d = make(2)
    prompt = [3, 1, 4, 1, 5]
    for _ in range(2):
        d.admit(_stream(prompt, 4))
        _drain(d)
    longer = _stream(prompt + [9, 2], 4)
    d.admit(longer)
    assert longer.cached == len(prompt) and not longer.exact_replay
    assert calls[-1] == (prompt + [9, 2], len(prompt))


def test_stepped_replay_is_one_zero_row_fill_and_followers_mirror(replay, monkeypatch):
    mod, make, calls = replay
    original = mod.prefill

    def steps(e, prompt, sampling, *, mtp, drafter, resume, rows, slice_end=None):
        begin = len(resume.ids) if resume else 0
        while begin + rows() < len(prompt):
            begin += rows()
            e.st.set_pos(begin)
            yield begin
        return original(e, prompt, sampling, mtp, drafter, resume)

    monkeypatch.setattr(mod, "prefill_steps", steps)
    leader = make(2, limit=256)
    leader.owner.e.prefill_rows = 4
    prompt = list(range(1, 11))
    cold = _stream(prompt, 6, policy="fq7:0.3")
    leader.begin_admit(cold)
    while leader.filling:
        leader.prefill_step()
    _drain(leader)
    warm = _stream(prompt, 6, policy="fq7:0.3")
    warm.prefill_slice_layers = 3                                   # a replay has no forward to slice
    leader.begin_admit(warm)
    assert warm.exact_replay and not warm.out
    leader.prefill_step()
    assert warm.out and sid_fills(mod, leader, warm.sid) == [[mod.FILL, warm.sid, len(prompt)]]
    _drain(leader)
    assert warm.out == cold.out and len(calls) == 1

    follower = make(2, limit=256)
    follower.owner.e.prefill_rows = 4
    follower.rank = follower.owner.rank = 1
    observed = {}
    original_finish = follower._finish

    def finish(sids):
        observed.update({sid: list(follower.streams[sid].out) for sid in sids})
        original_finish(sids)

    follower._finish = finish
    messages = iter(leader.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert observed == {cold.sid: cold.out, warm.sid: warm.out}
    assert follower.pool.available == follower.capacity


def sid_fills(mod, d, sid):
    return [m for m in d.messages if m[0] in (mod.FILL, mod.FILL_SLICE) and m[1] == sid]


def test_replay_fill_rejects_rows(replay, monkeypatch):
    mod, make, _ = replay
    d = make(2, limit=256)
    d.owner.e.prefill_rows = 4
    prompt = [4, 4, 2]
    d.admit(_stream(prompt, 3))
    _drain(d)
    warm = _stream(prompt, 3)
    d.begin_admit(warm)
    with pytest.raises(RuntimeError, match="one zero-row step"):
        d._fill(warm.sid, len(prompt), 2)


# -- restored state: every tensor a later token reads, and the rank vote -------------
@pytest.mark.parametrize('ring', [False, True])
def test_restore_bits_before_second_token(replay, ring):
    mod, make, _ = replay
    from tensorfold.families.glm5_next.cuda.decode import _snapshot_row_views, _ring_slots
    d = make(2)
    if ring:
        d.owner.drafter.ring, d.owner.drafter.window = 64, 3
    prompt = [5, 9, 2, 7, 11]
    cold = _stream(prompt, 1)
    d.admit(cold)
    snap = d.cache[-1]
    expected_rows = [x.clone() for x in _snapshot_row_views(cold.engine, snap, cold.drafter)]
    expected_rec, expected_conv = snap.rec.clone(), snap.conv.clone()
    expected_hidden = snap.last_hidden.clone()
    expected_ring = [x.clone() for x in snap.drafter_rows or []]
    _drain(d)
    e, draft = d.owner.e, d.owner.drafter
    for t in e.st.kc + [t for t in e.st.vc if t is not None]:
        t.fill_(float('nan'))
    for triple in e.st.index:
        for t in triple:
            t.fill_(float('nan'))
    e.st.rec.fill_(float('nan'))
    e.st.conv.fill_(float('nan'))
    e.last_hidden.fill_(float('nan'))
    for t in draft.kc + draft.vc:
        t.fill_(float('nan'))
    draft.context_end = 0
    draft.pos_dev.zero_()
    warm = _stream(prompt, 1)
    d.admit(warm)
    assert warm.exact_replay and warm.out == cold.out
    for got, expected in zip(_snapshot_row_views(warm.engine, snap, warm.drafter), expected_rows, strict=True):
        assert torch.equal(got, expected)
    assert torch.equal(warm.st.rec[warm.st.cur[0]], expected_rec)
    assert torch.equal(warm.st.conv, expected_conv)
    assert torch.equal(warm.engine.last_hidden, expected_hidden)
    assert warm.st.pos == len(prompt) and warm.st.pos_dev.item() == len(prompt)
    assert warm.st.mtp_len == 0 and warm.st.mtp_drafted == 0
    assert warm.drafter.context_end == len(prompt) and warm.drafter.pos_dev.item() == len(prompt)
    if ring:
        idx = _ring_slots(warm.drafter, len(prompt))
        for cache, expected in zip(warm.drafter.kc + warm.drafter.vc, expected_ring, strict=True):
            assert torch.equal(cache.index_select(1, idx), expected)


def test_missing_head_without_sessions_rejects_follower(replay):
    mod, make, _ = replay
    leader, follower = make(2), make(2)
    prompt = [1, 2, 3, 4, 5]
    for d in [leader, follower]:
        d.admit(_stream(prompt, 1))
        _drain(d)
        d.messages.clear()
    follower.cache[0].last_logits = None
    follower.rank = follower.owner.rank = 1
    warm = _stream(prompt, 1)
    leader.admit(warm)
    assert warm.exact_replay
    messages = iter(leader.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match='rank lost the named GLM prompt snapshot'):
        follower.follow()


@pytest.mark.parametrize('stepped', [False, True])
def test_three_ranks_missing_head_vote_is_cold(replay, stepped):
    _, make, _ = replay
    from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig
    ranks = [make(2) for _ in range(3)]
    prompt = [1, 2, 3, 4, 5]
    for rank, d in enumerate(ranks):
        d.owner.e.prefill_rows = 8
        d.admit(_stream(prompt, 1))
        _drain(d)
        d.messages.clear()
        d.rank = d.owner.rank = rank
        # Production restore vote; no disk writer needed for this memory-only case.
        d.sessions = SessionCache(d.owner, SessionConfig())
    ranks[1].cache[0].last_logits = None
    barrier, votes = threading.Barrier(3, timeout=5), [None] * 3
    qs = [queue.Queue(), queue.Queue()]
    def gather(rank, values):
        votes[rank] = list(values)
        barrier.wait()
        answer = [list(row) for row in votes]
        barrier.wait()
        return answer
    def broadcast(msg):
        for q in qs:
            q.put(list(msg))
    ranks[0].share = broadcast
    errors, observed, threads = [], [], []
    for rank, d in enumerate(ranks):
        d.owner._gather_ints = lambda values, rank=rank: gather(rank, values)
        if rank:
            d.share = lambda _, q=qs[rank - 1]: q.get(timeout=5)
            original = d._finish
            def finish(ids, d=d, original=original):
                observed.extend((d.rank, d.streams[sid].cached, d.streams[sid].exact_replay,
                                 list(d.streams[sid].out)) for sid in ids)
                original(ids)
            d._finish = finish
            def follow(d=d):
                try:
                    d.follow()
                except RuntimeError as exc:
                    if 'unknown GLM worker message' not in str(exc):
                        errors.append(exc)
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=follow, daemon=True)
            thread.start()
            threads.append(thread)
    s = _stream(prompt, 1)
    # Sync admission uses the fixture's cold forward; the vote is identical for stepped admission.
    try:
        if stepped:
            ranks[0].begin_admit(s)
            assert s.cached == 0 and not s.exact_replay
            ranks[0].finish([s])
        else:
            ranks[0].admit(s)
            _drain(ranks[0])
    finally:
        broadcast([99])
    for thread in threads:
        thread.join(timeout=6)
        assert not thread.is_alive()
    assert not errors
    assert s.cached == 0 and not s.exact_replay
    assert sorted(observed) == [(1, 0, False, s.out), (2, 0, False, s.out)]
