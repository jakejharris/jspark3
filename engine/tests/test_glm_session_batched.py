"""Pooled session/checkpoint protocol, ownership and checkpoint hooks on CPU workers."""

import queue
import threading
from types import SimpleNamespace

import pytest

from test_glm_batched_host import setup_decoder, allocations, _stream, _drain as _decode_drain  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig
from tensorfold.families.glm5_next.cuda.session_disk import SessionStore


@pytest.fixture(autouse=True)
def finish_writers(monkeypatch):
    stores = []
    original = SessionStore.__init__
    def initialize(store, *args, **kwargs):
        original(store, *args, **kwargs)
        stores.append(store)
    monkeypatch.setattr(SessionStore, "__init__", initialize)
    yield
    for store in stores:
        store.wait_pending()


def _drain(d):
    _decode_drain(d)
    if d.sessions is not None and d.sessions.store is not None:
        d.sessions.store.wait_pending()  # explicit durability barrier for disk-restore gates


def attach(d, path, **flags):
    d.owner.checkpoint_after = lambda pos: (pos // 4 + 1) * 4
    d.owner._gather_ints = lambda v: [v] * 3
    disk = SessionStore(path, 2**20, "fixture", min_free=0)
    d.sessions = SessionCache(d.owner, SessionConfig(disk=True, **flags), disk, role_ids=(50,))
    return disk


def test_evicted_prompt_returns_from_disk_and_continuation_matches_fresh(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    disk = attach(d, tmp_path)
    first = _stream([1, 2, 3, 4, 5], 4)
    d.admit(first)
    assert not disk.entries                         # no write on request's critical prefill path
    _drain(d)
    d.cache.clear()
    resumed = _stream(first.prompt + [7, 8], 8)
    d.admit(resumed)
    assert resumed.cached == 5
    _drain(d)
    fresh = make(2, drafter=False)
    control = _stream(resumed.prompt, 8)
    fresh.admit(control)
    _drain(fresh)
    assert resumed.out == control.out


@pytest.mark.parametrize("fault", ["missing", "corrupt", "incompatible"])
def test_three_rank_admission_votes_after_load_and_falls_back_without_hang(setup_decoder, tmp_path, fault):
    _, make = setup_decoder
    ranks = [make(2, drafter=False) for _ in range(3)]
    for rank, d in enumerate(ranks):
        disk = attach(d, tmp_path / str(rank))
        s = _stream([1, 2, 3, 4, 5], 4)
        d.admit(s)
        _drain(d)
        d.cache.clear()
        d.messages.clear()
        d.rank = d.owner.rank = rank
    bad = ranks[1].sessions.store
    path = next(iter(bad.entries.values())).path
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        from tensorfold.families.glm5_next.cuda.disk_io import TensorReader
        with TensorReader(path) as r:
            at = r.base + r.layout["kc0"]["offset"]
        with open(path, "r+b") as f:
            f.seek(at)
            byte = f.read(1)[0]
            f.seek(at)
            f.write(bytes([byte ^ 1]))
    else:
        ranks[1].sessions.store = SessionStore(path.parent, 2**20, "changed-weights", min_free=0)
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
    errors, observed = [], []
    threads = []
    for rank, d in enumerate(ranks):
        d.owner._gather_ints = lambda v, rank=rank: gather(rank, v)
        if rank:
            d.share = lambda _, q=qs[rank - 1]: q.get(timeout=5)
            original = d._finish
            def finish(ids, d=d, original=original):
                observed.extend((d.rank, s.cached, list(s.out)) for sid in ids if (s := d.streams.get(sid)))
                original(ids)
            d._finish = finish
            def follow(d=d):
                try:
                    d.follow()
                except RuntimeError as exc:
                    if "unknown GLM worker message" not in str(exc):
                        errors.append(exc)
                except BaseException as exc:
                    errors.append(exc)
            t = threading.Thread(target=follow, daemon=True)
            t.start()
            threads.append(t)
    resumed = _stream([1, 2, 3, 4, 5, 8, 9], 8)
    try:
        ranks[0].admit(resumed)
        _drain(ranks[0])
    finally:
        broadcast([99])
    for t in threads:
        t.join(timeout=6)
        assert not t.is_alive()
    assert not errors
    assert resumed.cached == 0
    assert sorted(observed) == [(1, 0, resumed.out), (2, 0, resumed.out)]
    assert all(d.pool.available == d.capacity for d in ranks)


def test_cache_disabled_write_pins_span_and_write_error_releases_it(setup_decoder, tmp_path, monkeypatch):
    _, make = setup_decoder
    d = make(2, drafter=False)
    d.owner.cache_entries = 0
    disk = attach(d, tmp_path)
    first = _stream([1, 2, 3, 4, 5], 1)
    d.admit(first)
    expected = first.st.kc[0][:5].clone()
    entered, release = threading.Event(), threading.Event()
    original = disk.put
    def delayed(e, snap):
        entered.set()
        assert release.wait(5)
        assert first.sid in d.pool.spans
        assert e.st.kc[0][:5].equal(expected)
        return original(e, snap)
    monkeypatch.setattr(disk, "put", delayed)
    failures = []
    def finish():
        try:
            d.finish([first])
        except BaseException as exc:
            failures.append(exc)
    t = threading.Thread(target=finish, daemon=True)
    t.start()
    assert entered.wait(5)
    assert first.sid in d.pool.spans and not disk.entries
    release.set()
    t.join(5)
    assert not failures and not t.is_alive()
    assert first.sid not in d.pool.spans and disk.entries
    other = _stream([7, 8, 9], 1)
    d.admit(other)
    assert other.start == first.start
    from tensorfold.families.glm5_next.cuda.disk_io import DirectFile
    def fail(*a, **kw):
        raise OSError("injected IO failure")
    monkeypatch.setattr(DirectFile, "write", fail)
    monkeypatch.setattr(disk, "put", original)
    d.finish([other])
    assert d.broken is None and d.pool.available == d.capacity and disk.errors == 1


def real_stepper(mod, monkeypatch):
    """Exercise production prefill_steps and mark/keep with deterministic CPU arithmetic."""
    import torch
    from tensorfold.families.glm5_next.cuda import decode
    def stage(w, st, b, tokens):
        return tokens
    def compute(w, st, b, tokens, **kw):
        data = torch.tensor([[token, st.pos + i] for i, token in enumerate(tokens)], dtype=torch.float32)
        st.kc[0][st.pos:st.pos + len(tokens)].copy_(data)
        b.fnormed[:len(tokens)].copy_(data)
        return data
    def commit(w, st, b, n, keep):
        st.set_pos(st.pos + keep)
        st.rec[st.cur[0]].fill_(st.pos)
        st.conv.fill_(st.pos)
    monkeypatch.setattr(decode, "stage", stage)
    monkeypatch.setattr(decode, "compute", compute)
    monkeypatch.setattr(decode, "commit", commit)
    monkeypatch.setattr(decode, "chunks_for", lambda *a: 1)
    monkeypatch.setattr(mod, "prefill_steps", decode.prefill_steps)
    monkeypatch.setattr(mod, "prefill", decode.prefill)


def step_engine(d):
    import torch
    e = d.owner.e
    e.prefill_rows = 8
    e.pbuf = SimpleNamespace(fnormed=torch.zeros(8, 2), overlay=None)
    e.reset = lambda: e.st.reset()
    e.overlay = lambda start, n: None
    e.sample = lambda logits, positions, sampling: [3]


@pytest.mark.parametrize("cancel_enabled", [False, True])
def test_completed_chunk_cancel_survives_and_retries_from_exact_anchor(setup_decoder, tmp_path, monkeypatch,
                                                                      cancel_enabled):
    mod, make = setup_decoder
    real_stepper(mod, monkeypatch)
    d = make(2, drafter=False)
    step_engine(d)
    disk = attach(d, tmp_path, cancel=cancel_enabled)
    s = _stream(list(range(20)), 1)
    stop = [False]
    s.cancelled = lambda: stop[0]
    d.begin_admit(s)
    d.prefill_step()
    assert s.fill_pos == 8
    stop[0] = True
    d.finish(d.prefill_step())
    assert d.pool.available == d.capacity and not d.sessions.staged
    assert bool(disk.entries) == cancel_enabled
    retry = _stream(list(range(20)), 1)
    d.begin_admit(retry)
    assert retry.cached == (8 if cancel_enabled else 0)
    while d.filling:
        d.prefill_step()
    _drain(d)
    assert retry.out == [3]


def test_grid_and_message_boundaries_are_rank_named_and_deferred_until_reply(setup_decoder, tmp_path, monkeypatch):
    mod, make = setup_decoder
    real_stepper(mod, monkeypatch)
    d = make(2, drafter=False)
    step_engine(d)
    disk = attach(d, tmp_path, checkpoints=True)
    s = _stream([1, 2, 3, 50, 5, 6, 7, 8, 9, 10, 11], 1)
    d.begin_admit(s)
    while d.filling:
        d.prefill_step()
        assert not disk.entries
    assert [m[2] for m in d.messages if m[0] == mod.FILL] == [3, 4, 8, 11]
    d.finish([s])
    disk.wait_pending()
    assert {len(e.ids) for e in disk.entries.values()} == {3, 4, 8, 11}
    d.cache.clear()
    fork = _stream(s.prompt[:9] + [22, 23], 1)
    d.begin_admit(fork)
    assert fork.cached == 8
    while d.filling:
        d.prefill_step()
    d.finish([fork])


def test_hash_observer_does_not_change_outputs_and_exposes_every_rank(setup_decoder, tmp_path):
    _, make = setup_decoder
    outputs = []
    for enabled in (False, True):
        d = make(2, drafter=False)
        attach(d, tmp_path / str(enabled), hash_gate=enabled)
        s = _stream([1, 2, 3, 4, 5], 4)
        d.admit(s)
        _drain(d)
        outputs.append(s.out)
        if enabled:
            assert len(s.stats()["session_state_sha256"]) == 3
            assert all(len(h) == 64 for h in s.stats()["session_state_sha256"])
            assert s.stats()["session_hash_s"] > 0
        else:
            assert "session_state_sha256" not in s.stats()
            assert "session_hash_s" not in s.stats()
    assert outputs[0] == outputs[1]


def test_miss_reasons_use_evidence_and_do_not_guess_compaction(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    attach(d, tmp_path)
    cache = d.sessions
    import numpy as np
    cache.history = [(np.array([1, -7, -7, 4], dtype=np.int32), (b"a" * 32,))]
    def request(prompt, digests=(), cached=0, event=None):
        s = SimpleNamespace(prompt=prompt, image_digests=digests, cached=cached, session_event=event)
        cache.begin(s)
        return s
    assert request([1, -8, -8, 4, 5], (b"b" * 32,)).session_miss_reason == "image-changed"
    assert request([1, -7, -7, 4, 5], (b"a" * 32,)).session_miss_reason == "store-evicted"
    assert request([1, -7, -7, 5, 6], (b"a" * 32,), cached=3).session_miss_reason == "fork"
    unknown = request([44, 45])
    assert unknown.session_miss_reason == "none" and unknown.session_reason_evidence == "unknown"
    compaction = request([44, 45], event="compaction")
    assert compaction.session_miss_reason == "compaction" and compaction.session_reason_evidence == "client-reported"


def test_stat_failure_during_finish_still_releases_span(setup_decoder, tmp_path, monkeypatch):
    _, make = setup_decoder
    d = make(2, drafter=False)
    disk = attach(d, tmp_path)
    s = _stream([1, 2, 3], 1)
    d.admit(s)
    def fail(*a):
        raise OSError("filesystem temporarily unavailable")
    monkeypatch.setattr("tensorfold.families.glm5_next.cuda.session_disk.shutil.disk_usage", fail)
    d.finish([s])
    assert d.broken is None and d.pool.available == d.capacity and not d.sessions.staged
    assert disk.errors == 1


def test_r2_repairs_only_newly_completed_pool_row_during_memory_extension(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2)
    d.owner.drafter.ring, d.owner.drafter.window = 16, 3
    attach(d, tmp_path)
    a = _stream([1, 2, 3, 4, 5], 1)
    d.admit(a)
    # Model an unwritten parent fence; the host fake normally initializes every pool.
    old = d.cache[-1]
    old.rows[-1][1].fill_(-31)
    d.finish([a])
    b = _stream(a.prompt + [6, 7, 8, 9], 1)
    d.admit(b)
    from tensorfold.families.glm5_next.cuda.decode import load_rows
    child = d.cache[-1]
    expected = b.st.index[0][2][1].clone()
    b.st.index[0][2].zero_()
    load_rows(b.engine, child, b.drafter)
    assert b.st.index[0][2][1].equal(expected)
    d.finish([b])


@pytest.mark.parametrize("cut", [1, 3, 5])
@pytest.mark.parametrize("completed", [False, True])
def test_sliced_cancel_writes_only_last_committed_state(sliced_decoder, tmp_path, completed, cut):
    import torch
    _, build = sliced_decoder
    d = build()
    disk = attach(d, tmp_path, cancel=True)
    s = _stream(list(range(20)), 1, policy="0")
    s.prefill_slice_layers = cut
    d.begin_admit(s)
    if completed:
        while s.fill_pos < 8:
            d.prefill_step()
        assert [len(snap.ids) for _, snap, _ in d.sessions.staged] == [8]
    d.prefill_step()                         # next chunk has touched some layers, but has not committed
    assert d.fill_owner == s.sid and s.st.pos == (8 if completed else 0)
    assert torch.all(s.st.conv == (16 if completed else 8))
    s.cancelled = lambda: True
    d.finish(d.prefill_step())
    assert d.fill_owner is None and not d.sessions.staged and d.pool.available == d.capacity
    assert {len(e.ids) for e in disk.entries.values()} == ({8} if completed else set())
    retry = _stream(s.prompt, 1, policy="0")
    retry.prefill_slice_layers = cut
    d.begin_admit(retry)
    assert retry.cached == (8 if completed else 0) and retry.start == s.start
    if completed:
        snap = disk.load(retry.engine, next(iter(disk.entries.values())))
        assert torch.all(snap.conv == 8)       # partial layer state (16) must never become a checkpoint
    drain_steps(d)
    fresh = build()
    control = _stream(s.prompt, 1, policy="0")
    control.prefill_slice_layers = cut
    fresh.begin_admit(control)
    drain_steps(fresh)
    assert retry.out == control.out
    assert torch.equal(retry.st.conv, control.st.conv)
    assert torch.equal(retry.st.kc[0][:20], control.st.kc[0][:20])


@pytest.mark.parametrize("cut", [1, 3, 5])
def test_sliced_grid_keeps_rank_named_stop_until_commit(sliced_decoder, tmp_path, cut):
    mod, build = sliced_decoder
    d = build()
    disk = attach(d, tmp_path, checkpoints=True)
    s = _stream([1, 2, 3, 50, 5, 6, 7, 8, 9, 10, 11], 1, policy="0")
    s.prefill_slice_layers = cut
    d.begin_admit(s)
    previous = 0
    while d.filling:
        d.prefill_step()
        if s.fill_layer:
            assert all(len(snap.ids) <= previous for _, snap, _ in d.sessions.staged)
        else:
            previous = s.fill_pos
        assert not disk.entries
    ends = [m[2] for m in d.messages if m[0] == mod.FILL_SLICE]
    assert list(dict.fromkeys(ends)) == [3, 4, 8, 11]
    d.finish([s])
    disk.wait_pending()
    assert {len(e.ids) for e in disk.entries.values()} == {3, 4, 8, 11}


@pytest.mark.parametrize("evict", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("dflash", [False, True], ids=["target", "dflash"])
def test_intervening_score_cannot_change_pooled_memory_or_disk_snapshot(setup_decoder, tmp_path, monkeypatch,
                                                                      evict, dflash):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda import quality_score

    _, make = setup_decoder
    def build(path):
        d = make(2, drafter=dflash)
        if dflash:
            d.owner.drafter.ring, d.owner.drafter.window = 64, 8
        attach(d, path, hash_gate=True)
        return d
    d = build(tmp_path / "saved")
    a = _stream([1, 2, 3, 4, 5], 1)
    d.admit(a)
    score = Stream([21, 22, 23], 1)
    score.quality_score_start, score.quality_result = 2, []
    assert not d.can_admit(score)              # post-reply persistence still owns the span
    d.finish([a])
    d.sessions.store.wait_pending()
    assert not d.sessions.staged and d.sessions.store.entries
    keys = set(d.sessions.store.entries)
    def overwrite(e, ids, start):
        e.st.reset()
        e.st.rec.fill_(777)
        e.st.conv.fill_(777)
        for t in e.st.kc + [t for group in e.st.index for t in group]:
            t.fill_(777)
        return [0.5]
    monkeypatch.setattr(quality_score, "score_prefill", overwrite)
    assert d.can_admit(score)
    d.admit(score)
    d.finish([score])
    assert score.quality_result == [0.5] and set(d.sessions.store.entries) == keys
    if evict:
        d.cache.clear()
    b = _stream(a.prompt + [7, 8], 4)
    d.admit(b)
    assert b.cached == 5 and b.session_cache_source == ("disk" if evict else "memory")
    fresh = build(tmp_path / "fresh")
    control = _stream(b.prompt, b.count)
    fresh.admit(control)
    assert b.session_state_sha256 == control.session_state_sha256
    _drain(d)
    _drain(fresh)
    assert b.out == control.out


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("cache_full", [False, True])
def test_blocked_writer_allows_warm_return_and_span_reuse(setup_decoder, tmp_path, monkeypatch, dflash, cache_full):
    from tensorfold.families.glm5_next.cuda import session_cache, session_writer
    from tensorfold.families.glm5_next.cuda.session_state import state_hash

    _, make = setup_decoder
    d = make(2, drafter=dflash)
    if dflash:
        d.owner.drafter.ring, d.owner.drafter.window = 64, 8
    disk = attach(d, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = session_writer.write_tensors
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("warm return performed a disk lookup, full hash, or duplicate snapshot")
    monkeypatch.setattr(session_writer, "write_tensors", delayed)
    monkeypatch.setattr(session_cache, "take_snapshot", forbidden)
    monkeypatch.setattr(session_cache, "state_hash", forbidden)
    a = _stream([1, 2, 3, 4, 5], 1)
    d.admit(a)
    want = state_hash(a.engine, d.cache[-1])
    try:
        d.finish([a])
        assert entered.wait(5)
        assert d.pool.available == d.capacity and not disk.entries
        writer = disk.pending
        assert disk.held() == writer.nbytes
        monkeypatch.setattr(disk, "resume", forbidden)
        if cache_full:
            d.owner.cache_bytes = d.held_bytes()
        b = _stream(a.prompt + [6, 7, 8, 9], 1)
        d.admit(b)
        assert b.cached == len(a.prompt) and b.session_cache_source == "memory"
        assert b.start == a.start
        d.finish([b])
        assert disk.pending is writer and d.sessions.dropped == 1
        assert not d.sessions.sources and not d.sessions.staged
        # The old pool fence may change as the memory prefix extends. It is not
        # part of the older file. All live tensors can now be reused as well.
        for t in a.st.kc + [t for group in a.st.index for t in group]:
            t.fill_(777)
        a.st.rec.fill_(777)
        a.st.conv.fill_(777)
    finally:
        release.set()
        disk.wait_pending()
    assert disk.errors == 0 and len(disk.entries) == 1
    restored = disk.load(a.engine, next(iter(disk.entries.values())), a.drafter)
    assert restored is not None and state_hash(a.engine, restored) == want
    assert d.sessions.held_bytes() == 0


def test_evicted_writer_rows_still_bound_memory_cache(setup_decoder, tmp_path, monkeypatch):
    from tensorfold.families.glm5_next.cuda import session_writer

    _, make = setup_decoder
    d = make(2, drafter=False)
    disk = attach(d, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = session_writer.write_tensors
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(session_writer, "write_tensors", delayed)
    a = _stream([1, 2, 3, 4, 5], 1)
    d.admit(a)
    cap = d.held_bytes()
    try:
        d.finish([a])
        assert entered.wait(5)
        assert d.sessions.extra_rows(d.cache) == 0
        d.cache.clear()
        assert d.sessions.extra_rows(d.cache) > 0
        d.owner.cache_bytes = cap
        b = _stream([11, 12, 13, 14, 15], 1)
        d.admit(b)
        assert not d.cache  # cache + writer cannot allocate two prefixes under a one-prefix cap
        d.finish([b])
        assert d.pool.available == d.capacity
    finally:
        release.set()
        disk.wait_pending()
    assert d.sessions.extra_rows(d.cache) == 0
    c = _stream([21, 22, 23, 24, 25], 1)
    d.admit(c)
    assert len(d.cache) == 1 and d.held_bytes() <= cap
    d.finish([c])


def test_serving_commands_pause_writer_until_every_span_is_released(setup_decoder, tmp_path, monkeypatch):
    _, make = setup_decoder
    d = make(2, drafter=False)
    disk = attach(d, tmp_path)
    disk.write_gate.quiet_s = 3600  # no elapsed-time assumption in this ownership test
    original = d.share
    def share(msg):
        assert not disk.write_gate.idle  # leader pauses before sending any command
        return original(msg)
    d.share = share
    a, b = _stream([1, 2, 3], 1), _stream([4, 5, 6], 1)
    d.admit(a)
    d.admit(b)
    d.finish([a])
    assert not disk.write_gate.idle and disk.pending is not None
    d.finish([])
    assert not disk.write_gate.idle
    d.finish([b])
    assert disk.write_gate.idle and d.pool.available == d.capacity
    # A declined admission/control transaction must not strand a writer paused.
    d._send([99])
    assert not disk.write_gate.idle
    d.finish([])
    assert disk.write_gate.idle


def test_follower_pauses_writer_on_doorbell_before_collective(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    disk = attach(d, tmp_path)
    d.rank = d.owner.rank = 1
    events = []
    def wait():
        assert disk.write_gate.idle
        events.append("wake")
    def share(_):
        assert not disk.write_gate.idle
        events.append("collective")
        return [99]
    d.owner.follower_doorbell = SimpleNamespace(wait=wait)
    d.share = share
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        d.follow()
    assert events == ["wake", "collective"]
