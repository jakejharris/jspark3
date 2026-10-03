"""Reply reservation ownership, pressure liveness, lossless live IO, and adversarial controls."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda.pool import TokenPool
from tensorfold.families.glm5_next.cuda.reply_reservations import (
    CREDIT, GROW, PARK, PREPARE, RESUME, MESSAGE, ReplyConfig, ReplyReservations,
)
from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig
from tensorfold.families.glm5_next.cuda.session_disk import SessionStore
from test_glm_batched_host import setup_decoder, allocations, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401


def enable(d, path, *, block=64, capacity=None, budget=64 * 2**20):
    d.pool = TokenPool(capacity or d.capacity, guard=d.guard)
    d.owner.world = 3
    d.owner._share = lambda v: v
    d.owner._gather_ints = lambda v: [v] * 3
    disk = SessionStore(path, budget, "test-weights", min_free=0)
    d.sessions = SessionCache(d.owner, SessionConfig(disk=True), disk)
    d.owner.cache_entries = 0
    d.growth = ReplyReservations(d, ReplyConfig(True, block))
    return d.growth


def drain(d, cap=2000):
    for _ in range(cap):
        if not d.live():
            return
        d.finish(d.round())
    pytest.fail("finite replies did not finish")


def test_in_place_growth_is_atomic_and_detects_stale_neighbors():
    pool = TokenPool(128)
    a, b = pool.allocate("a", 12), pool.allocate("b", 24)
    assert pool.growth("a", 16) is None
    before = dict(pool.spans)
    with pytest.raises(RuntimeError, match="overlaps"):
        pool.grow("a", a, 16)
    assert pool.spans == before
    pool.release("b")
    assert pool.grow("a", a, 64) == (0, 72)
    with pytest.raises(RuntimeError, match="stale"):
        pool.grow("a", a, 68)
    assert pool.spans == {"a": (0, 72)}


@pytest.mark.parametrize("block", [64, 2048])
def test_growth_retains_original_count_and_matches_uninterrupted_decode(setup_decoder, tmp_path, block):
    _, make = setup_decoder
    count = block + 25
    d = make(2, limit=count + 32, drafter=False)
    enable(d, tmp_path, block=block)
    s = _stream([1, 2, 3], count, draft=False)
    emitted = []
    s.emit = lambda new: emitted.extend(new)
    d.admit(s)
    assert s.count == count and s.reserved_tokens == 3 + block
    ptr = s.st.kc[0].data_ptr()
    drain(d, count + 5)
    assert s.count == len(s.out) == len(emitted) == count and emitted == s.out
    assert s.reply_grows and not s.reply_parks and s.st.kc[0].data_ptr() == ptr
    ref = make(2, limit=count + 32, drafter=False)
    control = _stream(s.prompt, count, draft=False)
    ref.admit(control)
    drain(ref, count + 5)
    assert s.out == control.out
    assert not d.growth.store.store.live_credits


def test_newest_parking_then_nonzero_oldest_relocation_preserves_response(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(4, limit=220, drafter=False)
    grow = enable(d, tmp_path, capacity=228)
    prefix = _stream([9] * 8, 1, draft=False)
    older = _stream([2] * 8, 200, draft=False)
    younger = _stream([3] * 8, 180, draft=False)
    emitted = {id(s): [] for s in (older, younger)}
    for s in (older, younger):
        s.emit = lambda new, s=s: emitted[id(s)].extend(new)
    for s in (prefix, older, younger):
        d.admit(s)
    assert older.start > 0
    d.finish([prefix])
    # Drive until pressure. Newer releases first; the nonzero oldest cannot fit
    # its whole cap until it too parks and returns at zero.
    for _ in range(80):
        done = d.round()
        d.finish(done)
        if older.reply_parks:
            break
    parks = [m[3] for m in d.messages if m[:1] == [MESSAGE] and m[2] == PARK]
    assert parks[:2] == [younger.sid, older.sid]
    assert older.start == 0 and older.reserved_tokens == older.request_limit
    assert older.engine is not d.owner.e and older.engine.graphs is None
    assert younger.sid in grow.parked and younger.st is None
    offered = _stream([4], 10, draft=False)
    assert not d.can_admit(offered)
    drain(d)
    for s in (older, younger):
        ref = make(2, limit=220, drafter=False)
        control = _stream(s.prompt, s.count, draft=False)
        ref.admit(control)
        drain(ref)
        assert s.out == control.out == emitted[id(s)]
        assert len(s.out) == s.count and s.reply_park_seconds >= 0
    assert not grow.parked and not grow.store.records and d.pool.available == 228
    assert d.can_admit(offered)


def test_park_prepare_failure_keeps_owned_span_and_open_response(setup_decoder, tmp_path, monkeypatch):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    before, ptr, out = dict(d.pool.spans), s.st.kc[0].data_ptr(), list(s.out)
    def fail(*a, **kw):
        raise OSError("injected disk full")
    monkeypatch.setattr("tensorfold.families.glm5_next.cuda.live_store.write_tensors", fail)
    assert not grow.park(s)
    assert d.pool.spans == before and s.st.kc[0].data_ptr() == ptr and s.out == out
    assert not s.done and not grow.parked and not grow.store.records and not d.broken
    assert not list(grow.store.dir.iterdir())


@pytest.mark.parametrize("fault", ["byte", "missing", "metadata"])
def test_bad_committed_live_record_remains_pinned_without_prefix_replay(setup_decoder, tmp_path, fault):
    from tensorfold.families.glm5_next.cuda.disk_io import TensorReader, write_tensors
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    assert grow.park(s)
    path, _ = grow.store.records[s.sid]
    if fault == "missing":
        path.unlink()
    elif fault == "byte":
        with TensorReader(path) as r:
            at = r.base + r.layout["rec"]["offset"]
        with path.open("r+b") as f:
            f.seek(at)
            byte = f.read(1)[0]
            f.seek(at)
            f.write(bytes([byte ^ 1]))
    else:
        with TensorReader(path) as r:
            meta, parts = dict(r.meta), [(name, r.get(name)) for name in r.layout]
        meta["mtp_drafted"] += 1
        write_tensors(path, meta, parts)
    out = list(s.out)
    assert not grow.resume(s)
    assert s.sid in grow.parked and s.sid in grow.store.records
    assert s.sid in grow.store.store.live_credits and s.sid not in d.pool.spans
    assert not s.done and s.out == out and s.engine is None


def test_rank_refusal_prevents_growth_and_epoch_replay_cannot_publish(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    before = dict(d.pool.spans)
    votes = 0
    def reject(v):
        nonlocal votes
        votes += 1
        return [v, [0] if votes == 3 else v, v]
    d.owner._gather_ints = reject
    with pytest.raises(RuntimeError, match="ownership"):
        grow.grow(s, s.request_limit)
    assert d.pool.spans == before and s.reserved_tokens == 67
    d.owner._gather_ints = lambda v: [v] * 3
    with pytest.raises(RuntimeError, match="epoch"):
        grow.follow([MESSAGE, grow.serial, GROW, s.sid, s.start, s.span, 212, 203])
    assert d.pool.spans == before


def test_credit_shortage_defers_before_first_emission(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path, budget=1024)
    s = _stream([1, 2, 3], 200, draft=False)
    assert d.admit(s) is False
    assert not s.out and not d.streams and not d.pool.spans and not grow.store.store.live_credits
    assert not any(m[0] == 1 for m in d.messages)


def test_parked_cancel_releases_credit_without_restoring_or_reemitting(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    assert grow.park(s)
    s.cancelled = lambda: True
    out = list(s.out)
    d.finish(d.round())
    assert s.out == out and s.done and not grow.store.records
    assert not grow.store.store.live_credits and not d.streams


def test_configuration_requires_backing_and_does_not_change_default(monkeypatch):
    for name in ("TF_GLM_GROW_REPLIES", "TF_GLM_REPLY_BLOCK_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    assert not ReplyConfig.from_env(2, SessionConfig()).enabled
    monkeypatch.setenv("TF_GLM_GROW_REPLIES", "1")
    with pytest.raises(ValueError, match="SESSION_DISK"):
        ReplyConfig.from_env(2, SessionConfig())
    monkeypatch.setenv("TF_GLM_REPLY_BLOCK_TOKENS", "65")
    with pytest.raises(ValueError, match="multiple"):
        ReplyConfig.from_env(2, SessionConfig(disk=True))


@pytest.mark.parametrize("cut", [0, 1, 3, 5])
def test_prefill_park_closes_generator_and_resumes_completed_chunk(sliced_decoder, tmp_path, cut):
    mod, build = sliced_decoder
    d = build()
    d.owner.drafter = None
    grow = enable(d, tmp_path)
    s = _stream(list(range(31)), 100, draft=False, policy="0")
    s.prefill_slice_layers = cut
    d.begin_admit(s)
    d.prefill_step()
    if cut:
        assert d.fill_owner == s.sid
        with pytest.raises(RuntimeError, match="incomplete"):
            # Direct malformed transaction is fatal; test the validation on a
            # standalone decode instance, then clear only this injected poison.
            grow.command(PREPARE, s.sid, s.start, s.span)
        d.broken = None
    assert grow.park(s)  # complete_chunk finishes the mixed-layer work first
    assert s.fill_pos == 8 and s.sid not in d.filling and d.fill_owner is None
    assert d.owner.e.pbuf.overlay is None and s.engine is None
    assert grow.resume(s)
    assert s.sid in d.filling and s.st.pos == 8
    drain_steps(d)
    ref = build()
    ref.owner.drafter = None
    control = _stream(s.prompt, s.count, draft=False, policy="0")
    control.prefill_slice_layers = cut
    ref.begin_admit(control)
    drain_steps(ref)
    assert s.out == control.out and len(s.out) == 100
    assert torch.equal(s.st.conv, control.st.conv)


def test_three_rank_protocol_parks_and_restores_identical_named_owners(setup_decoder, tmp_path):
    import queue
    import threading

    _, make = setup_decoder
    ranks = [make(4, limit=220, drafter=False) for _ in range(3)]
    for rank, d in enumerate(ranks):
        enable(d, tmp_path / str(rank), capacity=228)
        d.rank = d.owner.rank = d.growth.store.rank = rank
    barrier, votes = threading.Barrier(3, timeout=10), [None] * 3
    queues = [queue.Queue(), queue.Queue()]
    def gather(rank, values):
        votes[rank] = list(values)
        barrier.wait()
        result = [list(v) for v in votes]
        barrier.wait()
        return result
    def broadcast(msg):
        for q in queues:
            q.put(list(msg))
    ranks[0].share = broadcast
    seen, errors, threads = {}, [], []
    for rank, d in enumerate(ranks):
        d.owner._gather_ints = lambda v, rank=rank: gather(rank, v)
        if not rank:
            continue
        d.share = lambda _, q=queues[rank - 1]: q.get(timeout=10)
        original = d._finish
        def finish(ids, d=d, original=original):
            for sid in ids:
                seen[d.rank, sid] = list(d.streams[sid].out)
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
    source = ranks[0]
    streams = [_stream([9] * 8, 1, draft=False), _stream([2] * 8, 200, draft=False),
               _stream([3] * 8, 180, draft=False)]
    try:
        for s in streams:
            source.admit(s)
        source.finish([streams[0]])
        drain(source)
    finally:
        broadcast([99])
    for thread in threads:
        thread.join(12)
        assert not thread.is_alive()
    assert not errors
    assert all(seen[rank, s.sid] == s.out for rank in (1, 2) for s in streams)
    assert all(not d.pool.spans and not d.growth.store.store.live_credits for d in ranks)


@pytest.mark.parametrize("fair", [False, True])
def test_scheduler_retains_unstarted_response_on_credit_retry(fair):
    from test_glm_fair_schedule import scheduler, enqueue
    sch = scheduler(fair)
    stream, box = enqueue(sch, [1])
    attempts = []
    original = sch.decoder.admit
    def begin(s):
        attempts.append(s)
        if len(attempts) == 1:
            return False
        return original(s)
    sch.decoder.begin_admit = begin
    sch._admit()
    assert not stream.out and box.empty() and not sch.boxes
    first, sch.pending = sch.pending, None
    sch._admit(first)
    assert attempts == [stream, stream] and stream.out == [1]
    assert sch.boxes[id(stream)] is box


def test_prefill_completion_is_returned_once_before_done_message(sliced_decoder, tmp_path):
    _, build = sliced_decoder
    d = build()
    d.owner.drafter = None
    enable(d, tmp_path)
    s = _stream([1, 2, 3], 1, draft=False)
    d.begin_admit(s)
    done = d.prefill_step() + d.round()
    assert done == [s]
    d.finish(done)
    assert d.messages[-1] == [3, 1, s.sid]


@pytest.mark.parametrize("operation", [PARK, RESUME])
def test_repeated_commit_cannot_double_release_or_reemit(setup_decoder, tmp_path, operation):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    assert grow.park(s)
    if operation == RESUME:
        assert grow.resume(s)
    before, out = dict(d.pool.spans), list(s.out)
    with pytest.raises(RuntimeError, match="duplicate"):
        grow.command(operation, s.sid)
    assert d.pool.spans == before and s.out == out


def test_backed_credit_is_checked_before_park_and_all_rank_commit(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    before = dict(d.pool.spans)
    grow.store.store.live_credits[s.sid] = 1  # negative: a future capacity promise without backing
    assert not grow.park(s)
    assert d.pool.spans == before and s.st is not None and not s.done
    assert not any(m[0] == MESSAGE and m[2] == PARK for m in d.messages)


def test_duplicate_credit_cannot_delete_a_live_owners_backing(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 200, draft=False)
    d.admit(s)
    before = dict(grow.store.store.live_credits)
    with pytest.raises(RuntimeError, match="duplicate owner"):
        grow.command(CREDIT, s.sid, s.request_limit)
    assert grow.store.store.live_credits == before and s.sid in d.pool.spans


def test_infrastructure_failure_does_not_emit_a_normal_short_reply(setup_decoder, tmp_path, monkeypatch):
    _, make = setup_decoder
    d = make(2, limit=180, drafter=False)
    grow = enable(d, tmp_path, capacity=188)
    streams = [_stream([1] * 8, 160, draft=False), _stream([2] * 8, 150, draft=False)]
    for s in streams:
        d.admit(s)
    def fail(*a, **kw):
        raise OSError("injected persistent storage failure")
    monkeypatch.setattr(grow.store, "prepare", fail)
    before = dict(d.pool.spans)
    for _ in range(80):
        d.round()
        grow.retry_at = 0  # deterministic fault exercise, no wall-time wait
        if grow.error is not None and all(not grow.safe(s) for s in streams):
            break
    assert not any(s.done for s in streams) and not d.broken
    assert all(len(s.out) < s.count for s in streams) and d.pool.spans == before
    assert grow.error is not None and not d.can_admit(_stream([3], 10, draft=False))


def test_missing_image_fingerprint_refuses_before_any_credit_or_output(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1, 2, 3], 100, draft=False)
    s.images = [SimpleNamespace(digest=None)]
    with pytest.raises(ValueError, match="fingerprint"):
        d.admit(s)
    assert not s.out and not grow.store.store.live_credits and not d.messages


def test_c16_offered_work_drains_parked_fifo_before_new_admission(setup_decoder, tmp_path):
    import threading
    from tensorfold.cuda.scheduler import Scheduler

    _, make = setup_decoder
    d = make(8, limit=700, drafter=False)
    grow = enable(d, tmp_path, capacity=708)
    d.fair_schedule = True
    d.begin_admit = d.admit  # this gate isolates response/admission; real sliced prefill is gated above
    entered, release = threading.Event(), threading.Event()
    original = d.round
    def step():
        entered.set()
        assert release.wait(10)
        return original()
    d.round = step
    sch = Scheduler(d, max_streams=8, admit_per_round=1)
    outcomes, received = {}, {i: [] for i in range(16)}
    def request(i):
        try:
            outcomes[i] = sch.submit([i + 1] * 8, 200, None, False,
                                    lambda new: received[i].extend(new), stop_eos=False)
        except Exception as exc:
            outcomes[i] = exc
    threads = [threading.Thread(target=request, args=(i,), daemon=True) for i in range(16)]
    for thread in threads:
        thread.start()
    assert entered.wait(10)
    release.set()
    for thread in threads:
        thread.join(20)
        assert not thread.is_alive(), "a finite offered request starved"
    assert len(outcomes) == 16 and all(isinstance(result, dict) for result in outcomes.values())
    assert all(len(tokens) == 200 for tokens in received.values())
    assert sum(result["reply_parks"] for result in outcomes.values()) > 0
    parked, restored = set(), []
    for message in d.messages:
        if message[0] == 1:
            assert not parked, "new admission bypassed a parked response"
        if message[0] != MESSAGE:
            continue
        op, sid = message[2:4]
        if op == PARK:
            parked.add(sid)
        elif op == RESUME:
            assert sid == min(parked), "parked requests did not resume in admission order"
            parked.remove(sid)
            restored.append(sid)
    assert restored and not parked and not grow.store.store.live_credits
