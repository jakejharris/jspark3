"""TCPStore wake-up races and three-rank command/state parity, without CUDA."""

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import queue
import threading
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
from torch.distributed import DistNetworkError, DistStoreError, TCPStore

from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from tensorfold.families.glm5_next.cuda.idle_bell import FollowerDoorbell, configured
from tensorfold.engine.exact_sampling import Sampling
from test_glm_batched_host import setup_decoder, allocations, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from test_glm_reply_reservations import enable as enable_reservations, drain

pytestmark = pytest.mark.torch


@pytest.fixture
def bells():
    server = TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=5), wait_for_workers=False)
    clients = [TCPStore("127.0.0.1", server.port, None, False, timeout=timedelta(seconds=5)) for _ in range(2)]
    return [FollowerDoorbell(store, rank, 3) for rank, store in enumerate([server, *clients])]


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_default_off_and_explicit_switch(monkeypatch, value, expected):
    monkeypatch.delenv("TF_GLM_FOLLOWER_DOORBELL", raising=False)
    if value is not None:
        monkeypatch.setenv("TF_GLM_FOLLOWER_DOORBELL", value)
    assert configured() is expected


def test_invalid_switch_fails(monkeypatch):
    monkeypatch.setenv("TF_GLM_FOLLOWER_DOORBELL", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        configured()


def test_unrung_follower_blocks_and_fast_rank_cannot_steal_slow_rank_wake(bells):
    entered, finished = threading.Event(), threading.Event()
    def follower():
        entered.set()
        bells[1].wait()
        finished.set()
    thread = threading.Thread(target=follower, daemon=True)
    thread.start()
    assert entered.wait(2)
    try:
        assert not finished.wait(.05)  # negative control: no doorbell, no header/compute
    finally:
        bells[0].ring()
        thread.join(3)
    assert finished.is_set()
    assert not bells[0].store.check(["tf_glm_idle_1_1"])
    assert bells[0].store.check(["tf_glm_idle_1_2"])
    bells[2].wait()  # starts after rank 1 has already consumed/deleted its own key
    assert not bells[0].store.check(["tf_glm_idle_1_2"])
    for _ in range(4):
        bells[0].ring()  # wake-ups queued before either follower waits must not be lost
    for rank in (2, 1):
        for _ in range(4):
            bells[rank].wait()
    assert [b.epoch for b in bells] == [5, 5, 5]
    assert all(not bells[0].store.check([f"tf_glm_idle_{n}_{r}"])
               for n in range(1, 6) for r in (1, 2))


@pytest.mark.parametrize("error", [DistStoreError("wait timeout after an idle hour"),
                                        DistStoreError("invalid store state"),
                                        DistNetworkError("connection reset"),
                                        DistNetworkError("socket timeout")])
def test_only_idle_store_timeouts_retry(error):
    events = []
    def wait(keys, timeout):
        events.append((keys, timeout))
        if len(events) == 1:
            raise error
    deleted = []
    bell = FollowerDoorbell(NS(wait=wait, delete_key=deleted.append), 2, 3)
    if isinstance(error, DistStoreError) and "timeout" in str(error):
        bell.wait()
        assert len(events) == 2 and bell.epoch == 1 and deleted == ["tf_glm_idle_1_2"]
    else:
        with pytest.raises(type(error)):
            bell.wait()
        assert len(events) == 1 and bell.epoch == 0 and not deleted


def test_rank_zero_store_shutdown_releases_idle_follower():
    server = TCPStore("127.0.0.1", 0, None, True, wait_for_workers=False)
    client = TCPStore("127.0.0.1", server.port, None, False)
    entered, errors = threading.Event(), []
    def follower():
        entered.set()
        try:
            FollowerDoorbell(client, 1, 2).wait()
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=follower, daemon=True)
    thread.start()
    assert entered.wait(2)
    del server
    thread.join(3)
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], DistNetworkError)


def test_no_bell_preserves_serial_stand_in_communicator():
    engine = object.__new__(GlmEngine)
    engine.comm = NS()  # flag off / communicator without a store
    engine._ring()
    engine._await_bell()


@pytest.mark.parametrize("scoring", [False, True])
def test_serial_headers_wake_once_and_payload_does_not(scoring, monkeypatch):
    from tensorfold.families.glm5_next.cuda import quality_score

    engine = object.__new__(GlmEngine)
    events = []
    engine.follower_doorbell = NS(ring=lambda: events.append("wake"))
    engine._share = lambda frame: events.append(frame)
    engine.concurrent = False
    engine.quality_score_enabled = True
    engine.limit, engine.serial_only, engine.policy = 64, True, "0"
    engine.request, engine.e = NS(), object()
    engine._image_starts = lambda *a: None
    engine._effective = lambda code: code
    engine._take_over = lambda *a: None
    engine._run = lambda *a, **k: {}
    monkeypatch.setattr(quality_score, "score_prefill", lambda *a: [0.5])
    if scoring:
        engine.score([4, 5], 1)
    else:
        engine.generate([4, 5], 2, Sampling(23, .7, 16, .9), lambda _: None, draft=False)
    assert events[0] == "wake" and events[-1] == [4, 5] and len(events) == 3


@contextmanager
def running(ranks, bells):
    """Real socket bell; CPU collective barriers preserve the production command ordering."""
    barrier, votes = threading.Barrier(3, timeout=5), [None] * 3
    inboxes = [queue.Queue(), queue.Queue()]
    errors, threads, observed = [], [], {}
    frames, receives = [], [0, 0, 0]
    def gather(rank, values):
        votes[rank] = list(values)
        barrier.wait()
        result = [list(v) for v in votes]
        barrier.wait()
        return result
    def share(rank, values):
        receives[rank] += 1
        if rank == 0:
            frames.append(list(values))
            for q in inboxes:
                q.put(list(values))
        else:
            values = inboxes[rank - 1].get(timeout=5)
        barrier.wait()
        return values
    for rank, d in enumerate(ranks):
        d.rank = d.owner.rank = rank
        d.owner.follower_doorbell = bells[rank] if bells else None
        d.owner._gather_ints = lambda values, rank=rank: gather(rank, values)
        d.share = lambda values, rank=rank: share(rank, values)
        original = d._finish
        def finish(sids, d=d, original=original):
            for sid in sids:
                s = d.streams[sid]
                observed[d.rank, sid] = (list(s.out), s.st.conv.clone(), s.st.rec.clone())
            original(sids)
        d._finish = finish
        if rank:
            def follow(d=d):
                try:
                    d.follow()
                except RuntimeError as exc:
                    if "unknown GLM worker message 99" not in str(exc):
                        errors.append(exc)
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=follow, daemon=True)
            thread.start()
            threads.append(thread)
    try:
        yield frames, receives, observed
    finally:
        ranks[0]._send([99])
        for thread in threads:
            thread.join(6)
        assert not any(t.is_alive() for t in threads) and not errors


@pytest.mark.parametrize("reply", [False, True])
def test_batched_after_idle_matches_flag_off_tokens_state_and_commands(sliced_decoder, bells, reply):
    from test_glm_reply_prefill import background

    _, build = sliced_decoder
    results = []
    for enabled in (False, True):
        ranks = [build() for _ in range(3)]
        if reply:
            for d in ranks:
                d.reply_prefill, d.fair_schedule, d.reply_prefill_rows = True, True, 3
        with running(ranks, bells if enabled else None) as (frames, receives, observed):
            source = ranks[0]
            prompt = [2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8, 7, 2, 6, 1, 5]
            for turn in range(2):
                if enabled:
                    before = list(receives)
                    threading.Event().wait(.05)
                    assert receives == before  # neither follower entered NCCL while idle
                s = _stream(prompt[:5] if turn == 0 else prompt, 12,
                            sampling=Sampling(301, .7, 16, .93))
                source.begin_admit(s)
                epoch = bells[0].epoch
                drain_steps(source)
                if enabled:
                    assert bells[0].epoch == epoch  # active FILL/ROUND/DONE never touch store
                if reply and turn == 0:
                    job = background(prompt[:13], 5)
                    source.begin_admit(job)
                    source.prefill_step()
                    job.cancelled = lambda: True
                    source.finish(source.prefill_step())
                    job = background(prompt[:13], 5)
                    source.begin_admit(job)
                    drain_steps(source)
            # Synchronize the followers through another command before inspecting finish state.
        for sid in range(source.next_id):
            expected = observed[0, sid]
            for rank in (1, 2):
                got = observed[rank, sid]
                assert got[0] == expected[0]
                assert all(torch.equal(a, b) for a, b in zip(got[1:], expected[1:]))
        results.append((frames, observed))
    assert results[0][0] == results[1][0]
    for key, before in results[0][1].items():
        after = results[1][1][key]
        assert before[0] == after[0]
        assert all(torch.equal(a, b) for a, b in zip(before[1:], after[1:]))
    digest = lambda result: hashlib.sha256(json.dumps([result[1][k][0] for k in sorted(result[1])]).encode()).hexdigest()
    assert digest(results[0]) == digest(results[1])


def test_idle_reservation_credit_then_admit_and_pressure_wake_both_followers(setup_decoder, tmp_path, bells):
    _, make = setup_decoder
    ranks = [make(4, limit=220, drafter=False) for _ in range(3)]
    for rank, d in enumerate(ranks):
        enable_reservations(d, tmp_path / str(rank), capacity=228)
        d.growth.store.rank = rank
    with running(ranks, bells) as (frames, receives, observed):
        source = ranks[0]
        for turn in range(2):
            streams = [_stream([9] * 8, 1, draft=False), _stream([2] * 8, 200, draft=False),
                       _stream([3] * 8, 180, draft=False)]
            for s in streams:
                source.admit(s)
            assert bells[0].epoch == (turn + 1) * 2  # idle CREDIT and ADMIT; no prompt-payload bell
            source.finish([streams[0]])
            drain(source)
    assert all(not d.pool.spans and not d.growth.store.store.live_credits for d in ranks)
    for sid in range(source.next_id):
        assert observed[0, sid][0] == observed[1, sid][0] == observed[2, sid][0]
