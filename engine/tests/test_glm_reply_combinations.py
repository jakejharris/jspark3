"""Reply reservation ownership with integrated tiled attention, co-prefill and copy drafting, using their production controllers."""

import torch
import pytest

from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from test_glm_a2_prefill import express_decoder  # noqa: F401
from test_glm_cofill import cofill_decoder  # noqa: F401
from test_glm_copy_batched import copy_prices, edit_prompt  # noqa: F401
from test_glm_reply_reservations import enable, drain
from test_glm_copy_batched import joint_copy  # noqa: F401
from test_glm_copy_wide import wide_copy, long_edit  # noqa: F401
from test_glm_cofill_big import big_decoder  # noqa: F401


def test_park_completes_both_lanes_and_restores_express_buffer_binding(express_decoder, tmp_path):
    _, build = express_decoder
    d = build()
    d.owner.drafter = None
    grow = enable(d, tmp_path)
    d.owner.checkpoint_after = lambda pos: (pos // 4 + 1) * 4
    d.sessions.config.checkpoints = True
    d.sessions.role_ids = frozenset((50,))
    long = _stream([1] * 17, 100, policy="0")
    short = _stream([3, 50, 5], 100, policy="0")
    short.prefill_slice_layers = 1  # Park while both lanes own a partially executed chunk.
    d.begin_admit(long)
    d.prefill_step()
    d.begin_admit(short)
    d.prefill_step()
    assert d.fill_owner == long.sid and d.express_owner == short.sid
    assert grow.park(short)
    assert d.fill_owner is d.express_owner is None and long.fill_pos == 4 and short.fill_pos == 1
    assert grow.resume(short)
    assert short.engine.pbuf is d.express_buf and long.engine.pbuf is d.owner.e.pbuf
    assert short.engine.pbuf is not long.engine.pbuf
    drain_steps(d)
    for s in (long, short):
        ref = build()
        ref.owner.drafter = None
        control = _stream(s.prompt, s.count, draft=False)
        ref.begin_admit(control)
        drain_steps(ref)
        assert s.out == control.out


def test_cofilled_requests_grow_and_park_with_single_response_completion(cofill_decoder, tmp_path):
    mod, build, forwards = cofill_decoder
    d = build()
    d.owner.drafter, d.owner.pool_limit = None, 240
    grow = enable(d, tmp_path, capacity=248)
    streams = [_stream([token] * 17, 200, draft=False) for token in (2, 3)]
    for s in streams:
        d.begin_admit(s)
    before = len(forwards)
    assert not d.prefill_step()
    assert len(forwards) == before + 1 and not d.filling
    assert all(s.cofill_stats["streams"] == 2 for s in streams)
    drain(d)
    assert streams[1].reply_parks and not grow.parked
    for s in streams:
        ref = build()
        ref.owner.drafter = None
        control = _stream(s.prompt, s.count, draft=False)
        ref.admit(control)
        drain(ref)
        assert s.out == control.out and len(s.out) == s.count


def test_copy_lookup_survives_parking_and_prices_ignore_parked_peers(setup_decoder, copy_prices, tmp_path):
    _, make = setup_decoder
    d = make(4, limit=320)
    parent = d.owner.drafter
    parent.ring, parent.window, parent.cap = 512, 448, 4 * 512
    parent.kc = [torch.zeros(2, parent.cap, 2)]
    parent.vc = [torch.zeros(2, parent.cap, 2)]
    grow = enable(d, tmp_path / "live", capacity=328)
    prompt, _ = edit_prompt(d.w)
    streams = [_stream(prompt, 200, policy="f7") for _ in range(2)]
    for s in streams:
        d.admit(s)
    assert grow.metrics()["promised_tokens"] > d.owner.pool_limit
    drain(d)
    ref = make(2, limit=320, drafter=False)
    control = _stream(prompt, 200, draft=False)
    ref.admit(control)
    drain(ref)
    assert all(s.out == control.out and s.copies.lookup.ids == s.prompt + s.out for s in streams)
    assert streams[1].reply_parks and all(s.copy_stats["drafter_skips"] for s in streams)


def test_reservation_precedes_pricing_of_32_row_copy_candidates(wide_copy, tmp_path):
    _, make = wide_copy
    d = make(32, limit=700, pool_limit=720)
    parent = d.owner.drafter
    parent.ring, parent.window, parent.cap = 512, 448, 2 * 512
    parent.kc = [torch.zeros(2, parent.cap, 2)]
    parent.vc = [torch.zeros(2, parent.cap, 2)]
    grow = enable(d, tmp_path, capacity=752)
    prompt, expected = long_edit(d.w)
    streams = [_stream(prompt, len(expected), policy="fp7") for _ in range(2)]
    for s in streams:
        d.admit(s)
    drain(d)
    assert all(s.out == expected and 31 in s.depths for s in streams)
    assert streams[1].reply_parks and not grow.parked


def test_model_only_bound_negative_control_misses_wide_candidate(setup_decoder, tmp_path):
    _, make = setup_decoder
    d = make(2, drafter=False)
    grow = enable(d, tmp_path)
    s = _stream([1] * 20, 200, draft=False)
    d.admit(s)
    s.st.set_pos(s.reserved_tokens - 20)
    s.drafts, s.copy_drafts = [1] * 7, [1] * 31
    assert s.st.pos + 1 + len(s.drafts) + 8 <= s.reserved_tokens  # old check would allow this
    assert s.st.pos + 1 + len(s.copy_drafts) > s.reserved_tokens  # ROUND2 can cross it
    assert not grow.safe(s)


@pytest.mark.parametrize("suffix", [3, 7], ids=["express", "main"])
@pytest.mark.parametrize("cancel", [False, True], ids=["completed", "preempted"])
def test_reply_prefill_and_growth_share_protocol_without_leaking_owners(express_decoder, tmp_path, suffix, cancel):
    from test_glm_reply_prefill import background
    from tensorfold.families.glm5_next.cuda.reply_reservations import MESSAGE

    mod, build = express_decoder

    def configured(path):
        d = build()
        d.owner.drafter = None
        grow = enable(d, path)
        d.owner.cache_entries = 8
        d.reply_prefill, d.fair_schedule, d.reply_prefill_rows = True, True, 2
        return d, grow

    d, grow = configured(tmp_path / "leader")
    prompt = [2, 3, 4, 8, 9, 2, 5, 6, 7, 3, 8, 7, 1]
    warm = _stream(prompt[:5], 1, policy="0")
    d.begin_admit(warm)
    drain_steps(d)
    job = background(prompt[:5 + suffix], 5)
    job.policy = "0"
    d.begin_admit(job)
    assert job.sid in grow.store.store.live_credits
    assert job.express == (suffix == 3)
    if cancel:
        d.prefill_step()
        job.cancelled = lambda: True
        d.finish(d.prefill_step())
    else:
        drain_steps(d)
    assert not job.out and not d.live() and d.pool.available == d.capacity
    assert not grow.store.store.live_credits and not grow.parked
    actual = _stream(prompt, 90, policy="0")
    d.begin_admit(actual)
    assert actual.reply_prefill_hit_tokens == (0 if cancel else suffix)
    drain_steps(d)
    assert actual.reply_grows and len(actual.out) == 90
    reference = build()
    reference.owner.drafter = None
    control = _stream(prompt, 90, policy="0")
    reference.begin_admit(control)
    drain_steps(reference)
    assert actual.out == control.out
    assert MESSAGE != mod.ADMIT_REPLY
    assert any(m[0] == MESSAGE for m in d.messages)
    assert any(m[0] == mod.ADMIT_REPLY for m in d.messages)
    # Replay the real dispatcher on both followers: an opcode collision routes
    # ADMIT_REPLY into growth validation (or vice versa) and fails this gate.
    for rank in (1, 2):
        follower, follower_grow = configured(tmp_path / f"rank{rank}")
        follower.rank = follower.owner.rank = rank
        messages = iter(d.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert not follower.live() and not follower_grow.store.store.live_credits
        assert follower.pool.available == follower.capacity


@pytest.mark.parametrize("started", [False, True])
def test_large_cofill_keeps_its_stepper_and_prefix_when_parked(big_decoder, tmp_path, started):
    from tensorfold.families.glm5_next.cuda.cofill_big import Prefill

    _, make, _ = big_decoder
    d = make(4, rows=16)
    d.owner.drafter, d.owner.pool_limit = None, 240
    grow = enable(d, tmp_path, capacity=248)
    streams = [_stream([token] * 37, 200, policy="0") for token in (3, 7)]
    for s in streams:
        d.begin_admit(s)
        assert isinstance(d.filling[s.sid], Prefill)
    if started:
        d.prefill_step()
    for s in streams:
        assert grow.park(s)
        assert 0 < s.fill_pos < len(s.prompt)
    pos = streams[0].fill_pos
    assert grow.resume(streams[0])
    assert isinstance(d.filling[streams[0].sid], Prefill)
    assert d.filling[streams[0].sid].started and streams[0].st.pos == pos
    for _ in range(1000):
        d.finish(d.prefill_step())
        d.finish(d.round())
        if not d.live():
            break
    assert not d.live() and not grow.store.store.live_credits
    for s in streams:
        ref = make(2, rows=16)
        ref.owner.drafter = None
        control = _stream(s.prompt, s.count, policy="0")
        ref.admit(control)
        drain(ref)
        assert s.out == control.out and s.cofill_stats["prompt_rows"] == len(s.prompt)
