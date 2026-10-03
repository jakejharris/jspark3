"""Copy drafting: real batched control/accept/commit with CPU target/drafter fakes."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.copy_drafts import CopyOffer, CopyPrices
from tensorfold.families.glm5_next.cuda.copy_control import COPY_PROPOSE, COPY_CHECKED
from test_glm_batched_host import setup_decoder, _drain, _logits, _stream, _Drafter  # noqa: F401
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_copy_drafts import price_data, replay_prior
from test_glm_cofill import cofill_decoder  # noqa: F401


@pytest.fixture
def copy_prices(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GLM_COPY_PRIOR", raising=False)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(price_data(loads=range(1, 9))))
    monkeypatch.setenv("TF_GLM_COPY_PRICES", str(path))
    for key, value in (("WEIGHT", "weights"), ("DRAFTER", "drafter"), ("BUILD", "build")):
        monkeypatch.setenv(f"TF_GLM_{key}_ID", f"test-{value}")
    monkeypatch.setenv("TF_GLM_COPY_DRAFTS", "1")
    monkeypatch.setenv("TF_GLM_COPY_SKIP", "1")
    # The legacy one-stream settings must not widen this path.
    monkeypatch.setenv("TF_GLM_LOOKUP_DRAFTS", "31")
    return path


def edit_prompt(w, sampling=None, count=40):
    from tensorfold.families.glm5_next.cuda.decode import sample_rows

    tokens, token = [], 7
    for pos in range(64, 64 + count):
        token = sample_rows(w, _logits([token], pos - 1), [pos], sampling)[0]
        tokens.append(token)
    # Exact model continuation in earlier text, joined to the final seven prompt
    # tokens by the first sampled token. The fake depends on last token/position.
    source = list(range(1, 8)) + tokens
    return source + [50] * (56 - len(source)) + list(range(8)), tokens


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("rows", range(1, 9))
def test_every_width_equals_serial_with_real_lookup(setup_decoder, copy_prices, monkeypatch, rows, sampled):
    _, make = setup_decoder
    sampling = Sampling(2**64 - 3, .73123456789, 20, .92345678901) if sampled else None
    monkeypatch.setenv("TF_GLM_COPY_ROWS", str(max(2, rows)))
    d = make(8, limit=256)
    prompt, expected = edit_prompt(d.w, sampling)
    s = _stream(prompt, len(expected), sampling=sampling, draft=rows != 1, policy=f"f{max(1, rows - 1)}")
    d.admit(s)
    _drain(d)
    assert s.out == expected
    assert all(n <= rows - 1 for n in s.depths)
    assert rows - 1 in s.depths
    if rows > 1:
        assert s.copy_stats["accepted"] > 0 and s.copy_stats["drafter_skips"] > 0
        assert s.copies.lookup.ids == s.prompt + s.out
        assert len(s.drafter.proposals) < s.rounds


@pytest.mark.parametrize("slots", [2, 4, 8])
@pytest.mark.parametrize("sampled", [False, True])
def test_copies_are_isolated_at_load_and_include_64_total_rows(setup_decoder, copy_prices, monkeypatch, slots, sampled):
    mod, make = setup_decoder
    d = make(slots, limit=256, pool_limit=2048)
    totals, original = [], mod.compute_streams

    def compute(w, segs, buf, **kw):
        totals.append(segs[-1][2])
        return original(w, segs, buf, **kw)

    monkeypatch.setattr(mod, "compute_streams", compute)
    streams, expected = [], []
    for i in range(slots):
        sampling = Sampling(100 + i, .73, 20, .92) if sampled else None
        prompt, tokens = edit_prompt(d.w, sampling)
        s = _stream(prompt, len(tokens), sampling=sampling)
        d.admit(s)
        streams.append(s)
        expected.append(tokens)
    _drain(d)
    assert [s.out for s in streams] == expected
    assert len({id(s.copies) for s in streams}) == slots
    assert all(s.copies.lookup.ids == s.prompt + s.out for s in streams)
    assert max(totals) == slots * 8


def test_wrong_copy_is_rejected_without_changing_serial_output(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected))
    d.admit(s)
    assert s.copies.pending and s.drafts
    s.drafts[0] = (s.drafts[0] + 1) % 64  # plausible vocabulary token, intentionally wrong
    d.round()
    assert s.last[1] == 0 and s.copies.accepted == 0
    _drain(d)
    assert s.out == expected


def test_missing_prices_and_flag_off_preserve_baseline_messages_and_drafts(setup_decoder, monkeypatch):
    _, make = setup_decoder
    runs = []
    for enabled in (False, True):
        monkeypatch.delenv("TF_GLM_COPY_PRICES", raising=False)
        d = make(2)
        s = _stream([1, 2, 3], 30)
        s.copy_request = enabled
        d.admit(s)
        _drain(d)
        runs.append((s, d.messages))
    a, b = (r[0] for r in runs)
    assert (a.out, a.depths, a.keeps) == (b.out, b.depths, b.keeps)
    assert "copy_stats" not in a.stats()
    assert all(m[0] in (1, 2, 3, 4) for m in runs[0][1] if len(m) != 3 or m != a.prompt)
    assert not b.copy_stats["drafter_skips"]


def test_load_change_with_missing_peer_price_falls_back(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    a, b = _stream(prompt, len(expected)), _stream(prompt, len(expected))
    d.admit(a)
    assert a.copies.pending is not None
    d.copy_control.prices = CopyPrices(price_data(loads=(1,)))
    d.admit(b)  # the copied peer's depth-zero cost is also uncovered at load two
    _drain(d)
    assert a.out == b.out == expected
    assert b.copy_stats["unpriced"] > 0


@pytest.mark.parametrize("fair", [False, True])
def test_copy_depth_respects_explicit_and_background_policy_caps(setup_decoder, copy_prices, monkeypatch, fair):
    monkeypatch.setenv("TF_GLM_FAIR_SCHED", "1" if fair else "0")
    _, make = setup_decoder
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected), policy="f7" if fair else "f1")
    if fair:
        s.priority = "background"
    d.admit(s)
    _drain(d)
    assert s.out == expected and s.copy_stats["accepted"] > 0
    assert max(s.depths) == 1


def test_copy_model_copy_switch_keeps_taps_and_nonzero_slot_exact(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    blocker = _stream([1], 1)
    d.admit(blocker)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected))
    d.admit(s)
    assert s.slot == 1
    d.finish([blocker])
    prices = d.copy_control.prices
    d.copy_control.prices = None
    d.round()  # verify copy, then choose model
    assert s.copy_stats["sources"][-1] == "model"
    d.copy_control.prices = prices
    d.round()  # verify model, then choose copy
    assert s.copy_stats["sources"][-1] == "copy"
    assert s.drafter.context_end == s.st.pos
    _drain(d)
    assert s.out == expected and s.drafter.context_end == s.st.pos


def test_missing_tap_negative_control_is_caught_on_model_fallback(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    prompt, _ = edit_prompt(d.w)
    s = _stream(prompt, 30)
    d.admit(s)
    s.drafter.add_taps = lambda taps: None  # deliberately break required context maintenance
    d.copy_control.prices = None
    d.round()
    with pytest.raises(AssertionError):
        assert s.drafter.context_end == s.st.pos, "stale model context after copy"


def test_followers_never_choose_from_local_prices_and_reject_bad_controls(setup_decoder, copy_prices):
    _, make = setup_decoder
    leader = make(2)
    prompt, expected = edit_prompt(leader.w, Sampling(99, .73, 20, .92))
    s = _stream(prompt, len(expected), sampling=Sampling(99, .73, 20, .92))
    leader.admit(s)
    _drain(leader)
    for rank in (1, 2):
        follower = make(2)
        follower.rank = follower.owner.rank = rank
        follower.copy_control._choose = lambda *args: pytest.fail("follower used local steering numbers")
        outputs, finish = [], follower._finish

        def capture(sids):
            outputs.extend(list(follower.streams[sid].out) for sid in sids)
            finish(sids)

        follower._finish = capture
        messages = iter(leader.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == [expected]
    broken = make(2)
    broken.rank = broken.owner.rank = 1
    messages = [list(m) for m in leader.messages]
    control = next(m for m in messages if m[0] == COPY_PROPOSE)
    control[3] = 8  # illegal nine-row verification, even if legacy env permits it
    source = iter(messages)
    broken.share = lambda _: next(source)
    with pytest.raises(RuntimeError, match="invalid rank-zero copy proposal"):
        broken.follow()


def test_eos_room_cancellation_and_reused_span(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    eos = _stream([62], 20)
    eos.stop_eos = True
    d.admit(eos)
    assert eos.done and not eos.drafts and eos.copies.lookup.ids == [62, 63]
    d.finish([eos])
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, 1)
    d.admit(s)
    assert s.done and not s.drafts
    d.finish([s])
    cancelled = _stream(prompt, 30)
    d.admit(cancelled)
    cancelled.cancelled = lambda: True
    before = list(cancelled.copies.lookup.ids)
    d.finish(d.round())
    assert cancelled.copies.lookup.ids == before
    new = _stream(prompt, len(expected))
    d.admit(new)
    _drain(d)
    assert new.out == expected and new.copies is not cancelled.copies


@pytest.mark.parametrize("omit_absorb", [False, True])
def test_mtp_copy_and_model_switch_absorb_before_scratch_reuse(setup_decoder, copy_prices, monkeypatch, omit_absorb):
    mod, make = setup_decoder
    from tensorfold.families.glm5_next.cuda import decode

    prefill = mod.prefill
    absorbed = []

    def mtp_prefill(e, prompt, *args, **kwargs):
        first = prefill(e, prompt, *args, **kwargs)
        e.st.set_mtp_len(len(prompt) - 1)
        return first

    def absorb(e, hidden, new):
        assert torch.equal(hidden, e.st.kc[0][e.st.pos - len(new):e.st.pos])
        e.st.set_mtp_len(e.st.mtp_len + len(new))
        assert e.st.mtp_len == e.st.pos
        absorbed.append(e.st.pos)

    def draft(e, hidden, new, pos, depth, *args):
        absorb(e, hidden, new)
        return [int((new[-1] + pos + i) % 64) for i in range(depth)]

    monkeypatch.setattr(mod, "prefill", mtp_prefill)
    monkeypatch.setattr(mod, "draft", draft)
    monkeypatch.setattr(decode, "absorb", absorb)
    if omit_absorb:
        monkeypatch.setattr(decode, "absorb", lambda *args: None)
    d = make(2)
    d.w.mtp = object()
    streams = []
    for i in range(2):
        prompt, expected = edit_prompt(d.w)
        s = _stream(prompt, len(expected), policy="3")
        d.admit(s)
        streams.append((s, expected))
    prices = d.copy_control.prices
    d.copy_control.prices = None
    if omit_absorb:
        with pytest.raises(AssertionError):
            d.round()  # model fallback detects context rows omitted on the copy path
        return
    d.round()
    d.copy_control.prices = prices
    d.round()
    assert all(s.copy_stats["sources"][-1] == "copy" for s, _ in streams)
    _drain(d)
    assert all(s.out == expected for s, expected in streams)
    assert len(absorbed) > 4


def test_copy_model_switch_at_ring_wrap_preserves_aligned_context(setup_decoder, copy_prices, monkeypatch):
    class RingDrafter(_Drafter):
        def add_taps(self, taps):
            self.added.append(taps.clone())
            for i, row in enumerate(taps):
                for cache in self.kc + self.vc:
                    cache[:, (self.context_end + i) % self.ring].copy_(row[:2])
            self.context_end += len(taps)
            self.pos_dev.fill_(self.context_end)

        def propose(self, *args):
            # The aligned live tile preceding the window remains present after
            # any copy-only round; the drafter reads it when it resumes.
            lo = max(0, (self.context_end - self.window) // 64 * 64)
            for pos in range(lo, self.context_end):
                assert torch.all(self.kc[0][:, pos % self.ring, 1] == pos)
            return super().propose(*args)

    monkeypatch.setattr(_Drafter, "add_taps", RingDrafter.add_taps)
    monkeypatch.setattr(_Drafter, "ring", 96, raising=False)
    monkeypatch.setattr(_Drafter, "window", 24, raising=False)
    _, make = setup_decoder
    d = make(2)
    parent = d.owner.drafter
    parent.__class__ = RingDrafter
    prompt, _ = edit_prompt(d.w)
    reference = make(2)
    plain = _stream(prompt, 60, draft=False)
    reference.admit(plain)
    _drain(reference)
    expected = plain.out
    blocker = _stream([4], 1)
    d.admit(blocker)
    s = _stream(prompt, len(expected))
    d.admit(s)
    d.finish([blocker])
    # Keep copying through a physical ring wrap, then resume model proposals.
    while s.st.pos < parent.ring:
        d.round()
    assert not s.done
    d.copy_control.prices = None
    _drain(d)
    assert s.out == expected and s.drafter.context_end == s.st.pos
    assert s.drafter.proposals


@pytest.mark.parametrize("bad_field", [2, 3, 4, 5, 7])
def test_agreement_vote_detects_local_position_source_depth_or_token_on_every_rank(setup_decoder, bad_field):
    _, make = setup_decoder
    # Fixed records include local position/pending/source/depth/token ids, not
    # just a checksum of a received instruction. All ranks see the same failure.
    correct = [1, 3, 10, 7, 2, 2, 11, 12, -1, -1, -1, -1, -1]
    changed = correct[:]
    changed[bad_field] += 1
    for rank in (0, 1, 2):
        d = make(2)
        d.rank = d.owner.rank = rank
        d.copy_control.check = True
        d.share = lambda _: [COPY_CHECKED, 3, 2, 2, 11, 12]
        d.owner._gather_ints = lambda _: [correct, changed, correct]
        s = SimpleNamespace(sid=3, st=SimpleNamespace(pos=10))
        with pytest.raises(RuntimeError, match="invalid rank-zero copy proposal"):
            d.copy_control._exchange(s, 2, 2, [11, 12], 7, 7, check_local=True)


def test_default_copy_arm_runs_both_proposers_and_broadcasts_source(setup_decoder, copy_prices, monkeypatch):
    monkeypatch.delenv("TF_GLM_COPY_SKIP")
    _, make = setup_decoder
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected))
    d.admit(s)
    # A deliberately poor measured model history makes the next copy win even
    # when its price includes the already executed model pass.
    s.copies.model_rates = {i: [0, 99] for i in range(7)}
    d.round()
    assert s.copy_stats["sources"][-1] == "copy"
    _drain(d)
    assert s.out == expected and s.copy_stats["drafter_skips"] == 0
    assert len(s.drafter.proposals) == sum(src != "empty" for src in s.copy_stats["sources"])
    follower = make(2)
    follower.rank = follower.owner.rank = 1
    outputs, finish = [], follower._finish

    def capture(sids):
        outputs.extend(list(follower.streams[sid].out) for sid in sids)
        finish(sids)

    follower._finish = capture
    messages = iter(d.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert outputs == [expected]


def test_replay_prior_admission_checks_tokenizer_and_followers_do_not_load_it(setup_decoder, copy_prices,
                                                                            monkeypatch, tmp_path):
    import hashlib

    path = tmp_path / "prior.json"
    path.write_text(json.dumps(replay_prior()))
    monkeypatch.setenv("TF_GLM_COPY_PRIOR", str(path))
    monkeypatch.setenv("TF_GLM_COPY_TOKENIZER_SHA256", "b" * 64)
    _, make = setup_decoder
    invalid = make(2)
    with pytest.raises(ValueError, match="tokenizer"):
        invalid.admit(_stream([1, 2, 3], 10))
    assert invalid.messages == []
    monkeypatch.setenv("TF_GLM_COPY_TOKENIZER_SHA256", "a" * 64)
    lead = make(2)
    prompt, expected = edit_prompt(lead.w)
    s = _stream(prompt, len(expected))
    lead.admit(s)
    assert len(s.copies.prior_rates) == 153
    assert s.copy_stats["prior_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    _drain(lead)
    assert s.out == expected and s.copy_stats["accepted"] > 0
    # Follower may have a missing/different file and identity; only rank zero
    # consults these. The wire controls carry every source and copy token.
    path.unlink()
    monkeypatch.setenv("TF_GLM_COPY_TOKENIZER_SHA256", "invalid")
    follower = make(2)
    follower.rank = follower.owner.rank = 2
    outputs, finish = [], follower._finish

    def capture(sids):
        outputs.extend(list(follower.streams[sid].out) for sid in sids)
        finish(sids)

    follower._finish = capture
    messages = iter(lead.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert outputs == [expected]


@pytest.mark.parametrize("sampled", [False, True])
def test_copies_after_cofill_match_serial(cofill_decoder, copy_prices, sampled):
    _, make, _ = cofill_decoder
    d = make(2)
    streams, expected = [], []
    for i in range(2):
        sampling = Sampling(100 + i, .73, 20, .92) if sampled else None
        prompt, tokens = edit_prompt(d.w, sampling)
        s = _stream(prompt, len(tokens), sampling=sampling)
        d.begin_admit(s)
        streams.append(s)
        expected.append(tokens)
    d.prefill_step()
    assert all(s.cofill_stats["streams"] == 2 for s in streams)
    _drain(d)
    assert [s.out for s in streams] == expected
    assert all(s.copy_stats["accepted"] > 0 and s.copies.lookup.ids == s.prompt + s.out for s in streams)


@pytest.fixture
def joint_copy(setup_decoder, copy_prices, monkeypatch, tmp_path):
    path = tmp_path / "joint-prices.json"
    data = {"schema": "tf-price/v1", "kind": "measured", "build": "test-build",
            "weights": "test-weights", "drafter": "test-drafter",
            "cells": [{"rows": rows, "load": load, "ctx": "short", "path": "eager",
                       "median_ms": 10 + rows * .01, "spread": .1, "reps": 30}
                      for load in (1, 2, 4, 8) for rows in range(load, 8 * load + 1)]}
    data["cells"] += [{"rows": rows, "load": 1, "ctx": "short", "path": "graph-main",
                       "median_ms": 10 + rows * .01, "spread": .1, "reps": 30} for rows in range(1, 7)]
    path.write_text(json.dumps(data))
    monkeypatch.setenv("TF_GLM_PRICE_TABLE", str(path))
    monkeypatch.delenv("TF_GLM_COPY_PRICES")  # Joint copy drafting must consume the shared table only
    original = _Drafter.propose

    def observed(self, pending, depth, sampling, confidence):
        tokens = original(self, pending, depth, sampling, confidence)
        if getattr(self, "observe", False):
            self.last_observation = {"claims": [.05] * len(tokens), "candidates": [list(range(64)) for _ in tokens]}
        return tokens

    monkeypatch.setattr(_Drafter, "propose", observed)
    mod, make = setup_decoder

    def build(*args, **kwargs):
        d = make(*args, **kwargs)
        d.w.cfg.dense_limit = 256
        d.owner.e.st.parity = 0
        d.owner.e.graphs = SimpleNamespace(main={(n, p): object() for n in range(1, 7) for p in (0, 1)}, sparse={})
        return d

    return mod, build


@pytest.mark.parametrize("load", [1, 2, 4, 8])
@pytest.mark.parametrize("sampled", [False, True])
def test_joint_copy_uses_round2_and_calibrates_only_selected_source(joint_copy, load, sampled):
    mod, make = joint_copy
    d = make(max(2, load), limit=256, pool_limit=2048)
    streams, expected = [], []
    for i in range(load):
        sampling = Sampling(100 + i, .73, 20, .92) if sampled else None
        prompt, tokens = edit_prompt(d.w, sampling)
        s = _stream(prompt, len(tokens), sampling=sampling, policy="fp7")
        d.admit(s)
        assert len(s.drafts) == 7 and len(s.copy_drafts) == 2  # both complete chains remain available
        streams.append(s)
        expected.append(tokens)
    d.round()
    assert all(s.copy_stats["accepted"] > 0 for s in streams)
    assert all(s.calibration.counts["copy"][0][1] == 1 for s in streams)
    assert all(all(pair == [0., 0.] for pair in s.calibration.counts["text"]) for s in streams)
    _drain(d)
    assert [s.out for s in streams] == expected
    assert all(s.copy_stats["drafter_skips"] == 0 for s in streams)
    assert all(s.copies.lookup.ids == s.prompt + s.out for s in streams)
    rounds = [m for m in d.messages if m[0] == mod.ROUND2]
    assert rounds and any(mod.SRC_C in m[2 + m[1]:2 + 2 * m[1]] for m in rounds)
    assert all(len(m) == 2 + 3 * m[1] for m in rounds)


def test_joint_copy_followers_use_broadcast_chains_with_different_local_tables(joint_copy, monkeypatch):
    _, make = joint_copy
    lead = make(2)
    streams = []
    for i in range(2):
        sampling = Sampling(10 + i, .73, 20, .92)
        prompt, expected = edit_prompt(lead.w, sampling)
        s = _stream(prompt, len(expected), sampling=sampling, policy="fp7")
        lead.admit(s)
        streams.append(s)
    _drain(lead)
    for rank in (1, 2):
        follower = make(2)
        follower.rank = follower.owner.rank = rank
        follower.price_table = None
        follower.prior = {"copy": 0.0, "text": 99.0}
        follower.copy_control.prices = None
        outputs, finish = {}, follower._finish

        def capture(sids):
            outputs.update({sid: list(follower.streams[sid].out) for sid in sids})
            finish(sids)

        follower._finish = capture
        messages = iter(lead.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == {s.sid: s.out for s in streams}


def test_joint_copy_coverage_fallback_keeps_the_model_cap_chain(joint_copy):
    mod, make = joint_copy
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    streams = [_stream(prompt, len(expected), policy="fp7") for _ in range(2)]
    for s in streams:
        d.admit(s)
    # A source comparison may not borrow the load-one cells, even with a copy.
    d.price_table.cells = [c for c in d.price_table.cells if c["load"] == 1]
    d.round()
    msg = next(m for m in d.messages if m[0] == mod.ROUND2)
    assert msg[4:6] == [mod.SRC_F, mod.SRC_F] and msg[6:] == [1, 1]
    assert all(s.copy_stats["accepted"] == 0 and s.copies.pending is None for s in streams)
    _drain(d)
    assert [s.out for s in streams] == [expected, expected]


def test_joint_missing_table_does_not_activate_standalone_copy_prices(setup_decoder, copy_prices):
    _, make = setup_decoder
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    _drain(d)
    assert not s.copy_enabled and not hasattr(s, "copy_stats")
    assert s.policy == "fc7:0.3" and s.out == expected


def test_joint_wrong_copy_is_verified_and_rejected(joint_copy):
    _, make = joint_copy
    d = make(2)
    prompt, expected = edit_prompt(d.w)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    s.copy_drafts[0] = (s.copy_drafts[0] + 1) % 64
    d.round()
    assert s.last[1] == 0 and s.copy_stats["accepted"] == 0
    assert s.calibration.counts["copy"][0][1] == 0
    _drain(d)
    assert s.out == expected and s.copies.lookup.ids == s.prompt + s.out


def test_joint_copy_after_cofill_keeps_private_histories(joint_copy, cofill_decoder):
    _, make, _ = cofill_decoder
    d = make(2)
    streams, expected = [], []
    for i in range(2):
        sampling = Sampling(100 + i, .73, 20, .92)
        prompt, tokens = edit_prompt(d.w, sampling)
        s = _stream(prompt, len(tokens), sampling=sampling, policy="fp7")
        d.begin_admit(s)
        streams.append(s)
        expected.append(tokens)
    d.prefill_step()
    _drain(d)
    assert [s.out for s in streams] == expected
    assert all(s.cofill_stats["streams"] == 2 and s.copy_stats["accepted"] > 0 for s in streams)
    assert all(s.copy_stats["price_table"] == "shared:tf-price/v1" for s in streams)
