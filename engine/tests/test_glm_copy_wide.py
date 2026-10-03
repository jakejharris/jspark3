"""Wide copy drafting: actual lookup and batched control, with host target/drafter fakes."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.copy_drafts import configured_copy_rows
from tensorfold.families.glm5_next.cuda.decode import path_for, sample_rows
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _drain, _logits, _stream, _Drafter  # noqa: F401
from test_glm_copy_batched import copy_prices, joint_copy  # noqa: F401
from test_glm_copy_drafts import replay_prior


def long_edit(w, sampling=None, *, context=256, count=160):
    tokens, token = [], 7
    for pos in range(context, context + count):
        token = sample_rows(w, _logits([token], pos - 1), [pos], sampling)[0]
        tokens.append(token)
    source = list(range(1, 8)) + tokens
    return source + [50] * (context - 8 - len(source)) + list(range(8)), tokens


@pytest.fixture
def wide_copy(joint_copy, monkeypatch, tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(replay_prior(.97)))
    monkeypatch.setenv("TF_GLM_COPY_PRIOR", str(prior))
    monkeypatch.setenv("TF_GLM_COPY_TOKENIZER_SHA256", "a" * 64)
    mod, build = joint_copy

    def make(rows, slots=2, **kwargs):
        monkeypatch.setenv("TF_GLM_COPY_WIDE_ROWS", str(rows))
        d = build(slots, **kwargs)
        d.w.cfg.dense_limit = 1 << 20
        # Synthetic component prices exercise the controls, never serving data.
        d.price_table.cells = [
            {"rows": n, "load": load, "ctx": ctx, "path": path,
             "median_ms": 10 + n * .0001, "spread": .1, "reps": 30}
            for load in (1, 2, 4, 8) for n in range(load, rows * load + 1)
            for ctx in ("short", "8k", "32k")
            for path in (("eager", "graph-main") if load == 1 and n <= 6 else ("eager",))]
        return d

    return mod, make


@pytest.mark.parametrize("value", ["8", "15", "33", "-1", "no"])
def test_bad_wide_parameter_fails_before_allocation(monkeypatch, value):
    monkeypatch.setenv("TF_GLM_COPY_WIDE_ROWS", value)
    with pytest.raises(ValueError):
        configured_copy_rows()


def test_off_keeps_eight_rows_and_model_policy_caps(monkeypatch):
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    monkeypatch.delenv("TF_GLM_COPY_WIDE_ROWS", raising=False)
    assert configured_copy_rows() == 8
    monkeypatch.setenv("TF_GLM_COPY_WIDE_ROWS", "32")
    assert configured_copy_rows() == 32 and encode_policy("fp7")[1] == 7
    with pytest.raises(ValueError):
        encode_policy("fp31")


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("load", [1, 2, 4, 8])
@pytest.mark.parametrize("sampled", [False, True])
def test_widening_to_256_total_rows_equals_serial(wide_copy, monkeypatch, rows, load, sampled):
    mod, make = wide_copy
    d = make(rows, max(2, load), limit=1024, pool_limit=8192)
    totals, compute = [], mod.compute_streams

    def measured(w, segs, buf, **kwargs):
        totals.append(segs[-1][2])
        return compute(w, segs, buf, **kwargs)

    monkeypatch.setattr(mod, "compute_streams", measured)
    forward = d.owner.e.forward
    def measured_single(tokens):
        totals.append(len(tokens))
        return forward(tokens)
    d.owner.e.forward = measured_single
    streams = []
    for i in range(load):
        sampling = Sampling(100 + i, .73, 20, .92) if sampled else None
        prompt, expected = long_edit(d.w, sampling)
        s = _stream(prompt, len(expected), sampling=sampling, policy="fp7")
        s.expected = expected
        d.admit(s)
        assert len(s.drafts) == 7 and len(s.copy_drafts) == 2
        streams.append(s)
    _drain(d)
    assert max(totals) <= load * rows
    assert max(totals) > load * 8
    if not sampled:
        assert max(totals) == load * rows
    assert d.guard == rows and d.buf.rows == max(2, load) * rows
    for s in streams:
        assert s.out == s.expected and s.copies.lookup.ids == s.prompt + s.out
        assert rows - 1 in s.depths and max(s.depths) == rows - 1
        assert max(n for _, n, _ in s.drafter.proposals) == 7
        assert s.drafter.context_end == s.st.pos
        assert s.copy_stats["drafter_skips"] == 0 and s.copy_stats["max_rows"] == rows
        assert len(s.calibration.counts["copy"]) == rows - 1
        assert s.calibration.counts["copy"][rows - 2][1] > 0
        assert len(s.calibration.counts["text"]) == 7


@pytest.mark.parametrize("width", range(9, 33))
@pytest.mark.parametrize("sampled", [False, True])
def test_each_eager_width_is_verified_against_serial(wide_copy, width, sampled):
    _, make = wide_copy
    d = make(16 if width <= 16 else 32)
    sampling = Sampling(101, .73, 20, .92) if sampled else None
    prompt, expected = long_edit(d.w, sampling, count=80)
    s = _stream(prompt, len(expected), sampling=sampling, policy="fp7")
    d.admit(s)
    s.copies.width = width - 1
    s.copy_offer = s.copies.offer([], s.copy_cap)
    s.copy_drafts = s.copy_offer.tokens
    assert len(s.copy_drafts) == width - 1
    d.round()
    assert s.depths[0] == width - 1 and s.keeps[0] == width
    _drain(d)
    assert s.out == expected


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("policy,wide", [("fp1", False), ("fp7", True), ("f7", False)])
def test_only_full_shared_policy_can_widen(wide_copy, rows, policy, wide):
    _, make = wide_copy
    d = make(rows)
    prompt, expected = long_edit(d.w)
    s = _stream(prompt, len(expected), policy=policy)
    d.admit(s)
    _drain(d)
    assert s.out == expected
    assert max(s.depths) <= (rows - 1 if wide else 1 if policy == "fp1" else 7)
    assert s.copy_stats["max_rows"] == (rows if wide else 8)


@pytest.mark.parametrize("rows", [16, 32])
def test_missing_wide_prices_never_extrapolate(wide_copy, rows):
    _, make = wide_copy
    d = make(rows)
    d.price_table.cells = [c for c in d.price_table.cells if c["rows"] <= 8]
    prompt, expected = long_edit(d.w)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    _drain(d)
    assert s.out == expected and max(s.depths) <= 7


def test_wide_copy_then_missing_base_reverts_to_full_model_chain(wide_copy):
    mod, make = wide_copy
    d = make(32)
    prompt, expected = long_edit(d.w)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    while len(s.copy_drafts) < 31:
        d.round()
    d.round()
    assert 31 in s.depths
    d.price_table.cells = [c for c in d.price_table.cells if c["load"] != 1]
    d.round()
    assert s.priced_source == mod.SRC_F and s.depths[-1] <= 7
    _drain(d)
    assert s.out == expected and s.copies.pending is None


def test_edit_at_32k_and_lone_nonzero_slot(wide_copy):
    _, make = wide_copy
    d = make(32, limit=65536, pool_limit=131072)
    blocker = _stream([3], 1)
    d.admit(blocker)
    prompt, expected = long_edit(d.w, context=32768)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    d.finish([blocker])
    assert s.slot == 1
    _drain(d)
    assert s.out == expected and 31 in s.depths and s.drafter.context_end == s.st.pos


@pytest.mark.parametrize("action", ["wrong-tail", "eos", "cancel"])
def test_wide_tail_rejection_eos_and_cancel_then_span_reuse(wide_copy, action):
    _, make = wide_copy
    d = make(32)
    prompt, expected = long_edit(d.w, count=80)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    s.copies.width = 31
    s.copy_offer = s.copies.offer([], s.copy_cap)
    s.copy_drafts = s.copy_offer.tokens
    if action == "wrong-tail":
        s.copy_drafts[24] = (s.copy_drafts[24] + 1) % 64
        d.round()
        assert s.keeps[-1] == 25 and s.copies.accepted == 24
        assert s.last_outcomes[24] == "mismatch" and s.last_outcomes[25:] == ["censored"] * 6
    elif action == "eos":
        token = expected[25]
        s.eos = (token,)
        expected = expected[:expected.index(token, 1) + 1]
    else:
        s.cancelled = lambda: True
        expected = expected[:1]
    _drain(d)
    assert s.out == expected
    follow = _stream(prompt, 40, policy="fp7")
    d.admit(follow)
    _drain(d)
    assert follow.out == long_edit(d.w, count=80)[1][:40]
    assert follow.copies is not s.copies


def test_wide_taps_cross_ring_wrap_and_omission_is_detected(wide_copy, monkeypatch):
    original = _Drafter.propose

    def taps(self, rows):
        for i, row in enumerate(rows):
            for cache in self.kc + self.vc:
                cache[:, (self.context_end + i) % 96].copy_(row[:2])
        self.context_end += len(rows)
        self.pos_dev.fill_(self.context_end)

    def propose(self, *args):
        lo = max(0, (self.context_end - 24) // 64 * 64)
        for pos in range(lo, self.context_end):
            assert torch.all(self.kc[0][:, pos % 96, 1] == pos), "missing committed taps"
        return original(self, *args)

    monkeypatch.setattr(_Drafter, "add_taps", taps)
    monkeypatch.setattr(_Drafter, "propose", propose)
    _, make = wide_copy
    d = make(32)
    prompt, expected = long_edit(d.w)
    s = _stream(prompt, len(expected), policy="fp7")
    d.admit(s)
    _drain(d)
    assert s.out == expected and 31 in s.depths
    bad = _stream(prompt, len(expected), policy="fp7")
    d.admit(bad)
    # Negative control: stale tap position/content must fail before a model
    # proposal can silently use the context after a kept wide window.
    bad.drafter.kc[0][:, (bad.st.pos - 1) % 96, 1] = -1
    bad.drafter.add_taps = lambda rows: None
    with pytest.raises(AssertionError, match="missing committed taps"):
        d.round()


@pytest.mark.parametrize("rows", [9, 16, 32])
def test_wide_rows_are_eager_even_if_graphs_are_present(rows):
    e = SimpleNamespace(st=SimpleNamespace(pos=1, parity=0, index=None),
                        w=SimpleNamespace(cfg=SimpleNamespace(dense_limit=1024)),
                        graphs=SimpleNamespace(main={(rows, 0): object()}, sparse={}))
    assert path_for(e, rows) == "eager"


@pytest.mark.parametrize("checked", [False, True])
def test_wide_followers_replay_and_final_token_vote_is_checked(wide_copy, monkeypatch, checked):
    from tensorfold.families.glm5_next.cuda.copy_control import COPY_CHECKED

    monkeypatch.setenv("TF_GLM_COPY_CHECK", "1" if checked else "0")
    _, make = wide_copy
    lead = make(32, pool_limit=2048)
    records = []

    def gather(vote):
        records.append(list(vote))
        return [vote] * 3

    lead.owner._gather_ints = gather
    streams = []
    for i in range(2):
        sampling = Sampling(10 + i, .73, 20, .92)
        prompt, expected = long_edit(lead.w, sampling)
        s = _stream(prompt, len(expected), sampling=sampling, policy="fp7")
        lead.admit(s)
        streams.append(s)
    _drain(lead)
    for rank in (1, 2):
        follower = make(32, pool_limit=2048)
        follower.rank = follower.owner.rank = rank
        follower.price_table = None
        follower.owner._gather_ints = lambda vote: [vote] * 3
        outputs, finish = {}, follower._finish

        def capture(sids):
            outputs.update({sid: list(follower.streams[sid].out) for sid in sids})
            finish(sids)

        follower._finish = capture
        source = iter(lead.messages + [[99]])
        follower.share = lambda _: next(source)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == {s.sid: s.out for s in streams}
    if checked:
        assert any(m[0] == COPY_CHECKED and m[3] == 31 for m in lead.messages)
        assert all(len(v) == 37 for v in records)
        bad = make(32)
        bad.copy_control.check = True
        def disagree(vote):
            other = list(vote)
            other[-1] += 1  # the 31st token, beyond the old seven-token vote
            return [vote, other, vote]
        bad.owner._gather_ints = disagree
        s = SimpleNamespace(sid=0, st=SimpleNamespace(pos=10))
        with pytest.raises(RuntimeError, match="invalid rank-zero copy"):
            bad.copy_control._exchange(s, 2, 31, list(range(31)), 50, 1)
