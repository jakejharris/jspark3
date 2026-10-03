"""Host checks: atomic express threshold and express cofill.

Short prompts that arrive together must be read in one atomic express pass
(idle) or in passes bounded by TF_GLM_EXPRESS_ATOMIC_ROWS (while replies
decode), with the same tokens, state and follower commands as reading each
alone. Long express prompts keep one-layer slices. 32 rows with express cofill
off selects the previous scheduling behavior.
"""

import importlib

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.engine.exact_sampling import Sampling
from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _Buffers, _stream  # noqa: F401
from test_glm_cofill import cofill_decoder, state_hash  # noqa: F401

ROWS = 512
LENGTHS = [41, 59, 44, 62, 50, 47, 55, 53]   # varied short prompts: 41-62 tokens


@pytest.fixture
def express_decoder(cofill_decoder):
    mod, make, forwards = cofill_decoder

    def build(slots=8, *, atomic=512, express_cofill=True, rows=ROWS, busy=None):
        d = make(slots, rows=rows, limit=4096, pool_limit=8192)
        d.w.layers = list(range(45))
        b = _Buffers(d.w, rows, d.capacity)
        b.set_taps((0, 1), 2)
        b.prefill, b.ka = True, torch.empty(rows, 2)
        d.express_buf = b
        d.express_atomic_rows, d.express_cofill = atomic, express_cofill
        d.express_busy_rows = atomic if busy is None else busy
        return d

    return mod, build, forwards


def prompts(lengths=LENGTHS):
    return [[(3 + i + j) % 60 for j in range(n)] for i, n in enumerate(lengths)]


def samplers(sampled, n):
    return [Sampling(20261002 + i, .73, 20, .92) if sampled else None for i in range(n)]


def serve(d, requests, *, max_steps=400):
    """The scheduler's per-iteration order: one prefill step, then one decode round."""
    passes = []
    for _ in range(max_steps):
        if not d.live():
            break
        before = len(d.messages)
        done = d.prefill_step()
        passes += [m for m in d.messages[before:] if m[0] in (d_mod(d).FILL, d_mod(d).FILL_SLICE,
                                                                d_mod(d).cofill.FILL_COFILL)]
        done += d.round()
        d.finish(done)
    assert not d.live()
    return passes


def d_mod(d):
    return importlib.import_module(type(d).__module__)


def alone(build, prompt, sampling, count, **kw):
    d = build(**kw)
    s = _stream(prompt, count, sampling=sampling)
    s.cofill = False
    d.begin_admit(s)
    d.prefill_step()
    first = (s.out[:], state_hash(s), s.drafter.context_end)
    serve(d, [s])
    return first + (s.out[:],)


@pytest.mark.parametrize("sampled", [False, True])
def test_idle_burst_reads_all_short_prompts_in_one_pass_with_alone_tokens(express_decoder, sampled):
    mod, build, forwards = express_decoder
    ps, ss = prompts(), samplers(sampled, len(LENGTHS))
    refs = [alone(build, p, s, 24) for p, s in zip(ps, ss)]
    d = build()
    streams = [_stream(p, 24, sampling=s) for p, s in zip(ps, ss)]
    for s in streams:
        d.begin_admit(s)
        assert s.express and s.cofill_ready and s.slice_layers == 0
    assert all(m[0] == mod.cofill.ADMIT_COFILL for m in d.messages if m[0] in (mod.ADMIT, mod.cofill.ADMIT_COFILL))
    before = len(forwards)
    d.prefill_step()
    assert len(forwards) == before + 1 and len(forwards[-1]) == len(LENGTHS)   # one pass, all eight
    assert not d.filling and d.express_owner is None
    assert [(s.out, state_hash(s), s.drafter.context_end) for s in streams] == [r[:3] for r in refs]
    assert all(s.cofill_stats == {"streams": 8, "total_rows": sum(LENGTHS), "rows": len(s.prompt)}
               for s in streams)
    serve(d, streams)
    assert [s.out for s in streams] == [r[3] for r in refs]


@pytest.mark.parametrize("atomic", [512, 128])
def test_newcomers_while_decoding_are_capped_per_pass_and_exact(express_decoder, atomic):
    mod, build, forwards = express_decoder
    lengths = [100, 120, 90, 110, 100, 95, 105]
    ps, ss = prompts(lengths), samplers(True, len(lengths))
    incumbent_prompt = [5] * 30
    refs = [alone(build, p, s, 16, atomic=atomic) for p, s in zip(ps, ss)]
    d = build(atomic=atomic)
    incumbent = _stream(incumbent_prompt, 200)
    d.begin_admit(incumbent)
    d.finish(d.prefill_step() + d.round())
    assert incumbent.out and incumbent.sid not in d.filling          # now decoding
    streams = [_stream(p, 16, sampling=s) for p, s in zip(ps, ss)]
    for s in streams:
        d.begin_admit(s)
    sizes = []
    for _ in range(50):
        if not d.filling:
            break
        before = len(forwards)
        d.finish(d.prefill_step() + d.round())
        sizes += [sum(n for _, n in f) for f in forwards[before:]]
    assert not d.filling
    assert max(sizes, default=0) <= atomic                                       # an incumbent's pause is bounded
    if atomic == 512:
        assert len(sizes) == 2 and sizes[0] > 400                     # 7 x ~103 rows in two passes, not seven
    assert all(s.out[:1] == r[0] for s, r in zip(streams, refs))
    while d.live():
        d.finish(d.prefill_step() + d.round())
    assert [s.out for s in streams] == [r[3] for r in refs]
    assert len(incumbent.out) == 200


@pytest.mark.parametrize("tokens,atomic,layers,cofill_ready", [
    (32, 32, 0, False), (33, 32, 1, False),          # previous threshold with cofill off (see flags-off test)
    (512, 512, 0, True), (513, 512, 1, False), (600, 512, 1, False),
])
def test_atomic_threshold_and_long_prompts_keep_slices(express_decoder, tokens, atomic, layers, cofill_ready):
    mod, build, _ = express_decoder
    d = build(atomic=atomic, express_cofill=atomic != 32, rows=1024)
    s = _stream([2] * tokens, 8)
    d.begin_admit(s)
    assert s.express and s.slice_layers == layers and s.cofill_ready is cofill_ready
    d.prefill_step()
    assert d.messages[-1][0] == (mod.FILL_SLICE if layers else mod.FILL)


def test_sliced_long_express_prompt_is_never_cofilled_with_short_ones(express_decoder):
    mod, build, forwards = express_decoder
    d = build(rows=1024)
    long, shorts = _stream([4] * 700, 8), [_stream([7 + i] * 50, 8) for i in range(3)]
    for s in [long, *shorts]:
        d.begin_admit(s)
    assert long.slice_layers == 1 and not long.cofill_ready
    d.prefill_step()                                     # shortest-first express choice: the three shorts
    assert len(forwards[-1]) == 3 and all(s.out for s in shorts) and not long.out
    assert d.messages[-1][0] == mod.cofill.FILL_COFILL and long.sid not in d.messages[-1][2:]


def test_flags_off_reproduce_0530f05_commands(express_decoder):
    mod, build, forwards = express_decoder
    ps = prompts()
    d = build(atomic=32, express_cofill=False)
    streams = [_stream(p, 12) for p in ps]
    for s in streams:
        d.begin_admit(s)
        assert s.express and s.slice_layers == 1 and not s.cofill_ready
    serve(d, streams)
    kinds = {m[0] for m in d.messages}
    assert mod.cofill.ADMIT_COFILL not in kinds and mod.cofill.FILL_COFILL not in kinds
    assert mod.FILL_SLICE in kinds and not forwards                  # previous behavior: one-layer slices, no cofill


def test_suspended_main_lane_owner_keeps_express_newcomers_single(express_decoder):
    mod, build, _ = express_decoder
    d = build()
    streams = [_stream([i + 1] * 50, 5) for i in range(3)]
    for s in streams:
        d.begin_admit(s)
    d.fill_owner = 999
    assert mod.cofill.candidates(d, streams[0]) == []
    with pytest.raises(RuntimeError, match="suspended"):
        mod.cofill.run(d, [s.sid for s in streams])
    d.fill_owner = None
    d.express_cofill = False                                      # follower config drift is refused
    with pytest.raises(RuntimeError, match="cofill shape"):
        mod.cofill.run(d, [s.sid for s in streams])


def test_express_cofill_followers_replay_rank_zero_commands(express_decoder):
    mod, build, _ = express_decoder
    lead = build()
    ps, ss = prompts(), samplers(True, len(LENGTHS))
    streams = [_stream(p, 20, sampling=s) for p, s in zip(ps, ss)]
    for s in streams:
        lead.begin_admit(s)
    serve(lead, streams)
    assert any(m[0] == mod.cofill.FILL_COFILL for m in lead.messages)
    for rank in (1, 2):
        follower = build()
        follower.rank = follower.owner.rank = rank
        outputs, finish = {}, follower._finish

        def record(sids):
            outputs.update({sid: list(follower.streams[sid].out) for sid in sids})
            finish(sids)

        follower._finish = record
        messages = iter(lead.messages + [[99]])
        follower.share = lambda _: next(messages)
        with pytest.raises(RuntimeError, match="unknown GLM worker message"):
            follower.follow()
        assert outputs == {s.sid: s.out for s in streams}


@pytest.mark.parametrize("env,atomic,on", [
    ({}, 512, True),
    ({"TF_GLM_EXPRESS_ATOMIC_ROWS": "32", "TF_GLM_EXPRESS_COFILL": "0"}, 32, False),
    ({"TF_GLM_EXPRESS_ATOMIC_ROWS": "1024"}, 1024, True),
])
def test_env_keys_and_defaults(monkeypatch, env, atomic, on):
    from tensorfold.families.glm5_next.cuda import prefill_options as p1

    for key in ("TF_GLM_EXPRESS_ATOMIC_ROWS", "TF_GLM_EXPRESS_COFILL", "TF_GLM_PREFILL_SLICE_MS", "TF_GLM_EXPRESS_BUSY_ROWS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    try:
        importlib.reload(p1)
        assert (p1.EXPRESS_ATOMIC_ROWS, p1.EXPRESS_COFILL, p1.PREFILL_SLICE_MS, p1.EXPRESS_BUSY_ROWS) == (atomic, on, 0, 256)
    finally:
        monkeypatch.undo()
        importlib.reload(p1)


@pytest.mark.parametrize("key,value", [("TF_GLM_EXPRESS_ATOMIC_ROWS", "0"), ("TF_GLM_EXPRESS_ATOMIC_ROWS", "1025"),
                                       ("TF_GLM_EXPRESS_ATOMIC_ROWS", "x"), ("TF_GLM_EXPRESS_COFILL", "2"),
                                       ("TF_GLM_PREFILL_SLICE_MS", "-1"), ("TF_GLM_PREFILL_SLICE_MS", "1001"),
                                       ("TF_GLM_EXPRESS_BUSY_ROWS", "0"), ("TF_GLM_EXPRESS_BUSY_ROWS", "1025"),
                                       ("TF_GLM_EXPRESS_GATHER_MS", "-1"), ("TF_GLM_EXPRESS_GATHER_MS", "201")])
def test_env_refuses_unbounded_values(monkeypatch, key, value):
    from tensorfold.families.glm5_next.cuda import prefill_options as p1

    monkeypatch.setenv(key, value)
    try:
        with pytest.raises(ValueError):
            importlib.reload(p1)
    finally:
        monkeypatch.undo()
        importlib.reload(p1)


def test_idle_express_cofill_stays_below_the_four_warp_rows(express_decoder):
    mod, build, forwards = express_decoder
    d = build(rows=2048)
    streams = [_stream([(5 + i) % 60] * 300, 4) for i in range(5)]          # 1500 rows waiting, idle
    for s in streams:
        d.begin_admit(s)
    d.prefill_step()
    assert sum(n for _, n in forwards[-1]) == 900 <= mod.cofill.EXPRESS_IDLE_ROWS


@pytest.mark.parametrize("rows,busy,decoding,kind", [
    (300, 256, True, "FILL_SLICE"), (300, 256, False, "FILL"), (256, 256, True, "FILL"), (300, 512, True, "FILL"),
])
def test_busy_cap_slices_large_atomic_prompts_only_while_replies_decode(express_decoder, rows, busy, decoding, kind):
    mod, build, _ = express_decoder
    outputs = []
    for cap in (1024, busy):                         # 1024: never demoted (reference tokens)
        d = build(busy=cap)
        if decoding:
            incumbent = _stream([5] * 30, 300)
            d.begin_admit(incumbent)
            d.finish(d.prefill_step() + d.round())
        s = _stream([(3 * i) % 60 for i in range(rows)], 16, sampling=Sampling(9, .7, 20, .95))
        s.cofill = False
        d.begin_admit(s)
        assert s.slice_layers == 0
        before = len(d.messages)
        d.finish(d.prefill_step())
        if cap == busy:
            fills = [m for m in d.messages[before:] if m[0] in (mod.FILL, mod.FILL_SLICE)]
            assert fills[0][0] == getattr(mod, kind)
        while d.live():
            d.finish(d.prefill_step() + d.round())
        outputs.append(s.out)
    assert outputs[0] == outputs[1]


def test_busy_cap_bounds_cofill_groups(express_decoder):
    mod, build, forwards = express_decoder
    d = build(busy=128)
    incumbent = _stream([5] * 30, 300)
    d.begin_admit(incumbent)
    d.finish(d.prefill_step() + d.round())
    streams = [_stream([(7 + i) % 60] * 60, 8) for i in range(7)]          # 7 x 60 = 420 rows
    for s in streams:
        d.begin_admit(s)
    sizes = []
    while d.filling:
        before = len(forwards)
        d.finish(d.prefill_step() + d.round())
        sizes += [sum(n for _, n in f) for f in forwards[before:]]
    assert sizes == [120, 120, 120]                                         # then one 60-row single pass


def _gather_build(build, monkeypatch, mod, gather_ms, clock):
    d = build()
    d.express_gather_s = gather_ms / 1000.0
    monkeypatch.setattr(mod.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    return d


def test_idle_gather_waits_for_staggered_arrivals_then_one_pass(express_decoder, monkeypatch):
    mod, build, forwards = express_decoder
    clock = [100.0]
    d = _gather_build(build, monkeypatch, mod, 30, clock)
    ps = prompts()
    first = _stream(ps[0], 8)
    d.begin_admit(first)
    before = len(d.messages)
    assert d.prefill_step() == [] and len(d.messages) == before and not forwards   # waiting, nothing sent
    clock[0] += 0.010
    rest = [_stream(p, 8) for p in ps[1:6]]                                          # 6 of 8 slots
    for s in rest:
        d.begin_admit(s)
    assert d.prefill_step() == [] and not forwards                                   # still inside 30 ms
    clock[0] += 0.025                                                                # 35 ms after the first
    d.prefill_step()
    assert len(forwards) == 1 and len(forwards[0]) == 6 and not d.filling
    d2 = _gather_build(build, monkeypatch, mod, 30, clock)                            # all slots taken: no wait
    for p in ps:
        d2.begin_admit(_stream(p, 8))
    d2.prefill_step()
    assert len(forwards) == 2 and len(forwards[1]) == 8


@pytest.mark.parametrize("case", ["off", "decoding", "long", "full_rows"])
def test_idle_gather_never_waits_when_it_cannot_help(express_decoder, monkeypatch, case):
    mod, build, forwards = express_decoder
    clock = [100.0]
    d = _gather_build(build, monkeypatch, mod, 0 if case == "off" else 30, clock)
    if case == "decoding":
        incumbent = _stream([5] * 30, 300)
        incumbent.cofill = False
        d.express_gather_s = 0.0
        d.begin_admit(incumbent)
        d.finish(d.prefill_step() + d.round())
        d.express_gather_s = 0.030
    if case == "long":
        d.begin_admit(_stream([4] * 600, 8))                      # sliced express prompt is waiting
    n = 7 if case == "full_rows" else 1                                       # 7 x 150 rows >= 1023
    for i in range(n):
        d.begin_admit(_stream([(9 + i) % 60] * (150 if case == "full_rows" else 50), 8))
    before = len(d.messages)
    d.prefill_step()
    assert len(d.messages) > before                              # stepped at once, no wait
