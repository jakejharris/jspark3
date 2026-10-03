"""Real prefill/checkpoint/restore control flow, with deterministic CPU state updates replacing GPU kernels."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.glm5_next.cuda.engine import GlmEngine, configured_identical_cache


@pytest.fixture
def make_engine(monkeypatch):
    d = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    probe = importlib.import_module("tensorfold.families.glm5_next.cuda.tp3_probe")
    monkeypatch.setattr(probe, "recorder", None)
    monkeypatch.setattr(d.prof, "report", lambda _: None)
    monkeypatch.setattr(d.prof, "timed", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(d, "stage", lambda w, st, b, ids: ids)
    monkeypatch.setattr(d, "chunks_for", lambda *_: 1)

    def compute(w, st, b, ids, **kw):
        b.calls.append((st.pos, list(ids)))
        for i, token in enumerate(ids):
            st.rec[0].mul_(31).add_(token).remainder_(1000003)
            st.conv.copy_(torch.roll(st.conv, -1))
            st.conv[-1] = token
            b.fnormed[i, 0] = st.rec[0, 0]
            st.kc[0][st.pos + i] = st.rec[0, 0]
        return st.rec[0].clone()

    monkeypatch.setattr(d, "compute", compute)
    monkeypatch.setattr(d, "commit", lambda w, st, b, r, keep: st.set_pos(st.pos + keep))

    def absorb(e, hidden, tokens):
        for h, token in zip(hidden[:, 0].tolist(), tokens):
            e.st.mtp_kc[e.st.mtp_len] = int(h) + token
            e.st.set_mtp_len(e.st.mtp_len + 1)

    monkeypatch.setattr(d, "_absorb_rows", absorb)

    def make(enabled=True, mtp=False, dflash=False, rows=4, disk=False):
        st = SimpleNamespace(pos=0, mtp_len=0, mtp_drafted=0, cur=[0],
                             rec=torch.zeros((2, 1), dtype=torch.int64), conv=torch.zeros(3, dtype=torch.int64),
                             kc=[torch.zeros(256, dtype=torch.int64)], vc=[None], index=None,
                             mtp_kc=torch.zeros(256, dtype=torch.int64))
        st.set_pos = lambda pos: setattr(st, "pos", pos)
        st.set_mtp_len = lambda pos: setattr(st, "mtp_len", pos)
        b = SimpleNamespace(fnormed=torch.zeros((rows, 1), dtype=torch.int64), calls=[], overlay=None)
        e = SimpleNamespace(st=st, pbuf=b, w=SimpleNamespace(mtp=object() if mtp else None), prefill_rows=rows,
                            images=None, overlay=lambda *_: None, sample=lambda last, *_: last.tolist())
        def reset():
            st.rec.zero_(); st.conv.zero_(); st.set_pos(0); st.set_mtp_len(0)
            st.mtp_drafted = 0
        e.reset = reset
        e.tap_rows = lambda r, _: b.fnormed[:r].clone()
        drafter = SimpleNamespace(context_end=0, pos_dev=torch.zeros(1, dtype=torch.int32),
                                  kc=[torch.zeros((1, 256, 1), dtype=torch.int64)],
                                  vc=[torch.zeros((1, 256, 1), dtype=torch.int64)]) if dflash else None
        if drafter:
            drafter.reset = lambda: setattr(drafter, "context_end", 0)
            def add_taps(taps):
                start, end = drafter.context_end, drafter.context_end + len(taps)
                drafter.kc[0][:, start:end].copy_(taps.unsqueeze(0))
                drafter.vc[0][:, start:end].copy_((taps + 7).unsqueeze(0))
                drafter.context_end = end
                drafter.pos_dev.fill_(end)
            drafter.add_taps = add_taps
        g = GlmEngine.__new__(GlmEngine)
        g.cache_last_token = enabled
        g.e, g.w, g.rank, g.drafter, g.eos = e, e.w, 0, drafter, ()
        g.cache, g.live, g.cache_entries, g.cache_bytes = [], [], 1, 1 << 20
        g._image_rows = lambda *_: None
        g._drafters = lambda _: (False, mtp, dflash)
        g._gather_ints = lambda x: [x, x, x]
        g.disk = SimpleNamespace(put=lambda e, s: written.append(s), has=lambda _: False) if disk else None
        g.checkpoint_after = lambda pos: (pos // 4 + 1) * 4
        written = []
        g.written = written
        return g
    return make


def run(g, prompt, draft=True, hit=None, resuming=False):
    tokens = []
    stats = g._run(prompt, 1, None, False, lambda ids: tokens.extend(ids), [0, 0, 0, 0], hit, draft,
                   resuming=resuming)
    return tokens, stats


@pytest.mark.parametrize("mtp,dflash", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("n", [2, 4, 5, 9, 10, 17])
def test_real_prefix_state_and_one_token_repeat_equal_cold(make_engine, mtp, dflash, n):
    g = make_engine(mtp=mtp, dflash=dflash)
    prompt = list(range(1, n + 1))
    cold, stats = run(g, prompt)
    assert stats["cached"] == 0 and len(g.cache) == 1
    snap = g.cache[0]
    assert snap.ids == prompt[:-1]
    # The snapshot is the actual prefix state, not full state labelled with fewer IDs.
    prefix = make_engine(enabled=False, mtp=mtp, dflash=dflash)
    run(prefix, prompt[:-1])
    assert torch.equal(snap.rec, prefix.e.st.rec[0])
    assert torch.equal(snap.conv, prefix.e.st.conv)
    assert not torch.equal(snap.rec, g.e.st.rec[0])
    assert snap.drafter_end == (n - 1 if dflash else -1)
    assert snap.mtp_len == (n - 2 if mtp else -1)
    full_mtp = g.e.st.mtp_kc[:g.e.st.mtp_len].clone()
    # A reply may overwrite recurrent state and all positions after the prefix.
    g.e.st.rec.fill_(999); g.e.st.conv.fill_(999); g.e.st.set_pos(n + 7)
    g.e.pbuf.calls.clear()
    hit = g._resume(prompt, [0, 0, 0, 0])
    assert hit is snap
    warm, stats = run(g, prompt, hit=hit, resuming=True)
    assert warm == cold and stats["cached"] == n - 1
    assert g.e.pbuf.calls == [(n - 1, [prompt[-1]])]
    assert torch.equal(full_mtp, g.e.st.mtp_kc[:g.e.st.mtp_len])
    assert g.cache == [snap]       # entries=1 still retains the useful prefix


def test_unset_and_zero_keep_full_prompt_cache_and_identical_miss(make_engine, monkeypatch):
    for value in (None, "0"):
        if value is None: monkeypatch.delenv("TF_GLM_CACHE_LAST_TOKEN", raising=False)
        else: monkeypatch.setenv("TF_GLM_CACHE_LAST_TOKEN", value)
        g = make_engine(enabled=configured_identical_cache(1))
        run(g, list(range(10)))
        assert len(g.cache[0].ids) == 10
        assert g._resume(list(range(10)), [0] * 4) is None
        assert [len(ids) for _, ids in g.e.pbuf.calls] == [4, 4, 2]


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_serial_uses_same_cold_boundaries_but_keeps_nothing(make_engine, n):
    g = make_engine(mtp=True)
    prompt = list(range(n))
    drafted, _ = run(g, prompt)
    cold_calls = list(g.e.pbuf.calls)
    g.cache.clear(); g.e.pbuf.calls.clear()
    serial, stats = run(g, prompt, draft=False)
    assert serial == drafted and stats["cached"] == 0
    assert g.e.pbuf.calls == cold_calls and not g.cache


def test_append_changed_final_token_and_rank_miss(make_engine):
    g = make_engine(mtp=True)
    run(g, list(range(10)))
    for prompt in (list(range(9)) + [42], list(range(10)) + [11, 12]):
        hit = g._resume(prompt, [0] * 4)
        assert hit is not None
        cold, _ = run(make_engine(mtp=True), prompt)
        warm, _ = run(g, prompt, hit=hit, resuming=True)
        assert warm == cold
    prompt = list(range(10)) + [11, 12]
    hit = g._resume(prompt, [0] * 4)
    g._gather_ints = lambda x: [x, [0], x]
    _, stats = run(g, prompt, hit=hit, resuming=True)
    assert stats["cached"] == 0


def test_disk_checkpoints_coexist_and_only_one_memory_entry_survives(make_engine):
    g = make_engine(mtp=True, disk=True)
    run(g, list(range(10)))
    assert {len(s.ids) for s in g.written} == {4, 8, 9}
    assert [len(s.ids) for s in g.cache] == [9]


def test_evicted_dflash_cache_falls_back_instead_of_reusing_stale_draft_state(make_engine):
    g = make_engine(dflash=True)
    run(g, list(range(10)))
    snap = g.cache[0]
    # Fix (b) retains compatible draft rows when they fit; force actual budget eviction.
    g.cache_bytes = 0
    g._take_over([])
    assert snap not in g.cache and snap.rows is None and snap.nbytes == 0
    assert g._resume(list(range(10)), [0] * 4) is None


@pytest.mark.parametrize("value", ["", "yes", "2", "true"])
def test_invalid_flag(monkeypatch, value):
    monkeypatch.setenv("TF_GLM_CACHE_LAST_TOKEN", value)
    with pytest.raises(ValueError, match="0 or 1"):
        configured_identical_cache(1)


def test_parallel_rejected_only_when_enabled(monkeypatch):
    monkeypatch.setenv("TF_GLM_CACHE_LAST_TOKEN", "1")
    assert configured_identical_cache(1)
    with pytest.raises(ValueError, match="one GLM stream"):
        configured_identical_cache(2)
    monkeypatch.setenv("TF_GLM_CACHE_LAST_TOKEN", "0")
    assert not configured_identical_cache(2)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("disk", [False, True])
def test_stacked_drafter_snapshot_restores_after_other_conversation(make_engine, enabled, mtp, disk):
    g = make_engine(enabled=enabled, mtp=mtp, dflash=True, disk=disk)
    g.cache_entries = 3
    a, b = list(range(1, 11)), list(range(31, 42))
    run(g, a)
    snap = g.cache[0]
    boundary = len(a) - int(enabled)
    want_kv = [t[:, :boundary].clone() for t in (*g.drafter.kc, *g.drafter.vc)]
    run(g, b)
    assert snap in g.cache and snap.rows is not None and snap.drafter_end == boundary
    assert not torch.equal(g.drafter.kc[0][:, :boundary], want_kv[0])
    prompt = a + [19]
    hit = g._resume(prompt, [0] * 4)
    assert hit is snap
    cold_engine = make_engine(enabled=enabled, mtp=mtp, dflash=True, disk=disk)
    cold, _ = run(cold_engine, prompt)
    warm, stats = run(g, prompt, hit=hit, resuming=True)
    assert warm == cold and stats["cached"] == boundary
    assert torch.equal(g.e.st.rec, cold_engine.e.st.rec)
    assert torch.equal(g.e.st.conv, cold_engine.e.st.conv)
    for got, expected in zip((*g.drafter.kc, *g.drafter.vc), want_kv, strict=True):
        assert torch.equal(got[:, :boundary], expected)
    for got, expected in zip((*g.drafter.kc, *g.drafter.vc),
                             (*cold_engine.drafter.kc, *cold_engine.drafter.vc), strict=True):
        assert torch.equal(got[:, :len(prompt)], expected[:, :len(prompt)])


@pytest.mark.parametrize("enabled", [False, True], ids=["full-prompt-cache", "last-token-cache"])
@pytest.mark.parametrize("mtp,dflash", [(False, False), (True, False), (False, True), (True, True)])
def test_score_between_prompts_preserves_resume_state(make_engine, monkeypatch, enabled, mtp, dflash):
    from tensorfold.families.glm5_next.cuda import quality_score

    g = make_engine(enabled=enabled, mtp=mtp, dflash=dflash)
    g.quality_score_enabled, g.concurrent = True, False
    g._share = lambda values: values
    a, b = list(range(1, 11)), list(range(31, 42))
    run(g, a)
    snap = g.cache[0]

    def overwrite(e, ids, start):
        e.reset()
        e.st.rec.fill_(777)
        e.st.conv.fill_(777)
        e.st.kc[0].fill_(777)
        e.st.mtp_kc.fill_(777)
        return [0.5] * (len(ids) - start)

    monkeypatch.setattr(quality_score, "score_prefill", overwrite)
    assert g.score(b, len(b) - 1) == [0.5]
    assert snap.rows is not None
    assert g.live == []
    prompt = a + [19]
    hit = g._resume(prompt, [0, 0, 0, 0])
    assert hit is snap
    cold_engine = make_engine(enabled=enabled, mtp=mtp, dflash=dflash)
    cold, _ = run(cold_engine, prompt)
    warm, stats = run(g, prompt, hit=hit, resuming=True)
    assert warm == cold and stats["cached"] == len(snap.ids)
    assert torch.equal(g.e.st.rec[g.e.st.cur[0]], cold_engine.e.st.rec[cold_engine.e.st.cur[0]])
    assert torch.equal(g.e.st.conv, cold_engine.e.st.conv)
    assert torch.equal(g.e.st.kc[0][:len(prompt)], cold_engine.e.st.kc[0][:len(prompt)])
    g.score(b, len(b) - 1)
    serial, serial_stats = run(g, prompt, draft=False)
    fresh_serial, _ = run(make_engine(enabled=enabled, mtp=mtp, dflash=dflash), prompt, draft=False)
    assert serial == fresh_serial and serial_stats["cached"] == 0
