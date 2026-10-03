"""Real prefill/checkpoint/restore control flow with CPU kernels; discarded storage is poisoned."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.glm5_next.cuda import prefill_options as p1


@pytest.fixture
def rig(monkeypatch):
    d = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    monkeypatch.setattr(d.prof, "report", lambda *_: None)
    monkeypatch.setattr(d, "chunks_for", lambda *_: 1)
    monkeypatch.setattr(d, "stage", lambda w, st, b, ids: ids)
    monkeypatch.setattr(d, "commit", lambda w, st, b, r, keep: st.set_pos(st.pos + keep))
    calls = []

    def compute(w, st, b, ids, **kw):
        skip, lo = kw.get("skip_final_ffn", False), kw.get("tap_start", 0)
        calls.append((st.pos, len(ids), skip, lo))
        b.fnormed.fill_(float("nan"))
        for tap in b.taps:
            tap.fill_(float("nan"))
        for row, token in enumerate(ids):
            image = b.overlay.get(row, 0) if b.overlay else 0
            st.rec[0].mul_(31).add_(token + image).remainder_(100003)
            st.conv.copy_(torch.roll(st.conv, -1))
            st.conv[-1] = token + image
            st.kc[0][st.pos + row] = st.rec[0, 0]
            if not skip:
                b.fnormed[row, 0] = st.rec[0, 0] + 3
            if row >= lo:
                b.taps[0][row, 0] = st.rec[0, 0] + 7
                b.taps[1][row, 0] = st.rec[0, 0] + 11
        return None if skip else b.fnormed[len(ids) - 1].clone()
    monkeypatch.setattr(d, "compute", compute)
    def absorb(e, hidden, ids):
        assert torch.isfinite(hidden).all()
        for h, token in zip(hidden[:, 0], ids):
            e.st.mtp_kc[e.st.mtp_len] = h + token
            e.st.set_mtp_len(e.st.mtp_len + 1)
    monkeypatch.setattr(d, "_absorb_rows", absorb)

    def make(rows, *, mtp=False, draft="ring", images=False):
        st = SimpleNamespace(pos=0, mtp_len=0, mtp_drafted=0, cur=[0], rec=torch.zeros((2, 1)),
                             conv=torch.zeros(3), kc=[torch.zeros((2048, 1))], vc=[None], index=None,
                             mtp_kc=torch.zeros((2048, 1)))
        st.set_pos = lambda n: setattr(st, "pos", n)
        st.set_mtp_len = lambda n: setattr(st, "mtp_len", n)
        b = SimpleNamespace(prefill=True, fnormed=torch.empty((rows, 1)), taps=[torch.empty((rows, 1)) for _ in range(2)],
                            overlay=None)
        e = SimpleNamespace(w=SimpleNamespace(mtp=object() if mtp else None), st=st, pbuf=b, prefill_rows=rows,
                            sample=lambda last, *_: last.tolist(), tap_rows=lambda n, b: torch.cat([t[:n] for t in b.taps], 1))
        e.overlay = lambda at, n: {r: 100 for r in range(n) if images and (at + r) % 11 == 0}
        def reset():
            st.rec.zero_(); st.conv.zero_(); st.set_pos(0); st.set_mtp_len(0)
        e.reset = reset
        dr = None
        if draft:
            ring = 192 if draft == "ring" else 0
            dr = SimpleNamespace(context_end=0, pos_dev=torch.zeros(1, dtype=torch.int64), ring=ring,
                                 window=65 if ring else -1, tap_in=torch.zeros((64, 2)), groups=[],
                                 kc=[torch.full((1, ring or 2048, 2), float("nan"))],
                                 vc=[torch.full((1, ring or 2048, 2), float("nan"))])
            def reset_draft():
                dr.context_end = 0; dr.pos_dev.zero_(); dr.groups.clear()
                for cache in dr.kc + dr.vc: cache.fill_(float("nan"))
            dr.reset = reset_draft
            def taps(t):
                assert torch.isfinite(t).all(), "discarded target tap was consumed"
                for start in range(0, len(t), 64):
                    block = t[start:start + 64]
                    dr.groups.append((dr.context_end, len(block)))
                    idx = torch.arange(dr.context_end, dr.context_end + len(block))
                    if ring: idx %= ring
                    dr.kc[0].index_copy_(1, idx, block[None])
                    dr.vc[0].index_copy_(1, idx, (block + 17)[None])
                    dr.context_end += len(block)
                    dr.pos_dev.fill_(dr.context_end)
            dr.add_taps = taps
        return e, dr
    return d, make, calls


def state(e, dr):
    st = e.st
    values = [st.rec[0], st.conv, st.kc[0][:st.pos], st.mtp_kc[:st.mtp_len]]
    if dr:
        d = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
        values += d._ring_window(dr, dr.context_end) if dr.ring else [x[:, :dr.context_end] for x in dr.kc + dr.vc]
        assert dr.context_end == st.pos == int(dr.pos_dev.item())
    return [x.clone() for x in values]


def same(a, b):
    assert len(a) == len(b)
    assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))


@pytest.mark.parametrize("rows", [63, 64, 65, 128, 129])
@pytest.mark.parametrize("draft,mtp", [("ring", False), (None, False), ("unbounded", False), ("ring", True)])
@pytest.mark.parametrize("images", [False, True])
def test_dead_work_keeps_state_checkpoints_resume_and_groups(rig, monkeypatch, rows, draft, mtp, images):
    d, make, calls = rig
    prompt = list(range(1, 522))
    receipts = []
    for enabled in (False, True):
        monkeypatch.setattr(p1, "DEAD_WORK", enabled)
        e, dr = make(rows, mtp=mtp, draft=draft, images=images)
        kept, snapshots = [], []
        def keep(snap):
            d.save_rows(e, snap, dr)
            kept.append(state(e, dr))
            snapshots.append(snap)
        mark = lambda n: next((p for p in (131, 337) if p > n), None)
        calls.clear()
        first = d.prefill(e, prompt, None, mtp=mtp, drafter=dr, mark=mark, keep=keep)
        original = state(e, dr)
        groups = list(dr.groups) if dr else []
        flags = list(calls)
        # A kept state can restore after all live storage (including the drafter ring) is poisoned.
        e.st.rec.fill_(999); e.st.conv.fill_(999); e.st.kc[0].fill_(float("nan"))
        if dr:
            for c in dr.kc + dr.vc: c.fill_(float("nan"))
        snap = snapshots[-1]
        d.load_rows(e, snap, dr)
        resumed = d.prefill(e, prompt, None, mtp=mtp, drafter=dr, resume=snap)
        assert resumed == first
        same(state(e, dr), original)
        receipts.append((first, original, kept, groups, flags))
    assert receipts[0][0] == receipts[1][0]
    same(receipts[0][1], receipts[1][1])
    for a, b in zip(receipts[0][2], receipts[1][2]): same(a, b)
    assert all(group in receipts[0][3] for group in receipts[1][3])
    if mtp or draft == "unbounded":
        assert receipts[0][4] == receipts[1][4]
    else:
        assert any(skip or lo for _, _, skip, lo in receipts[1][4]), "optimization not exercised"


def test_poisoned_aligned_tile_is_live_even_when_nominally_masked(rig, monkeypatch):
    d, make, _ = rig
    monkeypatch.setattr(p1, "DEAD_WORK", True)
    e, dr = make(128)
    d.prefill(e, list(range(522)), None, mtp=False, drafter=dr)
    n = dr.context_end
    lo = max(0, n - dr.window) // 64 * 64
    idx = torch.arange(lo, n)
    values = dr.vc[0][:, idx % dr.ring]
    mask = (idx >= n - dr.window).view(1, -1, 1)
    assert torch.isfinite((values * mask).sum())
    # Emulates dropping the preceding tile: masked 0 * NaN is still NaN.
    assert lo < n - dr.window
    dr.vc[0][:, lo % dr.ring] = float("nan")
    assert not torch.isfinite((dr.vc[0][:, idx % dr.ring] * mask).sum())


def test_dead_prefix_retains_projection_group_shapes(rig):
    d, _, _ = rig
    dr = SimpleNamespace(ring=2176, window=2047, tap_in=torch.empty((64, 1)))
    for start in (0, 1, 63, 64, 65, 2047):
        for rows in (1, 63, 64, 65, 1024, 2048):
            for needed_at in (start + rows, start + rows + 2048, start + rows + 4096):
                skipped = d.dead_tap_prefix(start, start + rows, needed_at, dr)
                assert 0 <= skipped <= rows
                assert skipped == rows or skipped % 64 == 0
                lo = max(0, needed_at - dr.window) // 64 * 64
                assert skipped == 0 or start + skipped <= lo


def test_dead_work_options_survive_slices_and_cancel_cleanup(rig, monkeypatch):
    d, make, _ = rig
    monkeypatch.setattr(p1, "DEAD_WORK", True)
    prompt = list(range(521))
    plain, plain_dr = make(129, images=True)
    expected = d.prefill(plain, prompt, None, mtp=False, drafter=plain_dr)
    wanted = state(plain, plain_dr)
    original, options = d.compute, []
    def sliced(w, st, b, ids, *, layer_start=0, layer_stop=None, **kw):
        options.append((st.pos, layer_start, layer_stop, dict(kw)))
        return None if layer_stop == 1 else original(w, st, b, ids, **kw)
    monkeypatch.setattr(d, "compute", sliced)
    e, dr = make(129, images=True)
    e.w.layers = [0, 1]
    stop = 1
    steps = d.prefill_steps(e, prompt, None, mtp=False, drafter=dr, slice_end=lambda: stop)
    while True:
        try:
            progress = next(steps)
        except StopIteration as done:
            assert done.value == expected
            break
        stop = 2 if isinstance(progress, tuple) else 1
        if isinstance(progress, tuple):
            assert not d.prof.active
    same(state(e, dr), wanted)
    for first, second in zip(options[::2], options[1::2]):
        assert first[0] == second[0] and first[3] == second[3]
    assert any(item[3].get("skip_final_ffn") for item in options)
    e, dr = make(129, images=True)
    e.w.layers = [0, 1]
    steps = d.prefill_steps(e, prompt, None, mtp=False, drafter=dr, slice_end=lambda: 1)
    assert next(steps) == (129, 1)
    steps.close()
    assert e.pbuf.overlay is None and not d.prof.active and dr.context_end == 0
