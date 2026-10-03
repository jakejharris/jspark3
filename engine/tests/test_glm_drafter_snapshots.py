"""Real snapshot copies on CPU tensors: DFlash2 bits, fit rules, per-rank budgets and eviction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import decode  # noqa: E402
from tensorfold.families.glm5_next.cuda.dflash2 import rank_heads  # noqa: E402
from tensorfold.families.glm5_next.cuda.engine import GlmEngine, encode_policy  # noqa: E402


def _engine(n=9, *, rank=0, world=3, mtp=True, dflash=True):
    def rows(*shape):
        size = 1
        for dim in shape:
            size *= dim
        return (torch.arange(size).reshape(shape) + rank * 100).to(torch.bfloat16)

    st = SimpleNamespace(rec=rows(2, 3, 4).float(), cur=[1], conv=rows(3, 4),
                         kc=[rows(32, 3)], vc=[rows(32, 3)],
                         index=[(rows(32, 2), rows(32, 2), rows(9, 2)) for _ in range(2)],
                         mtp_kc=rows(32, 3), mtp_vc=rows(32, 3), mtp_len=n - 1, mtp_drafted=0)
    st.set_pos = lambda pos: setattr(st, "pos", pos)
    st.set_mtp_len = lambda pos: setattr(st, "mtp_len", pos)
    _, kvh = rank_heads(32, 8, world)
    d = SimpleNamespace(kc=[rows(kvh, 32, 4) for _ in range(2)],
                        vc=[rows(kvh, 32, 4) for _ in range(2)],
                        context_end=n, pos_dev=torch.tensor([n]), window=3)
    e = GlmEngine.__new__(GlmEngine)
    e.e, e.drafter = SimpleNamespace(st=st), d
    e.live, e.cache, e.cache_bytes, e.cache_entries = list(range(n)), [], 10**6, 8
    snap = decode.take_snapshot(e.e, e.live, rows(1, 4) if mtp else None,
                                mtp=mtp, drafter=d if dflash else None)
    e.cache.append(snap)
    return e, snap


def _bits(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("n", [1, 3, 4, 9])
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_committed_drafter_prefix_round_trips_bit_for_bit(n, world, rank):
    e, snap = _engine(n, rank=rank, world=world)
    d = e.drafter
    # Include signed zero, infinities and distinct NaN payloads, even outside the attention window.
    d.vc[0][0, 0] = torch.tensor([-32768, 32640, 32641, 32704], dtype=torch.int16).view(torch.bfloat16)
    want = [t[:, :n].clone() for t in (*d.kc, *d.vc)]
    target = [t.clone() for t in decode._row_views(e.e.st, n, n - 1)]
    base = decode.snapshot_bytes(snap)
    need = decode.row_bytes(e.e, snap, d)
    # Decode may already have advanced past this prompt's snapshot. Save its end, not the live end.
    d.context_end = n + 5
    d.pos_dev.fill_(n + 5)
    e._take_over([])
    assert snap.drafter_end == n and snap.rows is not None
    target_bytes = sum(t.numel() * t.element_size() for t in target)
    assert snap.nbytes == need == target_bytes + 2 * 2 * d.kc[0].shape[0] * n * 4 * 2
    assert e._held_bytes() == base + need
    # B overwrites the entire live allocation; the saved tensors must not alias it.
    for t in (*e.e.st.kc, *e.e.st.vc, e.e.st.mtp_kc, e.e.st.mtp_vc,
              *(t for group in e.e.st.index for t in group), *d.kc, *d.vc):
        t.fill_(17)
    e.e.st.rec.zero_()
    e.e.st.conv.zero_()
    d.context_end = 2
    d.pos_dev.fill_(2)
    assert e._resume(snap.ids + [77], encode_policy("fc7:0.3")) is snap
    decode.load_rows(e.e, snap, d)
    decode.restore(e.e, snap, d)
    for got, expected in zip((*d.kc, *d.vc), want, strict=True):
        assert torch.equal(_bits(got[:, :n]), _bits(expected))
        assert (got[:, n:] == 17).all()              # scratch past the end is not saved or restored
    for got, expected in zip(decode._row_views(e.e.st, n, n - 1), target, strict=True):
        assert torch.equal(_bits(got), _bits(expected))
    assert d.context_end == n and d.pos_dev.tolist() == [n]
    assert e.e.st.pos == n and e.e.st.mtp_len == n - 1 and e.e.st.mtp_drafted == 0
    assert torch.equal(_bits(e.e.st.rec[1]), _bits(snap.rec))
    assert torch.equal(_bits(e.e.st.conv), _bits(snap.conv))
    snap.rows, snap.nbytes = None, 0                 # _run's transition back to live rows
    assert e._held_bytes() == base


@pytest.mark.parametrize("policy,mtp,dflash,fits", [
    ("fc7:0.3", False, True, True), ("fc7:0.3", True, False, False),
    ("auto", True, True, True), ("auto", False, True, False), ("auto", True, False, False),
    ("2", True, False, True), ("2", False, True, False),
])
def test_saved_snapshot_fits_exactly_its_available_drafters(policy, mtp, dflash, fits):
    e, snap = _engine(mtp=mtp, dflash=dflash)
    e._take_over([])
    code = encode_policy(policy)
    assert (e._resume(snap.ids + [99], code) is snap) == fits
    assert e._resume(snap.ids, code) is None          # only strict prefixes
    assert e._resume([99] + snap.ids, code) is None


def test_resume_uses_longest_fitting_snapshot():
    e, short = _engine()
    _, long = _engine(12)
    e.cache.append(long)
    prompt = long.ids + [99]
    assert e._resume(prompt, encode_policy("fc7:0.3")) is long
    long.drafter_end = -1
    assert e._resume(prompt, encode_policy("fc7:0.3")) is short


@pytest.mark.parametrize("operation", [decode.row_bytes, decode.save_rows, decode.load_rows])
def test_drafter_rows_cannot_be_silently_omitted(operation):
    e, snap = _engine()
    with pytest.raises(ValueError, match="needs its drafter"):
        operation(e.e, snap)


def test_target_only_snapshot_needs_no_drafter():
    e, snap = _engine(dflash=False)
    need = decode.row_bytes(e.e, snap)
    decode.save_rows(e.e, snap)
    decode.load_rows(e.e, snap)
    assert snap.drafter_end == -1 and snap.nbytes == need


def test_ring_snapshot_restores_only_bounded_window_bits():
    e, _ = _engine(9)
    d = e.drafter
    d.ring, d.window = 128, 65
    d.kc = [torch.arange(2 * 128 * 4, dtype=torch.int16).view(2, 128, 4).view(torch.bfloat16)]
    d.vc = [d.kc[0].clone()]
    n = 140
    slots = decode._ring_slots(d, n)
    assert 0 < len(slots) <= d.ring and len(slots) < n
    want = decode._ring_window(d, n)
    d.kc[0].zero_()
    d.vc[0].zero_()
    decode._put_ring_window(d, n, want)
    for got, expected in zip(decode._ring_window(d, n), want, strict=True):
        assert torch.equal(_bits(got), _bits(expected))
    assert decode.ring_bytes(d, n) == sum(t.numel() * t.element_size() for t in want)


def test_extending_snapshot_transfers_prefix_rows_without_second_full_copy():
    e, old = _engine(5, dflash=False)
    decode.save_rows(e.e, old)
    first = old.rows[0]
    old_bytes = old.nbytes
    new = decode.take_snapshot(e.e, list(range(9)), torch.zeros(1, 4), mtp=True)
    expected = [v.clone() for v in decode._snapshot_row_views(e.e, new)]
    decode.extend_rows(e.e, old, new)
    assert old.rows is None and old.nbytes == 0
    assert new.rows[0][0] is first
    assert new.nbytes == decode.row_bytes(e.e, new)
    assert new.rows[0][-1].numel() < first.numel()
    for view in decode._snapshot_row_views(e.e, new):
        view.zero_()
    decode.load_rows(e.e, new)
    for got, want in zip(decode._snapshot_row_views(e.e, new), expected, strict=True):
        assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("spare", [-1, 0])
def test_all_three_ranks_include_drafter_rows_in_budget_before_copying(spare):
    outcomes = []
    for rank in range(3):
        e, snap = _engine(rank=rank)
        e.cache_bytes = e._held_bytes() + decode.row_bytes(e.e, snap, e.drafter) + spare
        e._take_over([])
        outcomes.append((snap in e.cache, snap.nbytes, e._held_bytes()))
        assert e._held_bytes() <= e.cache_bytes
        assert (snap.rows is not None) == (spare == 0)
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_take_over_frees_drafter_rows_and_protects_resume_point():
    e, hit = _engine()
    e._take_over([])
    hit_bytes = e._held_bytes()
    _, live = _engine()
    live.ids = list(range(100, 109))
    e.cache.append(live)
    e.live = live.ids
    e.cache_bytes = hit_bytes + decode.snapshot_bytes(live)
    e._take_over(hit.ids)
    assert e.cache == [hit] and hit.rows is not None and live.rows is None
    e._drop(hit)
    assert hit.rows is None and hit.nbytes == 0 and e._held_bytes() == 0


def test_remember_evicts_saved_drafter_rows_on_entry_limit_and_replacement():
    e, old = _engine()
    e._take_over([])
    _, new = _engine(12)
    e.cache_entries = 1
    e._remember(new)
    assert e.cache == [new] and old.rows is None and old.nbytes == 0
    decode.save_rows(e.e, new, e.drafter)
    _, again = _engine(12)
    e._remember(again)
    assert e.cache == [again] and new.rows is None and new.nbytes == 0


@pytest.mark.parametrize("dflash", [True, False])
def test_disk_target_copy_only_replaces_target_only_device_snapshot(dflash):
    e, snap = _engine(dflash=dflash)
    e.disk = SimpleNamespace(has=lambda ids: True)
    e._take_over([])
    assert (snap in e.cache) == dflash
    assert (snap.rows is not None) == dflash
