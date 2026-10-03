"""Disk session cache: real CPU tensor bits, O_DIRECT files, and fail-closed controls."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import decode
from tensorfold.families.glm5_next.cuda.disk_io import DirectFile, TensorReader, file_digest, write_tensors
from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig
from tensorfold.families.glm5_next.cuda.session_disk import SessionStore, exact_fingerprint
from tensorfold.families.glm5_next.cuda.session_state import state_hash, target_rows


def state(n=140, *, rank=0, offset=0, mtp=False, dflash=True, images=()):
    def rows(*shape):
        count = int(np.prod(shape))
        # Includes zeros, signed zeros, infinities and distinct BF16 NaN payloads.
        return ((torch.arange(count) * 41 + rank * 7) % 65536).short().view(torch.bfloat16).reshape(shape)

    def cache(width):
        parent = torch.zeros(1024 + offset, width, dtype=torch.bfloat16)
        parent[offset:].copy_(rows(1024, width))
        return parent[offset:]

    st = SimpleNamespace(kc=[cache(3)], vc=[None], rec=rows(2, 3, 4).float(), cur=[1], conv=rows(3, 4),
                         index=[(cache(2), cache(2), cache(2)) for _ in range(2 if mtp else 1)],
                         mtp_kc=cache(3), mtp_vc=None, mtp_len=n - 1 if mtp else 0, mtp_drafted=0, pos=n)
    st.set_pos = lambda pos: setattr(st, "pos", pos)
    st.set_mtp_len = lambda pos: setattr(st, "mtp_len", pos)
    st.rec[1].add_(n)
    st.conv = torch.full((3, 4), n, dtype=torch.bfloat16)
    e = SimpleNamespace(st=st, last_hidden=rows(1, 4))
    d = None
    if dflash:
        d = SimpleNamespace(kc=[rows(3, 512, 4)[:, :256]], vc=[rows(3, 512, 4)[:, :256]],
                            ring=256, window=65, context_end=n, pos_dev=torch.tensor([n]))
        idx = decode._ring_slots(d, n)
        for c in d.kc + d.vc:
            c.index_copy_(1, idx, rows(3, len(idx), 4))
    ids = list(range(n))
    if images:
        ids[1:3] = [-7, -7]
    snap = decode.take_snapshot(e, ids, e.last_hidden if mtp else None, mtp=mtp, drafter=d)
    snap.image_digests = images
    return e, d, snap


def store(tmp_path, **kw):
    return SessionStore(tmp_path, kw.pop("budget", 100 * 2**20), kw.pop("stamp", "test"), min_free=0, **kw)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("n", [3, 4, 5, 65, 140, 301])
@pytest.mark.parametrize("mtp,dflash", [(False, True), (True, False)])
def test_disk_restores_canonical_state_bits_in_nonzero_span(tmp_path, n, rank, mtp, dflash):
    e, d, snap = state(n, rank=rank, mtp=mtp, dflash=dflash)
    want = state_hash(e, snap)
    disk = store(tmp_path)
    assert disk.put(e, snap)
    # Re-open scans/checks identity just as an exact boot does.
    disk = store(tmp_path)
    entry = disk.resume(snap.ids + [999], mtp=mtp, dflash=dflash)
    dest, drafter, _ = state(n, rank=rank, offset=28, mtp=mtp, dflash=dflash)
    for _, t in target_rows(dest.st, n, max(snap.mtp_len, 0)):
        t.zero_()
    restored = disk.load(dest, entry, drafter)
    assert restored is not None
    decode.restore(dest, restored, drafter)
    fresh = decode.take_snapshot(dest, snap.ids, restored.pending, mtp=mtp, drafter=drafter)
    assert state_hash(dest, fresh) == want
    assert dest.st.pos == n and dest.st.mtp_drafted == 0


def test_non_four_aligned_parent_child_share_only_completed_rows(tmp_path):
    disk = store(tmp_path)
    entries = []
    for n in (3, 11, 17):
        e, d, snap = state(n)
        assert disk.put(e, snap)
        entry = disk.resume(snap.ids + [88], mtp=False, dflash=True)
        entries.append(entry)
        with TensorReader(entry.path) as r:
            assert r.meta["starts"]["pk0"] == (len(entries[-2].ids) // 4 if len(entries) > 1 else 0)
            assert r.layout["kc0"]["shape"][0] == n - (len(entries[-2].ids) if len(entries) > 1 else 0)
        for old in entries:
            restored = disk.load(e, old, d)
            reference, _, original = state(len(old.ids))
            assert state_hash(e, restored) == state_hash(reference, original)
    assert len(disk.entries) == 3 and sum(e.parent is None for e in disk.entries.values()) == 1


def test_fence_scratch_is_ignored_but_mixed_parity_state_is_detected():
    e, _, snap = state(7)
    want = state_hash(e, snap)
    for _, _, pk in e.st.index:
        pk[7 // 4:].fill_(float("nan"))
    assert state_hash(e, snap) == want
    snap.rec[0, 0] += 1
    assert state_hash(e, snap) != want                 # Mixed completed/incomplete KDA state


def test_full_image_identity_prevents_placeholder_collision(tmp_path):
    disk = store(tmp_path)
    e, d, snap = state(140, images=(b"a" * 32,))
    assert disk.put(e, snap)
    same = disk.resume(snap.ids + [1], mtp=False, dflash=True, image_digests=(b"a" * 32,))
    assert same is not None
    assert disk.resume(snap.ids + [1], mtp=False, dflash=True, image_digests=(b"b" * 32,)) is None
    assert disk.resume(snap.ids + [1], mtp=False, dflash=True) is None
    restored = disk.load(e, same, d)
    assert restored.image_digests == snap.image_digests
    assert state_hash(e, restored) == state_hash(e, snap)


@pytest.mark.parametrize("fault", ["tensor_byte", "missing_ring", "truncated", "header_byte", "parent"])
def test_corrupt_or_incomplete_snapshots_are_misses(tmp_path, fault):
    disk = store(tmp_path)
    e, d, parent = state(65)
    disk.put(e, parent)
    old = next(iter(disk.entries.values()))
    e, d, snap = state(140)
    disk.put(e, snap)
    entry = disk.resume(snap.ids + [1], mtp=False, dflash=True)
    if fault == "parent":
        old.path.unlink()
    elif fault == "missing_ring":
        with TensorReader(entry.path) as r:
            meta = dict(r.meta)
            tensors = [(name, r.get(name)) for name in r.layout if name != "draft0"]
        write_tensors(entry.path, meta, tensors)  # valid checksum, incomplete semantic state
    else:
        with open(entry.path, "r+b") as f:
            if fault == "truncated":
                f.truncate(4096)
            else:
                with TensorReader(entry.path) as r:
                    at = r.base + r.layout["draft0"]["offset"] if fault == "tensor_byte" else 80
                f.seek(at)
                value = f.read(1)[0]
                f.seek(at)
                f.write(bytes([value ^ 1]))
    assert disk.load(e, entry, d) is None
    assert entry.key not in disk.entries
    assert disk.errors > 0
    if fault == "parent":
        assert not store(tmp_path).entries


def test_bad_parent_chain_and_oversized_chain_never_exceed_cap(tmp_path):
    disk = store(tmp_path)
    e, d, parent = state(65)
    disk.put(e, parent)
    held = disk.held()
    disk.budget = held
    e, d, child = state(140)
    assert not disk.put(e, child)
    assert disk.held() == held <= disk.budget
    assert len(disk.entries) == 1
    # A cycle masquerading as a valid, checksummed file is removed at scan.
    entry = next(iter(disk.entries.values()))
    with TensorReader(entry.path) as r:
        meta, tensors = dict(r.meta), [(name, r.get(name)) for name in r.layout]
    meta["parent"] = entry.key
    write_tensors(entry.path, meta, tensors)
    assert not store(tmp_path).entries


def test_lru_cap_applies_on_write_and_restart_and_disk_floor(tmp_path, monkeypatch):
    disk = store(tmp_path)
    e, d, snap = state(65)
    disk.put(e, snap)
    cap = disk.held() * 2 + 4096
    disk.budget = cap
    for i in range(12):
        snap.ids[0] = 1000 + i
        assert disk.put(e, snap)
        assert disk.held() <= cap
    assert len(disk.entries) <= 2
    assert store(tmp_path, budget=cap // 2).held() <= cap // 2
    disk.min_free = 150 * 2**30
    monkeypatch.setattr("tensorfold.families.glm5_next.cuda.session_disk.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=disk.min_free))
    snap.ids[0] = 55555
    assert not disk.put(e, snap)


def test_write_failure_keeps_existing_chain_and_no_partial_file(tmp_path, monkeypatch):
    disk = store(tmp_path)
    e, d, snap = state(65)
    disk.put(e, snap)
    before = set(disk.entries)
    def fail(*a, **kw):
        raise OSError("injected short write")
    monkeypatch.setattr(DirectFile, "write", fail)
    e, d, snap = state(140)
    assert not disk.put(e, snap)
    assert set(disk.entries) == before and not list(tmp_path.glob("*.part"))


def checkpoint_files(path):
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"glm5_next"}')
    (path / "model.safetensors").write_bytes(b"original tensor bytes" * 1000)
    return path


def test_exact_boot_identity_uses_weight_bytes_not_source_or_timestamp(tmp_path):
    path = checkpoint_files(tmp_path / "model")
    opts = {"latent": True, "drafter_bits": 4, "drafter_window": 2048, "layout": 2}
    def stamp(**kw):
        return exact_fingerprint(path, kw.pop("rank", 0), kw.pop("world", 3),
                                 math_version=kw.pop("math_version", "E:fixture-baseline"), options=kw.pop("options", opts),
                                 **kw)
    original = stamp()
    (path / "engine.py").write_text("# reviewed exact source change")
    assert stamp() == original
    assert stamp(rank=1) != original and stamp(world=2) != original
    assert stamp(math_version="E:new-math") != original
    assert stamp(math_version="S:new-math") != stamp(math_version="S:new-math")
    for key in opts:
        assert stamp(options={**opts, key: "changed"}) != original
    (path / "model.safetensors").write_bytes(b"modified tensor bytes" * 1000)
    assert stamp() != original
    draft = checkpoint_files(tmp_path / "draft")
    drafted = stamp(drafter=draft)
    (draft / "model.safetensors").write_bytes(b"different drafter")
    assert stamp(drafter=draft) != drafted


@pytest.mark.parametrize("math_version", ["", "fixture-baseline", "E:", "S:"])
def test_identity_requires_explicit_math_contract(tmp_path, math_version):
    with pytest.raises(ValueError, match="MATH_VERSION"):
        exact_fingerprint(tmp_path, 0, 3, math_version=math_version, options={})


def test_mismatched_stamp_or_schema_cannot_load(tmp_path):
    e, d, snap = state()
    disk = store(tmp_path)
    disk.put(e, snap)
    assert not store(tmp_path, stamp="different actual weights/math/layout").entries
    disk.put(e, replace(snap, ids=[77] + snap.ids[1:]))
    entry = list(disk.entries.values())[-1]
    with TensorReader(entry.path) as r:
        meta, tensors = dict(r.meta), [(name, r.get(name)) for name in r.layout]
    meta["format"] = 999
    write_tensors(entry.path, meta, tensors)
    assert not store(tmp_path).entries


def test_direct_io_has_no_buffered_fallback_and_hashes_unaligned_tail(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"123" * 7001)
    import hashlib
    assert file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    import os
    original, seen = os.open, []
    def record(path, flags, *a):
        seen.append(flags)
        return original(path, flags, *a)
    monkeypatch.setattr(os, "open", record)
    file_digest(path)
    assert seen and all(flags & os.O_DIRECT for flags in seen)
    def fail(*a, **kw):
        raise OSError("O_DIRECT unsupported")
    monkeypatch.setattr(os, "open", fail)
    with pytest.raises(OSError, match="unsupported"):
        file_digest(path)


def test_staging_is_globally_bounded_and_cancel_keeps_only_finished_chunks(tmp_path):
    e, d, snap = state()
    cap = decode.snapshot_bytes(snap) * 2
    owner = SimpleNamespace(checkpoint_after=lambda pos: (pos // 16 + 1) * 16)
    cache = SessionCache(owner, SessionConfig(disk=True, cancel=True, stage_bytes=cap), store(tmp_path))
    s = SimpleNamespace(sid=1, image_digests=(), session_boundaries=(), prompt=list(range(500)),
                        fill_pos=140, engine=e)
    for sid in range(8):
        s.sid = sid
        cache.stage(s, snap)
        assert cache.held_bytes() <= cap
    assert cache.dropped > 0
    s.sid = 0  # the oldest stream's latest checkpoint was protected under pressure
    cache.finish(s)
    assert cache.store.resume(s.prompt, mtp=False, dflash=True) is not None
    assert all(sid != s.sid for sid, _, _ in cache.staged)


def test_rank_one_failed_load_forces_identical_cold_resume_decision(tmp_path):
    e, d, snap = state()
    disk = store(tmp_path)
    disk.put(e, snap)
    entry = disk.resume(snap.ids + [1], mtp=False, dflash=True)
    for rank in range(3):
        owner = SimpleNamespace(_gather_ints=lambda mine: [[1], [0], [1]])
        cache = SessionCache(owner, SessionConfig(disk=True), disk)
        s = SimpleNamespace(engine=e, drafter=d)
        assert cache.restore(s, None if rank == 1 else entry.stub()) is None


def test_flags_default_off_and_require_dependencies(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("TF_GLM_SESSION_"):
            monkeypatch.delenv(key)
    cfg = SessionConfig.from_env(8)
    assert not any((cfg.disk, cfg.cancel, cfg.checkpoints, cfg.hash_gate)) and cfg.reserve == 0
    monkeypatch.setenv("TF_GLM_SESSION_CANCEL", "1")
    with pytest.raises(ValueError, match="require"):
        SessionConfig.from_env(8)


def test_pending_mtp_row_cannot_be_omitted_or_mislabeled(tmp_path):
    disk = store(tmp_path)
    e, d, snap = state(140, mtp=True, dflash=False)
    assert disk.put(e, snap)
    entry = disk.resume(snap.ids + [1], mtp=True, dflash=False)
    with TensorReader(entry.path) as r:
        meta = dict(r.meta)
        tensors = [(name, r.get(name)) for name in r.layout if name != "pending"]
    write_tensors(entry.path, meta, tensors)
    assert disk.load(e, entry) is None


def test_content_hash_detects_same_length_same_mtime_weight_change(tmp_path):
    import os
    path = checkpoint_files(tmp_path / "model")
    weights = path / "model.safetensors"
    before = weights.stat()
    stamp = exact_fingerprint(path, 0, 3, math_version="E:v1", options={})
    content = bytearray(weights.read_bytes())
    content[100] ^= 1
    weights.write_bytes(content)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert weights.stat().st_size == before.st_size and weights.stat().st_mtime_ns == before.st_mtime_ns
    assert exact_fingerprint(path, 0, 3, math_version="E:v1", options={}) != stamp


def test_long_prefill_cannot_evict_another_active_streams_latest_checkpoint(tmp_path):
    e, d, snap = state(140)
    cap = decode.snapshot_bytes(snap) * 3
    owner = SimpleNamespace(checkpoint_after=lambda pos: (pos // 4 + 1) * 4)
    cache = SessionCache(owner, SessionConfig(disk=True, checkpoints=True, stage_bytes=cap), store(tmp_path))
    a = SimpleNamespace(sid=0, image_digests=(), session_boundaries=())
    b = SimpleNamespace(sid=1, image_digests=(), session_boundaries=())
    cache.stage(a, snap, final=True)
    for n in (144, 148, 152, 156, 160):
        _, _, anchor = state(n)
        cache.stage(b, anchor)
        assert cache.held_bytes() <= cap
        assert any(sid == 0 and saved is snap for sid, saved, _ in cache.staged)
        assert any(sid == 1 and len(saved.ids) == n for sid, saved, _ in cache.staged)
    assert cache.dropped > 0


def test_startup_refuses_staging_budget_too_small_for_active_slots(tmp_path, monkeypatch):
    from tensorfold.families.glm5_next.cuda.session_cache import configure
    e, d, _ = state()
    owner = SimpleNamespace(e=e, drafter=d, w=SimpleNamespace(cfg=SimpleNamespace(hidden=4)),
                            _gather_ints=lambda v: [v, v, v])
    with pytest.raises(RuntimeError, match="initialize") as caught:
        configure(owner, SessionConfig(disk=True, stage_bytes=1), tmp_path, None, {}, slots=8)
    assert "one completed checkpoint" in str(caught.value.__cause__)


@pytest.mark.parametrize("end", [5, 8, 11])
def test_transferred_pool_fence_must_be_refreshed_without_a_full_prefix_copy(end):
    from tensorfold.families.glm5_next.cuda.session_state import refresh_transferred_pool_fences

    def extend(fix):
        e, d, old = state(end)
        for _, _, pk in e.st.index:
            pk[end // 4].fill_(-31)       # a parent snapshot includes this unwritten fence
        decode.save_rows(e, old, d)
        first = old.rows[0]
        # Simulate the next prefill completing the parent's formerly partial pool.
        for _, _, pk in e.st.index:
            pk[end // 4].fill_(42)
        d.context_end = end + 5
        new = decode.take_snapshot(e, list(range(end + 5)), None, mtp=False, drafter=d)
        expected = state_hash(e, new)
        if fix:
            refresh_transferred_pool_fences(e, old, d)
        decode.extend_rows(e, old, new, d)
        assert new.rows[0][0] is first   # preserve target prefix ownership/tail-only copy
        for _, t in target_rows(e.st, end + 5, 0):
            t.zero_()
        decode.load_rows(e, new, d)
        return state_hash(e, new) == expected
    assert not extend(False)             # reproduces fixture-baseline's stale pool fence
    assert extend(True)
