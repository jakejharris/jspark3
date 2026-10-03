"""Async persistence keeps disk format, immutable ownership and resource caps."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_glm_session_disk import state, store
from tensorfold.families.glm5_next.cuda import decode, session_writer
from tensorfold.families.glm5_next.cuda.disk_io import TensorParts, TensorReader, tensor_chunks, write_tensors
from tensorfold.families.glm5_next.cuda.session_state import state_hash


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("n", [3, 4, 5, 301])
def test_async_saved_segments_restore_exact_bits(tmp_path, mtp, n):
    e, d, snap = state(n, mtp=mtp, dflash=not mtp)
    want = state_hash(e, snap)
    decode.save_rows(e, snap, d)
    # Segment across a non-four-aligned boundary, including empty tails.
    snap.rows = [(t[:3], t[3:7], t[7:]) for t in snap.rows]
    disk = store(tmp_path)
    assert disk.submit(e, [snap], snap, stage_bytes=2**20)
    disk.wait_pending()
    assert disk.errors == 0 and len(disk.entries) == 1
    target, drafter, _ = state(n, mtp=mtp, dflash=not mtp, offset=28)
    loaded = disk.load(target, next(iter(disk.entries.values())), drafter)
    assert loaded is not None and state_hash(target, loaded) == want
    assert store(tmp_path).held() == disk.held()  # same format and durable restart scan


def test_segmented_file_and_checksums_match_contiguous(tmp_path, monkeypatch):
    from tensorfold.families.glm5_next.cuda import disk_io
    monkeypatch.setattr(disk_io, "CHUNK", 4096)
    rows = torch.arange(12000, dtype=torch.int32).reshape(2000, 6)[:, ::2]
    parts = TensorParts([rows[:333], rows[333:777], rows[777:]])
    assert list(tensor_chunks(parts)) == list(tensor_chunks(rows))
    assert list(tensor_chunks(parts.slice(325, 780))) == list(tensor_chunks(rows[325:780]))
    write_tensors(tmp_path / "parts", {}, [("data", parts)])
    write_tensors(tmp_path / "plain", {}, [("data", rows)])
    assert (tmp_path / "parts").read_bytes() == (tmp_path / "plain").read_bytes()
    with TensorReader(tmp_path / "parts") as reader:
        assert reader.get("data").equal(rows)


@pytest.fixture
def blocked_writer(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = session_writer.write_tensors
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(session_writer, "write_tensors", blocked)
    yield entered, release
    release.set()


def test_pending_chain_protects_parents_and_reserves_live_budget(tmp_path, blocked_writer, monkeypatch):
    entered, release = blocked_writer
    disk = store(tmp_path)
    e, d, parent = state(65)
    assert disk.put(e, parent)
    old = next(iter(disk.entries.values()))
    e, d, child = state(140)
    decode.save_rows(e, child, d)
    try:
        assert disk.submit(e, [child], child, stage_bytes=2**20)
        assert entered.wait(5)
        disk.budget = disk.held()
        assert not disk.reserve_live(10, 1)
        assert old.key in disk.entries and old.path.exists()
        assert not disk.submit(e, [child], child, stage_bytes=2**20)
        # The filesystem-floor eviction loop must also protect pending parents.
        disk.budget *= 4
        disk.min_free = 1
        monkeypatch.setattr("tensorfold.families.glm5_next.cuda.session_disk.shutil.disk_usage",
                            lambda _: SimpleNamespace(free=1))
        assert not disk.reserve_live(11, 1)
        assert old.path.exists()
    finally:
        release.set()
        disk.wait_pending()
    assert len(disk.entries) == 2 and disk.errors == 0


def test_lost_parent_while_writing_never_publishes_broken_child(tmp_path, blocked_writer):
    entered, release = blocked_writer
    disk = store(tmp_path)
    e, d, parent = state(65)
    disk.put(e, parent)
    old = next(iter(disk.entries.values()))
    e, d, child = state(140)
    decode.save_rows(e, child, d)
    try:
        assert disk.submit(e, [child], child, stage_bytes=2**20)
        assert entered.wait(5)
        disk._drop_tree(old.key)  # a corrupt read can invalidate a protected parent
    finally:
        release.set()
        disk.wait_pending()
    assert not disk.entries and disk.errors == 1
    assert not list(tmp_path.glob("*.session"))


@pytest.mark.parametrize("fail_at", [1, 2])
def test_partial_batch_failure_only_publishes_complete_files(tmp_path, monkeypatch, fail_at):
    disk = store(tmp_path)
    _, _, parent = state(65)
    e, d, child = state(140)
    decode.save_rows(e, child, d)
    original, calls = session_writer.write_tensors, []
    def fail(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == fail_at:
            raise OSError("injected failed write")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(session_writer, "write_tensors", fail)
    assert disk.submit(e, [parent, child], child, stage_bytes=2**20)
    disk.wait_pending()
    assert disk.pending is None and disk.errors == 1
    assert len(disk.entries) == fail_at - 1
    assert len(list(tmp_path.glob("*.session"))) == fail_at - 1
    assert not list(tmp_path.glob("*.part"))


def test_async_fixed_staging_budget_is_binding(tmp_path):
    disk = store(tmp_path)
    e, d, snap = state(140)
    decode.save_rows(e, snap, d)
    assert not disk.submit(e, [snap], snap, stage_bytes=1)
    assert disk.pending is None and not disk.entries


def test_empty_delta_views_still_charge_their_pinned_allocation():
    rows = torch.zeros(100, 3)
    assert sum(session_writer.storages([rows[:0]]).values()) == rows.numel() * rows.element_size()
    assert session_writer.storages([rows, rows[:0]]) == session_writer.storages([rows])


@pytest.mark.parametrize("unlink_fails", [False, True])
def test_failure_after_rename_does_not_lose_disk_accounting(tmp_path, monkeypatch, unlink_fails):
    disk = store(tmp_path)
    e, d, snap = state(140)
    decode.save_rows(e, snap, d)
    original = session_writer.write_tensors
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("injected failure after rename")
    monkeypatch.setattr(session_writer, "write_tensors", fail)
    if unlink_fails:
        unlink = Path.unlink
        def refuse(path, *args, **kwargs):
            if path.suffix == ".session":
                raise OSError("injected unlink failure")
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", refuse)
    assert disk.submit(e, [snap], snap, stage_bytes=2**20)
    reserved = disk.pending.nbytes
    disk.wait_pending()
    assert disk.errors == 1 and not disk.entries
    assert disk.orphan_live_bytes == (reserved if unlink_fails else 0)
    assert bool(list(tmp_path.glob("*.session"))) == unlink_fails


def test_foreground_stops_writer_before_copy_and_between_tiles(tmp_path, monkeypatch):
    from tensorfold.families.glm5_next.cuda import disk_io

    monkeypatch.setattr(disk_io, "CHUNK", 4096)
    disk = store(tmp_path)
    disk.write_gate.quiet_s = 0
    disk.write_gate.set_idle(False)
    e, d, snap = state(301)
    decode.save_rows(e, snap, d)
    paused, copied = threading.Event(), threading.Event()
    copies = []
    wait, cpu = disk.write_gate.wait, torch.Tensor.cpu
    def observe_wait(draining):
        if not disk.write_gate.idle:
            paused.set()
        return wait(draining)
    def transfer(tensor, *args, **kwargs):
        copies.append(tensor.numel() * tensor.element_size())
        if len(copies) == 1:
            disk.write_gate.set_idle(False)  # arrival while this one tile is in flight
            copied.set()
        return cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(disk.write_gate, "wait", observe_wait)
    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    try:
        assert disk.submit(e, [snap], snap, stage_bytes=2**20)
        assert paused.wait(3) and not copies and not list(tmp_path.glob("*.part"))
        paused.clear()
        disk.write_gate.set_idle(True)
        assert copied.wait(3) and paused.wait(3)
        assert len(copies) == 1 and max(copies) <= disk_io.CHUNK
        assert not disk.pending.done.is_set() and not disk.entries
        disk.write_gate.set_idle(True)
        assert disk.pending.done.wait(3)
        disk.poll()
        assert disk.errors == 0 and len(disk.entries) == 1
        assert len(copies) > 1 and max(copies) <= disk_io.CHUNK
    finally:
        disk.wait_pending()


def test_idle_quiet_interval_restarts_after_foreground(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(session_writer, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    gate = session_writer.WriteGate(quiet_s=0.5)
    draining, ready = threading.Event(), threading.Event()
    thread = threading.Thread(target=lambda: (gate.wait(draining), ready.set()), daemon=True)
    thread.start()
    try:
        assert not ready.wait(0.02)
        gate.set_idle(False)
        clock[0] = 12.0
        gate.set_idle(True)
        assert not ready.wait(0.02)  # the old quiet deadline is not reusable
        clock[0] = 12.51
        with gate.condition:
            gate.condition.notify_all()
        assert ready.wait(3)
    finally:
        with gate.condition:
            draining.set()
            gate.condition.notify_all()
        thread.join(3)


def test_offline_drain_bypasses_busy_and_quiet_interval(tmp_path):
    disk = store(tmp_path)
    disk.write_gate.quiet_s = 3600
    disk.write_gate.set_idle(False)
    e, d, snap = state(140)
    decode.save_rows(e, snap, d)
    assert disk.submit(e, [snap], snap, stage_bytes=2**20)
    disk.wait_pending(timeout=3)
    assert disk.pending is None and len(disk.entries) == 1 and disk.errors == 0
    assert not disk.write_gate.idle  # offline drain does not change serving ownership


def test_finished_writer_no_longer_protects_files_from_live_credit(tmp_path):
    disk = store(tmp_path)
    disk.write_gate.quiet_s = 0
    e, d, snap = state(140)
    decode.save_rows(e, snap, d)
    assert disk.submit(e, [snap], snap, stage_bytes=2**20)
    disk.budget = disk.pending.nbytes
    assert disk.pending.done.wait(3)
    # No separate poll/restore: CREDIT itself must reap the completed writer.
    assert disk.reserve_live(7, disk.budget)
    assert disk.pending is None and not disk.entries


def test_simultaneous_http_preparations_keep_writer_paused():
    gate = session_writer.WriteGate(quiet_s=0)
    ready, draining = threading.Event(), threading.Event()
    with gate.prepare():
        with gate.prepare():
            gate.set_idle(False)
            gate.set_idle(True)  # finishing an older decode does not override preparation
            thread = threading.Thread(target=lambda: (gate.wait(draining), ready.set()), daemon=True)
            thread.start()
            assert gate.preparing == 2 and not ready.wait(0.02)
        assert gate.preparing == 1 and not ready.wait(0.02)
    assert ready.wait(3) and gate.preparing == 0
    thread.join(3)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_http_prepare_prioritizes_tokenization_and_releases_on_error(monkeypatch, enabled, failure):
    from tensorfold.cuda.server import App
    from tensorfold.families.glm5_next.cuda.app import GlmApp

    app = GlmApp.__new__(GlmApp)
    gate = session_writer.WriteGate(quiet_s=0)
    sessions = SimpleNamespace(store=SimpleNamespace(write_gate=gate)) if enabled else None
    app.engine = SimpleNamespace(session_cache=sessions)
    prepared = object()
    def prepare(self, body, chat):
        assert self is app and gate.preparing == int(enabled)
        if failure:
            raise ValueError("invalid request")
        return prepared
    monkeypatch.setattr(App, "prepare", prepare)
    if failure:
        with pytest.raises(ValueError, match="invalid request"):
            app.prepare({}, True)
    else:
        assert app.prepare({}, True) is prepared
    assert gate.preparing == 0
