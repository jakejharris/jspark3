"""The decode observer must not wait on CUDA/disk, change draft choices, or retain live tap buffers."""

import importlib
import json
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.glm5_next.cuda.decode_observer import DecodeObserver  # noqa: E402
from test_cuda_geometry import allocations  # noqa: E402, F401


@pytest.fixture
def observer(monkeypatch, tmp_path):
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_GLM_DECODE_TAP_EVERY", "1")
    d = DecodeObserver(0)
    yield d
    d.close()


def record(d, taps=None):
    d.begin_round()
    for _ in range(5):
        d.clock()
    d.record({"ms": dict.fromkeys(("forward", "sampling", "commit_taps", "drafter"), 1)}, taps, [])


def test_cuda_events_wait_only_in_writer(monkeypatch, tmp_path):
    serving = threading.get_ident()
    waits = []

    class Event:
        def __init__(self, **kw):
            pass
        def record(self):
            assert threading.get_ident() == serving
        def synchronize(self):
            assert threading.get_ident() != serving
            waits.append(self)
        def elapsed_time(self, other):
            assert threading.get_ident() != serving
            return 1.25

    def forbidden(*args, **kw):
        pytest.fail("The decode observer invoked the synchronizing memory profiler")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    for name in ("synchronize", "reset_peak_memory_stats", "memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    d = DecodeObserver(0)
    prof = importlib.import_module("tensorfold.families.glm5_next.cuda.prof")
    try:
        with d.phase("decode"):
            d.begin_round()
            d.clock()
            monkeypatch.setattr(prof, "decode_timer", d.block)
            with prof.timed("moe: gate/up"):
                with prof.timed("moe: all-gather"):
                    pass
            for _ in range(4):
                d.clock()
            d.record({"ms": {"forward": 99}}, None, [])
        d.flush()
        row = json.loads(d.file.read_text())
        assert row["ms"]["forward"] == 1.25
        assert row["host_ms"]["forward"] == 99
        assert row["forward_blocks_ms"] == {"moe: gate/up": 1.25, "moe: all-gather": 1.25}
        assert len(waits) == 2
        assert json.loads(d.phase_file.read_text())["cuda_stream_ms"] == 1.25
    finally:
        d.close()


def test_writer_backpressure_drops_without_waiting_or_retaining_live_taps(observer, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    save = torch.save
    def slow_save(*args, **kw):
        entered.set()
        assert release.wait(5)
        return save(*args, **kw)
    monkeypatch.setattr(torch, "save", slow_save)
    taps = [torch.arange(32, dtype=torch.int16).reshape(8, 4)]
    expected = taps[0][:3].clone()
    try:
        record(observer, observer.capture(taps, 3))
        assert entered.wait(5)
        taps[0].fill_(99)
        record(observer, observer.capture(taps, 3))
        assert observer.capture(taps, 3) is None
        assert observer.tap_dropped == 1
        for _ in range(256):
            record(observer)
        assert observer.dropped > 0
        assert observer._jobs.qsize() <= 127
    finally:
        release.set()
    observer.flush()
    payload = torch.load(observer.root / "rank0-round0.pt", weights_only=False)
    assert torch.equal(payload["taps"][0], expected)
    assert payload["taps"][0].untyped_storage().nbytes() == expected.numel() * expected.element_size()
    assert observer._tap_slots.qsize() == 2
    record(observer)
    observer.flush()
    last = json.loads(observer.file.read_text().splitlines()[-1])
    assert last["observer_dropped"] > 0 and last["tap_dropped"] == 1


def test_unwritable_or_existing_receipts_disable_observer_without_overwrite(monkeypatch, tmp_path):
    target = tmp_path / "rank0.jsonl"
    target.write_text("existing receipts\n")
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    d = DecodeObserver(0)
    assert not d.enabled and "exist" in d.error.lower()
    assert target.read_text() == "existing receipts\n"
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(target))
    d = DecodeObserver(0)
    assert not d.enabled


def test_close_drains_last_round(observer):
    record(observer)
    observer.close()
    assert json.loads(observer.file.read_text())["round"] == 0


@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("confidence", [0., .01, .3, .9])
def test_real_dflash_selector_observation_preserves_every_draft(sampled, confidence):
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.dflash2")
    from tensorfold.engine.exact_sampling import Sampling
    rng = np.random.default_rng(31)
    d = mod.Drafter.__new__(mod.Drafter)
    d.pred = rng.normal(size=(128, 8)).astype(np.float32)
    d.succ = rng.normal(size=(128, 8)).astype(np.float32)
    tokens = np.vstack([rng.choice(128, 32, replace=False) for _ in range(7)])
    values = rng.normal(size=(7, 32))
    proj = rng.normal(size=(7, 8))
    saved = [x.copy() for x in (tokens, values, proj, d.pred, d.succ)]
    sampling = Sampling(417, .7, 0, 1.) if sampled else None
    for anchor in (0, 11, 127):
        for first in (1, 4095, 14379):
            d.observe = False
            plain = d.chain(tokens, values, proj, anchor, first, sampling, confidence)
            d.observe = True
            shown = d.chain(tokens, values, proj, anchor, first, sampling, confidence)
            assert shown == plain
            assert len(d.last_observation["claims"]) >= len(shown)
    for before, after in zip(saved, (tokens, values, proj, d.pred, d.succ)):
        assert np.array_equal(before, after)


def test_capture_storage_survives_disable_before_record(observer):
    import weakref
    capture = observer.capture([torch.ones(8, 4)], 3)
    stored = weakref.ref(capture[0][0])
    observer.enabled = False
    del capture
    assert stored() is not None  # queued D2H still owns its CPU destination
