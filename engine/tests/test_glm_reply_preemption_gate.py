"""Forced-timing receipt gates reject a missed rendezvous and state/output drift."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder  # noqa: F401

spec = importlib.util.spec_from_file_location("preemption_gate", Path(__file__).parents[1] / "tools/check_glm_reply_preemption.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def reply():
    ids = [1, 2, 3]
    return {"tensorfold": {"token_ids": ids, "token_sha": hashlib.sha256(b"1,2,3").hexdigest()[:12],
                           "session_state_sha256": [str(i) * 64 for i in range(3)],
                           "cached": 14378, "session_cache_source": "memory", "reply_prefill_hit_tokens": 0,
                           "policy": "fc7:0.3", "drafts": True}}


def rendezvous():
    ready = dict(trial="000", selectors=[1, 0, 0], layer=1, base=14378, candidate=14380,
                 cached=14378, fill_pos=14378, already_cancelled=False)
    return ready, dict(ready, cancelled=True)


def test_valid_forced_case():
    ready, released = rendezvous()
    assert gate.check_case(reply(), reply(), ready, released, base=14378, candidate=14380)["tokens"] == 3


@pytest.mark.parametrize("fault", ["not_mixed", "not_cancelled", "early_cancel", "wrong_trial", "wrong_base",
                                  "no_slice", "completed", "disk", "no_state", "short_ids", "policy"])
def test_missed_or_invalid_forced_case_cannot_pass(fault):
    actual = reply()
    ready, released = rendezvous()
    stats = actual["tensorfold"]
    if fault == "not_mixed":
        ready["selectors"] = released["selectors"] = [1, 1, 1]
    elif fault == "not_cancelled":
        released["cancelled"] = False
    elif fault == "early_cancel":
        ready["already_cancelled"] = True
    elif fault == "wrong_trial":
        released["trial"] = "001"
    elif fault == "wrong_base":
        ready["base"] += 1
    elif fault == "no_slice":
        ready["layer"] = 0
    elif fault == "completed":
        stats["cached"], stats["reply_prefill_hit_tokens"] = 14380, 2
    elif fault == "disk":
        stats["session_cache_source"] = "disk"
    elif fault == "no_state":
        stats.pop("session_state_sha256")
    elif fault == "short_ids":
        stats["token_ids"].pop()
    else:
        stats["policy"] = "fq7:0.3"
    with pytest.raises(ValueError) as failure:
        gate.check_case(reply(), actual, ready, released, base=14378, candidate=14380)
    assert not isinstance(failure.value, gate.GateMismatch)  # invalid timing/receipt, not a valid numerical failure


@pytest.mark.parametrize("fault", ["state", "tokens"])
def test_valid_repro_with_drift_is_failure_not_invalid(fault):
    actual = reply()
    if fault == "state":
        actual["tensorfold"]["session_state_sha256"][1] = "f" * 64
    else:
        actual["tensorfold"]["token_ids"][-1] = 4
        actual["tensorfold"]["token_sha"] = hashlib.sha256(b"1,2,4").hexdigest()[:12]
    with pytest.raises(gate.GateMismatch):
        gate.check_case(reply(), actual, *rendezvous(), base=14378, candidate=14380)


def test_hook_waits_at_actual_slice_until_arrival(setup_decoder, monkeypatch, tmp_path):
    mod, _ = setup_decoder
    monkeypatch.setattr(mod.BatchedDecoder, "_fill", lambda *a, **k: None)
    gate.write_json(tmp_path / "arm.json", dict(trial="000", base=14378, candidate=14380))
    event = threading.Event()
    stream = SimpleNamespace(reply_prefill=True, done=False, fill_layer=1, st=SimpleNamespace(cur=[1, 0, 0]),
                             reply_base=14378, prompt=[0] * 14380, fill_pos=14378, cached=14378, reply_cancel=event)
    decoder = SimpleNamespace(rank=0, streams={2: stream})
    # install_hook mutates the class; explicitly register undo with monkeypatch.
    original = mod.BatchedDecoder._fill
    gate.install_hook(tmp_path, timeout=3)
    wrapped = mod.BatchedDecoder._fill
    monkeypatch.setattr(mod.BatchedDecoder, "_fill", original)
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (wrapped(decoder, 2, 14380, 1), finished.set()))
    thread.start()
    try:
        ready = gate.wait_json(tmp_path / "000-ready.json", 2)
        assert ready["selectors"] == [1, 0, 0] and not ready["already_cancelled"]
        assert not finished.is_set()
    finally:
        event.set()
        thread.join(3)
    assert finished.is_set()
    assert gate.wait_json(tmp_path / "000-released.json", 1)["cancelled"]


def test_twenty_trial_capture_and_dry_run_are_separate(tmp_path, monkeypatch):
    prefix, warm, control_dir, out = [tmp_path / name for name in ("prefix.json", "warm.json", "gate", "live")]
    control_dir.mkdir()
    for path in (prefix, warm):
        path.write_text(json.dumps({"request": {"model": "fixture", "messages": [], "max_tokens": 16}}))
    args = SimpleNamespace(prefix=prefix, warm=warm, gate_dir=control_dir, out=out, repeats=20,
                           base_tokens=14378, candidate_tokens=14380, min_layer=1,
                           expected_token_sha=reply()["tensorfold"]["token_sha"], dry_run=True, base="http://fixture")
    requests = []
    def urlopen(request, **kwargs):
        if isinstance(request, str):
            return io.BytesIO(f"tensorfold:requests_total {len(requests)}\nprocess_start_time_seconds 5\n".encode())
        body = json.loads(request.data)
        requests.append(body)
        result = reply()
        if body["tf_reply_prefill"]:
            arm = json.loads((control_dir / "arm.json").read_text())
            ready, released = rendezvous()
            ready["trial"] = released["trial"] = arm["trial"]
            gate.publish_json(control_dir / f"{arm['trial']}-ready.json", ready)
            gate.publish_json(control_dir / f"{arm['trial']}-released.json", released)
            result["tensorfold"]["reply_prefill"] = dict(status="queued", candidate_tokens=14380)
        return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(gate.urllib.request, "urlopen", urlopen)
    gate.capture(args)
    assert not requests and not out.exists() and not list(control_dir.iterdir())
    args.dry_run = False
    gate.capture(args)
    assert len(requests) == 42
    result = json.loads((out / "result.json").read_text())
    assert result["pass"] and len(result["repetitions"]) == 20
    assert not (out / "INVALID.md").exists() and not (out / "FAIL.md").exists()
    with pytest.raises(ValueError, match="empty"):
        gate.capture(args)  # live receipts can never be overwritten by a rerun
