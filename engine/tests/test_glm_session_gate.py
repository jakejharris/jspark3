"""Negative controls for the offline TP3 restored/fresh gate."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("session_gate", Path(__file__).parents[1] / "tools/check_glm_session_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def receipts():
    stats = {"session_state_sha256": [str(rank) * 64 for rank in range(3)], "token_ids": [1, 2, 3],
             "cached": 0, "session_cache_source": "cold", "policy": "fc7:0.3", "drafts": True}
    cold = {"tensorfold": stats}
    warm = copy.deepcopy(cold)
    warm["tensorfold"].update(cached=90000, session_cache_source="disk")
    return cold, warm


def test_gate_accepts_state_and_continuation_pair():
    assert gate.check_pair(*receipts(), expected_cached=90000)["pass"]


@pytest.mark.parametrize("fault", ["state", "tokens", "cached", "memory", "cold_was_warm", "missing_rank",
                                  "missing_tokens", "policy"])
def test_gate_rejects_corruption_and_mislabeled_receipts(fault):
    cold, warm = receipts()
    stats = warm["tensorfold"]
    if fault == "state":
        stats["session_state_sha256"][1] = "f" * 64
    elif fault == "tokens":
        stats["token_ids"][-1] += 1
    elif fault == "cached":
        stats["cached"] = 0
    elif fault == "memory":
        stats["session_cache_source"] = "memory"
    elif fault == "cold_was_warm":
        cold["tensorfold"]["cached"] = 5
    elif fault == "missing_rank":
        stats["session_state_sha256"].pop()
    elif fault == "missing_tokens":
        del stats["token_ids"]
    else:
        stats["policy"] = "0"
    with pytest.raises(ValueError):
        gate.check_pair(cold, warm, expected_cached=90000)
