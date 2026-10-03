"""Negative controls for the decode observer's offline row and path checks."""

import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "tools" / "glm_decode_log_validate.py"
_spec = importlib.util.spec_from_file_location("glm_decode_log_validate", _path)
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def test_mixed_round_ids_and_missing_graph_counter_are_rejected():
    record = {"rank": 0, "round": 2, "concurrency": 1, "verify_rows": 2,
              "slot": "zero", "path": "graph-main", "graph_paths": {"main": 1},
              "streams": [{"sid": 5, "rows": 2, "accepted_by_position": ["match"]}]}
    rows = [{"rank": 0, "round": 2, "sid": 5, "row": i} for i in range(2)]
    validator.validate_record(record)
    with pytest.raises(ValueError, match="dropped"):
        validator.validate_record(dict(record, observer_dropped=1))
    validator.validate_rows(record, rows)
    with pytest.raises(ValueError, match="another rank or round"):
        validator.validate_rows(record, [rows[0], dict(rows[1], round=3)])
    with pytest.raises(ValueError, match="unavailable"):
        validator.validate_record(dict(record, graph_paths=None))
