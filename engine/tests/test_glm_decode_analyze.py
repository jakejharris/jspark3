"""Synthetic decode receipt controls for priors, held-out tuning, phase share and sampled rows."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm_decode_analyze as analysis  # noqa: E402


def _write(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def _record(round_id, sid, prompt, claims, outcomes, segments, *, tap_file=None):
    stream = {"sid": sid, "rows": len(claims) + 1, "accepted_by_position": outcomes,
              "claims": claims, "row_segments": segments, "prompt_sha256": prompt,
              "family": "think", "segment": segments[0] if segments else "text"}
    record = {"rank": 0, "round": round_id, "concurrency": 1, "verify_rows": len(claims) + 1,
              "slot": "zero", "path": "eager", "graph_paths": {"eager": 1},
              "streams": [stream]}
    if tap_file:
        record["tap_file"] = tap_file
        record["tap_sha256"] = [f"tap{i}" for i in range(5)]
    return record


def test_prior_censors_tail_and_tuner_keeps_test_prompts_out_of_prior(tmp_path):
    records = []
    for i in range(5):
        prompt = f"prompt-{i}"
        records.append(_record(2 * i, i, prompt, [.8, .5],
                               ["match" if i % 2 else "mismatch", "censored"], ["think", "text"]))
        records.append(_record(2 * i + 1, i, prompt, [.25], ["eos-match"], ["tool"]))
    path = _write(tmp_path / "rank0.jsonl", records)
    rows = analysis.observations([path])
    prior, report = analysis.build_prior(rows)
    assert report["text"][1]["reached"] == 0
    assert prior["text"][1] == 1.0
    assert report["tool"][0]["accepted"] == 5
    result = analysis.tune(rows)["think"]
    assert result["test_prompts"] == 1 and result["validation_prompts"] == 1
    assert result["train_prompts"] == 3 and result["best_ratio"]["test_reached"] > 0
    train_prior = result["prior_from_train_only"]
    _, _, test_prompts = analysis.split_prompts(rows)
    changed = [dict(row, outcome="mismatch") if row["prompt"] in test_prompts and row["outcome"] in analysis.REACHED
               else row for row in rows]
    assert analysis.tune(changed)["think"]["prior_from_train_only"] == train_prior


def test_phase_share_uses_cuda_events_per_rank_and_rejects_host_only(tmp_path):
    path = _write(tmp_path / "phases.jsonl", [
        {"rank": 0, "phase": "decode", "cuda_stream_ms": 30, "host_start_s": 1, "host_end_s": 1.05},
        {"rank": 0, "phase": "prefill", "cuda_stream_ms": 70, "host_start_s": 2, "host_end_s": 2.1},
        {"rank": 1, "phase": "decode", "cuda_stream_ms": 40, "host_start_s": 1, "host_end_s": 1.05},
        {"rank": 1, "phase": "prefill", "cuda_stream_ms": 60, "host_start_s": 2, "host_end_s": 2.1},
    ])
    result = analysis.phase_share([path])
    assert result["decode_share_rank_median"] == pytest.approx(.35)
    bad = _write(tmp_path / "host-only.jsonl", [
        {"rank": 0, "phase": "decode", "cuda_stream_ms": None, "host_start_s": 1, "host_end_s": 2}])
    with pytest.raises(ValueError, match="CUDA event"):
        analysis.phase_share([bad])


def test_d8_manifest_references_all_taps_and_labels_doomed_tail(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    safetensors.save_file({"model.norm.weight": torch.ones(2), "lm_head.weight": torch.ones(4, 2)},
                          model_dir / "model.safetensors")
    tap_file = tmp_path / "rank0-round0.pt"
    rows = [{"rank": 0, "round": 0, "sid": 4, "row": i, "tap_row": i,
             "prompt_sha256": "prompt-d8", "family": "think", "segment": "think",
             "context": "short", "draft_token": 10 + i, "claim": None if i == 0 else [.8, .6][i - 1],
             "candidate_top8": [] if i == 0 else [10 + i, 20 + i],
             "target_pick": 11 + i, "outcome": ["pending", "match", "mismatch"][i]}
            for i in range(3)]
    torch.save({"taps": [torch.zeros(3, 2, dtype=torch.bfloat16) for _ in range(5)], "rows": rows}, tap_file)
    record = _record(0, 4, "prompt-d8", [.8, .6], ["match", "mismatch"], ["think", "think"],
                     tap_file=tap_file.name)
    path = _write(tmp_path / "rank0.jsonl", [record])
    peer = _write(tmp_path / "rank1.jsonl", [dict(record, rank=1, tap_sha256=["wrong"] * 5)])
    output = tmp_path / "private" / "manifest.jsonl"
    report = analysis.d8_manifest([path], output, minimum_rows=3, peer_paths=[peer], model_dir=model_dir)
    manifest = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["status"] for row in manifest] == ["pending", "useful", "doomed"]
    assert all(row["tap_layers"] == [5, 14, 24, 33, 42] for row in manifest)
    assert report["families"]["think"]["meets_20k_rows"]
    assert report["rank_tap_hash_agreement"]["1"] == {"checked_rounds": 1, "all_equal": False}
    assert report["checkpoint_weights"]["tensors"]["lm_head"]["shape"] == [4, 2]
    assert output.stat().st_mode & 0o777 == 0o600


def test_phase_share_rejects_dropped_observations(tmp_path):
    path = _write(tmp_path / "dropped.jsonl", [
        {"rank": 0, "phase": "decode", "cuda_stream_ms": 30, "host_start_s": 1, "host_end_s": 1.05,
         "observer_dropped": 1}])
    with pytest.raises(ValueError, match="dropped"):
        analysis.phase_share([path])
