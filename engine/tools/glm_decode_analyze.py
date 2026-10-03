"""Analyze local decode logs: priors, held-out calibration, phase share and sampled rows.

The sampled-row manifest contains token IDs and model-state paths. Keep its directory private.
CUDA event elapsed time includes stream gaps; it is not exclusive kernel-active time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

SEGMENTS = ("think", "text", "tool", "copy")
REACHED = {"match": 1, "eos-match": 1, "mismatch": 0}
TAP_LAYERS = (5, 14, 24, 33, 42)


def _jsonl(path: Path):
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def _family(item: dict, override: str | None) -> str:
    value = override or item.get("family") or "unknown"
    if value == "unknown":
        raise ValueError("Decode family is unknown; set TF_GLM_DECODE_FAMILY during collection or --family")
    return value


def observations(paths: list[Path], family_override: str | None = None) -> list[dict]:
    """Only rank zero supplies calibration labels; later rows after mismatch/EOS stay censored."""
    out = []
    for path in paths:
        for record in _jsonl(path):
            if record.get("rank") != 0:
                raise ValueError(f"{path}: calibration input must be a rank-0 log")
            for stream in record["streams"]:
                claims = stream.get("claims", [])
                labels = stream["accepted_by_position"]
                segments = stream.get("row_segments")
                prompt = stream.get("prompt_sha256")
                if not prompt or not isinstance(segments, list) or not (len(claims) == len(labels) == len(segments)):
                    raise ValueError(f"{path}: full prompt hash and per-row segments are required")
                family = _family(stream, family_override)
                for position, (claim, label, segment) in enumerate(zip(claims, labels, segments), 1):
                    if label not in (*REACHED, "censored") or segment not in SEGMENTS:
                        raise ValueError(f"{path}: invalid outcome or segment")
                    if not isinstance(claim, (float, int)) or not math.isfinite(claim) or not 0 <= claim <= 1:
                        raise ValueError(f"{path}: invalid conditional claim")
                    out.append({"family": family, "prompt": prompt, "session": (str(path), stream["sid"]),
                                "round": record["round"], "position": position, "segment": segment,
                                "claim": float(claim), "outcome": label})
    return out


def build_prior(rows: list[dict]) -> tuple[dict, dict]:
    totals = {seg: [[0.0, 0.0, 0] for _ in range(7)] for seg in SEGMENTS}
    for row in rows:
        if row["outcome"] not in REACHED:
            continue
        if not 1 <= row["position"] <= 7:
            raise ValueError("tf-price/v1 supports draft positions 1..7")
        c, a, n = totals[row["segment"]][row["position"] - 1]
        totals[row["segment"]][row["position"] - 1] = [c + row["claim"], a + REACHED[row["outcome"]], n + 1]
    prior = {seg: [a / c if c else 1.0 for c, a, _ in totals[seg]] for seg in SEGMENTS}
    report = {seg: [{"claimed": c, "accepted": a, "reached": n, "ratio": prior[seg][i]}
                    for i, (c, a, n) in enumerate(totals[seg])] for seg in SEGMENTS}
    return prior, report


def split_prompts(rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    prompts = sorted({row["prompt"] for row in rows})
    if len(prompts) < 3:
        raise ValueError("held-out tuning needs at least three distinct prompts per family")
    ranked = sorted(prompts, key=lambda p: hashlib.sha256(p.encode()).digest())
    ntest = max(1, min(len(ranked) - 2, round(len(ranked) * .2)))
    nvalid = max(1, min(len(ranked) - ntest - 1, round(len(ranked) * .2)))
    test = set(ranked[:ntest])
    validation = set(ranked[ntest:ntest + nvalid])
    return set(prompts) - test - validation, validation, test


class ReplayCalibration:
    def __init__(self, prior: dict, h: int, m: float, m0: float):
        self.prior, self.h, self.m, self.m0 = prior, h, m, m0
        self.counts = {seg: [[0.0, 0.0] for _ in range(7)] for seg in SEGMENTS}

    def chance(self, row: dict) -> float:
        seg, i, p = row["segment"], row["position"] - 1, row["claim"]
        counts = self.counts[seg]
        c_all, a_all = sum(x[0] for x in counts), sum(x[1] for x in counts)
        base = self.prior[seg][i]
        seg_ratio = (a_all + self.m0 * base) / (c_all + self.m0)
        c, a = counts[i]
        return min(.98, max(0.0, p * (a + self.m * seg_ratio) / (c + self.m)))

    def update(self, row: dict) -> None:
        if row["outcome"] not in REACHED:
            return
        i, seg = row["position"] - 1, row["segment"]
        c, a = self.counts[seg][i]
        g = 1 - 1 / self.h
        self.counts[seg][i] = [g * c + row["claim"], g * a + REACHED[row["outcome"]]]


def _brier(predictions: list[tuple[float, int]]) -> float:
    return sum((p - y) ** 2 for p, y in predictions) / len(predictions) if predictions else math.nan


def replay(rows: list[dict], prior: dict, h: int, m: float, m0: float) -> tuple[float, int]:
    sessions = defaultdict(list)
    for row in rows:
        sessions[row["session"]].append(row)
    predictions = []
    for group in sessions.values():
        cal = ReplayCalibration(prior, h, m, m0)
        rounds = defaultdict(list)
        for row in group:
            rounds[row["round"]].append(row)
        for round_id in sorted(rounds):
            current = rounds[round_id]
            # A production round prices its entire chain before any outcome updates.
            chances = [cal.chance(row) for row in current]
            for row, chance in zip(current, chances):
                if row["outcome"] in REACHED:
                    predictions.append((chance, REACHED[row["outcome"]]))
            for row in current:
                cal.update(row)
    return _brier(predictions), len(predictions)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-35, min(35, x))))


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def fit_logit(rows: list[dict]) -> tuple[float, float]:
    data = [(_logit(row["claim"]), REACHED[row["outcome"]]) for row in rows if row["outcome"] in REACHED]
    a, b = 1.0, 0.0
    for _ in range(30):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in data:
            q = _sigmoid(a * x + b)
            d, w = q - y, q * (1 - q)
            g0 += d * x
            g1 += d
            h00 += w * x * x
            h01 += w * x
            h11 += w
        g0 += .01 * (a - 1)
        g1 += .01 * b
        h00 += .01
        h11 += .01
        det = h00 * h11 - h01 * h01
        if det <= 0:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a -= max(-1, min(1, da))
        b -= max(-1, min(1, db))
        if max(abs(da), abs(db)) < 1e-6:
            break
    return a, b


def tune(rows: list[dict], hs=(16, 32, 64), ms=(2, 4, 8), m0s=(4, 8, 16)) -> dict:
    if not hs or not ms or not m0s or any(x <= 1 for x in hs) or any(x <= 0 for x in (*ms, *m0s)):
        raise ValueError("H must exceed 1; m and m0 must be positive")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    result = {}
    for family, group in sorted(by_family.items()):
        train_prompts, validation_prompts, test_prompts = split_prompts(group)
        train = [r for r in group if r["prompt"] in train_prompts]
        validation = [r for r in group if r["prompt"] in validation_prompts]
        test = [r for r in group if r["prompt"] in test_prompts]
        prior, _ = build_prior(train)
        scores = []
        for h in hs:
            for m in ms:
                for m0 in m0s:
                    score, reached = replay(validation, prior, h, m, m0)
                    scores.append({"H": h, "m": m, "m0": m0, "validation_brier": score,
                                   "validation_reached": reached})
        if not scores or scores[0]["validation_reached"] == 0:
            raise ValueError(f"{family}: validation prompts have no reached draft rows")
        best = min(scores, key=lambda x: (x["validation_brier"], x["H"], x["m"], x["m0"]))
        test_score, test_reached = replay(test, prior, best["H"], best["m"], best["m0"])
        if not test_reached:
            raise ValueError(f"{family}: test prompts have no reached draft rows")
        best = dict(best, test_brier=test_score, test_reached=test_reached)
        a, b = fit_logit(train)
        logit_score = _brier([(_sigmoid(a * _logit(r["claim"]) + b), REACHED[r["outcome"]])
                              for r in test if r["outcome"] in REACHED])
        result[family] = {"train_prompts": len(train_prompts),
                          "validation_prompts": len(validation_prompts), "test_prompts": len(test_prompts),
                          "best_ratio": best, "logit": {"a": a, "b": b, "brier": logit_score},
                          "ratio_more_than_10pct_worse": test_score > 1.1 * logit_score,
                          "grid": scores, "prior_from_train_only": prior}
    return result


def phase_share(paths: list[Path]) -> dict:
    by_rank = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for path in paths:
        for row in _jsonl(path):
            if row.get("observer_dropped", 0):
                raise ValueError(f"{path}: observer dropped phases; share would be biased")
            phase = row.get("phase")
            if phase not in ("prefill", "decode"):
                raise ValueError(f"{path}: unknown phase {phase!r}")
            ms = row.get("cuda_stream_ms")
            if not isinstance(ms, (float, int)) or not math.isfinite(ms) or ms < 0:
                raise ValueError(f"{path}: CUDA event timing is required for GPU share")
            if row["host_end_s"] < row["host_start_s"]:
                raise ValueError(f"{path}: reversed host interval")
            by_rank[row["rank"]][phase] += float(ms)
            counts[row["rank"]][phase] += 1
    ranks = {}
    for rank, phases in sorted(by_rank.items()):
        total = phases["decode"] + phases["prefill"]
        if total <= 0:
            raise ValueError(f"rank {rank}: no timed CUDA work")
        ranks[str(rank)] = {"decode_cuda_stream_ms": phases["decode"],
                            "prefill_cuda_stream_ms": phases["prefill"],
                            "decode_share": phases["decode"] / total,
                            "prefill_share": phases["prefill"] / total,
                            "phase_counts": dict(counts[rank])}
    if not ranks:
        raise ValueError("no phase intervals")
    return {"ranks": ranks, "decode_share_rank_median": statistics.median(x["decode_share"] for x in ranks.values()),
            "measurement": "CUDA event elapsed within serialized scheduler phases; includes stream gaps and waits; not exclusive kernel-active time"}


def _row_status(outcomes: list[str], row: int) -> str:
    if row == 0:
        return "pending"
    prefix = outcomes[:row]
    if "mismatch" in prefix or "eos-match" in prefix[:-1]:
        return "doomed"
    if "censored" in prefix:
        return "unknown"
    return "useful" if prefix[-1] in ("match", "eos-match") else "doomed"


def checkpoint_inventory(model_dir: Path) -> dict:
    """Locate final norm and rank-sharded head sources without loading model tensors."""
    from safetensors import safe_open

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                weight_map.update({key: shard.name for key in reader.keys()})
    required = {"norm": next((key for key in ("model.norm.weight", "norm.weight") if key in weight_map), None),
                "lm_head": "lm_head.weight" if "lm_head.weight" in weight_map else None}
    if any(value is None for value in required.values()):
        raise ValueError("checkpoint lacks final norm or LM head weight")
    optional = {key: key for key in ("lm_head.scales", "lm_head.biases") if key in weight_map}
    chosen = {**required, **optional}
    tensors = {}
    for role, name in chosen.items():
        shard = model_dir / weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as reader:
            view = reader.get_slice(name)
            tensors[role] = {"name": name, "file": str(shard.resolve()),
                             "shape": view.get_shape(), "dtype": view.get_dtype()}
    return {"tensors": tensors, "reconstruction":
            "Use weights.py vocab_share and make_b16/make_q4 for the served TP3 head; tensor files are read-only."}


def d8_manifest(paths: list[Path], output: Path, family_override: str | None = None,
                minimum_rows: int = 20_000, peer_paths: list[Path] | None = None,
                model_dir: Path | None = None) -> dict:
    import torch
    from glm_decode_log_validate import validate_record, validate_rows

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    counts = defaultdict(lambda: defaultdict(int))
    peer_hashes = {}
    for path in peer_paths or []:
        for record in _jsonl(path):
            peer_hashes[(path.parent.resolve(), record["rank"], record["round"])] = record.get("tap_sha256")
    agreements = defaultdict(list)
    fd = os.open(output, flags, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as sink:
        for path in paths:
            for record in _jsonl(path):
                if record.get("rank") != 0:
                    raise ValueError("Sampled-row input must be a rank-0 log")
                if "tap_file" not in record:
                    continue
                validate_record(record)
                payload = torch.load(path.parent / record["tap_file"], map_location="cpu", weights_only=False)
                validate_rows(record, payload["rows"])
                taps = payload["taps"]
                if len(taps) != len(TAP_LAYERS) or any(t.shape[0] != record["verify_rows"] for t in taps):
                    raise ValueError("Sampled-row analysis needs all five tap layers and every executed row")
                for (directory, rank, round_id), hashes in peer_hashes.items():
                    if directory == path.parent.resolve() and round_id == record["round"] and hashes is not None:
                        agreements[rank].append(hashes == record.get("tap_sha256"))
                streams = {s["sid"]: s for s in record["streams"]}
                for row in payload["rows"]:
                    stream = streams[row["sid"]]
                    family = _family(row, family_override)
                    prompt = row["prompt_sha256"]
                    if prompt != stream.get("prompt_sha256"):
                        raise ValueError("tap prompt differs from round log")
                    i = row["row"]
                    if i and (row["outcome"] != stream["accepted_by_position"][i - 1]
                              or row["segment"] != stream["row_segments"][i - 1]
                              or row["claim"] != stream["claims"][i - 1]):
                        raise ValueError("tap row labels differ from round log")
                    status = _row_status(stream["accepted_by_position"], row["row"])
                    item = {"family": family, "segment": row["segment"], "prompt_sha256": prompt,
                            "split": "test" if int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 5 == 0 else "train",
                            "round": record["round"], "sid": row["sid"], "row": row["row"],
                            "load": record["concurrency"], "context": row["context"],
                            "path": record["path"], "rows_in_round": record["verify_rows"],
                            "tap_file": str(path.parent / record["tap_file"]), "tap_row": row["tap_row"],
                            "tap_layers": TAP_LAYERS, "tap_widths": [t.shape[1] for t in taps],
                            "draft_token": row["draft_token"], "claim": row["claim"],
                            "candidate_top8": row["candidate_top8"], "target_pick": row["target_pick"],
                            "outcome": row["outcome"], "status": status}
                    sink.write(json.dumps(item, separators=(",", ":")) + "\n")
                    counts[family][status] += 1
    return {"manifest": str(output), "families": {family: {**statuses,
             "total": sum(statuses.values()), "meets_20k_rows": sum(statuses.values()) >= minimum_rows}
             for family, statuses in sorted(counts.items())},
            "tap_layers": TAP_LAYERS,
            "checkpoint_weights": checkpoint_inventory(model_dir) if model_dir else "model directory not supplied",
            "rank_tap_hash_agreement": {str(rank): {"checked_rounds": len(matches), "all_equal": all(matches)}
                                        for rank, matches in sorted(agreements.items())}}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as sink:
        json.dump(data, sink, indent=2)
        sink.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prior", "tune", "share", "d8"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--log", type=Path, action="append", required=True, help="rank-0 decode JSONL, or phase JSONL for share")
        cmd.add_argument("--out", type=Path, required=True)
        if name in ("prior", "tune", "d8"):
            cmd.add_argument("--family", help="override unknown family label")
        if name == "tune":
            cmd.add_argument("--H", dest="hs", default="16,32,64")
            cmd.add_argument("--m", dest="ms", default="2,4,8")
            cmd.add_argument("--m0", dest="m0s", default="4,8,16")
        if name == "d8":
            cmd.add_argument("--peer-log", type=Path, action="append", default=[])
            cmd.add_argument("--minimum-rows", type=int, default=20_000)
            cmd.add_argument("--model-dir", type=Path, help="read-only inventory of final norm and LM head shards")
    args = parser.parse_args()
    if args.command == "prior":
        rows = observations(args.log, args.family)
        prior, report = build_prior(rows)
        _write_json(args.out, prior)  # exact TF_GLM_DRAFT_PRIOR file schema
        _write_json(args.out.with_suffix(".report.json"), {"rows": len(rows), "counts": report})
        print(json.dumps({"prior": str(args.out), "rows": len(rows)}))
    elif args.command == "tune":
        result = tune(observations(args.log, args.family),
                      hs=tuple(int(x) for x in args.hs.split(",")),
                      ms=tuple(float(x) for x in args.ms.split(",")),
                      m0s=tuple(float(x) for x in args.m0s.split(",")))
        _write_json(args.out, result)
        print(json.dumps({k: {"best_ratio": v["best_ratio"], "ratio_more_than_10pct_worse":
                                v["ratio_more_than_10pct_worse"]} for k, v in result.items()}))
    elif args.command == "share":
        result = phase_share(args.log)
        _write_json(args.out, result)
        print(json.dumps({"decode_share_rank_median": result["decode_share_rank_median"],
                          "measurement": result["measurement"]}))
    else:
        result = d8_manifest(args.log, args.out, args.family, args.minimum_rows, args.peer_log,
                             args.model_dir)
        _write_json(args.out.with_suffix(".report.json"), result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
