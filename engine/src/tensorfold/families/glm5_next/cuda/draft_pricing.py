"""Host-only draft price lookup, reached-row calibration, and deterministic cuts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

SEGMENTS = ("think", "text", "tool", "copy")
SRC_D, SRC_C, SRC_F, SRC_N = 0, 1, 2, 3


def segment_patterns(model_dir: Path | None = None) -> dict[str, list[int]] | None:
    """Read the four literal markers in the served chat template using its tokenizer."""
    override = load_json_env("TF_GLM_SEGMENT_IDS")
    if override is not None:
        return {k: [int(x) for x in v] for k, v in override.items()}
    if model_dir is None:
        return None
    template = model_dir / "chat_template.jinja"
    tokenizer = model_dir / "tokenizer.json"
    if not template.exists() or not tokenizer.exists():
        return None
    markers = {"think_open": "<think>", "think_close": "</think>",
               "tool_open": "<tool_call>", "tool_close": "</tool_call>"}
    if any(v not in template.read_text() for v in markers.values()):
        return None
    try:
        from tokenizers import Tokenizer
        encoded = Tokenizer.from_file(str(tokenizer))
        return {k: encoded.encode(v, add_special_tokens=False).ids for k, v in markers.items()}
    except (ImportError, ValueError, OSError):
        return None


class SegmentTracker:
    def __init__(self, patterns: dict[str, list[int]] | None, prompt: list[int]):
        self.patterns = patterns or {}
        self.width = max((len(v) for v in self.patterns.values()), default=1)
        self.tail = list(prompt[-self.width:])
        self.segment = "think" if self._ends("think_open") else "text"

    def _ends(self, marker: str) -> bool:
        pattern = self.patterns.get(marker)
        return bool(pattern) and self.tail[-len(pattern):] == pattern

    def feed(self, tokens: list[int]) -> str:
        for token in tokens:
            self.tail = (self.tail + [token])[-self.width:]
            if self._ends("think_open"):
                self.segment = "think"
            elif self._ends("think_close"):
                self.segment = "text"
            elif self._ends("tool_open"):
                self.segment = "tool"
            elif self._ends("tool_close"):
                self.segment = "text"
        return self.segment

    def row_segments(self, drafts: list[int]) -> list[str]:
        tracker = SegmentTracker(self.patterns, [])
        tracker.tail, tracker.segment = self.tail[:], self.segment
        out = []
        for token in drafts:
            out.append(tracker.segment)
            tracker.feed([token])
        return out


def load_json_env(name: str):
    value = os.environ.get(name)
    if not value:
        return None
    if value[0] in "{[0123456789.-":
        return json.loads(value)
    return json.loads(Path(value).read_text())


class Calibration:
    def __init__(self, prior: dict | None = None, *, copy_depth: int = 7):
        self.prior = ({seg: float(prior) for seg in SEGMENTS}
                      if isinstance(prior, (int, float)) else prior or {})
        self.counts = {seg: [[0.0, 0.0] for _ in range(copy_depth if seg == "copy" else 7)] for seg in SEGMENTS}

    def ratio(self, segment: str, pos: int) -> float:
        rows = self.counts[segment]
        claimed = sum(x[0] for x in rows)
        accepted = sum(x[1] for x in rows)
        base = self.prior.get(segment, 1.0)
        if isinstance(base, list):
            base = base[min(pos, len(base) - 1)]
        seg_ratio = (accepted + 8.0 * float(base)) / (claimed + 8.0)
        c, a = rows[pos]
        return (a + 4.0 * seg_ratio) / (c + 4.0)

    def chances(self, segment: str, claims: list[float], raw: bool = False) -> list[float]:
        return [min(0.98, max(0.0, p * (1.0 if raw else self.ratio(segment, i))))
                for i, p in enumerate(claims)]

    def chances_rows(self, segments: list[str], claims: list[float], raw: bool = False) -> list[float]:
        return [min(0.98, max(0.0, p * (1.0 if raw else self.ratio(seg, i))))
                for i, (seg, p) in enumerate(zip(segments, claims))]

    def update_rows(self, segments: list[str], claims: list[float], drafts: list[int],
                    target: list[int], room: int, eos: tuple[int, ...]) -> list[str]:
        outcomes = ["censored"] * len(claims)
        for i, (seg, p, token) in enumerate(zip(segments, claims, drafts)):
            if i + 1 >= room:
                break
            pick = target[i]
            matched = pick == token
            c, a = self.counts[seg][i]
            self.counts[seg][i] = [c * (31.0 / 32.0) + p,
                                   a * (31.0 / 32.0) + float(matched)]
            outcomes[i] = "eos-match" if matched and pick in eos else "match" if matched else "mismatch"
            if not matched or pick in eos:
                break
        return outcomes

    def update(self, segment: str, claims: list[float], accepted: int, *, rejected: bool = True) -> None:
        # The first rejected row is reached; later rows are censored.
        reached = accepted + int(rejected)
        for i, p in enumerate(claims[:min(len(claims), reached)]):
            c, a = self.counts[segment][i]
            self.counts[segment][i] = [c * (31.0 / 32.0) + p, a * (31.0 / 32.0) + float(i < accepted)]


class PriceTable:
    """tf-price/v1 with step-up lookup inside one context/load/path."""

    def __init__(self, data: dict, *, weights: str, drafter: str, build: str, build_ok: str = ""):
        if data.get("schema") != "tf-price/v1" or data.get("kind") not in ("measured", "estimate"):
            raise ValueError("invalid price table schema or kind")
        if not weights or not drafter or not build:
            raise ValueError("price table identity needs TF_GLM_WEIGHT_ID, TF_GLM_DRAFTER_ID and TF_GLM_BUILD_ID")
        if data.get("weights") != weights or data.get("drafter") != drafter:
            raise ValueError("price table weight or drafter identity differs")
        if data.get("build") != build and (not build_ok or data.get("build") != build_ok):
            raise ValueError("price table build is not the running or explicitly allowed build")
        self.kind = data["kind"]
        self.cells = data["cells"]
        if not self.cells or any(c["rows"] < 1 or c["load"] < 1 or c["reps"] < (30 if self.kind == "measured" else 3)
                                 or not math.isfinite(c["median_ms"]) or c["median_ms"] <= 0
                                 or not math.isfinite(c["spread"]) or c["spread"] < 0
                                 or c["ctx"] not in ("short", "8k", "32k", "128k")
                                 or not (c["path"] in ("graph-main", "eager") or
                                         c["path"].startswith("graph-sparse:")) for c in self.cells):
            raise ValueError("price table has invalid or under-repeated cells")
        keys = [(c["rows"], c["ctx"], c["load"], c["path"]) for c in self.cells]
        if len(set(keys)) != len(keys):
            raise ValueError("price table has duplicate cells")
        if self.kind == "estimate" and any(c["load"] != 1 or c["rows"] > 8
                                           for c in self.cells):
            raise ValueError("estimate price tables are c1 only")

    def lookup(self, rows: int, context: str, load: int, path: str) -> float:
        classes = ("short", "8k", "32k", "128k")
        at = classes.index(context)
        for ctx in classes[at:at + 2]:
            cells = [c for c in self.cells if c["load"] == load and c["ctx"] == ctx and c["path"] == path
                     and c["rows"] >= rows]
            if cells:
                return min(cells, key=lambda c: c["rows"])["median_ms"]
        raise ValueError(f"price table lacks {context}/{path}/load{load} at {rows} rows")


def context_class(length: int) -> str:
    if length < 8192:
        return "short"
    if length < 32768:
        return "8k"
    if length < 131072:
        return "32k"
    return "128k"


def choose_depths(streams: list[dict], table: PriceTable, *, context: str, load: int,
                  base_rows: int, other_expected: float = 0.0, solo: bool = False,
                  path_for_rows=None, priced_count: int | None = None) -> tuple[dict[int, int], dict]:
    """Each stream has sid and cumulative per-row pass chances; ties use sid and position."""
    if solo and len(streams) > 1:
        parts = [choose_depths([s], table, context=context, load=load, base_rows=base_rows,
                               other_expected=other_expected, path_for_rows=path_for_rows,
                               priced_count=len(streams)) for s in streams]
        return ({sid: depth for depths, _ in parts for sid, depth in depths.items()}, {"solo": True})
    offers = [(chance, s["sid"], i + 1) for s in streams for i, chance in enumerate(s["pass_chances"])]
    offers.sort(key=lambda x: (-x[0], x[1], x[2]))
    expected = (len(streams) if priced_count is None else priced_count) + other_expected
    best = (-math.inf, 0, 0.0)
    trace = []
    for m in range(len(offers) + 1):
        total = base_rows + m
        path = path_for_rows(total) if path_for_rows else "eager"
        try:
            ms = table.lookup(total, context, load, path)
        except ValueError:
            if m == 0:
                raise
            if m < len(offers):
                expected += offers[m][0]
            continue
        rate = expected / ms
        trace.append({"rows": total, "path": path, "expected": expected, "ms": ms, "rate": rate})
        if rate > best[0]:
            best = (rate, m, ms)
        if m < len(offers):
            expected += offers[m][0]
    depths = {s["sid"]: 0 for s in streams}
    for _, sid, _ in offers[:best[1]]:
        depths[sid] += 1
    return depths, {"cut": best[1], "rate": best[0], "ms": best[2], "trace": trace}


def cumulative(chances: list[float]) -> list[float]:
    out, product = [], 1.0
    for q in chances:
        product *= q
        out.append(product)
    return out


def threshold_depth(chances: list[float], threshold: float) -> int:
    passing = cumulative(chances)
    return next((i for i, q in enumerate(passing) if i > 0 and q < threshold), len(passing))
