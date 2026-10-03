"""Rank-zero copy proposals and continuation learning; no target arithmetic or GPU allocations.

The price adapter is deliberately separate from lookup/learning so the draft allocator can consume
``offer()``'s cumulative acceptance probabilities in its round-wide allocator.
Missing measured cells keep the existing model drafter. No boot-line timing fit
is treated as a measured price. Replay priors are estimates, reset per request.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from .lookup import PromptLookup


def configured_copy_rows() -> int:
    """Boot capacity; wide copies are separately opt-in and require shared draft prices."""
    rows = int(os.environ.get("TF_GLM_COPY_WIDE_ROWS", "0"))
    if rows not in (0, 16, 32):
        raise ValueError("TF_GLM_COPY_WIDE_ROWS must be 0, 16 or 32")
    return rows or 8


def context_class(tokens: int) -> str:
    # Same lower-bound classes as the draft collector/allocator.
    for end, name in ((8192, "short"), (32768, "8k"), (131072, "32k")):
        if tokens < end:
            return name
    return "128k"


def run_bucket(tokens: int) -> int:
    return min(16, tokens)


@dataclass(frozen=True)
class CopyOffer:
    tokens: list[int]
    match: int
    run: int
    chances: list[float]       # P(all copies through this position survive)

    @property
    def claims(self) -> list[float]:
        previous, out = 1.0, []
        for chance in self.chances:
            out.append(chance / previous if previous else 0.0)
            previous = chance
        return out


def prefer_copy(table, model: list[float], copies: list[float], *, base_rows: int, base_expected: float,
                context: str, load: int, path_for_rows) -> tuple[bool, dict]:
    """Draft source comparison: other priced streams held at zero; ties to copy."""
    # Missing base coverage is a round fallback, even if a wider path is priced.
    base_ms = table.lookup(base_rows, context, load, path_for_rows(base_rows))

    def best(chances):
        value, expected = base_expected / base_ms, base_expected
        for depth, chance in enumerate(chances, 1):
            expected += chance
            rows = base_rows + depth
            try:
                ms = table.lookup(rows, context, load, path_for_rows(rows))
            except ValueError:
                continue
            value = max(value, expected / ms)
        return value

    model_rate, copy_rate = best(model), best(copies)
    return copy_rate >= model_rate, {"model_rate": model_rate, "copy_rate": copy_rate}


class CopyDrafts:
    """A request's private index, conditional copy rates, and model baseline rates."""

    def __init__(self, prompt, *, max_rows: int = 8, agree: int = 8, prior: dict | None = None) -> None:
        if max_rows not in (*range(2, 9), 16, 32):
            raise ValueError("copy windows must be 2..8, 16 or 32 rows")
        if not 3 <= agree <= 16:
            raise ValueError("copy agreement must be 3..16 tokens")
        self.lookup = PromptLookup(prompt, agree=agree, drafts=max_rows - 1)
        self.max_drafts = max_rows - 1
        self.width = min(2, self.max_drafts)
        self.run = 0
        self.rates: dict[tuple[int, int], list[int]] = {}
        self.prior_rates: dict[tuple[int, int], float] = {}
        if prior is not None:
            expected = {"gram": 3, "agree": agree, "reach": 16, "max_drafts": 7, "latest_start_wins": True}
            if prior.get("schema") != "glm-copy-continuation-prior-v1" or prior.get("lookup") != expected:
                raise ValueError("copy prior schema or lookup policy differs")
            # Below AGREE the replay never offered rows. Do not load diagnostic
            # one-step probes or borrow their global fallback as evidence.
            for match in range(agree, 17):
                row = prior["match_x_run"]["16+" if match == 16 else str(match)]
                for run in range(17):
                    rate = row["16+" if run == 16 else str(run)]["price_rate"]
                    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
                        raise ValueError("invalid copy continuation prior")
                    self.prior_rates[match, run] = rate
        self.model_rates: dict[int, list[int]] = {}
        self.pending: CopyOffer | None = None
        self.rounds = self.drafted = self.accepted = self.skipped = 0

    def offer(self, new: list[int], room: int) -> CopyOffer:
        self.lookup.extend(new)
        tokens, match = self.lookup.match(min(room, self.width))
        # Image placeholders are host keys, never vocabulary ids to verify/copy.
        tokens = tokens[:next((i for i, token in enumerate(tokens) if token < 0), len(tokens))]
        if not tokens:
            self.run = 0
        chances, chance = [], 1.0
        for i in range(len(tokens)):
            key = match, run_bucket(self.run + i)
            accepted, trials = self.rates.get(key, (0, 0))
            # Eight pseudo-observations preserve price_rate exactly at admission
            # while letting this request adapt. Unseen cells use Beta(1,1).
            mass = 8 if key in self.prior_rates else 2
            chance *= (accepted + mass * self.prior_rates.get(key, .5)) / (trials + mass)
            chances.append(chance)
        return CopyOffer(tokens, match, self.run, chances)

    def model_chances(self, depth: int) -> list[float]:
        # Optimistic until observed: copies must beat the model, not an invented
        # weak baseline. These are prefix rates, not conditional probabilities.
        chances, previous = [], 1.0
        for i in range(depth):
            accepted, trials = self.model_rates.get(i, (0, 0))
            previous = min(previous, (accepted + 1) / (trials + 1))
            chances.append(previous)
        return chances

    def selected(self, offer: CopyOffer, depth: int, *, skipped: bool = True) -> list[int]:
        self.pending = CopyOffer(offer.tokens[:depth], offer.match, offer.run, offer.chances[:depth])
        self.skipped += int(skipped)
        return self.pending.tokens

    def observe(self, drafted: int, accepted: int, *, censored: bool = False) -> None:
        """Learn only reached copy rows. EOS/count truncation is not a rejection."""
        offer, self.pending = self.pending, None
        if offer is None:
            for i in range(drafted if not censored else accepted):
                rate = self.model_rates.setdefault(i, [0, 0])
                rate[0] += int(i < accepted)
                rate[1] += 1
            self.run = 0
            return
        self.rounds += 1
        self.drafted += drafted
        self.accepted += accepted
        reached = min(drafted, accepted + int(not censored))
        for i in range(reached):
            rate = self.rates.setdefault((offer.match, run_bucket(offer.run + i)), [0, 0])
            rate[0] += int(i < accepted)
            rate[1] += 1
        if accepted == drafted:
            self.run += accepted
            self.width = min(self.max_drafts, max(self.width + 1, 2 * drafted))
        else:
            self.run = 0
            self.width = max(1, min(self.width, accepted + 1))


class CopyPrices:
    """Measured round prices, keyed by context class, load and graph/eager mode.

    File format: {"version": 1, "build": "sha", "weights": "id", "drafter": "id",
    "cells": [{"context": "32k", "load": 1, "slot": "zero", "reps": 3,
    "mode": "eager|graph-main|graph-sparse:<bucket>", "source": "receipt path", "verify_ms": {"1": ..., "8": ...},
    "draft_ms": {"f": [...depth 0..7...], "m": [...]}, "lookup_ms": ...}]}.
    Only supplied cells/row counts are used. Costs include the lookup overhead;
    draft_ms includes necessary MTP absorption for that arm (including depth 0).
    """

    def __init__(self, data: dict) -> None:
        if data.get("version") != 1 or not isinstance(data.get("cells"), list):
            raise ValueError("copy prices need version 1 and measured cells")
        self.cells = {}
        for cell in data["cells"]:
            key = (cell["context"], cell["load"], cell["mode"], cell["slot"])
            path_ok = (key[2] in ("graph-main", "eager") or
                       isinstance(key[2], str) and key[2].startswith("graph-sparse:") and key[2][13:].isdigit())
            if (key in self.cells or key[0] not in ("short", "8k", "32k", "128k")
                    or type(key[1]) is not int or not 1 <= key[1] <= 8 or not path_ok
                    or key[3] not in ("zero", "nonzero") or type(cell.get("reps")) is not int or cell["reps"] < 3
                    or not isinstance(cell.get("source"), str) or not cell["source"].strip()):
                raise ValueError("invalid or duplicate measured copy price cell")
            verify = {int(n): float(ms) for n, ms in cell["verify_ms"].items()}
            draft = {arm: [float(ms) for ms in values] for arm, values in cell["draft_ms"].items()}
            overhead = float(cell["lookup_ms"])
            if (not verify or any(n < key[1] or n > 8 * key[1] for n in verify)
                    or any(not math.isfinite(ms) or ms <= 0 for ms in verify.values())
                    or any(arm not in ("f", "m") or not 2 <= len(values) <= 8 for arm, values in draft.items())
                    or any(not math.isfinite(ms) or ms < 0 for values in draft.values() for ms in values)
                    or not math.isfinite(overhead) or overhead < 0):
                raise ValueError("copy prices must be finite measured costs within the eight-row window")
            self.cells[key] = (verify, draft, overhead)

    @classmethod
    def read(cls, path: str, *, weights: str, drafter: str, build: str, build_ok: str = "") -> CopyPrices:
        data = json.loads(Path(path).read_text())
        if (not weights or not drafter or not build or data.get("weights") != weights or data.get("drafter") != drafter
                or not data.get("build") or data["build"] not in (build, build_ok)):
            raise ValueError("copy price build/weight/drafter identity differs or is unset")
        return cls(data)

    def choose(self, offer: CopyOffer, *, context: int, load: int, mode, arm: str,
               model_chances: list[float], peer_rows: int = 0, peer_tokens: float = 0.0,
               peer_ms: float = 0.0, model_depth: int | None = None, skip_model: bool = True,
               slot: str = "zero") -> int:
        """Price each measured step edge; ties and missing baseline prices keep the drafter.

        Maximizing E[commits]/ms is the draft allocator's marginal inequality integrated over each
        measured step. Peer rows/tokens are held fixed for this source comparison;
        The draft round allocator may subsequently truncate the offered copy prefix.
        """
        def cell_for(rows):
            return self.cells.get((context_class(context), load, mode(rows) if callable(mode) else mode, slot))

        depth = len(model_chances) if model_depth is None else model_depth
        cell = cell_for(peer_rows + 1 + depth)
        if cell is None or not offer.tokens:
            return 0
        verify, draft, overhead = cell
        costs = draft.get(arm, [])
        baseline_ms = verify.get(peer_rows + 1 + depth)
        if baseline_ms is None or depth >= len(costs) or not costs:
            return 0
        best = (peer_tokens + 1 + sum(model_chances)) / (baseline_ms + costs[depth] + peer_ms)
        chosen, expected = 0, peer_tokens + 1.0
        for i, chance in enumerate(offer.chances, 1):
            expected += chance
            cell = cell_for(peer_rows + 1 + i)
            if cell is None:
                continue
            verify, draft, overhead = cell
            costs = draft.get(arm, [])
            if not costs or (not skip_model and depth >= len(costs)):
                continue
            ms = verify.get(peer_rows + 1 + i)
            if ms is None:
                continue
            rate = expected / (ms + costs[0 if skip_model else depth] + overhead + peer_ms)
            if rate > best:
                chosen, best = i, rate
        return chosen
