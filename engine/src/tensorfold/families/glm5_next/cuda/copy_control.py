"""Isolated batched-decoder adapter. Rank zero owns lookup, prices and cuts."""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path

from .copy_drafts import CopyDrafts, CopyPrices, context_class, prefer_copy
from .draft_pricing import SRC_C, SRC_D, SRC_N, cumulative

ADMIT_COPY, COPY_PROPOSE, COPY_CHECKED = 21, 22, 26   # 5 belongs to FILL_SLICE
MODEL, COPY, EMPTY, BOTH = 1, 2, 0, 3


class CopyController:
    def __init__(self, decoder) -> None:
        self.decoder = decoder
        self.prices = None
        self.configured = False
        self.check = False
        self.price_error = "missing"

    def enabled(self, stream, code) -> bool:
        enabled = getattr(stream, "copy_request", None)
        if enabled is None:
            enabled = os.environ.get("TF_GLM_COPY_DRAFTS", "0") == "1"
        if type(enabled) is not bool:
            raise ValueError("copy_drafts must be a boolean")
        if not enabled or not stream.draft or not code[0] or code[0] == 17 or getattr(stream, "price_table_missing", None):
            return False
        if not self.configured:
            self.max_rows = int(os.environ.get("TF_GLM_COPY_ROWS", "8"))
            if not 2 <= self.max_rows <= 8:
                raise ValueError("TF_GLM_COPY_ROWS must be 2..8; use TF_GLM_COPY_WIDE_ROWS for 16/32")
            self.agree = int(os.environ.get("TF_GLM_COPY_AGREE", "8"))
            self.skip_model = os.environ.get("TF_GLM_COPY_SKIP", "0") == "1"
            self.check = os.environ.get("TF_GLM_COPY_CHECK", "0") == "1"
            prior_path = os.environ.get("TF_GLM_COPY_PRIOR")
            prior_bytes = Path(prior_path).read_bytes() if prior_path else None
            self.prior = json.loads(prior_bytes) if prior_bytes else None
            self.prior_sha256 = hashlib.sha256(prior_bytes).hexdigest() if prior_bytes else None
            if self.prior is not None:
                tokenizer = os.environ.get("TF_GLM_COPY_TOKENIZER_SHA256", "")
                if not tokenizer or self.prior.get("tokenizer_sha256") != tokenizer:
                    raise ValueError("copy prior tokenizer identity differs or is unset")
            CopyDrafts([], max_rows=self.max_rows, agree=self.agree, prior=self.prior)  # before ADMIT/collectives
            path = os.environ.get("TF_GLM_COPY_PRICES")
            if path:
                try:
                    self.prices = CopyPrices.read(path, weights=os.environ.get("TF_GLM_WEIGHT_ID", ""),
                                                  drafter=os.environ.get("TF_GLM_DRAFTER_ID", ""),
                                                  build=os.environ.get("TF_GLM_BUILD_ID", ""),
                                                  build_ok=os.environ.get("TF_GLM_PRICE_TABLE_BUILD_OK", ""))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.price_error = "invalid:" + str(exc)
            self.configured = True
        return True

    def admit(self, s, enabled: bool) -> None:
        s.copy_enabled = enabled
        s.copy_drafts = []
        if enabled and self.decoder.rank == 0:
            wide = self.decoder.max_rows > 8 and s.price_mode == 16 and s.policy_code[1] == 7
            rows = self.decoder.max_rows if wide else self.max_rows
            s.copies = CopyDrafts(s.prompt, max_rows=rows, agree=self.agree, prior=self.prior)
            s.copy_stats = {"rounds": 0, "drafted": 0, "accepted": 0, "drafter_skips": 0,
                            "unpriced": 0, "sources": [], "price_table": (
                                "shared:tf-price/v1" if getattr(s, "priced", False) and s.price_mode == 16 else
                                "loaded" if self.prices else self.price_error)}
            s.copy_stats["prior_sha256"] = self.prior_sha256
            s.copy_stats["max_rows"] = rows

    def _exchange(self, s, mode: int, depth: int, tokens: list[int], room: int, pending: int,
                  *, check_local: bool = False):
        d = self.decoder
        msg = [COPY_CHECKED if self.check else COPY_PROPOSE, s.sid, mode, depth, *tokens]
        if d.rank == 0:
            d._send(msg)
        else:
            msg = d.share(None)
        valid = not (len(msg) < 4 or msg[0] not in (COPY_PROPOSE, COPY_CHECKED) or msg[1] != s.sid
                or msg[2] not in (EMPTY, MODEL, COPY, BOTH)
                or not 0 <= msg[3] <= min(d.max_rows - 1, max(0, room))
                or len(msg) != 4 + (msg[3] if msg[2] == COPY else 0)
                or (msg[2] == EMPTY and msg[3] != 0) or (msg[2] != EMPTY and msg[3] == 0)
                or any(token < 0 for token in msg[4:]))
        # Optional debug vote before any next model collective. Every rank sees a
        # stale position, different local fallback chain, or malformed control;
        # a lone follower must not raise while peers enter target/drafter work.
        chosen = tokens if check_local else msg[4:]
        local_mode, local_depth = (mode, depth) if check_local else (msg[2:4] if len(msg) >= 4 else (-1, -1))
        width = d.max_rows - 1
        vote = [int(valid), s.sid, s.st.pos, pending, local_mode, local_depth,
                *(chosen[:width] + [-1] * (width - len(chosen[:width])))]
        gather = getattr(d.owner, "_gather_ints", None) if msg[0] == COPY_CHECKED else None
        votes = gather(vote) if gather is not None else [vote]
        if not valid or any(v[0] != 1 or v != votes[0] for v in votes):
            raise RuntimeError("invalid rank-zero copy proposal")
        return msg[2], msg[3], msg[4:]

    def _mode(self, s, load: int, rows: int) -> str:
        # Mirror Engine.forward's graph dispatch, including rows 7/8 being eager.
        e = s.engine
        graphs = e.graphs
        if rows > 8 or load != 1 or e is not self.decoder.owner.e or graphs is None:
            return "eager"
        limit = getattr(e.w.cfg, "dense_limit", 0)
        if s.st.pos + rows <= limit:
            return "graph-main" if graphs.main.get((rows, s.st.parity)) is not None else "eager"
        if s.st.pos >= limit and s.st.index is not None and hasattr(graphs, "sparse"):
            from .sparse import pool_bucket

            bucket = pool_bucket(s.st.pos, rows, s.st.index[0][2].shape[0] - 2)
            return f"graph-sparse:{bucket}" if graphs.sparse.get((rows, s.st.parity, bucket)) is not None else "eager"
        return "eager"

    def _choose(self, s, offer, depth: int, room: int, *, skip_model: bool) -> int:
        if self.prices is None:
            s.copy_stats["unpriced"] += 1
            return 0
        d = self.decoder
        peers = [p for p in d.streams.values() if p is not s and not p.done and p.sid not in d.filling
                 and (d.growth is None or d.growth.safe(p))]
        load = len(peers) + 1
        context = max(p.st.pos for p in [s, *peers])
        peer_rows = sum(1 + len(p.drafts) for p in peers)
        peer_tokens, peer_ms = 0.0, 0.0
        # The shared draft allocator can replace this source comparison. Here peers'
        # currently published windows are fixed; do not invent missing peer costs.
        for p in peers:
            cell = self.prices.cells.get((context_class(context), load, "eager", "zero"))
            costs = cell[1].get("f" if p.use_dflash else "m", []) if cell else []
            n = len(p.drafts)
            copied = p.copies.pending if p.copies is not None else None
            cost_depth = getattr(p, "copy_model_depth", n)
            if cell is None or not costs or cost_depth >= len(costs):
                s.copy_stats["unpriced"] += 1
                return 0
            chances = (copied.chances if copied else p.copies.model_chances(n)
                       if p.copies is not None else [1.0] * n)
            peer_tokens += 1 + sum(chances)
            peer_ms += costs[cost_depth] if costs else 0
            if copied:
                peer_ms += cell[2]
        return self.prices.choose(offer, context=context, load=load, mode=lambda n: self._mode(s, load, n),
                                  slot="nonzero" if load == 1 and s.engine is not d.owner.e else "zero",
                                  arm="f" if s.use_dflash else "m", model_depth=depth,
                                  model_chances=s.copies.model_chances(min(depth, max(0, room - 1))),
                                  peer_rows=peer_rows, peer_tokens=peer_tokens, peer_ms=peer_ms, skip_model=skip_model)

    def propose(self, s, pending, hidden, new, room: int, depth: int) -> None:
        d = self.decoder
        joint = s.priced and s.price_mode == 16
        s.copy_drafts = []
        # A full fp7 model cap allows the separate wide-copy ceiling. Explicit
        # smaller policies (including the background prefill cap) keep their bound.
        cap = d.max_rows - 1 if joint and s.policy_code[1] == 7 else min(7, depth)
        s.copy_cap = min(cap, max(0, room - 1)) if depth > 0 and pending not in s.eos else 0
        mode, tokens = MODEL, []
        if d.rank == 0:
            offer = s.copies.offer(new, s.copy_cap)
            if depth <= 0 or pending in s.eos:
                mode, depth = EMPTY, 0
            elif self.skip_model and not joint:
                chosen = self._choose(s, offer, depth, room, skip_model=True) if offer.tokens else 0
                if chosen:
                    mode, depth = COPY, chosen
                    tokens = s.copies.selected(offer, depth)
            else:
                mode = BOTH
        mode, depth, tokens = self._exchange(s, mode, depth, tokens, room, pending)
        s.copy_model_depth = 0
        if mode == COPY:
            if s.use_mtp:
                # Keep the MTP cache current without chaining predictions. DFlash
                # already absorbed kept taps in _round, even on copy-only rounds.
                from .decode import absorb

                if (d.growth is not None and s.st.mtp_len - s.st.mtp_drafted + len(new) > s.span - 8):
                    raise RuntimeError("copy proposal MTP absorption would cross its token reservation")
                absorb(s.engine, hidden, new)
            s.drafts = tokens
        elif mode in (MODEL, BOTH):
            d._model_propose(s, pending, hidden, new, depth)
            # DFlash confidence can shorten its block; broadcast the actual cut
            # and tokens as well as the pre-collective choice of drafter/depth.
            local = s.drafts
            s.copy_model_depth = len(local)
            _, _, s.drafts = self._exchange(s, COPY if local else EMPTY, len(local), local, room, pending, check_local=True)
            if d.rank != 0 and s.drafts != local:
                raise RuntimeError("model draft differs from rank-zero copy fallback")
            if mode == BOTH:
                if joint:
                    # Preserve the cap chain and its claims until ROUND2. Every
                    # rank also retains the copy chain that ROUND2 may select.
                    if d.rank == 0:
                        s.copy_offer = offer
                        copies = offer.tokens
                    else:
                        copies = []
                    _, _, s.copy_drafts = self._exchange(s, COPY if copies else EMPTY, len(copies), copies,
                                                       room, pending)
                    return
                selected_mode, selected_depth, copies = MODEL if s.drafts else EMPTY, len(s.drafts), []
                if d.rank == 0:
                    chosen = self._choose(s, offer, len(s.drafts), room, skip_model=False) if offer.tokens else 0
                    if chosen:
                        selected_mode, selected_depth = COPY, chosen
                        copies = s.copies.selected(offer, chosen, skipped=False)
                mode, _, copies = self._exchange(s, selected_mode, selected_depth, copies, room, pending)
                if mode == COPY:
                    s.drafts = copies   # model pass already maintained MTP context, if present
        if d.rank == 0:
            s.copy_stats["sources"].append("copy" if mode == COPY else "model" if mode == MODEL else "empty")
            s.copy_stats["drafter_skips"] = s.copies.skipped
            s.copy_source = SRC_C if mode == COPY else SRC_D if mode == MODEL else SRC_N
            if mode == COPY:
                self._copy_observation(s, offer)

    @staticmethod
    def _copy_observation(s, offer) -> None:
        s.draft_observation = {"claims": offer.claims, "candidates": [[t] for t in offer.tokens],
                               "claim_convention": "copy-continuation-conditional"}

    def joint_offer(self, s, model: list[float], **kwargs):
        """Choose the source, but defer learner updates until the joint cut is applied."""
        if not s.copy_enabled or not s.copy_drafts:
            return False, model
        raw = bool(s.policy_code[2] & 1)
        s.copy_chances = s.calibration.chances("copy", s.copy_offer.claims, raw=raw)
        copies = cumulative(s.copy_chances[:s.copy_cap])
        selected, detail = prefer_copy(self.decoder.price_table, model, copies, **kwargs)
        s.copy_stats["source_price"] = detail
        return selected, copies if selected else model

    @staticmethod
    def applied(s, source: int, depth: int) -> None:
        if not getattr(s, "copy_enabled", False) or not s.priced or s.copies is None:
            return
        s.copy_stats["sources"].append("copy" if source == SRC_C else "model")
        if source == SRC_C:
            if depth:
                s.copies.selected(s.copy_offer, depth, skipped=False)
            CopyController._copy_observation(s, s.copy_offer)
            s.priced_segments = ["copy"] * len(s.copy_offer.tokens)
            s.priced_chances = s.copy_chances
            s.price_ratios = [1.0 if s.policy_code[2] & 1 else s.calibration.ratio("copy", i)
                              for i in range(len(s.copy_offer.tokens))]

    def observe(self, s, keep: int, got) -> None:
        if self.decoder.rank != 0 or not getattr(s, "copy_enabled", False):
            return
        accepted, censored = 0, True
        for i, token in enumerate(s.drafts):
            if i + 1 >= s.count - len(s.out):
                break
            if got[i] != token:
                censored = False
                break
            accepted += 1
            if got[i] in s.eos:
                break
        s.copies.observe(len(s.drafts), accepted, censored=censored)
        s.copy_stats.update(rounds=s.copies.rounds, drafted=s.copies.drafted, accepted=s.copies.accepted)
