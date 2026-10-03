"""Isolated real-weight DFlash launch screen, stock A / slot graphs B / stock A.

Requires a dedicated GPU with no TP3 rank running. Rank-zero TP3 weights
and repeated local partials isolate launch/replay cost, not network transport or
end-to-end serving throughput. The controller owns the MEASURING/rank guard.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from tensorfold.families.glm5_next.cuda import draft_graphs
from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
from tensorfold.families.glm5_next.cuda.qmm import as_i32, make_q4
from tensorfold.families.glm5_next.cuda.split import RankReader
from tensorfold.families.glm5_next.cuda.weights import Config, PREFIX, vocab_share


class Copies:
    def __init__(self, world):
        self.world = world

    def all_gather(self, send, recv):
        flat = send.reshape(-1)
        for r in range(self.world):
            recv.reshape(-1)[r * flat.numel():(r + 1) * flat.numel()].copy_(flat)


def load_draft(model, path):
    """Load only target embeddings/head and the actual DFlash weights."""
    cfg = Config.read(model)
    if cfg.quant != "mlx":
        raise ValueError("this real-weight screen expects the serving MLX-4bit rank folder")
    rd = RankReader(model, 0, 3)
    embed = (as_i32(rd.get(PREFIX + "embed_tokens.weight")).cuda(),
             rd.get(PREFIX + "embed_tokens.scales").cuda(), rd.get(PREFIX + "embed_tokens.biases").cuda())
    lo, hi = vocab_share(cfg.vocab, 0, 3)
    head = make_q4(as_i32(rd.get("lm_head.weight")[lo:hi]).cuda(),
                   rd.get("lm_head.scales")[lo:hi].cuda(), rd.get("lm_head.biases")[lo:hi].cuda())
    w = SimpleNamespace(embed=embed, head=head, draft_head=None, device=torch.device("cuda"),
                        rank=0, world=3, vocab_offset=lo, comm=Copies(3))
    return Drafter(path, w, capacity=262144, streams=8)


def digest(d, result):
    h = hashlib.sha256()
    for value in result:
        h.update(value.tobytes())
    for value in [d.pos_dev, *d.kc, *d.vc]:
        h.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--drafter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=24)
    args = p.parse_args()
    torch.cuda.set_device(0)
    parent = load_draft(args.model, args.drafter)
    parent.capture()
    owner = SimpleNamespace(drafter=parent, rank=0, _gather_ints=lambda v: [v] * 3)
    bank = draft_graphs.capture(owner, 8)
    stock = [parent]
    for d in bank[1:]:
        e = copy.copy(d)
        e.block_graph, e.tap_graphs = None, {}
        stock.append(e)
    generator = torch.Generator(device="cuda").manual_seed(20261002)
    taps = torch.randn((64, len(parent.tap_layers) * parent.D), generator=generator,
                       device="cuda", dtype=torch.bfloat16)

    def seed(ds, context):
        for d in ds:
            d.reset()
            for c in d.kc + d.vc:
                c.zero_()
            # Only the sliding-window suffix is read, at its real absolute pos.
            start = max(0, context - d.ring)
            d.context_end = start
            d.pos_dev.fill_(start)
            for at in range(start, context, 64):
                d.add_taps(taps[:min(64, context - at)])

    report = {"kind": "isolated-local-partials-real-weights", "target": args.model,
              "draft": args.drafter, "torch": torch.__version__,
              "incremental_reserved_bytes": owner.draft_graph_bytes,
              "admitted_budget_bytes": 7 * draft_graphs.SLOT_BUDGET, "cells": []}
    for context in (128, 32768):
        for slots in (1, 8):
            cell = {"context": context, "slots": slots, "arms": {}}
            for arm, ds in (("A1", stock[:slots]), ("B", bank[:slots]), ("A2", stock[:slots])):
                seed(ds, context)
                checks = [digest(d, d.candidates(11 + i, 7)) for i, d in enumerate(ds)]
                times, outputs = [], []
                for turn in range(args.rounds + 5):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    out = []
                    for i, d in enumerate(ds):
                        d.add_taps(taps[:4])
                        out.append(d.propose(11 + i, 7, None, .3))
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - begin) * 1e3
                    if turn >= 5:
                        times.append(elapsed)
                        outputs.append(out)
                cell["arms"][arm] = {"round_ms": times, "median_ms": statistics.median(times),
                                     "state_hashes": checks,
                                     "output_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest()}
                print(json.dumps({"context": context, "slots": slots, "arm": arm,
                                  "median_ms": cell["arms"][arm]["median_ms"]}), flush=True)
            a, b, a2 = [cell["arms"][arm] for arm in ("A1", "B", "A2")]
            cell["drift_fraction"] = abs(a2["median_ms"] / a["median_ms"] - 1)
            baseline = (a["median_ms"] + a2["median_ms"]) / 2
            cell["draft_round_saved_ms"] = baseline - b["median_ms"]
            cell["draft_speedup_fraction"] = baseline / b["median_ms"] - 1
            cell["exact"] = (a["state_hashes"] == b["state_hashes"] == a2["state_hashes"] and
                             a["output_sha256"] == b["output_sha256"] == a2["output_sha256"])
            cell["valid"] = cell["exact"] and cell["drift_fraction"] <= .1
            report["cells"].append(cell)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    report["valid"] = all(c["valid"] for c in report["cells"])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["valid"]:
        raise SystemExit("rejected cohort: drift >10% or graph/eager bytes differ")


if __name__ == "__main__":
    main()
