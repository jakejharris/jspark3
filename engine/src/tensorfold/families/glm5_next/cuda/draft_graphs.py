"""Startup-only graph bank for the fixed per-slot DFlash rings."""

import copy
import os

import torch


# A budget, not an allocation estimate: reject a graph bank exceeding it. The
# ordinary admission planner reserves it before weights or graphs are loaded.
SLOT_BUDGET = 128 * 2**20


def configured(parallel: int, drafter) -> bool:
    value = os.environ.get("TF_GLM_DRAFT_SLOT_GRAPHS", "0")
    if value not in ("0", "1"):
        raise ValueError("TF_GLM_DRAFT_SLOT_GRAPHS must be 0 or 1")
    if value == "1" and (parallel < 2 or drafter is None):
        raise ValueError("TF_GLM_DRAFT_SLOT_GRAPHS requires --parallel >=2 and a DFlash drafter")
    return value == "1"


def capture(owner, slots: int) -> list:
    """All ranks call this at startup, never from ADMIT or reservation LOAD.

    Separate graph pools retain each slot's outputs and position pointer. Input
    scratch remains shared: the decoder consumes one proposal before the next.
    """
    parent = owner.drafter
    if not parent.ring:
        raise ValueError("TF_GLM_DRAFT_SLOT_GRAPHS requires a bounded drafter ring")
    bank = [parent]
    before = torch.cuda.memory_reserved()
    for slot in range(1, slots):
        d = copy.copy(parent)
        first, last = slot * parent.ring, (slot + 1) * parent.ring
        d.kc = [c[:, first:last] for c in parent.kc]
        d.vc = [c[:, first:last] for c in parent.vc]
        d.pos_dev = torch.zeros_like(parent.pos_dev)
        d.block_graph, d.tap_graphs = None, {}
        d.capture()
        bank.append(d)
        used = max(0, torch.cuda.memory_reserved() - before)
        votes = owner._gather_ints([int(used <= slot * SLOT_BUDGET)])
        if any(v != [1] for v in votes):
            raise RuntimeError("DFlash slot graph pools exceeded their admitted memory budget")
    owner.draft_graph_bytes = max(0, torch.cuda.memory_reserved() - before)
    print(f"[tensorfold] rank {owner.rank} DFlash slot graphs: {slots}; "
          f"extra reserved {owner.draft_graph_bytes} B; budget {(slots - 1) * SLOT_BUDGET} B", flush=True)
    return bank
