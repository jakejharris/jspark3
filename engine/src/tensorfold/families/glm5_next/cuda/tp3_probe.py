"""Three-rank reduction exactness probes, off by default.

- ``TF_TP3_REDUCE=nccl`` sums the ranks' fp32 partials with NCCL's ring all-reduce instead of TensorFold's gather and
  rank-order sum. With three operands NCCL's grouping follows its chunking, which moves with the row count, so a
  drafted reply can differ from its serial reference when the reduction grouping changes.
- ``TF_TP3_CAPTURE=DIR`` writes, for the main model's eager decode steps of each request (at most
  ``TF_TP3_CAPTURE_STEPS``, default 64), every cross-rank reduction's local partial and result, one file a step:
  ``DIR/rank<R>/req<NNN>-<draft|serial>/step<NNNNN>.pt``. Compare the saved inputs and results to locate the first
  differing reduction between drafted and serial requests. Capturing turns the engine's CUDA graphs off
  (a step has to run eagerly to be copied out).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

REDUCE = os.environ.get("TF_TP3_REDUCE", "gather")
CAPTURE = os.environ.get("TF_TP3_CAPTURE", "")
STEPS = int(os.environ.get("TF_TP3_CAPTURE_STEPS", "64"))
if REDUCE not in ("gather", "nccl"):
    raise ValueError(f"TF_TP3_REDUCE={REDUCE!r}: gather (TensorFold's rank-order sum) or nccl (the ring all-reduce probe)")


class Recorder:
    """One request's steps: begin_request, then begin_step / record... / end_step for each main-model decode step."""

    def __init__(self, folder: str) -> None:
        self.root, self.folder, self.requests = Path(folder), None, 0
        self.step = self.steps = 0
        self.calls: list | None = None
        self.meta: dict = {}

    def begin_request(self, rank: int, draft: bool) -> None:
        self.folder = self.root / f"rank{rank}" / f"req{self.requests:03d}-{'draft' if draft else 'serial'}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.requests += 1
        self.steps = 0

    def begin_step(self, pos: int, ids: torch.Tensor) -> None:
        if self.folder is None or self.steps >= STEPS:
            self.calls = None
            return
        self.calls = []
        self.meta = {"pos": int(pos), "ids": ids.cpu().tolist()}

    def record(self, part: torch.Tensor, result: torch.Tensor) -> None:
        if self.calls is not None:
            self.calls.append((part.float().cpu(), result.float().cpu()))

    def end_step(self) -> None:
        if self.calls is None:
            return
        torch.save({**self.meta, "calls": self.calls}, self.folder / f"step{self.steps:05d}.pt")
        self.steps += 1
        self.calls = None


recorder = Recorder(CAPTURE) if CAPTURE else None
