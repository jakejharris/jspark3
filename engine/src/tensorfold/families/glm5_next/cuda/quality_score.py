"""Default-off, teacher-forced GLM quality scoring on the ordinary prefill path.

Only the requested continuation needs a head projection. Prompt chunks still use
exactly the serving prefill stage/compute/commit sequence and chunk sizes.
"""
from __future__ import annotations

import math
from typing import Any

import torch

MAX_SCORE_TOKENS = 33024
MAX_SCORED_TOKENS = 256


def validate_request(body: dict[str, Any], model: str, vocab: int, context: int) -> tuple[list[int], int]:
    """Reject malformed scoring work before rank zero sends any collective command."""
    if set(body) != {"model", "token_ids", "score_start"} or body["model"] != model:
        raise ValueError("quality score needs model, token_ids and score_start for the served model")
    ids, start = body["token_ids"], body["score_start"]
    if not isinstance(ids, list) or not 2 <= len(ids) <= min(MAX_SCORE_TOKENS, context):
        raise ValueError(f"token_ids must contain 2 to {min(MAX_SCORE_TOKENS, context)} tokens")
    if any(type(token) is not int or not 0 <= token < vocab for token in ids):
        raise ValueError("token_ids must be integers in the served tokenizer vocabulary")
    if type(start) is not int or not 1 <= start < len(ids):
        raise ValueError("score_start must be between 1 and len(token_ids)-1")
    if len(ids) - start > MAX_SCORED_TOKENS:
        raise ValueError(f"a quality score request can score at most {MAX_SCORED_TOKENS} tokens")
    return ids, start


def distributed_nll(logits, token_id: int, offset: int, comm, world: int) -> float:
    """Full-vocabulary logsumexp from vocabulary shards; one all-gather of two FP32 scalars."""
    local = logits.reshape(-1).float()
    target = (local[token_id - offset] if offset <= token_id < offset + local.numel()
              else torch.full((), float("-inf"), dtype=torch.float32, device=local.device))
    packed = torch.stack((torch.logsumexp(local, dim=0), target))
    gathered = torch.empty((world * 2,), dtype=torch.float32, device=local.device)
    comm.all_gather(packed, gathered)
    parts = gathered.view(world, 2)
    value = torch.logsumexp(parts[:, 0], dim=0) - parts[:, 1].max()
    nll = float(value.item())
    if not math.isfinite(nll) or nll < -1e-4:
        raise RuntimeError("quality scorer produced an invalid NLL")
    return max(0.0, nll)


@torch.no_grad()
def score_prefill(e, ids: list[int], score_start: int) -> list[float]:
    """All TP ranks call this in the same order. The head projects one row at a time."""
    from . import glue
    from .forward import chunks_for, commit, compute, mm, stage

    w, st, b = e.w, e.st, e.pbuf
    e.reset()
    scores: list[float] = []
    for pos in range(0, len(ids), e.prefill_rows):
        chunk = ids[pos:pos + e.prefill_rows]
        rows = stage(w, st, b, chunk)
        compute(w, st, b, rows, logits=False, nch=chunks_for(st, rows), host_pos=st.pos)
        # Row at absolute position j-1 predicts token j. Only project rows
        # whose target token belongs to the frozen continuation.
        first = max(0, score_start - pos - 1)
        last = min(rows, len(ids) - pos - 1)
        for row in range(first, last):
            source = b.hidden[row:row + 1]
            normed = b.fnormed[row:row + 1]
            scales = b.fxs[row:row + 1]
            glue.rmsnorm(source, w.norm, w.cfg.eps, normed, scales)
            logits = mm(b, normed, w.head, scales, b.logits[:1])
            scores.append(distributed_nll(logits, ids[pos + row + 1], w.vocab_offset, w.comm, w.world))
        commit(w, st, b, rows, rows)
    if len(scores) != len(ids) - score_start:
        raise RuntimeError("quality scorer lost a continuation row")
    return scores
