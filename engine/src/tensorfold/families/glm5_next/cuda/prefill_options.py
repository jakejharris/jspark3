"""Default-off prefill trials; all TP ranks must use the same flags for the lifetime of a boot."""

import os


def _flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


PROMPT_SCRATCH = _flag("TF_GLM_P1_PROMPT_SCRATCH")
SELECT_BLOCKS = _flag("TF_GLM_P1_SELECT_BLOCKS")
VISIBLE_POOLS = _flag("TF_GLM_P1_VISIBLE_POOLS")
DENSE_ROWS = _flag("TF_GLM_P1_DENSE_ROWS")
DEAD_WORK = _flag("TF_GLM_P1_DEAD_WORK")
PROMPT_ATT_ROWS = 512
SELECT_ROWS = 512

# Tiled attention keeps the expert pass intact while attention consumes bounded row tiles.
ATTENTION_TILES = _flag("TF_GLM_A2_ATTENTION_TILES")
EXPRESS = _flag("TF_GLM_A2_EXPRESS")
ATTENTION_ROWS = 512
EXPRESS_ROWS = 2048

# Express prompts with at most this many uncached rows run all layers in one
# pass instead of one-layer slices; the same bound caps an express cofill pass
# while any pooled reply is decoding (an incumbent's worst single pause). 32
# with EXPRESS_COFILL=0 selects the previous scheduling behavior.
EXPRESS_ATOMIC_ROWS = int(os.environ.get("TF_GLM_EXPRESS_ATOMIC_ROWS", "512"))
if not 1 <= EXPRESS_ATOMIC_ROWS <= 1024:
    raise ValueError("TF_GLM_EXPRESS_ATOMIC_ROWS must be an integer from 1 to 1024")
# Waiting atomic express prompts share one cofill pass (needs TF_GLM_COFILL=1).
EXPRESS_COFILL = _flag("TF_GLM_EXPRESS_COFILL", "1")
# Rank zero only: while any pooled reply decodes, an express pass (one atomic
# prompt or a cofill group) holds at most this many rows; a larger atomic
# prompt is then sliced by layer (or by TF_GLM_PREFILL_SLICE_MS). This bounds
# an incumbent's pause; TF_GLM_EXPRESS_ATOMIC_ROWS applies to idle passes.
EXPRESS_BUSY_ROWS = int(os.environ.get("TF_GLM_EXPRESS_BUSY_ROWS", "256"))
if not 1 <= EXPRESS_BUSY_ROWS <= 1024:
    raise ValueError("TF_GLM_EXPRESS_BUSY_ROWS must be an integer from 1 to 1024")
# Rank zero only: on an idle server, hold waiting atomic express prompts up to
# this many ms after the first admission so near-simultaneous arrivals (staggered
# by tokenization) share one cofill pass. 0 = off (no wait).
EXPRESS_GATHER_MS = int(os.environ.get("TF_GLM_EXPRESS_GATHER_MS", "0"))
if not 0 <= EXPRESS_GATHER_MS <= 200:
    raise ValueError("TF_GLM_EXPRESS_GATHER_MS must be an integer from 0 to 200")
# Rank zero only: an automatically sliced chunk advances as many layers per
# step as fit this estimated budget (ms) instead of one; 0 keeps one layer.
# Followers take each layer stop from FILL_SLICE, so ranks may differ.
PREFILL_SLICE_MS = int(os.environ.get("TF_GLM_PREFILL_SLICE_MS", "0"))
if not 0 <= PREFILL_SLICE_MS <= 1000:
    raise ValueError("TF_GLM_PREFILL_SLICE_MS must be an integer from 0 to 1000")
# Whole-pass cost model for the budget: fixed sweep plus rows at the cold rate.
SLICE_FIXED_MS = 100.0
SLICE_ROWS_PER_S = 2150.0

# Co-prefill uses one tiled expert pass for chunks from several independent prompts.
COFILL_BIG = _flag("TF_GLM_COFILL_BIG")

# Keep the recurrence's arithmetic while changing independent value-row ownership.
KDA_VALUE_TILES = _flag("TF_GLM_KDA_VALUE_TILES")

# Sparse prefill keeps token/chunk order while trimming repeated empty tiles.
SPARSE_TRIM = _flag("TF_GLM_SPARSE_TRIM")

# Zero preserves full decode rounds at every yield. N serves one target token
# every N sliced-prefill steps, retaining DFlash taps until drafting resumes.
PREFILL_DECODE_QUANTUM = int(os.environ.get("TF_GLM_PREFILL_DECODE_QUANTUM", "0"))
if not 0 <= PREFILL_DECODE_QUANTUM <= 4:
    raise ValueError("TF_GLM_PREFILL_DECODE_QUANTUM must be an integer from 0 to 4")

# Expert prefill keeps stock expert arithmetic; only large g64 SwiGLU prefill passes opt in.
EXPERT_PREFILL128 = _flag("TF_GLM_EXPERT_PREFILL128")

# Four-warp gate/up for actual 1024-4095-row g64 SwiGLU passes; down stays stock.
EXPERT_PREFILL64 = _flag("TF_GLM_EXPERT_PREFILL64")

# Keep attention overlap halves, but let MoE consume both halves' normed rows together.
EXPERT_WHOLE_PASS = _flag("TF_GLM_EXPERT_WHOLE_PASS")

# Exact replay (default on): a memory snapshot keeps its prompt-end head row,
# so a prompt equal to a stored one samples its first token without prefill.
# 0 restores strict-prefix-only lookup.
_exact = os.environ.get("TF_GLM_EXACT_REPLAY", "1")
if _exact not in ("0", "1"):
    raise ValueError("TF_GLM_EXACT_REPLAY must be 0 or 1")
EXACT_REPLAY = _exact == "1"
