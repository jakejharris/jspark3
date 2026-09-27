#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Align prefill chunks to the Mamba block, not to the smallest cache block.

``Scheduler._mamba_block_aligned_split`` keeps the "align" cache-mode
invariant -- slot p holds the SSM state after exactly (p + 1) * block_size
tokens, written at chunk ends, so chunk ends must be block aligned -- using
``cache_config.block_size``. The engine recomputes that value as the minimum
block size over the prefix-caching groups (engine core), which the DFlash
drafter group drags down to its own page while the Mamba groups keep their
block. Chunk ends then land off the Mamba block:

* the state block hashed at the next chunk end holds the running state of an
  unaligned chunk end, so a prefix hit on it resumes the SSM layers short of
  the attention prefix (the align kernel writes one running-state slot per
  request, never intermediate boundary states);
* a boundary crossed mid-chunk gets no state block at all, so the checkpoint
  is lost and a repeat backs up one Mamba block.

The split also backs ``last_cache_position`` off by one block whenever
speculative decoding is EAGLE-like, assuming full attention drops its last
matching block. With patch_hybrid_prefix_hit.py only the drafter group is
EAGLE-flagged and full attention keeps every block, so the back-off only
discards the last reachable checkpoint.

This patch derives the alignment from the Mamba groups' own block size and
applies the back-off only when the full-attention group (group 0) is among
the coordinator's EAGLE groups. Sub-block progress is unchanged: an
allowance smaller than the Mamba block still advances inside a block and
stops at the next boundary. With a budget of two blocks minus the draft
slots (7168 - 8 against 3584) the aligned chunk is one block.

Requires patch_scheduler_decode_floor.py v5 (or an unpatched scheduler):
v5 bounds the alignment's ``max_prefill_tokens`` by the per-request mixed
cap, so a capped request keeps sub-block progress instead of rounding to
zero; earlier decode-floor forms are rejected. No shared anchors with that
overlay, either application order yields the same file.

Usage:
    python3 patch_mamba_align_chunking.py

Idempotent; a partially applied or drifted source fails before writing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-mamba-align-chunking-v1]"
DECODE_FLOOR_MARK = "# [glm53-decode-floor"  # every decode-floor version
DECODE_FLOOR_V5 = "# [glm53-decode-floor:v5]"

IMPORT_OLD = """from vllm.v1.kv_cache_interface import KVCacheConfig
"""
IMPORT_NEW = """from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
"""

INIT_OLD = """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
"""
INIT_NEW = """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # [glm53-mamba-align-chunking-v1] The block whose boundaries carry
        # cacheable SSM states is the Mamba groups' own; cache_config.block_size
        # is the smallest prefix-caching block (the drafter's page here).
        self.mamba_align_block_size = max(
            (
                group.kv_cache_spec.block_size
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            default=self.cache_config.block_size,
        )
        # Back off the last cacheable position only when full attention
        # (group 0) really drops its last matching block, i.e. it is one of
        # the coordinator's EAGLE groups.
        self.mamba_align_eagle_backoff = self.use_eagle and 0 in getattr(
            self.kv_cache_manager.coordinator, "eagle_group_ids", {0}
        )
"""

SPLIT_OLD = """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
"""
SPLIT_NEW = """        block_size = self.mamba_align_block_size  # [glm53-mamba-align-chunking-v1]
        # The last block-aligned position whose state can be cached. When
        # full attention prunes its last matching block (EAGLE), back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.mamba_align_eagle_backoff:
            last_cache_position = max(last_cache_position - block_size, 0)
"""

EDITS = (
    ("import", IMPORT_OLD, IMPORT_NEW),
    ("init", INIT_OLD, INIT_NEW),
    ("split", SPLIT_OLD, SPLIT_NEW),
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def verify_complete(text: str) -> list[str]:
    problems = [
        f"{label}: expected exactly one applied block, found {n}"
        for label, _, new in EDITS
        if (n := text.count(new)) != 1
    ]
    # An applied block may extend its anchor; judge the superseded forms on
    # the text with every applied block removed.
    stripped = text
    for _, _, new in EDITS:
        stripped = stripped.replace(new, "")
    problems += [
        f"{label}: superseded form still present" for label, old, _ in EDITS if old in stripped
    ]
    try:
        compile(text, str(P), "exec")
    except SyntaxError as exc:
        problems.append(f"syntax: {exc.msg}")
    return problems


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if DECODE_FLOOR_MARK in text and DECODE_FLOOR_V5 not in text:
        raise SystemExit(
            f"{P}: patch_scheduler_decode_floor.py older than v5 present; its "
            "per-request cap is not applied to the Mamba alignment"
        )
    if MARK not in text:
        for label, old, new in EDITS:
            text = replace_once(text, old, new, label)
    if problems := verify_complete(text):
        raise SystemExit(f"{P}: incomplete or drifted overlay state: " + "; ".join(problems))
    if text != P.read_text():
        P.write_text(text)
    print(f"patched {P.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
