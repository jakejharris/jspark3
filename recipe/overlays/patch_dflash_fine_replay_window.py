#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Retain the compact DFlash drafter window at the fine-grained replay boundary.

Mia's fine-grained hybrid APC (patch_hybrid_prefix_hit.py at f497020) lets the
MLA and Mamba target groups reuse a producer's last-prompt hash boundary
(floor(P / hash_block_size) * hash_block_size, the partial-tail entry). A reused
target boundary is only safe with a complete DFlash drafter window ending there;
otherwise the replay check backs the hit up to the preceding scheduler page
unless the fresh suffix is at least the drafter window (2048 tokens) and can
rebuild it.

JSpark3 retains drafter windows sparsely (VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA
=0): SingleTypeKVCacheManager.cache_blocks keeps only the window ending at the
replay boundary aligned to the scheduler block (2560). A fine target tail
therefore never has a drafter window, and every continuation shorter than the
window falls back to the 2560 page (live TP3: replay clamp 7040->5120,
5760->5120).

This patch keeps both boundaries under sparse retention when fine-grained hits
are enabled: the coarse window (unchanged) and the window ending at the replay
boundary aligned to the hash grain. Only the compact DFlash boundary-lookup
drafter group is affected, and only when the coordinator enabled partial hash
hits; otherwise the manager flag stays None and caching is byte-for-byte the
previous behavior. Dense retention already caches every drafter block (four
640-token blocks per hit fill a 2560 segment), so it is unaffected.

Cost: up to cdiv(window - 1, block) = 4 more retained drafter block ids per
request in the prefix cache (evictable; the boot KV pool is unchanged).

Installation states: pristine-after-SWA-chain source is patched; an already
patched source is a byte-identical no-op; partial or drifted state fails
closed before either file is written.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
COORD = Path(os.environ.get("GLM53_KV_COORDINATOR_PY", f"{_VLLM}/v1/core/kv_cache_coordinator.py"))
STM = Path(os.environ.get("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY",
                          f"{_VLLM}/v1/core/single_type_kv_cache_manager.py"))
MARK = "# [glm53-dflash-fine-replay-window-v1]"

COORD_OLD = """            self.single_type_managers[boundary_group_id].use_eagle = False
        logger.info(
            "[glm53-dflash-boundary-lookup-v1] boundary_group_ids=%s",
            sorted(self.dflash_boundary_group_ids),
        )
"""
COORD_NEW = """            self.single_type_managers[boundary_group_id].use_eagle = False
        logger.info(
            "[glm53-dflash-boundary-lookup-v1] boundary_group_ids=%s",
            sorted(self.dflash_boundary_group_ids),
        )
        # [glm53-dflash-fine-replay-window-v1] Fine target hits need a drafter
        # window ending at the fine replay boundary; sparse retention keeps it
        # in addition to the scheduler-page window.
        for boundary_group_id in self.dflash_boundary_group_ids:
            self.single_type_managers[
                boundary_group_id
            ]._glm53_fine_replay_alignment = (
                self.hash_block_size if self.enable_partial_hash_hits else None
            )
"""
STM_OLD = """        block_mask = self.reachable_block_mask(
            start_block=num_cached_blocks,
            end_block=num_full_blocks,
            alignment_tokens=self.scheduler_block_size,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
        )
"""
STM_NEW = STM_OLD + """        # [glm53-dflash-fine-replay-window-v1] Also keep the window ending at the
        # replay boundary aligned to the fine hash grain (set only on the compact
        # DFlash boundary-lookup group when partial hash hits are enabled).
        fine_alignment = getattr(self, "_glm53_fine_replay_alignment", None)
        if fine_alignment and block_mask is not None:
            fine_mask = self.reachable_block_mask(
                start_block=num_cached_blocks,
                end_block=num_full_blocks,
                alignment_tokens=fine_alignment,
                kv_cache_spec=self.kv_cache_spec,
                use_eagle=self.use_eagle,
                retention_interval=retention_interval,
                reachable_boundaries=[request.num_prompt_tokens - 1],
            )
            block_mask = (
                None
                if fine_mask is None
                else [coarse or fine for coarse, fine in zip(block_mask, fine_mask)]
            )
"""
EDITS = ((COORD, COORD_OLD, COORD_NEW), (STM, STM_OLD, STM_NEW))


def prepare(path: Path, old: str, new: str) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    original = path.read_text()
    if original.count(new) == 1 and original.count(MARK) == new.count(MARK):
        return original, original
    if MARK in original:
        raise SystemExit(f"{path}: partial or drifted {MARK} stage; refusing")
    if original.count(old) != 1:
        raise SystemExit(f"{path}: anchor not found exactly once; refusing")
    text = original.replace(old, new, 1)
    compile(text, str(path), "exec")
    return original, text


def write(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(text)
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    # Preflight both files before writing either.
    updates = [(path, *prepare(path, old, new)) for path, old, new in EDITS]
    for path, original, text in updates:
        if text != original:
            write(path, text)
        print(f"patched {path.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
