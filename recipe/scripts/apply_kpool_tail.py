#!/usr/bin/env python3
"""Apply the exact vcruz305-inspired K-pool tail correction transactionally."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _atomic import Refusal, execute, print_receipt, safe_target
from _contracts import KPOOL

MAMBA = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
INDEXER = "vllm/v1/attention/backends/mla/indexer.py"
MAMBA_OLD = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""
MAMBA_NEW = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # Hybrid models never passed positions here, unlike default.py.
            # The K-pool tail builder needs them: without positions it skips
            # compute_kpool_tail_slot_mapping and uses the generic paged
            # mapping against a one-entry block-table row, which writes the
            # tail cache out of bounds. See docs/KPOOL_TAIL_BUG.md.
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""
INDEXER_OLD = """    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
"""
INDEXER_NEW = """    # In place: slot_mapping is the tail group's persistent buffer. A fresh
    # clone here is captured by CUDA graphs at a transient address and read
    # back stale on replay (illegal memory access). See docs/KPOOL_TAIL_BUG.md.
    out = slot_mapping
    if num_actual_tokens == 0:
        return out
"""


def replace_once(data: bytes, old: str, new: str, label: str) -> bytes:
    text = data.decode("utf-8")
    if text.count(old) != 1:
        raise Refusal(f"{label}: source seam drift")
    return text.replace(old, new, 1).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        vllm = args.vllm_root.resolve(strict=True)
        root = vllm.parent
        if (root / "vllm").resolve(strict=True) != vllm:
            raise Refusal("--vllm-root must name the vllm package directory")

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            mamba = safe_target(root, MAMBA)
            indexer = safe_target(root, INDEXER)
            return {
                mamba: replace_once(before[mamba], MAMBA_OLD, MAMBA_NEW, "mamba positions"),
                indexer: replace_once(before[indexer], INDEXER_OLD, INDEXER_NEW, "persistent slot mapping"),
            }

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_kpool_tail.py",
            expected_section=KPOOL,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        print_receipt(receipt)
        return 0
    except (OSError, ValueError, UnicodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
