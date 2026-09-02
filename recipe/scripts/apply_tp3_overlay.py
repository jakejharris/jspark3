#!/usr/bin/env python3
"""Construct the exact base-recipe TP3 overlay from hash-pinned Fly sources."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

from _atomic import Refusal, execute, print_receipt, safe_target, sha_file
from _contracts import TP3

MODEL = "vllm/models/glm5next/nvidia/model.py"
EXL3 = "vllm/model_executor/layers/quantization/exl3.py"
VOCAB = "vllm/model_executor/layers/vocab_parallel_embedding.py"
PARAMETER = "vllm/model_executor/parameter.py"
FLASH = "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"

SOURCE_PATHS = {
    "patch_tp3_glm.py": "overlay/patch_tp3_glm.py",
    "exl3.py": "overlay/miaai/exl3.py",
    "vocab_parallel_embedding.py": "overlay/vllm/model_executor/layers/vocab_parallel_embedding.py",
    "parameter.py": "overlay/vllm/model_executor/parameter.py",
    "flashinfer_mla_sparse_sm120.py": "overlay/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
}

FLASH_EDITS = (
    (
        "3× TP pad: stock 64 heads → 66 so local=22. FlashInfer SM120 decode only\n"
        "instantiates local heads in {8,16,32,64,128}; 22 falls through to prefill\n"
        "which asserts num_tokens>64. For decode-sized calls (≤64 tokens) pad Q/out\n"
        "to 32 local heads and slice back. Prefill (>64) keeps 22. Do not pad the\n"
        "model to 96 — that makes one rank all dummy KDA heads and destroys logits.\n",
        "3× TP pad: stock 64 heads → 66 so local=22. FlashInfer SM120 decode and\n"
        "prefill instantiate local heads only in {8,16,32,64,128}; 22 is unsupported\n"
        "in both dispatch tables. Pad Q/out to 32 local heads for every call and slice\n"
        "back. Do not pad the model to 96 — that makes one rank all dummy KDA heads\n"
        "and destroys logits.\n",
    ),
    (
        "# FlashInfer SM120 sparse-MLA decode instantiation set (local heads).\n"
        "_SM120_DECODE_HEADS = (8, 16, 32, 64, 128)\n"
        "_SM120_DECODE_MAX_TOKENS = 64\n",
        "# FlashInfer SM120 sparse-MLA decode and prefill instantiation set (local heads).\n"
        "_SM120_KERNEL_HEADS = (8, 16, 32, 64, 128)\n",
    ),
    (
        "def _decode_kernel_heads(n_local: int) -> int:\n"
        "    for t in _SM120_DECODE_HEADS:\n",
        "def _kernel_heads(n_local: int) -> int:\n"
        "    for t in _SM120_KERNEL_HEADS:\n",
    ),
    (
        "        kernel_heads = self.num_heads\n"
        "        q_kernel = q\n"
        "        if num_actual_toks <= _SM120_DECODE_MAX_TOKENS:\n"
        "            kernel_heads = _decode_kernel_heads(self.num_heads)\n"
        "            if kernel_heads != self.num_heads:\n"
        "                # [T, H, D] → pad H on the right. Extra heads are zero-Q\n"
        "                # and discarded after the kernel; they do not join softmax\n"
        "                # of the real heads (softmax is per-head).\n"
        "                q_kernel = torch.nn.functional.pad(\n"
        "                    q, (0, 0, 0, kernel_heads - self.num_heads)\n"
        "                )\n",
        "        kernel_heads = _kernel_heads(self.num_heads)\n"
        "        q_kernel = q\n"
        "        if kernel_heads != self.num_heads:\n"
        "            # [T, H, D] → pad H on the right. Extra heads are zero-Q and\n"
        "            # discarded after the kernel; they do not join softmax of the\n"
        "            # real heads (softmax is per-head). This applies to both the\n"
        "            # <=64-token decode dispatch and the >64-token prefill dispatch.\n"
        "            q_kernel = torch.nn.functional.pad(\n"
        "                q, (0, 0, 0, kernel_heads - self.num_heads)\n"
        "            )\n",
    ),
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise Refusal(f"{label}: expected exactly one source seam")
    return text.replace(old, new, 1)


def literal_bindings(source: bytes) -> dict[str, object]:
    """Read literal patch data without importing or executing the upstream script."""
    tree = ast.parse(source.decode("utf-8"))
    values: dict[str, object] = {}
    pending = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    changed = True
    while changed:
        changed = False
        for node in pending:
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name in values:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            values[name] = value
            changed = True
    return values


def build_model(before: bytes, patch_source: bytes) -> bytes:
    values = literal_bindings(patch_source)
    required = {
        "HEAD_PAD",
        "ALOG_PAD",
        "load_pads",
        "old_emb",
        "new_emb",
        "old_lm",
        "new_lm",
        "old_shared",
        "new_shared",
        "old_mlp",
        "new_mlp",
        "old_down",
        "new_down",
    }
    if not required.issubset(values):
        raise Refusal(f"TP3 patch-data bindings missing: {sorted(required - values.keys())}")
    text = before.decode("utf-8")
    text = replace_once(
        text,
        "from collections.abc import Iterable\n",
        "from collections.abc import Iterable\nfrom math import gcd\n",
        "gcd import",
    )
    anchor = "        config = vllm_config.model_config.hf_config\n        self.config = config\n"
    text = replace_once(text, anchor, anchor + str(values["HEAD_PAD"]), "head pad")
    load = "    def load_weights(self, weights:"
    index = text.find(load)
    if index < 0:
        raise Refusal("load_weights seam missing")
    newline = text.find("\n", index)
    text = text[: newline + 1] + str(values["ALOG_PAD"]) + text[newline + 1 :]
    load_pads = values["load_pads"]
    if not isinstance(load_pads, tuple):
        raise Refusal("load-pad plan is not a literal tuple")
    for ordinal, pair in enumerate(load_pads, 1):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise Refusal("load-pad pair drift")
        text = replace_once(text, str(pair[0]), str(pair[1]), f"loader {ordinal}")
    for old_name, new_name, label in (
        ("old_emb", "new_emb", "embedding"),
        ("old_lm", "new_lm", "lm-head"),
        ("old_shared", "new_shared", "shared-width"),
        ("old_mlp", "new_mlp", "shared-gate"),
        ("old_down", "new_down", "shared-down"),
    ):
        replacement = str(values[new_name])
        if old_name == "old_mlp":
            replacement = "            # TP3-SHARED-MLP\n" + replacement
        text = replace_once(text, str(values[old_name]), replacement, label)
    return text.encode()


def build_flash(source: bytes) -> bytes:
    text = source.decode("utf-8")
    for ordinal, (old, new) in enumerate(FLASH_EDITS, 1):
        text = replace_once(text, old, new, f"FlashInfer TP3 overlay delta {ordinal}")
    return text.encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
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
        source_root = args.source_root.resolve(strict=True)
        source_files = {
            name: source_root / relative for name, relative in SOURCE_PATHS.items()
        }
        for name, path in source_files.items():
            expected = TP3["sources"][name]
            if sha_file(path) != expected:
                raise Refusal(f"pinned Fly source drift: {name}")

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            paths = {name: safe_target(root, name) for name in (MODEL, EXL3, VOCAB, PARAMETER, FLASH)}
            return {
                paths[MODEL]: build_model(before[paths[MODEL]], source_files["patch_tp3_glm.py"].read_bytes()),
                paths[EXL3]: source_files["exl3.py"].read_bytes(),
                paths[VOCAB]: source_files["vocab_parallel_embedding.py"].read_bytes(),
                paths[PARAMETER]: source_files["parameter.py"].read_bytes(),
                paths[FLASH]: build_flash(source_files["flashinfer_mla_sparse_sm120.py"].read_bytes()),
            }

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_tp3_overlay.py",
            expected_section=TP3,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        print_receipt(receipt)
        return 0
    except (OSError, SyntaxError, ValueError, UnicodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
