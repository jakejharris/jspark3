#!/usr/bin/env python3
"""Reproduce the base recipe's image-provided GLM/DFlash seams without running patchers."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

from _atomic import Refusal, execute, print_receipt, safe_target, sha_file
from _contracts import IMAGE_DFLASH

MODEL = "vllm/models/glm5next/nvidia/model.py"
KV = "vllm/v1/core/kv_cache_utils.py"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
DETOKENIZER = "vllm/v1/engine/detokenizer.py"
VIDEO = "glm53_video_patch.py"
VIDEO_PTH = "glm53_video.pth"
KPOOL_VERIFY = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
QWEN2 = "vllm/model_executor/models/qwen3_dflash2.py"
SPECULATOR = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
SPEC_INIT = "vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py"
QWEN = "vllm/model_executor/models/qwen3_dflash.py"
REGISTRY = "vllm/model_executor/models/registry.py"
DFLASH_UTILS = "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"
DECODE_INIT = "vllm/v1/worker/gpu/spec_decode/__init__.py"

SOURCE_PATHS = {
    "chat_template.jinja": "files/chat_template.jinja",
    "dflash2_speculator.py": "overlay/miaai/dflash2_speculator.py",
    "qwen3_dflash2.py": "overlay/miaai/qwen3_dflash2.py",
    "patch_dflash2.py": "overlay/miaai/patch_dflash2.py",
    "patch_exl3_ext_aarch64.py": "overlay/miaai/patch_exl3_ext_aarch64.py",
    "patch_glm5_drafter_group.py": "overlay/miaai/patch_glm5_drafter_group.py",
    "patch_glm_eagle3.py": "overlay/miaai/patch_glm_eagle3.py",
    "patch_glm_video_placeholders.py": "overlay/miaai/patch_glm_video_placeholders.py",
    "patch_model_overrides.py": "overlay/miaai/patch_model_overrides.py",
    "patch_scheduler_decode_floor.py": "overlay/miaai/patch_scheduler_decode_floor.py",
    "patch_suppress_stops_in_reasoning.py": "overlay/miaai/patch_suppress_stops_in_reasoning.py",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise Refusal(f"{label}: expected one source seam, found {count}")
    return text.replace(old, new, 1)


def literal_assignments(source: bytes) -> dict[str, object]:
    tree = ast.parse(source.decode("utf-8"))
    values: dict[str, object] = {}
    for node in sorted(
        (item for item in ast.walk(tree) if isinstance(item, ast.Assign)),
        key=lambda item: item.lineno,
    ):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return values


def call_pairs(source: bytes, arity: int, first_name: str | None = None) -> list[tuple[str, str]]:
    tree = ast.parse(source.decode("utf-8"))
    pairs: list[tuple[str, str]] = []
    calls = sorted(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "replace_once"
        ),
        key=lambda item: item.lineno,
    )
    for call in calls:
        if len(call.args) != arity:
            continue
        offset = 0
        if first_name is not None:
            first = call.args[0]
            if not isinstance(first, ast.Name) or first.id != first_name:
                continue
            offset = 1
        try:
            old = ast.literal_eval(call.args[offset])
            new = ast.literal_eval(call.args[offset + 1])
        except (ValueError, TypeError):
            continue
        if isinstance(old, str) and isinstance(new, str):
            pairs.append((old, new))
    return pairs


def eagle(before: bytes, patch: bytes) -> bytes:
    # The pinned image already has the Eagle interface. Its historical build
    # reapplied the substring-containing aux-state seam once, producing the
    # exact duplicated declaration captured by the base recipe. Reproduce only that
    # exact delta; never execute the non-idempotent upstream patcher.
    if len(call_pairs(patch, 2)) != 7:
        raise Refusal("Eagle source patch plan drift")
    text = before.decode("utf-8")
    old = "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    text = replace_once(text, old, old + old, "Eagle historical second insertion")
    return text.encode()


def drafter_group(before: bytes, patch: bytes) -> bytes:
    values = literal_assignments(patch)
    try:
        v4_old = str(values["v4_old"])
        v4_new = str(values["v4_new"])
        new_padded = str(values["new_padded"])
        old_standalone = str(values["old_standalone"])
    except KeyError as exc:
        raise Refusal("drafter-group upgrade literals missing") from exc
    text = replace_once(before.decode("utf-8"), v4_old, v4_new, "drafter-group v4 validation")
    if text.count(old_standalone) == 1:
        return text.replace(old_standalone, new_padded, 1).encode()
    start_marker = "            # STANDALONE: compact per-layer tensors."
    end_marker = "        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)"
    start, end = text.find(start_marker), text.find(end_marker)
    if start < 0 or end < 0 or end <= start or text.count(start_marker) != 1:
        raise Refusal("drafter-group compact block boundary drift")
    return (text[:start] + new_padded + text[end:]).encode()


def scheduler(before: bytes, patch: bytes) -> bytes:
    values = literal_assignments(patch)
    required = {"IMPORT_OLD", "IMPORT_NEW", "HELPER", "RUNNING_OLD", "RUNNING_NEW", "WAITING_OLD", "WAITING_NEW"}
    if not required.issubset(values):
        raise Refusal(f"scheduler patch-data drift: {sorted(required - values.keys())}")
    text = before.decode("utf-8")
    text = replace_once(text, str(values["IMPORT_OLD"]), str(values["IMPORT_NEW"]), "scheduler import")
    anchor = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    text = replace_once(text, anchor, str(values["HELPER"]) + anchor, "scheduler helper")
    text = replace_once(text, str(values["RUNNING_OLD"]), str(values["RUNNING_NEW"]), "scheduler running")
    text = replace_once(text, str(values["WAITING_OLD"]), str(values["WAITING_NEW"]), "scheduler waiting")
    return text.encode()


def detokenizer(before: bytes, patch: bytes) -> bytes:
    values = literal_assignments(patch)
    names = ("IMPORT", "FACTORY", "INIT", "STOP")
    text = before.decode("utf-8")
    for name in names:
        old, new = values.get(f"{name}_OLD"), values.get(f"{name}_NEW")
        if not isinstance(old, str) or not isinstance(new, str):
            raise Refusal(f"detokenizer {name.lower()} patch-data drift")
        text = replace_once(text, old, new, f"detokenizer {name.lower()}")
    return text.encode()


def registry(before: bytes, patch: bytes) -> bytes:
    pairs = call_pairs(patch, 3, "registry")
    if len(pairs) != 1:
        raise Refusal(f"DFlash registry patch plan drift: {len(pairs)} edits")
    return replace_once(before.decode("utf-8"), pairs[0][0], pairs[0][1], "DFlash registry second insertion").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
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
        asset_root = args.asset_root.resolve(strict=True)
        source_files = {name: source_root / relative for name, relative in SOURCE_PATHS.items()}
        for name, path in source_files.items():
            if sha_file(path) != IMAGE_DFLASH["sources"][name]:
                raise Refusal(f"pinned Fly source drift: {name}")
        for record in IMAGE_DFLASH["verify_only"]:
            if "path" in record:
                path = safe_target(root, str(record["path"]))
            else:
                path = asset_root / str(record["asset_path"])
            if sha_file(path) != record["sha256"]:
                raise Refusal(f"image verify-only seam drift: {path.name}")

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            paths = {
                name: safe_target(root, name)
                for name in (
                    MODEL, KV, SCHEDULER, DETOKENIZER, VIDEO, VIDEO_PTH,
                    KPOOL_VERIFY, QWEN2, SPECULATOR, SPEC_INIT, QWEN,
                    REGISTRY, DFLASH_UTILS, DECODE_INIT,
                )
            }
            outputs = {
                paths[MODEL]: eagle(before[paths[MODEL]], source_files["patch_glm_eagle3.py"].read_bytes()),
                paths[KV]: drafter_group(before[paths[KV]], source_files["patch_glm5_drafter_group.py"].read_bytes()),
                paths[SCHEDULER]: scheduler(before[paths[SCHEDULER]], source_files["patch_scheduler_decode_floor.py"].read_bytes()),
                paths[DETOKENIZER]: detokenizer(before[paths[DETOKENIZER]], source_files["patch_suppress_stops_in_reasoning.py"].read_bytes()),
                paths[VIDEO]: source_files["patch_glm_video_placeholders.py"].read_bytes(),
                paths[VIDEO_PTH]: b"import glm53_video_patch\n",
                paths[REGISTRY]: registry(before[paths[REGISTRY]], source_files["patch_dflash2.py"].read_bytes()),
            }
            for fixed in (KPOOL_VERIFY, QWEN2, SPECULATOR, SPEC_INIT, QWEN, DFLASH_UTILS, DECODE_INIT):
                outputs[paths[fixed]] = before[paths[fixed]]
            return outputs

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_image_glm_dflash.py",
            expected_section=IMAGE_DFLASH,
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
