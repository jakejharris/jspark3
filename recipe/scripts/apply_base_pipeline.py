#!/usr/bin/env python3
"""Resume the five exact base-recipe transforms from a hash-proven stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

from _atomic import (ABSENT, Refusal, canonical, compiled, observed,
                     read_image_receipt, safe_target, sha_file,
                     transaction_debris, transaction_names)

STAGES = (
    "apply_tp3_overlay.py",
    "apply_image_glm_dflash.py",
    "apply_kpool_tail.py",
    "apply_kda_mixed.py",
    "apply_kda_fg.py",
)
MODULES = {
    "apply_tp3_overlay.py": "apply_tp3_overlay",
    "apply_image_glm_dflash.py": "apply_image_glm_dflash",
}


def contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value.get("transforms", {})) != set(STAGES):
        raise Refusal("pipeline transform inventory drift")
    return value


def snapshots(value: dict) -> tuple[list[dict[str, str]], dict[str, dict]]:
    state: dict[str, str] = {}
    records: dict[str, dict] = {}
    for stage in STAGES:
        section = value["transforms"][stage]
        targets = section.get("targets")
        if not isinstance(targets, list) or not targets:
            raise Refusal(f"{stage}: empty target inventory")
        for record in targets:
            path = str(record["path"])
            state.setdefault(path, str(record["before_sha256"]))
    result = []
    for stage in STAGES:
        targets = value["transforms"][stage]["targets"]
        for record in targets:
            path = str(record["path"])
            if state[path] != str(record["before_sha256"]):
                raise Refusal(f"{stage}: transform chain does not join at {path}")
        result.append(dict(state))
        for record in targets:
            state[str(record["path"])] = str(record["after_sha256"])
            records[str(record["path"])] = record
    result.append(dict(state))
    return result, records


def verify_sources(source_root: Path, value: dict) -> None:
    for stage, module_name in MODULES.items():
        module = importlib.import_module(module_name)
        expected = value["transforms"][stage]["sources"]
        for name, relative in module.SOURCE_PATHS.items():
            path = source_root / relative
            if not path.is_file() or sha_file(path) != expected[name]:
                raise Refusal(f"{stage}: pinned source drift: {name}")


def verify_fixed(root: Path, asset_root: Path, value: dict) -> None:
    for record in value["transforms"]["apply_image_glm_dflash.py"].get("verify_only", []):
        path = safe_target(root, str(record["path"])) if "path" in record else asset_root / str(record["asset_path"])
        if not path.is_file() or sha_file(path) != record["sha256"]:
            raise Refusal(f"image verify-only identity drift: {path.name}")


def stage(root: Path, states: list[dict[str, str]]) -> int:
    matches = []
    union = states[-1]
    for number, expected in enumerate(states):
        if all(observed(safe_target(root, path)) == expected.get(path, ABSENT) for path in union):
            matches.append(number)
    if len(matches) != 1:
        raise Refusal("pipeline target set is mixed, partial, or unknown")
    return matches[0]


def command(args: argparse.Namespace, name: str) -> list[str]:
    base = [sys.executable, str(Path(__file__).resolve().with_name(name)),
            "--vllm-root", str(args.vllm_root), "--contract", str(args.contract),
            "--image-receipt", str(args.image_receipt), "--apply"]
    if name == "apply_tp3_overlay.py":
        base.extend(("--source-root", str(args.source_root)))
    elif name == "apply_image_glm_dflash.py":
        base.extend(("--source-root", str(args.source_root), "--asset-root", str(args.asset_root)))
    return base


def pending_transactions(root: Path, value: dict) -> list[str]:
    pending = []
    for name in STAGES:
        records = value["transforms"][name]["targets"]
        journal, files = transaction_names(root, name, records)
        if transaction_debris(root, name, journal, files):
            pending.append(name)
    return pending


def recover_journals(args: argparse.Namespace, root: Path, names: list[str]) -> list[dict]:
    receipts = []
    for name in names:
        process = subprocess.run(command(args, name), text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, check=False)
        if process.returncode:
            raise Refusal(f"{name}: transaction recovery failed: {process.stderr.strip()}")
        receipts.append(json.loads(process.stdout))
    return receipts


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
        args.vllm_root = vllm
        args.source_root = args.source_root.resolve(strict=True)
        args.asset_root = args.asset_root.resolve(strict=True)
        args.contract = args.contract.resolve(strict=True)
        args.image_receipt = args.image_receipt.resolve(strict=True)
        read_image_receipt(args.image_receipt)
        value = contract(args.contract)
        verify_sources(args.source_root, value)
        verify_fixed(root, args.asset_root, value)
        states, final_records = snapshots(value)
        pending = pending_transactions(root, value)
        if pending and not args.apply:
            raise Refusal("prepared transaction or orphan artifact requires --apply recovery")
        receipts = recover_journals(args, root, pending) if args.apply else []
        current = stage(root, states)
        if not args.apply:
            status = "ALREADY_APPLIED" if current == len(STAGES) else f"READY_STAGE_{current}"
        else:
            for name in STAGES[current:]:
                process = subprocess.run(command(args, name), text=True, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, check=False)
                if process.returncode:
                    raise Refusal(f"{name}: apply failed: {process.stderr.strip()}")
                receipts.append(json.loads(process.stdout))
            if stage(root, states) != len(STAGES):
                raise Refusal("pipeline did not reach exact final state")
            status = "ALREADY_APPLIED" if current == len(STAGES) and not receipts else "APPLIED"
        if current == len(STAGES) or args.apply:
            for path, record in final_records.items():
                target = safe_target(root, path)
                if observed(target) != str(record["after_sha256"]):
                    raise Refusal(f"final target drift: {path}")
                if target.is_file():
                    compiled(target, target.read_bytes())
        result = {
            "schema_version": 1, "state": status, "detected_stage": current,
            "final_stage": len(STAGES), "transforms_executed": [row["transform"] for row in receipts],
            "contract_sha256": sha_file(args.contract),
            "pipeline_sha256": sha_file(Path(__file__).resolve()),
            "target_set_sha256": hashlib.sha256(canonical(states[-1])).hexdigest(),
        }
        sys.stdout.buffer.write(canonical(result))
        return 0
    except (OSError, SyntaxError, ValueError, KeyError, TypeError, json.JSONDecodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
