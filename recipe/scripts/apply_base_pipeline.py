#!/usr/bin/env python3
"""Resume the eight exact base-recipe transforms from a hash-proven stage.

Stages 1-5 are the frozen v1.3.0 set; stages 6-7 are the v1.4 core
(pinned Mia/vLLM installers) and the display-reserve KV wiring; stage 8 is
the v1.5 EXL3 fat-expert path.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from _atomic import (ABSENT, Refusal, canonical, compiled, observed,
                     read_image_receipt, safe_target, sha_file,
                     transaction_debris, transaction_names)

STAGES = (
    "apply_tp3_overlay.py",
    "apply_image_glm_dflash.py",
    "apply_kpool_tail.py",
    "apply_kda_mixed.py",
    "apply_kda_fg.py",
    "apply_v14_core.py",
    "apply_display_kv.py",
    "apply_exl3_fatpath.py",
)
MODULES = {
    "apply_tp3_overlay.py": "apply_tp3_overlay",
    "apply_image_glm_dflash.py": "apply_image_glm_dflash",
}
V16_NORMALIZED = {
    ("JSPARK3_V16_COOP", "1", "vllm/model_executor/layers/quantization/exl3.py"):
        ("89111aaf1d3082dd62c76ecb8c378fabf3f98619bee5124ee4ac3622e1e2eb4f",
         "71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174"),
    ("GLM53_ADAPTIVE_K", "ema", "vllm/v1/core/sched/scheduler.py"):
        ("0d58e688019ddfa5952990be2ea751b71d96bea7a35c4c31c19c672b0890f764",
         "0b086dd3cc0febc4924202ca1fef8809b8290b9a59ef15fa80004cef4edeb937"),
    ("GLM53_ADAPTIVE_K", "ema", "vllm/v1/worker/gpu/cudagraph_utils.py"):
        ("6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa",
         "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f"),
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


def verify_v14_sources() -> None:
    """Stages 6 and 7 pin their sources inside the recipe. Check them on every
    run, including a final-state --check that executes no stage."""
    for module_name in ("apply_v14_core", "apply_display_kv"):
        importlib.import_module(module_name).verify_sources()


def verify_v15_sources() -> None:
    """Stage 8 pins its sources inside the recipe too; checked on every run."""
    importlib.import_module("apply_exl3_fatpath").verify_sources()


def verify_fixed(root: Path, asset_root: Path, value: dict) -> None:
    for record in value["transforms"]["apply_image_glm_dflash.py"].get("verify_only", []):
        path = safe_target(root, str(record["path"])) if "path" in record else asset_root / str(record["asset_path"])
        got = normalized_observed(root, str(record.get("path", "")), path)
        if not path.is_file() or got != record["sha256"]:
            raise Refusal(f"image verify-only identity drift: {path.name}")


def normalized_observed(root: Path, relative: str, path: Path | None = None) -> str:
    """Project a sealed v1.6 option target back onto its exact v1.5 input.

    This is used only by ``--check``/resume detection after optional transforms
    have run.  The base stage inventory and its hashes remain unchanged.
    """
    got = observed(path if path is not None else safe_target(root, relative))
    # The always-on S9.11 startup hook is a separate sealed post-base transform.
    # Normalize only its exact worker hash; never accept arbitrary worker edits.
    from _contracts import V16_WARMJIT
    worker = V16_WARMJIT["targets"][0]
    if relative == worker["path"] and got == worker["after_sha256"]:
        return worker["before_sha256"]
    for (name, enabled, target), (option_hash, base_hash) in V16_NORMALIZED.items():
        if target == relative and os.environ.get(name) == enabled and got == option_hash:
            return base_hash
    return got


def stage(root: Path, states: list[dict[str, str]]) -> int:
    matches = []
    union = states[-1]
    for number, expected in enumerate(states):
        if all(normalized_observed(root, path) == expected.get(path, ABSENT) for path in union):
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


def run_stage(args: argparse.Namespace, name: str, mode: str) -> dict:
    """Run one stage child and collect its receipt on the dedicated file channel.

    Stage stdout is shared with interpreter startup chatter (site .pth imports),
    so it can never serve as a JSON channel. The child writes canonical receipt
    bytes to a fresh, nonexistent file named by JSPARK3_RECEIPT_OUT; the parent
    reads that file only after the child exits successfully.
    """
    with tempfile.TemporaryDirectory(prefix="stage-receipt-") as scratch:
        receipt_path = Path(scratch) / f"{name}.receipt.json"
        environment = dict(os.environ)
        environment["JSPARK3_RECEIPT_OUT"] = str(receipt_path)
        process = subprocess.run(command(args, name), text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, check=False, env=environment)
        diagnostics.retain(process.stdout + process.stderr)
        if process.returncode:
            raise Refusal(f"{name}: transaction {mode} failed: {process.stderr.strip()}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Refusal(f"{name}: stage receipt unavailable on receipt channel: {exc}") from exc
        if not isinstance(receipt, dict):
            raise Refusal(f"{name}: stage receipt is not a JSON object")
        if receipt.get("schema_version") != 1 or receipt.get("transform") != name \
                or receipt.get("state") not in ("APPLIED", "ALREADY_APPLIED") \
                or not isinstance(receipt.get("targets"), list):
            raise Refusal(f"{name}: stage receipt rejected")
        return receipt


def recover_journals(args: argparse.Namespace, root: Path, names: list[str]) -> list[dict]:
    return [run_stage(args, name, "recovery") for name in names]


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
        verify_v14_sources()
        verify_v15_sources()
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
                receipts.append(run_stage(args, name, "apply"))
            if stage(root, states) != len(STAGES):
                raise Refusal("pipeline did not reach exact final state")
            status = "ALREADY_APPLIED" if current == len(STAGES) and not receipts else "APPLIED"
        if current == len(STAGES) or args.apply:
            for path, record in final_records.items():
                target = safe_target(root, path)
                if normalized_observed(root, path, target) != str(record["after_sha256"]):
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
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
