#!/usr/bin/env python3
"""Install JSpark3 v1.6's boot-time TP3 ABI2 cooperative-MoE adapter.

State ``0`` is an exact no-op over v1.5's stage-8 EXL3 bytes. State ``1``
requires a twice-built, newly profiled, hash-bound native bundle and appends a
sealed footer to that exact stage-8 file. The footer runs before model
construction, so the adapter prepares shared scratch after weight loading and
before FULL CUDA-graph capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from _atomic import (
    IMAGE_CONFIG,
    IMAGE_MANIFEST,
    Refusal,
    canonical,
    execute,
    load_contract,
    read_image_receipt,
    safe_target,
    sha_bytes,
)

OVERLAY = Path(__file__).resolve().parent.parent / "overlays" / "v16" / "coop"
SOURCE_ROOT = OVERLAY / "source"
SOURCE_MANIFEST = OVERLAY / "SOURCE_MANIFEST.json"
INSTALL_CONTRACT = OVERLAY / "INSTALL_CONTRACT.json"
DEFAULT_BUNDLE = OVERLAY / "bundle"
DEFAULT_BUILD_RECORD = OVERLAY / "BUILD.json"

TRANSFORM = "apply_coop_moe.py"
EXL3 = "vllm/model_executor/layers/quantization/exl3.py"
V15_EXL3_SHA256 = "71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174"
COOP_EXL3_SHA256 = "89111aaf1d3082dd62c76ecb8c378fabf3f98619bee5124ee4ac3622e1e2eb4f"
SOURCE_MANIFEST_SHA256 = "38d93fdef5c10cece0d97e66fd402d9669598bf09c784af69db706f469cce9da"
RUNTIME_SHA256 = "65660930bc9965ff7b7379afaa270e0f80f418276bad4c42c10dbdee7ac3d0eb"

FOOTER = b'''
# [jspark3-v16-coop] Boot-time TP3 ABI2 cooperative MoE. This footer is
# installed only for JSPARK3_V16_COOP=1 and runs before weight load/graph capture.
import runpy as _jspark3_coop_runpy
import sys as _jspark3_coop_sys
_jspark3_coop_setup = _jspark3_coop_runpy.run_path(
    "/recipe/overlays/v16/coop/bundle/runtime.py"
)
_jspark3_coop_setup["install"](
    _jspark3_coop_sys.modules[__name__],
    library_root="/recipe/overlays/v16/coop/bundle",
    enabled=True,
)
del _jspark3_coop_setup, _jspark3_coop_runpy, _jspark3_coop_sys
'''

EXPECTED_SECTION = {
    "sources": {
        "SOURCE_MANIFEST.json": SOURCE_MANIFEST_SHA256,
        "source/runtime.py": RUNTIME_SHA256,
    },
    "targets": [
        {
            "path": EXL3,
            "before_sha256": V15_EXL3_SHA256,
            "after_sha256": COOP_EXL3_SHA256,
            "required_before_seams": [
                {"text": "def apply_exl3_fused_moe(\n", "count": 1},
                {"text": "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n", "count": 1},
                {"text": "_jspark3_fatpath.probe(", "count": 1},
                {"text": "_jspark3_fatpath.dispatch(\n", "count": 1},
                {"text": "_exl3_expert_map_device", "count": 2},
            ],
            "required_after_seams": [
                {"text": "# [jspark3-v16-coop]", "count": 1},
                {"text": "_jspark3_coop_setup[\"install\"](\n", "count": 1},
                {"text": "_jspark3_fatpath.probe(", "count": 1},
                {"text": "_jspark3_fatpath.dispatch(\n", "count": 1},
                {"text": "_exl3_expert_map_device", "count": 2},
            ],
            "forbidden_after_seams": [],
        }
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _object(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"{label} missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"{label} must be an object")
    return value


def verify_sources() -> None:
    value = _object(SOURCE_MANIFEST, "cooperative-MoE source manifest")
    if digest(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise Refusal("cooperative-MoE source manifest drift")
    if value.get("schema_version") != 1:
        raise Refusal("cooperative-MoE source manifest schema drift")
    if SOURCE_ROOT.is_symlink() or not SOURCE_ROOT.is_dir() or any(
        path.is_symlink() for path in SOURCE_ROOT.rglob("*")
    ):
        raise Refusal("cooperative-MoE source tree is missing or contains a symlink")
    expected = value.get("files")
    observed = {
        path.relative_to(SOURCE_ROOT).as_posix(): digest(path)
        for path in sorted(SOURCE_ROOT.rglob("*")) if path.is_file()
    }
    if expected != observed:
        raise Refusal("pinned cooperative-MoE source drift")
    if observed.get("runtime.py") != RUNTIME_SHA256:
        raise Refusal("pinned cooperative-MoE runtime drift")


def _safe_bundle_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise Refusal(f"unsafe bundle path: {relative}")
    path = root.joinpath(rel)
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"bundle file missing or symlinked: {relative}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise Refusal(f"bundle path escapes root: {relative}")
    return path


def verify_bundle(bundle: Path, build_record: Path) -> dict:
    verify_sources()
    if bundle.is_symlink() or not bundle.is_dir():
        raise Refusal("cooperative-MoE bundle root is missing or symlinked")
    record = _object(build_record, "cooperative-MoE build record")
    if set(record) != {
        "schema_version", "image", "source_manifest_sha256", "reproducibility",
        "qualification", "bundle",
    } or record.get("schema_version") != 1:
        raise Refusal("cooperative-MoE build record schema drift")
    if record.get("image") != {"manifest": IMAGE_MANIFEST, "config": IMAGE_CONFIG}:
        raise Refusal("cooperative-MoE build image drift")
    if record.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise Refusal("cooperative-MoE build source drift")

    reproducibility = record.get("reproducibility")
    if (
        not isinstance(reproducibility, dict)
        or set(reproducibility) != {"runs", "comparison", "binary_sha256"}
        or reproducibility.get("runs") != 2
        or reproducibility.get("comparison") != "bit-identical"
        or not is_sha256(reproducibility.get("binary_sha256"))
    ):
        raise Refusal("cooperative-MoE binary lacks two-build bit identity")
    qualification = record.get("qualification")
    profiles = qualification.get("profile_log_sha256") if isinstance(qualification, dict) else None
    if (
        not isinstance(qualification, dict)
        or set(qualification) != {"ep_ranks", "geometries", "cases", "profile_log_sha256"}
        or not isinstance(profiles, dict)
        or set(profiles) != {f"rank{rank}-geo{geometry}.jsonl"
                             for rank in range(3) for geometry in range(3)}
        or not all(is_sha256(value) for value in profiles.values())
        or qualification.get("cases") != 468
        or qualification.get("ep_ranks") != [0, 1, 2]
        or qualification.get("geometries") != [0, 1, 2]
    ):
        raise Refusal("cooperative-MoE build lacks complete 3x3 profile qualification")

    manifest_path = _safe_bundle_file(bundle, "manifest.json")
    manifest = _object(manifest_path, "cooperative-MoE bundle manifest")
    expected_contract = {
        "abi": 2,
        "hidden": 4096,
        "intermediate_local": 2048,
        "topk": 8,
        "experts_max": 96,
        "row_capacities": [32, 64],
        "counter_lengths": {"32": 5635, "64": 11267},
        "params_size": 344,
        "activation_limit": 10.0,
    }
    if manifest.get("schema") != 1 or manifest.get("contract") != expected_contract:
        raise Refusal("cooperative-MoE bundle ABI/shape contract drift")
    files = manifest.get("files")
    if not isinstance(files, dict) or not {
        "cooperative_moe.so", "runtime.py", "dispatch_policy.json", "PROVENANCE.json",
        "LICENSE.MIT", "LICENSE.upstream-AGPL-3.0", "LICENSE.exllamav3",
    }.issubset(files):
        raise Refusal("incomplete cooperative-MoE bundle manifest")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise Refusal("malformed cooperative-MoE bundle manifest")
        if digest(_safe_bundle_file(bundle, relative)) != expected_hash:
            raise Refusal(f"cooperative-MoE bundle digest drift: {relative}")

    native = files["cooperative_moe.so"]
    policy = _object(_safe_bundle_file(bundle, "dispatch_policy.json"), "cooperative-MoE row policy")
    rows = policy.get("rows")
    if (
        policy.get("schema") != 1 or policy.get("native_sha256") != native
        or not isinstance(rows, dict) or set(rows) != {str(row) for row in range(1, 65)}
        or not all(value == "stock" or type(value) is int and value in (0, 1, 2)
                   for value in rows.values())
    ):
        raise Refusal("cooperative-MoE row policy is incomplete or not bound to this binary")
    binary = _safe_bundle_file(bundle, "cooperative_moe.so")
    if not binary.read_bytes().startswith(b"\x7fELF"):
        raise Refusal("cooperative-MoE native artifact is not ELF")
    if digest(_safe_bundle_file(bundle, "runtime.py")) != RUNTIME_SHA256:
        raise Refusal("bundle runtime does not equal the pinned adapter")
    for relative in ("PROVENANCE.json", "LICENSE.MIT", "LICENSE.upstream-AGPL-3.0"):
        if digest(_safe_bundle_file(bundle, relative)) != digest(SOURCE_ROOT / relative):
            raise Refusal(f"bundle source/license does not equal the pinned copy: {relative}")
    if digest(_safe_bundle_file(bundle, "LICENSE.exllamav3")) != digest(SOURCE_ROOT / "vendor/LICENSE.exllamav3"):
        raise Refusal("bundle ExLlamaV3 license does not equal the pinned copy")

    bundle_record = record.get("bundle")
    expected_record = {
        "manifest_sha256": digest(manifest_path),
        "native_sha256": native,
        "runtime_sha256": files["runtime.py"],
        "dispatch_policy_sha256": files["dispatch_policy.json"],
    }
    if (
        not isinstance(bundle_record, dict)
        or set(bundle_record) != set(expected_record)
        or bundle_record != expected_record
        or reproducibility.get("binary_sha256") != native
    ):
        raise Refusal("cooperative-MoE build record does not bind the served bundle")
    return {"native_sha256": native, "policy_sha256": files["dispatch_policy.json"]}


def compose(before: bytes) -> bytes:
    if sha_bytes(before) != V15_EXL3_SHA256:
        raise Refusal("cooperative-MoE requires exact v1.5 stage-8 EXL3 bytes")
    result = before + FOOTER
    if sha_bytes(result) != COOP_EXL3_SHA256:
        raise Refusal("cooperative-MoE generated EXL3 hash mismatch")
    return result


def identity(rank: int, state: str, exl3_sha: str, native: str = "none") -> None:
    print(
        f"[jspark3-v16:coop] rank={rank} state={state} boot_time_only=1 "
        f"exl3_sha256={exl3_sha} native_sha256={native}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=INSTALL_CONTRACT)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--build-record", type=Path, default=DEFAULT_BUILD_RECORD)
    parser.add_argument("--state", choices=("0", "1"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        state = args.state if args.state is not None else os.environ.get("JSPARK3_V16_COOP")
        if state not in ("0", "1"):
            raise Refusal("JSPARK3_V16_COOP must be explicitly 0 or 1")
        receipt = read_image_receipt(args.image_receipt)
        rank = receipt["rank"]
        node_rank = os.environ.get("NODE_RANK")
        if node_rank is not None and node_rank != str(rank):
            raise Refusal("NODE_RANK disagrees with the image receipt")
        vllm = args.vllm_root.resolve(strict=True)
        root = vllm.parent
        if (root / "vllm").resolve(strict=True) != vllm:
            raise Refusal("--vllm-root must name the vllm package directory")
        exl3 = safe_target(root, EXL3)
        load_contract(args.contract, TRANSFORM, EXPECTED_SECTION)

        overrides = args.bundle_root != DEFAULT_BUNDLE or args.build_record != DEFAULT_BUILD_RECORD
        if overrides and os.environ.get("JSPARK3_V16_COOP_MAINTENANCE") != "1":
            raise Refusal("bundle/build-record overrides require the explicit maintenance gate")

        if state == "0":
            observed = digest(exl3)
            if observed != V15_EXL3_SHA256:
                raise Refusal(f"coop-off requires exact v1.5 EXL3 bytes, got {observed}")
            receipt_out = {
                "schema_version": 1,
                "transform": TRANSFORM,
                "state": "DISABLED_EXACT_V15",
                "contract_sha256": digest(args.contract),
                "target": {"path": EXL3, "observed_sha256": observed},
            }
            identity(rank, "off", observed)
            sys.stdout.buffer.write(canonical(receipt_out))
            return 0

        bundle_identity = verify_bundle(args.bundle_root, args.build_record)

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            return {exl3: compose(before[exl3])}

        receipt_out = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform=TRANSFORM,
            expected_section=EXPECTED_SECTION,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        observed = digest(exl3)
        identity(rank, "on", observed, bundle_identity["native_sha256"])
        sys.stdout.buffer.write(canonical(receipt_out))
        return 0
    except (OSError, ValueError, UnicodeError, KeyError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
