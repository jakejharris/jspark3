#!/usr/bin/env python3
"""Apply the JSpark3 v1.6 dense-FP8 boot overlay transactionally.

The entrypoint invokes this transform only for ``JSPARK3_V16_DENSE_FP8`` modes
other than ``off``.  Skipping it in off mode is what preserves the exact v1.5
installed bytes and code path.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path

from _atomic import Refusal, execute, print_receipt, safe_target


OVERLAY = Path(__file__).resolve().parent.parent / "overlays" / "v16" / "dense_fp8"
PATCHER = "patch_ablit_dense_fp8.py"
MODULE = "jspark3_dense_fp8.py"
REFERENCE = "jspark3_dense_fp8_reference.py"

ABLIT = "vllm/model_executor/layers/quantization/ablit_transplant.py"
INSTALLED_MODULE = "vllm/model_executor/layers/quantization/jspark3_dense_fp8.py"
INSTALLED_REFERENCE = (
    "vllm/model_executor/layers/quantization/jspark3_dense_fp8_reference.py"
)
EDITED = (ABLIT,)
INSTALLED = (INSTALLED_MODULE, INSTALLED_REFERENCE)
TRANSFORM = "apply_dense_fp8.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refusal(f"cannot load sealed dense-FP8 source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_module():
    return _load_source("jspark3_dense_fp8_reference_build", OVERLAY / REFERENCE)


def patcher_module():
    return _load_source("patch_ablit_dense_fp8_build", OVERLAY / PATCHER)


def expected_section() -> dict:
    try:
        import _contracts

        section = _contracts.V16_DENSE_FP8
    except (ImportError, AttributeError) as exc:
        raise Refusal(
            "v1.6 dense-FP8 contract is not wired into recipe/scripts/_contracts.py"
        ) from exc
    if not isinstance(section, dict):
        raise Refusal("v1.6 dense-FP8 embedded contract is not an object")
    return section


def verify_sources(section: dict | None = None) -> None:
    section = expected_section() if section is None else section
    expected = section.get("sources", {})
    names = (PATCHER, MODULE, REFERENCE)
    if set(expected) != set(names):
        raise Refusal("dense-FP8 source inventory drift")
    for name in names:
        path = OVERLAY / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected[name]:
            raise Refusal(f"pinned dense-FP8 source drift: {name}")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    """Patch the staged v1.5 finalizer and install the two sealed modules."""

    if set(files) != set(EDITED):
        raise Refusal("dense-FP8 target-set drift")
    try:
        before = files[ABLIT].decode("utf-8")
        patched = patcher_module().patch_text(before).encode("utf-8")
    except (UnicodeError, SyntaxError, ValueError) as exc:
        raise Refusal(f"{PATCHER}: installer refused: {exc}") from None
    return {
        ABLIT: patched,
        INSTALLED_MODULE: (OVERLAY / MODULE).read_bytes(),
        INSTALLED_REFERENCE: (OVERLAY / REFERENCE).read_bytes(),
    }


def run_transform(
    *,
    vllm: Path,
    contract: Path,
    image_receipt: Path,
    section: dict,
    apply: bool,
) -> dict:
    root = vllm.parent

    def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
        paths = {rel: safe_target(root, rel) for rel in (*EDITED, *INSTALLED)}
        outputs = compose({rel: before[paths[rel]] for rel in EDITED})
        return {paths[rel]: outputs[rel] for rel in outputs}

    return execute(
        root=root,
        contract_path=contract,
        receipt_path=image_receipt,
        transform=TRANSFORM,
        expected_section=section,
        builder=build,
        apply=apply,
        script_path=Path(__file__),
    )


def off_receipt() -> dict:
    return {
        "schema_version": 1,
        "transform": TRANSFORM,
        "state": "OFF_UNCHANGED",
        "mode": "off",
        "targets": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        reference = reference_module()
        mode = reference.resolve_mode(os.environ.get(reference.ENV))
        if mode == "off":
            # Do not resolve, read, hash, or touch the target tree in off mode.
            print_receipt(off_receipt())
            return 0
        reference.validate_composition(mode, os.environ)
        vllm = args.vllm_root.resolve(strict=True)
        root = vllm.parent
        if (root / "vllm").resolve(strict=True) != vllm:
            raise Refusal("--vllm-root must name the vllm package directory")
        section = expected_section()
        verify_sources(section)
        receipt = run_transform(
            vllm=vllm,
            contract=args.contract.resolve(strict=True),
            image_receipt=args.image_receipt.resolve(strict=True),
            section=section,
            apply=args.apply,
        )
        receipt["mode"] = mode
        print_receipt(receipt)
        return 0
    except (OSError, SyntaxError, ValueError, UnicodeError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
