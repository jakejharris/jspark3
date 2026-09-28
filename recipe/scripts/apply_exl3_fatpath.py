#!/usr/bin/env python3
"""JSpark3 v1.5 EXL3 fat-expert path.

Stage 8 of the base pipeline. Runs the sealed ``overlays/v15/
patch_exl3_fatpath.py`` against a staged copy of the served EXL3 MoE
(``exl3.py``, already at its stage 1 bytes) and installs the sealed
``jspark3_exl3_fatpath.py`` beside it, committed through the exact
before/after hash transaction. The mode (legacy, syncfree, fused) is chosen at
runtime by JSPARK3_EXL3_FATPATH; ``legacy`` runs the original loop unchanged.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from _atomic import Refusal, execute, print_receipt, safe_target
from _contracts import V15_FATPATH
from _installer_guard import require_overrides

OVERLAY = Path(__file__).resolve().parent.parent / "overlays" / "v15"
PATCHER = "patch_exl3_fatpath.py"
MODULE = "jspark3_exl3_fatpath.py"

EXL3 = "vllm/model_executor/layers/quantization/exl3.py"
INSTALLED = "vllm/model_executor/layers/quantization/jspark3_exl3_fatpath.py"
EDITED = (EXL3,)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> None:
    for name, digest in V15_FATPATH["sources"].items():
        path = OVERLAY / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise Refusal(f"pinned EXL3 fat-path source drift: {name}")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    """Run the sealed fat-path patcher over a staged copy of exl3.py.

    Returns the patched exl3.py plus the installed module (the sealed overlay
    bytes), keyed by relative path. Shared with the offline contract generator.
    """
    if set(files) != set(EDITED):
        raise Refusal("EXL3 fat-path target-set drift")
    with tempfile.TemporaryDirectory(prefix="jspark3-v15-fatpath-") as scratch:
        stage_root = Path(scratch)
        for rel, data in files.items():
            staged = stage_root / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GLM53_EXL3_PY": str(stage_root / EXL3),
        }
        try:
            require_overrides(OVERLAY / PATCHER, env)
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
        process = subprocess.run(
            [sys.executable, "-S", str(OVERLAY / PATCHER)], env=env, cwd=stage_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        diagnostics.retain(process.stdout + process.stderr)
        if process.returncode:
            tail = (process.stderr or process.stdout).strip().splitlines()[-3:]
            raise Refusal(f"{PATCHER}: installer refused: {' | '.join(tail)}")
        out = {rel: (stage_root / rel).read_bytes() for rel in EDITED}
        out[INSTALLED] = (OVERLAY / MODULE).read_bytes()
        return out


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
        verify_sources()

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            paths = {rel: safe_target(root, rel) for rel in (*EDITED, INSTALLED)}
            after = compose({rel: before[paths[rel]] for rel in EDITED})
            return {path: after[rel] for rel, path in paths.items()}

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_exl3_fatpath.py",
            expected_section=V15_FATPATH,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        print_receipt(receipt)
        return 0
    except (OSError, ValueError, UnicodeError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
