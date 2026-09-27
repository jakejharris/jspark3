#!/usr/bin/env python3
"""JSpark3 v1.4 display-reserve KV wiring (MiaAI-Lab #234, forked).

Stage 7 of the base pipeline. Runs the sealed ``overlays/v14/display_kv/
patch_display_kv.py`` against a staged copy of the four worker allocation
targets and installs ``glm53_display_kv.py`` beside them, committed through
the exact before/after hash transaction. The shared library is not installed:
workers load the sealed build from the read-only recipe mount
(GLM53_DISPLAY_KV_LIB), and ``GLM53_DISPLAY_KV=0`` never opens it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from _atomic import Refusal, execute, print_receipt, safe_target
from _contracts import V14_DISPLAY
from _installer_guard import require_overrides

OVERLAY = Path(__file__).resolve().parent.parent / "overlays" / "v14" / "display_kv"
PATCHER = "patch_display_kv.py"
MODULE = "glm53_display_kv.py"

WORKER = "vllm/v1/worker/gpu_worker.py"
RUNNER = "vllm/v1/worker/gpu_model_runner.py"
ATTN = "vllm/v1/worker/gpu/attn_utils.py"
MIXIN = "vllm/v1/worker/kv_connector_model_runner_mixin.py"
INSTALLED = "vllm/v1/worker/glm53_display_kv.py"
EDITED = (WORKER, RUNNER, ATTN, MIXIN)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> None:
    for name, digest in V14_DISPLAY["sources"].items():
        path = OVERLAY / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise Refusal(f"pinned display-KV source drift: {name}")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    """Run the sealed #234 patcher over staged copies of the EDITED files.

    Returns the edited files plus the installed module, keyed by relative
    path. Shared with the offline contract generator.
    """
    if set(files) != set(EDITED):
        raise Refusal("display-KV target-set drift")
    with tempfile.TemporaryDirectory(prefix="jspark3-v14-display-") as scratch:
        stage_root = Path(scratch)
        for rel, data in files.items():
            staged = stage_root / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GLM53_VLLM_ROOT": str(stage_root / "vllm"),
            "GLM53_DISPLAY_KV_STAGE": str(OVERLAY),
        }
        try:
            require_overrides(OVERLAY / PATCHER, env)
        except ValueError as exc:
            raise Refusal(str(exc)) from None
        process = subprocess.run(
            [sys.executable, "-S", str(OVERLAY / PATCHER)], env=env, cwd=stage_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode:
            tail = (process.stderr or process.stdout).strip().splitlines()[-3:]
            raise Refusal(f"{PATCHER}: installer refused: {' | '.join(tail)}")
        return {rel: (stage_root / rel).read_bytes() for rel in (*EDITED, INSTALLED)}


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
            transform="apply_display_kv.py",
            expected_section=V14_DISPLAY,
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
