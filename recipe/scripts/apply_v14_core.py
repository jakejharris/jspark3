#!/usr/bin/env python3
"""JSpark3 v1.4 core transform: pinned Mia/vLLM patchers over the v1.3 state.

Stage 6 of the base pipeline. Runs the sealed installers under
``overlays/v14/`` as ``python3 -S`` children against a staged copy of the
v1.3 target set (a mirrored ``vllm/`` tree, so the drafter patcher finds
``v1/worker/utils.py`` beside ``v1/core``), in Mia's Dockerfile order, and
commits the result through the same exact before/after hash transaction as
every other stage. The installers never touch the live tree: each one's
live-path fallbacks must all be redirected (``_installer_guard``) or the
stage refuses before it runs.

  1. patch_glm5_drafter_group.py   Mia@7cded8e   kv_cache_utils.py, worker/utils.py
  2. patch_scheduler_decode_floor.py  cadence-sched v6 (Mia v5 + #246 + cap)
  3. patch_mamba_align_chunking.py Mia@7cded8e   scheduler.py
  4. patch_kv_capacity_log.py      Mia@7cded8e   kv_cache_utils.py
  5. patch_indexer_warmup_range.py Mia #203      mla/indexer.py
  6. patch_kpool_seed_stride.py    vLLM #57477   ops/kpool_compress.py
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
from _contracts import V14_CORE
from _installer_guard import require_overrides

OVERLAYS = Path(__file__).resolve().parent.parent / "overlays" / "v14"

KV = "vllm/v1/core/kv_cache_utils.py"
WORKER_UTILS = "vllm/v1/worker/utils.py"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
INDEXER = "vllm/v1/attention/backends/mla/indexer.py"
KPOOL_COMPRESS = "vllm/models/glm5next/nvidia/ops/kpool_compress.py"
TARGETS = (KV, WORKER_UTILS, SCHEDULER, INDEXER, KPOOL_COMPRESS)

# (installer, extra argv, {env var: staged target}); order is Mia's.
STEPS = (
    ("patch_glm5_drafter_group.py", ("--kv-file", KV), {}),
    ("patch_scheduler_decode_floor.py", (), {"GLM53_SCHEDULER_PY": SCHEDULER}),
    ("patch_mamba_align_chunking.py", (), {"GLM53_SCHEDULER_PY": SCHEDULER}),
    ("patch_kv_capacity_log.py", (), {"GLM53_KV_CACHE_UTILS_PY": KV}),
    ("patch_indexer_warmup_range.py", (), {"GLM53_INDEXER_BACKEND_PY": INDEXER}),
    ("patch_kpool_seed_stride.py", (), {"GLM53_KPOOL_COMPRESS_PY": KPOOL_COMPRESS}),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> None:
    sources = V14_CORE["sources"]
    for name, _, _ in STEPS:
        path = OVERLAYS / name
        if path.is_symlink() or not path.is_file() or sha(path) != sources[name]:
            raise Refusal(f"pinned v1.4 installer drift: {name}")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    """Run every installer, in order, over staged copies of ``files``.

    ``files`` maps each TARGETS path to its before bytes; the result maps the
    same paths to the composed bytes. Shared with the offline contract
    generator (tools/v14/build_v14_contracts.py) so the sealed after hashes
    come from this exact code path.
    """
    if set(files) != set(TARGETS):
        raise Refusal("v1.4 core target-set drift")
    with tempfile.TemporaryDirectory(prefix="jspark3-v14-core-") as scratch:
        stage_root = Path(scratch)
        for rel, data in files.items():
            staged = stage_root / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
        run_installers(stage_root)
        return {rel: (stage_root / rel).read_bytes() for rel in TARGETS}


def run_installers(stage_root: Path) -> None:
    for name, argv, targets in STEPS:
        try:
            require_overrides(OVERLAYS / name, {*targets, *(a for a in argv if a.startswith("--"))})
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update({var: str(stage_root / rel) for var, rel in targets.items()})
        args = [str(stage_root / a) if a in TARGETS else a for a in argv]
        process = subprocess.run(
            [sys.executable, "-S", str(OVERLAYS / name), *args],
            env=env, cwd=stage_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        diagnostics.retain(process.stdout + process.stderr)
        if process.returncode:
            tail = (process.stderr or process.stdout).strip().splitlines()[-3:]
            raise Refusal(f"{name}: installer refused: {' | '.join(tail)}")


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
            paths = {rel: safe_target(root, rel) for rel in TARGETS}
            after = compose({rel: before[path] for rel, path in paths.items()})
            return {path: after[rel] for rel, path in paths.items()}

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_v14_core.py",
            expected_section=V14_CORE,
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
