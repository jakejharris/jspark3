#!/usr/bin/env python3
"""Apply the JSpark3 v1.6 adaptive verification-width overlay exactly once."""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path

from _atomic import Refusal, canonical, execute, read_image_receipt, safe_target


OVERLAY = Path(__file__).resolve().parent.parent / "overlays" / "v16" / "adaptive_k"
PATCHER = "patch_adaptive_k.py"
POLICY = "jspark3_adaptive_k.py"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
CUDAGRAPH = "vllm/v1/worker/gpu/cudagraph_utils.py"
INSTALLED_POLICY = "vllm/v1/core/sched/jspark3_adaptive_k.py"
EDITED = (SCHEDULER, CUDAGRAPH)
INSTALLED = (INSTALLED_POLICY,)
TRANSFORM = "apply_adaptive_k.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patcher_module():
    path = OVERLAY / PATCHER
    spec = importlib.util.spec_from_file_location("jspark3_adaptive_k_patcher", path)
    if spec is None or spec.loader is None:
        raise Refusal("cannot load sealed adaptive-k patcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_section() -> dict:
    try:
        import _contracts

        section = _contracts.V16_ADAPTIVE_K
    except (ImportError, AttributeError) as exc:
        raise Refusal(
            "v1.6 adaptive-k contract is not wired into recipe/scripts/_contracts.py"
        ) from exc
    if not isinstance(section, dict):
        raise Refusal("v1.6 adaptive-k embedded contract is not an object")
    return section


def verify_sources(section: dict) -> None:
    expected = section.get("sources", {})
    if set(expected) != {PATCHER, POLICY}:
        raise Refusal("adaptive-k source inventory drift")
    for name in (PATCHER, POLICY):
        path = OVERLAY / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected[name]:
            raise Refusal(f"pinned adaptive-k source drift: {name}")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    if set(files) != set(EDITED):
        raise Refusal("adaptive-k target-set drift")
    try:
        scheduler, cudagraph = patcher_module().patch_bytes(
            files[SCHEDULER], files[CUDAGRAPH]
        )
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise Refusal(f"{PATCHER}: installer refused: {exc}") from None
    return {
        SCHEDULER: scheduler,
        CUDAGRAPH: cudagraph,
        INSTALLED_POLICY: (OVERLAY / POLICY).read_bytes(),
    }


def identity(rank: int, state: str) -> None:
    print(f"[jspark3-v16:adaptive-k] rank={rank} state={state}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--state", choices=("off", "ema"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        mode = args.state if args.state is not None else os.environ.get("GLM53_ADAPTIVE_K")
        if mode not in ("off", "ema"):
            raise Refusal("GLM53_ADAPTIVE_K must be explicitly off or ema")
        expected = os.environ.get("JSPARK3_V16_ADAPTIVE_K_EXPECT")
        if expected != mode:
            raise Refusal(
                "JSPARK3_V16_ADAPTIVE_K_EXPECT must exactly match GLM53_ADAPTIVE_K"
            )
        image = read_image_receipt(args.image_receipt.resolve(strict=True))
        rank = image["rank"]
        if os.environ.get("NODE_RANK", str(rank)) != str(rank):
            raise Refusal("NODE_RANK disagrees with the image receipt")
        vllm = args.vllm_root.resolve(strict=True)
        root = vllm.parent
        if (root / "vllm").resolve(strict=True) != vllm:
            raise Refusal("--vllm-root must name the vllm package directory")
        paths = {rel: safe_target(root, rel) for rel in (*EDITED, *INSTALLED)}

        if mode == "off":
            patcher = patcher_module()
            observed = {rel: sha(paths[rel]) for rel in EDITED}
            required = {
                SCHEDULER: patcher.SCHEDULER_V15_SHA256,
                CUDAGRAPH: patcher.CUDAGRAPH_V15_SHA256,
            }
            if observed != required or paths[INSTALLED_POLICY].exists():
                raise Refusal("adaptive-k off requires exact post-v1.5 target state")
            identity(rank, "off")
            sys.stdout.buffer.write(
                canonical(
                    {
                        "schema_version": 1,
                        "transform": TRANSFORM,
                        "state": "DISABLED_EXACT_V15",
                        "targets": observed,
                    }
                )
            )
            return 0

        section = expected_section()
        verify_sources(section)

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            outputs = compose({rel: before[paths[rel]] for rel in EDITED})
            return {paths[rel]: data for rel, data in outputs.items()}

        receipt = execute(
            root=root,
            contract_path=args.contract.resolve(strict=True),
            receipt_path=args.image_receipt,
            transform=TRANSFORM,
            expected_section=section,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        identity(rank, "ema")
        sys.stdout.buffer.write(canonical(receipt))
        return 0
    except (OSError, SyntaxError, ValueError, UnicodeError, KeyError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
