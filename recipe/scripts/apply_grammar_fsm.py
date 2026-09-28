#!/usr/bin/env python3
"""Apply the sealed XGrammar reasoning/termination backport after the base pipeline."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import importlib.util
from pathlib import Path

from _atomic import Refusal, execute, print_receipt, safe_target, sha_file
from _contracts import V16_GRAMMAR_FSM

OVERLAY = Path(__file__).resolve().parent.parent / "overlays/v16/grammar_fsm"
PATCHER = "patch_xgrammar_termination.py"
BACKEND = "vllm/v1/structured_output/backend_xgrammar.py"
MANAGER = "vllm/v1/structured_output/__init__.py"
EDITED = (BACKEND, MANAGER)
TRANSFORM = "apply_grammar_fsm.py"


def verify_sources(section=V16_GRAMMAR_FSM):
    if section["sources"] != {PATCHER: sha_file(OVERLAY / PATCHER)}:
        raise Refusal("grammar FSM patcher source drift")


def compose(files: dict[str, bytes]) -> dict[str, bytes]:
    if set(files) != set(EDITED):
        raise Refusal("grammar FSM target inventory drift")
    verify_sources()
    spec = importlib.util.spec_from_file_location("grammar_fsm_patch", OVERLAY / PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = {}
    for name, prepare in ((BACKEND, module.prepare_backend), (MANAGER, module.prepare_manager)):
        text, _ = prepare(files[name].decode())
        compile(text, name, "exec")
        outputs[name] = text.encode()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        root = args.vllm_root.resolve(strict=True).parent
        if (root / "vllm").resolve(strict=True) != args.vllm_root.resolve(strict=True):
            raise Refusal("--vllm-root must name the vllm package")
        verify_sources()
        paths = {name: safe_target(root, name) for name in EDITED}

        def build(before):
            outputs = compose({name: before[path] for name, path in paths.items()})
            return {paths[name]: data for name, data in outputs.items()}

        print_receipt(execute(
            root=root, contract_path=args.contract, receipt_path=args.image_receipt,
            transform=TRANSFORM, expected_section=V16_GRAMMAR_FSM,
            builder=build, apply=args.apply, script_path=Path(__file__),
        ))
        return 0
    except (OSError, ValueError, SyntaxError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
