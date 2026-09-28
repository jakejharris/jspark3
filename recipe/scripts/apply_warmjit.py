#!/usr/bin/env python3
"""Install the sealed S9.11 worker-startup warmup after the base pipeline."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
from pathlib import Path

from _atomic import Refusal, execute, print_receipt, safe_target, sha_file
from _contracts import V16_WARMJIT

TRANSFORM = "apply_warmjit.py"
OVERLAY = Path(__file__).resolve().parent.parent / "overlays/v16/warmjit"
MODULE = "warmup_jit.py"
WORKER = "vllm/v1/worker/gpu_worker.py"
INSTALLED = "vllm/v1/worker/jspark3_warmup_jit.py"
DEPENDENCIES = (
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
    "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
    "vllm/v1/worker/gpu/sample/gumbel.py",
    "vllm/v1/worker/gpu/sample/logprob.py",
    "vllm/v1/sample/ops/topk_topp_triton.py",
    "vllm/third_party/flash_linear_attention/ops/l2norm.py",
    "vllm/models/glm5next/nvidia/ops/kpool_compress.py",
    "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "vllm/model_executor/kernels/mhc/tilelang_kernels.py",
    "vllm/model_executor/kernels/mhc/tilelang.py",
    "vllm/models/glm5next/nvidia/kda.py",
    "vllm/v1/attention/backends/gdn_attn.py",
)
EDITED = (WORKER, *DEPENDENCIES)
ANCHOR = "        # Reset the seed to ensure that the random state is not affected by\n"
HOOK = ("        # [jspark3-s911-warmjit] Boot-only; before seed reset and JIT monitor.\n"
        "        from vllm.v1.worker.jspark3_warmup_jit import warmup as warmup_traffic_jit\n"
        "        warmup_traffic_jit(self)\n\n")


def verify_sources():
    if V16_WARMJIT["sources"] != {MODULE: sha_file(OVERLAY / MODULE)}:
        raise Refusal("S9.11 warmup source drift")


def compose(files):
    if set(files) != set(EDITED):
        raise Refusal("S9.11 target inventory drift")
    verify_sources()
    text = files[WORKER].decode()
    if text.count(ANCHOR) != 1 or "jspark3-s911-warmjit" in text:
        raise Refusal("S9.11 worker warmup anchor drift")
    text = text.replace(ANCHOR, HOOK + ANCHOR)
    compile(text, WORKER, "exec")
    return dict(files, **{WORKER: text.encode(), INSTALLED: (OVERLAY / MODULE).read_bytes()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        root = args.vllm_root.resolve(strict=True).parent
        if root / "vllm" != args.vllm_root.resolve(strict=True):
            raise Refusal("--vllm-root must name the vllm package")
        verify_sources()
        paths = {name: safe_target(root, name) for name in (*EDITED, INSTALLED)}

        def build(before):
            outputs = compose({name: before[paths[name]] for name in EDITED})
            return {paths[name]: data for name, data in outputs.items()}

        print_receipt(execute(
            root=root, contract_path=args.contract, receipt_path=args.image_receipt,
            transform=TRANSFORM, expected_section=V16_WARMJIT, builder=build,
            apply=args.apply, script_path=Path(__file__)))
        return 0
    except (OSError, ValueError, Refusal) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
