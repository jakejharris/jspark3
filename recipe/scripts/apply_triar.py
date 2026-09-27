#!/usr/bin/env python3
"""Optional boot stage after v1.6. Gate 0 leaves every served byte untouched."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from _atomic import Refusal, execute, print_receipt, read_image_receipt, safe_target

OVERLAY = Path(__file__).resolve().parents[1] / "overlays/v17/triar"
TRANSFORM = "apply_triar.py"
PREFIX = "vllm/distributed/device_communicators/"
CUDA = PREFIX + "cuda_communicator.py"
PYNCCL = PREFIX + "pynccl.py"
MODULES = ("jspark3_triar_arithmetic.py", "jspark3_triar_kernel.py", "jspark3_triar_runtime.py", "jspark3_triar_dual.py")
CONTRACT_SHA256 = "703820079200f547c6fa117560d73e0e5986d401f5a7a880dec7e091f24bec86"
FOOTER = b'''
# [jspark3-triar] Installed only when JSPARK3_TRIAR=1 at boot.
from vllm.distributed.device_communicators.jspark3_triar_runtime import install as _triar_install
_triar_install(CudaCommunicator)
del _triar_install
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section():
    raw = (OVERLAY / "INSTALL_CONTRACT.json").read_bytes()
    if sha(raw) != CONTRACT_SHA256:
        raise Refusal("TRIAR install contract drift")
    value = json.loads(raw)["transforms"][TRANSFORM]
    for name, digest in value["sources"].items():
        path = OVERLAY / name
        if path.is_symlink() or sha(path.read_bytes()) != digest:
            raise Refusal(f"TRIAR source drift: {name}")
    return value


def compose(before):
    return {CUDA: before[CUDA] + FOOTER, PYNCCL: before[PYNCCL],
            **{PREFIX + name: (OVERLAY / name).read_bytes() for name in MODULES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=OVERLAY / "INSTALL_CONTRACT.json")
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--state", choices=("0", "1"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        gate = args.state if args.state is not None else os.environ.get("JSPARK3_TRIAR", "0")
        if gate not in ("0", "1"):
            raise Refusal("JSPARK3_TRIAR must be 0 or 1")
        expected = section()
        image = read_image_receipt(args.image_receipt)
        if os.environ.get("NODE_RANK", str(image["rank"])) != str(image["rank"]):
            raise Refusal("TRIAR image receipt rank mismatch")
        vllm = args.vllm_root.resolve(strict=True)
        if vllm.name != "vllm":
            raise Refusal("--vllm-root must name the vllm package directory")
        root = vllm.parent
        paths = {r["path"]: safe_target(root, r["path"]) for r in expected["targets"]}
        if gate == "0":
            for record in expected["targets"]:
                path = paths[record["path"]]
                observed = sha(path.read_bytes()) if path.exists() else "ABSENT"
                if observed != record["before_sha256"]:
                    raise Refusal("TRIAR off requires exact original files and no installed modules")
            print(json.dumps({"transform": TRANSFORM, "state": "DISABLED_EXACT_V16"}))
            return 0

        def build(before):
            result = compose({rel: before[paths[rel]] for rel in (CUDA, PYNCCL)})
            return {paths[rel]: data for rel, data in result.items()}

        receipt = execute(root=root, contract_path=args.contract,
                          receipt_path=args.image_receipt, transform=TRANSFORM,
                          expected_section=expected, builder=build, apply=args.apply,
                          script_path=Path(__file__))
        print_receipt(receipt)
        return 0
    except (OSError, ValueError, KeyError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
