#!/usr/bin/env python3
"""Hash- and seam-bound JSpark3 v1 hook after vLLM weight post-processing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py"
)
IMPORT_ANCHOR = "from abc import ABC, abstractmethod\n\n"
IMPORT_LINE = "import os\n\n"
CALL_ANCHOR = "            process_weights_after_loading(model, model_config, target_device)\n"
CALL_REPLACEMENT = '''            process_weights_after_loading(model, model_config, target_device)

            if (
                os.environ.get("JSPARK3_TRUNK_W8A16") == "1"
                and type(model).__name__ == "Glm5NextForConditionalGeneration"
            ):
                from vllm.model_executor.layers.quantization.trunk_w8a16 import (
                    finalize_trunk_w8a16,
                )

                language_model = model.language_model
                if type(language_model).__name__ != "Glm5NextForCausalLM":
                    raise RuntimeError(
                        "JSPARK3_REFUSE: conditional-generation language model type drift "
                        f"observed={type(language_model).__name__}"
                    )
                finalize_trunk_w8a16(language_model)
'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--expected-before-sha256", required=True)
    parser.add_argument("--expected-after-sha256", default="")
    args = parser.parse_args()
    source = args.target.read_bytes()
    before = sha256_bytes(source)
    text = source.decode("utf-8")
    already = text.count(IMPORT_LINE) == 1 and text.count(CALL_REPLACEMENT) == 1
    if already:
        compile(text, str(args.target), "exec")
        if args.expected_after_sha256 and before != args.expected_after_sha256:
            raise SystemExit(
                "REFUSE: already-applied base-loader SHA drift "
                f"expected={args.expected_after_sha256} observed={before}"
            )
        print(
            f"JSPARK3_BASE_LOADER_PATCH_ALREADY_APPLIED before_sha256={before} "
            f"after_sha256={before}",
            flush=True,
        )
        return 0
    if before != args.expected_before_sha256:
        raise SystemExit(
            "REFUSE: pristine base-loader SHA drift "
            f"expected={args.expected_before_sha256} observed={before}"
        )
    if text.count(IMPORT_LINE) or text.count(CALL_REPLACEMENT):
        raise SystemExit("REFUSE: partial JSpark3 base-loader hook detected")
    if text.count(IMPORT_ANCHOR) != 1 or text.count(CALL_ANCHOR) != 1:
        raise SystemExit("REFUSE: base-loader seam count drift")
    text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    text = text.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
    if text.count(IMPORT_LINE) != 1 or text.count(CALL_REPLACEMENT) != 1:
        raise SystemExit("REFUSE: base-loader result marker mismatch")
    compile(text, str(args.target), "exec")
    patched = text.encode("utf-8")
    after = sha256_bytes(patched)
    if args.expected_after_sha256 and after != args.expected_after_sha256:
        raise SystemExit(
            "REFUSE: patched base-loader SHA drift "
            f"expected={args.expected_after_sha256} observed={after}"
        )
    args.target.write_bytes(patched)
    print(
        f"JSPARK3_BASE_LOADER_PATCH_PASS before_sha256={before} after_sha256={after}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
