#!/usr/bin/env python3
"""Route the existing post-load trunk finalizer through dense FP8 when enabled."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# [jspark3-v16-dense-fp8]"
OLD = """def finalize_with_ablit(model):
    from . import trunk_w8a16 as trunk
    mode = os.environ.get('ABLIT')
"""
NEW = """def finalize_with_ablit(model):
    dense_fp8_mode = os.environ.get('JSPARK3_V16_DENSE_FP8', 'off').strip().lower()
    if dense_fp8_mode == 'off':
        from . import trunk_w8a16 as trunk
    elif dense_fp8_mode in ('trunk', 'negative-coarse'):
        from . import jspark3_dense_fp8 as trunk  # [jspark3-v16-dense-fp8]
    else:
        raise RuntimeError(
            "JSPARK3_V16_DENSE_FP8 must be off, trunk or negative-coarse; "
            f"got {dense_fp8_mode!r}"
        )
    mode = os.environ.get('ABLIT')
"""


def patch_text(text: str) -> str:
    if text.count(NEW) == 1 and text.count(OLD) == 0:
        compile(text, "ablit_transplant.py", "exec")
        return text
    if text.count(OLD) != 1 or text.count(NEW) != 0 or MARKER in text:
        raise ValueError("dense-FP8 ablit finalizer anchor drifted or is partial")
    result = text.replace(OLD, NEW, 1)
    if result.count(MARKER) != 1:
        raise ValueError("dense-FP8 ablit finalizer marker mismatch")
    compile(result, "ablit_transplant.py", "exec")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    text = args.target.read_text(encoding="utf-8")
    patched = patch_text(text)
    if patched == text:
        print(f"{args.target.name}: {MARKER} already present - verified")
        return 0
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched {args.target.name} ({MARKER})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
