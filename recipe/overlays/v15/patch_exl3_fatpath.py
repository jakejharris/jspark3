#!/usr/bin/env python3
"""JSpark3 v1.5: route the EXL3 fat-expert fallback through jspark3_exl3_fatpath.

Target: GLM53_EXL3_PY (default: the served FlyCockpit
vllm/model_executor/layers/quantization/exl3.py, sha256 9e823926...). Three
exact hunks:

  1. import the sealed ``jspark3_exl3_fatpath`` module (installed beside
     exl3.py by the same stage);
  2. in ``apply_exl3_fused_moe``, start the single asynchronous host copy of the
     per-expert row counts BEFORE the fused ``exl3_moe`` launch;
  3. hand the fat experts to ``_jspark3_fatpath.dispatch``; the original
     ``nonzero()/tolist()`` + ``apply_exl3_python_loop`` block stays in place,
     byte for byte, and runs only when dispatch declines (mode ``legacy``).

The thin fused path, the decode graphs and every other function are untouched.
Idempotent; a partial or drifted file fails before writing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_EXL3_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py",
    )
)

HUNKS = (
    (
        "from vllm.model_executor.utils import set_weight_attrs\n",
        "from vllm.model_executor.utils import set_weight_attrs\n"
        "from vllm.model_executor.layers.quantization import jspark3_exl3_fatpath as _jspark3_fatpath  # [jspark3-fatpath]\n",
    ),
    (
        "    counts = expert_count[:n_exp]\n"
        "    fn = exllamav3_ext.exl3_moe\n",
        "    counts = expert_count[:n_exp]\n"
        "    # [jspark3-fatpath] one async host copy of the row counts, issued before the fused launch\n"
        "    fat_handle = _jspark3_fatpath.probe(expert_count, n_exp, tokens, TEMP_ROWS_FUSED)\n"
        "    fn = exllamav3_ext.exl3_moe\n",
    ),
    (
        "    if tokens > TEMP_ROWS_FUSED:\n"
        "        fat = (counts > TEMP_ROWS_FUSED).nonzero(as_tuple=False).view(-1)\n"
        "        if fat.numel():\n",
        "    if tokens > TEMP_ROWS_FUSED and not _jspark3_fatpath.dispatch(\n"
        "        fat_handle, x2d, ids, weights, layer, inners, expert_map, limit, local, out,\n"
        "        TEMP_ROWS_FUSED, apply_exl3_python_loop,\n"
        "    ):  # [jspark3-fatpath] the legacy loop below runs only in mode legacy\n"
        "        fat = (counts > TEMP_ROWS_FUSED).nonzero(as_tuple=False).view(-1)\n"
        "        if fat.numel():\n",
    ),
)


def patch_text(text: str) -> str:
    new_count = [text.count(new) for _, new in HUNKS]
    if all(n == 1 for n in new_count):
        return text
    for old, new in HUNKS:
        if text.count(old) != 1 or text.count(new) != 0:
            raise SystemExit(f"{P}: [jspark3-fatpath] anchor drifted (partial or unknown source)")
        text = text.replace(old, new, 1)
    compile(text, str(P), "exec")
    return text


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    patched = patch_text(text)
    if patched == text:
        print(f"{P.name}: [jspark3-fatpath] already present - verified")
        return 0
    P.write_text(patched)
    print(f"patched {P.name} ([jspark3-fatpath] dispatch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
