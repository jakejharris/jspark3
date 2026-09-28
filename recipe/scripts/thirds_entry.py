#!/usr/bin/env python3
"""Candidate container entrypoint; use `python3 -S`, before site hooks.

Off delegates directly to the sealed entry script. On runs those exact stages
and adds one thirds stage immediately before the final vllm exec. No copy of
the served script or any sealed runtime source is edited on disk.
"""
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import hashlib
import os
from pathlib import Path

# Reviewed base entry including the cooperative maintenance-override refusals.
# TRIAR and thirds compose their stages only over these exact source bytes.
ENTRY_SHA256 = {'35e2256f6be0032c5b4fdcdc1ec5cccb7b56251bdfd5fa59b3029fcbeff3a374'}
ANCHOR = 'exec vllm serve "$JSPARK_TARGET_RUNTIME" "$@"\n'
STAGE = '''python3 -S "$recipe/scripts/apply_cyclic_thirds.py" --vllm-root "$vllm" \\
  --state 1 --image-receipt "$receipt" --apply
'''


def compose(entry):
    if hashlib.sha256(entry).hexdigest() not in ENTRY_SHA256:
        raise RuntimeError("sealed v1.6 entrypoint hash drift")
    text = entry.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError("entrypoint seam drift")
    return text.replace(ANCHOR, STAGE + ANCHOR)


def main():
    entry = Path(__file__).resolve().with_name("container_entry.sh")
    state = os.environ.get("GLM53_CYCLIC_THIRDS", "0")
    if state == "0":
        os.execv("/bin/bash", ["bash", str(entry), *sys.argv[1:]])
    if state != "1":
        raise SystemExit("REFUSE: GLM53_CYCLIC_THIRDS must be 0 or 1")
    if os.environ.get("JSPARK3_V16_PROFILE") != "qa":
        raise SystemExit("REFUSE: unqualified cyclic thirds is a QA candidate only")
    script = compose(entry.read_bytes())
    os.execv("/bin/bash", ["bash", "-c", script, "thirds-entry", *sys.argv[1:]])


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    main()
