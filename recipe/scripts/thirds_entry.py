#!/usr/bin/env python3
"""Candidate container entrypoint; use `python3 -S`, before site hooks.

Off delegates directly to the sealed entry script. On runs those exact stages
and adds one thirds stage immediately before the final vllm exec. No copy of
the served script or any sealed runtime source is edited on disk.
"""

import hashlib
import os
from pathlib import Path
import sys

# Sealed v1.6 entry, and the same entry with the Lab D TRIAR stage appended
# after adaptive-k (JSPARK3_TRIAR-gated; off leaves the served tree unpatched).
ENTRY_SHA256 = {'b830b123102777aea3080bfda50acd8b17417f8b8e370a77daf328c4a3b75aec'}
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
    main()
