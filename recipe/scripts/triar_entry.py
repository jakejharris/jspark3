#!/usr/bin/env python3
"""Compose TRIAR with either thirds setting, preserving the sealed base entry."""
import os
from pathlib import Path
import sys
import thirds_entry

ANCHOR = 'overlay_source=$recipe/overlays/trunk_w8a16.py\n'
STAGE = '''python3 -S "$recipe/scripts/apply_triar.py" --vllm-root "$vllm" \\
  --state 1 --image-receipt "$receipt" --apply
'''


def compose(entry, thirds):
    # Also validates the exact base entry and its final-exec seam.
    script = thirds_entry.compose(entry)
    if not thirds:
        script = script.replace(thirds_entry.STAGE, '')
    if script.count(ANCHOR) != 1:
        raise RuntimeError('TRIAR entry seam drift')
    script=script.replace(ANCHOR, STAGE + ANCHOR)
    return script.replace(thirds_entry.ANCHOR, 'timeout --signal=TERM --kill-after=15s 210s python3 -S /recipe/scripts/final_container_preflight.py\n' + thirds_entry.ANCHOR)


def main():
    if os.environ.get('JSPARK3_TRIAR') != '1':
        raise SystemExit('REFUSE: TRIAR entry requires JSPARK3_TRIAR=1')
    state = os.environ.get('GLM53_CYCLIC_THIRDS', '0')
    if state not in ('0', '1'):
        raise SystemExit('REFUSE: invalid thirds state')
    if os.environ.get('JSPARK3_V16_PROFILE') not in ('qa','production','production-stock'):
        raise SystemExit('REFUSE: TRIAR requires QA, production or production-stock profile')
    script = compose(Path(__file__).with_name('container_entry.sh').read_bytes(), state == '1')
    os.execv('/bin/bash', ['bash', '-c', script, 'triar-entry', *sys.argv[1:]])


if __name__ == '__main__':
    main()
