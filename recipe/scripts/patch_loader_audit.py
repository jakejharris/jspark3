#!/usr/bin/env python3
"""Install a byte-bound audit wrapper without changing the loader algorithm."""
import hashlib
import json
from pathlib import Path
import sys

recipe, target = map(Path, sys.argv[1:])
contract = json.loads((recipe / 'config/loader-audit.json').read_text())
raw = target.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
if digest == contract['after']:
    raise SystemExit(0)
if digest != contract['before']:
    raise SystemExit('REFUSE: loader audit source drift')
anchor = '\nlogger = init_logger(__name__)\n'
text = raw.decode()
if text.count(anchor) != 1:
    raise SystemExit('REFUSE: loader audit seam drift')
text = text.replace(anchor, '\nfrom vllm.model_executor.layers.quantization.instanttensor_audit import audited_iterator as instanttensor_weights_iterator\n' + anchor)
compile(text, str(target), 'exec')
if hashlib.sha256(text.encode()).hexdigest() != contract['after']:
    raise SystemExit('REFUSE: loader audit result drift')
target.write_text(text)
