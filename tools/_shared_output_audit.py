"""Require explicit review when executable output paths or their inputs change.

This is an audit coverage guard, not a taint analyser. Whole-file hashes also
cover changes upstream of a writer. Runtime-only channels are included so
moving one into shared evidence cannot silently escape the review inventory.
"""
import hashlib
import json
from pathlib import Path

SUFFIXES = {'.py', '.sh', '.c', '.cu', '.cuh', '.h', '.patch', '.pth'}


def candidates(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts)
            and (p.suffix in SUFFIXES or p.name == 'Dockerfile')}


def verify(root):
    root = Path(root)
    audit = json.loads((root / 'manifests/shared-output-audit.json').read_text())
    actual = candidates(root)
    recorded = audit['files']
    issues = []
    if set(actual) != set(recorded):
        issues.append('executable inventory changed; shared-output review required')
    for name in actual.keys() & recorded.keys():
        row = recorded[name]
        if (hashlib.sha256(actual[name].read_bytes()).hexdigest() != row['sha256']
                or not row.get('disposition') or not isinstance(row.get('sites'), list)):
            issues.append('shared-output audit stale: ' + name)
    return issues
