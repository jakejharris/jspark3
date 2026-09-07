#!/usr/bin/env python3
"""Exercise the documented CLI without bytecode environment workarounds."""
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ)
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    env.pop('PYTHONPYCACHEPREFIX', None)
    command = ['python3', 'tools/validate_release.py', '.', '--report', 'validation.json']
    with tempfile.TemporaryDirectory() as raw:
        for supplied_cache in (False, True):
            root = Path(raw) / ('stale' if supplied_cache else 'clean')
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(
                '.git', '__pycache__', '.pytest_cache', 'dist', 'validation.json'))
            cache = None
            if supplied_cache:
                cache = Path(py_compile.compile(
                    str(root / 'tools/validate_live_evidence.py'),
                    cfile=str(root / 'tools/__pycache__' /
                              f'validate_live_evidence.{sys.implementation.cache_tag}.pyc'),
                    doraise=True))
                original = cache.read_bytes()
            result = subprocess.run(command, cwd=root, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report = json.loads((root / 'validation.json').read_text())
            inventory = next(check for check in report['checks'] if check['check'] == 'inventory')
            if supplied_cache:
                assert result.returncode != 0, result.stdout
                assert inventory['status'] == 'FAIL', result.stdout
                assert 'compiled cache' in inventory['detail'], inventory
                assert cache.read_bytes() == original, 'validator removed or rewrote supplied cache'
                assert sum(check['status'] == 'FAIL' for check in report['checks']) == 1, result.stdout
            else:
                assert result.returncode == 0, result.stdout
                assert inventory['status'] == 'PASS', result.stdout
                assert not list(root.rglob('__pycache__')), 'validator created a cache directory'
                assert not list(root.rglob('*.pyc')), 'validator wrote bytecode'
    print('PASS documented CLI: clean tree passes all gates without bytecode writes; '
          'supplied helper cache fails inventory and remains unchanged')


if __name__ == '__main__':
    main()
