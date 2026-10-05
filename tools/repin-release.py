#!/usr/bin/env python3
"""Repin a draft candidate from a source-matching wheel, then regenerate SHA256SUMS."""
import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--engine-commit', required=True, help='committed source containing the revised fix')
    args = parser.parse_args()
    gate = runpy.run_path(str(HERE / 'scripts/check-release.py'))
    meta = json.loads((HERE / 'manifests/release.json').read_text())
    if meta['status'] != 'draft':
        parser.error('only an unpublished draft may be repinned')
    pins = gate['settings'](HERE / 'pins.env')
    if args.wheel.name != pins['ENGINE_WHEEL']:
        parser.error('wheel filename does not match ENGINE_WHEEL')
    commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--verify',
                                      args.engine_commit + '^{commit}'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(HERE), 'diff', commit, '--', 'engine'], text=True)
    if changed:
        parser.error('engine/ must match the specified committed source')
    payload = runpy.run_path(str(HERE / 'tools/payload.py'))
    untracked = sorted(set(payload['present']()) - set(payload['tracked']()) - {'SHA256SUMS'})
    if untracked:
        parser.error('stage or move untracked payload files before repinning: ' + ', '.join(untracked))
    digest = gate['check_wheel'](args.wheel)
    namespace = f"{pins['RELEASE']}-{digest[:12]}"
    updates = {'pins.env': {'ENGINE_COMMIT': commit, 'WHEEL_CONTENT_SHA256': digest},
               'config/serve.conf': {'SERVE_SESSION_NAMESPACE': namespace},
               'config/serve.env': {'TF_GLM_DISK_MATH_VERSION': f'E:{namespace}'}}
    for name, values in updates.items():
        path = HERE / name
        text = path.read_text()
        for key, value in values.items():
            text, count = re.subn(rf'^{key}=.*$', f'{key}={value}', text, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f'{name} must contain exactly one {key}')
        path.write_text(text)
    meta['engine_commit'] = commit
    meta['engine_package_sha256'] = gate['WHEEL']['digest'](gate['package_files']())
    meta['wheel']['content_sha256'] = digest
    meta['session_namespace'] = namespace
    (HERE / 'manifests/release.json').write_text(json.dumps(meta, indent=2) + '\n')
    return payload['manifest']()


if __name__ == '__main__':
    sys.exit(main())
