#!/usr/bin/env python3
"""Check the draft release identity and, when supplied, its built engine wheel."""
import argparse
import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
WHEEL = runpy.run_path(str(HERE / 'scripts/wheel-content.py'))


def settings(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines()
                if line and not line.startswith('#'))


def package_files():
    source = HERE / 'engine/src'
    return {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / 'tensorfold').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def check_wheel(path):
    files = WHEEL['contents'](path)
    package = {name: digest for name, digest in files.items() if name.startswith('tensorfold/')}
    # These developer guides are not package data in engine/pyproject.toml.
    guides = {'tensorfold/families/README.md', 'tensorfold/families/qwen3_5/cuda/README.md',
              'tensorfold/kernels/README.md'}
    expected = {name: digest for name, digest in package_files().items() if name not in guides}
    if package != expected:
        raise ValueError('wheel package files differ from engine/src/tensorfold')
    return WHEEL['digest'](files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, help='also verify a built wheel and the payload manifest')
    args = parser.parse_args()
    pins = settings(HERE / 'pins.env')
    meta = json.loads((HERE / 'manifests/release.json').read_text())
    version = pins['RELEASE']
    failures = []

    def check(ok, name):
        print(f'{"PASS" if ok else "FAIL"}  {name}')
        if not ok:
            failures.append(name)

    check(version == 'v2.0.2' and meta['version'] == version, 'v2.0.2 release identity')
    check(meta['status'] == 'draft' and meta['source_ref'] == f'release/{version}', 'draft source reference')
    check(meta['base_version'] == 'v2.0.1' and meta['base_commit'] ==
          '0be336670bb1ce8827ff7dbf1f33d1d61ea33cd7', 'v2.0.1 base')
    check(re.fullmatch(r'[0-9a-f]{40}', pins['ENGINE_COMMIT']) is not None and
          meta['engine_commit'] == pins['ENGINE_COMMIT'], 'engine source commit')
    check(meta['engine_package_sha256'] == WHEEL['digest'](package_files()), 'engine package source digest')
    check(re.fullmatch(r'[0-9a-f]{64}', pins['WHEEL_CONTENT_SHA256']) is not None and
          meta['wheel'] == {'filename': pins['ENGINE_WHEEL'], 'content_sha256': pins['WHEEL_CONTENT_SHA256']},
          'wheel filename and content pin')
    namespace = f"{version}-{pins['WHEEL_CONTENT_SHA256'][:12]}"
    check(settings(HERE / 'config/serve.conf')['SERVE_SESSION_NAMESPACE'] == namespace and
          settings(HERE / 'config/serve.env')['TF_GLM_DISK_MATH_VERSION'] == f'E:{namespace}' and
          meta['session_namespace'] == namespace, 'wheel-specific session namespace and arithmetic identity')
    readme = (HERE / 'README.md').read_text()
    check(f'**Release candidate: JSpark3 {version}' in readme and
          '**Current published release:** [v2.0.1]' in readme, 'README candidate versus published release')
    for name, title in [('CHANGELOG.md', f'# JSpark3 {version} (draft)'),
                        ('INSTALL.md', f'# Installing JSpark3 {version}'),
                        ('UPGRADING.md', f'# Upgrading to JSpark3 {version}'),
                        ('RELEASE-GATE.md', f'# JSpark3 {version} release gate'),
                        ('release/RELEASE-NOTES.md', f'# JSpark3 {version} release candidate')]:
        check((HERE / name).read_text().splitlines()[0] == title, f'{name} identity')
    for name in ('CITATION.cff', 'CITATION.bib'):
        citation = (HERE / name).read_text()
        check(set(re.findall(r'v\d+\.\d+\.\d+', citation)) == {version} and
              f'/tree/release/{version}' in citation, f'{name} candidate identity')
    check(f'git clone --branch release/{version} ' in (HERE / 'INSTALL.md').read_text(),
          'install selects the candidate branch')
    check(meta['performance']['measured_version'] == 'v2.0.1' and
          meta['performance']['v2_0_2_measured'] is False, 'no relabeled performance measurement')
    for name, expected in meta['performance']['historical_sha256'].items():
        check(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, f'historical evidence: {name}')
    check((HERE / meta['evidence']).is_file() and meta['live_validation']['retry'] == 'pending',
          'live retry remains pending with evidence')
    if args.wheel:
        try:
            actual = check_wheel(args.wheel)
            check(args.wheel.name == pins['ENGINE_WHEEL'] and actual == pins['WHEEL_CONTENT_SHA256'],
                  'built wheel matches package source and content pin')
        except (OSError, ValueError) as error:
            check(False, str(error))
        check(subprocess.run([sys.executable, str(HERE / 'tools/payload.py'), 'verify']).returncode == 0,
              'complete payload matches SHA256SUMS')
    print(f'release gate: {len(failures)} failed' + (' (identity and wheel)' if args.wheel else ' (identity only)'))
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
