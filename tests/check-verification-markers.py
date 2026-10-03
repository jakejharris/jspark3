#!/usr/bin/env python3
"""Recheck previously verified tiny payloads, then exercise the real serve/preflight trust checks."""
import contextlib
import hashlib
import io
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def run(tree, env, script, *args):
    return subprocess.run([str(tree / 'scripts' / script), *args], env=env,
                          capture_output=True, text=True, timeout=15)


def seed(directory, manifest):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for name in ('first.bin', 'last.bin'):
        (directory / name).write_bytes(b'valid payload\n')
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
                                for p in sorted(directory.iterdir())))
    Path(str(directory) + '.verified').unlink(missing_ok=True)


def consumers_reject(tree, env, directory, manifest, kind):
    lib = subprocess.run(['bash', '-c', 'source "$1/scripts/lib.sh"; verified "$2" "$3"',
                          'marker-fixture', str(tree), str(directory), str(manifest)],
                         env=env, capture_output=True, text=True, timeout=10)
    preflight = runpy.run_path(str(tree / 'scripts/preflight.py'))
    rejected = lib.returncode != 0 and not preflight['verified'](directory, manifest)
    if kind in ('split', 'ablit-rank0', 'ablit-rank1', 'drafter'):
        weights = 'base' if kind == 'drafter' else directory.parent.name
        rank = '0' if kind == 'drafter' else directory.name[-1]
        # These are the exact serve.sh predicates and remedies, without its Docker prerequisites.
        source = (tree / 'scripts/serve.sh').read_text()
        start = source.index('  { [[ -d $third ]]')
        stop = source.index('  [[ -f $template', start)
        setup = '''set -euo pipefail
source "$1/scripts/lib.sh"
OPT_WEIGHTS=$2
load_cluster
RANK=$3
third=$(rank_dir "$RANK"); manifest=$(rank_manifest "$RANK")
'''
        p = subprocess.run(['bash', '-c', setup + source[start:stop], 'serve-marker-fixture', str(tree), weights, rank],
                           env=env, capture_output=True, text=True, timeout=10)
        remedy = 'scripts/fetch-weights.sh' if kind == 'drafter' else 'scripts/split.sh'
        rejected = rejected and p.returncode == 2 and remedy in p.stderr
        s = {'WEIGHTS': weights, 'RANK': rank, 'DRAFTER': 'dflash2', 'DATA': str(directory.parent if kind == 'drafter'
                                                                                   else directory.parent.parent),
             'TEMPLATE_SHA256': ''}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            preflight['check_third'](s)
        lines = output.getvalue().splitlines()
        subject = 'drafter' if kind == 'drafter' else f'rank {rank} third'
        rejected = rejected and any('FAIL' in line and subject in line for line in lines) and remedy in output.getvalue()
    return rejected


def interrupted(tree, env, directory, manifest, command, kind, t):
    seed(directory, manifest)
    run(tree, env, *command)
    marker = Path(str(directory) + '.verified')
    bin_dir = t / f'pause-{kind}'
    bin_dir.mkdir()
    ready = bin_dir / 'checking'
    executable = 'sha256sum' if command[0] == 'fetch-weights.sh' else 'python3'
    real = shutil.which(executable)
    stub = bin_dir / executable
    stub.write_text(f'''#!/usr/bin/env bash
if [[ $1 == -c || ${{2:-}} == check ]]; then
  touch "$CHECK_READY"
  sleep 30
else
  exec {real} "$@"
fi
''')
    stub.chmod(0o755)
    paused_env = dict(env, PATH=f'{bin_dir}:{env["PATH"]}', CHECK_READY=str(ready))
    p = subprocess.Popen([str(tree / 'scripts' / command[0]), *command[1:]], env=paused_env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        check(ready.exists() and not marker.exists(), f'{kind}: marker is absent while the recheck is running')
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
        p.communicate(timeout=5)
        check(p.returncode != 0 and not marker.exists() and consumers_reject(tree, env, directory, manifest, kind),
              f'{kind}: interrupted recheck remains unverified; trust checks reject it')
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tree, data = t / 'recipe', t / 'data'
        for part in ('scripts', 'config', 'manifests', 'template'):
            shutil.copytree(HERE / part, tree / part, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(HERE / 'pins.env', tree / 'pins.env')
        data.mkdir()
        (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                               (HERE / 'cluster.env.example').read_text()))
        env = dict(os.environ, CLUSTER_ENV=str(tree / 'cluster.env'), PYTHONDONTWRITEBYTECODE='1')
        cases = (
            ('split', 'base/rank0', 'manifests/base/rank0.sha256', ['split.sh', '--verify-only']),
            ('ablit-rank0', 'ablit/rank0', 'manifests/ablit/rank0.sha256',
             ['split.sh', '--weights', 'ablit', '--rank', '0', '--verify-only']),
            ('ablit-rank1', 'ablit/rank1', 'manifests/ablit/rank1.sha256',
             ['split.sh', '--weights', 'ablit', '--rank', '1', '--verify-only']),
            ('drafter', 'drafter', 'manifests/inputs/drafter.sha256',
             ['fetch-weights.sh', '--drafter-only', '--verify-only']),
            ('base', 'base/weights', 'manifests/inputs/base-weights.sha256',
             ['fetch-weights.sh', '--drafter', 'none', '--verify-only']),
            ('source', 'ablit/source', 'manifests/inputs/ablit-source.sha256',
             ['fetch-weights.sh', '--weights', 'ablit', '--drafter', 'none', '--verify-only']),
            ('convert', 'ablit/weights', 'manifests/inputs/ablit-weights.sha256', ['convert-ablit.sh', '--verify-only']),
        )
        for kind, rel, listed, command in cases:
            directory, manifest = data / rel, tree / listed
            marker = Path(str(directory) + '.verified')
            for damage in ('corrupt', 'delete-file', 'missing-dir', 'missing-manifest'):
                seed(directory, manifest)
                # Keep the third trusted while testing the drafter's later serve predicate.
                if kind == 'drafter':
                    third, third_list = data / 'base/rank0', tree / 'manifests/base/rank0.sha256'
                    seed(third, third_list)
                    run(tree, env, 'split.sh', '--verify-only')
                first = run(tree, env, *command)
                check(first.returncode == 0 and marker.exists(), f'{kind}/{damage}: initial check verifies')
                old_marker = marker.read_bytes() if marker.exists() else b''
                manifest_bytes = manifest.read_bytes()
                payload = directory / 'last.bin'
                if damage == 'corrupt':
                    payload.write_bytes(b'wrong payload\n')
                elif damage == 'delete-file':
                    payload.unlink()
                elif damage == 'missing-dir':
                    shutil.rmtree(directory)
                else:
                    payload.write_bytes(b'wrong payload\n')
                    manifest.unlink()
                # A dry run may reject a missing manifest, but must not mutate data or trust.
                before = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
                dry_run = run(tree, env, *command, '--dry-run')
                after = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
                check(before == after and marker.read_bytes() == old_marker,
                      f'{kind}/{damage}: dry run leaves payloads and markers untouched')
                if kind == 'ablit-rank0' and damage == 'missing-manifest':
                    ordinary = run(tree, env, 'split.sh', '--weights', 'ablit', '--rank', '0')
                    refusal = '--weights ablit is not available in this tree (no manifests/ablit); use base'
                    check(ordinary.returncode == dry_run.returncode == 2 and refusal in ordinary.stderr
                          and refusal in dry_run.stderr and marker.read_bytes() == old_marker
                          and before == {p: p.read_bytes() for p in data.rglob('*') if p.is_file()},
                          'ablit-rank0/missing-manifest: normal split and dry run keep the availability refusal and data')
                result = run(tree, env, *command)
                check(result.returncode != 0 and not marker.exists(),
                      f'{kind}/{damage}: real recheck fails and revokes the marker')
                manifest.write_bytes(manifest_bytes)
                check(consumers_reject(tree, env, directory, manifest, kind),
                      f'{kind}/{damage}: trust checks reject it and served inputs name the recovery command')
                if damage == 'missing-manifest':
                    result = run(tree, env, *command)
                    check(result.returncode != 0 and not marker.exists(),
                          f'{kind}/{damage}: restoring only the manifest still fails the corrupt payload recheck')
                seed(directory, manifest)
                check(consumers_reject(tree, env, directory, manifest, kind),
                      f'{kind}/{damage}: repaired bytes alone remain unverified until a real check passes')
                repaired = run(tree, env, *command)
                check(repaired.returncode == 0 and marker.read_text().strip() == hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      f'{kind}/{damage}: repairing and rechecking restores verification')
            if kind in ('split', 'drafter', 'convert'):
                interrupted(tree, env, directory, manifest, command, kind, t)

        directory, manifest = data / 'base/rank0', tree / 'manifests/base/rank0.sha256'
        marker = Path(str(directory) + '.verified')
        partial = Path(str(directory) + '.partial')
        partial_marker = Path(str(partial) + '.verified')
        for valid in (False, True):
            seed(directory, manifest)
            marker.write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + '\n')
            directory.rename(partial)
            partial_marker.write_bytes(marker.read_bytes())
            if not valid:
                (partial / 'last.bin').unlink()
            result = run(tree, env, 'split.sh', '--verify-only')
            check((result.returncode == 0 and directory.is_dir() and marker.exists()) if valid
                  else (result.returncode != 0 and not marker.exists() and partial.is_dir()),
                  f'partial third: {"valid output promoted and marked" if valid else "missing file clears destination marker"}')
            check(not partial_marker.exists(), 'partial third: old partial marker is revoked')
            if partial.exists():
                shutil.rmtree(partial)
    print(f'check-verification-markers: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
