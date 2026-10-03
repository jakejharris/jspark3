#!/usr/bin/env python3
"""Version prechecks read hf's venv as data and never execute either venv's startup hooks."""
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
PIN = FIXTURE['PINS']['HF_HUB_VERSION']
failures = []

# Actively attempt DNS at startup, stopping it before any network operation.
GUARD = '''import json, os, sys
def audit(event, args):
    forbidden = event.startswith('socket.') or (event == 'import' and args[0].startswith('huggingface_hub'))
    if forbidden or event == 'cpython.run_command':
        with open(os.environ['VERSION_AUDIT'], 'a') as stream:
            stream.write(json.dumps({'event': event, 'args': repr(args), 'python': sys.executable,
                'env': {k: os.environ.get(k) for k in ('HF_ENDPOINT', 'HF_TOKEN', 'HUGGINGFACE_CO_STAGING',
                    'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN')}}) + '\\n')
    if forbidden:
        raise RuntimeError('version test forbids network and huggingface_hub imports')
sys.addaudithook(audit)
import socket
try:
    socket.getaddrinfo('startup.example.invalid', 443)
except RuntimeError:
    pass
'''
CLI = '''#!/usr/bin/env python3
import os, sys
with open(os.environ['CLI_AUDIT'], 'a') as stream:
    stream.write(repr(sys.argv) + '\\n')
raise SystemExit('the version precheck must not execute hf')
'''


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tree = FIXTURE['prepare'](t)
        data = t / 'data'
        data.mkdir()
        cluster = tree / 'cluster.env'
        cluster.write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}', (tree / 'cluster.env.example').read_text()))
        bin_dir = t / 'bin'
        bin_dir.mkdir()
        hf = bin_dir / 'hf'
        packages = install_hf(hf, CLI, PIN)
        (packages / 'version_audit.py').write_text(GUARD)
        (packages / 'version_audit.pth').write_text('import version_audit\n')
        poison = packages / 'huggingface_hub'
        poison.mkdir()
        (poison / '__init__.py').write_text('raise RuntimeError("the Hub package must not be imported")\n')
        system = t / 'system'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(system)
        system_packages = system / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        (system_packages / 'version_audit.py').write_text(GUARD)
        (system_packages / 'version_audit.pth').write_text('import version_audit\n')
        env = dict(os.environ, PATH=f'{system / "bin"}:{bin_dir}:/usr/bin:/bin', CLUSTER_ENV=str(cluster),
                   HUGGINGFACE_CO_STAGING='1', VERSION_AUDIT=str(t / 'audit'), CLI_AUDIT=str(t / 'cli'))
        for key in ('PYTHONPATH', 'PYTHONHOME', 'HF_HUB_OFFLINE', 'HF_ENDPOINT', 'HF_TOKEN'):
            env.pop(key, None)
        install = (f'python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub=={PIN}", '
                   'then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md, \'What you need\')')

        def fetch():
            # The existing empty-selection refusal comes immediately AFTER the real version check.
            return subprocess.run([str(tree / 'scripts/fetch-weights.sh'), '--weights', 'base',
                                   '--drafter-only', '--drafter', 'none'], env=env,
                                  capture_output=True, text=True, timeout=40)

        def preflight():
            code = ('import runpy; p=runpy.run_path(' + repr(str(tree / 'scripts/preflight.py')) + '); '
                    + 'p["check_fetch"]({"HF_HUB_VERSION": ' + repr(PIN) + ', "WEIGHTS": "base"})')
            return subprocess.run([str(system / 'bin/python3'), '-I', '-S', '-c', code], env=env,
                                  capture_output=True, text=True, timeout=40)

        metadata = packages / 'huggingface_hub-0.0.0.dist-info/METADATA'
        original = metadata.read_text()
        for system_version in (None, '9.8.7'):
            if system_version:
                other = system_packages / 'huggingface_hub-9.8.7.dist-info'
                other.mkdir()
                (other / 'METADATA').write_text(f'Name: huggingface-hub\nVersion: {system_version}\n')
            for endpoint in (None, 'https://mirror.invalid:8443/'):
                for token in (None, 'fake-version-token-not-a-credential'):
                    for key, value in (('HF_ENDPOINT', endpoint), ('HF_TOKEN', token)):
                        env.pop(key, None)
                        if value is not None:
                            env[key] = value
                    p = fetch()
                    label = f'system={system_version}, mirror={bool(endpoint)}, token={bool(token)}'
                    check(p.returncode == 2 and p.stderr.strip() ==
                          'fetch-weights.sh: --drafter-only with --drafter none fetches nothing',
                          f'{label}: pinned hf venv passes the real precheck')
                    p = preflight()
                    check(p.returncode == 0 and p.stdout.startswith('PASS'), f'{label}: preflight passes')

        for version in ('0.0.1', PIN + 'b1', PIN + '.post1'):
            metadata.write_text(original.replace('Version: ' + PIN, 'Version: ' + version))
            p = fetch()
            expected = (f'fetch-weights.sh: the Hugging Face CLI is version {version}; this release pins '
                        f'HF_HUB_VERSION={PIN}. Install it in a virtual environment: {install}')
            check(p.returncode == 2 and p.stderr.strip() == expected, f'{version}: exact existing refusal and hint')
            p = preflight()
            check(p.returncode == 0 and p.stdout.startswith('FAIL') and f'it is {version}:' in p.stdout,
                  f'{version}: preflight refuses the entire version, without a regex substring match')
        metadata.write_text(original)

        # Standard env shebangs use the PATH interpreter; the unknown launcher never runs.
        body = hf.read_text().partition('\n')[2]
        env['PATH'] = f'{bin_dir / "hf-env/bin"}:{bin_dir}:/usr/bin:/bin'
        for header in ('#!/usr/bin/env python3', '#!/usr/bin/env -S python3 -s'):
            hf.write_text(header + '\n' + body)
            p = fetch()
            check('fetches nothing' in p.stderr, f'{header}: resolves the same interpreter as the kernel')
        hf.write_text('#!/bin/sh\nexit 0\n')
        p = fetch()
        check('CLI is version unknown;' in p.stderr, 'unsupported launcher fails closed')
        hf.unlink()
        p = fetch()
        check(p.returncode == 2 and p.stderr.strip() ==
              f"fetch-weights.sh: no 'hf' command: install the pinned Hugging Face CLI in a virtual environment: {install}",
              'missing hf: exact existing refusal and hint')
        check('not installed:' in preflight().stdout, 'missing hf: preflight keeps its refusal')

        # Exercise the pip/distlib trampoline used for venv paths containing spaces.
        spaced = t / 'venv with spaces'
        spaced.mkdir()
        packages2 = install_hf(spaced / 'hf', CLI, PIN)
        (packages2 / 'version_audit.py').write_text(GUARD)
        (packages2 / 'version_audit.pth').write_text('import version_audit\n')
        env['PATH'] = f'{system / "bin"}:{spaced}:/usr/bin:/bin'
        check('fetches nothing' in fetch().stderr, 'pip shell trampoline selects its quoted venv interpreter')
        check(not (t / 'audit').exists(), 'zero startup hook executions in the outer and hf venvs')
        check(not (t / 'cli').exists(), 'zero CLI executions')
        # Prove the audit guard is live; no connection can be made by this control.
        p = subprocess.run([str(system / 'bin/python3'), '-c', 'import socket; socket.socket()'], env=env,
                           capture_output=True, text=True)
        check(p.returncode != 0 and 'version test forbids network' in p.stderr,
              'negative control: an attempted socket is caught before it can connect')
        rows = [json.loads(line) for line in (t / 'audit').read_text().splitlines()]
        check(any(row['event'] == 'socket.getaddrinfo' for row in rows),
              'negative control: the startup hook actively attempts DNS when isolation is removed')
    print(f'check-hf-version: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
