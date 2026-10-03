#!/usr/bin/env python3
"""Record every downloader's endpoint/auth environment; all wrapper HTTP responses are in memory."""
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
TOKEN = 'fake-endpoint-token-not-a-credential'
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


# Supply only the wrapper's tiny metadata/access responses. Neither the wrapper nor the fake CLI can use sockets.
HTTP = '''import io, json, os, socket, urllib.request
def no_network(*args, **kwargs):
    raise AssertionError('endpoint fixture forbids network access')
socket.socket = no_network
def open_request(self, request, *args, **kwargs):
    with open(os.environ['WRAPPER_CALLS'], 'a') as f:
        f.write(json.dumps({'url': request.full_url, 'auth': request.get_header('Authorization')}) + '\\n')
    return io.BytesIO(b'x' if '/resolve/' in request.full_url else b'[]')
urllib.request.OpenerDirector.open = open_request
'''

HF = '''#!/usr/bin/env python3
import json, os, pathlib, sys
assert sys.argv[1] == 'download', 'the CLI must not run for a version check'
with open(os.environ['HF_CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in (
        'HF_ENDPOINT', 'HUGGINGFACE_CO_STAGING', 'HF_HUB_ENDPOINT', 'HF_INFERENCE_ENDPOINT',
        'HF_TOKEN', 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'HF_HUB_DISABLE_XET')}}) + '\\n')
repo, *args = sys.argv[2:]
names = []
while args and not args[0].startswith('--'):
    names.append(args.pop(0))
dest = pathlib.Path(args[args.index('--local-dir') + 1])
dest.mkdir(parents=True, exist_ok=True)
for name in names or ['stand-in.txt']:
    (dest / name).write_text(repo + '\\n')
'''


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for label, endpoint, staging, args in (
            ('default with staging', None, '1', ['--weights', 'ablit']),
            ('empty endpoint with staging', '', 'YES', ['--weights', 'ablit']),
            ('explicit mirror with staging', 'https://mirror.invalid:8443/', '1', ['--weights', 'ablit']),
            ('explicit hub-ci with staging', 'https://hub-ci.huggingface.co', '1', ['--weights', 'ablit']),
            ('default without staging', None, None, ['--weights', 'ablit']),
            ('anonymous base with staging', None, '1', ['--weights', 'base', '--drafter', 'none']),
            ('anonymous drafter with staging', None, '1', ['--weights', 'base', '--drafter-only']),
        ):
            t = Path(tmp) / label.replace(' ', '-')
            tree = FIXTURE['prepare'](t)
            data, bin_dir = t / 'data', t / 'bin'
            data.mkdir()
            bin_dir.mkdir()
            (bin_dir / 'sitecustomize.py').write_text(HTTP)
            hf = bin_dir / 'hf'
            install_hf(hf, HF, FIXTURE['PINS']['HF_HUB_VERSION'])
            hf.chmod(0o755)
            (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                                   (tree / 'cluster.env.example').read_text()))
            env = dict(os.environ, CLUSTER_ENV=str(tree / 'cluster.env'), PATH=f'{bin_dir}:{os.environ["PATH"]}',
                       PYTHONPATH=str(bin_dir), PYTHONDONTWRITEBYTECODE='1', HF_TOKEN=TOKEN,
                       HF_HOME=str(t / 'hf-home'), HF_CALLS=str(t / 'calls'), WRAPPER_CALLS=str(t / 'wrapper'),
                       HF_HUB_ENDPOINT='https://hub-ci.huggingface.co',
                       HF_INFERENCE_ENDPOINT='https://inference.invalid')
            for key, value in (('HF_ENDPOINT', endpoint), ('HUGGINGFACE_CO_STAGING', staging)):
                env.pop(key, None)
                if value is not None:
                    env[key] = value
            env.pop('HF_HUB_DISABLE_IMPLICIT_TOKEN', None)
            env.pop('HF_HUB_DISABLE_XET', None)
            if args[1] == 'base':
                env.pop('HF_TOKEN')
            selected = (endpoint or 'https://huggingface.co').rstrip('/')
            repos = [FIXTURE['PINS'][name] for name in ('BASE_REPO', 'ABLIT_SOURCE_REPO', 'DRAFTER_REPO')]
            if args[1] == 'base':
                repos = [repos[2] if '--drafter-only' in args else repos[0]]
            partial = data / 'base/weights/.cache/huggingface/download/fixture.incomplete'
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b'existing partial download')
            for attempt in range(2 if label == 'explicit mirror with staging' else 1):
                p = subprocess.run([str(tree / 'scripts/fetch-weights.sh'), *args], env=env,
                                   capture_output=True, text=True, timeout=30)
                calls = [json.loads(line) for line in (t / 'calls').read_text().splitlines()] if (t / 'calls').exists() else []
                calls = calls[attempt * len(repos):]
                wrapper = [json.loads(line) for line in (t / 'wrapper').read_text().splitlines()] if (t / 'wrapper').exists() else []
                what = label + (' resumed' if attempt else '')
                check(p.returncode == 0 and [c['args'][1] for c in calls] == repos,
                      f'{what}: expected downloads complete and verify')
                check(bool(calls) and all(c['env']['HF_ENDPOINT'] == selected
                                         and c['env']['HUGGINGFACE_CO_STAGING'] is None for c in calls),
                      f'{what}: every hf download is pinned to {selected}, staging override removed')
                check(bool(wrapper) and all(c['url'].startswith(selected + '/') for c in wrapper),
                      f'{what}: wrapper uses the same endpoint')
                check(bool(calls) and all((c['env']['HF_TOKEN'] == TOKEN if c['args'][1] == FIXTURE['PINS']['ABLIT_SOURCE_REPO']
                                          else c['env']['HF_TOKEN'] is None and c['env']['HF_HUB_DISABLE_IMPLICIT_TOKEN'] == '1')
                                         and c['env']['HF_HUB_DISABLE_XET'] == '1' for c in calls),
                      f'{what}: only the source receives the fake token; Xet default preserved')
                check(partial.read_bytes() == b'existing partial download'
                      and bool(calls) and all('--force-download' not in c['args'] and '--revision' in c['args']
                                             and c['args'][c['args'].index('--local-dir') + 1].startswith(str(data) + '/')
                                             for c in calls),
                      f'{what}: partial cache retained and pinned download arguments preserve resume')
    print(f'check-fetch-endpoint: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
