#!/usr/bin/env python3
"""Check that a Hugging Face token never reaches the console or the script logs, on the gated (ablit) fetch path.

    python3 tests/check-token-canary.py

Works on a copy of this tree that is made ready for the ablit fetch: stand-in converter and conversion pins, and
input lists that describe small stand-in files. It runs scripts/fetch-weights.sh --weights ablit against a local
stand-in for Hugging Face (HF_ENDPOINT) and a stand-in hf at the pinned version. HF_TOKEN is set to a fresh canary, and
a second canary is stored as a login token (HF_HOME/token).
Passes when:
  - neither canary appears in stdout, stderr or logs/*.log;
  - the gated requests did carry HF_TOKEN, so the path really ran;
  - the access check reads the first listed weight shard, and the gated download asks for the listed files only;
  - the log is mode 0600;
  - a stand-in hf that prints both tokens is caught, so the check can fail.
No network, no data, no docker.
"""
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cluster_fixture import select
from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
PINS = dict(re.findall(r'^([A-Z_0-9]+)=(\S*)', (HERE / 'pins.env').read_text(), re.M))
requests = []
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


class Hub(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        requests.append({'path': self.path, 'auth': self.headers.get('Authorization', '')})
        tree = '/tree/' in self.path
        body = b'[]' if tree else b'{'
        self.send_response(200 if tree else 206)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


STUB_HF = '''#!/usr/bin/env bash
if [[ ${1:-} != download ]]; then exit 97; fi
printf '%s\\n' "$*" >>"$HF_ARGS"
repo=$2 dir='' files=()
shift 2
while (($#)) && [[ $1 != --* ]]; do files+=("$1"); shift; done
while (($#)); do [[ $1 == --local-dir ]] && dir=$2; shift; done
((${#files[@]})) || files=(stand-in.txt)
mkdir -p "$dir" && for f in "${files[@]}"; do printf '%s\\n' "$repo" >"$dir/$f"; done
echo "stand-in hf: downloaded $repo"
if [[ -n ${LEAK:-} ]]; then echo "token ${HF_TOKEN:-} stored $(cat "$HF_HOME/token")"; fi
'''


def prepare(t):
    tree = t / 'tree'
    for part in ('scripts', 'config', 'manifests'):
        shutil.copytree(HERE / part, tree / part, symlinks=True)
    for part in ('pins.env', 'cluster.env.example'):
        shutil.copy2(HERE / part, tree / part)
    pins = (tree / 'pins.env').read_text()
    pins = re.sub(r'(?m)^ABLIT_CONVERTER_SHA256=.*$', 'ABLIT_CONVERTER_SHA256=' + 'a' * 64, pins)
    (tree / 'pins.env').write_text(re.sub(r'(?m)^ABLIT_CONVERSION_ID=.*$', 'ABLIT_CONVERSION_ID=' + 'b' * 64, pins))
    inputs = tree / 'manifests/inputs'
    for name, repo in (('base-weights', PINS['BASE_REPO']), ('ablit-source', PINS['ABLIT_SOURCE_REPO']),
                       ('drafter', PINS['DRAFTER_REPO']), ('ablit-weights', 'converted')):
        digest = hashlib.sha256(f'{repo}\n'.encode()).hexdigest()
        names = (SHARD, 'stand-in.txt') if name == 'ablit-source' else ('stand-in.txt',)
        (inputs / f'{name}.sha256').write_text(''.join(f'{digest}  {n}\n' for n in names))
    return tree


def run(t, leak):
    tree = prepare(t / ('leak' if leak else 'clean'))
    home = t / ('leak' if leak else 'clean') / 'hf-home'
    data = t / ('leak' if leak else 'clean') / 'data'
    home.mkdir()
    data.mkdir()
    stored, token = 'hf_STORED' + secrets.token_hex(12), 'hf_CANARY' + secrets.token_hex(12)
    (home / 'token').write_text(stored)
    (t / 'bin').mkdir(exist_ok=True)
    hf = t / 'bin/hf'
    install_hf(hf, STUB_HF, PINS['HF_HUB_VERSION'])
    hf.chmod(0o755)
    cluster = tree / 'cluster.env'
    cluster.write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}', (tree / 'cluster.env.example').read_text()))
    args = t / ('hf-args-leak' if leak else 'hf-args-clean')
    env = dict(os.environ, HF_TOKEN=token, HF_HOME=str(home), HF_ENDPOINT=f'http://127.0.0.1:{PORT}',
               PATH=f'{t / "bin"}:{os.environ["PATH"]}', HF_ARGS=str(args))
    # Loopback only: drop any proxy the caller set, and never send a local request through one.
    for name in ('http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(name, None)
        env.pop(name.upper(), None)
    env.update(no_proxy='127.0.0.1,localhost', NO_PROXY='127.0.0.1,localhost')
    env.pop('LEAK', None)
    if leak:
        env['LEAK'] = '1'
    select(env, tree, t)
    before = len(requests)
    p = subprocess.run([str(tree / 'scripts/fetch-weights.sh'), '--weights', 'ablit'], capture_output=True, text=True,
                       env=env, timeout=300)
    logs = ''.join(f.read_text() for f in sorted((tree / 'logs').glob('*.log')))
    seen = p.stdout + p.stderr + logs
    found = sum(seen.count(c) for c in (token, stored))
    return p, tree, token, requests[before:], found, args.read_text().splitlines() if args.exists() else []


def main():
    global PORT
    server = ThreadingHTTPServer(('127.0.0.1', 0), Hub)
    PORT = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        p, tree, token, mine, found, calls = run(t, leak=False)
        check(p.returncode == 0 and 'fetch-weights.sh: done' in p.stdout,
              'the ablit fetch path ran to the end (base, gated source, drafter)')
        check(any(r['auth'] == f'Bearer {token}' for r in mine) and any('/resolve/' in r['path'] for r in mine),
              'the gated requests carried HF_TOKEN (access check and size request)')
        check(not any(r['auth'] for r in mine if PINS['ABLIT_SOURCE_REPO'] not in r['path']),
              'requests for the base weights and the drafter carried no token')
        probe = f"/{PINS['ABLIT_SOURCE_REPO']}/resolve/{PINS['ABLIT_SOURCE_REV']}/{SHARD}"
        check(any(r['path'] == probe and r['auth'] == f'Bearer {token}' for r in mine),
              'the access check reads the first weight shard the source list names, not a public small file')
        source = [c.split() for c in calls if c.split()[1:2] == [PINS['ABLIT_SOURCE_REPO']]]
        check(len(source) == 1 and source[0][2:4] == [SHARD, 'stand-in.txt'] and source[0][4] == '--revision',
              'the gated download asks for exactly the files manifests/inputs/ablit-source.sha256 lists')
        check(all(c.split()[2] == '--revision' for c in calls if c.split()[1] != PINS['ABLIT_SOURCE_REPO']),
              'the base weights and the drafter are downloaded whole, as before')
        log = tree / 'logs/fetch-weights.log'
        check(log.exists() and (log.stat().st_mode & 0o777) == 0o600, 'logs/fetch-weights.log is mode 0600')
        check(found == 0, f'neither canary is in stdout, stderr or logs/*.log ({found} found)')
        p, tree, token, mine, found, calls = run(t, leak=True)
        check(found >= 2, f'control: a stand-in hf that prints both tokens is caught ({found} found)')
    server.shutdown()
    print(f'check-token-canary: {len(failures)} failed')
    sys.exit(1 if failures else 0)


PORT = 0
SHARD = 'model-00001-of-00002.safetensors'
if __name__ == '__main__':
    main()
