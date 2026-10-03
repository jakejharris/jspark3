#!/usr/bin/env python3
"""Reject missing/invalid ablit credentials before fetching any bytes, including base or drafter weights."""
import os
import re
import runpy
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
TOKEN = 'fake-gate-order-token-not-a-credential'
failures = []
requests = []
received_bytes = 0
gate_status = 200


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


class Hub(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        global received_bytes
        requests.append(self.path)
        if self.path.startswith('/payload/'):
            code, body = 200, (urllib.parse.unquote(self.path[len('/payload/'):]) + '\n').encode()
        elif '/resolve/' in self.path:
            code, body = gate_status, b'x' if gate_status == 200 else b''
        else:
            code, body = 200, b'[]'
        self.send_response(code)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        received_bytes += len(body)

    def log_message(self, *args):
        pass


HF = '''#!/usr/bin/env python3
import os, pathlib, sys, urllib.parse, urllib.request
assert sys.argv[1] == 'download', 'the CLI must not run for a version check'
repo, *args = sys.argv[2:]
with open(os.environ['HF_ARGS'], 'a') as f:
    f.write(repo + '\\n')
names = []
while args and not args[0].startswith('--'):
    names.append(args.pop(0))
dest = pathlib.Path(args[args.index('--local-dir') + 1])
with urllib.request.urlopen(os.environ['HF_ENDPOINT'] + '/payload/' + urllib.parse.quote(repo), timeout=5) as response:
    body = response.read()
dest.mkdir(parents=True, exist_ok=True)
for name in names or ['stand-in.txt']:
    (dest / name).write_bytes(body)
'''


def prepare(t, endpoint):
    tree = FIXTURE['prepare'](t)
    data = t / 'data'
    data.mkdir()
    (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                           (tree / 'cluster.env.example').read_text()))
    (t / 'bin').mkdir()
    hf = t / 'bin/hf'
    install_hf(hf, HF, FIXTURE['PINS']['HF_HUB_VERSION'])
    hf.chmod(0o755)
    env = dict(os.environ, CLUSTER_ENV=str(tree / 'cluster.env'), PATH=f'{t / "bin"}:{os.environ["PATH"]}',
               HF_ENDPOINT=endpoint, HF_TOKEN=TOKEN, HF_ARGS=str(t / 'calls'),
               NO_PROXY='127.0.0.1', no_proxy='127.0.0.1', PYTHONDONTWRITEBYTECODE='1')
    return tree, data, env


def fetch(tree, env, *args):
    return subprocess.run([str(tree / 'scripts/fetch-weights.sh'), '--weights', 'ablit', *args],
                          env=env, capture_output=True, text=True, timeout=30)


def main():
    global received_bytes, gate_status
    help_result = subprocess.run([str(HERE / 'scripts/fetch-weights.sh'), '--help'],
                                 capture_output=True, text=True, timeout=10)
    help_text = ' '.join(help_result.stdout.split())
    check(help_result.returncode == 0 and 'With ablit selected, a fetch checks source access first, --drafter-only included.' in help_text,
          'help explains the ablit access check, including --drafter-only')
    check('--verify-only needs no token' in help_text, 'help says --verify-only needs no token')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://127.0.0.1:{server.server_port}'
    errors = {
        'missing': 'the refusal-removed source is gated: accept its terms on its Hugging Face page with your account, then export HF_TOKEN=<your token> and run this again',
        '401': 'Hugging Face did not accept HF_TOKEN: create a read token in your Hugging Face settings, export HF_TOKEN=<your token>, and run this again',
        '403': "your Hugging Face account has not accepted the source's terms yet: open https://huggingface.co/" + FIXTURE['PINS']['ABLIT_SOURCE_REPO'] + ', accept them, and run this again',
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for mode, args in (('missing', ()), ('401', ()), ('403', ()),
                               ('missing', ('--drafter-only',)), ('401', ('--drafter-only',))):
                case = t / (mode + ('-drafter' if args else ''))
                tree, data, env = prepare(case, endpoint)
                if mode == 'missing':
                    env.pop('HF_TOKEN', None)
                gate_status = 200 if mode == 'missing' else int(mode)
                requests.clear()
                received_bytes = 0
                p = fetch(tree, env, *args)
                label = f'{mode}{" drafter-only" if args else ""}'
                check(p.returncode == 2 and f'fetch-weights.sh: {errors[mode]}' in p.stderr.splitlines(),
                      f'{label}: original refusal message, exit 2')
                check(p.stdout.startswith((tree / 'config/ablit-notice.txt').read_text()),
                      f'{label}: ablit notice prints first')
                check(received_bytes == 0 and not (case / 'calls').exists() and not list(data.rglob('*')),
                      f'{label}: zero bytes fetched ({received_bytes}), zero downloads and no data changes')
                check(not requests if mode == 'missing' else len(requests) == 1 and '/resolve/' in requests[0],
                      f'{label}: no metadata or payload request before access is accepted')

            tree, data, env = prepare(t / 'accepted', endpoint)
            gate_status = 200
            requests.clear()
            received_bytes = 0
            p = fetch(tree, env)
            calls = (t / 'accepted/calls').read_text().splitlines()
            check(p.returncode == 0 and calls == [FIXTURE['PINS'][k] for k in ('BASE_REPO', 'ABLIT_SOURCE_REPO', 'DRAFTER_REPO')]
                  and received_bytes > 0 and '/resolve/' in requests[0],
                  'accepted credentials: probe precedes all three downloads, and valid files are fetched')
            check(p.stdout.startswith((tree / 'config/ablit-notice.txt').read_text()), 'accepted credentials: notice prints first')
            requests.clear()
            received_bytes = 0
            env.pop('HF_TOKEN')
            p = fetch(tree, env, '--verify-only')
            check(p.returncode == 0 and not requests and received_bytes == 0
                  and (t / 'accepted/calls').read_text().splitlines() == calls,
                  '--verify-only still checks the files without credentials or downloads')
    finally:
        server.shutdown()
        server.server_close()
    print(f'check-fetch-gate-order: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
