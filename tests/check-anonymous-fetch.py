#!/usr/bin/env python3
"""Check that the base weights and the drafter are fetched anonymously, metadata included, even with HF_TOKEN set.

    python3 tests/check-anonymous-fetch.py [--real-hf]

Runs scripts/fetch-weights.sh (base weights, then --drafter-only) against a local stand-in for Hugging Face
(HF_ENDPOINT) and a stand-in 'hf' command at the pinned version, with HF_TOKEN set to a sentinel and the same
sentinel stored as a login token (HF_HOME/token). Passes when no request to the stand-in carries an Authorization
header or the sentinel, and the download command runs without HF_TOKEN and with HF_HUB_DISABLE_IMPLICIT_TOKEN=1. Also
checks that another hf version is refused before anything is fetched. No network, no data, no docker.

--real-hf uses the pinned hf on PATH instead of the stand-in command, so the CLI's own requests are checked too, and
first proves the check can fail: hf run directly with HF_TOKEN must send it to the stand-in.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
SENTINEL = 'sentinel-token-not-real-q7k'
PINS = dict(re.findall(r'^([A-Z_]+)=(\S*)', (HERE / 'pins.env').read_text(), re.M))
requests = []
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


class Hub(BaseHTTPRequestHandler):
    def answer(self, send_body):
        requests.append({'method': self.command, 'path': self.path, 'headers': dict(self.headers)})
        tree = '/tree/' in self.path
        body = b'[]' if tree else b'{"error": "stand-in: not found"}'
        self.send_response(200 if tree else 404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        self.answer(True)

    def do_HEAD(self):  # noqa: N802 - http.server API
        self.answer(False)

    def log_message(self, *args):
        pass


FAKE_HF = '''#!/usr/bin/env bash
if [[ ${1:-} != download ]]; then exit 97; fi
{ printf 'ARGS %s\\n' "$*"; printf 'HF_TOKEN=%s\\n' "${HF_TOKEN-<unset>}";
  printf 'HF_HUB_DISABLE_IMPLICIT_TOKEN=%s\\n' "${HF_HUB_DISABLE_IMPLICIT_TOKEN-<unset>}"; } >>"$HF_CALLS"
'''


def carries_token(reqs):
    return any('authorization' in {k.lower() for k in r['headers']} for r in reqs) or SENTINEL in json.dumps(reqs)


def main():
    real = '--real-hf' in sys.argv[1:]
    server = ThreadingHTTPServer(('127.0.0.1', 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / 'bin').mkdir()
        (t / 'data').mkdir()
        (t / 'hf-home').mkdir()
        (t / 'hf-home/token').write_text(SENTINEL)  # as 'hf auth login' stores it
        hf = t / 'bin/hf'
        calls = t / 'hf-calls'
        example = (HERE / 'cluster.env.example').read_text()
        cluster = t / 'cluster.env'
        cluster.write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={t / "data"}', example))
        env = dict(os.environ, CLUSTER_ENV=str(cluster), HF_TOKEN=SENTINEL, HF_CALLS=str(calls),
                   HF_HOME=str(t / 'hf-home'), HF_ENDPOINT=f'http://127.0.0.1:{server.server_port}',
                   PATH=os.environ['PATH'] if real else f'{t / "bin"}:{os.environ["PATH"]}')
        for name in ('HF_HUB_DISABLE_IMPLICIT_TOKEN', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_XET'):
            env.pop(name, None)
        # Loopback only: drop any proxy the caller set, and never send a local request through one.
        for name in ('http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name, None)
            env.pop(name.upper(), None)
        env.update(no_proxy='127.0.0.1,localhost', NO_PROXY='127.0.0.1,localhost')

        def fetch(*args):
            return subprocess.run([str(HERE / 'scripts/fetch-weights.sh'), *args], capture_output=True, text=True,
                                  env=env, timeout=120)

        if real:
            out = subprocess.run([sys.executable, '-I', '-S', str(HERE / 'scripts/hf-version.py')],
                                 capture_output=True, text=True, env=env).stdout.strip()
            check(out == PINS['HF_HUB_VERSION'],
                  f'the hf on PATH is the pinned {PINS["HF_HUB_VERSION"]}')
            subprocess.run(['hf', 'download', PINS['BASE_REPO'], 'config.json', '--revision', PINS['BASE_REV'],
                            '--local-dir', str(t / 'control')], capture_output=True, text=True, env=env, timeout=120)
            check(carries_token(requests), 'control: hf run directly with HF_TOKEN sends it to the stand-in')
            requests.clear()
        else:
            install_hf(hf, FAKE_HF, '0.0.1')
            hf.chmod(0o755)
            p = fetch()
            check(p.returncode == 2 and f'HF_HUB_VERSION={PINS["HF_HUB_VERSION"]}' in p.stderr and not requests
                  and not calls.exists(), 'another hf version is refused before anything is fetched')
            install_hf(hf, FAKE_HF, PINS['HF_HUB_VERSION'])

        for args, repo in (((), PINS['BASE_REPO']), (('--drafter-only',), PINS['DRAFTER_REPO'])):
            what = 'drafter' if args else 'base weights'
            before = len(requests)
            fetch(*args)  # the stand-in downloads nothing, so the check after the download fails; that is expected
            mine = requests[before:]
            check(any(f'/api/models/{repo}/tree/' in r['path'] for r in mine), f'{what}: size metadata was requested')
            check(not carries_token(mine), f'{what}: no request carries a token (HF_TOKEN set, a login token stored)')
            if real:
                check(any(r['path'] != mine[0]['path'] for r in mine), f'{what}: the real hf made its own requests')
                continue
            log = calls.read_text() if calls.exists() else ''
            runs = [chunk for chunk in log.split('ARGS ')[1:] if repo in chunk]
            check(len(runs) == 1 and 'HF_TOKEN=<unset>' in runs[0] and 'HF_HUB_DISABLE_IMPLICIT_TOKEN=1' in runs[0],
                  f'{what}: hf download runs without HF_TOKEN and with HF_HUB_DISABLE_IMPLICIT_TOKEN=1')
    server.shutdown()
    print(f'check-anonymous-fetch: {len(failures)} failed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
