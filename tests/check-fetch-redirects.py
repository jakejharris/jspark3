#!/usr/bin/env python3
"""Exercise authenticated fetch redirects through two real loopback origins; never contact the network."""
import os
import re
import runpy
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
TOKEN = 'fake-redirect-token-not-a-credential'
requests = []
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


class Hub(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        requests.append((self.server.server_port, self.path, self.headers.get('Authorization')))
        if self.path.startswith('/same/'):
            location, code = f'{cdn}/cdn/{self.path[6:]}', 307
        elif self.path.startswith('/cdn/'):
            location, code = f'{endpoint}/back/{self.path[5:]}', 303
        elif not self.path.startswith('/back/'):
            location, code = f'{endpoint}/same{self.path}', 302
        else:
            body = b'[]' if '/tree/' in self.path else b'x'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(code)
        self.send_header('Location', location)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, *args):
        pass


def local_redirects():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tree = FIXTURE['prepare'](t)
        data = t / 'data'
        data.mkdir()
        (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                               (tree / 'cluster.env.example').read_text()))
        (t / 'bin').mkdir()
        hf = t / 'bin/hf'
        install_hf(hf, FIXTURE['STUB_HF'], FIXTURE['PINS']['HF_HUB_VERSION'])
        hf.chmod(0o755)
        env = dict(os.environ, PATH=f'{t / "bin"}:{os.environ["PATH"]}', HF_TOKEN=TOKEN,
                   HF_ENDPOINT=endpoint, HF_ARGS=str(t / 'hf-args'), CLUSTER_ENV=str(tree / 'cluster.env'),
                   NO_PROXY='127.0.0.1', no_proxy='127.0.0.1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('LEAK', None)
        p = subprocess.run([str(tree / 'scripts/fetch-weights.sh'), '--weights', 'ablit'],
                           env=env, capture_output=True, text=True, timeout=60)
        check(p.returncode == 0 and (data / 'ablit/source.verified').exists(),
              'gated download succeeds through same-origin, CDN and return redirects')
        source = FIXTURE['PINS']['ABLIT_SOURCE_REPO']
        for kind, part in (('access probe', '/resolve/'), ('size metadata', '/tree/')):
            mine = [(port, path, auth) for port, path, auth in requests if source in path and part in path]
            check(len(mine) == 4 and all(auth == f'Bearer {TOKEN}' for _, _, auth in mine[:2]),
                  f'{kind}: explicitly selected endpoint and its same-origin hop receive the token')
            check(len(mine) == 4 and all(auth is None for _, _, auth in mine[2:]),
                  f'{kind}: CDN hop has no token, and a return redirect does not restore it')
        anonymous = [auth for _, path, auth in requests if source not in path]
        check(bool(anonymous) and all(auth is None for auth in anonymous),
              'base and drafter metadata stay anonymous across every redirect')
        logs = ''.join(f.read_text() for f in (tree / 'logs').glob('*.log'))
        check(TOKEN not in p.stdout + p.stderr + logs, 'token never reaches console or logs')


def origin_rules():
    redirect = runpy.run_path(str(HERE / 'scripts/ablit/fetch-source.py'))['SafeRedirect']
    for origin, target, keep in (
        ('https://huggingface.co', 'https://HUGGINGFACE.CO:443/file', True),
        ('https://huggingface.co', 'https://huggingface.co:444/file', False),
        ('https://huggingface.co', 'https://huggingface.co:0/file', False),
        ('https://huggingface.co', 'https://cdn.huggingface.co/file', False),
        ('https://mirror.example.invalid:9443', 'https://mirror.example.invalid:9443/file', True),
        ('https://mirror.example.invalid:9443', 'https://huggingface.co/file', False),
        ('http://127.0.0.1', 'http://127.0.0.1:80/file', True),
        ('http://127.0.0.1', 'https://127.0.0.1/file', False),
    ):
        try:
            handler = redirect(origin)
            req = urllib.request.Request(origin + '/start', headers={'Authorization': f'Bearer {TOKEN}'})
            moved = handler.redirect_request(req, None, 302, 'Found', {}, target)
            ok = bool(moved.get_header('Authorization')) == keep
        except TypeError:  # The old helper has no configurable origin.
            ok = False
        check(ok, f'origin (scheme, host, effective port): {origin} -> {target}, token={keep}')
    req = urllib.request.Request('https://huggingface.co/start', headers={'Authorization': f'Bearer {TOKEN}'})
    refused = False
    try:
        redirect().redirect_request(req, None, 302, 'Found', {}, 'http://huggingface.co/file')
    except urllib.error.HTTPError:
        refused = True
    check(refused, 'HTTPS-to-HTTP redirect is refused before the next request')
    source = (HERE / 'scripts/fetch-weights.sh').read_text()
    endpoint_code = '\n'.join(line for line in source.splitlines() if line.startswith('ENDPOINT='))
    for value, expected in ((None, 'https://huggingface.co'), ('', 'https://huggingface.co'),
                            ('http://127.0.0.1:12345/', 'http://127.0.0.1:12345')):
        env = dict(os.environ)
        env.pop('HF_ENDPOINT', None)
        if value is not None:
            env['HF_ENDPOINT'] = value
        p = subprocess.run(['bash', '-c', endpoint_code + '\nprintf "%s" "$ENDPOINT"'],
                           env=env, capture_output=True, text=True, check=True)
        check(p.stdout == expected, f'HF_ENDPOINT={value!r}: endpoint is {expected}')


def main():
    global endpoint, cdn
    servers = [ThreadingHTTPServer(('127.0.0.1', 0), Hub) for _ in range(2)]
    endpoint, cdn = [f'http://127.0.0.1:{s.server_port}' for s in servers]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        local_redirects()
        origin_rules()
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
    print(f'check-fetch-redirects: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
