#!/usr/bin/env bash
# Check the upstream model revisions and NGC image in a release's pins.env.
# Anonymous, read-only API/HEAD requests only; no models, layers or wheels downloaded.
# Usage: scripts/check-sources.sh [PATH/TO/pins.env]
# Defaults to pins.env beside scripts/. On main, pass the v2.0.1 checkout's pins.env.
# Exit 0: all expected responses match; 1: a source changed or could not be checked;
# 2: invalid arguments or pins. Needs only bash and Python 3.10+; no HF/Docker login.
set -euo pipefail
case ${1:-} in
  -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
if [[ $# -gt 1 || ${1:-} == -* ]]; then
  echo 'usage: check-sources.sh [PATH/TO/pins.env]' >&2
  exit 2
fi
exec python3 -I -S - "${1:-$(dirname "$0")/../pins.env}" <<'PY'
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import build_opener, HTTPRedirectHandler, Request


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# No HF client, saved tokens, Docker config or netrc. Redirects are changes, not
# permission to forward the anonymous registry bearer token to another origin.
opener = build_opener(NoRedirect)
LIMIT = 1024 * 1024


class SourceChanged(ValueError):
    """A diagnostic made locally, safe to print without upstream response bodies."""


def request(url, method='GET', headers=None):
    req = Request(url, method=method, headers=headers or {})
    try:
        response = opener.open(req, timeout=20)
    except HTTPError as error:
        response = error
    with response:
        body = response.read(LIMIT + 1) if method == 'GET' and response.status == 200 else b''
        if len(body) > LIMIT:
            raise SourceChanged('API response exceeds 1 MiB')
        return response.status, response.headers, body


def get_json(url, headers=None, digest=None):
    status, _, body = request(url, headers=headers)
    if status != 200:
        raise SourceChanged(f'API returned HTTP {status}, expected 200')
    if digest and 'sha256:' + hashlib.sha256(body).hexdigest() != digest:
        raise SourceChanged('registry manifest does not match its pinned digest')
    return json.loads(body)


def read_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or key in pins:
            raise ValueError('malformed or duplicate pin')
        pins[key] = value
    for prefix in ('BASE', 'DRAFTER', 'ABLIT_SOURCE'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', pins[prefix + '_REPO']):
            raise ValueError(f'invalid {prefix}_REPO')
        if not re.fullmatch(r'[0-9a-f]{40}', pins[prefix + '_REV']):
            raise ValueError(f'invalid {prefix}_REV')
    if not re.fullmatch(r'nvcr\.io/[A-Za-z0-9_./-]+@sha256:[0-9a-f]{64}', pins['IMAGE']):
        raise ValueError('IMAGE must pin an NGC manifest digest')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', pins['IMAGE_ID']):
        raise ValueError('invalid IMAGE_ID')
    return pins


def check_model(pins, prefix, gated=False):
    repo, rev = pins[prefix + '_REPO'], pins[prefix + '_REV']
    metadata = get_json(f'https://huggingface.co/api/models/{repo}/revision/{rev}')
    if metadata.get('sha') != rev:
        raise SourceChanged('API revision differs from the pin')
    if metadata.get('private') is not False:
        raise SourceChanged('repository is no longer public')
    files = sorted(row['rfilename'] for row in metadata['siblings']
                   if row['rfilename'].endswith('.safetensors'))
    if not files:
        raise SourceChanged('no weight file in the pinned revision')
    if gated:
        if metadata.get('gated') not in ('auto', 'manual', True):
            raise SourceChanged('ablit source is no longer gated')
        status, headers, _ = request(
            f'https://huggingface.co/{repo}/resolve/{rev}/{quote(files[0], safe="/")}', method='HEAD')
        if status != 401 or headers.get('X-Error-Code') != 'GatedRepo':
            raise SourceChanged(f'ablit weight HEAD returned HTTP {status}; expected 401 / GatedRepo')
        return 'pinned revision resolves; anonymous weight HEAD is 401 / GatedRepo (expected)'
    if metadata.get('gated') is not False:
        raise SourceChanged('public source now requires gated access')
    return 'public pinned revision resolves'


def check_image(pins):
    repo, digest = pins['IMAGE'].removeprefix('nvcr.io/').split('@')
    # NGC issues this pull-only token to anonymous clients; no credentials sent.
    auth = get_json('https://nvcr.io/proxy_auth?' + urlencode({
        'service': 'nvcr.io', 'scope': f'repository:{repo}:pull'}))
    token = auth.get('token') or auth.get('access_token')
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9._~+/=-]+', token):
        raise SourceChanged('NGC did not issue an anonymous pull token')
    headers = {'Authorization': 'Bearer ' + token, 'Accept': ', '.join((
        'application/vnd.oci.image.index.v1+json',
        'application/vnd.docker.distribution.manifest.list.v2+json',
        'application/vnd.oci.image.manifest.v1+json',
        'application/vnd.docker.distribution.manifest.v2+json'))}
    url = f'https://nvcr.io/v2/{repo}/manifests/'
    index = get_json(url + digest, headers, digest)
    arm = [m['digest'] for m in index['manifests']
           if m.get('platform', {}).get('os') == 'linux'
           and m.get('platform', {}).get('architecture') == 'arm64']
    if len(arm) != 1 or not re.fullmatch(r'sha256:[0-9a-f]{64}', arm[0]):
        raise SourceChanged('index must contain one Linux arm64 manifest')
    manifest = get_json(url + arm[0], headers, arm[0])
    if manifest['config']['digest'] != pins['IMAGE_ID']:
        raise SourceChanged('arm64 image config differs from IMAGE_ID')
    return 'index digest, arm64 manifest and pinned IMAGE_ID resolve'


def main():
    try:
        pins = read_pins(sys.argv[1])
    except (OSError, ValueError, KeyError):
        print('FAIL pins: supply a valid release pins.env (see --help)', file=sys.stderr)
        return 2
    failed = 0
    for name, check in (
        ('base', lambda: check_model(pins, 'BASE')),
        ('drafter', lambda: check_model(pins, 'DRAFTER')),
        ('ablit', lambda: check_model(pins, 'ABLIT_SOURCE', gated=True)),
        ('NGC image', lambda: check_image(pins)),
    ):
        try:
            print(f'PASS {name}: {check()}', flush=True)
        except SourceChanged as error:
            failed += 1
            print(f'FAIL {name}: {error}', flush=True)
        except (URLError, OSError, HTTPException):
            failed += 1
            print(f'FAIL {name}: network request failed or timed out', flush=True)
        except (ValueError, KeyError, TypeError, AttributeError):
            # Never echo response bodies or credentials, including registry tokens.
            failed += 1
            print(f'FAIL {name}: unexpected status, metadata or digest; source may have changed', flush=True)
    print(f'check-sources: {4 - failed} passed, {failed} failed')
    return int(failed != 0)


if __name__ == '__main__':
    sys.exit(main())
PY
