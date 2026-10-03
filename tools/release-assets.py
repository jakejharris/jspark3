#!/usr/bin/env python3
"""Build the release assets from a commit of this repository, reproducibly.

    python3 tools/release-assets.py --out DIR [--commit REF] [--final]

Writes, into DIR (which must be outside this tree):
  jspark3-recipe-v<version>.tar.gz   `git archive` of the commit under jspark3-v<version>/, gzip with no name or time
  jspark3.sbom.cdx.json              CycloneDX 1.5 SBOM: the recipe, the engine and its wheel digest, the container
                                     image by digest, every wheel in wheels.lock, the weights, draft model and
                                     refusal-removed source at their pinned revisions, the chat template and the
                                     fabric launcher, each with its license
  PROMPT-MIX.jsonl                   the concurrency benchmark's prompt texts, as release/PROMPT-MIX.jsonl in the commit
  SHA256SUMS                         sha256 of the files above
Every byte derives from the commit (timestamps are the commit time), so two runs give identical files. A file the
commit does not have yet is reported as an empty slot; --final refuses to build with an empty slot.
Standard library and git only.
"""
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def git(*args, binary=False):
    out = subprocess.run(['git', '-C', str(HERE), *args], capture_output=True, check=True).stdout
    return out if binary else out.decode().strip()


def show(commit, path):
    return git('show', f'{commit}:{path}')


def env_file(text):
    values = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def lic(expression):
    if expression.startswith('LicenseRef-'):
        return [{'license': {'name': expression}}]
    if any(op in expression for op in (' AND ', ' OR ', ' WITH ')):
        return [{'expression': expression}]
    return [{'license': {'id': expression}}]


def tarball(commit, version):
    raw = git('archive', '--format=tar', f'--prefix=jspark3-{version}/', commit, binary=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def sbom(commit, version, archive_name, archive_sha):
    pins = env_file(show(commit, 'pins.env'))
    when = datetime.fromtimestamp(int(git('show', '-s', '--format=%ct', commit)), timezone.utc)
    licenses = json.loads(show(commit, 'tools/wheel-licenses.json'))['wheels']
    manifests = {p: sha256(git('show', f'{commit}:{p}', binary=True)) for p in (
        'manifests/inputs/base-weights.sha256', 'manifests/inputs/drafter.sha256')}
    components = []

    def add(ref, kind, name, version_, license_, **extra):
        item = {'type': kind, 'bom-ref': ref, 'name': name, 'version': version_, 'licenses': lic(license_)}
        item.update(extra)
        components.append(item)

    add('engine', 'library', 'tensorfold (JSpark3 fork)', '0.3.6.2', 'MIT AND Apache-2.0',
        description='Fork of TensorFold 0.3.6.2 (MIT) vendored in engine/; MIT, with third-party code under MIT and '
                    'Apache-2.0 and two Apache-2.0 lines ported from upstream TensorFold (engine/THIRD_PARTY_NOTICES.md, '
                    'engine/NOTICE)',
        properties=[{'name': 'jspark3:engine-commit', 'value': pins['ENGINE_COMMIT']},
                    {'name': 'jspark3:wheel', 'value': pins['ENGINE_WHEEL']},
                    {'name': 'jspark3:wheel-content-sha256', 'value': pins['WHEEL_CONTENT_SHA256']}],
        externalReferences=[{'type': 'vcs', 'url': 'https://github.com/ashhart/TensorFold', 'comment': 'upstream'}])
    image, digest = pins['IMAGE'].split('@', 1)
    add('image', 'container', image, digest, 'LicenseRef-NVIDIA-Container-License', author='NVIDIA',
        hashes=[{'alg': 'SHA-256', 'content': digest.split(':', 1)[1]}],
        purl=f'pkg:docker/{image.split("/", 1)[1]}@{digest}?repository_url={image.split("/", 1)[0]}',
        properties=[{'name': 'jspark3:image-id', 'value': pins['IMAGE_ID']},
                    {'name': 'jspark3:distribution', 'value': 'pulled by digest at install; not redistributed'}])
    for line in show(commit, 'wheels.lock').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        wheel, digest = line.split()
        meta = licenses[wheel]
        add(f'wheel:{wheel}', 'library', meta['name'], meta['version'], meta['license'], author=meta['authors'],
            hashes=[{'alg': 'SHA-256', 'content': digest.split(':', 1)[1]}],
            purl=f'pkg:pypi/{meta["name"].lower()}@{meta["version"]}',
            properties=[{'name': 'jspark3:file', 'value': wheel},
                        {'name': 'jspark3:distribution', 'value': 'downloaded from PyPI at install; not redistributed'}])
    hf = 'https://huggingface.co/'
    add('weights:base', 'machine-learning-model', pins['BASE_REPO'], pins['BASE_REV'], 'MIT', author='Z.AI',
        externalReferences=[{'type': 'distribution', 'url': hf + pins['BASE_REPO'] + '/tree/' + pins['BASE_REV']}],
        properties=[{'name': 'jspark3:files-manifest-sha256', 'value': manifests['manifests/inputs/base-weights.sha256']},
                    {'name': 'jspark3:distribution', 'value': 'downloaded anonymously at install; not redistributed'}])
    add('weights:drafter', 'machine-learning-model', pins['DRAFTER_REPO'], pins['DRAFTER_REV'], 'CC-BY-NC-ND-4.0',
        author='Inco AI',
        externalReferences=[{'type': 'distribution', 'url': hf + pins['DRAFTER_REPO'] + '/tree/' + pins['DRAFTER_REV']}],
        properties=[{'name': 'jspark3:files-manifest-sha256', 'value': manifests['manifests/inputs/drafter.sha256']},
                    {'name': 'jspark3:distribution', 'value': 'downloaded unmodified at install; never redistributed; '
                                                              'optional (--drafter none)'}])
    if pins.get('ABLIT_SOURCE_REPO', 'PENDING') != 'PENDING':
        add('weights:ablit-source', 'machine-learning-model', pins['ABLIT_SOURCE_REPO'], pins['ABLIT_SOURCE_REV'],
            pins.get('ABLIT_SOURCE_LICENSE', 'MIT'), author='orcarouter',
            description='refusal-removed (abliterated) source weights, opt-in; MIT plus the use conditions on its model '
                        'card (config/ablit-notice.txt)',
            externalReferences=[{'type': 'distribution',
                                 'url': hf + pins['ABLIT_SOURCE_REPO'] + '/tree/' + pins['ABLIT_SOURCE_REV']}],
            properties=[{'name': 'jspark3:access', 'value': "gated: the user's own Hugging Face token"},
                        {'name': 'jspark3:distribution', 'value': 'converted on the user\'s machine; not redistributed'},
                        {'name': 'jspark3:converter-sha256', 'value': pins.get('ABLIT_CONVERTER_SHA256', 'PENDING')}])
    add('template', 'file', 'template/chat-template.jinja', pins['TEMPLATE_SHA256'][:12], 'MIT',
        hashes=[{'alg': 'SHA-256', 'content': pins['TEMPLATE_SHA256']}],
        description="the base weights' chat template (MIT, Copyright (c) 2026 Z.AI Co., Ltd) plus six recipe lines")
    launcher = git('show', f'{commit}:scripts/fabric/launch.py', binary=True)
    add('fabric', 'file', 'scripts/fabric/launch.py', sha256(launcher)[:12], 'Apache-2.0',
        hashes=[{'alg': 'SHA-256', 'content': sha256(launcher)}], description='the recipe\'s NCCL ring launcher')

    serial = uuid.UUID(hashlib.sha256(f'jspark3:{commit}'.encode()).hexdigest()[:32], version=5)
    doc = {
        'bomFormat': 'CycloneDX', 'specVersion': '1.5', 'serialNumber': f'urn:uuid:{serial}', 'version': 1,
        'metadata': {
            'timestamp': when.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'component': {'type': 'application', 'bom-ref': 'recipe', 'name': 'jspark3', 'version': version,
                          'licenses': lic('Apache-2.0'),
                          'hashes': [{'alg': 'SHA-256', 'content': archive_sha}],
                          'properties': [{'name': 'jspark3:commit', 'value': commit},
                                         {'name': 'jspark3:archive', 'value': archive_name}]},
        },
        'components': components,
        'dependencies': [{'ref': 'recipe', 'dependsOn': [c['bom-ref'] for c in components]}],
    }
    return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


# Files of the commit that are also published as separate assets.
SLOTS = ('release/PROMPT-MIX.jsonl',)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', default='HEAD')
    ap.add_argument('--final', action='store_true', help='refuse to build while a slot is empty')
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == HERE or HERE in out.parents:
        sys.exit('release-assets: --out must be outside this tree')
    commit = git('rev-parse', '--verify', a.commit + '^{commit}')
    version = env_file(show(commit, 'pins.env'))['RELEASE']
    out.mkdir(parents=True, exist_ok=True)
    archive_name = f'jspark3-recipe-{version}.tar.gz'
    archive = tarball(commit, version)
    files = {archive_name: archive, 'jspark3.sbom.cdx.json': sbom(commit, version, archive_name, sha256(archive))}
    tracked = set(git('ls-tree', '-r', '--name-only', commit).splitlines())
    empty = [path for path in SLOTS if path not in tracked]
    if empty and a.final:
        sys.exit(f'release-assets: empty slot(s) in {commit[:12]}: {", ".join(empty)}; nothing written')
    for path in SLOTS:
        if path in tracked:
            files[path.rsplit('/', 1)[-1]] = git('show', f'{commit}:{path}', binary=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    sums = ''.join(f'{sha256(data)}  {name}\n' for name, data in sorted(files.items()))
    (out / 'SHA256SUMS').write_text(sums)
    print(f'release-assets: {version} from commit {commit[:12]}:')
    print(sums, end='')
    for path in empty:
        print(f'release-assets: empty slot: {path} (not in this commit yet)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
