#!/usr/bin/env python3
"""Validate a recipe-only source export. No deployment or publication is performed."""
from __future__ import annotations

import argparse
import ast
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
from _release_checks import Report, sha256, pth_payload_allowed, check_syntax, check_links, check_dry_runs

# Split policy literals so the scanner also checks its own source.
NAMES = [b"For" b"ge", b"E" b"li", b"Ar" b"gus", b"As" b"tra", b"Co" b"dex",
         b"Clau" b"de", b"Op" b"us", b"Fa" b"ble", b"S" b"ol", b"Lu" b"na", b"G" b"PT"]
PRIVATE_DIR = b".co" b"dex-tasks"
PATTERNS = [
    ("private identity", rb"(?i)(?<![a-z0-9])(?:" + b"|".join(NAMES) + rb")(?![a-z0-9])"),
    ("attribution trailer", rb"(?i)co-authored" rb"-by\s*:"),
    ("private path", rb"(?i)/ho" rb"me/[^/\s]+|/Us" rb"ers/[^/\s]+|/mn" rb"t/[a-z]/|[A-Z]:\\Us" rb"ers\\|~/assis" rb"tants/|/tmp/(?:tm" rb"ux|pi)-|" + re.escape(PRIVATE_DIR)),
    ("private run label", rb"(?i)(?<![a-z0-9])boot[s]?[ _-]*\d+[a-z0-9_-]*|(?<![a-z0-9])spark[123](?![a-z0-9])|jspark3-perf" rb"-run"),
    ("worker label", rb"(?i)\bpane[ _:#%-]+[a-z0-9][a-z0-9_-]*|\blane[ _:#-]+(?:\d+|[a-z][0-9]+|(?-i:[A-Z]))\b|\blane[-_:]+[a-z]\b|(?<![a-z0-9])%\d{3,}\b|\btm" rb"ux\b"),
    ("credential", rb"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{16,}|(?<![a-z0-9])(?:sk[-_]|hf_|gh[pousr]_)[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("credential assignment", rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_+/=-]{12,}"),
    ("credential URL", rb"[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@"),
    ("service credential", rb"(?:ASIA[0-9A-Z]{16}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,})"),
    ("private donor repository", rb"(?i)drow" rb"zeys/"),
    ("private mailbox", rb"(?i)jjh" rb"digital"),
]
QUAD = re.compile(rb"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
IPV6 = re.compile(rb"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*(?:%[\w]+)?(?![\w:])")
ALLOWED_V4 = [ipaddress.ip_network(x) for x in ("127.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
REQUIRED = ['README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'REQUIRED_ATTRIBUTION.md',
            'SHA256SUMS', 'manifests/release.json', 'manifests/derivation.json', 'manifests/sbom.cdx.json',
            'manifests/license-review.json', 'manifests/binaries.json', 'REUSE.toml', 'manifests/dependencies.json', 'docs/LICENSING.md', 'recipe/scripts/production_stock.py', 'manifests/toggles.json',
            'recipe/SHA256SUMS', 'recipe/.env.example', 'recipe/scripts/verify_stock.py',
            'recipe/scripts/fleetctl.py', 'recipe/scripts/remote_preflight.py',
            'recipe/config/patch-contract.json', 'docs/INSTALL.md', 'docs/OPERATIONS.md',
            'docs/LIMITATIONS.md', 'docs/REPRODUCIBILITY.md', 'tools/validate_release.py']


def git_admin(root):
    admin = root / '.git'
    if not admin.exists() or admin.is_symlink():
        return False
    process = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True, timeout=10)
    return process.returncode == 0 and Path(process.stdout.strip()).resolve() == root.resolve()


def inventory(root):
    files, errors = [], []
    has_git = git_admin(root)
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            p = Path(parent, name)
            if p == root / '.git' and has_git:
                dirs.remove(name)
                continue
            if p.is_symlink():
                errors.append('symlink directory: ' + str(p.relative_to(root)))
                dirs.remove(name)
            if name in {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'dist'} or name.startswith('.co' + 'dex'):
                errors.append('forbidden directory: ' + str(p.relative_to(root)))
                dirs.remove(name)
        for name in names:
            p = Path(parent, name)
            if p == root / '.git' and has_git:
                continue
            rel = p.relative_to(root).as_posix()
            if not stat.S_ISREG(p.lstat().st_mode) or p.is_symlink():
                errors.append('not a regular file: ' + rel)
                continue
            if p.suffix in {'.so', '.pyc', '.pyo', '.log', '.safetensors', '.gguf', '.bin', '.pt', '.ckpt', '.npy', '.npz', '.pcap', '.pcapng', '.key', '.pem', '.p12', '.pfx'} or name in {'.git', '.env', 'id_rsa', 'id_ed25519'}:
                errors.append('forbidden payload: ' + rel)
            data = p.read_bytes()
            if (p.suffix == '.pth' and not pth_payload_allowed(p)) or (b'\0' in data or data.startswith(b'\x7fELF')):
                errors.append('unrecognized binary payload: ' + rel)
            if p.stat().st_size > 12 * 1024 * 1024:
                errors.append('oversize file: ' + rel)
            files.append(p)
    if sum(p.stat().st_size for p in files) > 64 * 1024 * 1024:
        errors.append('oversize tree')
    return sorted(files), errors


def leaks(data):
    errors = [label for label, pattern in PATTERNS if re.search(pattern, data)]
    for raw in QUAD.findall(data):
        try:
            addr = ipaddress.ip_address(raw.decode())
        except ValueError:
            continue
        if str(addr) != '0.0.0.0' and not any(addr in network for network in ALLOWED_V4):
            errors.append('non-documentation IPv4')
            break
    for raw in IPV6.findall(data):
        try:
            addr = ipaddress.ip_address(raw.decode())
        except ValueError:
            continue
        if addr not in (ipaddress.ip_address('::'), ipaddress.ip_address('::1')) and addr not in ipaddress.ip_network('2001:db8::/32'):
            errors.append('non-documentation IPv6')
            break
    return sorted(set(errors))


def commit_privacy(root):
    if not git_admin(root):
        return []
    process = subprocess.run(['git', '-C', str(root), 'log', '--all', '--format=%B'],
                             capture_output=True, timeout=30)
    if process.returncode:
        return ['commit messages could not be inspected']
    return ['commit message: ' + label for label in leaks(process.stdout)]


def sums(root, files):
    return ''.join(f'{sha256(p)}  {p.relative_to(root).as_posix()}\n' for p in sorted(files) if p != root / 'SHA256SUMS')


def load(root, name):
    return json.loads((root / name).read_text())


def verify(root, require_final=False):
    report = Report()
    files, errors = inventory(root)
    def check(name, action):
        try:
            problems = action() or []
            if problems: report.fail(name, '; '.join(problems[:15]))
            else: report.ok(name)
        except Exception as exc:
            # Do not echo external data or secret-bearing matches.
            report.fail(name, type(exc).__name__)
    check('inventory', lambda: errors)
    check('required-files', lambda: [n for n in REQUIRED if not (root / n).is_file()])
    check('privacy-scan', lambda: [p.relative_to(root).as_posix() + ': ' + ', '.join(found)
          for p in files if (found := leaks(p.relative_to(root).as_posix().encode() + b'\n' + p.read_bytes()))] + commit_privacy(root))
    check('owner-identity', lambda: [] if load(root, 'manifests/release.json')['repository'] == 'https://github.com/jakejharris/jspark3'
          and not any(re.search(rb'(?:github.com/|ghcr.io/|pkg:github/)jakejh(?:/|$)', p.read_bytes()) for p in files) else ['repository identity mismatch'])
    check_syntax(root, files, report)
    check_links(root, files, report)

    def derivation():
        d = load(root, 'manifests/derivation.json')
        rows = d['files']
        expected = {p.relative_to(root).as_posix() for p in files} - {'SHA256SUMS', 'recipe/SHA256SUMS', 'manifests/derivation.json'}
        issues = []
        if set(rows) != expected: issues.append('derivation inventory mismatch')
        for rel, row in rows.items():
            if sha256(root / rel) != row['public_sha256']: issues.append('public hash drift: ' + rel)
            if not re.fullmatch('[0-9a-f]{64}', row['input_sha256']): issues.append('missing input hash: ' + rel)
        if not d.get('source_inventory_sha256') or d.get('scope') != 'recipe-only': issues.append('source identity missing')
        return issues
    check('derivation', derivation)

    def contracts():
        recipe = root / 'recipe'
        code = '''import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import _contracts as c
import apply_base_pipeline as p
import apply_v14_core as core
import apply_display_kv as display
import apply_exl3_fatpath as fat
import apply_coop_moe as coop
import apply_dense_fp8 as dense
import apply_adaptive_k as adaptive
import apply_warmjit as warmjit
import apply_grammar_fsm as grammar
from _atomic import canonical
r=Path(sys.argv[1]).parent
v=json.loads((r/'config/patch-contract.json').read_text())
assert v['transforms']==c.SECTIONS
states,_=p.snapshots(v)
core.verify_sources(); fat.verify_sources()
sys.path.insert(0,str(r.parent/"tools"))
from _source_contracts import verify_sources
verify_sources(r)
dense.verify_sources(); warmjit.verify_sources(); grammar.verify_sources()
adaptive.verify_sources(c.V16_ADAPTIVE_K)
import hashlib,re
pin=re.findall(r'"transform_target_set_sha256": "([0-9a-f]{64})"',(r/'scripts/_fleetctl.py').read_text())
assert pin==[hashlib.sha256(canonical(states[-1])).hexdigest()]
cad=json.loads((r/'config/cadence-contract.json').read_text())
for n,h in cad['modules'].items(): assert hashlib.sha256((r/'modules'/n).read_bytes()).hexdigest()==h
for n,h in json.loads((r/'config/swa-contract.json').read_text())['patches'].items():
 assert hashlib.sha256((r/'overlays'/n).read_bytes()).hexdigest()==h
'''
        p = subprocess.run([sys.executable, '-B', '-S', '-c', code, str(recipe / 'scripts')], capture_output=True, timeout=30)
        return [] if p.returncode == 0 else ['source pins or transform contracts disagree']
    check('identity-contracts', contracts)
    check('license-copies', lambda: [n for n in ('LICENSE', 'REQUIRED_ATTRIBUTION.md', 'THIRD_PARTY_NOTICES.md') if (root / n).read_bytes() != (root / 'recipe' / n).read_bytes()])
    def license_inventory():
        review = load(root, 'manifests/license-review.json')
        assert review['status'] == 'approved-per-file-source-posture'
        assert set(review['components']) >= {'mia-patches', 'instanttensor', 'flycockpit', 'display-kv', 'abliteration', 'cooperative-kernel'}
        annotations = {}
        for block in (root / 'REUSE.toml').read_text().split('[[annotations]]')[1:]:
            fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
            for path in json.loads(fields['path']):
                assert path not in annotations
                annotations[path] = json.loads(fields['SPDX-License-Identifier'])
        for name, license in review['files'].items():
            data = (root / name).read_text()
            if license == 'LicenseRef-Retained-Notice': continue
            if '/coop/' in name or not re.search(r'(?m)^(?:#|//) SPDX-License-Identifier:', data):
                assert annotations[name] == license
            else:
                assert 'SPDX-License-Identifier: ' + license in data
        assert review['files']['recipe/overlays/patch_dflash_fine_replay_window.py'] == 'AGPL-3.0-only'
        for name in ('ExLlamaV3-MIT.txt', 'display-kv-AGPL-3.0.txt'):
            assert (root / 'third_party/licenses' / name).is_file()
    check('license-review-inventory', license_inventory)

    def binaries():
        b = load(root, 'manifests/binaries.json')
        actual = {p.relative_to(root).as_posix() for p in files if p.read_bytes().startswith(b'\x7fELF')}
        issues = []
        if actual: issues.append('native binary forbidden in source export')
        for rel, row in b.items():
            if (root / rel).exists() or row['distributed'] is not False or not re.fullmatch('[0-9a-f]{64}', row['expected_sha256']): issues.append('binary exclusion or expected pin drift: ' + rel)
            for source in [*row['sources'], row['build_script']]:
                if not (root / source).is_file(): issues.append('missing binary source/build step')
        return issues
    check('binary-provenance', binaries)

    def release():
        r = load(root, 'manifests/release.json')
        from _qualification import verify as qualification
        qualification(root, require_final)
        return [] if (r['schema_version'] == 1 and r['scope'] == 'recipe-only' and r['publication_authorized'] is False
                      and r['hardware_validation'] == 'not-performed' and r['license_clearance'] == 'approved-per-file-source-posture' and re.fullmatch(r'v\d+\.\d+\.\d+', r['base_version'])) else ['release identity or status invalid']
    check('release-manifest', release)

    def toggles():
        env = dict(line.split('=', 1) for line in (root / 'recipe/.env.example').read_text().splitlines() if line and not line.startswith('#') and '=' in line)
        t = load(root, 'manifests/toggles.json')
        final = load(root, 'manifests/final-binding.json')['state'] == 'bound'
        return [name + ': configuration drift' for name, row in t.items()
                if row['example_default'] != env.get(name) or row['sealed_base_value'] not in row['modes']
                or row['release_ship_value'] != (env.get(name) if final else None)]
    check('toggle-defaults', toggles)

    def replay():
        r = load(root, 'manifests/composition.json')
        return [] if r['verdict'] == 'PASS' and r['original_targets_verified'] and r['public_targets_verified'] and r['public_contracts_verified'] and r['executable_ast_equal'] else ['sanitized transform composition unproved']
    check('offline-composition', replay)

    def sbom():
        s = load(root, 'manifests/sbom.cdx.json')
        actual = {p.relative_to(root).as_posix(): sha256(p) for p in files if p.relative_to(root).parts[0] == 'recipe' and p.name != 'SHA256SUMS'}
        recorded = {c['name']: c['hashes'][0]['content'] for c in s['components'] if c['type'] == 'file'}
        assert [c for c in s['components'] if c['type'] != 'file'] == load(root, 'manifests/dependencies.json')['components']
        return [] if s['bomFormat'] == 'CycloneDX' and s['specVersion'] == '1.5' and recorded == actual else ['file SBOM drift']
    check('sbom', sbom)
    check('sha256sums', lambda: [str(folder.relative_to(root)) + ': checksum inventory drift' for folder, selected in
          [(root, files), (root / 'recipe', [p for p in files if p.is_relative_to(root / 'recipe')])]
          if (folder / 'SHA256SUMS').read_text() != sums(folder, selected)])
    check_dry_runs(root, report)
    return {'checks': report.checks, 'failed': report.failed, 'verdict': 'PASS' if not report.failed else 'FAIL',
            'scope': 'offline source export with approved per-file license posture; hardware admission remains separate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('.'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-final', action='store_true', help='refuse an unselected, prepared or mismatched final recipe')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.report and args.report.resolve().is_relative_to(root):
        parser.error('write the validation report outside the hashed export')
    result = verify(root, require_final=args.require_final)
    if args.report: args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(f"VERDICT {result['verdict']} ({len(result['checks'])} checks, {result['failed']} failed)")
    return bool(result['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
