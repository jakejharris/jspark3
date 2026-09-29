#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit prepared (component-pending) v1.8.4 source for tests of pre-integration behavior.

The release tree is integrated and bound, so tests of the seal integrator and of the
pending-seal refusals cannot start from the live tree. prepare() copies the live source
and restores the prepared component/binding state byte-exact from
test_prepared_v184_source.json: the state files, the installer's source-manifest pin
line and the v1.8.4 catalog entry. It removes the two files the integrator adds, with
their inventory rows, and rehashes the inventories. Every other file is the live tree.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tools/test_prepared_v184_source.json'
COOP = 'recipe/overlays/v16/coop/'
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'recipe/scripts')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture():
    return json.loads(FIXTURE.read_text())


def legacy_build_record():
    """The pre-integration (legacy, schema 1) cooperative BUILD record."""
    return json.loads(fixture()['files'][COOP + 'BUILD.json'])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def refresh(root):
    """Rehash the source inventories of a test tree after explicit fixture edits."""
    from validate_release import inventory, sums
    import _shared_output_audit as audit
    files, errors = inventory(root)
    assert not errors, errors
    review = json.loads((root / 'manifests/shared-output-audit.json').read_text())
    for name, path in audit.candidates(root).items():
        review['files'][name]['sha256'] = sha(path)
    write(root / 'manifests/shared-output-audit.json', review)
    sbom = json.loads((root / 'manifests/sbom.cdx.json').read_text())
    for row in sbom['components']:
        if row['type'] == 'file':
            row['hashes'][0]['content'] = sha(root / row['name'])
    write(root / 'manifests/sbom.cdx.json', sbom)
    recipe = root / 'recipe'
    (recipe / 'SHA256SUMS').write_text(sums(recipe, [p for p in files if p.is_relative_to(recipe)]))
    derivation = json.loads((root / 'manifests/derivation.json').read_text())
    for name, row in derivation['files'].items():
        row['public_sha256'] = sha(root / name)
    write(root / 'manifests/derivation.json', derivation)
    (root / 'SHA256SUMS').write_text(sums(root, files))


def prepare(dest):
    """Build the prepared v1.8.4 source at the new directory dest; return dest."""
    dest = Path(dest)
    data = fixture()
    # Copy the shipped inventory only, so untracked working files never enter the fixture.
    names = [line.split('  ', 1)[1] for line in (ROOT / 'SHA256SUMS').read_text().splitlines()]
    for name in [*names, 'SHA256SUMS']:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name, text in data['files'].items():
        (dest / name).write_bytes(text.encode())
    for name, entries in data['json_entries'].items():
        value = json.loads((dest / name).read_text())
        value.update(entries)
        write(dest / name, value)
    pin = data['installer_pin']
    installer = dest / pin['path']
    text, count = re.subn(r'^SOURCE_MANIFEST_SHA256 = "[0-9a-f]{64}"$', pin['line'], installer.read_text(), flags=re.M)
    assert count == 1, 'installer source pin not found'
    installer.write_text(text)
    for name, digest in data['sha256'].items():
        assert sha(dest / name) == digest, 'prepared fixture differs: ' + name
    absent = set(data['absent'])
    for name in absent:
        (dest / name).unlink()
    licenses = json.loads((dest / 'manifests/license-review.json').read_text())
    licenses['files'] = {k: v for k, v in licenses['files'].items() if k not in absent}
    write(dest / 'manifests/license-review.json', licenses)
    blocks = (dest / 'REUSE.toml').read_text().split('\n[[annotations]]\n')
    kept = [blocks[0]]
    for block in blocks[1:]:
        lines = block.splitlines()
        paths = [p for p in json.loads(lines[0].split(' = ', 1)[1]) if p not in absent]
        if paths:
            kept.append('\n'.join(['path = ' + json.dumps(paths), *lines[1:]]) + ('\n' if block.endswith('\n') else ''))
    (dest / 'REUSE.toml').write_text('\n[[annotations]]\n'.join(kept))
    sbom = json.loads((dest / 'manifests/sbom.cdx.json').read_text())
    sbom['components'] = [c for c in sbom['components'] if c.get('name') not in absent]
    write(dest / 'manifests/sbom.cdx.json', sbom)
    derivation = json.loads((dest / 'manifests/derivation.json').read_text())
    derivation['files'] = {k: v for k, v in derivation['files'].items() if k not in absent}
    write(dest / 'manifests/derivation.json', derivation)
    refresh(dest)
    return dest


class PreparedFixtureTests(unittest.TestCase):
    def test_fixture_is_the_authentic_prepared_source(self):
        import _coop_qualification as q
        data = fixture()
        index = q.read(ROOT / (COOP + 'QUALIFICATION.json'))
        # The integrator retains the original source manifest; the seed policy is sealed by hash.
        self.assertEqual(data['files'][COOP + 'SOURCE_MANIFEST.json'].encode(),
                         (ROOT / (COOP + 'QUALIFICATION_SOURCE_MANIFEST.json')).read_bytes())
        self.assertEqual(hashlib.sha256(data['files'][COOP + 'source/dispatch_policy.json'].encode()).hexdigest(),
                         index['raw_bundle']['dispatch_policy_sha256'])
        self.assertEqual(q.digest_value(legacy_build_record()), q.LEGACY_RECORD_SHA256)
        self.assertEqual(json.loads(data['files']['recipe/config/coop-release.json'])['status'],
                         'pending-component-qualification')

    def test_prepared_source_validates_and_refuses_final(self):
        with tempfile.TemporaryDirectory() as directory:
            source = prepare(Path(directory) / 'source')
            for name in fixture()['absent']:
                self.assertFalse((source / name).exists())
            run = lambda *args: subprocess.run([sys.executable, '-B', str(source / 'tools/validate_release.py'), str(source), *args],
                                               capture_output=True, text=True, timeout=300)
            proc = run()
            self.assertEqual(proc.returncode, 0, proc.stdout)
            proc = run('--require-final')
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('FAIL release-manifest', proc.stdout)
            self.assertEqual(json.loads((source / 'manifests/final-binding.json').read_text())['state'], 'prepared')


if __name__ == '__main__':
    unittest.main()
