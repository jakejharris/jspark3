#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic component integration through the real source and runtime gates."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/v16'), str(ROOT / 'recipe/scripts')]
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import _coop_qualification as q
import _shared_output_audit as audit
import apply_coop_moe as installer
import build_native as native
import integrate_coop_seal as integration
from test_coop_qualification import RecordTests, host_identity
from test_operator_image import write_record
from validate_release import inventory, sums


def refresh_fixture(root):
    """Rehash test-only authorities after explicit synthetic edits, never a release."""
    files, errors = inventory(root)
    assert not errors, errors
    review = q.read(root / 'manifests/shared-output-audit.json')
    for name, path in audit.candidates(root).items():
        review['files'][name]['sha256'] = q.sha(path)
    integration.write(root / 'manifests/shared-output-audit.json', review)
    sbom = q.read(root / 'manifests/sbom.cdx.json')
    for row in sbom['components']:
        if row['type'] == 'file':
            row['hashes'][0]['content'] = q.sha(root / row['name'])
    integration.write(root / 'manifests/sbom.cdx.json', sbom)
    recipe = root / 'recipe'
    (recipe / 'SHA256SUMS').write_text(sums(recipe, [p for p in files if p.is_relative_to(recipe)]))
    derivation = q.read(root / 'manifests/derivation.json')
    for name, row in derivation['files'].items():
        row['public_sha256'] = q.sha(root / name)
    integration.write(root / 'manifests/derivation.json', derivation)
    (root / 'SHA256SUMS').write_text(sums(root, files))


class ServingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        files, _ = inventory(ROOT)
        for path in files:
            dest = self.source / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        fixture = RecordTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.native_bytes = fixture.native_bytes
        self.seal = fixture.coop
        self.record = fixture.record
        self.output = self.root / 'integrated'
        # Model the actual seal: original source/seed, newly selected bundle.
        original = ROOT / integration.COOP
        shutil.copyfile(original / 'source/dispatch_policy.json', self.seal / 'source/dispatch_policy.json')
        shutil.copyfile(original / 'SOURCE_MANIFEST.json', self.seal / 'SOURCE_MANIFEST.json')
        raw_source_hash = q.sha(self.seal / 'SOURCE_MANIFEST.json')
        raw_policy_hash = q.sha(self.seal / 'source/dispatch_policy.json')
        index = fixture.index
        index['source_manifest_sha256'] = raw_source_hash
        raw_manifest = index['raw_bundle_manifest']
        raw_manifest['files']['dispatch_policy.json'] = raw_policy_hash
        raw = index['raw_bundle']
        raw.update(manifest_sha256=q.digest_value(raw_manifest), dispatch_policy_sha256=raw_policy_hash)
        for row in index['gates'].values():
            if row['kind'] != 'policy':
                row['bundle'] = copy.deepcopy(raw)
        for build in (index['qualification_build'], index['independent_build']['native_receipt']):
            build['build_inputs'][integration.COOP + '/SOURCE_MANIFEST.json'] = raw_source_hash
            build['build_inputs'][integration.COOP + '/source/dispatch_policy.json'] = raw_policy_hash
            build['payload_sha256'] = q.digest_value({k: v for k, v in build.items() if k != 'payload_sha256'})
        integration.write(self.seal / 'QUALIFICATION.json', index)
        self.record.update(source_manifest_sha256=raw_source_hash, qualification_source_manifest_sha256=raw_source_hash,
                           gate_index_sha256=q.sha(self.seal / 'QUALIFICATION.json'))
        self.record['reproducibility']['evidence_sha256'] = q.digest_value(index['qualification_build'])
        integration.write(self.seal / 'BUILD.json', self.record)
        (self.seal / 'bundle/cooperative_moe.so').write_bytes(self.native_bytes)
        # The only substituted authority is the native target in this temporary
        # test tree. All source, seal, policy and admission verifiers run intact.
        module = self.source / 'recipe/scripts/_coop_qualification.py'
        real_target = q.read(ROOT / 'recipe/config/coop-release.json')['native_sha256']
        module.write_text(module.read_text().replace(real_target, q.TARGET_NATIVE))
        for name in ('recipe/config/coop-release.json', 'manifests/final-binding.json',
                     'manifests/final-catalog.json', 'release/results-v1.8.4.json'):
            path = self.source / name
            path.write_text(path.read_text().replace(real_target, q.TARGET_NATIVE))
        for obj, name, value in ((q, 'RECIPE', self.source / 'recipe'),
                                  (installer, 'SOURCE_MANIFEST_SHA256', raw_source_hash)):
            control = patch.object(obj, name, value)
            control.start()
            self.addCleanup(control.stop)
        refresh_fixture(self.source)

    def command(self, root, script, *args):
        return subprocess.run([sys.executable, '-B', str(root / script), *map(str, args)],
                              capture_output=True, text=True, timeout=90)

    def integrate(self):
        result = self.command(self.source, 'tools/integrate_coop_seal.py',
                              '--sealed-output', self.seal, '--output', self.output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def validate(self, *args):
        return self.command(self.output, 'tools/validate_release.py', self.output, *args)

    def test_integrate_validate_and_prepare_with_real_gates(self):
        before = {p.relative_to(self.seal): q.sha(p) for p in self.seal.rglob('*') if p.is_file()}
        frozen = ['release/results.json', 'release/results-v1.8.0.json', 'release/RELEASE-NUMBERS.md',
                  'release/results-v1.8.4.json', 'manifests/final-binding-v1.8.3.json', 'REQUIRED_ATTRIBUTION.md']
        self.integrate()
        self.assertEqual(before, {p.relative_to(self.seal): q.sha(p) for p in self.seal.rglob('*') if p.is_file()})
        for name in frozen:
            self.assertEqual(q.sha(self.output / name), q.sha(self.source / name), name)
        coop = self.output / integration.COOP
        record = q.read(coop / 'BUILD.json')
        self.assertEqual(record, {**self.record, 'source_manifest_sha256': q.sha(coop / 'SOURCE_MANIFEST.json')})
        self.assertEqual(q.sha(coop / 'QUALIFICATION_SOURCE_MANIFEST.json'), self.record['qualification_source_manifest_sha256'])
        self.assertNotEqual(record['source_manifest_sha256'], record['qualification_source_manifest_sha256'])
        self.assertFalse(list(self.output.rglob('*.so')))
        self.assertEqual(self.validate().returncode, 0)
        self.assertNotEqual(self.validate('--require-final').returncode, 0)
        # Produce test receipts bound to the integrated source. Only Docker image
        # inspection is stubbed; validation and all receipt/binary/seal checks run.
        image = copy.deepcopy(self.record['qualification_image'])
        image['source_recipe_sha256'] = q.sha(self.output / 'recipe/SHA256SUMS')
        write_record(self.root / 'image.json', image)
        built = self.root / 'built'
        outputs = {}
        for group in native.OUTPUTS.values():
            for name in group.values():
                path = built / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.native_bytes)
                outputs[name] = q.sha(path)
        image = q.read(self.root / 'image.json')
        with patch.object(native, 'ROOT', self.output):
            build = {'schema_version': 2, 'verification': 'fixed-native-build-v2',
                     'source_recipe_sha256': image['source_recipe_sha256'], 'image_receipt_sha256': image['payload_sha256'],
                     'build_inputs': native.build_inputs(), 'builder_host': host_identity('serving fixture'),
                     'reproducibility': {'runs': 2, 'comparison': 'bit-identical'}, 'binary_sha256': outputs,
                     'hardware_qualified': False}
        write_record(self.root / 'native.json', build)
        runtime = self.root / 'runtime'
        program = '''import sys
from pathlib import Path
from unittest.mock import patch
root = Path(sys.argv.pop(1))
sys.path[:0] = [str(root / 'tools'), str(root / 'recipe/scripts')]
import prepare_runtime
with patch('_image_identity.verify_local_image'):
    prepare_runtime.main()
'''
        process = subprocess.run([sys.executable, '-B', '-c', program, str(self.output), '--binary-root', str(built),
            '--image-receipt', str(self.root / 'image.json'), '--native-receipt', str(self.root / 'native.json'),
            '--output', str(runtime)], capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        receipt = q.read(runtime / 'runtime-build-receipt.json')
        self.assertFalse(receipt['hardware_qualified'])
        self.assertFalse(receipt['coop_hardware_seal_required'])
        self.assertEqual(receipt['component_qualification'], {'native_sha256': q.TARGET_NATIVE,
            'policy_sha256': record['bundle']['dispatch_policy_sha256'],
            'component_seal_sha256': q.digest_value(record), 'gate_index_sha256': record['gate_index_sha256']})
        self.assertIn('JSPARK3_V16_COOP=1\n', (runtime / 'recipe/.env.example').read_text())
        self.assertEqual(q.read(runtime / integration.COOP / 'BUILD.json'), record)
        preflight = self.command(runtime, 'recipe/scripts/remote_preflight.py', '--recipe-only',
                                 '--recipe-root', runtime / 'recipe')
        self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
        # Recipe-only preflight does not run this full per-rank contract/bundle
        # gate. Exercise the actual remote gate and controller expectation with
        # fresh imports from the prepared runtime; no gate or Docker stub here.
        program = '''import sys
from pathlib import Path
recipe = Path(sys.argv[1])
sys.path.insert(0, str(recipe / 'scripts'))
import _fleetctl as fleet
import remote_preflight as preflight
row = preflight.v16_artifacts(recipe, 'production-stock', '1', 'ema', 'trunk')
assert row['coop'] == 'on'
assert row == fleet.expected_v16_row(fleet.load_env(recipe / '.env.example'))
assert preflight.verify_manifest(recipe)
'''
        def full_contract_gate():
            return subprocess.run([sys.executable, '-B', '-c', program, str(runtime / 'recipe')],
                                  capture_output=True, text=True, timeout=30)
        preflight = full_contract_gate()
        self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
        contract_path = runtime / integration.COOP / 'INSTALL_CONTRACT.json'
        original = contract_path.read_bytes()
        for mutation in ('source', 'runtime', 'before', 'after', 'seam'):
            with self.subTest(contract_drift=mutation):
                contract = json.loads(original)
                section = contract['transforms'][installer.TRANSFORM]
                if mutation in ('source', 'runtime'):
                    name = 'SOURCE_MANIFEST.json' if mutation == 'source' else 'source/runtime.py'
                    section['sources'][name] = '0' * 64
                elif mutation in ('before', 'after'):
                    section['targets'][0][mutation + '_sha256'] = '0' * 64
                else:
                    section['targets'][0]['required_after_seams'][0]['count'] += 1
                try:
                    integration.write(contract_path, contract)
                    refused = full_contract_gate()
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertIn('cooperative-MoE contract differs from its embedded seal', refused.stderr)
                finally:
                    contract_path.write_bytes(original)

    def test_install_contract_drift_refuses_source_validation(self):
        self.integrate()
        path = self.output / integration.COOP / 'INSTALL_CONTRACT.json'
        contract = q.read(path)
        contract['transforms'][installer.TRANSFORM]['sources']['SOURCE_MANIFEST.json'] = '0' * 64
        integration.write(path, contract)
        refresh_fixture(self.output)
        report = self.root / 'contract-drift.json'
        result = self.validate('--report', report)
        self.assertNotEqual(result.returncode, 0)
        failures = [row['check'] for row in q.read(report)['checks'] if row['status'] == 'FAIL']
        self.assertEqual(failures, ['identity-contracts'])

    def test_missing_mismatched_foreign_seals_leave_no_output(self):
        path = self.seal / 'BUILD.json'
        original = path.read_bytes()
        for mutation in ('missing', 'foreign', 'mismatched'):
            with self.subTest(mutation=mutation):
                if mutation == 'missing':
                    path.unlink()
                else:
                    record = copy.deepcopy(self.record)
                    record['qualification_source_manifest_sha256' if mutation == 'foreign' else 'gate_index_sha256'] = q.digest_value(mutation)
                    integration.write(path, record)
                result = self.command(self.source, 'tools/integrate_coop_seal.py',
                    '--sealed-output', self.seal, '--output', self.output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_policy_mismatch_refuses_after_checksums_refreshed(self):
        self.integrate()
        path = self.output / integration.COOP / 'source/dispatch_policy.json'
        path.write_bytes(path.read_bytes() + b' ')
        refresh_fixture(self.output)
        self.assertNotEqual(self.validate().returncode, 0)
        # Isolate the source/bundle equality gate even with a rehashed source.
        self.record['source_manifest_sha256'] = q.sha(self.seal / 'SOURCE_MANIFEST.json')
        with self.assertRaisesRegex(ValueError, 'source seed policy differs'):
            pin = self.source / 'recipe/config/coop-release.json'
            integration.write(pin, {'schema_version': 1, 'status': 'qualified', 'native_sha256': q.TARGET_NATIVE,
                'build_sha256': q.digest_value(self.record), 'gate_index_sha256': self.record['gate_index_sha256']})
            q.verify_record(self.record, self.seal / 'bundle', self.seal)

    def test_final_refuses_every_stage_and_bound_without_measurements(self):
        self.integrate()
        release = self.output / 'manifests/release.json'
        value = q.read(release)
        for stage in ('prepared', 'component-qualified', 'final'):
            with self.subTest(stage=stage):
                value['stage'] = stage
                integration.write(release, value)
                refresh_fixture(self.output)
                self.assertNotEqual(self.validate('--require-final').returncode, 0)
        binding = self.output / 'manifests/final-binding.json'
        value = q.read(binding)
        value['state'] = 'bound'
        integration.write(binding, value)
        refresh_fixture(self.output)
        self.assertNotEqual(self.validate('--require-final').returncode, 0)

    def test_private_state_refuses_inherited_admission(self):
        self.integrate()
        path = self.output / 'manifests/final-binding.json'
        value = q.read(path)
        value['admission_receipt_sha256'] = q.digest_value('historical admission')
        integration.write(path, value)
        refresh_fixture(self.output)
        self.assertNotEqual(self.validate().returncode, 0)

    def test_original_qualification_manifest_remains_hash_bound(self):
        self.integrate()
        path = self.output / integration.COOP / 'QUALIFICATION_SOURCE_MANIFEST.json'
        path.write_bytes(path.read_bytes() + b' ')
        refresh_fixture(self.output)
        self.assertNotEqual(self.validate().returncode, 0)


if __name__ == '__main__':
    unittest.main()
