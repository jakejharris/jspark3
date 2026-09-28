#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline native receipt policy tests. Real builds require an ARM64 Docker runtime."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import build_native as native


class NativeReceiptTests(unittest.TestCase):
    def test_runtime_policy_matches_build_inputs_and_reference(self):
        policy = json.loads((native.ROOT / 'recipe/config/native-build-policy.json').read_text())
        self.assertEqual(policy['build_inputs'], native.build_inputs())
        reference = policy['coop_reference']
        coop = native.ROOT / native.COOP
        self.assertEqual(reference['manifest'], json.loads((coop / 'bundle/manifest.json').read_text()))
        self.assertEqual(reference['policy_sha256'], native.sha(coop / 'bundle/dispatch_policy.json'))
        self.assertEqual(reference['build_record_sha256'], native.sha(coop / 'BUILD.json'))
        self.assertEqual(reference['rows'], json.loads((coop / 'bundle/dispatch_policy.json').read_text())['rows'])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'receipt.json'
        self.image = {'payload_sha256': '2' * 64,
                      'source_recipe_sha256': native.sha(native.ROOT / 'recipe/SHA256SUMS')}
        self.record = {'schema_version': 1, 'verification': 'fixed-native-build-v1',
                       'source_recipe_sha256': self.image['source_recipe_sha256'],
                       'image_receipt_sha256': self.image['payload_sha256'],
                       'build_inputs': native.build_inputs(),
                       'reproducibility': {'runs': 2, 'comparison': 'bit-identical'},
                       'binary_sha256': {name: '3' * 64 for group in native.OUTPUTS.values() for name in group.values()},
                       'hardware_qualified': False}

    def write(self, record):
        payload = {k: v for k, v in record.items() if k != 'payload_sha256'}
        payload['payload_sha256'] = hashlib.sha256(native.canonical(payload)).hexdigest()
        self.path.write_bytes(native.canonical(payload))

    def test_operator_outputs_accepted_without_historical_pin(self):
        self.write(self.record)
        self.assertEqual(native.read_native_record(self.path, self.image)['binary_sha256'],
                         self.record['binary_sha256'])

    def test_source_and_recipe_drift_refused_even_when_reselfhashed(self):
        for name in ('tools/build_native.py', f'{native.DISPLAY}/display_kv.c',
                     f'{native.COOP}/build_repro.sh', f'{native.COOP}/source/native/cooperative_moe.cu'):
            with self.subTest(name=name):
                record = copy.deepcopy(self.record)
                record['build_inputs'][name] = '0' * 64
                self.write(record)
                with self.assertRaisesRegex(native.ImageRefusal, 'inputs/recipe drift'):
                    native.read_native_record(self.path, self.image)

    def test_source_image_and_qualification_drift_refused(self):
        for field, value in (('source_recipe_sha256', '0' * 64), ('image_receipt_sha256', '0' * 64),
                             ('hardware_qualified', True), ('reproducibility', {'runs': 1})):
            with self.subTest(field=field):
                self.write({**self.record, field: value})
                with self.assertRaises(native.ImageRefusal):
                    native.read_native_record(self.path, self.image)

    def test_output_inventory_is_exact(self):
        for outputs in ({}, {**self.record['binary_sha256'], '../extra.so': '4' * 64},
                        {name: 'invalid' for name in self.record['binary_sha256']}):
            self.write({**self.record, 'binary_sha256': outputs})
            with self.assertRaisesRegex(native.ImageRefusal, 'output inventory drift'):
                native.read_native_record(self.path, self.image)

    def test_receipt_tamper_and_symlink_refused(self):
        self.write(self.record)
        self.path.write_bytes(self.path.read_bytes().replace(b'3' * 64, b'4' * 64))
        with self.assertRaisesRegex(native.ImageRefusal, 'payload hash mismatch'):
            native.read_native_record(self.path, self.image)
        self.path.unlink()
        self.path.symlink_to(self.path.parent / 'absent')
        with self.assertRaisesRegex(native.ImageRefusal, 'missing or symlinked'):
            native.read_native_record(self.path, self.image)


class DefaultProfileTests(unittest.TestCase):
    def test_prepare_and_default_consumers_with_operator_receipts(self):
        # Synthetic receipt/output fixtures exercise the trust policy and real
        # preparation/consumer code. They are not a compiler or GPU test.
        from test_operator_image import fixture, write_record
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, binaries, runtime = (work / n for n in ('source', 'binaries', 'runtime'))
            for line in (native.ROOT / 'SHA256SUMS').read_text().splitlines():
                name = line.split('  ', 1)[1]
                dest = source / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(native.ROOT / name, dest)
            shutil.copyfile(native.ROOT / 'SHA256SUMS', source / 'SHA256SUMS')
            image_path = work / 'image.json'
            write_record(image_path, fixture())
            image = native.read_operator_record(image_path)
            outputs = {}
            for group in native.OUTPUTS.values():
                for name in group.values():
                    path = binaries / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'\x7fELF synthetic ARM64 build fixture ' + name.encode())
                    outputs[name] = native.sha(path)
            record = {'schema_version': 1, 'verification': 'fixed-native-build-v1',
                      'source_recipe_sha256': image['source_recipe_sha256'],
                      'image_receipt_sha256': image['payload_sha256'], 'build_inputs': native.build_inputs(),
                      'reproducibility': {'runs': 2, 'comparison': 'bit-identical'},
                      'binary_sha256': outputs, 'hardware_qualified': False}
            receipt = work / 'native.json'
            write_record(receipt, record)
            process = subprocess.run([sys.executable, str(source / 'tools/prepare_runtime.py'),
                                      '--binary-root', str(binaries), '--image-receipt', str(image_path),
                                      '--native-receipt', str(receipt), '--output', str(runtime)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            code = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import _fleetctl as fleet, remote_preflight as preflight, apply_coop_moe as coop
r = Path(sys.argv[1]).parent
values = fleet.load_env(r / '.env.example')
assert values['JSPARK3_V16_PROFILE'] == 'production-stock'
assert values['JSPARK3_V16_COOP'] == '1' and values['JSPARK3_V14_PROFILE'] == 'full'
row = fleet.expected_v16_row(values)
assert row['coop'] == 'on' and row['profile'] == 'production-stock'
assert coop.verify_bundle(coop.DEFAULT_BUNDLE, coop.DEFAULT_BUILD_RECORD) == row['coop_bundle']
policy = json.loads((coop.DEFAULT_BUNDLE / 'dispatch_policy.json').read_text())
assert 'profile_log_sha256' not in policy and 'reference_policy_sha256' in policy
record = json.loads(coop.DEFAULT_BUILD_RECORD.read_text())
assert record['hardware_qualified'] is False
import os, runpy, types
os.environ.pop('GLM53_COOP_QUALIFICATION', None)
os.environ.pop('GLM53_COOP_MAINTENANCE_TEST', None)
# Exercise the actual adapter's file/policy verifier without importing a GPU
# framework or executing kernels. These functions do not access torch.
sys.modules['torch'] = types.ModuleType('torch')
adapter = runpy.run_path(str(coop.DEFAULT_BUNDLE / 'runtime.py'))
assert adapter['verify_bundle'](coop.DEFAULT_BUNDLE) == coop.DEFAULT_BUNDLE / 'cooperative_moe.so'
assert adapter['load_row_policy'](coop.DEFAULT_BUNDLE) == {int(k): v for k, v in policy['rows'].items()}
# Both normal host preflight and the patch installer use this verifier.
for path in (coop.DEFAULT_BUNDLE / 'cooperative_moe.so', coop.DEFAULT_BUNDLE / 'dispatch_policy.json',
             coop.DEFAULT_BUNDLE / 'manifest.json', coop.DEFAULT_BUILD_RECORD,
             r / 'config/operator-native.json', r / 'config/operator-image.json',
             coop.SOURCE_ROOT / 'native/cooperative_moe.cu'):
    original = path.read_bytes()
    path.write_bytes(original + b'tamper')
    try:
        fleet.expected_v16_row(values)
    except (ValueError, preflight.Refusal, coop.Refusal):
        pass
    else:
        raise AssertionError('accepted tamper: ' + str(path))
    finally:
        path.write_bytes(original)
assert preflight.verify_manifest(r)
print('PASS default production-stock coop=1 full: host/controller/installer/adapter artifact checks; seven tamper refusals')
'''
            process = subprocess.run([sys.executable, '-B', '-c', code, str(runtime / 'recipe/scripts')],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertFalse(list(runtime.rglob('__pycache__')))


if __name__ == '__main__':
    unittest.main()
