#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline native receipt policy tests. Real builds require an ARM64 Docker runtime."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import build_native as native


class NativeReceiptTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
