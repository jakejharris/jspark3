#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline native receipt policy tests. Real builds require an ARM64 Docker runtime."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
                       'builder_host': native.builder_host(),
                       'reproducibility': {'runs': 2, 'comparison': 'bit-identical'},
                       'binary_sha256': {name: (native.TARGET_NATIVE if group == native.OUTPUTS['coop'] else '3' * 64) for group in native.OUTPUTS.values() for name in group.values()},
                       'hardware_qualified': False}

    def write(self, record):
        payload = {k: v for k, v in record.items() if k != 'payload_sha256'}
        payload['payload_sha256'] = hashlib.sha256(native.canonical(payload)).hexdigest()
        self.path.write_bytes(native.canonical(payload))

    def test_operator_outputs_accepted_without_historical_pin(self):
        for names in (native.OUTPUTS['display'].values(), self.record['binary_sha256']):
            record = {**self.record, 'binary_sha256': {name: self.record['binary_sha256'][name] for name in names}}
            self.write(record)
            self.assertEqual(native.read_native_record(self.path, self.image)['binary_sha256'],
                             record['binary_sha256'])

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
        for outputs in ({}, {name: '3' * 64 for name in native.OUTPUTS['coop'].values()},
                        {next(iter(native.OUTPUTS['display'].values())): '3' * 64},
                        {**self.record['binary_sha256'], '../extra.so': '4' * 64},
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


class BuildSelectionTests(unittest.TestCase):
    def test_default_opt_in_and_two_build_refusals(self):
        # Exercise main's selection/publication with compiler and Docker stubs.
        from test_operator_image import fixture, write_record
        cases = ((None, None, ['display', 'display', 'coop', 'coop']),
                 ('--with-coop', None, ['display', 'display', 'coop', 'coop']),
                 ('--display-only', None, ['display', 'display']),
                 (None, 'display', ['display', 'display']),
                 (None, 'coop', ['display', 'display', 'coop', 'coop']),
                 (None, 'wrong-pin', ['display', 'display', 'coop', 'coop']))
        for option, drift, expected_calls in cases:
            with self.subTest(option=option, drift=drift), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                image_path, output = work / 'image.json', work / 'binaries'
                write_record(image_path, fixture())
                calls = []
                def fake_build(kind, stage, image):
                    calls.append(kind)
                    hashes = {}
                    for name, relative in native.OUTPUTS[kind].items():
                        path = stage / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'synthetic compiler output')
                        value = native.TARGET_NATIVE if kind == 'coop' else native.sha(path)
                        if kind == drift and calls.count(kind) == 2 or drift == 'wrong-pin' and kind == 'coop':
                            value = native.sha(path)
                            value = 'f' * 64
                        hashes[relative] = value
                    return hashes
                argv = ['build_native.py', '--image-receipt', str(image_path), '--output', str(output)]
                if option:
                    argv.append(option)
                log = io.StringIO()
                with patch.object(native, 'TARGET_NATIVE', hashlib.sha256(b'synthetic compiler output').hexdigest()), \
                        patch.object(sys, 'argv', argv), patch.object(native, 'verify_local_image'), \
                        patch('validate_release.verify', return_value={'failed': 0}), \
                        patch.object(native, 'build', side_effect=fake_build), \
                        redirect_stdout(log), redirect_stderr(log):
                    status = native.main()
                self.assertEqual(calls, expected_calls)
                self.assertEqual(status, 9 if drift else 0, log.getvalue())
                self.assertEqual(output.exists(), drift is None)
                if drift:
                    self.assertIn('candidate pin' if drift == 'wrong-pin' else drift + ': two native builds differ', log.getvalue())
                else:
                    with patch.object(native, 'TARGET_NATIVE', hashlib.sha256(b'synthetic compiler output').hexdigest()):
                        record = native.read_native_record(output / 'native-build-receipt.json',
                                                           native.read_operator_record(image_path))
                    expected = {name for kind in set(expected_calls) for name in native.OUTPUTS[kind].values()}
                    self.assertEqual(set(record['binary_sha256']), expected)
                    self.assertEqual((output / native.COOP / 'bundle/cooperative_moe.so').exists(), option != '--display-only')


class DefaultProfileTests(unittest.TestCase):
    def test_prepare_and_default_consumers_with_operator_receipts(self):
        self.prepare_and_check(with_coop=False)

    def test_default_refuses_missing_coop(self):
        self.prepare_and_check(with_coop=False, default_refusal=True)

    def prepare_and_check(self, with_coop, default_refusal=False):
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
            for kind in (['display', 'coop'] if with_coop else ['display']):
                group = native.OUTPUTS[kind]
                for name in group.values():
                    path = binaries / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'\x7fELF synthetic ARM64 build fixture ' + name.encode())
                    outputs[name] = native.sha(path)
            record = {'schema_version': 1, 'verification': 'fixed-native-build-v1',
                      'source_recipe_sha256': image['source_recipe_sha256'],
                      'image_receipt_sha256': image['payload_sha256'], 'build_inputs': native.build_inputs(),
                      'builder_host': native.builder_host(),
                       'reproducibility': {'runs': 2, 'comparison': 'bit-identical'},
                      'binary_sha256': outputs, 'hardware_qualified': False}
            receipt = work / 'native.json'
            write_record(receipt, record)
            process = subprocess.run([sys.executable, str(source / 'tools/prepare_runtime.py'),
                                      '--binary-root', str(binaries), '--image-receipt', str(image_path),
                                      '--native-receipt', str(receipt), '--output', str(runtime),
                                      *([] if default_refusal else ['--coop-off'])],
                                     capture_output=True, text=True)
            if default_refusal:
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('default preparation requires the pinned coop native', process.stderr)
                self.assertFalse(runtime.exists())
                return
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual((runtime / native.COOP / 'bundle/cooperative_moe.so').exists(), with_coop)
            process = subprocess.run([sys.executable, str(runtime / 'recipe/scripts/remote_preflight.py'),
                                      '--recipe-only', '--recipe-root', str(runtime / 'recipe')],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            for name in ('BUILD.json', 'bundle/manifest.json', 'bundle/dispatch_policy.json'):
                self.assertEqual((runtime / 'recipe/overlays/v16/coop' / name).read_bytes(),
                                 (source / 'recipe/overlays/v16/coop' / name).read_bytes())
            self.assertIn('JSPARK3_V16_COOP=1\n', (source / 'recipe/.env.example').read_text())
            code = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import _fleetctl as fleet, remote_preflight as preflight
r = Path(sys.argv[1]).parent
os.environ['JSPARK3_V16_COOP'] = '1'
values = fleet.load_env(r / '.env.example')
assert values['JSPARK3_V16_PROFILE'] == 'production-stock'
assert values['JSPARK3_V16_COOP'] == '0' and values['JSPARK3_V14_PROFILE'] == 'full'
row = fleet.expected_v16_row(values)
assert row == preflight.v16_artifacts(r, values['JSPARK3_V16_PROFILE'], values['JSPARK3_V16_COOP'],
                                    values['GLM53_ADAPTIVE_K'], values['JSPARK3_V16_DENSE_FP8'])
assert row['coop'] == 'off' and row['coop_bundle'] == 'UNSEALED_ALLOWED_ONLY_WHEN_OFF'
argv = fleet.preflight_argv(values, 0)
assert argv[argv.index('--v16-coop') + 1] == '0'
assert preflight.verify_manifest(r)
values['JSPARK3_V16_COOP'] = '1'
for check in (lambda: fleet.expected_v16_row(values),
              lambda: preflight.v16_artifacts(r, 'production-stock', '1', 'ema', 'trunk')):
    try:
        check()
    except preflight.Refusal as exc:
        assert 'hardware-sealed' in str(exc) and 'component seal' in str(exc), str(exc)
    else:
        raise AssertionError('coop=1 accepted an unsealed operator binary')
for name in ('overlays/v14/display_kv/display_kv_probe', 'overlays/v14/display_kv/libglm53_display_kv.so',
             'overlays/v16/coop/bundle/cooperative_moe.so'):
    path = r / name
    if not path.exists():
        assert name == 'overlays/v16/coop/bundle/cooperative_moe.so'
        continue
    original = path.read_bytes()
    path.write_bytes(original + b'tamper')
    try:
        preflight.verify_manifest(r)
    except preflight.Refusal as exc:
        assert 'recipe manifest mismatch' in str(exc), str(exc)
    else:
        raise AssertionError('accepted runtime tamper: ' + name)
    finally:
        path.write_bytes(original)
print('PASS prepared production-stock coop=0 full: controller/preflight; coop=1 and installed binary tampers refused')
'''
            process = subprocess.run([sys.executable, '-B', '-c', code, str(runtime / 'recipe/scripts')],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertFalse(list(runtime.rglob('__pycache__')))


if __name__ == '__main__':
    unittest.main()
