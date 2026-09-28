#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real native command construction and ELF comparisons; only external I/O stubbed."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import experiment_native_build as experiment

SECRET = 'opaqueNativeExperimentDiagnostic'


def host(board):
    return {'architecture': 'aarch64', 'machine_id_sha256': 'ab' * 32,
            'physical_identity': {'kind': 'gb10-gpu-uuid', 'uuid_sha256': experiment.diagnostics.fingerprint(board)}}


class NativeExperimentTests(unittest.TestCase):
    def run_fixture(self, root, mode='match'):
        output = root / 'builds'
        image = {'config_digest': 'sha256:' + '1' * 64, 'payload_sha256': '2' * 64,
                 'source_recipe_sha256': experiment.native.sha(experiment.native.ROOT / 'recipe/SHA256SUMS')}
        calls = []

        def docker(command, stage, log):
            calls.append(command)
            kind = stage.name.split('-')[0]
            for name in experiment.native.OUTPUTS[kind]:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = Path(sys.executable).read_bytes()
                if mode == 'drift' and stage.name == 'display-2' and name == 'display_kv_probe':
                    data += SECRET.encode()
                target.write_bytes(data)
            log.write(SECRET + '\n')
            return subprocess.CompletedProcess(command, 17 if mode == 'compiler-failure' else 0)

        console = io.StringIO()
        argv = ['experiment_native_build.py', '--image-receipt', str(root / 'image.json'), '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch('validate_release.verify', return_value={'failed': 0}), \
                patch.object(experiment.native, 'read_operator_record', return_value=image), \
                patch.object(experiment.native, 'verify_local_image'), \
                patch.object(experiment.native, 'local_docker', return_value='unix:///var/run/docker.sock'), \
                patch.object(experiment.native, 'builder_host', return_value=host('first')), \
                patch.object(experiment, 'run_container', side_effect=docker), \
                redirect_stdout(console), redirect_stderr(console):
            status = experiment.main()
        self.assertNotIn(SECRET, console.getvalue())
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        report = json.loads((output / 'report.json').read_text())
        self.assertNotIn(SECRET, json.dumps(report))
        self.assertFalse(report['hardware_qualified'])
        self.assertFalse(list(output.rglob('native-build-receipt.json')))
        self.assertTrue(any(SECRET in p.read_text() for p in output.rglob('*' + experiment.diagnostics.PRIVATE_SUFFIX)))
        return output, report, calls, status

    def test_three_fresh_scoped_builds_and_all_pairwise_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report, calls, status = self.run_fixture(Path(directory))
            self.assertEqual(status, 0)
            self.assertEqual(report['status'], 'PASS')
            experiment.checked_report(output / 'report.json')
            self.assertEqual(len(calls), 6)
            self.assertEqual(len({c[c.index('-v') + 1] for c in calls}), 6)
            self.assertEqual([c[c.index('native-experiment') + 1] for c in calls], ['0', '0', '7', '7', '19', '19'])
            for command in calls:
                self.assertEqual(command[:2], ['docker', 'create'])
                for option, value in [('--platform', 'linux/arm64'), ('--network', 'none'),
                                      ('--cpus', '4'), ('--memory', '8g'), ('--memory-swap', '8g')]:
                    self.assertEqual(command[command.index(option) + 1], value)
                self.assertNotIn('--gpus', command)
                self.assertIn('NVIDIA_VISIBLE_DEVICES=void', command)
                self.assertIn('CUDA_VISIBLE_DEVICES=', command)
                self.assertIn('/w/build_repro.sh', command)
            self.assertEqual(len(report['comparisons']), 9)
            self.assertTrue(all(row['identical'] for row in report['comparisons']))

    def test_mismatch_retains_every_elf_and_projects_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report, calls, status = self.run_fixture(Path(directory), 'drift')
            self.assertEqual(status, 9)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(len(calls), 6)
            self.assertEqual(sum(not row['identical'] for row in report['comparisons']), 2)
            for name, (kind, relative) in experiment.ARTIFACTS.items():
                for i in range(1, 4):
                    self.assertEqual(experiment.native.sha(output / f'{kind}-{i}' / relative),
                                     report['runs'][i - 1]['artifacts'][name])
            shared = json.loads((output / 'display_kv_probe-1-2.json').read_text())
            self.assertNotIn(SECRET, json.dumps(shared))
            self.assertNotIn('left', shared['file_bytes']['first_differences'][0] if shared['file_bytes']['first_differences'] else {})
            with self.assertRaises(ValueError):
                experiment.checked_report(output / 'report.json')

    def test_compiler_failure_retains_partial_stage_and_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report, calls, status = self.run_fixture(Path(directory), 'compiler-failure')
            self.assertEqual((status, report['status'], len(calls)), (9, 'REFUSED', 1))
            self.assertTrue((output / 'display-1/display_kv_probe').is_file())
            self.assertEqual(report['runs'], [])

    def test_cross_host_hashes_identity_and_incomplete_or_opaque_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output, record, _, _ = self.run_fixture(Path(directory))
            first, second = output / 'report.json', output / 'second.json'
            other = copy.deepcopy(record)
            other['builder_host'] = host('second')
            experiment.write(second, other)
            result = experiment.compare_hosts([first, second])
            self.assertEqual(result['status'], 'PASS')
            self.assertTrue(result['independent_native_builders'])
            self.assertFalse(result['hardware_qualified'])
            artifact = next(iter(experiment.ARTIFACTS))
            for row in other['runs']:
                row['artifacts'][artifact] = '12' * 32
            experiment.write(second, other)
            self.assertEqual(experiment.compare_hosts([first, second])['status'], 'FAIL')
            for field, value in [('builder_host', host('first')), ('comparisons', []),
                                 ('status', 'INCOMPLETE'), ('build_inputs', {SECRET: SECRET})]:
                damaged = copy.deepcopy(record)
                damaged['builder_host'] = host('second')
                damaged[field] = value
                experiment.write(second, damaged)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    experiment.compare_hosts([first, second])
            damaged = copy.deepcopy(record)
            damaged['runs'][0]['artifacts'][artifact] = SECRET
            experiment.write(second, damaged)
            console = io.StringIO()
            with patch.object(sys, 'argv', ['experiment_native_build.py', '--compare', str(first), str(second)]), \
                    redirect_stdout(console), redirect_stderr(console):
                self.assertEqual(experiment.main(), 9)
            self.assertNotIn(SECRET, console.getvalue())


if __name__ == '__main__':
    unittest.main()
