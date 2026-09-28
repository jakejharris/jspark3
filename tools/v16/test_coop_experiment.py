#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline diagnostic/runner tests; Docker builds and image checks are stubbed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import diff_native_elf as diff
import experiment_coop_build as experiment


class ElfTests(unittest.TestCase):
    def test_identity(self):
        result = diff.compare(sys.executable, sys.executable)
        self.assertTrue(result['identical'])
        self.assertEqual(result['changed_sections'], [])
        self.assertEqual(result['symbols']['left_only_count'], 0)

    def test_changed_code_locates_section_and_symbol(self):
        elf = diff.Elf(sys.executable)
        symbol = next(s for s in elf.symbols() if s[6] and 0 < s[4] < len(elf.sections)
                      and elf.sections[s[4]]['name'] == '.text' and s[7])
        section = elf.sections[symbol[4]]
        data = bytearray(elf.data)
        data[section['offset'] + symbol[5] - section['address']] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'changed.so'
            target.write_bytes(data)
            result = diff.compare(sys.executable, target)
        self.assertFalse(result['identical'])
        self.assertEqual([r['name'] for r in result['changed_sections']], ['.text'])
        self.assertEqual(result['file_bytes']['different_bytes'], 1)
        self.assertIn(symbol[1], [s['name'] for s in result['symbols']['left_only']])

    def test_non_elf_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'not-elf'
            path.write_bytes(b'not an ELF')
            with self.assertRaisesRegex(ValueError, 'ELF64'):
                diff.Elf(path)


class RunnerTests(unittest.TestCase):
    def test_three_fresh_builds_and_mismatch_evidence(self):
        for mode in ('match', 'drift', 'baseline-failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                receipt = work / 'image.json'
                receipt.write_text('{}')
                output = work / 'experiment'
                calls = []

                def fake_docker(command, stdout, stderr):
                    calls.append(command)
                    stage = Path(command[command.index('-v') + 1].removesuffix(':/w'))
                    (stage / 'out').mkdir()
                    raw = Path(sys.executable).read_bytes()
                    if mode == 'drift' and len(calls) == 2:
                        raw += b'changed output'
                    failed = mode == 'baseline-failure' and stage.name.startswith('baseline-')
                    if failed:
                        raw = b'incomplete linker output'
                    (stage / 'out/cooperative_moe.so').write_bytes(raw)
                    (stage / 'out/elf.txt').write_text('Shared library: [libcudart.so.13]\n')
                    (stage / 'out/ldd.txt').write_text('libcudart.so.13 => /cuda/libcudart.so.13\n')
                    (stage / 'out/cuda-elf.txt').write_text('CUDA diagnostic fixture\n')
                    stdout.write('synthetic compiler output; no GPU\n')
                    return subprocess.CompletedProcess(command, 1 if failed else 0)

                image = {'config_digest': 'sha256:' + '1' * 64,
                         'source_recipe_sha256': experiment.native.sha(experiment.ROOT / 'recipe/SHA256SUMS')}
                argv = ['experiment_coop_build.py', '--image-receipt', str(receipt), '--output', str(output)]
                if mode == 'baseline-failure':
                    argv += ['--baseline-builder', str(experiment.ROOT / experiment.native.COOP / 'build_repro.sh')]
                with patch.object(sys, 'argv', argv), patch.object(experiment.native, 'verify_local_image'), \
                        patch.object(experiment.native, 'read_operator_record', return_value=image), \
                        patch('validate_release.verify', return_value={'failed': 0}), \
                        patch.object(experiment.subprocess, 'check_output', return_value='aarch64\n'), \
                        patch.object(experiment.subprocess, 'run', side_effect=fake_docker), redirect_stdout(io.StringIO()):
                    status = experiment.main()
                self.assertEqual(status, 9 if mode == 'drift' else 0)
                groups = 2 if mode == 'baseline-failure' else 1
                self.assertEqual([c[-1] for c in calls], ['0', '37', '74'] * groups)
                self.assertEqual(len({c[c.index('-v') + 1] for c in calls}), 3 * groups)
                self.assertTrue(all('--gpus' not in c and c[c.index('--network') + 1] == 'none' for c in calls))
                result = json.loads((output / 'experiment.json').read_text())
                self.assertEqual(result['status'], 'FAIL' if mode == 'drift' else 'PASS')
                if mode == 'baseline-failure':
                    self.assertEqual(len(result['builders']['baseline']['runs']), 3)
                    self.assertIn('elf_error', result['builders']['baseline']['runs'][0])
                self.assertFalse(result['hardware_qualified'])
                self.assertEqual(len(result['builders']['candidate']['comparisons']), 2)
                self.assertTrue(all((output / f'candidate-{n}/console.log').exists() for n in (1, 2, 3)))
                comparison = json.loads((output / 'candidate-1-vs-2.json').read_text())
                self.assertEqual(comparison['identical'], mode != 'drift')


if __name__ == '__main__':
    unittest.main()
