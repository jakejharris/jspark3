#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline refusal controls for the sanitizer worker; no GPU qualification."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import coop_sanitizer as worker
import coop_evidence as evidence
import qualify_coop


class SanitizerTests(unittest.TestCase):
    def execute(self, *, code=0, summary='ERROR SUMMARY: 0 errors', records=True,
                replay_code=0, launches=None):
        calls = []
        launch_text = ('========= Launch #1 (filtered: not in include list)\n'
                       '=========   Kernel: stock_control()\n'
                       '========= Launch #2\n'
                       '=========   Kernel: exl3_moe_coop_detector_control(int*, int)\n'
                       '========= ERROR SUMMARY: 2 errors\n') if launches is None else launches
        def child(command, **kwargs):
            calls.append(command)
            if '--read' in command:
                self.assertTrue(Path(command[command.index('--read') + 1]).is_file())
                kwargs['stdout'].write(launch_text)
                return subprocess.CompletedProcess(command, replay_code)
            Path(command[command.index('--log-file') + 1]).write_text('========= ' + summary + '\n')
            if records:
                Path(command[command.index('--save') + 1]).write_bytes(b'fake saved execution')
            return subprocess.CompletedProcess(command, code)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(tempfile, 'tempdir', directory), \
                patch.object(worker.subprocess, 'run', side_effect=child), contextlib.redirect_stdout(output):
            result = worker.run('memcheck', ['test-app'])
        return result, calls, output.getvalue()

    def test_clean_gate_preserves_saved_instrumentation_without_info_summary(self):
        result, calls, text = self.execute()
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(evidence.sanitizer(text, 'memcheck'), 1)
        live, replay = calls
        self.assertEqual(live[live.index('--error-exitcode') + 1], '9')
        self.assertEqual(live[live.index('--print-level') + 1], 'warn')
        self.assertEqual(live[live.index('--kernel-name') + 1], evidence.KERNEL_FILTER)
        self.assertIn('--dump-kernel-launches', live)
        self.assertNotIn('test-app', replay)
        self.assertNotIn('stock_control', text)
        self.assertNotIn('ERROR SUMMARY: 2', text)

    def test_live_failure_is_never_replayed_or_converted_to_success(self):
        for code in (1, 9, 137, -15):
            result, calls, text = self.execute(code=code)
            self.assertEqual(result, 9 if code < 0 else code)
            self.assertEqual(len(calls), 1)
            self.assertEqual(text, '')

    def test_zero_exit_cannot_override_bad_or_missing_summary(self):
        for summary in ('ERROR SUMMARY: 1 errors', '', 'ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 1 errors',
                        'ERROR SUMMARY: 0 errors\n========= Error: tool incomplete'):
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                self.execute(summary=summary)

    def test_missing_records_readback_failure_and_filtered_only_refuse(self):
        for arguments in ({'records': False}, {'replay_code': 9}, {'launches': ''},
                          {'launches': 'Internal Sanitizer Error'},
                          {'launches': '========= Launch #1 (filtered: not in include list)\n'
                                       '=========   Kernel: exl3_moe_coop_detector_control()\n'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.execute(**arguments)

    def test_all_remaining_campaign_siblings_use_the_worker(self):
        plan = [('environment', []), *qualify_coop.matrix()]
        self.assertEqual(len(plan), 56)
        self.assertEqual(plan[4][0], 'smoke-r0-g0-memcheck')
        remaining = plan[5:]
        self.assertEqual(len(remaining), 51)
        sanitizers = [(name, cmd) for name, cmd in plan if name.endswith(('-memcheck', '-racecheck'))]
        self.assertEqual(len(sanitizers), 24)
        self.assertEqual(sum(name.endswith('-memcheck') for name, _ in remaining), 11)
        self.assertEqual(sum(name.endswith('-racecheck') for name, _ in remaining), 12)
        for name, command in sanitizers:
            self.assertIn('/src/tools/v16/coop_sanitizer.py', command)
            self.assertNotIn('/sanitizer/compute-sanitizer', command)
        self.assertIn('tools/v16/coop_sanitizer.py', qualify_coop.runner_inputs())


if __name__ == '__main__':
    unittest.main()
