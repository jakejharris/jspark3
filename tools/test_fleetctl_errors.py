#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline remote failure controls through the real verify receipt writer."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe/scripts"))
import _fleetctl as fleet

TRACEBACK = ('Traceback (most recent call last):\n'
             '  File "remote_check.py", line 23, in check\n'
             '    assert actual == expected, "native identity drift"\n'
             'AssertionError: native identity drift\n')


class RemoteFailureTests(unittest.TestCase):
    def verify_failure(self, stderr):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "verify.json"
            manifest = root / "manifest.json"
            manifest.write_text('{}\n')
            process = subprocess.CompletedProcess(['ssh'], 1, '', stderr)

            def failing_check(args, values, bound, candidate):
                fleet.remote(values, 2, ['python3', 'remote_check.py'])

            argv = ['fleetctl', 'verify', '--env-file', str(root / 'env'),
                    '--manifest', str(manifest), '--output', str(output),
                    '--log-output', str(root / 'rank0.log')]
            with (patch.object(sys, 'argv', argv),
                  patch.object(fleet, 'load_env', return_value={}),
                  patch.object(fleet, 'validate_env'),
                  patch.object(fleet, 'bound_manifest', return_value={}),
                  patch.object(fleet, '_verify_bound', side_effect=failing_check),
                  patch.object(fleet, 'preserve_rank0_logs') as preserve,
                  patch.object(fleet, 'ssh_argv', return_value=['ssh']),
                  patch.object(fleet.subprocess, 'run', return_value=process),
                  redirect_stderr(io.StringIO())):
                self.assertEqual(fleet.main(), 9)
                preserve.assert_called_once()
            record = json.loads(output.read_text())
            digest = record.pop('payload_sha256')
            self.assertEqual(digest, fleet.sha_bytes(fleet.canonical(record)))
            self.assertEqual(record['status'], 'VERIFY_REFUSED')
            self.assertEqual(record['manifest_sha256'], fleet.sha_file(manifest))
            return record['reason']

    def test_verify_record_keeps_multiline_traceback(self):
        reason = self.verify_failure(TRACEBACK)
        self.assertIn('rank2 remote command failed (exit 1)', reason)
        self.assertIn(TRACEBACK.rstrip(), reason)

    def test_large_stderr_keeps_bounded_tail_and_marks_truncation(self):
        reason = self.verify_failure('OLD-OUTPUT\n' + 'x' * 32000 + '\n' + TRACEBACK)
        self.assertNotIn('OLD-OUTPUT', reason)
        self.assertIn('[stderr truncated; tail follows]', reason)
        self.assertIn(TRACEBACK.rstrip(), reason)
        self.assertLess(len(reason), 16 * 1024 + 128)

    def test_empty_stderr_has_rank_exit_and_fallback(self):
        self.assertIn('rank2 remote command failed (exit 1):\nno detail',
                      self.verify_failure(' \n'))

    def test_unchecked_remote_returns_original_process(self):
        process = subprocess.CompletedProcess(['ssh'], 4, 'partial', TRACEBACK)
        with (patch.object(fleet, 'ssh_argv', return_value=['ssh']),
              patch.object(fleet.subprocess, 'run', return_value=process)):
            self.assertIs(fleet.remote({}, 0, ['check'], check=False), process)


if __name__ == '__main__':
    unittest.main()
