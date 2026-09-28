#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline remote failure controls through the real verify receipt writer."""
from contextlib import redirect_stderr
import io
import json
import os
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
    def verify_failure(self, stderr, *, local_child=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "verify.json"
            manifest = root / "manifest.json"
            manifest.write_text('{}\n')
            process = subprocess.CompletedProcess(['ssh'], 1, '', stderr)
            console = io.StringIO()

            def failing_check(args, values, bound, candidate):
                fleet.remote(values, 2, ['python3', 'remote_check.py'])

            argv = ['fleetctl', 'verify', '--env-file', str(root / 'env'),
                    '--manifest', str(manifest), '--output', str(output),
                    '--log-output', str(root / 'rank0.log')]
            with (patch.object(sys, 'argv', argv),
                  patch.object(fleet, 'load_env', return_value={}),
                  patch.object(fleet, 'validate_env'),
                  patch.object(fleet, 'bound_manifest', return_value={}),
                  patch.object(fleet, 'wait_ready'),
                  patch.object(fleet, '_verify_bound', side_effect=failing_check),
                  patch.object(fleet, 'preserve_rank0_logs') as preserve,
                  patch.object(fleet, 'ssh_argv', return_value=[sys.executable, '-c',
                               'import sys;sys.stderr.write(sys.argv[1]);sys.exit(1)', stderr]),
                  patch.object(fleet.subprocess, 'run', wraps=subprocess.run if local_child else None,
                               **({} if local_child else {'return_value': process})),
                  redirect_stderr(console)):
                self.assertEqual(fleet.main(), 9)
                preserve.assert_called_once()
            record = json.loads(output.read_text())
            digest = record.pop('payload_sha256')
            self.assertEqual(digest, fleet.sha_bytes(fleet.canonical(record)))
            self.assertEqual(record['status'], 'VERIFY_REFUSED')
            self.assertEqual(record['manifest_sha256'], fleet.sha_file(manifest))
            self.console = console.getvalue()
            return record['reason']

    def test_secret_bearing_traceback_is_redacted_in_receipt_and_console(self):
        secret = 'synthetic-' + 'environment-credential'
        flag = 'synthetic-' + 'argv-credential'
        header = 'synthetic-' + 'header-credential'
        diagnostics = (secret + '\n' + f"--api-key '{flag}'\nAuthorization: Bearer {header}\n" + TRACEBACK)
        with patch.dict(os.environ, {'TEST_ACCESS_TOKEN': secret}):
            reason = self.verify_failure(diagnostics, local_child=True)
        for output in (reason, self.console):
            for value in (secret, flag, header):
                self.assertNotIn(value, output)
            self.assertIn('remote_check.py', output)
            self.assertIn('AssertionError: native identity drift', output)
            self.assertIn('[REDACTED]', output)

    def test_redaction_happens_before_tail_truncation(self):
        secret = 'sensitive-' + 'x' * 20000
        reason = self.verify_failure('TOKEN=' + secret + '\n' + TRACEBACK)
        self.assertNotIn('x' * 100, reason)
        self.assertIn(TRACEBACK.rstrip(), reason)

    def test_quoted_assignments_argv_and_credential_urls(self):
        for diagnostic in ('PASSWORD="private value"', "['--api-key', 'private value']",
                           '{"access_token": "private value"}', 'https://' + 'user:' + 'private@host.invalid/'):
            self.assertNotIn('private', fleet.redact_diagnostics(diagnostic))

    def test_escaped_quoted_secret_is_redacted_whole(self):
        secret = 'private' + chr(34) + 'suffix'
        for diagnostic in (json.dumps({'password': secret}), repr(['--api-key', secret])):
            self.assertNotIn('suffix', fleet.redact_diagnostics(diagnostic))
        with patch.dict(os.environ, {'TEST_SECRET': secret}):
            self.assertNotIn('suffix', fleet.redact_diagnostics(json.dumps(secret)))

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
