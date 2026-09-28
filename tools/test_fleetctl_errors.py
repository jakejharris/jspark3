#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real failing children -> verify -> shared receipt and private diagnostic tail."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recipe/scripts'))
import _fleetctl as fleet
from _diagnostic_cases import TRACE, cases


class RemoteFailureTests(unittest.TestCase):
    def verify_failure(self, stderr, *, write_error=None, private_error=None, local_error=None,
                       preserve_error=None, argv=None, values=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'verify.json'
            manifest = root / 'manifest.json'
            manifest.write_text('{}\n')
            source = root / 'input.txt'
            source.write_text(stderr)
            console = io.StringIO()

            def failing_check(args, env, bound, candidate):
                if local_error:
                    raise local_error
                fleet.remote(env, 2, argv or ['python3', 'remote_check.py'])

            cli = ['fleetctl', 'verify', '--env-file', str(root / 'env'),
                   '--manifest', str(manifest), '--output', str(output),
                   '--log-output', str(root / 'rank0.log')]
            child = [sys.executable, '-B', '-c',
                     'import sys;from pathlib import Path;sys.stderr.write(Path(sys.argv[1]).read_text());sys.exit(17)',
                     str(source)]
            with (patch.object(sys, 'argv', cli),
                  patch.object(fleet, 'load_env', return_value=values or {}),
                  patch.object(fleet, 'validate_env'),
                  patch.object(fleet, 'bound_manifest', return_value={}),
                  patch.object(fleet, 'wait_ready'),
                  patch.object(fleet, '_verify_bound', side_effect=failing_check),
                  patch.object(fleet, 'preserve_rank0_logs', side_effect=preserve_error) as preserve,
                  patch.object(fleet, 'atomic_json', wraps=fleet.atomic_json, side_effect=write_error),
                  patch.object(fleet.diagnostics, 'private_tail', wraps=fleet.diagnostics.private_tail,
                               side_effect=private_error),
                  patch.object(fleet, 'ssh_argv', return_value=child),
                  redirect_stderr(console)):
                self.assertEqual(fleet.main(), 9)
                preserve.assert_called_once()
            self.console = console.getvalue()
            if write_error:
                self.assertFalse(output.exists())
                self.assertTrue(list(root.glob('*' + fleet.diagnostics.PRIVATE_SUFFIX)))
                return None
            self.record = json.loads(output.read_text())
            record = dict(self.record)
            digest = record.pop('payload_sha256')
            self.assertEqual(digest, fleet.sha_bytes(fleet.canonical(record)))
            self.assertEqual(record['status'], 'VERIFY_REFUSED')
            self.assertEqual(record['manifest_sha256'], fleet.sha_file(manifest))
            if private_error:
                self.assertIsNone(record['private_stderr_tail'])
            else:
                name = record['private_stderr_tail']
                self.assertEqual(Path(name).name, name)
                self.assertTrue(name.endswith(fleet.diagnostics.PRIVATE_SUFFIX))
                self.assertIn(name, self.console)
                private = root / name
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
                self.raw_tail = private.read_text()
            return record

    def test_reviewer_76_case_matrix(self):
        self.assertEqual(len(cases), 76)
        for case in cases:
            with self.subTest(case=case['name']), patch.dict(os.environ, case['known']):
                text = case['diagnostic'] + ('\n' + TRACE if case['append_trace'] else '')
                record = self.verify_failure(text)
                public = json.dumps(record) + self.console
                for secret in case['secrets']:
                    self.assertNotIn(secret, public)
                self.assertEqual(self.raw_tail, text.replace('\r\n', '\n')[-fleet.REMOTE_STDERR_LIMIT:])
                # Free-form diagnostic markers are retained privately, not
                # promoted to shared output just because they look innocuous.
                for marker in case['markers']:
                    self.assertIn(marker, self.raw_tail)
                detail = record['diagnostics']
                self.assertEqual((detail['rank'], detail['exit_code'], detail['command']),
                                 (2, 17, 'python3 remote_check.py'))
                self.assertIn('rank2 remote command failed (exit 17)', record['reason'])
                if case['append_trace'] or 'File "remote_check.py"' in text:
                    self.assertIn({'file': 'remote_check.py', 'line': 23}, detail['frames'])
                    self.assertIn('AssertionError', detail['exception_types'])
                    self.assertIn('remote_check.py:23', self.console)
                self.assertNotIn('native identity drift', public)

    def test_unknown_fields_paths_names_and_arguments_cannot_become_shared_text(self):
        secret = 'opaque-field-credential'
        text = (f'  File "/{secret}/verify_stock.py", line 42, in {secret}\n'
                f'  File "{secret}.py", line 9, in check\n'
                f'{secret}Error: {secret}\nValueError: {secret}\n' + TRACE)
        record = self.verify_failure(text, argv=['python3', '-S', '/recipe/scripts/verify_stock.py',
                                                '--argument', secret])
        self.assertNotIn(secret, json.dumps(record) + self.console)
        self.assertIn(secret, self.raw_tail)
        self.assertIn({'file': 'verify_stock.py', 'line': 42}, record['diagnostics']['frames'])
        self.assertIn({'file': '<unrecognized file>', 'line': 9}, record['diagnostics']['frames'])
        self.assertEqual(record['diagnostics']['command'], 'python3 verify_stock.py')

    def test_docker_identity_and_inline_program_never_enter_command_summary(self):
        secret = 'opaque-argument-credential'
        for argv in (['docker', 'exec', secret, 'python3', '-c', secret], ['python3', '-c', secret]):
            with self.subTest(argv=argv):
                record = self.verify_failure(TRACE, argv=argv)
                self.assertNotIn(secret, json.dumps(record) + self.console)
                self.assertIn(record['diagnostics']['command'], ('docker exec', 'python3 -c'))

    def test_receipt_log_and_private_write_errors_cannot_echo_secrets(self):
        secret = 'opaque-write-error-credential'
        for field in ('write_error', 'private_error', 'preserve_error'):
            with self.subTest(field=field):
                self.verify_failure(TRACE, **{field: OSError(secret)})
                self.assertNotIn(secret, self.console)
                self.assertIn('remote_check.py:23', self.console)
                self.assertIn('AssertionError', self.console)

    def test_nonremote_failures_also_exclude_freeform_messages(self):
        secret = 'opaque-local-credential'
        record = self.verify_failure('', local_error=fleet.Refusal('gate failed: ' + secret))
        self.assertNotIn(secret, json.dumps(record) + self.console)
        self.assertIn(secret, self.raw_tail)
        self.assertIn('Refusal', record['diagnostics']['exception_types'])
        self.assertTrue(record['diagnostics']['controller_frames'])

    def test_large_adversarial_input_finishes_and_writes_receipt(self):
        # Bound the whole real writer externally, not merely its string helper.
        source = Path(__file__).resolve()
        for count in (32000, 200000):
            with self.subTest(chars=count * 6):
                code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                        "from test_fleetctl_errors import RemoteFailureTests, TRACE;"
                        "t=RemoteFailureTests();"
                        "t.verify_failure('https://host.invalid/?'+'token.'*int(sys.argv[2])+'\\n'+TRACE);"
                        "assert t.record['diagnostics']['stderr_truncated'];"
                        "assert len(t.raw_tail)==16384")
                start = time.monotonic()
                proc = subprocess.run([sys.executable, '-B', '-c', code, str(source.parent), str(count)],
                                      capture_output=True, text=True, timeout=5)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertLess(time.monotonic() - start, 5)

    def test_empty_stderr_still_has_context_and_private_file(self):
        record = self.verify_failure('')
        self.assertEqual(self.raw_tail, '')
        self.assertEqual(record['diagnostics']['frames'], [])
        self.assertIn('rank2 remote command failed (exit 17)', self.console)

    def test_rank0_log_is_a_safe_summary_with_private_mode_tail(self):
        secret = 'opaque-rank0-credential'
        process = subprocess.CompletedProcess(['ssh'], 0, secret + '\n', TRACE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rank0.log'
            with patch.object(fleet, 'remote', return_value=process):
                fleet.preserve_rank0_logs({}, {'containers': [{'rank': 0, 'container_id': 'local-fixture'}]}, output)
            self.assertNotIn(secret, output.read_text())
            private, = output.parent.glob('*' + fleet.diagnostics.PRIVATE_SUFFIX)
            self.assertEqual(private.read_text(), secret + '\n' + TRACE)
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
            self.assertIn(private.name, output.read_text())

    def test_private_files_are_unique_and_do_not_follow_prior_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / fleet.diagnostics.private_tail(root / 'verify.json', 'first')
            target = root / 'keep'
            target.write_text('untouched')
            first.unlink()
            first.symlink_to(target)
            second = root / fleet.diagnostics.private_tail(root / 'verify.json', 'second')
            self.assertNotEqual(first, second)
            self.assertEqual(target.read_text(), 'untouched')
            self.assertEqual(second.read_text(), 'second')

    def test_shared_compatibility_helper_is_bounded_and_discards_free_text(self):
        secret = 'opaque-any-encoding'
        text = 'token.' * 32000 + secret + '\n' + TRACE
        shared = fleet.redact_diagnostics(text)
        self.assertNotIn(secret, shared)
        self.assertIn('AssertionError', shared)
        self.assertEqual(fleet.redact_diagnostics(secret), fleet.redact_diagnostics('arbitrary text'))
        known = TRACE.replace('remote_check.py', 'verify_stock.py')
        summary = fleet.redact_diagnostics(known)
        self.assertIn('verify_stock.py:23', summary)
        self.assertEqual(fleet.redact_diagnostics(summary), summary)

    def test_unchecked_remote_returns_original_process(self):
        process = subprocess.CompletedProcess(['ssh'], 4, 'partial', TRACE)
        with (patch.object(fleet, 'ssh_argv', return_value=['ssh']),
              patch.object(fleet.subprocess, 'run', return_value=process)):
            self.assertIs(fleet.remote({}, 0, ['check'], check=False), process)


if __name__ == '__main__':
    unittest.main()
