#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real clone/export privacy regressions; build execution stops at a Docker stub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


class ReleasePrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        origin = cls.work / 'origin'
        origin.mkdir()
        # Use the shipped inventory, so this test also works from a source tarball.
        names = [line.split('  ', 1)[1] for line in (ROOT / 'SHA256SUMS').read_text().splitlines()]
        for name in [*names, 'SHA256SUMS']:
            target = origin / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

        def git(*args):
            return subprocess.run(['git', '-C', str(origin), '-c', 'user.name=Release Test',
                '-c', 'user.email=release@example.invalid', '-c', 'commit.gpgsign=false',
                '-c', 'core.hooksPath=/dev/null', *args], check=True, capture_output=True, text=True).stdout.strip()

        git('init', '-b', 'main')
        git('add', '.')
        git('commit', '-m', 'Source export')
        git('tag', 'release-test')
        git('checkout', '-b', 'legacy-docs')
        cls.private_message = 'spark' + '1 historical note'
        git('commit', '--allow-empty', '-m', cls.private_message)
        cls.unrelated = git('rev-parse', 'HEAD')
        git('checkout', 'main')
        cls.clone = cls.work / 'clone'
        subprocess.run(['git', 'clone', '--no-local', '--branch', 'release-test', str(origin), str(cls.clone)],
                       check=True, capture_output=True, text=True)

    def private_diagnostic(self, console):
        name = console.split('Private diagnostics (do not share): ', 1)[1].splitlines()[0]
        return Path(name).read_text()

    def validate(self, root):
        report = self.work / 'validation.json'
        proc = subprocess.run([sys.executable, '-B', str(root / 'tools/validate_release.py'), str(root),
                               '--report', str(report)], capture_output=True, text=True)
        return proc, json.loads(report.read_text())

    def test_full_clone_ignores_unrelated_history_and_unblocks_build_entrypoints(self):
        log = subprocess.check_output(['git', '-C', str(self.clone), 'log', '--all', '--format=%B'], text=True)
        self.assertIn(self.private_message, log)
        proc = subprocess.run(['git', '-C', str(self.clone), 'merge-base', '--is-ancestor',
                               self.unrelated, 'HEAD'], capture_output=True)
        self.assertEqual(proc.returncode, 1)  # Ref exists in the clone, outside the checked-out ancestry.
        proc, report = self.validate(self.clone)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(report['failed'], 0)
        self.assertEqual(len(report['checks']), 18)

        executables = self.work / 'bin'
        executables.mkdir()
        docker = executables / 'docker'
        docker.write_text('#!/bin/sh\necho IMAGE_BUILD_REACHED >&2\nexit 73\n')
        docker.chmod(0o755)
        proc = subprocess.run([sys.executable, '-B', str(self.clone / 'tools/build_operator_image.py'),
                               '--output', str(self.work / 'image.json')], capture_output=True, text=True,
                              env={**os.environ, 'PATH': str(executables) + os.pathsep + os.environ['PATH']})
        self.assertEqual(proc.returncode, 9)
        self.assertNotIn('IMAGE_BUILD_REACHED', proc.stderr)
        private = self.private_diagnostic(proc.stderr)
        child = private.split('Complete private child output: ', 1)[1].splitlines()[0]
        self.assertIn('IMAGE_BUILD_REACHED', Path(child).read_text())
        self.assertNotIn('source export failed validation', proc.stderr)
        self.assertFalse((self.work / 'image.json').exists())
        proc = subprocess.run([sys.executable, '-B', str(self.clone / 'tools/prepare_runtime.py'),
                               '--binary-root', str(self.work / 'absent-binaries'),
                               '--output', str(self.work / 'runtime')], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('missing or mismatched local build', self.private_diagnostic(proc.stderr))
        self.assertNotIn('source export failed validation', proc.stderr)
        self.assertFalse((self.work / 'runtime').exists())

    def test_pending_component_candidate_refuses_final_validation(self):
        proc = subprocess.run([sys.executable, '-B', str(self.clone / 'tools/validate_release.py'),
                               str(self.clone), '--require-final'], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('FAIL release-manifest', proc.stdout)

    def test_export_and_checkout_still_reject_private_file_bytes_and_names(self):
        export = self.work / 'export'
        shutil.copytree(self.clone, export, ignore=shutil.ignore_patterns('.git'))
        secret = 'hf' + '_' + 'x' * 32
        for root in (self.clone, export):
            for name, data in (('README.md', secret), ('untracked.txt', secret),
                               ('spark' + '1-notes.txt', 'ordinary text')):
                with self.subTest(tree=root.name, path=name):
                    path = root / name
                    original = path.read_bytes() if path.exists() else None
                    try:
                        path.write_text(data)
                        proc, report = self.validate(root)
                        self.assertNotEqual(proc.returncode, 0)
                        privacy = next(row for row in report['checks'] if row['check'] == 'privacy-scan')
                        self.assertEqual(privacy['status'], 'FAIL')
                        self.assertIn(name, Path(privacy['private_diagnostic']).read_text())
                        self.assertNotIn(secret, proc.stdout + proc.stderr)
                    finally:
                        if original is None:
                            path.unlink()
                        else:
                            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
