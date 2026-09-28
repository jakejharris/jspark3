#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline controls against a patched release-packaging checkout.

Usage: python3 -B tools/test_release_tooling.py --packaging-root PACKAGING_DIR
All publication commands are replaced with local recording executables.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PACKAGING = None


class ReleaseToolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PACKAGING is None:
            raise unittest.SkipTest('requires --packaging-root PACKAGING_DIR')
        spec = importlib.util.spec_from_file_location('release_render', PACKAGING / 'tools/render.py')
        cls.render = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.render)
        cls.raw = (Path(__file__).resolve().parents[1] / 'release/results-v1.8.0.json').read_bytes()

    def test_card_license_matches_model_and_rejects_old_metadata(self):
        for variant in ('b', 'c'):
            for comparison in ('strict', 'median'):
                card = self.render.render(self.raw, variant, True, comparison)['hf-card']
                self.assertTrue(card.startswith('---\nlicense: other\n'
                                               'license_name: shapleymcg-license-1.0\n'
                                               'license_link: LICENSE\n'))
                self.render.check_hf_license(card)
                self.assertNotIn('library_name: transformers', card)
                self.assertIn('CC BY-NC-ND 4.0', card)
                self.assertNotIn("Apache-2.0 metadata follows", card)
                for bad in (card.replace('license: other', 'license: apache-2.0'),
                            card.replace('license_link: LICENSE\n', ''),
                            card.replace('shapleymcg-license-1.0', 'apache-2.0')):
                    with self.assertRaisesRegex(ValueError, 'ShapleyMcg'):
                        self.render.check_hf_license(bad)

    def test_public_copy_identifies_quantization_and_coop_without_changing_figures(self):
        content = self.render.render(self.raw, 'b', True)
        for name in ('README.md', 'release/RELEASE-NOTES.md', 'hf-card'):
            text = content[name]
            self.render.check_headline_copy(text, self.raw)
            self.assertIn('unedited EXL3-quantized GLM-5.3 Flash weights', text)
            self.assertIn('`ABLIT=1`', text)
            self.assertIn('`JSPARK3_V16_COOP=1`', text)
            self.assertIn('`JSPARK3_V16_COOP=0`', text)
            self.assertIn('Decode ranges span both sweeps.', text)
            self.assertIn('from one post-hygiene gate pass', text)

    def test_variant_copy_gate_accepts_corrected_wording_and_refuses_missing_mode(self):
        # Keep the real comparison gate; supply only its filesystem helpers.
        package = types.ModuleType('package')
        package.load = lambda path: json.loads(path.read_text())
        package.sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        package.FROZEN_RESULTS_SHA256 = hashlib.sha256(self.raw).hexdigest()
        spec = importlib.util.spec_from_file_location('comparison_under_test',
                                                     PACKAGING / 'tools/comparison_variants.py')
        comparison = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, package=package, render=self.render):
            spec.loader.exec_module(comparison)
        for rule in ('strict', 'median'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                content = self.render.render(self.raw, 'b', True, rule)

                def write(name, data):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data if isinstance(data, bytes) else data.encode())

                for name, key in (('release-body.md', 'release/RELEASE-NOTES.md'),
                                  ('hf/README.md', 'hf-card'), ('github/README.md', 'README.md'),
                                  ('github/release/RELEASE-NOTES.md', 'release/RELEASE-NOTES.md')):
                    write(name, content[key])
                for name in ('github/release/results.json', 'github/release/results-v1.8.0.json',
                             'assets/results-v1.8.0.json', 'hf/jspark3/v1.8.0/results-v1.8.0.json'):
                    write(name, self.raw)
                numbers = (Path(__file__).resolve().parents[1] / 'release/RELEASE-NUMBERS.md').read_bytes()
                for name in ('github/release/RELEASE-NUMBERS.md', 'assets/RELEASE-NUMBERS.md',
                             'hf/jspark3/v1.8.0/RELEASE-NUMBERS.md'):
                    write(name, numbers)
                write('freeze.json', json.dumps({'readable_numbers_sha256': hashlib.sha256(numbers).hexdigest()}))
                write('github/manifests/final-binding.json', '{"hardware_qualified":false}')
                report = comparison.copy_checks(root, rule)
                self.assertEqual(report['passed'], report['total'], report)
                write('hf/README.md', content['hf-card'].replace('`ABLIT=1`', 'edited'))
                report = comparison.copy_checks(root, rule)
                self.assertLess(report['passed'], report['total'])

    def publish(self, fail):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'publish.sh'
            # Relocate the actual script; retain its complete command ordering.
            script.write_text('\n'.join(
                'PACKAGE_ROOT=' + str(root) if line.startswith('PACKAGE_ROOT=') else line
                for line in (PACKAGING / '9AM-COMMANDS.sh').read_text().splitlines()) + '\n')
            stub = '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
event = name
if name == 'python3':
    if any('comparison_variants.py' in x for x in args): event = 'choose'
    elif any('publish_checks.py' in x for x in args): event = 'preflight'
    elif any('hf_upload.py' in x for x in args): event = 'hf-upload'
    elif any('publish_release.py' in x for x in args): event = 'github-reconcile'
    elif 'package.json' in args[-1]: event = 'tag'
    elif 'git-plan.json' in args[-1]: event = 'commit'
    else: raise RuntimeError(args)
elif name == 'git':
    assert 'push' in args and '--atomic' in args, args
    event = 'git-dry-run' if '--dry-run' in args else 'git-push'
elif name == 'gh': event = 'gh-' + '-'.join(args[:2])
with open(os.environ['EVENT_LOG'], 'a') as stream:
    stream.write(json.dumps([event, os.environ.get('HF_HUB_DISABLE_XET')]) + '\\n')
if event == os.environ.get('FAIL_AT'): sys.exit(17)
if event == 'choose': print('strict')
if event == 'tag': print('v1.8.0')
if event == 'commit': print('a' * 40)
'''
            for name in ('python3', 'git', 'gh'):
                executable = root / name
                executable.write_text('#!' + sys.executable + '\n' + stub)
                executable.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DISABLE_XET'}
            env.update(PATH=str(root) + os.pathsep + env['PATH'],
                       EVENT_LOG=str(root / 'events.jsonl'), FAIL_AT=fail)
            process = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            return process, events

    def test_upload_failure_does_not_push_tag_or_create_release(self):
        process, events = self.publish('hf-upload')
        self.assertEqual(process.returncode, 17, process.stderr)
        names = [event[0] for event in events]
        self.assertIn('git-dry-run', names)
        self.assertNotIn('git-push', names)
        self.assertFalse(any(name.startswith('gh-') for name in names))
        self.assertIn(['hf-upload', '1'], events)

    def test_success_uploads_before_atomic_refs_and_release(self):
        process, events = self.publish('')
        self.assertEqual(process.returncode, 0, process.stderr)
        names = [event[0] for event in events]
        self.assertLess(names.index('hf-upload'), names.index('git-push'))
        self.assertLess(names.index('git-push'), names.index('github-reconcile'))
        self.assertIn(['hf-upload', '1'], events)

    def test_preflight_and_dry_run_failures_stop_before_upload(self):
        for fail in ('preflight', 'git-dry-run'):
            process, events = self.publish(fail)
            self.assertEqual(process.returncode, 17, process.stderr)
            self.assertNotIn('hf-upload', [event[0] for event in events])
            self.assertNotIn('git-push', [event[0] for event in events])

    def test_standalone_uploader_disables_xet_before_sdk_import(self):
        # The SDK freezes this environment setting at import time.
        import builtins
        real_import = builtins.__import__
        sdk = types.ModuleType('huggingface_hub')
        sdk.HfApi = sdk.CommitOperationAdd = sdk.hf_hub_download = object
        imported = []

        def checked_import(name, *args, **kwargs):
            if name == 'huggingface_hub':
                imported.append(os.environ.get('HF_HUB_DISABLE_XET'))
                return sdk
            if name == 'package':
                return types.ModuleType('package')
            return real_import(name, *args, **kwargs)

        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DISABLE_XET'}
        with patch.dict(os.environ, env, clear=True), patch.object(builtins, '__import__', checked_import):
            runpy.run_path(str(PACKAGING / 'tools/hf_upload.py'), run_name='upload_under_test')
        self.assertEqual(imported, ['1'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packaging-root', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    PACKAGING = args.packaging_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
