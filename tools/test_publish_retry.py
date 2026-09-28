#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline lost-acknowledgement and conflict controls for the release assembler."""
import argparse
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PACKAGING = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


class RetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PACKAGING is None:
            raise unittest.SkipTest('requires --packaging-root PACKAGING_DIR')
        cls.package = types.ModuleType('package')
        cls.package.need = need
        cls.package.sha = lambda path: sha(path.read_bytes())
        cls.package.load = lambda path: json.loads(path.read_text())
        cls.package.save = lambda path, data: path.write_text(json.dumps(data, sort_keys=True))
        cls.sdk = types.ModuleType('huggingface_hub')
        cls.sdk.HfApi = cls.sdk.hf_hub_download = object
        cls.sdk.CommitOperationAdd = lambda **kwargs: types.SimpleNamespace(**kwargs)
        cls.modules = {'package': cls.package, 'huggingface_hub': cls.sdk}
        for name in ('publish_checks', 'publish_release', 'hf_upload'):
            spec = importlib.util.spec_from_file_location(name, PACKAGING / 'tools' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, cls.modules):
                spec.loader.exec_module(module)
            cls.modules[name] = module
        cls.checks = cls.modules['publish_checks']
        cls.release = cls.modules['publish_release']
        cls.hf = cls.modules['hf_upload']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'package.json').write_text('{}')
        self.plan = {'branch': 'release/v-test', 'tag': 'v-test', 'commit': 'a' * 40, 'tag_object': 'b' * 40}
        self.ctx = {'out': self.root, 'plan': self.plan, 'body': 'Approved notes\n',
                    'assets': {'recipe.tar.gz': b'approved tar', 'SHA256SUMS': b'approved checksums'}}

    def test_existing_refs_accept_only_exact_approved_objects(self):
        rows = [('a' * 40, 'refs/heads/release/v-test'), ('b' * 40, 'refs/tags/v-test'),
                ('a' * 40, 'refs/tags/v-test^{}')]
        encode = lambda seq: '\n'.join(oid + '\t' + ref for oid, ref in seq)
        for subset in ([], rows[:1], rows[1:], rows):
            self.checks.verify_refs(encode(subset), self.plan)
        self.checks.verify_refs(encode(rows), self.plan, complete=True)
        for bad in (rows[:2], [rows[0], rows[2]], rows + rows[:1], [('c' * 40, rows[0][1])],
                    [('a' * 40, 'refs/heads/unapproved')]):
            with self.assertRaises(ValueError):
                self.checks.verify_refs(encode(bad), self.plan)
        for index in range(3):
            bad = list(rows)
            bad[index] = ('c' * 40, rows[index][1])
            with self.assertRaises(ValueError):
                self.checks.verify_refs(encode(bad), self.plan)

    def github(self, fail=''):
        ctx = self.ctx
        class Fake:
            current = None
            pr = 'OPEN'
            def __init__(self):
                self.events = []
                self.fail = fail
            def event(self, name):
                self.events.append(name)
                if self.fail == name:
                    self.fail = ''
                    raise OSError('lost acknowledgement: ' + name)
            def refs(self, complete=True):
                return None
            def release(self):
                return copy.deepcopy(self.current)
            def before_create(self):
                self.event('before-create')
            def create(self):
                self.event('create-before')
                self.current = {'id': 1, 'tag_name': ctx['plan']['tag'], 'name': 'JSpark3 v-test',
                                'target_commitish': ctx['plan']['commit'], 'prerelease': False,
                                'draft': True, 'body': ctx['body'], 'assets': []}
                self.event('create-after')
            def upload(self, name):
                self.event('upload-before:' + name)
                row = {'id': len(self.current['assets']) + 1, 'name': name, 'state': 'starter', 'size': 0}
                self.current['assets'].append(row)
                self.event('upload-placeholder:' + name)
                row.update(state='uploaded', size=len(ctx['assets'][name]), digest='sha256:' + sha(ctx['assets'][name]))
                self.event('upload-after:' + name)
            def remove_placeholder(self, row):
                self.current['assets'] = [r for r in self.current['assets'] if r['id'] != row['id']]
                self.event('remove-placeholder')
            def asset_bytes(self, row):
                return ctx['assets'][row['name']]
            def publish(self):
                self.event('publish-before')
                self.current['draft'] = False
                self.event('publish-after')
            def pr_state(self):
                return self.pr
            def close_pr(self):
                self.event('close-before')
                self.pr = 'CLOSED'
                self.event('close-after')
        return Fake()

    def test_resume_before_and_after_every_github_mutation(self):
        failures = ['before-create', 'create-before', 'create-after', 'remove-placeholder',
                    'publish-before', 'publish-after', 'close-before', 'close-after']
        failures += [phase + ':' + name for phase in ('upload-before', 'upload-placeholder', 'upload-after')
                     for name in self.ctx['assets']]
        for failure in failures:
            with self.subTest(failure=failure):
                github = self.github(failure)
                if failure == 'remove-placeholder':
                    github.create()
                    github.current['assets'] = [{'id': 1, 'name': 'recipe.tar.gz', 'size': 0, 'state': 'starter'}]
                with self.assertRaises(OSError):
                    self.release.reconcile(self.ctx, github)
                result = self.release.reconcile(self.ctx, github)
                self.assertEqual(result['status'], 'PUBLISHED')
                before = list(github.events)
                self.release.reconcile(self.ctx, github)
                self.assertEqual(github.events, before, 'completed retry must make no writes')
                self.assertEqual(github.current['draft'], False)
                self.assertEqual(github.pr, 'CLOSED')
                self.assertEqual({a['name'] for a in github.current['assets']}, set(self.ctx['assets']))

    def test_conflicting_release_is_refused_without_mutation(self):
        good = self.github()
        self.release.reconcile(self.ctx, good)
        for field, bad in [('body', 'different notes'), ('target_commitish', 'c' * 40),
                           ('name', 'Unapproved'), ('tag_name', 'wrong'), ('prerelease', True)]:
            github = self.github()
            github.current = copy.deepcopy(good.current)
            github.current[field] = bad
            with self.assertRaises(ValueError):
                self.release.reconcile(self.ctx, github)
            self.assertEqual(github.events, [])
        for change in ('digest', 'size', 'extra', 'duplicate', 'starter-published'):
            github = self.github()
            github.current = copy.deepcopy(good.current)
            rows = github.current['assets']
            if change == 'digest': rows[0]['digest'] = 'sha256:' + '0' * 64
            elif change == 'size': rows[0]['size'] += 1
            elif change == 'extra': rows.append({**rows[0], 'name': 'unapproved.bin'})
            elif change == 'duplicate': rows.append(rows[0])
            else: rows[0].update(state='starter', size=0, digest=None)
            with self.assertRaises(ValueError):
                self.release.reconcile(self.ctx, github)
            self.assertEqual(github.events, [])

    def hf_api(self, failure=''):
        class Fake:
            def __init__(self):
                self.head = '1' * 40
                self.states = {'1' * 40: {'README.md': b'old card', 'model.safetensors': b'weights'}}
                self.commits = 0
                self.fail = failure
            def model_info(self, repo, revision=None, files_metadata=False):
                rev = revision or self.head
                data = self.states[rev]
                return types.SimpleNamespace(sha=rev, card_data={'license': 'other', 'license_name': 'shapleymcg-license-1.0'},
                    siblings=[types.SimpleNamespace(rfilename=name, size=len(value), blob_id=sha(value), lfs=None)
                              for name, value in data.items()])
            def create_commit(self, **kwargs):
                need(kwargs['parent_commit'] == self.head, 'CAS refused')
                if self.fail == 'before':
                    self.fail = ''
                    raise OSError('failed before commit')
                data = dict(self.states[self.head])
                data.update({op.path_in_repo: op.path_or_fileobj for op in kwargs['operations']})
                self.head = '2' * 40
                self.states[self.head] = data
                self.commits += 1
                if self.fail == 'after':
                    self.fail = ''
                    raise OSError('lost HF acknowledgement')
                return types.SimpleNamespace(oid=self.head)
            def read(self, rev, name):
                return self.states[rev][name]
        return Fake()

    def hf_payload(self):
        (self.root / 'hf').mkdir(exist_ok=True)
        (self.root / 'hf/README.md').write_bytes(b'approved card')
        (self.root / 'package.json').write_text(json.dumps({'tag': 'v-test', 'hf_files': {'README.md': sha(b'approved card')}}))

    def test_hub_retry_lost_ack_is_one_commit_and_preserves_weights(self):
        self.hf_payload()
        for failure in ('before', 'after'):
            with self.subTest(failure=failure):
                (self.root / 'hf-upload-plan.json').unlink(missing_ok=True)
                api = self.hf_api(failure)
                with self.assertRaises(OSError): self.hf.reconcile(self.root, api, api.read)
                self.hf.reconcile(self.root, api, api.read)
                result = self.hf.reconcile(self.root, api, api.read)
                self.assertEqual(api.commits, 1)
                self.assertEqual(result['action'], 'skip-exact-bytes')
                self.assertEqual(api.states[api.head]['model.safetensors'], b'weights')

    def test_hub_conflicting_payload_or_weights_refused_on_retry(self):
        self.hf_payload()
        for name in ('README.md', 'model.safetensors'):
            (self.root / 'hf-upload-plan.json').unlink(missing_ok=True)
            api = self.hf_api('before')
            with self.assertRaises(OSError): self.hf.reconcile(self.root, api, api.read)
            api.states['3' * 40] = {**api.states['1' * 40], name: b'conflict'}
            api.head = '3' * 40
            with self.assertRaises(ValueError): self.hf.reconcile(self.root, api, api.read)
            self.assertEqual(api.commits, 0)

    def test_hub_shared_plan_hashes_opaque_remote_metadata(self):
        self.hf_payload()
        api = self.hf_api()
        secret = 'opaqueRemoteFilenameCredentialValue'
        api.states[api.head][secret] = b'unrelated remote bytes'
        self.hf.reconcile(self.root, api, api.read)
        self.hf.reconcile(self.root, api, api.read)
        for name in ('hf-upload-plan.json', 'hf-published.json'):
            self.assertNotIn(secret, (self.root / name).read_text())
        self.assertEqual(api.commits, 1)
        self.assertEqual(api.states[api.head][secret], b'unrelated remote bytes')

    def test_actual_preflight_cli_sets_xet_before_sdk_import(self):
        # Execute the real CLI block and imports, replacing only remote/local IO.
        source = ast.parse((PACKAGING / 'tools/publish_checks.py').read_text())
        source.body = [node for node in source.body if not isinstance(node, ast.FunctionDef)]
        fake_release = types.ModuleType('publish_release')
        fake_release.context = lambda *a: {}
        fake_release.GitHub = lambda ctx: types.SimpleNamespace(refs=lambda **kw: None, release=lambda: {'approved': True})
        fake_release.check_release = lambda *a, **kw: None
        sdk = types.ModuleType('huggingface_hub')
        sdk.HfApi = lambda: self.hf_api()
        observed = []
        real_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name == 'huggingface_hub':
                observed.append(os.environ.get('HF_HUB_DISABLE_XET'))
                return sdk
            return real_import(name, *args, **kwargs)
        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DISABLE_XET'}
        namespace = {'__name__': '__main__', 'verify_local': lambda *a: ({'github_remote_baseline': {}}, self.plan)}
        with patch.dict(sys.modules, {**self.modules, 'publish_release': fake_release}), \
                patch.dict(os.environ, env, clear=True), patch.object(builtins, '__import__', checked_import), \
                patch.object(sys, 'argv', ['publish_checks.py', str(self.root), '--checkout', str(self.root), '--go', 'go']):
            exec(compile(source, 'publish_checks.py', 'exec'), namespace)
        self.assertEqual(observed, ['1'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packaging-root', type=Path, required=True)
    args, rest = parser.parse_known_args()
    PACKAGING = args.packaging_root.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
