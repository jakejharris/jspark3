#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic offline controls; never creates or pins hardware qualification."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'recipe/scripts'), str(ROOT / 'tools'), str(Path(__file__).parent)]
import _coop_qualification as q
import _image_identity as images
import coop_evidence as evidence
import qualify_coop as runner
from test_operator_image import fixture, write_record


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recipe = self.root / 'recipe'
        shutil.copytree(ROOT / 'recipe/config', self.recipe / 'config')
        self.coop = self.recipe / 'overlays/v16/coop'
        shutil.copytree(ROOT / 'recipe/overlays/v16/coop', self.coop)
        self.bundle = self.coop / 'bundle'
        self.profiles = {f'rank{r}-geo{g}.jsonl': hashed(f'profile {r} {g}') for r in range(3) for g in range(3)}
        policy = {'schema': 1, 'native_sha256': q.TARGET_NATIVE,
                  'profile_log_sha256': self.profiles, 'rows': {str(n): 'stock' for n in range(1, 65)}}
        runner.write(self.bundle / 'dispatch_policy.json', policy)
        shutil.copyfile(self.bundle / 'dispatch_policy.json', self.coop / 'source/dispatch_policy.json')
        manifest = q.read(self.bundle / 'manifest.json')
        manifest['files']['dispatch_policy.json'] = q.sha(self.bundle / 'dispatch_policy.json')
        manifest['files']['cooperative_moe.so'] = q.TARGET_NATIVE
        runner.write(self.bundle / 'manifest.json', manifest)
        image = fixture()
        for key in ('config_digest', 'manifest_digest'):
            image[key] = 'sha256:' + hashed(key)
        image['diff_ids'] = ['sha256:' + hashed('layer')]
        write_record(self.root / 'image.json', image)
        image = images.read_operator_record(self.root / 'image.json')
        checkpoint = {'revision': '25a44fdbf16862a46b7cc9921142c6c81350af2f', 'files': {'index': hashed('checkpoint')}}
        self.index = {'schema_version': 1, 'status': 'PASS', 'native_sha256': q.TARGET_NATIVE,
            'source_manifest_sha256': q.sha(self.coop / 'SOURCE_MANIFEST.json'),
            'image_receipt_sha256': image['payload_sha256'], 'helper': q.read(self.recipe / 'config/coop-helper.json'),
            'sanitizer': q.read(self.recipe / 'config/coop-sanitizer.json'), 'checkpoint': checkpoint,
            'profile_log_sha256': self.profiles,
            'gates': {name: {'kind': kind, 'status': 'PASS', **{key: hashed(name + key) for key in
                ('receipt_sha256', 'log_sha256', 'environment_sha256', 'execution_sha256')}} for name, kind in q.gate_names().items()}}
        runner.write(self.coop / 'QUALIFICATION.json', self.index)
        self.record = {'schema_version': 2, 'source_manifest_sha256': self.index['source_manifest_sha256'],
            'qualification_source_manifest_sha256': self.index['source_manifest_sha256'],
            'compiled_inputs': q.compiled_inputs(self.coop), 'builder_sha256': q.sha(self.coop / 'build_repro.sh'),
            'build_policy_sha256': q.digest_value(images.build_policy()), 'qualification_image': image,
            'helper': self.index['helper'], 'sanitizer': self.index['sanitizer'], 'checkpoint': checkpoint,
            'reproducibility': {'runs': 2, 'comparison': 'bit-identical', 'binary_sha256': q.TARGET_NATIVE,
                                'evidence_sha256': hashed('two build records')},
            'bundle': {'manifest_sha256': q.sha(self.bundle / 'manifest.json'), 'native_sha256': q.TARGET_NATIVE,
                       'runtime_sha256': manifest['files']['runtime.py'], 'dispatch_policy_sha256': manifest['files']['dispatch_policy.json']},
            'gate_index_sha256': q.sha(self.coop / 'QUALIFICATION.json')}
        self.patch = patch.object(q, 'RECIPE', self.recipe)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def pin_fixture(self):
        runner.write(self.recipe / 'config/coop-release.json', {'schema_version': 1, 'status': 'qualified',
            'native_sha256': q.TARGET_NATIVE, 'build_sha256': q.digest_value(self.record),
            'gate_index_sha256': self.record['gate_index_sha256']})

    def verify(self):
        return q.verify_record(self.record, self.bundle, self.coop)

    def test_pending_cannot_become_release_seal(self):
        with self.assertRaisesRegex(ValueError, 'pending or differs'):
            self.verify()
        self.pin_fixture()
        self.assertEqual(self.verify()['native_sha256'], q.TARGET_NATIVE)

    def test_independent_eligible_operator_config_is_accepted(self):
        self.pin_fixture()
        independent = fixture()
        independent['config_digest'] = 'sha256:' + hashed('different config')
        write_record(self.root / 'operator.json', independent)
        identity = q.verify_record(self.record, self.bundle, self.coop, operator=self.root / 'operator.json')
        self.assertEqual(identity['component_seal_sha256'], q.digest_value(self.record))
        independent['build_policy']['dockerfile_sha256'] = hashed('wrong dockerfile')
        write_record(self.root / 'operator.json', independent)
        with self.assertRaisesRegex(ValueError, 'build policy drift'):
            q.verify_record(self.record, self.bundle, self.coop, operator=self.root / 'operator.json')

    def test_missing_gate_profile_and_placeholder_refuse_even_if_repinned(self):
        original = copy.deepcopy(self.index)
        for kind in set(q.gate_names().values()):
            self.index = copy.deepcopy(original)
            name = next(n for n, k in q.gate_names().items() if k == kind)
            del self.index['gates'][name]
            runner.write(self.coop / 'QUALIFICATION.json', self.index)
            self.record['gate_index_sha256'] = q.sha(self.coop / 'QUALIFICATION.json')
            self.pin_fixture()
            with self.assertRaisesRegex(ValueError, 'component gates'):
                self.verify()
        self.index = copy.deepcopy(original)
        self.index['profile_log_sha256']['rank0-geo0.jsonl'] = '0' * 64
        runner.write(self.coop / 'QUALIFICATION.json', self.index)
        self.record['gate_index_sha256'] = q.sha(self.coop / 'QUALIFICATION.json')
        self.pin_fixture()
        with self.assertRaisesRegex(ValueError, 'profile hashes'):
            self.verify()

    def test_wrong_image_source_native_policy_and_tampering(self):
        for field in ('source_manifest_sha256', 'builder_sha256', 'build_policy_sha256', 'gate_index_sha256'):
            with self.subTest(field=field):
                original = self.record[field]
                self.record[field] = hashed('wrong')
                self.pin_fixture()
                with self.assertRaises(ValueError): self.verify()
                self.record[field] = original
        self.pin_fixture()
        for name in ('source/dispatch_policy.json', 'QUALIFICATION.json'):
            path = self.coop / name
            data = path.read_bytes()
            path.write_bytes(data + b' ')
            with self.assertRaises(ValueError): self.verify()
            path.write_bytes(data)
        path = self.coop / 'QUALIFICATION.json'
        path.rename(self.coop / 'saved.json')
        path.symlink_to(self.coop / 'saved.json')
        with self.assertRaisesRegex(ValueError, 'symlink'): self.verify()

    def test_qualification_image_policy_is_not_operator_receipt_override(self):
        self.record['qualification_image']['build_policy']['dockerfile_sha256'] = hashed('wrong')
        image = self.record['qualification_image']
        image['payload_sha256'] = q.digest_value({k:v for k,v in image.items() if k != 'payload_sha256'})
        self.pin_fixture()
        with self.assertRaisesRegex(ValueError, 'build policy drift'): self.verify()


class RunnerTests(unittest.TestCase):
    def test_complete_command_matrix_and_no_serving_mutation(self):
        commands = dict(runner.matrix())
        self.assertEqual(set(commands), set(q.gate_names()) | {'select-policy'})
        self.assertEqual(sum(k == 'sanitizer' for k in q.gate_names().values()), 24)
        for name, command in commands.items():
            text = ' '.join(command)
            self.assertNotIn('fleetctl', text)
            self.assertNotIn('vllm serve', text)
            if name.startswith(('h1-', 'policy-')):
                self.assertNotIn('GLM53_COOP_QUALIFICATION=', text)
                self.assertNotIn('GLM53_COOP_GEOMETRY=', text)
            if name.startswith(('smoke-', 'geometry2-')):
                self.assertIn('--error-exitcode 9', text)
                self.assertIn('--dump-kernel-launches', text)
                self.assertIn(evidence.KERNEL_FILTER, text)

    def test_instrumentation_and_error_controls(self):
        launch = '========= Launch #1\n=========   Kernel: exl3_moe_coop_a_kernel()\n'
        for tool, summary in [('memcheck', 'ERROR SUMMARY: 0 errors'), ('racecheck', 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')]:
            log = launch + '========= ' + summary + '\n'
            self.assertEqual(evidence.sanitizer(log, tool), 1)
            for bad in (summary, log.replace('Launch #1', 'Launch #1 (filtered: kernel name)'),
                        log.replace('exl3_moe_coop_', 'stock_'), log + 'Internal Sanitizer Error'):
                with self.assertRaises(ValueError): evidence.sanitizer(bad, tool)

    def test_missing_and_symlinked_raw_campaign_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runner.write(root / 'campaign.json', {'schema_version': 1, 'status': 'INCOMPLETE'})
            with self.assertRaisesRegex(ValueError, 'campaign incomplete'): evidence.validate_campaign(root)
            (root / 'campaign.json').unlink()
            (root / 'campaign.json').symlink_to(root / 'missing')
            with self.assertRaises(ValueError): evidence.validate_campaign(root)


if __name__ == '__main__':
    unittest.main()
