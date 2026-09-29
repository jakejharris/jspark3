#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic offline controls; never creates or pins hardware qualification."""
import copy
import contextlib
import io
import subprocess
import hashlib
import json
import os
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
import _coop_checkpoint as checkpoint
import _coop_bundle as bundles
import coop_evidence as evidence
import qualify_coop as runner
from test_operator_image import fixture, write_record


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def host_identity(board):
    return {'architecture': 'aarch64', 'machine_id_sha256': hashed('cloned OS'),
            'physical_identity': {'kind': 'gb10-gpu-uuid', 'uuid_sha256': hashed(board)}}


def profile_rows(rank, geometry, native_hash):
    rows = [{'stage':'profile_identity','native_sha256':native_hash}]
    for n in evidence.select_policy.ROWS:
        for pattern in evidence.select_policy.PATTERNS:
            ratio = 1.1 if n == 64 else .95 + .01 * geometry
            rows += [{'stage':'compare','label':f'profile/rank={rank}/geometry={geometry}/rows={n}/{pattern}',
                      'pass':True,'peak_rel':.001,'row_rel':.01,'rel_l2':.01},
                     {'stage':'profile','rank':rank,'geometry':geometry,'rows':n,'pattern':pattern,
                      'stock':{'min_ms':1.,'median_ms':1.,'max_ms':1.},
                      'candidate':{'min_ms':ratio,'median_ms':ratio,'max_ms':ratio}}]
    return rows + [{'stage':'profile_complete','rank':rank,'geometry':geometry}]


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recipe = self.root / 'recipe'
        shutil.copytree(ROOT / 'recipe', self.recipe)
        self.native_bytes = b'\x7fELF synthetic native fixture, never GPU evidence'
        native_patch = patch.object(q, 'TARGET_NATIVE', hashlib.sha256(self.native_bytes).hexdigest())
        native_patch.start()
        self.addCleanup(native_patch.stop)
        self.coop = self.recipe / 'overlays/v16/coop'
        self.bundle = self.coop / 'bundle'
        self.profiles = {f'rank{r}-geo{g}.jsonl': hashed(f'profile {r} {g}') for r in range(3) for g in range(3)}
        policy = {'schema': 1, 'native_sha256': q.TARGET_NATIVE,
                  'profile_log_sha256': [self.profiles[n] for n in sorted(self.profiles)], 'rows': {str(n): 'stock' for n in range(1, 65)}}
        runner.write(self.bundle / 'dispatch_policy.json', policy)
        shutil.copyfile(self.bundle / 'dispatch_policy.json', self.coop / 'source/dispatch_policy.json')
        source_manifest = q.read(self.coop / 'SOURCE_MANIFEST.json')
        source_manifest['files']['dispatch_policy.json'] = q.sha(self.coop / 'source/dispatch_policy.json')
        runner.write(self.coop / 'SOURCE_MANIFEST.json', source_manifest)
        source_pin = patch('apply_coop_moe.SOURCE_MANIFEST_SHA256', q.sha(self.coop / 'SOURCE_MANIFEST.json'))
        source_pin.start()
        self.addCleanup(source_pin.stop)
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
        checkpoint_pin = checkpoint.authority()
        self.index = {'schema_version': 1, 'status': 'PASS', 'native_sha256': q.TARGET_NATIVE,
            'source_manifest_sha256': q.sha(self.coop / 'SOURCE_MANIFEST.json'),
            'image_receipt_sha256': image['payload_sha256'], 'helper': q.read(self.recipe / 'config/coop-helper.json'),
            'sanitizer': q.read(self.recipe / 'config/coop-sanitizer.json'), 'checkpoint': checkpoint_pin,
            'raw_bundle_manifest': copy.deepcopy(manifest), 'raw_bundle': bundles.identity(self.bundle),
            'selected_bundle': bundles.identity(self.bundle),
            'profile_log_sha256': self.profiles,
            'gates': {name: {'kind': kind, 'status': 'PASS', 'bundle':bundles.identity(self.bundle), **{key: hashed(name + key) for key in
                ('receipt_sha256', 'log_sha256', 'environment_sha256', 'execution_sha256')}} for name, kind in q.gate_names().items()}}
        second = {'schema_version': 2, 'verification': 'fixed-native-build-v2', 'image_receipt_sha256': image['payload_sha256'],
                  'build_inputs':{**{runner.native.COOP + '/' + name:value for name,value in q.compiled_inputs(self.coop).items()},
                                  runner.native.COOP + '/SOURCE_MANIFEST.json':q.sha(self.coop / 'SOURCE_MANIFEST.json'),
                                  runner.native.COOP + '/source/runtime.py':q.sha(self.coop / 'source/runtime.py'),
                                  runner.native.COOP + '/source/dispatch_policy.json':q.sha(self.coop / 'source/dispatch_policy.json')},
                  'source_recipe_sha256': image['source_recipe_sha256'],
                  'reproducibility': {'runs': 2, 'comparison': 'bit-identical'}, 'hardware_qualified': False,
                  'binary_sha256': {'recipe/overlays/v16/coop/bundle/cooperative_moe.so': q.TARGET_NATIVE},
                  'builder_host': host_identity('second')}
        second['payload_sha256'] = q.digest_value(second)
        self.index['independent_build'] = {'native_receipt': second, 'image_receipt': image}
        first = copy.deepcopy(second)
        first['builder_host'] = host_identity('first')
        first['payload_sha256'] = q.digest_value({k:v for k,v in first.items() if k != 'payload_sha256'})
        self.index['qualification_build'] = first
        runner.write(self.coop / 'QUALIFICATION.json', self.index)
        self.record = {'schema_version': 2, 'source_manifest_sha256': self.index['source_manifest_sha256'],
            'qualification_source_manifest_sha256': self.index['source_manifest_sha256'],
            'compiled_inputs': q.compiled_inputs(self.coop), 'builder_sha256': q.sha(self.coop / 'build_repro.sh'),
            'build_policy_sha256': q.digest_value(images.build_policy()), 'qualification_image': image,
            'helper': self.index['helper'], 'sanitizer': self.index['sanitizer'], 'checkpoint': checkpoint_pin,
            'reproducibility': {'runs': 2, 'comparison': 'bit-identical', 'binary_sha256': q.TARGET_NATIVE,
                                'evidence_sha256': q.digest_value(first)},
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

    def test_seal_refuses_same_board_even_with_different_os_identity(self):
        first = self.index['qualification_build']['builder_host']
        second = self.index['independent_build']['native_receipt']
        second['builder_host']['physical_identity'] = copy.deepcopy(first['physical_identity'])
        second['builder_host']['machine_id_sha256'] = hashed('new OS identity')
        second['payload_sha256'] = q.digest_value({k:v for k,v in second.items() if k != 'payload_sha256'})
        runner.write(self.coop / 'QUALIFICATION.json', self.index)
        self.record['gate_index_sha256'] = q.sha(self.coop / 'QUALIFICATION.json')
        self.pin_fixture()
        with self.assertRaisesRegex(ValueError, 'second physical machine'):
            self.verify()

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

    def test_checkpoint_and_original_bundle_cannot_be_reminted(self):
        self.record['checkpoint'] = copy.deepcopy(self.record['checkpoint'])
        self.record['checkpoint']['files']['model-00057-of-00120.safetensors'] = hashed('substituted shard')
        with self.assertRaisesRegex(ValueError, 'checkpoint authority'):
            q.verify_record(self.record, self.bundle, self.coop, release=False)
        self.record['checkpoint'] = checkpoint.authority()
        for rank in range(3):
            value = copy.deepcopy(self.index)
            value['gates'][f'policy-r{rank}']['bundle']['runtime_sha256'] = hashed('other adapter')
            runner.write(self.coop / 'QUALIFICATION.json', value)
            self.record['gate_index_sha256'] = q.sha(self.coop / 'QUALIFICATION.json')
            with self.assertRaisesRegex(ValueError, 'incomplete gate'):
                q.verify_record(self.record, self.bundle, self.coop, release=False)

    def test_copied_source_and_builder_cannot_define_their_own_pins(self):
        for name in ('source/runtime.py', 'build_repro.sh'):
            other = self.root / ('altered-' + name.replace('/', '-'))
            shutil.copytree(self.coop, other)
            (other / name).write_bytes((other / name).read_bytes() + b'\n# substituted\n')
            manifest = q.read(other / 'SOURCE_MANIFEST.json')
            manifest['files']['runtime.py'] = q.sha(other / 'source/runtime.py')
            runner.write(other / 'SOURCE_MANIFEST.json', manifest)
            record = {**self.record, 'source_manifest_sha256':q.sha(other / 'SOURCE_MANIFEST.json'),
                      'builder_sha256':q.sha(other / 'build_repro.sh'), 'compiled_inputs':q.compiled_inputs(other)}
            with self.assertRaisesRegex(ValueError, 'trusted recipe'):
                q.verify_record(record, other / 'bundle', other, release=False)

    def test_complete_synthetic_campaign_seals_without_mutating_raw_evidence(self):
        import argparse
        shutil.copytree(ROOT / 'tools', self.root / 'tools')
        for obj, key, value in ((runner, 'COOP', self.coop), (runner.native, 'ROOT', self.root), (evidence, 'ROOT', self.root)):
            control = patch.object(obj, key, value)
            control.start()
            self.addCleanup(control.stop)
        raw = self.root / 'campaign'
        raw.mkdir()
        (raw / 'profiles').mkdir()
        (self.bundle / 'cooperative_moe.so').write_bytes(self.native_bytes)
        shutil.copytree(self.bundle, raw / 'raw-bundle')
        shutil.copytree(self.bundle, raw / 'selected-bundle')
        logs = []
        for r in range(3):
            for g in range(3):
                path = raw / f'profiles/rank{r}-geo{g}.jsonl'
                path.write_text(runner.projection.gate_log(f'profile-r{r}-g{g}',
                    ''.join(json.dumps(row) + '\n' for row in profile_rows(r, g, q.TARGET_NATIVE))))
                logs.append(path)
        policy = evidence.select_policy.select(logs, q.TARGET_NATIVE)
        runner.write(raw / 'selected-bundle/dispatch_policy.json', policy)
        manifest = q.read(raw / 'selected-bundle/manifest.json')
        manifest['files']['dispatch_policy.json'] = q.sha(raw / 'selected-bundle/dispatch_policy.json')
        runner.write(raw / 'selected-bundle/manifest.json', manifest)
        identity = {k:self.index[k] for k in ('native_sha256','source_manifest_sha256','image_receipt_sha256',
                                             'helper','checkpoint','sanitizer')}
        identity.update(raw_bundle=bundles.identity(raw / 'raw-bundle'),
                        raw_bundle_manifest=q.read(raw / 'raw-bundle/manifest.json'))
        identity['runner_sha256'] = runner.runner_inputs()
        image = self.record['qualification_image']
        runner.write(raw / 'image.json', image)
        args = argparse.Namespace(fly_root=self.root / 'fly', helpers_root=self.root / 'helpers',
                                  model_root=self.root / 'model', sanitizer_root=self.root / 'sanitizer')
        plan = {name:runner.container(args, raw, raw / ('container-' + name), image, cmd, True)
                for name, cmd in runner.matrix()}
        runner.write(raw / 'plan.json', plan)
        env = {'status':'PASS','exl3_sha256':'71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174',
               'checkpoint':checkpoint.authority(),
               'fatpath_sha256':'69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e',
               'ldd':'libcudart.so.13 => synthetic','nvcc':'fixture','gcc':'fixture','sanitizer':'fixture',
               'torch':'fixture','cuda':'fixture','gpu':{'name':'NVIDIA GB10','capability':[12,1],'driver':'fixture'}}
        env = runner.projection.environment({**env, 'bundle': {}})
        for name, kind in q.gate_names().items():
            stage = raw / ('container-' + name)
            stage.mkdir()
            binding = bundles.identity(raw / ('selected-bundle' if kind == 'policy' else 'raw-bundle'))
            runner.write(stage / 'environment.json', {**env, 'bundle':binding})
            runner.write(stage / 'execution.json', {'exit_code':0,'command':plan[name],'started_at':'1','completed_at':'2'})
            rank = int(name.split('-')[1][1:])
            if kind == 'profile':
                geometry = int(name[-1])
                text = (raw / f'profiles/rank{rank}-geo{geometry}.jsonl').read_text()
            elif kind == 'h1':
                geometry, mode = int(name.split('-')[2][1:]), name.split('-')[3]
                proof = dict(schema_version=1, control='coop-h1-raw-output', mode=mode, status='PASS', ep_rank=rank,
                    qualification_geometry=geometry, native_sha256=q.TARGET_NATIVE,
                    test_source_sha256=q.sha(ROOT / 'recipe/overlays/v16/coop/source/test_cuda_integration.py'),
                    bundle_manifest_sha256=q.sha(raw / 'raw-bundle/manifest.json'),
                    policy_sha256=q.sha(raw / 'raw-bundle/dispatch_policy.json'),
                    control_sha256=q.sha(ROOT / 'tools/v16/coop_h1_control.py'),
                    exl3_sha256=env['exl3_sha256'], fatpath_sha256=env['fatpath_sha256'],
                    thresholds={'peak':.003,'row_peak':.05,'rel_l2':.05},real_weight_comparisons=156,
                    expected_real_weight_comparisons=156,detector_rejections=0 if mode == 'baseline' else 156,
                    detector_outcome='unmodified-pass' if mode == 'baseline' else 'expected-peak-tolerance-failure-observed',
                    minimum_observed_peak_rel=.1,serving_path_reachable=False)
                proof['payload_sha256'] = q.digest_value(proof)
                runner.write(raw / (name + '-proof.json'), proof)
                text = json.dumps(proof) + '\n'
            elif kind == 'sanitizer':
                geometry2 = name.startswith('geometry2')
                done = dict(stage='geometry2_sanitizer_complete' if geometry2 else 'race_smoke_complete', rank=rank)
                done['pass'] = True
                if geometry2:
                    done.update(rows=[2,3,4,5,8,10,12,16,18,20,24,28,32,64],graph_replays=87,host_launches=1)
                summary = 'ERROR SUMMARY: 0 errors' if name.endswith('memcheck') else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                text = json.dumps(done) + '\n========= Launch #1\n=========   Kernel: exl3_moe_coop_fixture()\n========= ' + summary + '\n'
            else:
                text = json.dumps({'stage':'production_policy_complete','pass':True,'choices':
                    {str(n):policy['rows'].get(str(n),'stock') for n in (*evidence.select_policy.ROWS,33,65)}}) + '\n'
            text = runner.projection.gate_log(name, text)
            (raw / (name + '.log')).write_text(text)
            runner.write(raw / (name + '.json'), {'schema_version':1,'name':name,'kind':kind,'status':'PASS',
                         'exit_code':0,'identity':identity,'bundle':binding,'log_sha256':q.sha(raw / (name + '.log'))})
        runner.write(raw / 'campaign.json', {'schema_version':1,'status':'COMPLETE','identity':identity})
        second = self.root / 'second'
        binary = second / runner.native.COOP / 'bundle/cooperative_moe.so'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(self.native_bytes)
        for receipt, host in ((raw / 'native-build.json','first'),(second / 'native-build-receipt.json','second')):
            build = {**self.index['qualification_build'], 'schema_version':2,'verification':'fixed-native-build-v2',
                     'build_inputs':runner.native.build_inputs(),
                     'builder_host':host_identity(host)}
            build['binary_sha256'] = {**{name:hashed(name) for name in runner.native.OUTPUTS['display'].values()},
                                      runner.native.COOP + '/bundle/cooperative_moe.so':q.TARGET_NATIVE}
            build.pop('payload_sha256')
            write_record(receipt, build)
        args = argparse.Namespace(seal=raw,output=self.root / 'sealed',independent_build_root=second,
                                  independent_image_receipt=self.root / 'image.json')
        runner.write(raw / 'independent-build.json', {'native_receipt': q.read(second / 'native-build-receipt.json'), 'image_receipt': image})
        before = {p.relative_to(raw):q.sha(p) for p in raw.rglob('*') if p.is_file()}
        with patch.object(runner,'TARGET_NATIVE',q.TARGET_NATIVE), patch.object(evidence,'TARGET_NATIVE',q.TARGET_NATIVE), \
                patch.object(runner.native,'TARGET_NATIVE',q.TARGET_NATIVE), contextlib.redirect_stdout(io.StringIO()):
            runner.seal(args)
            record = q.read(args.output / 'BUILD.json')
            q.verify_record(record,args.output / 'bundle',args.output,release=False)
            # Review P2: mutate only the selected copy after every gate, then
            # repair its own manifest. Neither campaign validation nor sealing
            # may adopt the replacement, even with internally consistent hashes.
            selected = raw / 'selected-bundle'
            original_manifest = (selected / 'manifest.json').read_bytes()
            for relative in ('runtime.py', 'headers/compat.cuh', 'toolchain.txt'):
                path = selected / relative
                original = path.read_bytes()
                path.write_bytes(original + b'\n# post-gate mutation\n')
                changed = q.read(selected / 'manifest.json')
                changed['files'][relative] = q.sha(path)
                runner.write(selected / 'manifest.json', changed)
                with self.assertRaises(ValueError): evidence.validate_campaign(raw)
                with self.assertRaises(ValueError): runner.seal(argparse.Namespace(**{**vars(args),'output':self.root / 'bad-seal'}))
                self.assertFalse((self.root / 'bad-seal').exists())
                path.write_bytes(original)
                (selected / 'manifest.json').write_bytes(original_manifest)
            for rank in range(3):
                path = raw / f'container-policy-r{rank}/environment.json'
                original = path.read_bytes()
                env_changed = q.read(path)
                env_changed['bundle']['manifest_sha256'] = hashed('other policy-tested bundle')
                runner.write(path, env_changed)
                with self.assertRaisesRegex(ValueError, 'different bundle'): evidence.validate_campaign(raw)
                path.write_bytes(original)
            name = 'policy-r0'
            saved = copy.deepcopy(plan[name])
            plan[name][plan[name].index(image['config_digest'])] = 'sha256:' + hashed('wrong image')
            runner.write(raw / 'plan.json', plan)
            stage = raw / ('container-' + name)
            execution = q.read(stage / 'execution.json')
            execution['command'] = plan[name]
            runner.write(stage / 'execution.json', execution)
            with self.assertRaisesRegex(ValueError, 'qualification image differs'): evidence.validate_campaign(raw)
            plan[name] = saved
            execution['command'] = saved
            runner.write(raw / 'plan.json', plan)
            runner.write(stage / 'execution.json', execution)
            (raw / 'smoke-r0-g0-memcheck.json').unlink()
            with self.assertRaises(ValueError): evidence.validate_campaign(raw)
        self.assertEqual({p.relative_to(raw):q.sha(p) for p in raw.rglob('*') if p.is_file()},
                         {name:digest for name,digest in before.items() if str(name) != 'smoke-r0-g0-memcheck.json'})

    def test_complete_sealed_bundle_prepares_coop_on_and_preserves_inventory(self):
        import apply_coop_moe as coop
        import build_native as native
        import prepare_runtime as prepare
        # Synthetic source authority is confined to this temporary fixture.
        # Production hashes, source tree and pending release pin are unchanged.
        source_manifest = q.read(self.coop / 'SOURCE_MANIFEST.json')
        source_manifest['files']['dispatch_policy.json'] = q.sha(self.coop / 'source/dispatch_policy.json')
        runner.write(self.coop / 'SOURCE_MANIFEST.json', source_manifest)
        self.record['source_manifest_sha256'] = q.sha(self.coop / 'SOURCE_MANIFEST.json')
        runner.write(self.coop / 'BUILD.json', self.record)
        self.pin_fixture()
        for name in ('SHA256SUMS', 'manifests/binaries.json', 'tools/build_native.py'):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        binaries = self.root / 'built'
        runtime_directory = tempfile.TemporaryDirectory()
        self.addCleanup(runtime_directory.cleanup)
        output = Path(runtime_directory.name) / 'prepared'
        outputs = {}
        for kind in native.OUTPUTS.values():
            for name in kind.values():
                dest = binaries / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(self.native_bytes)
                outputs[name] = q.sha(dest)
        image = images.read_operator_record(self.root / 'image.json')
        with contextlib.ExitStack() as stack:
            for module, name, value in ((native, 'ROOT', self.root), (native, 'TARGET_NATIVE', q.TARGET_NATIVE),
                    (prepare, '__file__', str(self.root / 'tools/prepare_runtime.py')),
                    (coop, 'OVERLAY', self.coop), (coop, 'SOURCE_ROOT', self.coop / 'source'),
                    (coop, 'SOURCE_MANIFEST', self.coop / 'SOURCE_MANIFEST.json'),
                    (coop, 'SOURCE_MANIFEST_SHA256', self.record['source_manifest_sha256']),
                    (coop, 'DEFAULT_BUNDLE', self.bundle), (coop, 'DEFAULT_BUILD_RECORD', self.coop / 'BUILD.json')):
                stack.enter_context(patch.object(module, name, value))
            build = {'schema_version': 2, 'verification': 'fixed-native-build-v2',
                     'source_recipe_sha256': image['source_recipe_sha256'], 'image_receipt_sha256': image['payload_sha256'],
                     'build_inputs': native.build_inputs(), 'builder_host': host_identity('local fixture'),
                     'reproducibility': {'runs': 2, 'comparison': 'bit-identical'}, 'binary_sha256': outputs,
                     'hardware_qualified': False}
            write_record(self.root / 'native.json', build)
            argv = ['prepare_runtime.py', '--binary-root', str(binaries), '--native-receipt', str(self.root / 'native.json'),
                    '--image-receipt', str(self.root / 'image.json'), '--output', str(output)]
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch('validate_release.verify', return_value={'failed': 0}))
            stack.enter_context(patch.object(images, 'verify_local_image'))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.main()
        receipt = q.read(output / 'runtime-build-receipt.json')
        self.assertIs(receipt['coop_hardware_seal_required'], False)
        self.assertIs(receipt['hardware_qualified'], False)
        self.assertEqual(receipt['component_qualification']['component_seal_sha256'], q.digest_value(self.record))
        self.assertIn('JSPARK3_V16_COOP=1\n', (output / 'recipe/.env.example').read_text())
        self.assertEqual((output / native.COOP / 'bundle/cooperative_moe.so').read_bytes(), self.native_bytes)
        proc = subprocess.run([sys.executable, str(output / 'recipe/scripts/remote_preflight.py'), '--recipe-only',
                               '--recipe-root', str(output / 'recipe')], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse(list(output.rglob('__pycache__')))


class RunnerTests(unittest.TestCase):
    def test_duplicate_or_missing_builder_refuses_before_any_container(self):
        import argparse
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'image.json'
            write_record(image_path, fixture())
            image = images.read_operator_record(image_path)
            build = {'builder_host': host_identity('same board')}
            args = argparse.Namespace(output=root / 'campaign', check_only=True,
                independent_build_root=root, independent_image_receipt=image_path)
            (root / 'native-build-receipt.json').write_text('{}')
            with patch.object(runner, 'inputs', return_value=(image, build, {}, {})), \
                    patch.object(runner.native, 'read_native_record', return_value=build), \
                    patch.object(runner, 'run_container') as execute:
                for check_only in (True, False):
                    args.check_only = check_only
                    with self.assertRaisesRegex(ValueError, 'second physical machine'):
                        runner.campaign(args)
                args.independent_build_root = None
                with self.assertRaisesRegex(ValueError, 'second-machine build'):
                    runner.campaign(args)
                execute.assert_not_called()
                self.assertFalse(args.output.exists())
                self.assertFalse(Path(str(args.output) + '.check').exists())

    def test_legacy_seal_cannot_be_reminted_with_placeholder_profiles(self):
        # The release tree carries the integrated schema-2 seal; start from the
        # pre-integration legacy record explicitly.
        import test_prepared_fixture as prepared_fixture
        record = prepared_fixture.legacy_build_record()
        q.verify_legacy_record(record)
        record['qualification']['profile_log_sha256']['rank0-geo0.jsonl'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'legacy seals cannot be reminted'):
            q.verify_legacy_record(record)
        with self.assertRaisesRegex(ValueError, 'legacy seals cannot be reminted'):
            q.verify_legacy_record(q.read(ROOT / 'recipe/overlays/v16/coop/BUILD.json'))

    def test_serving_refuses_every_maintenance_override(self):
        import _fleetctl as fleet
        values = fleet.load_env(ROOT / 'recipe/.env.example')
        for name in ('GLM53_COOP_QUALIFICATION', 'GLM53_COOP_GEOMETRY', 'GLM53_COOP_SANITIZER',
                     'GLM53_COOP_EP_RANK', 'GLM53_COOP_TEST_HELPERS', 'GLM53_COOP_BUNDLE',
                     'GLM53_COOP_MAINTENANCE_TEST', 'JSPARK3_V16_COOP_MAINTENANCE'):
            with self.subTest(name=name):
                with patch.dict(os.environ, {name: '0'}):
                    with self.assertRaisesRegex(fleet.Refusal, 'TP3 override'): fleet.validate_env(values)
                with self.assertRaisesRegex(fleet.Refusal, 'TP3 override'): fleet.validate_env({**values, name: '0'})

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
                self.assertEqual(command[4:8], ['python3', '-B', '/src/tools/v16/coop_sanitizer.py', name.rsplit('-', 1)[1]])

    def test_check_only_validates_plan_without_exposing_gpu_or_sealing(self):
        import argparse
        import types
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_root = root / 'build'
            for run in ('a', 'b'):
                stage = build_root / ('coop-' + run) / 'out'
                shutil.copytree(ROOT / 'recipe/overlays/v16/coop/bundle', stage)
                for name in ('cooperative_moe.so', 'build32.log', 'build64.log', 'link.log'):
                    (stage / name).write_text('synthetic check-only fixture')
            for name in runner.native.OUTPUTS['display'].values():
                path = build_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic display fixture')
            args = argparse.Namespace(output=root / 'qualification', check_only=True, build_root=build_root,
                fly_root=root / 'fly', helpers_root=root / 'helpers', model_root=root / 'snapshot',
                sanitizer_root=root / 'sanitizer')
            image = {'config_digest': 'sha256:' + hashed('image'), 'payload_sha256': hashed('receipt')}
            calls = []
            def execute(command, stage, log):
                calls.append(command)
                log.write('opaqueCampaignDiagnosticValue\n')
                runner.write(stage / 'environment.json', runner.projection.environment({'status': 'PASS', 'bundle': {},
                    'checkpoint':checkpoint.authority(),
                    'exl3_sha256': '71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174',
                    'fatpath_sha256': '69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e',
                    'ldd': 'libcudart.so.13 => synthetic', 'torch': 'fixture', 'cuda': 'fixture',
                    'nvcc': 'fixture', 'gcc': 'fixture', 'sanitizer': 'fixture'}))
                return types.SimpleNamespace(returncode=0)
            with patch.object(runner, 'inputs', return_value=(image, {}, {}, {})), \
                    patch.object(runner.native, 'verify_local_image'), patch.object(runner, 'independent_build', return_value={}), \
                    patch.object(runner, 'run_container', side_effect=execute), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.campaign(args)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('--gpus', calls[0])
            self.assertEqual(calls[0][calls[0].index('--memory') + 1], '4g')
            self.assertEqual(calls[0][calls[0].index('--memory-swap') + 1], '4g')
            self.assertIn('NVIDIA_VISIBLE_DEVICES=void', calls[0])
            self.assertIn('CUDA_VISIBLE_DEVICES=', calls[0])
            check = Path(str(args.output) + '.check')
            plan = q.read(check / 'plan.json')
            self.assertEqual(set(plan), set(q.gate_names()) | {'environment', 'select-policy'})
            self.assertTrue(all('--gpus' in command for name, command in plan.items() if name != 'environment'))
            estimates = q.read(check / 'plan-estimates.json')
            self.assertEqual(set(estimates['steps']), set(plan))
            self.assertEqual(estimates['confidence'], 'low')
            self.assertEqual(estimates['gpu_window_seconds'], [5417, 14430])
            self.assertTrue(all(command[command.index('--memory') + 1] == '16g' for name, command in plan.items() if name != 'environment'))
            self.assertFalse(args.output.exists())
            self.assertFalse((check / 'QUALIFICATION.json').exists())
            self.assertEqual(q.read(check / 'campaign.json')['status'], 'INCOMPLETE')
            self.assertNotIn('opaqueCampaignDiagnosticValue', (check / 'environment.log').read_text())
            private = list(check.glob('*' + runner.diagnostics.PRIVATE_SUFFIX))
            self.assertTrue(any('opaqueCampaignDiagnosticValue' in p.read_text() for p in private))
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in private))
            self.assertFalse((check / 'raw-bundle/build32.log').exists())
            for path in private:
                path.unlink()
            runner.validate_environment(check / 'container-environment/environment.json', gpu=False)

    def test_instrumentation_and_error_controls(self):
        launch = '========= Launch #1\n=========   Kernel: exl3_moe_coop_a_kernel()\n'
        for tool, summary in [('memcheck', 'ERROR SUMMARY: 0 errors'), ('racecheck', 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')]:
            log = launch + '========= ' + summary + '\n'
            self.assertEqual(evidence.sanitizer(log, tool), 1)
            for bad in (summary, log.replace('Launch #1', 'Launch #1 (filtered: kernel name)'),
                        log.replace('exl3_moe_coop_', 'stock_'), log + 'Internal Sanitizer Error',
                        log + '========= ERROR SUMMARY: 1 errors\n'):
                with self.assertRaises(ValueError): evidence.sanitizer(bad, tool)

    def test_policy_uses_all_468_new_cases_and_keeps_nonwinners_stock(self):
        import select_policy
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            for rank in range(3):
                for geometry in range(3):
                    path = Path(directory) / f'rank{rank}-geo{geometry}.jsonl'
                    rows = [{'stage':'profile_identity','native_sha256': q.TARGET_NATIVE}]
                    for n in select_policy.ROWS:
                        for pattern in select_policy.PATTERNS:
                            ratio = 1.1 if n == 64 else .95 + .01 * geometry
                            rows += [{'stage':'compare','label':f'profile/rank={rank}/geometry={geometry}/rows={n}/{pattern}',
                                      'pass':True,'peak_rel':.001,'row_rel':.01,'rel_l2':.01},
                                     {'stage':'profile','rank':rank,'geometry':geometry,'rows':n,'pattern':pattern,
                                      'stock':{'min_ms':1.,'median_ms':1.,'max_ms':1.},
                                      'candidate':{'min_ms':ratio,'median_ms':ratio,'max_ms':ratio}}]
                    rows += [{'stage':'profile_complete','rank':rank,'geometry':geometry}]
                    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    logs.append(path)
            policy = select_policy.select(logs, q.TARGET_NATIVE)
            self.assertEqual(policy['rows']['2'], 0)
            self.assertEqual(policy['rows']['64'], 'stock')
            self.assertEqual(policy['rows']['11'], 'stock')
            self.assertEqual(len(policy['rows']), 64)
            with self.assertRaises(ValueError): select_policy.select(logs[:-1], q.TARGET_NATIVE)
            logs[0].write_text(logs[0].read_text().replace(q.TARGET_NATIVE, hashed('old binary')))
            with self.assertRaisesRegex(ValueError, 'bound'): select_policy.select(logs, q.TARGET_NATIVE)

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
