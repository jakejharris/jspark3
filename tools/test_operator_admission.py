#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Readiness, real wrapper launch, hygiene and operator admission controls."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'recipe/scripts'), str(ROOT / 'tools/v16')]
import _fleetctl as fleet
import admission_gate
import apc_gate
import page_cache_hygiene
import qualify_runtime as qualification
import triar_inactive
from v16_common import sha256_json, write_json


class ReadinessTests(unittest.TestCase):
    values = {'JSPARK_MASTER_ADDR': '127.0.0.1', 'JSPARK_API_PORT': '8888'}
    rows = [dict(running=True, oom_killed=False, restart_count=0) for _ in range(3)]

    def response(self, body=b'{}'):
        response = io.BytesIO(body)
        response.status = 200
        return response

    def test_delayed_health_and_model_ready(self):
        responses = [urllib.error.URLError('loading'), self.response(),
                     urllib.error.HTTPError('model', 503, 'loading', {}, None),
                     self.response(), self.response(b'{"data":[{"id":"glm-5.3-flash"}]}')]
        with patch.object(fleet, 'collect_status', return_value=self.rows), \
                patch.object(fleet.urllib.request, 'urlopen', side_effect=responses), \
                patch.object(fleet.time, 'sleep') as sleep:
            fleet.wait_ready(self.values, {}, 60)
            self.assertEqual(sleep.call_count, 2)

    def test_timeout_dead_rank_and_wrong_model_refuse(self):
        with patch.object(fleet, 'collect_status', return_value=self.rows), \
                patch.object(fleet.urllib.request, 'urlopen', side_effect=urllib.error.URLError('loading')):
            with self.assertRaisesRegex(fleet.Refusal, 'timeout'):
                fleet.wait_ready(self.values, {}, 0)
        with patch.object(fleet, 'collect_status', return_value=[dict(running=False)]):
            with self.assertRaisesRegex(fleet.Refusal, 'startup rank'):
                fleet.wait_ready(self.values, {}, 60)
        with patch.object(fleet, 'collect_status', return_value=self.rows), \
                patch.object(fleet.urllib.request, 'urlopen', side_effect=[self.response(), self.response(b'{"data":[{"id":"wrong"}]}')]):
            with self.assertRaisesRegex(fleet.Refusal, 'identity drift'):
                fleet.wait_ready(self.values, {}, 60)
        for timeout in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(fleet.Refusal, 'finite'):
                fleet.wait_ready(self.values, {}, timeout)

    def test_real_non_b_wrapper_launch_does_not_write_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / 'recipe'
            shutil.copytree(ROOT / 'recipe', recipe, ignore=shutil.ignore_patterns('__pycache__'))
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONDONTWRITEBYTECODE'}
            proc = subprocess.run(['bash', 'scripts/preflight.sh', '--env-file', '.env.example', '--dry-run'],
                                  cwd=recipe, env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(list(recipe.rglob('__pycache__')), [])


class HygieneTests(unittest.TestCase):
    def test_real_fadvise_preserves_bytes_skips_symlinks_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'file').write_bytes(b'payload')
            os.link(root / 'file', root / 'hardlink')
            (root / 'link').symlink_to(root / 'file')
            os.mkfifo(root / 'fifo')
            report = page_cache_hygiene.evict([root])
            self.assertEqual((report['verdict'], report['files'], report['bytes']), ('PASS', 1, 7))
            self.assertEqual((root / 'file').read_bytes(), b'payload')
            self.assertTrue((root / 'link').is_symlink())

    def test_missing_root_and_permission_failure_are_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'file').write_text('data')
            with patch.object(page_cache_hygiene.os, 'open', side_effect=PermissionError(13, 'denied', 'file')):
                report = page_cache_hygiene.evict([root])
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertEqual(report['errors'][0]['errno'], 13)
            self.assertEqual(page_cache_hygiene.evict([root / 'absent'])['verdict'], 'FAIL')


class InactiveTests(unittest.TestCase):
    def fixture(self):
        activation = dict(rank=0, time=2, active=True, hash_gate=True, banks={'bf16_0': 1, 'int8_0': 1},
                          _file='activation-123.jsonl')
        caps, dots = [], {}
        for phase in ('profile', 'serving'):
            for bank in ('bf16_0', 'int8_0'):
                name = phase + bank
                dots[name] = dict(sha256='digest', triar_marker=False)
                caps.append(dict(rank=0, time=2, phase=phase, bank=bank, capture_order=['bf16_0', 'int8_0'],
                    _file='captures-123.jsonl', object=name, dot=name, dot_sha256='digest', desc={'mode': 'FULL', 'tokens': 4},
                    calls=[dict(prefix=str(i), path='bf16' if bank == 'bf16_0' else 'int8', weight_ptr=1, control_ptr=2)
                           for i in range(34)]))
        adaptive = [dict(rank=0, time=2, phase=phase, status='PASS', _file='adaptive-capture-123.jsonl',
                        descriptors=[dict(requests=r, query_width=4, physical_rows=4*r) for r in range(1, 9)])
                    for phase in ('profile', 'serving')]
        return dict(rank=0, observed_at=3, sources={'source': 'digest'}, epoch=None, dots=dots,
                    rows={'activation': [activation], 'captures': caps, 'adaptive-capture': adaptive})

    def test_positive_capture_and_all_refusal_controls(self):
        good = self.fixture()
        evaluate = lambda value: triar_inactive.evaluate(value, 0, {'source': 'digest'}, '1970-01-01T00:00:01Z')
        self.assertEqual(evaluate(good)['status'], 'PASS')
        changes = [lambda v: v.update(epoch={'schema': 'triar-epoch/1', 'mode': 'on'}),
                   lambda v: v.update(sources={'source': 'drift'}),
                   lambda v: v['rows']['captures'].pop(),
                   lambda v: v['rows']['captures'][0].update(time=0),
                   lambda v: v['rows']['captures'][0]['calls'].pop(),
                   lambda v: v['rows']['adaptive-capture'].pop(),
                   lambda v: v['dots']['profilebf16_0'].update(triar_marker=True),
                   lambda v: v['dots']['profilebf16_0'].update(sha256='drift'),
                   lambda v: v['rows']['captures'][1].update(object='profilebf16_0')]
        for change in changes:
            value = copy.deepcopy(good)
            change(value)
            with self.assertRaises((ValueError, KeyError)):
                evaluate(value)
        with self.assertRaisesRegex(qualification.QAError, 'contradicts'):
            qualification.triar_off('dual-replay', True)


class AdmissionTests(unittest.TestCase):
    def test_gate_defaults_match_fine_hit_runtime(self):
        args = apc_gate.build_parser().parse_args(['--base-url', 'http://localhost:8888', '--env-file', 'env',
                                                 '--fixtures', 'fixtures', '--out', 'out'])
        self.assertEqual(args.expect, 'finehit')

    def test_real_admission_rejects_changed_boot_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'coop': 'off', 'adaptive-k': 'ema', 'dense-fp8': 'trunk'}
            first = dict(schema=admission_gate.FIRST_SCHEMA, verdict='PASS', identity_config=identity,
                         producer='qualify_runtime.py', boot=[{'container_id': 'one'}], manifest_sha256='manifest')
            write_json(root / 'first-prompt.json', first)
            digest = hashlib.sha256((root / 'first-prompt.json').read_bytes()).hexdigest()
            final = {**first, 'schema': admission_gate.FINAL_SCHEMA, 'evidence_sha256': {'first-prompt.json': digest},
                     'input_hashes': {'client_evidence': [{'schema': admission_gate.FIRST_SCHEMA, 'sha256': digest}]}}
            final['payload_sha256'] = sha256_json(final)
            write_json(root / 'finalize.json', final)
            argv = ['--first-prompt', str(root / 'first-prompt.json'), '--finalize', str(root / 'finalize.json'),
                    '--out', str(root / 'admission.json')]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(admission_gate.main(argv), 0)
                (root / 'first-prompt.json').write_text(json.dumps({**first, 'boot': [{'container_id': 'other'}]}))
                self.assertEqual(admission_gate.main(argv), 1)
                write_json(root / 'first-prompt.json', first)
                (root / 'first-prompt.json').write_text((root / 'first-prompt.json').read_text() + '\n')
                self.assertEqual(admission_gate.main(argv), 1)

    def workflow(self, failure=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, manifest_path = root / 'env', root / 'service.json'
            env.write_text('fixture')
            manifest_path.write_text('{}')
            out = root / 'proof'
            values = dict(JSPARK3_V16_PROFILE='production-stock', ABLIT='0', JSPARK3_V16_COOP='0',
                          JSPARK3_V16_APC_LRU='1', JSPARK_MASTER_ADDR='127.0.0.1', JSPARK_API_PORT='8888',
                          GLM53_ADAPTIVE_K='ema', JSPARK3_V16_DENSE_FP8='trunk', JSPARK3_TRIAR='1')
            bindings = [dict(rank=r, container_id=str(r) * 64) for r in range(3)]
            manifest = dict(containers=bindings, recipe_manifest_sha256='recipe')
            events = []
            def remote(values, rank, argv, **kwargs):
                if qualification.EPOCH_READ in argv:
                    doc = {name: None for name in ('triar-epoch.json', 'adaptive-k-epoch.json', 'sched-epoch.json', 'fatpath-epoch.json')}
                elif '/recipe/scripts/page_cache_hygiene.py' in argv:
                    events.append('hygiene-' + str(rank))
                    doc = dict(verdict='FAIL' if failure == 'hygiene' else 'PASS', files=3, bytes=10, errors=[])
                elif '/recipe/scripts/triar_inactive.py' in argv:
                    doc = dict(verdict='FAIL' if failure == 'triar' else 'PASS', attestation={'rank': rank})
                else:
                    return types.SimpleNamespace(stdout='TileLang begins to compile' if failure == 'jit' else '', stderr='', returncode=0)
                return types.SimpleNamespace(stdout=json.dumps(doc), stderr='', returncode=0)
            def run(command, **kwargs):
                script = Path(command[2]).name
                output = Path(command[command.index('--out') + 1]) if '--out' in command else Path(command[command.index('--output') + 1])
                if script == 'fleetctl.py':
                    events.append(output.stem)
                    doc = dict(status='VERIFY_PASS', manifest_sha256=fleet.sha_file(manifest_path), production_stock={'status': 'PASS'})
                    doc['payload_sha256'] = fleet.sha_bytes(fleet.canonical(doc))
                    write_json(output, doc)
                elif script == 'prefill_gate.py':
                    events.append(output.stem)
                    first = output.stem == 'prefill-first'
                    passing = not first and failure != 'prefill'
                    doc = dict(floor_tok_s=1100, gate_start='2026-09-28T00:00:00Z', compile_lines_after_warmup=[],
                               gate=[dict(tag=f'pi-turn-{i}', requests=1, new_tokens=3000, cached_tokens=2560,
                                          prefill_tok_s=1200 if passing else 900) for i in range(8)],
                               verdict='PASS' if passing else 'FAIL')
                    doc['receipt_sha256'] = sha256_json(doc)
                    write_json(output, doc)
                    return types.SimpleNamespace(returncode=0 if passing else 1, stdout='', stderr='')
                elif script == 'apc_gate.py':
                    events.append('apc')
                    write_json(output, dict(expect='finehit', analysis={'verdict': 'PASS'}, checker_controls={'negative': 'FAIL'}))
                elif script == 'admission_gate.py':
                    events.append('admission')
                    with redirect_stdout(io.StringIO()):
                        code = admission_gate.main(command[3:])
                    return types.SimpleNamespace(returncode=code, stdout='', stderr='')
                else:
                    raise AssertionError(script)
                return types.SimpleNamespace(returncode=0, stdout='', stderr='')
            native = types.SimpleNamespace(load_env=lambda p: values, validate_env=lambda v: None,
                bound_manifest=lambda *a, **kw: manifest, verify_remote_recipe=lambda *a: None,
                inspect_identity=lambda *a: {'State': {'Running': True, 'OOMKilled': False,
                    'StartedAt': 'changed' if failure == 'restart' and events else 'start'}, 'RestartCount': 0},
                remote=remote, sha_file=fleet.sha_file, sha_bytes=fleet.sha_bytes, canonical=fleet.canonical,
                atomic_text=fleet.atomic_text, redact_diagnostics=fleet.redact_diagnostics)
            args = argparse.Namespace(recipe=ROOT / 'recipe', env_file=env, manifest=manifest_path, output=out)
            with patch.object(qualification.subprocess, 'run', side_effect=run), redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(qualification.QAError): qualification.run(args, native)
                    self.assertFalse((out / 'admission.json').exists())
                    self.assertFalse((out / 'finalize.json').exists())
                else:
                    qualification.run(args, native)
                    self.assertEqual(json.loads((out / 'admission.json').read_text())['verdict'], 'PASS')
                    self.assertEqual(events, ['verify-first', 'prefill-first', 'hygiene-0', 'hygiene-1', 'hygiene-2',
                                              'prefill-post', 'apc', 'verify-final', 'admission'])

    def test_producer_drives_real_admission_and_first_pass_failure_is_retained(self):
        self.workflow()

    def test_producer_refuses_failed_repeat_hygiene_triar_compile_or_changed_boot(self):
        for failure in ('prefill', 'hygiene', 'triar', 'jit', 'restart'):
            with self.subTest(failure=failure): self.workflow(failure)

    def test_first_prefill_failure_recorded_repeat_blocks(self):
        rows = [dict(tag=f'pi-turn-{i}', requests=1, new_tokens=3000, cached_tokens=2560, prefill_tok_s=900) for i in range(8)]
        doc = dict(floor_tok_s=1100, gate=rows, compile_lines_after_warmup=[], verdict='FAIL')
        doc['receipt_sha256'] = sha256_json(doc)
        qualification.prefill_pass(doc, blocking=False)
        with self.assertRaisesRegex(qualification.QAError, 'post-hygiene'):
            qualification.prefill_pass(doc, blocking=True)
        doc['gate'][0]['prefill_tok_s'] = 1500
        with self.assertRaisesRegex(qualification.QAError, 'hash drift'):
            qualification.prefill_pass(doc, blocking=False)


if __name__ == '__main__':
    unittest.main()
