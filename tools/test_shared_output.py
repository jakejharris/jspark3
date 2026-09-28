#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R4 integration probes with R5 expectations; real producers, writers and gates."""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
from unittest.mock import patch

import tempfile
import unittest
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / 'recipe/scripts'), str(SOURCE / 'tools'), str(SOURCE / 'tools/v16')]
import _fleetctl as fleet
import qualify_runtime as q
import prefill_gate as prefill
import admission_gate as admission
import test_operator_admission as existing
from v16_common import write_json, sha256_json


def integration(source, evidence):
    evidence.mkdir(parents=True, exist_ok=True)
    secret = 'opaqueR4IntegrationCredentialValue'
    result = {}

    # Complete the shipped positive qualification fixture, retaining its actual output.
    workflow_root = evidence / 'qualification-inventory'
    workflow_root.mkdir()
    class RetainDirectory:
        def __enter__(self): return str(workflow_root)
        def __exit__(self, *args): pass
    with patch.object(existing.tempfile, 'TemporaryDirectory', RetainDirectory):
        existing.AdmissionTests().workflow()
    original = workflow_root / 'proof'
    final = json.loads((original / 'finalize.json').read_text())
    private_names = [n for n in final['evidence_sha256'] if n.endswith(fleet.diagnostics.PRIVATE_SUFFIX)]
    shared = workflow_root / 'shared-bundle'
    shutil.copytree(original, shared, ignore=shutil.ignore_patterns('*' + fleet.diagnostics.PRIVATE_SUFFIX))
    with redirect_stdout(io.StringIO()):
        recheck = admission.main(['--first-prompt', str(shared/'first-prompt.json'), '--finalize', str(shared/'finalize.json'), '--out', str(shared/'recheck.json')])
    result['private_inventory'] = dict(original_verdict=json.loads((original/'admission.json').read_text())['verdict'],
        private_dependencies=private_names, recheck_returncode=recheck, recheck=json.loads((shared/'recheck.json').read_text()))
    assert not private_names and recheck == 0
    assert not any(n.endswith(fleet.diagnostics.PRIVATE_SUFFIX) for n in json.loads((original/'first-prompt.json').read_text())['evidence_sha256'])

    def environment(root):
        root.mkdir()
        env, manifest_path = root/'env', root/'manifest.json'
        env.write_text('fixture'); manifest_path.write_text('{}')
        values = dict(JSPARK3_V16_PROFILE='production-stock', ABLIT='0', JSPARK3_V16_COOP='0',
                      JSPARK3_V16_APC_LRU='1', JSPARK_MASTER_ADDR='127.0.0.1', JSPARK_API_PORT='8888',
                      GLM53_ADAPTIVE_K='ema', JSPARK3_V16_DENSE_FP8='trunk', JSPARK3_TRIAR='1')
        manifest = dict(containers=[dict(rank=r, container_id=str(r)*64) for r in range(3)], recipe_manifest_sha256='recipe')
        native = types.SimpleNamespace(load_env=lambda p: values, validate_env=lambda v: None,
            bound_manifest=lambda *a, **kw: manifest, verify_remote_recipe=lambda *a: None,
            inspect_identity=lambda *a: {'State': {'Running': True, 'OOMKilled': False, 'StartedAt': 'start'}, 'RestartCount': 0},
            sha_file=fleet.sha_file, sha_bytes=fleet.sha_bytes, canonical=fleet.canonical,
            save_diagnostics=fleet.save_diagnostics, redact_diagnostics=fleet.redact_diagnostics, Refusal=fleet.Refusal)
        args = argparse.Namespace(recipe=source/'recipe', env_file=env, manifest=manifest_path, output=root/'proof')
        return native, args

    # Run real qualify_runtime.run -> real prefill.main -> real JSON writer with synthetic remote log.
    native, args = environment(evidence/'prefill-leak')
    raw_log = 'TileLang begins to compile Authorization: Custom ' + secret + '\n'
    def remote(values, rank, argv, **kwargs):
        if q.EPOCH_READ in argv:
            doc = {name: None for name in ('triar-epoch.json','adaptive-k-epoch.json','sched-epoch.json','fatpath-epoch.json')}
            return types.SimpleNamespace(stdout=json.dumps(doc), stderr='', returncode=0)
        if '/recipe/scripts/page_cache_hygiene.py' in argv:
            return types.SimpleNamespace(stdout=json.dumps(dict(verdict='FAIL', files=0, bytes=0, errors=[])), stderr='', returncode=1)
        raise AssertionError(argv)
    native.remote = remote
    def measured(base, body, tag):
        return dict(tag=tag, requests=1, new_tokens=3000, cached_tokens=2560, prefill_tok_s=1200)
    def command(command, **kwargs):
        script = Path(command[2]).name
        if script == 'fleetctl.py':
            output = Path(command[command.index('--output')+1])
            doc = dict(status='VERIFY_PASS', manifest_sha256=fleet.sha_file(args.manifest), production_stock={'status':'PASS'})
            doc['payload_sha256'] = fleet.sha_bytes(fleet.canonical(doc)); write_json(output, doc)
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        assert script == 'prefill_gate.py'
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(prefill, 'measured', side_effect=measured), patch.object(prefill, 'class_body', return_value={}), \
             patch.object(prefill, 'pi_turn', side_effect=lambda *a: ({}, 'pi-turn-'+str(a[-1]))), \
             patch.object(prefill.time, 'sleep'), patch.object(prefill, 'fetch_remote_logs', return_value=(raw_log, [])), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            code = prefill.main(command[3:])
        return types.SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())
    with patch.object(q.subprocess, 'run', side_effect=command):
        try: q.run(args, native)
        except q.QAError as exc: assert str(exc) == 'rank0 hygiene incomplete'
    receipt = args.output/'prefill-first.json'
    result['prefill_receipt'] = dict(shared_receipt=str(receipt), secret_in_shared_receipt=secret in receipt.read_text(),
        secret_in_shared_log=secret in (args.output/'prefill-first.log').read_text(),
        fields=[k for k,v in json.loads(receipt.read_text()).items() if secret in json.dumps(v)])
    assert not result['prefill_receipt']['secret_in_shared_receipt'] and not result['prefill_receipt']['secret_in_shared_log']
    assert any(secret in p.read_text() for p in args.output.glob('*'+fleet.diagnostics.PRIVATE_SUFFIX))

    # Direct qualification remote failure: real local child, real RemoteFailure, real main catcher.
    native, args = environment(evidence/'lost-qualification-tail')
    native.remote = fleet.remote
    stderr = 'Traceback (most recent call last):\n  File "<string>", line 11, in check\nAssertionError: epoch inspect failed '+secret+'\n'
    child = [sys.executable, '-B', '-c', 'import sys;sys.stderr.write(sys.argv[1]);sys.exit(17)', stderr]
    argv = ['qualify_runtime.py', '--recipe', str(args.recipe), '--env-file', str(args.env_file), '--manifest', str(args.manifest), '--output', str(args.output)]
    console = io.StringIO()
    with patch.object(sys, 'argv', argv), patch.object(q.importlib, 'import_module', return_value=native), \
         patch.object(fleet, 'ssh_argv', return_value=child), redirect_stderr(console):
        code = q.main()
    (args.output.parent/'console.txt').write_text(console.getvalue())
    result['lost_qualification_tail'] = dict(returncode=code, console=console.getvalue(),
        private_files=[str(p) for p in args.output.parent.rglob('*'+fleet.diagnostics.PRIVATE_SUFFIX)])
    assert code == 1 and result['lost_qualification_tail']['private_files']
    assert secret not in console.getvalue()
    assert 'rank0 remote command failed (exit 17)' in console.getvalue()
    assert 'docker exec' in console.getvalue() and '<string>:11' in console.getvalue()
    assert any(secret in Path(p).read_text() for p in result['lost_qualification_tail']['private_files'])

    # Start cleanup: after one created identity, preserve the real nested manifest-write error path.
    start_root = evidence/'start-write-error'; start_root.mkdir()
    preflight = start_root/'preflight.json'; preflight.write_text('{}')
    manifest = start_root/'service.json'
    argv = ['fleetctl', 'start', '--env-file', str(start_root/'env'), '--manifest', str(manifest),
            '--preflight', str(preflight), '--preflight-sha256', fleet.sha_file(preflight), '--confirm', 'START-JSPARK3']
    def start_remote(values, rank, command, **kwargs):
        if command[:3] == ['docker','container','inspect']:
            if len(command[-1]) == 64: raise fleet.Refusal('post-create inspect failed')
            return types.SimpleNamespace(returncode=1, stdout='', stderr='')
        return types.SimpleNamespace(returncode=0, stdout='c'*64, stderr='')
    console = io.StringIO()
    with ExitStack() as stack:
        for obj, attr, kw in [
            (sys,'argv',{'new':argv}), (fleet,'load_env',{'return_value':{'JSPARK_WORK_ROOT':'/work'}}),
            (fleet,'validate_env',{}), (fleet,'read_receipt',{'return_value':{}}), (fleet,'validate_preflight_receipt',{'return_value':({},'r'*64)}),
            (fleet,'configuration_digest',{'return_value':'config'}), (fleet,'verify_remote_recipe',{}),
            (fleet,'container_argv',{'return_value':['docker','create']}), (fleet,'remote',{'side_effect':start_remote}),
            (fleet,'stop_created',{'return_value':True}),
            (fleet,'atomic_json',{'side_effect':OSError(28,'No space left on device','/private/'+secret+'/service.json')})]:
            stack.enter_context(patch.object(obj,attr,**kw))
        stack.enter_context(redirect_stderr(console))
        code = fleet.main()
    (start_root/'console.txt').write_text(console.getvalue())
    result['start_write_error'] = dict(returncode=code, secret_in_shared_console=secret in console.getvalue(), console=console.getvalue())
    assert code == 9 and not result['start_write_error']['secret_in_shared_console']
    assert any(secret in p.read_text() for p in start_root.glob('*'+fleet.diagnostics.PRIVATE_SUFFIX))

    (evidence/'integration.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


class IntegrationTests(unittest.TestCase):
    def test_reviewers_four_real_integration_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            integration(SOURCE, Path(directory))


class OutputBoundaryTests(unittest.TestCase):
    def test_api_and_metrics_project_opaque_fields_and_retain_raw(self):
        import apc_gate
        import v16_common as common
        secret = 'opaqueProtocolCredentialValue'
        event = dict(id=secret, usage=dict(completion_tokens=1, vendor=secret),
                     choices=[dict(token_ids=[41], finish_reason='length', logprobs=dict(
                         tokens=[secret], token_logprobs=[-.1], top_logprobs=[{secret: -.1}]))])
        with tempfile.TemporaryDirectory() as directory, patch.object(fleet.diagnostics, 'private_directory', return_value=Path(directory)):
            result = apc_gate.consume(['data: ' + json.dumps(event), 'data: [DONE]'], 0, now=lambda: 1)
            metrics = common.parse_prometheus(secret + ' 1\nvllm:num_requests_running 0\n')
            self.assertNotIn(secret, json.dumps(result) + json.dumps(metrics))
            self.assertEqual(result['usage'], {'completion_tokens': 1})
            self.assertEqual(metrics, {'vllm:num_requests_running': 0.0})
            self.assertEqual(result['token_strs'][0], next(iter(result['top_logprobs'][0])))
            private = list(Path(directory).glob('*' + fleet.diagnostics.PRIVATE_SUFFIX))
            self.assertTrue(any(secret in p.read_text() for p in private))
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in private))

    def test_failure_keeps_complete_diagnostic_beyond_bounded_tail(self):
        raw = 'earlyPrivateDiagnosticValue\n' + 'x' * 50000 + '\nlatePrivateDiagnosticValue'
        with tempfile.TemporaryDirectory() as directory:
            error = fleet.RemoteFailure(1, ['docker', 'exec', 'privateArgument'], 17, raw)
            record = fleet.diagnostics.record_failure(error, Path(directory) / 'failure.json')
            self.assertEqual((Path(directory) / record['private_diagnostic']).read_text(), raw)
            self.assertEqual((Path(directory) / record['private_stderr_tail']).read_text(), raw[-16384:])
            self.assertNotIn('earlyPrivateDiagnosticValue', json.dumps(record))
            self.assertNotIn('privateArgument', json.dumps(record))

    def test_child_output_is_private_and_cli_exception_hook_is_structural(self):
        secret = 'opaqueChildCompilerDiagnosticValue'
        with tempfile.TemporaryDirectory() as directory:
            child = [sys.executable, '-B', '-c', 'import sys;print(sys.argv[1]);raise ValueError(sys.argv[1])', secret]
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                fleet.diagnostics.run_private(child, check=True)
            self.assertIn(secret, Path(raised.exception.private_child_output).read_text())
            console = io.StringIO()
            with redirect_stderr(console):
                record = fleet.diagnostics.report_failure(raised.exception, Path(directory) / 'failure.json')
            self.assertNotIn(secret, console.getvalue())
            self.assertIn('Complete private child output:', (Path(directory) / record['private_diagnostic']).read_text())
            probe = "import sys;sys.path.insert(0,sys.argv[1]);import _diagnostics as d;d.install_exception_hook();raise ValueError(sys.argv[2])"
            process = subprocess.run([sys.executable, '-B', '-c', probe, str(SOURCE/'recipe/scripts'), secret], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertNotIn(secret, process.stdout + process.stderr)
            self.assertIn('ValueError', process.stderr)

    def test_relocated_validator_import_has_no_recipe_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'ablit_artifacts.py'
            shutil.copy2(SOURCE / 'recipe/scripts/validate_ablit_artifacts.py', target)
            process = subprocess.run([sys.executable, '-B', '-I', '-S', '-c',
                'import runpy,sys;runpy.run_path(sys.argv[1],run_name="relocated_module")', str(target)],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_audit_refuses_changed_writer_and_new_output_path(self):
        import _shared_output_audit as audit
        self.assertEqual(audit.verify(SOURCE), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(SOURCE / 'manifests', root / 'manifests')
            for name, path in audit.candidates(SOURCE).items():
                dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
            self.assertEqual(audit.verify(root), [])
            existing = root / 'tools/v16/prefill_gate.py'
            existing.write_text(existing.read_text() + '\nprint("new unaudited output")\n')
            self.assertTrue(any('prefill_gate.py' in issue for issue in audit.verify(root)))
            (root / 'tools/new_writer.py').write_text('print("new output path")\n')
            self.assertIn('executable inventory changed; shared-output review required', audit.verify(root))


if __name__ == '__main__':
    unittest.main()
