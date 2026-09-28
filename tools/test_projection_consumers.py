#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Projected facts must still drive accounting, parity and actionable failures."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'recipe/scripts'), str(ROOT / 'tools/v16')]
import argparse
import copy
import io
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch
import _diagnostics as diagnostics
import _shared_output_audit as audit
import apc_gate as apc
import page_cache_hygiene as hygiene
import prefill_gate as prefill
import qualify_runtime as qualification
import v16_common as common
from test_verify_healthy import HealthyFleet

SECRET = 'opaqueProjectionCredentialValue'


def stream(tokens, prompt=5560, cached=2560):
    event = dict(id=SECRET, usage=dict(completion_tokens=tokens, prompt_tokens=prompt,
        prompt_tokens_details=dict(cached_tokens=cached, vendor=SECRET), vendor=SECRET),
        choices=[dict(token_ids=list(range(tokens)), finish_reason='length', delta={'content': SECRET},
            logprobs=dict(tokens=[SECRET + str(i) for i in range(tokens)], token_logprobs=[-.1]*tokens,
                top_logprobs=[{SECRET + str(i): -.1, SECRET + 'alternative': -.2} for i in range(tokens)]))])
    return io.BytesIO(('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n').encode())


class ProjectionConsumerTests(unittest.TestCase):
    def test_apc_raw_and_hashed_tokens_keep_all_parity_and_checker_decisions(self):
        cells = apc.cells_from({k: list(range(17000)) for k in ('prose', 'tools', 'long')})
        apc.validate_cells(cells)
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, 'private_directory', return_value=Path(directory)):
            def document(raw):
                # Identity projection gives the pre-redesign protocol representation.
                with patch.object(diagnostics, 'fingerprint', side_effect=(lambda x: x) if raw else diagnostics.fingerprint):
                    probe = apc.consume(stream(apc.PROBE_TOKENS), 0, now=lambda: 1)
                rows = []
                for cell in cells:
                    cold = dict(probe, cached_tokens=0, prefill_s=1)
                    warm = dict(probe, cached_tokens=cell['expected_hit'], prefill_s=.5)
                    rows.append(dict(id=cell['id'], producer={'cached_tokens': 0},
                                     cold=[copy.deepcopy(cold) for _ in range(3)], warm=copy.deepcopy(warm)))
                return dict(cells=cells, rows=rows, expect='finehit', requests_delta=len(rows)*5, apc_identity=[['6', '']])
            raw, projected = document(True), document(False)
            self.assertEqual(apc.analyze(raw), apc.analyze(projected))
            self.assertEqual(apc.analyze(projected)['verdict'], 'PASS')
            self.assertEqual(apc.checker_controls(raw), apc.checker_controls(projected))
            self.assertEqual(set(apc.checker_controls(projected).values()), {'FAIL'})
            # Exercise top-token lookup at divergence, not just identical sequences.
            for doc in (raw, projected):
                warm = doc['rows'][0]['warm']
                warm['token_ids'][0] = 999
                warm['token_strs'][0] = (SECRET + 'alternative' if doc is raw else diagnostics.fingerprint(SECRET + 'alternative'))
            self.assertEqual(apc.compare(raw['rows'][0]['warm'], raw['rows'][0]['cold'][0]),
                             apc.compare(projected['rows'][0]['warm'], projected['rows'][0]['cold'][0]))
            self.assertEqual(apc.compare(projected['rows'][0]['warm'], projected['rows'][0]['cold'][0])['tie_gap'], 0.0)
            self.assertNotIn(SECRET, json.dumps(projected))

    def test_real_api_metric_projections_feed_prefill_and_apc_accounting(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, 'private_directory', return_value=Path(directory)):
            def metrics(hit=2560):
                return ('vllm:request_success_total 1\nvllm:prefix_cache_queries_total 5560\n'
                        f'vllm:prefix_cache_hits_total {hit}\nvllm:request_prefill_time_seconds_sum 2\n'
                        f'vllm:num_requests_running{{label="{SECRET}"}} 0\nvllm:num_requests_waiting 0\n{SECRET} 5\n')
            def http(request, **kwargs):
                if isinstance(request, str):
                    raise AssertionError('HTTP must use a Request')
                return stream(64)
            with patch.object(common, 'http_get_text', side_effect=['', metrics()]), \
                    patch.object(prefill.urllib.request, 'urlopen', side_effect=http), patch.object(prefill.time, 'sleep'):
                row = prefill.measured('http://localhost', {}, 'pi-turn-0')
            self.assertEqual((row['new_tokens'], row['cached_tokens'], row['prefill_tok_s']), (3000, 2560, 1500))
            self.assertEqual(prefill.evaluate([row], [], 1100), [])
            self.assertTrue(prefill.evaluate([dict(row, cached_tokens=0)], [], 1100))
            self.assertTrue(prefill.evaluate([dict(row, prefill_tok_s=1000)], [], 1100))
            markers = prefill.compile_lines('x TileLang begins to compile ' + SECRET)
            self.assertTrue(prefill.evaluate([row], markers, 1100))
            self.assertNotIn(SECRET, json.dumps(row) + json.dumps(markers))
            payload = apc.body('glm-5.3-flash', list(range(5560)), 'synthetic-salt', 64, True)
            for hit in (2560, 0):
                with self.subTest(hit=hit), patch.object(common, 'http_get_text', side_effect=['', metrics(hit)]), \
                        patch.object(apc.urllib.request, 'urlopen', side_effect=http), patch.object(apc.time, 'sleep'):
                    if hit == 0:
                        with self.assertRaisesRegex(common.QAError, 'usage cached'):
                            apc.request('http://localhost', payload)
                    else:
                        result = apc.request('http://localhost', payload)
                        self.assertEqual((result['cached_tokens'], result['requests_delta']), (2560, 1))
                        self.assertNotIn(SECRET, json.dumps(result))
            parsed = common.parse_prometheus(metrics())
            self.assertEqual(common.idle_issues(parsed), [])
            self.assertTrue(common.idle_issues(dict(parsed, **{'vllm:num_requests_running': 1})))

    def test_hygiene_errors_retain_filename_errno_trace_and_qualification_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, private = root/'inputs', root/'rank-private'
            inputs.mkdir(); private.mkdir()
            target = inputs / SECRET
            target.write_text('data')
            original_open = os.open
            def denied(path, *args, **kwargs):
                if Path(path) == target:
                    raise PermissionError(13, 'Permission denied', str(target))
                return original_open(path, *args, **kwargs)
            with patch.object(diagnostics, 'private_directory', return_value=private):
                with patch.object(os, 'open', side_effect=denied):
                    report = hygiene.evict([inputs])
                with patch.object(os, 'posix_fadvise', side_effect=OSError(5, 'I/O error')):
                    fd_error = hygiene.evict([inputs])
                missing = hygiene.evict([inputs / 'absent'])
            for doc in (report, fd_error, missing):
                self.assertEqual(doc['verdict'], 'FAIL')
                self.assertNotIn(SECRET, json.dumps(doc))
                detail = Path(doc['errors'][0]['private_diagnostic'])
                self.assertEqual(detail.stat().st_mode & 0o777, 0o600)
                self.assertIn(str(inputs / 'absent' if doc is missing else target), detail.read_text())
            self.assertEqual(report['errors'][0]['errno'], 13)
            self.assertIn('PermissionError', Path(report['errors'][0]['private_diagnostic']).read_text())
            # Actual qualification writer retains the rank reply, including its
            # actionable rank-local private pointer, behind a local private pointer.
            (root/'fleet').mkdir()
            fixture = HealthyFleet(root/'fleet')
            self.assertEqual(fixture.run(), 0)
            verified = fixture.receipt
            out = root/'qualification'
            args = argparse.Namespace(env_file=fixture.env, manifest=fixture.manifest_path, recipe=ROOT/'recipe', output=out)
            initial = [dict(rank=r, container_id=fixture.manifest['containers'][r]['container_id'], started_at='start', epochs={}) for r in range(3)]
            def child(argv, **kwargs):
                name = Path(argv[2]).name
                dest = Path(argv[argv.index('--output' if name == 'fleetctl.py' else '--out')+1])
                doc = verified if name == 'fleetctl.py' else dict(gate=[dict(tag=f'pi-turn-{r}', requests=1) for r in range(8)],
                    floor_tok_s=1100, gate_start='start', verdict='PASS')
                if name != 'fleetctl.py':
                    doc['receipt_sha256'] = common.sha256_json(doc)
                common.write_json(dest, doc)
                return types.SimpleNamespace(returncode=0, stdout='', stderr='')
            reply = types.SimpleNamespace(returncode=1, stdout=json.dumps(report), stderr='')
            import _fleetctl as fleet
            with patch.object(qualification, 'snapshot', return_value=initial), \
                    patch.object(fleet, 'verify_remote_recipe'), patch.object(fleet, 'remote', return_value=reply), \
                    patch.object(qualification.subprocess, 'run', side_effect=child):
                with self.assertRaisesRegex(qualification.QAError, 'rank0 hygiene incomplete'):
                    qualification.run(args, fleet)
            summary = json.loads((out/'hygiene-rank0.json').read_text())
            self.assertEqual(summary['verdict'], 'FAIL')
            captured = json.loads((out/summary['private_diagnostic']).read_text())
            self.assertEqual(captured, report)
            self.assertIn(str(target), Path(captured['errors'][0]['private_diagnostic']).read_text())
            self.assertNotIn(SECRET, (out/'hygiene-rank0.json').read_text())

    def test_audit_discovers_exec_shebang_and_hidden_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifests').mkdir()
            (root/'manifests/shared-output-audit.json').write_text(json.dumps({'files': {}}))
            for name, content, mode in [('exec-only', 'print(1)\n', 0o755),
                                        ('shebang-only', '#!/bin/sh\necho 1\n', 0o644),
                                        ('.hidden/writer', '#!/bin/sh\necho 1\n', 0o644)]:
                with self.subTest(name=name):
                    path = root/name; path.parent.mkdir(exist_ok=True)
                    path.write_text(content); path.chmod(mode)
                    self.assertIn(name, audit.candidates(root))
                    self.assertIn('executable inventory changed; shared-output review required', audit.verify(root))
                    path.unlink()


if __name__ == '__main__':
    unittest.main()
