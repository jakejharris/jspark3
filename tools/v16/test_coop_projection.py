#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Component projections preserve gate decisions and keep diagnostics private."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe/scripts'))
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
import _diagnostics as diagnostics
import coop_projection as projection
import coop_evidence as evidence
import qualify_coop as runner
from test_coop_qualification import profile_rows, hashed
import test_coop_qualification as fixtures

SECRET = 'opaqueComponentDiagnosticValue'


class ProjectionTests(unittest.TestCase):
    def test_all_profile_decisions_survive_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for rank in range(3):
                for geometry in range(3):
                    rows = profile_rows(rank, geometry, evidence.TARGET_NATIVE)
                    for row in rows:
                        row['vendor_diagnostic'] = SECRET
                    raw = SECRET + '\n' + ''.join(json.dumps(row) + '\n' for row in rows)
                    path = root / f'rank{rank}-geo{geometry}.jsonl'
                    path.write_text(projection.gate_log('profile-r0-g0', raw))
                    self.assertNotIn(SECRET, path.read_text())
                    paths.append(path)
            selected = evidence.select_policy.select(paths, evidence.TARGET_NATIVE)
            self.assertEqual(selected['rows']['2'], 0)
            self.assertEqual(selected['rows']['64'], 'stock')
            for mutation in ('failure', 'nonfinite', 'missing'):
                original = paths[0].read_text()
                rows = [json.loads(line) for line in original.splitlines()]
                if mutation == 'failure': rows[1]['pass'] = False
                elif mutation == 'nonfinite': rows[1]['peak_rel'] = float('nan')
                else: rows.pop(1)
                with self.assertRaises(ValueError):
                    paths[0].write_text(projection.gate_log('profile-r0-g0', '\n'.join(map(json.dumps, rows))))
                    evidence.select_policy.select(paths, evidence.TARGET_NATIVE)
                paths[0].write_text(original)

    def test_complete_sanitizer_output_checked_before_projection(self):
        for tool, summary in [('memcheck', 'ERROR SUMMARY: 0 errors'),
                              ('racecheck', 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')]:
            text = '\n'.join([SECRET, json.dumps({'stage': 'race_smoke_complete', 'rank': 0, 'pass': True, 'vendor': SECRET}),
                '========= Launch #1 ' + SECRET, '=========   Kernel: exl3_moe_coop_' + SECRET,
                '========= ' + summary])
            projected = projection.gate_log('smoke-r0-g0-' + tool, text)
            self.assertNotIn(SECRET, projected)
            self.assertEqual(evidence.sanitizer(projected, tool), evidence.sanitizer(text, tool))
            for bad in ('Internal Sanitizer Error', '========= ERROR SUMMARY: 1 errors'):
                with self.assertRaises(ValueError):
                    projection.gate_log('smoke-r0-g0-' + tool, bad + '\n' + 'x' * 20000 + '\n' + text)

    def test_environment_identities_hashed_linkage_decision_retained(self):
        raw = {'status': 'PASS', 'checkpoint': {}, 'bundle': {}, 'exl3_sha256': hashed('exl3'),
               'fatpath_sha256': hashed('fatpath'), 'ldd': 'libcudart.so.13 => ' + SECRET,
               **{k: SECRET for k in projection.VERSIONS},
               'gpu': {'name': 'NVIDIA GB10', 'capability': [12, 1], 'driver': SECRET}}
        projected = projection.environment(raw)
        self.assertNotIn(SECRET, json.dumps(projected))
        self.assertEqual(projected['ldd'], projection.LINKAGE)
        for key in ('ldd', 'gpu', 'nvcc'):
            bad = copy.deepcopy(raw)
            if key == 'ldd': bad[key] += ' not found'
            elif key == 'gpu': bad[key]['name'] = SECRET
            else: bad[key] = ''
            with self.assertRaises(ValueError): projection.environment(bad)

    def test_runner_failure_retains_raw_without_console_leak(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, 'argv', ['qualify_coop', '--seal', '/unused', '--output', '/unused']), \
                patch.object(runner, 'seal', side_effect=ValueError(SECRET)), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(runner.main(), 9)
            self.assertNotIn(SECRET, stderr.getvalue())
            files = [Path(stderr.getvalue().split('Private diagnostics (do not share): ', 1)[1].splitlines()[0])]
            self.assertTrue(any(SECRET in p.read_text() for p in files))
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in files))

    def test_seal_has_no_private_diagnostic_dependencies(self):
        # Run the complete real seal producer from the retained 468-case fixture.
        # Add private diagnostics and unlisted build output, then remove them
        # before revalidating and emitting the seal.
        fixture = fixtures.RecordTests('test_complete_synthetic_campaign_seals_without_mutating_raw_evidence')
        original = runner.seal
        seen = []
        def seal(args):
            if args.output.name != 'sealed': return original(args)
            private = diagnostics.private_text(args.seal / 'gate.log', SECRET)
            path = args.seal / private
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            before = evidence.validate_campaign(args.seal)
            path.unlink()
            self.assertEqual(before, evidence.validate_campaign(args.seal))
            extra = args.seal / 'selected-bundle/compiler.log'
            extra.write_text(SECRET)
            try: original(args)
            finally: extra.unlink()
            self.assertFalse((args.output / 'bundle/compiler.log').exists())
            self.assertFalse(list(args.output.rglob('*' + diagnostics.PRIVATE_SUFFIX)))
            self.assertNotIn(SECRET, (args.output / 'QUALIFICATION.json').read_text())
            seen.append(True)
        with patch.object(runner, 'seal', side_effect=seal):
            result = unittest.TestResult()
            fixture.run(result)
        self.assertEqual(result.errors + result.failures, [])
        self.assertEqual(seen, [True])

    def test_admission_component_failure_is_private(self):
        import admission_gate as admission
        component = {key: hashed(key) for key in ('native_sha256', 'policy_sha256', 'component_seal_sha256', 'gate_index_sha256')}
        component['operator_image_config'] = 'sha256:' + hashed('image')
        receipt = {'identity_config': {'coop': 'on', 'adaptive-k': 'ema', 'dense-fp8': 'off'},
                   'component_qualification': component}
        with patch.object(admission, 'release_component', side_effect=ValueError(SECRET)):
            report = admission.evaluate(receipt, receipt)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertNotIn(SECRET, json.dumps(report))
        reason = next(row for row in report['findings'] if row.startswith('release component qualification refused:'))
        private = Path(reason.split('Private diagnostics (do not share): ', 1)[1].splitlines()[0])
        self.assertIn(SECRET, private.read_text())
        self.assertEqual(private.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
