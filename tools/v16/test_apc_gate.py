#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline APC checker controls: derived logprob sabotage and the cold/cold nondeterminism ceiling."""
import copy
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import apc_gate as apc


def probe(lps, first=0, cached=0):
    """64 greedy tokens of a synthetic probe; every top-20 row also holds one alternative token."""
    ids = [first] + list(range(1, apc.PROBE_TOKENS))
    return {'token_ids': ids, 'token_strs': [f't{i}' for i in ids], 'logprobs': list(lps),
            'top_logprobs': [{f't{i}': lp, 'alt': lp - 3.0} | ({'t0': lp - 1.0} if i == 7 else {})
                             for i, lp in zip(ids, lps)],
            'cached_tokens': cached, 'ttft_s': 1.0, 'prefill_s': 1.0}


def capture(cold_noise=0.3, warm_offset=0.2):
    """A passing finehit capture: cold[1] differs from cold[0] by cold_noise at token 3 of every
    cell, warm differs from cold[0] by warm_offset at token 0."""
    cells = apc.cells_from({k: list(range(17000)) for k in ('prose', 'tools', 'long')})
    apc.validate_cells(cells)
    base = [-0.5] * apc.PROBE_TOKENS
    noisy = base[:3] + [-0.5 - cold_noise] + base[4:]
    rows = [{'id': cell['id'], 'producer': {'cached_tokens': 0},
             'cold': [probe(base), probe(noisy), probe(base)],
             'warm': probe([-0.5 - warm_offset] + base[1:], cached=cell['expected_hit'])} for cell in cells]
    return {'cells': cells, 'rows': rows, 'expect': 'finehit', 'requests_delta': len(rows) * 5,
            'apc_identity': [['6', '']]}


class DerivedSabotageTests(unittest.TestCase):
    def test_passing_capture_derives_a_sabotage_that_clears_the_limit_by_the_margin(self):
        doc = capture()
        result, controls, plan = apc.gate(doc)
        self.assertEqual((result['verdict'], result['findings']), ('PASS', []))
        self.assertEqual(len(controls), 8)
        self.assertEqual(set(controls.values()), {'FAIL'})
        self.assertEqual((plan['cell'], plan['limit'], plan['reason']), ('pi-prose', 0.6, None))
        self.assertAlmostEqual(plan['size'], 0.6 + apc.SABOTAGE_MARGIN + 0.2)
        broken = copy.deepcopy(doc)
        broken['rows'][0]['warm']['logprobs'][0] -= plan['size']
        after = apc.analyze(broken)
        self.assertGreaterEqual(after['warm_spread'], after['spread_limit'] + apc.SABOTAGE_MARGIN - 1e-6)
        self.assertEqual(after['verdict'], 'FAIL')

    def test_offset_in_either_direction_still_clears_the_margin(self):
        for offset in (0.25, -0.25):
            with self.subTest(offset=offset):
                result, controls, plan = apc.gate(capture(warm_offset=offset))
                self.assertEqual((result['verdict'], controls['warm_logprob']), ('PASS', 'FAIL'))
                self.assertAlmostEqual(plan['offset'], abs(offset))

    def test_sabotage_larger_than_the_reviewed_corruption_is_not_discriminating(self):
        # Cold noise 1.9 stays under the ceiling (limit 3.8), but a 0.8 first-token offset needs a
        # 5.1-nat corruption: over the 5.0 bound, so APC refuses instead of using a bigger one.
        doc = capture(cold_noise=1.9, warm_offset=0.8)
        self.assertEqual(apc.analyze(copy.deepcopy(doc))['verdict'], 'PASS')
        result, controls, plan = apc.gate(doc)
        self.assertEqual(controls['warm_logprob'], 'NOT_DISCRIMINATING')
        self.assertTrue(plan['reason'] and plan['size'] > apc.SABOTAGE_MAX)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual([f for f in result['findings'] if f.startswith('sabotage not discriminating:')],
                         ['sabotage not discriminating: ' + plan['reason']])
        self.assertFalse(any('checker negative control passed' in f for f in result['findings']))

    def test_no_shared_first_token_is_not_discriminating(self):
        doc = capture()
        for row in doc['rows']:
            # cold[0] alone picks another first token: no row has shared history to corrupt.
            row['cold'][0] = probe(row['cold'][0]['logprobs'], first=7)
        self.assertEqual(apc.analyze(copy.deepcopy(doc))['verdict'], 'PASS')
        result, controls, plan = apc.gate(doc)
        self.assertEqual((controls['warm_logprob'], plan['cell']), ('NOT_DISCRIMINATING', None))
        self.assertIn('sabotage not discriminating: no warm first token matches its first cold probe', result['findings'])
        self.assertEqual(result['verdict'], 'FAIL')

    def test_corruption_that_does_not_clear_the_margin_is_not_discriminating(self):
        original = apc.sabotage
        undersized = lambda doc: dict(original(doc), size=original(doc)['limit'] / 2)  # noqa: E731
        with patch.object(apc, 'sabotage', side_effect=undersized):
            result, controls, _ = apc.gate(capture())
        self.assertEqual((controls['warm_logprob'], result['verdict']), ('NOT_DISCRIMINATING', 'FAIL'))
        self.assertIn('sabotage not discriminating: corrupted spread did not clear the limit by the margin',
                      result['findings'])

    def test_checker_that_ignores_the_spread_still_fails_its_control(self):
        original = apc.parity
        def blind(rows):
            result = original(rows)
            result['findings'] = [f for f in result['findings'] if 'warm/cold logprob spread' not in f]
            return result
        with patch.object(apc, 'parity', side_effect=blind):
            result, controls, _ = apc.gate(capture())
        self.assertEqual(controls['warm_logprob'], 'PASS')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any(f.startswith('checker negative control passed') for f in result['findings']))


class ColdCeilingTests(unittest.TestCase):
    def test_cold_spread_above_the_ceiling_refuses_as_nondeterminism_without_widening(self):
        doc = capture(cold_noise=2.683196)   # the largest retained cold/cold spread
        result, controls, plan = apc.gate(doc)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('nondeterminism: cold/cold logprob spread 2.683 > 2.000 ceiling', result['findings'])
        self.assertEqual(result['spread_limit'], 2 * apc.COLD_SPREAD_CEILING)
        self.assertNotIn('PASS', controls.values())
        self.assertLessEqual(plan['size'], apc.SABOTAGE_MAX)

    def test_ceiling_is_exclusive(self):
        result, controls, _ = apc.gate(capture(cold_noise=apc.COLD_SPREAD_CEILING))
        self.assertEqual((result['verdict'], set(controls.values())), ('PASS', {'FAIL'}))
        self.assertEqual(result['spread_limit'], 2 * apc.COLD_SPREAD_CEILING)

    def test_without_the_ceiling_the_old_fixed_corruption_is_refused_as_not_discriminating(self):
        # The pre-fix shape: a 5.366 limit under which a fixed 5.0-nat corruption passed.
        doc = capture(cold_noise=2.683196, warm_offset=0.161336)
        with patch.object(apc, 'COLD_SPREAD_CEILING', float('inf')):
            fixed = copy.deepcopy(doc)
            fixed['rows'][0]['warm']['logprobs'][0] -= 5.0
            self.assertEqual(apc.analyze(fixed)['verdict'], 'PASS')
            result, controls, plan = apc.gate(doc)
        self.assertEqual((controls['warm_logprob'], result['verdict']), ('NOT_DISCRIMINATING', 'FAIL'))
        self.assertAlmostEqual(plan['size'], 5.366392 + apc.SABOTAGE_MARGIN + 0.161336)


class NonFiniteTests(unittest.TestCase):
    """Rows 1+ are never the sabotaged cell (pi-prose, row 0), so only the finite check can refuse."""
    def assert_refused(self, doc):
        result, controls, plan = apc.gate(doc)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(result['findings'][0].startswith('non-finite chosen-token logprobs in '))
        self.assertIn('sabotage not discriminating: ' + plan['reason'], result['findings'])
        self.assertEqual((controls['warm_logprob'], plan['size']), ('NOT_DISCRIMINATING', None))
        self.assertNotIn('PASS', controls.values())

    def test_nan_outside_the_sabotage_cell_refuses(self):
        doc = capture()
        run = doc['rows'][1]['cold'][1]  # the review's reproduction
        run['logprobs'][0] = float('nan')
        run['top_logprobs'][0][run['token_strs'][0]] = float('nan')
        self.assert_refused(doc)

    def test_equal_infinities_across_a_cell_refuse(self):
        for value in (float('inf'), float('-inf')):
            with self.subTest(value=value):
                doc = capture()
                row = doc['rows'][1]
                for run in (row['warm'], *row['cold']):
                    run['logprobs'][0] = value
                    run['top_logprobs'][0][run['token_strs'][0]] = value
                self.assert_refused(doc)

    def test_non_finite_or_non_numeric_anywhere_refuses(self):
        # Inside and beyond shared histories, warm and cold, including the last token.
        for row, run, position, value in ((5, 'warm', 40, float('nan')), (13, 2, 63, float('inf')),
                                          (3, 0, 10, float('-inf')), (8, 1, 3, None)):
            with self.subTest(row=row, run=run, position=position, value=value):
                doc = capture()
                probe_run = doc['rows'][row]['warm'] if run == 'warm' else doc['rows'][row]['cold'][run]
                probe_run['logprobs'][position] = value
                self.assert_refused(doc)


if __name__ == '__main__':
    unittest.main()
