"""Offline rejection checks for C4 publication evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from validate_c4_followup import EVIDENCE, validate

ROOT = Path(__file__).resolve().parents[1]


class FollowupTests(unittest.TestCase):
    def test_valid_receipts(self):
        claims = validate(ROOT)
        self.assertEqual(claims['display']['v11.c4.original.aggregate_decode_tok_s.median'], '223.14')

    def test_rejects_selection_pooling_and_receipt_drift(self):
        for mutation in ('selected_take', 'instrumented_substitution', 'dropped_repeat', 'receipt_bytes'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / EVIDENCE, root / EVIDENCE)
                shutil.copyfile(ROOT / 'results/results.json', root / 'results/results.json')
                path = root / EVIDENCE / 'CLAIMS.json'
                claims = json.loads(path.read_text())
                if mutation == 'receipt_bytes':
                    receipt = root / EVIDENCE / 'ORIGINAL-RESULTS.json'
                    receipt.write_text(receipt.read_text() + ' ')
                elif mutation == 'dropped_repeat':
                    claims['series']['original']['aggregate_decode_tok_s'].pop(0)
                else:
                    value = 248.61 if mutation == 'selected_take' else 231.90
                    claims['series']['original']['median_aggregate_decode_tok_s'] = value
                    claims['display']['v11.c4.original.aggregate_decode_tok_s.median'] = f'{value:.2f}'
                path.write_text(json.dumps(claims))
                with self.assertRaises(AssertionError):
                    validate(root)


if __name__ == '__main__':
    unittest.main()
