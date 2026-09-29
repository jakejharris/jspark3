#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Public attestation of a private admission: bound receipts, exact private set."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/v16')]
import _qualification as qualification
from v16_common import sha256_json

ADMISSION = 'release/v1.8.4-admission'
REFUSED = (AssertionError, subprocess.CalledProcessError, KeyError, ValueError)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, indent=2):
    Path(path).write_text(json.dumps(value, indent=indent, sort_keys=True) + '\n')


class AttestationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
        for name in ('tools', 'recipe', ADMISSION):
            shutil.copytree(ROOT / name, self.root / name, ignore=ignore)
        self.admission = self.root / ADMISSION
        self.attestation = self.admission / 'attestation.json'

    def final(self):
        return read(self.admission / 'finalize.json')

    def verify(self):
        qualification.verify_admission(self.root, self.admission, self.final())

    def refused(self, pattern=None, kind=REFUSED):
        with self.assertRaises(kind) as caught:
            self.verify()
        if pattern:
            self.assertRegex(str(caught.exception), pattern)

    def edit_attestation(self, change):
        record = read(self.attestation)
        change(record)
        write(self.attestation, record)

    def rebind(self, first_change=None, final_change=None, first_indent=2):
        """Rewrite receipts consistently, as a producer would, without private bytes."""
        first = read(self.admission / 'first-prompt.json')
        if first_change:
            first_change(first)
        write(self.admission / 'first-prompt.json', first, first_indent)
        digest = sha(self.admission / 'first-prompt.json')
        final = self.final()
        final['evidence_sha256']['first-prompt.json'] = digest
        final['input_hashes']['client_evidence'][0]['sha256'] = digest
        if final_change:
            final_change(final)
        final['payload_sha256'] = sha256_json({k: v for k, v in final.items() if k != 'payload_sha256'})
        write(self.admission / 'finalize.json', final)
        if self.attestation.exists():
            self.edit_attestation(lambda r: r.update(input_sha256={
                'first_prompt': digest, 'finalize': sha(self.admission / 'finalize.json')}))

    def test_shipped_attestation_passes(self):
        self.verify()
        declared = read(self.attestation)['private_evidence_sha256']
        self.assertEqual(len(declared), 11)
        self.assertEqual(sum(name.endswith('.log') for name in declared), 8)
        for name, digest in declared.items():
            self.assertFalse((self.admission / name).exists())
            self.assertEqual(self.final()['evidence_sha256'][name], digest)

    def test_shipped_dependency_bytes_are_bound(self):
        path = self.admission / 'hygiene-rank0.json'
        path.write_bytes(path.read_bytes() + b' ')
        self.refused('operator evidence changed')

    def test_unshipped_set_must_equal_declared_private_set(self):
        (self.admission / 'verify-final.json').unlink()
        self.refused('declared private set')

    def test_declared_private_set_cannot_hide_or_add_names(self):
        self.edit_attestation(lambda r: r['private_evidence_sha256'].pop('apc.log'))
        self.refused('declared private set')
        self.setUp()
        self.edit_attestation(lambda r: r['private_evidence_sha256'].update(
            {'hygiene-rank0.json': sha(self.admission / 'hygiene-rank0.json')}))
        self.refused('declared private set')

    def test_declared_hash_must_equal_receipt_hash(self):
        self.edit_attestation(lambda r: r['private_evidence_sha256'].update({'apc.json': '0' * 64}))
        self.refused('declared private evidence hash differs')

    def test_missing_attestation_refuses_unshipped_evidence(self):
        self.attestation.unlink()
        self.refused('declared private set')

    def test_unbound_admission_file_refused(self):
        write(self.admission / 'notes.json', {'verdict': 'PASS'})
        self.refused('unbound admission file')

    def test_attestation_record_fields_are_checked(self):
        for change in (lambda r: r.update(verdict='FAIL'),
                       lambda r: r.update(findings=['admission evidence refused']),
                       lambda r: r['input_sha256'].update(finalize='0' * 64),
                       lambda r: r.update(gate_sha256='0' * 64),
                       lambda r: r.update(schema='other'),
                       lambda r: r.update(source_commit='adda4a1'),
                       lambda r: r.update(extra=True)):
            with self.subTest(change=change):
                self.setUp()
                self.edit_attestation(change)
                self.refused()

    def test_shipped_qualification_tools_must_match_receipt(self):
        path = self.root / 'tools/v16/apc_gate.py'
        path.write_text(path.read_text() + '\n# drift\n')
        self.refused('qualification tools differ')

    def test_receipt_level_gate_checks_still_run(self):
        # Control: a consistent rebinding with different bytes still passes.
        self.rebind(first_indent=1)
        self.verify()
        for first_change, final_change in (
                (lambda f: f.update(verdict='FAIL'), None),
                (lambda f: f['identity_config'].update(coop='off'), None),
                (lambda f: f.update(stock_profile={'ABLIT': '1', 'APC': '1', 'profile': 'production-stock'}), None),
                (lambda f: f.update(manifest_sha256='0' * 64), None),
                (None, lambda f: f['input_hashes']['client_evidence'][0].update(sha256='0' * 64)),
                (lambda f: f['component_qualification'].update(policy_sha256='1' * 63 + '2'),
                 lambda f: f['component_qualification'].update(policy_sha256='1' * 63 + '2'))):
            with self.subTest(first=first_change, final=final_change):
                self.setUp()
                self.rebind(first_change, final_change)
                self.refused(kind=subprocess.CalledProcessError)

    def test_receipt_payload_hash_still_checked(self):
        final = self.final()
        final['completed_at'] = '2026-01-01T00:00:00+00:00'
        write(self.admission / 'finalize.json', final)
        self.edit_attestation(lambda r: r['input_sha256'].update(finalize=sha(self.admission / 'finalize.json')))
        self.refused(kind=subprocess.CalledProcessError)

    def test_complete_public_evidence_uses_unchanged_operator_gate(self):
        private = read(self.attestation)['private_evidence_sha256']
        self.attestation.unlink()
        self.rebind(final_change=lambda f: [f['evidence_sha256'].pop(name) for name in private])
        self.verify()
        path = self.admission / 'triar-inactive-rank1.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        self.refused('operator evidence changed')
        path.write_bytes(original)
        # Only the unchanged operator gate inspects receipt verdicts on this path.
        self.rebind(lambda f: f.update(verdict='FAIL'))
        self.refused(kind=subprocess.CalledProcessError)


if __name__ == '__main__':
    unittest.main()
