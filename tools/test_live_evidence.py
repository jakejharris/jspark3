#!/usr/bin/env python3
"""Offline rejection tests for the sealed live evidence; no network or models."""
import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import validate_live_evidence as live
import validate_release as release

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    receipt = live.validate(ROOT)
    assert receipt['long_context_witness']['prompt_tokens'] == 48957
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        shutil.copytree(ROOT / 'recipe', root / 'recipe')
        (root / 'manifests').mkdir()
        shutil.copyfile(ROOT / 'manifests/release.json', root / 'manifests/release.json')
        evidence_dir = Path(live.EVIDENCE_PATH).parent
        shutil.copytree(ROOT / evidence_dir, root / evidence_dir)
        evidence = json.loads((root / live.EVIDENCE_PATH).read_text())
        changes = [
            (('verification', 'status'), 'VERIFY_REFUSED'),
            (('verification', 'long_context_witness', 'prompt_tokens'), 32768),
            (('verification', 'long_context_witness', 'completion_tokens'), 19),
            (('verification', 'long_context_witness', 'min_completion_tokens'), 1),
            (('verification', 'long_context_witness', 'code_word_verbatim'), False),
            (('verification', 'long_context_witness', 'request_payload_sha256'), '0' * 64),
            (('verification', 'focused_witness', 'median_decode_tok_s'), 999),
            (('verification', 'focused_witness', 'request_window_rank_causality'), []),
            (('verification', 'runtime_identity', 0, 'transform_target_set_sha256'), '0' * 64),
            (('verification', 'runtime_identity', 1, 'cadence_b45', 'capture_receipts'), 15),
            (('verification', 'image_and_safety', 2, 'cgroup', 'swap_current'), 1),
            (('verification', 'load', 'startup_complete'), False),
            (('verification', 'candidate_recipe_manifest_sha256'), '0' * 64),
            (('provenance', 'verifier_commit'), evidence['provenance']['candidate_commit']),
            (('provenance', 'final_archive_cold_boot_tested'), True),
            (('limits', 'pi_slowdown_status'), 'RESOLVED'),
            (('prior_verifier_refusals',), []),
        ]
        for keys, value in changes:
            changed = copy.deepcopy(evidence)
            parent = changed
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            (root / live.EVIDENCE_PATH).write_text(json.dumps(changed))
            try:
                live.validate(root)
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted drift: {keys}')
        (root / live.EVIDENCE_PATH).write_text(json.dumps(evidence))
        # Byte drift must fail even when the checksum manifest is untouched.
        path = root / 'recipe/scripts/container_entry.sh'
        path.write_text(path.read_text() + '\n# changed serving bytes\n')
        try:
            live.validate(root)
        except ValueError as exc:
            assert 'package recipe bytes drifted' in str(exc)
        else:
            raise AssertionError('accepted untested serving bytes')
        shutil.copyfile(ROOT / 'recipe/scripts/container_entry.sh', path)
        # A notice fix with a refreshed manifest keeps the pin; the same edit to a
        # serving file with a refreshed manifest does not.
        manifest = root / 'recipe/SHA256SUMS'
        for name, accepted in (('THIRD_PARTY_NOTICES.md', True), ('scripts/container_entry.sh', False)):
            path = root / 'recipe' / name
            path.write_text(path.read_text() + '\n# refreshed\n')
            manifest.write_text(release.sums(root, root / 'recipe', {'SHA256SUMS'}))
            try:
                live.validate(root)
            except ValueError:
                assert not accepted, f'refused a notice-only edit: {name}'
            else:
                assert accepted, f'accepted untested serving bytes with a refreshed manifest: {name}'
            shutil.copyfile(ROOT / 'recipe' / name, path)
            shutil.copyfile(ROOT / 'recipe/SHA256SUMS', manifest)
        # Removing evidence cannot leave a PASS claim admitted by the full validator.
        (root / live.EVIDENCE_PATH).unlink()
        report = release.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            release.check_current_claims(root, report)
        assert report.failed == 1
    print(f'PASS live evidence: accepted receipt, {len(changes)} gate/provenance mutations refused, '
          'serving-byte drift refused, notice-only edit accepted, missing evidence fails current claims')


if __name__ == '__main__':
    main()
