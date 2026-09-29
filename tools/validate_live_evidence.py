#!/usr/bin/env python3
"""Validate the sanitized integrated receipt and its limited package equivalence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

EVIDENCE_PATH = 'results/evidence/candidate/cadence-v11/LIVE-VERIFY.json'
# Recipe legal text: it never executes, so it is outside the live-tested byte pin.
NOTICE_FILES = ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'REQUIRED_ATTRIBUTION.md')


def validate(root: Path) -> dict:
    evidence = json.loads((root / EVIDENCE_PATH).read_text())

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError('live verification: ' + message)

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def sums(data: bytes) -> dict[str, str]:
        rows = [line.split('  ', 1) for line in data.decode().splitlines()]
        require(len({name for _, name in rows}) == len(rows), 'duplicate recipe entry')
        return {name: value for value, name in rows}

    source, provenance = evidence['source'], evidence['provenance']
    require(evidence['schema_version'] == 1, 'export schema drift')
    require(source['raw_receipt_sha256'] ==
            '1fbba426b8ac7ef9dc7b019241a70351335e82ebc8064c92a19c310e7db628d6',
            'accepted original receipt identity drift')
    require(source['raw_receipt_retained_privately'] is True, 'original receipt retention absent')
    require(provenance['candidate_commit'] == 'a729583fc1e286583023f4b9b92db87efefcb33a' and
            provenance['verifier_commit'] == '456a2624d0c54e2096327b57fcb54e8333f10114',
            'candidate/verifier provenance drift')
    require(provenance['final_archive_cold_boot_tested'] is False, 'unproven archive cold-start claim')
    release = json.loads((root / 'manifests/release.json').read_text())['verification_evidence']
    require(release['status'] == 'INTEGRATED_LIVE_PASS' and release['record'] == EVIDENCE_PATH and
            release['candidate_commit'] == provenance['candidate_commit'] and
            release['verifier_commit'] == provenance['verifier_commit'] and
            release['final_archive_cold_boot_tested'] is False, 'release evidence binding drift')
    require(evidence['prior_verifier_refusals'] == [{
        'raw_receipt_sha256': 'e2ee2a155c76f400de5bb564c4c0241d03dd67d5984d07894928d74cdece80b7',
        'host_verifier_commit': 'e58c825cde635c6ac6f2366f36df72a4ae0e2fe9',
        'status': 'VERIFY_REFUSED', 'reason': 'rank0 runtime-view/transform identity drift',
        'raw_receipt_retained_privately': True,
    }], 'prior verifier refusal must remain recorded')
    limits = evidence['limits']
    require(limits['pi_slowdown_status'] == 'UNRESOLVED' and
            all(limits[k] is False for k in ('configured_context_verified',
                'sustained_concurrency_certified', 'independent_fleet_reproduction')),
            'unsupported operating-envelope or Pi claim')
    receipt = evidence['verification']
    require(receipt['schema_version'] == 1 and receipt['status'] == 'VERIFY_PASS' and
            receipt['grade'] == 'ENGINEERING-EVIDENCE', 'integrated pass absent')
    require('payload_sha256' not in receipt and 'manifest_sha256' not in receipt,
            'sanitized projection must not impersonate the original receipt')
    require(provenance['candidate_recipe_manifest_file'] == 'CANDIDATE-SHA256SUMS',
            'candidate manifest path drift')
    candidate_bytes = (root / EVIDENCE_PATH).with_name('CANDIDATE-SHA256SUMS').read_bytes()
    verifier_bytes = (root / 'recipe/SHA256SUMS').read_bytes()
    require(digest(candidate_bytes) == receipt['candidate_recipe_manifest_sha256'] ==
            '32784962571dfaa21634d8d290b011b3bbbe36ac8bdeaae0ceff5bd38fed7627',
            'candidate recipe binding drift')
    candidate, verifier = sums(candidate_bytes), sums(verifier_bytes)
    # Legal text never executes: a notice fix may change these rows. Every other row
    # must still rebuild the exact live-tested manifest, byte for byte.
    tested = ''.join(f'{candidate[name] if name in NOTICE_FILES else value}  {name}\n'
                     for value, name in (line.split('  ', 1) for line in verifier_bytes.decode().splitlines()))
    require(digest(tested.encode()) == receipt['verifier_recipe_manifest_sha256'] ==
            'afd1fca1f58bcece36a441e2479bc41f39504fcc0c4cad45ae9905e629e62a43',
            'package recipe differs from the live-tested host verifier recipe')
    require(candidate.keys() == verifier.keys(), 'recipe inventory drift')
    require([name for name in candidate if candidate[name] != verifier[name] and name not in NOTICE_FILES] ==
            provenance['changed_recipe_files'] == ['scripts/fleetctl.py'],
            'candidate/package recipe delta exceeds the verifier controller')
    for name, expected in verifier.items():
        require(digest((root / 'recipe' / name).read_bytes()) == expected,
                f'package recipe bytes drifted: {name}')
    require(receipt['health_http'] == 200 and receipt['served_model'] == 'glm-5.3-flash' and
            receipt['arithmetic'] == 323, 'health/model/arithmetic gate drift')
    require(set(receipt['load']) == {'target_shards_120', 'draft_shards_1',
            'cadence_capture_evidence', 'draft_graphs_5', 'startup_complete',
            'b5_controller_calibrated'} and all(v is True for v in receipt['load'].values()),
            'load gate drift')
    for key in ('runtime_identity', 'image_and_safety'):
        require([r['rank'] for r in receipt[key]] == [0, 1, 2], f'{key} rank set drift')
    for row in receipt['runtime_identity']:
        require(row['image_receipt_bound'] is True and 'image_receipt_sha256' not in row,
                'image receipt binding or redaction drift')
        require(row['target_runtime_config'] ==
                '55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd' and
                row['draft_runtime_config'] ==
                'c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f' and
                row['transform_pipeline_state'] == 'ALREADY_APPLIED' and
                row['transform_target_set_sha256'] ==
                '1f3beb88157da0a7782cc94d49bc5c8d93103fa708b620f8b3fb51f110a8f635',
                'runtime identity drift')
        cadence = row['cadence_b45']
        require(cadence['modules_verified'] == 5 and cadence['kda_original_untouched'] is True and
                cadence['capture_dots_intact'] is True and cadence['capture_receipts'] >= 16 and
                cadence['serving_graph_dumps'] >= 8, 'Cadence capture gate drift')
    for row in receipt['image_and_safety']:
        cgroup = row['cgroup']
        require('name' not in row and row['running'] is True and row['oom_killed'] is False and
                row['restart_count'] == 0 and row['exit_code'] == 0 and row['restart_policy'] == 'no' and
                row['image_config_match'] is True and row['image_reference_match'] is True and
                row['actual_image_config'] ==
                'sha256:ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263' and
                row['memory_bytes'] == row['memory_swap_bytes'] == 68719476736 and
                cgroup['memory_max'] == '68719476736' and cgroup['swap_max'] == '0' and
                cgroup['swap_current'] == 0 and all(int(cgroup['events'][k]) == 0
                for k in ('oom', 'oom_kill', 'oom_group_kill')), 'image/safety gate drift')
    focused = receipt['focused_witness']
    require(focused['pass'] is True and focused['automatic_retries'] == 0 and
            focused['warmups'] == 1 and focused['scored_requests'] == len(focused['decode_tok_s']) == 3 and
            statistics.median(focused['decode_tok_s']) == focused['median_decode_tok_s'] and
            focused['median_decode_tok_s'] >= focused['admission_floor'] and
            focused['request_body_sha256'] ==
            '681a284f6d441734ccaade4d7b7c4731c736e7fb7ce87cabff1312326c5187a7',
            'focused witness gate or median drift')
    ranks = {'rank0': True, 'rank1': True, 'rank2': True}
    require(focused['all_three_ranks_observed'] == ranks and
            focused['request_window_rank_causality'] == [ranks] * 3, 'focused rank causality drift')
    witness = receipt['long_context_witness']
    require(witness['pass'] is True and witness['reasons'] == [] and
            witness['payload_pinned'] is True and witness['code_word_verbatim'] is True and
            witness['min_prompt_tokens'] == 32768 and witness['prompt_tokens'] > 32768 and
            witness['min_completion_tokens'] == 20 and witness['completion_tokens'] >= 20 and
            witness['finish_reason'] in ('stop', 'length') and
            witness['served_model'] == 'glm-5.3-flash' and witness['request_payload_sha256'] ==
            '882c504c49c626977c32e0ecb4035f29636a6c15ad1aeccdb31a38b6c7e8e652',
            'strict long-context witness gate drift')
    return receipt
