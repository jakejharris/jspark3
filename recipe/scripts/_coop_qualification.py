# SPDX-License-Identifier: Apache-2.0
"""Component seal checks shared by preparation, preflight and boot.

A self hash detects corruption; the release pin is the authority. Operator image
receipts prove eligibility, never GPU qualification. Raw evidence stays with the
owner; its complete, reviewed index is shipped alongside the pinned BUILD.
"""
import hashlib
import json
from pathlib import Path
import re

from _image_identity import build_policy, canonical, read_operator_record, verify_operator_record, sha

TARGET_NATIVE = '3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07'
RECIPE = Path(__file__).resolve().parents[1]
LEGACY_RECORD_SHA256 = '1f81dd90014951faad19d3eb15dcc1c2c7607dbbe0f268cef3bf65058f8295bf'


def need(ok, message):
    if not ok:
        raise ValueError('cooperative qualification: ' + message)


def digest_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hash_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) and len(set(value)) > 1


def verify_builder_host(host):
    """Validate operator-attested QUALIFICATION build fields, not their authenticity.

    A trusted operator collects these for a two-board determinism check. Receipt
    self-hashes detect drift; they do not authenticate edits or prove execution
    against a malicious operator. Outsiders reproduce and compare TARGET_NATIVE.
    Field descriptions: docs/COOP_REPRODUCIBILITY.md, Builder evidence fields.
    """
    need(isinstance(host, dict) and set(host) == {'architecture', 'machine_id_sha256', 'physical_identity'}
         and host['architecture'] in ('aarch64', 'x86_64')
         and hash_ok(host['machine_id_sha256']), 'builder host identity malformed; rebuild old receipts')
    physical = host['physical_identity']
    need(physical is None or (isinstance(physical, dict)
         and set(physical) == {'kind', 'uuid_sha256'} and physical['kind'] == 'gb10-gpu-uuid'
         and hash_ok(physical['uuid_sha256'])), 'physical builder identity malformed')


def independent_builders(first, second):
    for host in (first, second):
        verify_builder_host(host)
        need(host['architecture'] == 'aarch64' and host['physical_identity'] is not None,
             'independent builds require native ARM64 GB10 physical identities')
    # machine-id is diagnostic only: cloned operating systems share it. GB10's
    # integrated GPU identifies the physical board, independent of OS/hostname.
    need(first['physical_identity'] != second['physical_identity'], 'second physical machine required')


def regular(root, relative):
    rel = Path(relative)
    need(not rel.is_absolute() and '..' not in rel.parts and rel.parts, 'unsafe evidence path')
    path = root / rel
    need(not root.is_symlink() and all(not p.is_symlink() for p in [path, *path.parents]
         if p == root or root in p.parents), 'symlink in evidence path')
    need(path.is_file() and path.resolve().is_relative_to(root.resolve()), 'missing evidence: ' + relative)
    return path


def read(path):
    regular(path.parent, path.name)
    value = json.loads(path.read_text())
    need(isinstance(value, dict), 'expected object')
    return value


def verify_legacy_record(record):
    need(digest_value(record) == LEGACY_RECORD_SHA256, 'historical record differs; legacy seals cannot be reminted')


def gate_names():
    result = {}
    for rank in range(3):
        for geometry in range(3):
            suffix = f'r{rank}-g{geometry}'
            for mode in ('baseline', 'perturb'):
                result[f'h1-{suffix}-{mode}'] = 'h1'
            result[f'profile-{suffix}'] = 'profile'
            for tool in ('memcheck', 'racecheck'):
                result[f'smoke-{suffix}-{tool}'] = 'sanitizer'
                if geometry == 2:
                    result[f'geometry2-{suffix}-{tool}'] = 'sanitizer'
        result[f'policy-r{rank}'] = 'policy'
    return result


def compiled_inputs(coop):
    paths = [coop / 'build_repro.sh', *sorted((coop / 'source/native').rglob('*')),
             *sorted((coop / 'source/vendor').rglob('*'))]
    return {p.relative_to(coop).as_posix(): sha(regular(coop, p.relative_to(coop).as_posix()))
            for p in paths if p.is_file()}


def verify_record(record, bundle, coop, *, release=True, operator=None):
    from _coop_bundle import verify_bundle, verify_manifest, verify_selection
    keys = {'schema_version', 'source_manifest_sha256', 'qualification_source_manifest_sha256',
            'compiled_inputs', 'builder_sha256', 'build_policy_sha256', 'qualification_image',
            'helper', 'sanitizer', 'checkpoint', 'reproducibility', 'bundle', 'gate_index_sha256'}
    need(set(record) == keys and record['schema_version'] == 2, 'record schema drift')
    need(record['source_manifest_sha256'] == sha(coop / 'SOURCE_MANIFEST.json'), 'source drift')
    need(record['compiled_inputs'] == compiled_inputs(coop), 'compiled inputs drift')
    need(record['builder_sha256'] == sha(coop / 'build_repro.sh'), 'builder drift')
    need(record['build_policy_sha256'] == digest_value(build_policy()), 'image build policy drift')
    image = record['qualification_image']
    verify_operator_record(image)
    helper = read(RECIPE / 'config/coop-helper.json')
    need(record['helper'] == helper, 'helper provenance drift')
    need(record['sanitizer'] == read(RECIPE / 'config/coop-sanitizer.json'), 'sanitizer provenance drift')
    checkpoint = record['checkpoint']
    from _coop_checkpoint import authority
    need(checkpoint == authority(), 'checkpoint authority differs')
    for key in ('qualification_source_manifest_sha256', 'gate_index_sha256'):
        need(hash_ok(record[key]), 'missing qualification hash: ' + key)
    rep = record['reproducibility']
    need(isinstance(rep, dict) and set(rep) == {'runs', 'comparison', 'binary_sha256', 'evidence_sha256'}
         and rep['runs'] >= 2 and rep['comparison'] == 'bit-identical'
         and rep['binary_sha256'] == TARGET_NATIVE and hash_ok(rep['evidence_sha256']), 'build evidence')
    observed_bundle = verify_bundle(bundle, coop, native_required=False)
    manifest = read(bundle / 'manifest.json')
    files = manifest['files']
    need(record['bundle'] == {'manifest_sha256': sha(bundle / 'manifest.json'),
         'native_sha256': TARGET_NATIVE, 'runtime_sha256': files['runtime.py'],
         'dispatch_policy_sha256': files['dispatch_policy.json']}, 'bundle identity')
    need(files['cooperative_moe.so'] == TARGET_NATIVE, 'wrong native digest')
    index = read(coop / 'QUALIFICATION.json')
    need(sha(coop / 'QUALIFICATION.json') == record['gate_index_sha256'], 'gate index hash drift')
    need(index.get('schema_version') == 1 and index.get('status') == 'PASS'
         and index.get('native_sha256') == TARGET_NATIVE
         and index.get('source_manifest_sha256') == record['qualification_source_manifest_sha256']
         and index.get('image_receipt_sha256') == image['payload_sha256']
         and index.get('helper') == helper and index.get('checkpoint') == checkpoint
         and index.get('sanitizer') == record['sanitizer'], 'gate index binding')
    raw_manifest = index.get('raw_bundle_manifest', {})
    verify_manifest(raw_manifest, coop)
    verify_selection(raw_manifest, manifest)
    raw = index.get('raw_bundle', {})
    need(set(raw) == set(observed_bundle) and hash_ok(raw.get('manifest_sha256'))
         and raw.get('native_sha256') == TARGET_NATIVE
         and raw.get('runtime_sha256') == observed_bundle['runtime_sha256']
         and raw.get('dispatch_policy_sha256') == raw_manifest['files']['dispatch_policy.json']
         and index.get('selected_bundle') == observed_bundle, 'original/selected qualification bundle differs')
    independent = index.get('independent_build', {})
    need(set(independent) == {'native_receipt', 'image_receipt'}, 'second-machine evidence missing')
    verify_operator_record(independent['image_receipt'])
    first = index.get('qualification_build', {})
    second = independent['native_receipt']
    for build, build_image in ((first, image), (second, independent['image_receipt'])):
        inputs = build.get('build_inputs', {})
        prefix = 'recipe/overlays/v16/coop/'
        need(build.get('payload_sha256') == digest_value({k:v for k,v in build.items() if k != 'payload_sha256'})
             and build.get('image_receipt_sha256') == build_image['payload_sha256']
             and build.get('source_recipe_sha256') == build_image['source_recipe_sha256']
             and build.get('reproducibility') == {'runs': 2, 'comparison': 'bit-identical'}
             and build.get('schema_version') == 2 and build.get('verification') == 'fixed-native-build-v2'
             and build.get('hardware_qualified') is False
             and build.get('binary_sha256', {}).get('recipe/overlays/v16/coop/bundle/cooperative_moe.so') == TARGET_NATIVE
             and build.get('builder_host', {}).get('architecture') == 'aarch64', 'invalid native build evidence')
        need(all(inputs.get(prefix + name) == expected for name, expected in record['compiled_inputs'].items())
             and inputs.get(prefix + 'SOURCE_MANIFEST.json') == record['qualification_source_manifest_sha256']
             and inputs.get(prefix + 'source/runtime.py') == raw['runtime_sha256']
             and inputs.get(prefix + 'source/dispatch_policy.json') == raw['dispatch_policy_sha256'],
             'native receipt differs from qualified source/builder inputs')
    independent_builders(first['builder_host'], second['builder_host'])
    need(record['reproducibility']['evidence_sha256'] == digest_value(first), 'qualification build receipt drift')
    gates = index.get('gates', {})
    need(set(gates) == set(gate_names()), 'missing/extra component gates')
    for name, kind in gate_names().items():
        row = gates[name]
        need(set(row) == {'kind', 'status', 'receipt_sha256', 'log_sha256', 'environment_sha256', 'execution_sha256', 'bundle'}
             and row['kind'] == kind and row['status'] == 'PASS'
             and row['bundle'] == (observed_bundle if kind == 'policy' else raw)
             and all(hash_ok(row[k]) for k in ('receipt_sha256', 'log_sha256', 'environment_sha256', 'execution_sha256')), 'incomplete gate: ' + name)
    profiles = index.get('profile_log_sha256', {})
    need(set(profiles) == {f'rank{r}-geo{g}.jsonl' for r in range(3) for g in range(3)}
         and all(hash_ok(h) for h in profiles.values()), 'nine profile hashes required')
    policy = read(bundle / 'dispatch_policy.json')
    need(policy.get('profile_log_sha256') == [profiles[name] for name in sorted(profiles)] and policy.get('native_sha256') == TARGET_NATIVE,
         'policy was not selected from these profiles')
    if release:
        pin = read(RECIPE / 'config/coop-release.json')
        need(pin == {'schema_version': 1, 'status': 'qualified', 'native_sha256': TARGET_NATIVE,
                     'build_sha256': digest_value(record), 'gate_index_sha256': record['gate_index_sha256']},
             'release component seal is pending or differs')
        need((coop / 'source/dispatch_policy.json').read_bytes() == (bundle / 'dispatch_policy.json').read_bytes(),
             'source seed policy differs from measured bundle')
    if operator is not None:
        read_operator_record(operator)
    return {'native_sha256': TARGET_NATIVE, 'policy_sha256': files['dispatch_policy.json'],
            'component_seal_sha256': digest_value(record), 'gate_index_sha256': record['gate_index_sha256']}
