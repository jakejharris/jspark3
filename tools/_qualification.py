"""Offline qualification-policy and final recipe/configuration binding checks."""
import ast
import hashlib
import json
from pathlib import Path
import re


def load(root, rel):
    return json.loads((root / rel).read_text())


def verify(root, require_final=False):
    policy = load(root, 'manifests/qualification.json')
    assert policy['prefill_floor_tok_s'] == 1100
    assert policy['first_pass_canaries'] == 'recorded-only'
    assert policy['quick'] == 'diagnostic'
    assert policy['post_hygiene_canaries'] == 'blocking'
    assert policy['hardware_qualified'] is False
    assert load(root, 'manifests/stock-profile.json')['hardware_qualified'] is False
    tree = ast.parse((root/'tools/v16/prefill_gate.py').read_text())
    defaults = [kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                and any(isinstance(a, ast.Constant) and a.value == '--floor-tok-s' for a in node.args)
                for kw in node.keywords if kw.arg == 'default' and isinstance(kw.value, ast.Constant)]
    assert defaults == [1100.0]
    for path in [root/'README.md', root/'CHANGELOG.md', *list((root/'docs').glob('*.md')),
                 root/'release/RELEASE-NOTES.md']:
        text = path.read_text()
        assert not re.search(r'--floor-tok-s\s+1000\b|floor is 1000', text)
        assert not re.search(r'\bA0-' + r'Q\b|\bK' + r'1\b', text)
        assert '{{' not in text and '<!--' not in text and 'CONFIRM' not in text
    docs = load(root, 'manifests/docs-inputs.json')
    assert hashlib.sha256((root/'release/results.json').read_bytes()).hexdigest() == docs['results_sha256']
    numbers = load(root, 'release/results.json')
    cohort = numbers['sets']['base_m0_q']
    assert cohort['mode'] == 0 and cohort['serving_starts'] == 1 and cohort['sweeps'] == 2
    assert 'QA' in cohort['label'] and numbers['mode_switch'] == 'B'
    binding = load(root, 'manifests/final-binding.json')
    release = load(root, 'manifests/release.json')
    if release.get('release_version') == 'v1.8.4':
        verify_v184(root, binding, release, require_final)
        return
    derivation = load(root, 'manifests/derivation.json')
    final = require_final or release.get('stage') == 'final' or release.get('release_version') is not None
    assert binding['hardware_qualified'] is False
    assert binding['state'] in ('unselected', 'prepared', 'bound')
    if final:
        assert binding['state'] == 'bound', 'final recipe is unbound'
        assert release['stage'] == 'final'
        assert release['release_version'] == binding['release_version']
        assert re.fullmatch(r'v\d+\.\d+\.\d+', binding['release_version'])
        for key in ('selection_sha256', 'settings_record_sha256'):
            assert re.fullmatch('[0-9a-f]{64}', binding[key])
    elif binding['state'] == 'bound':
        raise ValueError('bound final lacks final release identity')
    if binding['state'] == 'unselected':
        assert set(binding) == {'state', 'hardware_qualified'} and release.get('release_version') is None
        return
    catalog = load(root, 'manifests/final-catalog.json')
    selected = catalog[binding['delivery_seal_sha256']]
    assert selected['variant'] == binding['variant']
    assert selected['recipe_sha256'] == binding['source_recipe_sha256']
    for key in ('candidate_manifest_sha256', 'source_inventory_sha256', 'sealed_profile_sha256'):
        assert binding[key] == derivation[key], 'candidate/profile/inventory binding mismatch'
    rows = derivation['files']
    # Every retained source recipe file must come from this delivery, including
    # executables after sanitization/repinning. Generated wrappers/docs retain
    # their explicit export origins and are independently checksummed.
    binaries = load(root, 'manifests/binaries.json')
    overrides = {'recipe/README.md', 'recipe/transforms/README.md',
                 'recipe/docs/LIMITATIONS.md', 'recipe/docs/REPRODUCIBILITY.md'}
    for name, expected in selected['recipe_inputs'].items():
        if name in overrides or Path(name).suffix == '.log' or name in binaries:
            continue
        assert name in rows, 'missing final recipe source'
        assert rows[name]['input_sha256'] == expected, 'mixed final recipe source'
    for name, row in binaries.items():
        assert row['expected_sha256'] == selected['recipe_inputs'][name], 'wrong final binary pin'
    env = dict(line.split('=', 1) for line in (root/'recipe/.env.example').read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    settings = binding['startup_settings']
    assert settings == selected['startup_settings'], 'profile differs from sealed final variant'
    assert all(env.get(k) == v for k, v in settings.items()), 'final settings disagree with example'
    assert settings['ABLIT'] == '0' and settings['JSPARK3_V16_PROFILE'] == 'production-stock'
    assert settings.get('JSPARK3_TRIAR','0') == ('0' if binding['variant'] == 'c' else '1')
    assert settings.get('JSPARK3_V18_THIRDS','0') == ('1' if binding['variant'] == 'a' else '0')
    if final:
        runtime = binding['runtime_settings']
        assert set(runtime) == {'triar','adaptive','b5'}
        assert runtime['triar'] in ('off','on') and runtime['adaptive'] in ('ema','off') and runtime['b5']=='default'
        assert binding['variant'] != 'c' or runtime == {'triar':'off','adaptive':'ema','b5':'default'}


def verify_v184(root, binding, release, require_final):
    """Pending fields are null, never invented hashes or inherited GPU evidence."""
    import sys
    sys.path.insert(0, str(root / 'recipe/scripts'))
    from _coop_qualification import TARGET_NATIVE
    from _coop_checkpoint import authority
    authority()
    assert binding['schema_version'] == 2 and binding['hardware_qualified'] is False
    assert binding['release_version'] == 'v1.8.4' and binding['native_sha256'] == TARGET_NATIVE
    historical = root / 'manifests/final-binding-v1.8.3.json'
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == binding['historical_binding_sha256']
    catalog = load(root, 'manifests/final-catalog.json')[binding['delivery_id']]
    assert catalog == {'kind': 'source-candidate', 'native_sha256': TARGET_NATIVE,
                       'component_seal_sha256': binding['component_seal_sha256'],
                       'policy_sha256': binding['policy_sha256'], 'startup_settings': binding['startup_settings']}
    env = dict(line.split('=', 1) for line in (root / 'recipe/.env.example').read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    assert all(env.get(k) == v for k, v in binding['startup_settings'].items())
    assert env['JSPARK3_V16_COOP'] == '1' and env['ABLIT'] == '0' and env['JSPARK3_V16_PROFILE'] == 'production-stock'
    pin = load(root, 'recipe/config/coop-release.json')
    results = load(root, 'release/results-v1.8.4.json')
    assert results['native_sha256'] == TARGET_NATIVE and results['release_version'] == 'v1.8.4'
    if binding['state'] == 'prepared':
        assert not require_final and release['stage'] == 'prepared', 'v1.8.4 component/boot/measurement gates are pending'
        assert pin == {'schema_version': 1, 'status': 'pending-component-qualification', 'native_sha256': TARGET_NATIVE,
                       'build_sha256': None, 'gate_index_sha256': None}
        assert all(binding[key] is None for key in ('component_seal_sha256', 'policy_sha256', 'admission_receipt_sha256', 'results_sha256'))
        assert results['status'] == 'pending-measurement' and all(results[key] is None for key in
               ('candidate_commit', 'component_seal_sha256', 'policy_sha256', 'stock', 'edited'))
        return
    assert binding['state'] in ('component-qualified', 'bound'), 'unknown v1.8.4 binding state'
    from _coop_qualification import verify_record
    coop = root / 'recipe/overlays/v16/coop'
    record = load(root, 'recipe/overlays/v16/coop/BUILD.json')
    component = verify_record(record, coop / 'bundle', coop)
    assert hashlib.sha256((coop / 'QUALIFICATION_SOURCE_MANIFEST.json').read_bytes()).hexdigest() == record['qualification_source_manifest_sha256']
    assert binding['component_seal_sha256'] == component['component_seal_sha256']
    assert binding['policy_sha256'] == component['policy_sha256']
    assert load(root, 'manifests/binaries.json')['recipe/overlays/v16/coop/bundle/cooperative_moe.so']['expected_sha256'] == TARGET_NATIVE
    if binding['state'] == 'component-qualified':
        assert not require_final, 'component-qualified source lacks serving measurements and admission'
        assert release['stage'] == 'component-qualified', 'private serving stage differs'
        assert binding['admission_receipt_sha256'] is None and binding['results_sha256'] is None
        assert results['status'] == 'pending-measurement' and all(results[key] is None for key in
               ('candidate_commit', 'component_seal_sha256', 'policy_sha256', 'stock', 'edited'))
        return
    assert release['stage'] == 'final'
    assert results['status'] == 'measured' and re.fullmatch('[0-9a-f]{40}', results['candidate_commit'])
    assert results['component_seal_sha256'] == binding['component_seal_sha256']
    assert results['policy_sha256'] == binding['policy_sha256']
    assert hashlib.sha256((root / 'release/results-v1.8.4.json').read_bytes()).hexdigest() == binding['results_sha256']
    stock = results['stock']
    assert stock['weight_mode'] == 'production-stock' and stock['ABLIT'] == 0
    assert len(stock['decode_sweeps']) == 2 and len(stock['prefill_turns']) == 8 and stock['quality_evidence']
    # The public admission producer revalidates boot/environment/component and
    # all supplied evidence. A standalone PASS string never qualifies a release.
    import subprocess, sys, tempfile
    admission = root / 'release/v1.8.4-admission'
    assert hashlib.sha256((admission / 'finalize.json').read_bytes()).hexdigest() == binding['admission_receipt_sha256']
    final = load(root, 'release/v1.8.4-admission/finalize.json')
    assert final['producer'] == 'qualify_runtime.py' and final['identity_config']['coop'] == 'on'
    assert all(final['component_qualification'][key] == value for key, value in component.items())
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run([sys.executable, '-B', str(root / 'tools/v16/admission_gate.py'),
                        '--first-prompt', str(admission / 'first-prompt.json'), '--finalize', str(admission / 'finalize.json'),
                        '--out', str(Path(temp) / 'admission.json')], check=True, capture_output=True)
