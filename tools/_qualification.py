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
