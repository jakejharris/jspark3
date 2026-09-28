#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Integrate a reviewed component seal into a new, private serving source tree.

Run from a validated clean source export. The original source and seal remain
unchanged. Only component evidence is bound; serving results/admission stay null.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recipe/scripts'))
import argparse
import copy
import json
import re
import shutil
import subprocess
import tempfile
import _diagnostics as diagnostics
import _coop_qualification as q
from _coop_bundle import verify_bundle
from validate_release import inventory, sums, verify

COOP = 'recipe/overlays/v16/coop'
INSTALLER = 'recipe/scripts/apply_coop_moe.py'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def refresh(root, seal):
    """Refresh only the metadata touched by this reviewed integration operation."""
    additions = [COOP + '/QUALIFICATION.json', COOP + '/QUALIFICATION_SOURCE_MANIFEST.json']
    licenses = q.read(root / 'manifests/license-review.json')
    for name in additions:
        q.need(name not in licenses['files'], 'component evidence already inventoried')
        licenses['files'][name] = 'AGPL-3.0-only'
    write(root / 'manifests/license-review.json', licenses)
    reuse = root / 'REUSE.toml'
    reuse.write_text(reuse.read_text() + '\n[[annotations]]\npath = ' + json.dumps(additions)
                     + '\nprecedence = "aggregate"\nSPDX-License-Identifier = "AGPL-3.0-only"\n')
    # The only executable change is the checked source-manifest constant.
    # Preserve its output review: no output site or behavior has changed.
    audit = q.read(root / 'manifests/shared-output-audit.json')
    audit['files'][INSTALLER]['sha256'] = q.sha(root / INSTALLER)
    write(root / 'manifests/shared-output-audit.json', audit)
    files, errors = inventory(root)
    q.need(not errors, 'integrated source inventory refused')
    sbom = q.read(root / 'manifests/sbom.cdx.json')
    for row in sbom['components']:
        if row['type'] == 'file':
            row['hashes'] = [{'alg': 'SHA-256', 'content': q.sha(root / row['name'])}]
    for name in additions:
        sbom['components'].append({'type': 'file', 'name': name,
            'licenses': [{'license': {'id': licenses['files'][name]}}],
            'hashes': [{'alg': 'SHA-256', 'content': q.sha(root / name)}]})
    write(root / 'manifests/sbom.cdx.json', sbom)
    recipe = root / 'recipe'
    (recipe / 'SHA256SUMS').write_text(sums(recipe, [p for p in files if p.is_relative_to(recipe)]))
    derivation = q.read(root / 'manifests/derivation.json')
    derivation['delivery_candidate'].update(state='component-qualified',
        binding_sha256=q.sha(root / 'manifests/final-binding.json'),
        source_recipe_sha256=q.sha(recipe / 'SHA256SUMS'))
    derivation['delivery_candidate']['qualification_build_sha256'] = q.digest_value(q.read(seal / 'BUILD.json'))
    original = {COOP + '/QUALIFICATION.json': seal / 'QUALIFICATION.json',
                COOP + '/QUALIFICATION_SOURCE_MANIFEST.json': seal / 'SOURCE_MANIFEST.json'}
    for path in files:
        name = path.relative_to(root).as_posix()
        if name in ('SHA256SUMS', 'recipe/SHA256SUMS', 'manifests/derivation.json'):
            continue
        row = derivation['files'].get(name)
        if row is None:
            row = {'origin': 'reviewed-component-seal', 'input_sha256': q.sha(original[name])}
        digest = q.sha(path)
        if row.get('public_sha256') != digest:
            row.update(public_sha256=digest, changed=digest != row['input_sha256'],
                       revision='v1.8.4-component-qualified')
        derivation['files'][name] = row
    write(root / 'manifests/derivation.json', derivation)
    (root / 'SHA256SUMS').write_text(sums(root, files))


def integrate(root, seal, output):
    q.need(not output.exists() and not output.is_symlink(), 'output must be new')
    q.need(not output.resolve().is_relative_to(root.resolve())
           and not output.resolve().is_relative_to(seal.resolve()), 'output must be outside source and seal')
    q.need(not verify(root)['failed'], 'source export failed validation')
    binding = q.read(root / 'manifests/final-binding.json')
    release = q.read(root / 'manifests/release.json')
    q.need(binding['release_version'] == 'v1.8.4' and binding['state'] == 'prepared'
           and release['stage'] == 'prepared', 'requires the prepared v1.8.4 source')
    record = q.read(q.regular(seal, 'BUILD.json'))
    q.verify_record(record, seal / 'bundle', seal, release=False)
    verify_bundle(seal / 'bundle', seal)  # Includes the real native bytes.
    original_manifest = q.sha(root / COOP / 'SOURCE_MANIFEST.json')
    q.need(record['source_manifest_sha256'] == record['qualification_source_manifest_sha256'] == original_manifest,
           'seal belongs to a different qualification source')
    index = q.read(seal / 'QUALIFICATION.json')
    source = q.read(root / COOP / 'SOURCE_MANIFEST.json')
    q.need(index['raw_bundle_manifest']['files']['dispatch_policy.json'] == source['files']['dispatch_policy.json'],
           'raw qualification policy differs from source')
    # Stage beside the destination so failure never leaves a partially integrated
    # candidate at the requested output path. Copy only validated source files.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.coop-integration-', dir=output.parent) as temporary:
        candidate = Path(temporary) / 'source'
        files, errors = inventory(root)
        q.need(not errors, 'source inventory changed')
        for path in files:
            dest = candidate / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        coop = candidate / COOP
        manifest = q.read(seal / 'bundle/manifest.json')
        for name in ['manifest.json', *manifest['files']]:
            if name != 'cooperative_moe.so':
                shutil.copyfile(q.regular(seal / 'bundle', name), coop / 'bundle' / name)
        shutil.copyfile(q.regular(seal, 'QUALIFICATION.json'), coop / 'QUALIFICATION.json')
        shutil.copyfile(q.regular(seal, 'SOURCE_MANIFEST.json'), coop / 'QUALIFICATION_SOURCE_MANIFEST.json')
        shutil.copyfile(coop / 'bundle/dispatch_policy.json', coop / 'source/dispatch_policy.json')
        source['files']['dispatch_policy.json'] = q.sha(coop / 'source/dispatch_policy.json')
        write(coop / 'SOURCE_MANIFEST.json', source)
        source_hash = q.sha(coop / 'SOURCE_MANIFEST.json')
        installer = candidate / INSTALLER
        text, count = re.subn(r'^SOURCE_MANIFEST_SHA256 = "' + original_manifest + '"$',
                             'SOURCE_MANIFEST_SHA256 = "' + source_hash + '"',
                             installer.read_text(), flags=re.MULTILINE)
        q.need(count == 1, 'installer source pin differs')
        installer.write_text(text)
        contract = q.read(coop / 'INSTALL_CONTRACT.json')
        contract['transforms']['apply_coop_moe.py']['sources']['SOURCE_MANIFEST.json'] = source_hash
        write(coop / 'INSTALL_CONTRACT.json', contract)
        rebound = copy.deepcopy(record)
        rebound['source_manifest_sha256'] = source_hash
        q.need(rebound['compiled_inputs'] == q.compiled_inputs(coop), 'compiled inputs changed during integration')
        write(coop / 'BUILD.json', rebound)
        seal_hash = q.digest_value(rebound)
        policy_hash = manifest['files']['dispatch_policy.json']
        write(candidate / 'recipe/config/coop-release.json', {'schema_version': 1, 'status': 'qualified',
            'native_sha256': q.TARGET_NATIVE, 'build_sha256': seal_hash,
            'gate_index_sha256': record['gate_index_sha256']})
        binaries = q.read(candidate / 'manifests/binaries.json')
        binaries[COOP + '/bundle/cooperative_moe.so']['expected_sha256'] = q.TARGET_NATIVE
        write(candidate / 'manifests/binaries.json', binaries)
        binding.update(state='component-qualified', component_seal_sha256=seal_hash, policy_sha256=policy_hash)
        write(candidate / 'manifests/final-binding.json', binding)
        catalog = q.read(candidate / 'manifests/final-catalog.json')
        catalog[binding['delivery_id']].update(component_seal_sha256=seal_hash, policy_sha256=policy_hash)
        write(candidate / 'manifests/final-catalog.json', catalog)
        release['stage'] = 'component-qualified'
        release['qualification'] = ('Component-qualified for private serving and measurement; final-source rebuilds, '
                                    'serving measurements and same-boot admission remain required.')
        write(candidate / 'manifests/release.json', release)
        refresh(candidate, seal)
        # Fresh imports must bind to the integrated source, not this process's
        # original source pins. This runs every source/privacy/inventory check.
        result = subprocess.run([sys.executable, '-B', str(candidate / 'tools/validate_release.py'), str(candidate)],
                                capture_output=True, text=True)
        diagnostics.retain(result.stdout + result.stderr)
        q.need(result.returncode == 0, 'integrated source failed validation')
        q.need(not output.exists() and not output.is_symlink(), 'output appeared during integration')
        candidate.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private source candidate directory')
    args = parser.parse_args()
    integrate(ROOT, args.sealed_output, args.output)
    print('PASS component-qualified source prepared; serving measurement, admission and publication remain closed')


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    main()
