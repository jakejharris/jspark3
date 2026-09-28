# SPDX-License-Identifier: Apache-2.0
"""Receipt-bound native artifacts; build provenance is not GPU qualification."""
import hashlib
import json
from pathlib import Path

from _image_identity import ImageRefusal, canonical, read_operator_record, sha

COOP_NATIVE = 'recipe/overlays/v16/coop/bundle/cooperative_moe.so'


def operator_bundle_documents(recipe, native):
    policy = json.loads((recipe / 'config/native-build-policy.json').read_text())
    reference = policy['coop_reference']
    row_policy = {'schema': 1, 'rows': reference['rows'],
                  'native_sha256': native['binary_sha256'][COOP_NATIVE],
                  'basis': 'reference rows for verified source build; fresh hardware qualification required',
                  'reference_policy_sha256': reference['policy_sha256']}
    manifest = reference['manifest']
    manifest['files']['cooperative_moe.so'] = row_policy['native_sha256']
    manifest['files']['dispatch_policy.json'] = hashlib.sha256(canonical(row_policy)).hexdigest()
    record = {'schema_version': 2, 'verification': 'operator-native-receipt-v1',
              'native_receipt_sha256': native['payload_sha256'],
              'reference_build_sha256': reference['build_record_sha256'],
              'hardware_qualified': False,
              'manifest_sha256': hashlib.sha256(canonical(manifest)).hexdigest()}
    return row_policy, manifest, record


def seal_operator_bundle(recipe, native):
    policy, manifest, record = operator_bundle_documents(recipe, native)
    coop = recipe / 'overlays/v16/coop'
    for path, value in ((coop / 'bundle/dispatch_policy.json', policy),
                        (coop / 'bundle/manifest.json', manifest), (coop / 'BUILD.json', record)):
        path.write_bytes(canonical(value))


def verify_operator_bundle(bundle, build_record):
    recipe = bundle.parents[3]
    policy = json.loads((recipe / 'config/native-build-policy.json').read_text())
    image = read_operator_record(recipe / 'config/operator-image.json')
    receipt_path = recipe / 'config/operator-native.json'
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ImageRefusal('operator native receipt missing or symlinked')
    native = json.loads(receipt_path.read_text())
    expected_keys = {'schema_version', 'verification', 'source_recipe_sha256', 'image_receipt_sha256',
                     'build_inputs', 'reproducibility', 'binary_sha256', 'hardware_qualified', 'payload_sha256'}
    if not isinstance(native, dict) or set(native) != expected_keys:
        raise ImageRefusal('operator native receipt schema drift')
    payload = {k: v for k, v in native.items() if k != 'payload_sha256'}
    if native['payload_sha256'] != hashlib.sha256(canonical(payload)).hexdigest():
        raise ImageRefusal('operator native receipt payload hash mismatch')
    if (native['schema_version'] != 1 or native['verification'] != 'fixed-native-build-v1'
            or native['build_inputs'] != policy['build_inputs']
            or native['reproducibility'] != {'runs': 2, 'comparison': 'bit-identical'}
            or native['hardware_qualified'] is not False
            or native['source_recipe_sha256'] != image['source_recipe_sha256']
            or native['image_receipt_sha256'] != image['payload_sha256']
            or set(native['binary_sha256']) != set(policy['binary_paths'])):
        raise ImageRefusal('operator native source/build/image binding drift')
    # The source export's policy pins the external builder too; all installed
    # source inputs and all three outputs must still equal their receipt.
    files = {k: v for k, v in native['build_inputs'].items() if k.startswith('recipe/')}
    files.update(native['binary_sha256'])
    for name, expected in files.items():
        path = recipe / Path(name).relative_to('recipe')
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ImageRefusal('operator native file drift: ' + name)
    row_policy, manifest, record = operator_bundle_documents(recipe, native)
    for path, expected in ((bundle / 'dispatch_policy.json', row_policy),
                           (bundle / 'manifest.json', manifest), (build_record, record)):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != canonical(expected):
            raise ImageRefusal('operator native seal drift: ' + path.name)
    for name, expected in manifest['files'].items():
        path = bundle / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ImageRefusal('operator native bundle drift: ' + name)
    if not (bundle / 'cooperative_moe.so').read_bytes().startswith(b'\x7fELF'):
        raise ImageRefusal('operator native artifact is not ELF')
    return {'native_sha256': row_policy['native_sha256'],
            'policy_sha256': manifest['files']['dispatch_policy.json']}
