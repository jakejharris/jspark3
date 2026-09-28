#!/usr/bin/env python3
"""Create a runtime candidate from validated source and pinned or receipted local builds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-root', type=Path, required=True,
                        help='local build tree with every path listed in manifests/binaries.json')
    parser.add_argument('--output', type=Path, required=True, help='new private runtime directory')
    parser.add_argument('--image-receipt', type=Path,
                        help='verified local build receipt; omit only for the historical reference image')
    parser.add_argument('--native-receipt', type=Path,
                        help='build_native.py receipt; requires the same --image-receipt')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root): parser.error('output must be new and outside the source export')
    from validate_release import verify
    if verify(root)['failed']: raise SystemExit('REFUSE: source export failed validation')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sys.path.insert(0, str(root / 'recipe/scripts'))
    from _image_identity import read_operator_record
    image_record = None
    if args.image_receipt:
        image_record = read_operator_record(args.image_receipt)
        if image_record['source_recipe_sha256'] != sha(root / 'recipe/SHA256SUMS'):
            raise SystemExit('REFUSE: image receipt belongs to a different source recipe')
    native_record = None
    if args.native_receipt:
        if not image_record:
            raise SystemExit('REFUSE: native receipt requires its operator image receipt')
        from build_native import read_native_record
        native_record = read_native_record(args.native_receipt, image_record)
    binaries = json.loads((root / 'manifests/binaries.json').read_text())
    for name, row in binaries.items():
        path = args.binary_root / name
        expected = native_record['binary_sha256'][name] if native_record else row['expected_sha256']
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise SystemExit('REFUSE: missing or mismatched local build: ' + name)
    shutil.copytree(root / 'recipe', output / 'recipe')
    for name in binaries:
        shutil.copyfile(args.binary_root / name, output / name)
        (output / name).chmod(0o755)
    recipe = output / 'recipe'
    if image_record:
        (recipe / 'config/operator-image.json').write_text(json.dumps(image_record, sort_keys=True) + '\n')
    if native_record:
        (recipe / 'config/operator-native.json').write_text(json.dumps(native_record, sort_keys=True) + '\n')
    sys.path.insert(0, str(output / 'recipe/scripts'))
    import apply_display_kv as display
    import apply_coop_moe as coop
    display.verify_sources()
    coop_native = 'recipe/overlays/v16/coop/bundle/cooperative_moe.so'
    operator_coop = sha(output / coop_native) != binaries[coop_native]['expected_sha256']
    if operator_coop:
        from _native_identity import seal_operator_bundle
        seal_operator_bundle(recipe, native_record)
    coop.verify_bundle(coop.DEFAULT_BUNDLE, coop.DEFAULT_BUILD_RECORD)
    (recipe / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(recipe).as_posix()}\n'
        for p in sorted(recipe.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
    (output / 'runtime-build-receipt.json').write_text(json.dumps({
        'source_tree_sha256': sha(root / 'SHA256SUMS'), 'recipe_manifest_sha256': sha(recipe / 'SHA256SUMS'),
        'operator_image_receipt_sha256': sha(recipe / 'config/operator-image.json') if image_record else None,
        'operator_native_receipt_sha256': sha(recipe / 'config/operator-native.json') if native_record else None,
        'coop_artifact_seal': 'operator-native-receipt-v1' if operator_coop else 'historical',
        'binary_sha256': {n: sha(output / n) for n in binaries}, 'hardware_qualified': False}, indent=2) + '\n')
    print('PASS local runtime recipe prepared; hardware admission remains closed')


if __name__ == '__main__': main()
