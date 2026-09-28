#!/usr/bin/env python3
"""Create a runtime candidate from validated source and pinned or receipted local builds."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe/scripts"))
import _diagnostics as diagnostics
import argparse
import hashlib
import json
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-root', type=Path, required=True,
                        help='local build tree with receipted artifacts, or all historical pinned artifacts')
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
    if verify(root)['failed']: raise ValueError('source export failed validation')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sys.path.insert(0, str(root / 'recipe/scripts'))
    from _image_identity import read_operator_record
    image_record = None
    if args.image_receipt:
        image_record = read_operator_record(args.image_receipt)
        if image_record['source_recipe_sha256'] != sha(root / 'recipe/SHA256SUMS'):
            raise ValueError('image receipt belongs to a different source recipe')
    native_record = None
    if args.native_receipt:
        if not image_record:
            raise ValueError('native receipt requires its operator image receipt')
        from build_native import read_native_record
        native_record = read_native_record(args.native_receipt, image_record)
    binaries = json.loads((root / 'manifests/binaries.json').read_text())
    if native_record:
        binaries = {name: binaries[name] for name in native_record['binary_sha256']}
    for name, row in binaries.items():
        path = args.binary_root / name
        expected = native_record['binary_sha256'][name] if native_record else row['expected_sha256']
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError('missing or mismatched local build: ' + name)
    shutil.copytree(root / 'recipe', output / 'recipe')
    for name in binaries:
        shutil.copyfile(args.binary_root / name, output / name)
        (output / name).chmod(0o755)
    sys.path.insert(0, str(output / 'recipe/scripts'))
    import apply_display_kv as display
    import apply_coop_moe as coop
    display.verify_sources()
    coop_native = 'recipe/overlays/v16/coop/bundle/cooperative_moe.so'
    needs_coop_seal = (coop_native not in binaries or
                       sha(output / coop_native) != binaries[coop_native]['expected_sha256'])
    if needs_coop_seal:
        # The receipt proves a source build, not the historical 3x3 GPU profiles.
        # Keep BUILD.json, manifest and row policy unchanged: verify_bundle must
        # continue to refuse coop=1. Operator sealing needs future runtime support.
        coop.verify_sources()
    else:
        coop.verify_bundle(coop.DEFAULT_BUNDLE, coop.DEFAULT_BUILD_RECORD)
    recipe = output / 'recipe'
    if image_record:
        (recipe / 'config/operator-image.json').write_text(json.dumps(image_record, sort_keys=True) + '\n')
    if native_record:
        (recipe / 'config/operator-native.json').write_text(json.dumps(native_record, sort_keys=True) + '\n')
        # A source-build receipt is not a hardware seal. The prepared example
        # must explicitly select the supported off path; omission would allow
        # callers to inherit a different default.
        env = recipe / '.env.example'
        env.write_text('\n'.join('JSPARK3_V16_COOP=0' if line.startswith('JSPARK3_V16_COOP=') else line
                                 for line in env.read_text().splitlines()) + '\n')
    (recipe / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(recipe).as_posix()}\n'
        for p in sorted(recipe.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
    (output / 'runtime-build-receipt.json').write_text(json.dumps({
        'source_tree_sha256': sha(root / 'SHA256SUMS'), 'recipe_manifest_sha256': sha(recipe / 'SHA256SUMS'),
        'operator_image_receipt_sha256': sha(recipe / 'config/operator-image.json') if image_record else None,
        'operator_native_receipt_sha256': sha(recipe / 'config/operator-native.json') if native_record else None,
        'coop_hardware_seal_required': needs_coop_seal,
        'binary_sha256': {n: sha(output / n) for n in binaries}, 'hardware_qualified': False}, indent=2) + '\n')
    print('PASS local runtime recipe prepared; hardware admission remains closed')
    if native_record:
        print('Operator native build: prepared .env.example selects JSPARK3_V16_COOP=0; keep coop disabled')
    if needs_coop_seal:
        print('Coop-MoE is absent or differs from the historical seal; operator coop=1 support requires a future implementation change')


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    main()
