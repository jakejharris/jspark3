"""Source-only contract checks; deployment still uses the native verifiers."""
import hashlib
import json
from pathlib import Path


def verify_sources(recipe):
    import apply_display_kv as display
    import apply_coop_moe as coop
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    artifacts = json.loads((recipe.parent / 'manifests/binaries.json').read_text())
    for name, expected in display.V14_DISPLAY['sources'].items():
        path = display.OVERLAY / name
        rel = path.relative_to(recipe.parent).as_posix()
        if rel in artifacts:
            assert not path.exists() and artifacts[rel]['expected_sha256'] == expected
        else:
            assert path.is_file() and not path.is_symlink() and sha(path) == expected
    coop.verify_sources()
    bundle = coop.DEFAULT_BUNDLE
    manifest = json.loads((bundle / 'manifest.json').read_text())
    record = json.loads(coop.DEFAULT_BUILD_RECORD.read_text())
    for name, expected in manifest['files'].items():
        path = bundle / name
        assert path.resolve().is_relative_to(bundle.resolve())
        rel = path.relative_to(recipe.parent).as_posix()
        if rel in artifacts:
            assert not path.exists() and artifacts[rel]['expected_sha256'] == expected
        else:
            assert path.is_file() and not path.is_symlink() and sha(path) == expected
    native = manifest['files']['cooperative_moe.so']
    assert record['source_manifest_sha256'] == coop.SOURCE_MANIFEST_SHA256
    assert record['image'] == {'manifest': coop.IMAGE_MANIFEST, 'config': coop.IMAGE_CONFIG}
    assert record['bundle'] == {'manifest_sha256': sha(bundle / 'manifest.json'), 'native_sha256': native,
        'runtime_sha256': manifest['files']['runtime.py'], 'dispatch_policy_sha256': manifest['files']['dispatch_policy.json']}
    assert record['reproducibility'] == {'runs': 2, 'comparison': 'bit-identical', 'binary_sha256': native}
    assert json.loads((bundle / 'dispatch_policy.json').read_text())['native_sha256'] == native
    build = json.loads((display.OVERLAY / 'BUILD.json').read_text())
    for name, expected in build['sources'].items(): assert sha(display.OVERLAY / name) == expected
    for name, expected in build['outputs'].items():
        assert artifacts[(display.OVERLAY / name).relative_to(recipe.parent).as_posix()]['expected_sha256'] == expected
    if (recipe/'scripts/apply_triar.py').exists():
        import apply_triar
        apply_triar.section()
    if (recipe/'scripts/apply_cyclic_thirds.py').exists():
        import apply_cyclic_thirds
        apply_cyclic_thirds.verify_sources()
        bundle = recipe/'overlays/v17/thirds/bundle'
        manifest = json.loads((bundle/'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            path = bundle/name
            assert not path.is_symlink() and path.resolve().is_relative_to(bundle.resolve())
            rel = path.relative_to(recipe.parent).as_posix()
            if rel in artifacts:
                assert not path.exists() and artifacts[rel]['expected_sha256'] == expected
            else:
                assert path.is_file() and sha(path) == expected
