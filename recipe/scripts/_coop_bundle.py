# SPDX-License-Identifier: Apache-2.0
"""Bind copied bundle inputs to source pins; selection changes only the policy."""
import _coop_qualification as q


def source_files(coop):
    from apply_coop_moe import SOURCE_MANIFEST_SHA256, RUNTIME_SHA256
    trusted = q.RECIPE / 'overlays/v16/coop'
    q.need(q.sha(q.regular(coop, 'SOURCE_MANIFEST.json')) == q.sha(trusted / 'SOURCE_MANIFEST.json') == SOURCE_MANIFEST_SHA256,
           'source manifest differs from trusted recipe')
    q.need(q.sha(q.regular(coop, 'build_repro.sh')) == q.sha(trusted / 'build_repro.sh'),
           'builder differs from trusted recipe')
    files = q.read(coop / 'SOURCE_MANIFEST.json')['files']
    q.need(files.get('runtime.py') == RUNTIME_SHA256, 'source adapter differs from install pin')
    actual = {p.relative_to(coop / 'source').as_posix() for p in (coop / 'source').rglob('*') if p.is_file()}
    q.need(actual == set(files), 'source inventory differs')
    for name, expected in files.items():
        q.need(q.sha(q.regular(coop / 'source', name)) == expected, 'source file differs from pin: ' + name)
    return files


def verify_manifest(manifest, coop, *, source_policy=False):
    source = source_files(coop)
    expected = {name: source[name] for name in ('runtime.py', 'PROVENANCE.json', 'LICENSE.MIT', 'LICENSE.upstream-AGPL-3.0')}
    expected['LICENSE.exllamav3'] = source['vendor/LICENSE.exllamav3']
    expected.update({'source/' + name: value for name, value in source.items() if name.startswith('native/')})
    expected.update({'headers/' + name.removeprefix('vendor/exllamav3_ext/'): value
                     for name, value in source.items() if name.startswith('vendor/exllamav3_ext/')})
    expected['headers/quant/glm53_coop_kernel.cuh'] = source['native/cooperative_moe_kernel.cuh']
    expected['headers/quant/exl3_moe_coop.cuh'] = source['native/exl3_moe_coop.cuh']
    expected['cooperative_moe.so'] = q.TARGET_NATIVE
    reference = q.read(q.RECIPE / 'overlays/v16/coop/bundle/manifest.json')
    q.need(set(manifest) == set(reference) and all(manifest[k] == reference[k] for k in reference if k != 'files'),
           'bundle manifest contract differs')
    files = manifest['files']
    q.need(set(files) == set(expected) | {'toolchain.txt', 'dispatch_policy.json'}
           and all(files.get(k) == v for k, v in expected.items())
           and all(q.hash_ok(v) for v in files.values()), 'bundle files differ from pinned source/native')
    if source_policy:
        q.need(files['dispatch_policy.json'] == source['dispatch_policy.json'], 'raw seed policy differs from source')
    return files


def identity(bundle):
    files = q.read(q.regular(bundle, 'manifest.json'))['files']
    return {'manifest_sha256': q.sha(bundle / 'manifest.json'), 'native_sha256': files['cooperative_moe.so'],
            'runtime_sha256': files['runtime.py'], 'dispatch_policy_sha256': files['dispatch_policy.json']}


def verify_bundle(bundle, coop, *, source_policy=False, native_required=True):
    manifest = q.read(q.regular(bundle, 'manifest.json'))
    files = verify_manifest(manifest, coop, source_policy=source_policy)
    for name, expected in files.items():
        if name == 'cooperative_moe.so' and not native_required and not (bundle / name).exists():
            continue  # source exports omit the locally built native library
        q.need(q.sha(q.regular(bundle, name)) == expected, 'bundle artifact changed: ' + name)
    # Build intermediates/logs may remain in raw output, but importable extras
    # and symlinks cannot shadow the pinned adapter/dependencies.
    for path in bundle.rglob('*'):
        q.need(not path.is_symlink(), 'symlink in bundle')
        if path.is_file() and (path.suffix in ('.py', '.pyc', '.pyo', '.so') or '__pycache__' in path.parts):
            q.need(path.relative_to(bundle).as_posix() in files, 'unlisted executable bundle input')
    return identity(bundle)


def verify_selection(raw, selected):
    q.need({k:v for k,v in raw.items() if k != 'files'} == {k:v for k,v in selected.items() if k != 'files'}
           and set(raw['files']) == set(selected['files'])
           and all(selected['files'][name] == digest for name, digest in raw['files'].items()
                   if name != 'dispatch_policy.json'), 'selected bundle changed a non-policy input')
