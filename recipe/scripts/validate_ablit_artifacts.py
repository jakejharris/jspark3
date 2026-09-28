#!/usr/bin/env python3
"""Validate the immutable, explicitly scoped donor artifact (stdlib only)."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import re

# The repository is operator-supplied in the hash-pinned MANIFEST.json.
REVISION = '80b6d18d77e3020f2384597081d405f19893f101'
INDEX_SHA256 = '015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def shape_for_layer(layer: int) -> list[int]:
    # MTP is explicitly MLA even though 45 % 4 != 3 (mtp.py is_mtp_layer=True).
    return [4096, 16384 if layer % 4 == 3 or layer == 45 else 8192]


def validate(root: Path, expected: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('invalid donor manifest digest')
    if root.is_symlink() or not root.is_dir():
        raise ValueError('unsafe donor root')
    path = root / 'MANIFEST.json'
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError('donor manifest identity mismatch or incomplete artifact')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema_version') != 1 or not isinstance(manifest.get('repository'), str) or
            not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest['repository']) or
            manifest.get('revision') != REVISION or manifest.get('index_sha256') != INDEX_SHA256 or
            manifest.get('layer_range') != [15, 45] or manifest.get('method') != 'transplant' or
            manifest.get('dtype') != 'BF16'):
        raise ValueError('donor manifest contract mismatch')
    rows = manifest.get('tensors', [])
    if [row.get('layer') for row in rows] != list(range(15, 46)):
        raise ValueError('missing, duplicated or unordered donor layer')
    expected_files = {'MANIFEST.json'}
    for row in rows:
        layer = row['layer']
        shape = shape_for_layer(layer)
        name = f'L{layer:02d}.bin'
        expected_files.update({name, f'L{layer:02d}.receipt.json', row['shard'] + '.header.json'})
        if (row.get('tensor') != f'model.language_model.layers.{layer}.self_attn.o_proj.weight' or
                row.get('dtype') != 'BF16' or row.get('shape') != shape or row.get('file') != name or
                row.get('bytes') != shape[0] * shape[1] * 2 or
                not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', row.get('shard', ''))):
            raise ValueError(f'L{layer} identity/dtype/shape mismatch')
        tensor_path = root / name
        if (tensor_path.is_symlink() or not tensor_path.is_file() or
                tensor_path.stat().st_size != row['bytes'] or sha(tensor_path) != row.get('sha256')):
            raise ValueError(f'L{layer} payload corrupt, short, or absent')
        header_path = root / (row['shard'] + '.header.json')
        if (header_path.is_symlink() or header_path.stat().st_size != row['header_bytes'] or
                sha(header_path) != row['header_sha256']):
            raise ValueError(f'L{layer} safetensors header mismatch')
        header = json.loads(header_path.read_text())[row['tensor']]
        lo, hi = header['data_offsets']
        if (header['dtype'] != row['dtype'] or header['shape'] != shape or
                row['range'] != [8 + row['header_bytes'] + lo, 8 + row['header_bytes'] + hi - 1] or
                hi - lo != row['bytes'] or row['range'][1] >= row['shard_bytes']):
            raise ValueError(f'L{layer} source range mismatch')
    if {p.name for p in root.iterdir()} != expected_files or any(p.is_symlink() for p in root.iterdir()):
        raise ValueError('donor artifact contains partial or unrecognized files')
    if sum(row['bytes'] for row in rows) != manifest.get('total_tensor_bytes'):
        raise ValueError('donor byte total mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    manifest = validate(args.root, args.manifest_sha256)
    print(json.dumps({'status': 'PASS', 'revision': REVISION, 'layers': len(manifest['tensors']),
                      'total_tensor_bytes': manifest['total_tensor_bytes'],
                      'manifest_sha256': args.manifest_sha256}, sort_keys=True))


if __name__ == '__main__':
    import _diagnostics as diagnostics
    diagnostics.install_exception_hook()
    main()
