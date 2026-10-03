#!/usr/bin/env python3
"""Compare a rank directory with a per-rank manifest, or write one.

    python3 scripts/verify-split.py check DIR MANIFEST     exit 0: every file matches; 1: differences (listed)
    python3 scripts/verify-split.py write DIR OUT          write DIR's manifest to OUT
    python3 scripts/verify-split.py digest MANIFEST        print the manifest's own sha256
    python3 scripts/verify-split.py seal DIR --rank R --conversion-id ID
                                                           finish a fresh ablit third: write its .complete marker
                                                           ("<ID> rank=R world=3") and then DIR/MANIFEST.sha256,
                                                           the list of every other file, as the measured thirds have

A manifest is `sha256sum` format: "<sha256>  <path>" for every regular file under DIR (symlinks followed), paths
relative to DIR, sorted bytewise. `cd DIR && sha256sum -c MANIFEST` checks the same thing without this script.
On a mismatch, every missing, extra or different file is named with its size, so a difference in one shard is
visible at once. Output names files relative to DIR, never DIR itself. Standard library only.
"""
import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 8 << 20


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def listing(root):
    files = []
    for folder, dirs, names in os.walk(root, followlinks=True):
        dirs.sort()
        for name in names:
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                files.append(os.path.relpath(full, root))
    return sorted(files, key=lambda p: p.encode())


def hashes(root, workers=4):
    files = listing(root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(lambda rel: file_sha256(os.path.join(root, rel)), files))
    return dict(zip(files, digests))


def read_manifest(path):
    entries = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition('  ')
        if not sep or len(digest) != 64 or not rel:
            sys.exit(f'verify-split: {path} line {number} is not "<sha256>  <path>"')
        entries[rel[2:] if rel.startswith('./') else rel] = digest
    return entries


def render(entries):
    return ''.join(f'{entries[rel]}  {rel}\n' for rel in sorted(entries, key=lambda p: p.encode()))


def seal(root, rank, conversion_id):
    """The ablit conversion's own record of a third: the marker first, then a list of every other file."""
    if not re.fullmatch(r'[0-9a-f]{64}', conversion_id):
        sys.exit('verify-split: the conversion id must be 64 lowercase hex digits (ABLIT_CONVERSION_ID in pins.env)')
    if not root.is_dir():
        sys.exit('verify-split: the rank directory does not exist')
    inner = root / 'MANIFEST.sha256'
    if inner.exists():
        sys.exit('verify-split: this third already has MANIFEST.sha256; split it again instead of sealing it twice')
    if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        sys.exit('verify-split: a third holds plain files only; this one has a folder or a link')
    (root / '.complete').write_text(f'{conversion_id} rank={rank} world=3\n')
    body = render(hashes(root))
    inner.write_text(body)
    print(f'verify-split: sealed rank {rank}; MANIFEST.sha256 lists {body.count(chr(10))} files, '
          f'sha256 {hashlib.sha256(body.encode()).hexdigest()}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('dir')
    c.add_argument('manifest')
    w = sub.add_parser('write')
    w.add_argument('dir')
    w.add_argument('out')
    d = sub.add_parser('digest')
    d.add_argument('manifest')
    s = sub.add_parser('seal')
    s.add_argument('dir')
    s.add_argument('--rank', required=True, choices=('0', '1', '2'))
    s.add_argument('--conversion-id', required=True)
    a = ap.parse_args()

    if a.command == 'digest':
        print(hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest())
        return 0
    if a.command == 'seal':
        return seal(Path(a.dir), a.rank, a.conversion_id)
    if not os.path.isdir(a.dir):
        sys.exit('verify-split: the rank directory does not exist')
    got = hashes(a.dir)
    if a.command == 'write':
        Path(a.out).write_text(render(got))
        print(f'verify-split: wrote {len(got)} entries; manifest sha256 '
              f'{hashlib.sha256(render(got).encode()).hexdigest()}')
        return 0
    want = read_manifest(a.manifest)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    different = sorted(rel for rel in set(want) & set(got) if want[rel] != got[rel])
    if not (missing or extra or different):
        print(f'verify-split: all {len(want)} files match')
        return 0
    size = lambda rel: os.path.getsize(os.path.join(a.dir, rel))  # noqa: E731
    for rel in missing:
        print(f'MISSING    {rel}')
    for rel in extra:
        print(f'EXTRA      {rel} ({size(rel)} bytes)')
    for rel in different:
        print(f'DIFFERENT  {rel} ({size(rel)} bytes): sha256 {got[rel]}, expected {want[rel]}')
    print(f'verify-split: {len(missing)} missing, {len(extra)} extra, {len(different)} different '
          f'of {len(want)} expected files')
    return 1


if __name__ == '__main__':
    sys.exit(main())
