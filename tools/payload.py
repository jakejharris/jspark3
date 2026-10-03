#!/usr/bin/env python3
"""Check this recipe's files against SHA256SUMS, write SHA256SUMS, or write an external seal.

    python3 tools/payload.py verify                 every listed file matches, and nothing unlisted is present
    python3 tools/payload.py manifest               write SHA256SUMS from the files git tracks (release tooling)
    python3 tools/payload.py seal --tag TAG --archive FILE --out FILE [--facts FILE] [--receipt FILE ...]

SHA256SUMS lists every shipped file except itself, in `sha256sum` format, sorted bytewise; `sha256sum -c SHA256SUMS`
checks the same files without this script. verify ignores what you create while installing: cluster.env, wheels/,
logs/, Python caches and the .git folder. The seal is written outside the tree: it binds the tag, the commit, the
SHA256SUMS digest, the release archive's digest and, when given, the release-facts file and receipts.
Exit 0: everything matches. Exit 1: something differs (listed). Standard library only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SUMS = 'SHA256SUMS'
# Created by installing, never shipped.
IGNORED_TOP = {'.git', 'cluster.env', 'wheels', 'logs'}
IGNORED_PARTS = {'__pycache__'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def present():
    found = []
    for folder, dirs, names in os.walk(HERE):
        rel_folder = Path(folder).relative_to(HERE)
        dirs[:] = sorted(d for d in dirs if not (rel_folder == Path('.') and d in IGNORED_TOP) and d not in IGNORED_PARTS)
        for name in names:
            rel = (rel_folder / name).as_posix()
            if rel_folder == Path('.') and name in IGNORED_TOP or name.endswith('.pyc'):
                continue
            found.append(rel)
    return found


def read_sums():
    entries = {}
    for line in (HERE / SUMS).read_text().splitlines():
        digest, sep, rel = line.partition('  ')
        if sep and len(digest) == 64:
            entries[rel] = digest
    return entries


def render(entries):
    return ''.join(f'{entries[rel]}  {rel}\n' for rel in sorted(entries, key=lambda p: p.encode()))


def verify():
    if not (HERE / SUMS).is_file():
        print(f'payload: no {SUMS} in this tree')
        return 1
    want = read_sums()
    bad = 0
    for rel, digest in want.items():
        path = HERE / rel
        if not path.is_file():
            print(f'MISSING    {rel}')
            bad += 1
        elif sha256(path) != digest:
            print(f'DIFFERENT  {rel}')
            bad += 1
    for rel in present():
        if rel != SUMS and rel not in want:
            print(f'UNLISTED   {rel}')
            bad += 1
    if bad:
        print(f'payload: {bad} of {len(want)} files differ from {SUMS} (listed above)')
        return 1
    digest = sha256(HERE / SUMS)
    print(f'payload: all {len(want)} files match {SUMS} (sha256 {digest})')
    return 0


def tracked():
    out = subprocess.run(['git', '-C', str(HERE), 'ls-files', '-z'], capture_output=True, check=True).stdout
    return [p for p in out.decode().split('\0') if p and p != SUMS]


def manifest():
    files = tracked()
    untracked = sorted(set(present()) - set(files) - {SUMS})
    if untracked:
        print('payload: commit or remove these first (they are not tracked):\n  ' + '\n  '.join(untracked))
        return 1
    text = render({rel: sha256(HERE / rel) for rel in files})
    (HERE / SUMS).write_text(text)
    print(f'payload: wrote {SUMS} over {len(files)} files (sha256 {hashlib.sha256(text.encode()).hexdigest()})')
    return 0


def seal(a):
    out = Path(a.out).resolve()
    if HERE in out.parents:
        sys.exit('payload: the seal is written outside the tree it seals')
    if verify() != 0:
        sys.exit('payload: refusing to seal a tree that does not match its SHA256SUMS')
    git = lambda *args: subprocess.run(['git', '-C', str(HERE), *args], capture_output=True, text=True,  # noqa: E731
                                       check=True).stdout.strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        sys.exit('payload: refusing to seal: tracked files have uncommitted changes')
    record = {
        'tag': a.tag,
        'commit': git('rev-parse', 'HEAD'),
        'tree': git('rev-parse', 'HEAD^{tree}'),
        'sha256sums_sha256': sha256(HERE / SUMS),
        'archive': {'name': Path(a.archive).name, 'sha256': sha256(a.archive)},
    }
    if a.facts:
        record['release_facts'] = {'name': Path(a.facts).name, 'sha256': sha256(a.facts)}
    record['receipts'] = [{'name': Path(r).name, 'sha256': sha256(r)} for r in a.receipt]
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(f'payload: wrote the seal for {a.tag} at commit {record["commit"][:12]} (sha256 {sha256(out)})')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    sub.add_parser('manifest')
    s = sub.add_parser('seal')
    s.add_argument('--tag', required=True)
    s.add_argument('--archive', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--facts')
    s.add_argument('--receipt', action='append', default=[])
    a = ap.parse_args()
    return {'verify': verify, 'manifest': manifest}.get(a.command, lambda: seal(a))()


if __name__ == '__main__':
    sys.exit(main())
