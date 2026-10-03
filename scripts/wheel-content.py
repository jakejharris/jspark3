#!/usr/bin/env python3
"""SHA-256 of sorted '<file sha256>  <path>\n' records, excluding only *.dist-info/RECORD and *.dist-info/WHEEL.

This includes package metadata, entry points and licences while ignoring archive bookkeeping.
"""
import hashlib
import sys
import zipfile


def contents(path):
    with zipfile.ZipFile(path) as wheel:
        return {name: hashlib.sha256(wheel.read(name)).hexdigest()
                for name in sorted(wheel.namelist())
                if not name.endswith('/') and not (len(name.split('/')) == 2 and name.split('/')[0].endswith('.dist-info') and name.split('/')[1] in ('RECORD', 'WHEEL'))}


def digest(files):
    return hashlib.sha256(''.join(f'{sha}  {name}\n' for name, sha in sorted(files.items())).encode()).hexdigest()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect', help='fail unless every wheel matches this content SHA-256')
    parser.add_argument('wheels', nargs='+')
    args = parser.parse_args()
    failed = False
    for path in args.wheels:
        files = contents(path)
        actual = digest(files)
        print(f'{actual}  {path} ({len(files)} files, excluding only *.dist-info/RECORD and *.dist-info/WHEEL)')
        if args.expect is not None and actual != args.expect:
            print(f'wheel content mismatch: expected {args.expect}, got {actual}', file=sys.stderr)
            failed = True
    sys.exit(1 if failed else 0)
