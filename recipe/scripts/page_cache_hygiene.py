#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Advise DONTNEED for regular files below explicit roots; never delete files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat


def evict(roots):
    report = {'schema': 'jspark3-page-cache-hygiene/1', 'started_at': datetime.now(timezone.utc).isoformat(),
              'roots': [str(root) for root in roots], 'files': 0, 'bytes': 0, 'errors': []}
    seen = set()
    def error(exc):
        report['errors'].append({'path': str(exc.filename), 'errno': exc.errno})
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            report['errors'].append({'path': str(root), 'reason': 'root is not a real directory'})
            continue
        for directory, dirs, files in os.walk(root, onerror=error, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                try:
                    mode = path.lstat().st_mode
                    if not stat.S_ISREG(mode):
                        continue
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        info = os.fstat(fd)
                        if not stat.S_ISREG(info.st_mode):
                            raise ValueError('file changed type during hygiene')
                        key = (info.st_dev, info.st_ino)
                        if key in seen:
                            continue
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        seen.add(key)
                        report['files'] += 1
                        report['bytes'] += info.st_size
                    finally:
                        os.close(fd)
                except OSError as exc:
                    error(exc)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['verdict'] = 'PASS' if report['files'] and not report['errors'] else 'FAIL'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    args = parser.parse_args()
    report = evict(args.roots)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
