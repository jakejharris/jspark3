#!/usr/bin/env python3
"""Build one sorted SHA256SUMS without following symlinks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", default="SHA256SUMS")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = root / args.output
    rows: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            print(f"REFUSE symlink: {relative}", file=sys.stderr)
            return 9
        if path.is_dir():
            if path.name in {".git", "__pycache__", ".cache"}:
                print(f"REFUSE generated/private directory: {relative}", file=sys.stderr)
                return 9
            continue
        if not path.is_file():
            print(f"REFUSE non-regular entry: {relative}", file=sys.stderr)
            return 9
        if path == output:
            continue
        rows.append((relative, path))
    payload = "".join(f"{digest(path)}  {relative}\n" for relative, path in sorted(rows))
    output.write_text(payload, encoding="utf-8")
    print(f"PASS {output} files={len(rows)} sha256={digest(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
