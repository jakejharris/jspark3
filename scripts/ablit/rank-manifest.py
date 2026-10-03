#!/usr/bin/env python3
"""Write a sorted per-file SHA-256 manifest, excluding the manifest itself."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
a = p.parse_args()
manifest = a.directory / "MANIFEST.sha256"
if manifest.exists():
    raise SystemExit("Refusing to replace an existing manifest")
lines = []
total = 0
for path in sorted(a.directory.iterdir()):
    if not path.is_file() or path.is_symlink():
        raise SystemExit("Unexpected non-file or symlink in rank directory")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    lines.append(f"{digest}  {path.name}\n")
    total += path.stat().st_size
    print(f"{digest}  {path.name}", flush=True)
if not lines:
    raise SystemExit("Refusing an empty manifest")
body = "".join(lines)
manifest.write_text(body)
print(json.dumps({"files": len(lines), "bytes": total, "manifest_sha256": hashlib.sha256(body.encode()).hexdigest()}), flush=True)
