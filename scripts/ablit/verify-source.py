#!/usr/bin/env python3
"""Verify every root source file against the pinned public API inventory."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("source", type=Path)
p.add_argument("api", type=Path)
a = p.parse_args()
info = json.loads(a.api.read_text())
if info["sha"] != "c02a5f6fa06f0aa444877b44d19fd5c96390329f":
    raise SystemExit("Source revision mismatch")
files = sorted((x for x in info["siblings"] if "/" not in x["rfilename"]), key=lambda x: x["rfilename"])
tsv = "".join(f"{x['rfilename']}\t{x['size']}\t{x.get('lfs', {}).get('sha256', '-')}\n" for x in files)
if hashlib.sha256(tsv.encode()).hexdigest() != "483b8bec0a086353fc55903371408940f0481612441a4a745ac3e0088bf0c802":
    raise SystemExit("Source inventory hash mismatch")
for meta in files:
    name = meta["rfilename"]
    path = a.source / name
    if path.stat().st_size != meta["size"]:
        raise SystemExit("Source size mismatch: " + name)
    h = hashlib.sha256() if "lfs" in meta else hashlib.sha1(f"blob {meta['size']}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    if h.hexdigest() != meta.get("lfs", {}).get("sha256", meta["blobId"]):
        raise SystemExit("Source hash mismatch: " + name)
    print("VERIFIED", name, h.hexdigest(), flush=True)
print(json.dumps({"files_verified": len(files), "bytes_verified": sum(x["size"] for x in files)}))
