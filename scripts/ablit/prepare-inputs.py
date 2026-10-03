#!/usr/bin/env python3
"""Verify the pinned base snapshot and derive the converter's public inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vontra", type=Path)
    p.add_argument("source_api", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    sizes = 0
    for line in (here / "vontra.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = a.vontra / name
        with path.open("rb") as source:
            got = hashlib.file_digest(source, "sha256").hexdigest()
        if got != expected:
            raise SystemExit("Base input SHA-256 mismatch: " + name)
        sizes += path.stat().st_size
        print(f"VERIFIED {got}  {name}", flush=True)
    targets = {}
    for path in sorted(a.vontra.glob("model-*.safetensors")):
        with path.open("rb") as source:
            size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in targets:
                raise SystemExit("Duplicate base tensor: " + name)
            targets[name] = [info["dtype"], info["shape"], path.name]
    if len(targets) != 114160:
        raise SystemExit("Unexpected target tensor count")
    (a.output / "target.json").write_text(json.dumps(targets, sort_keys=True) + "\n")
    info = json.loads(a.source_api.read_text())
    if info["sha"] != "c02a5f6fa06f0aa444877b44d19fd5c96390329f":
        raise SystemExit("Source API revision mismatch")
    files = sorted((x for x in info["siblings"] if "/" not in x["rfilename"]), key=lambda x: x["rfilename"])
    tsv = "".join(f"{x['rfilename']}\t{x['size']}\t{x.get('lfs', {}).get('sha256', '-')}\n" for x in files)
    (a.output / "source-FILES.tsv").write_text(tsv)
    if hashlib.sha256(tsv.encode()).hexdigest() != "483b8bec0a086353fc55903371408940f0481612441a4a745ac3e0088bf0c802":
        raise SystemExit("Source inventory mismatch")
    names = {"config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"}
    small = "".join(line + "\n" for line in (here / "vontra.sha256").read_text().splitlines() if line.split("  ", 1)[1] in names)
    (a.output / "small.sha256").write_text(small)
    print(json.dumps({"base_files_verified": 54, "base_bytes_verified": sizes, "target_tensors": len(targets), "source_inventory_sha256": hashlib.sha256(tsv.encode()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
