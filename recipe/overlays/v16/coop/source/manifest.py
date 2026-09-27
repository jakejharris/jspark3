"""Offline provenance and artifact hashes; no Torch/CUDA import needed."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root):
    data = json.loads((root / "PROVENANCE.json").read_text())
    for relative, expected in data["upstreams"][1]["header_sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"vendored pinned header changed: {relative}")


def create(root):
    contract = {"abi": 2, "hidden": 4096, "intermediate_local": 2048,
                "topk": 8, "experts_max": 96, "row_capacities": [32, 64],
                "counter_lengths": {"32": 5635, "64": 11267}, "params_size": 344, "activation_limit": 10.0}
    paths = [root / name for name in ("runtime.py", "dispatch_policy.json", "cooperative_moe.so", "PROVENANCE.json", "toolchain.txt", "LICENSE.MIT", "LICENSE.upstream-AGPL-3.0", "LICENSE.exllamav3")]
    paths += sorted(q for q in (root / "source").rglob("*") if q.is_file())
    paths += sorted(q for q in (root / "headers").rglob("*") if q.is_file())
    data = {"schema": 1, "contract": contract,
            "files": {str(q.relative_to(root)): digest(q) for q in paths},
            "gpu_validation": "pending; build success is not correctness or performance proof"}
    (root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def verify_artifacts(root):
    root = root.resolve()
    data = json.loads((root / "manifest.json").read_text())
    if data.get("schema") != 1 or data.get("contract", {}).get("abi") != 2:
        raise RuntimeError("expected TP3 ABI2 manifest")
    files = data["files"]
    if not {"runtime.py", "cooperative_moe.so", "dispatch_policy.json"} <= files.keys():
        raise RuntimeError("incomplete manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise RuntimeError(f"artifact missing, changed or outside bundle: {relative}")
    policy = json.loads((root / "dispatch_policy.json").read_text())
    if policy.get("native_sha256") != files["cooperative_moe.so"]:
        raise RuntimeError("reprofile this native binary before preparing a serving overlay")
    return data


if __name__ == "__main__":
    action, directory = sys.argv[1:]
    {"verify-sources": verify_sources, "create": create,
     "verify-artifacts": verify_artifacts}[action](Path(directory))
