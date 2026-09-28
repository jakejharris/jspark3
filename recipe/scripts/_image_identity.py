# SPDX-License-Identifier: Apache-2.0
"""One image selection for build tools, host gates and container transforms.

The optional operator record is part of the private runtime's SHA256SUMS.
The original OCI document remains the historical reference, never a pull URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

RECIPE = Path(__file__).resolve().parents[1]
POLICY = {"policy": "verified-local-build-v1"}


class ImageRefusal(ValueError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_identity():
    path = RECIPE / "config/image-oci.json"
    value = json.loads(path.read_text())
    return {"manifest_digest": "sha256:" + sha(path),
            "config_digest": value["config"]["digest"],
            "diff_ids": [row["digest"] for row in value["layers"]]}


def build_policy():
    return json.loads((RECIPE / "config/image-build-policy.json").read_text())


def read_operator_record(path):
    if path.is_symlink() or not path.is_file():
        raise ImageRefusal("operator image receipt missing or symlinked")
    record = json.loads(path.read_text())
    keys = {"schema_version", "verification", "manifest_digest", "config_digest",
            "diff_ids", "build_policy", "source_recipe_sha256", "payload_sha256"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ImageRefusal("operator image receipt schema drift")
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    if record["payload_sha256"] != hashlib.sha256(canonical(payload)).hexdigest():
        raise ImageRefusal("operator image receipt payload hash mismatch")
    if (record["schema_version"] != 1 or
            record["verification"] != "fixed-dockerfile-build-and-instanttensor-check" or
            record["build_policy"] != build_policy()):
        raise ImageRefusal("operator image build policy drift")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["source_recipe_sha256"])):
        raise ImageRefusal("operator image source binding malformed")
    layers = record["diff_ids"]
    if not isinstance(layers, list) or not layers:
        raise ImageRefusal("operator image layers missing")
    for digest in [record["manifest_digest"], record["config_digest"], *layers]:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)):
            raise ImageRefusal("operator image digest malformed")
    return record


def selected_identity():
    path = RECIPE / "config/operator-image.json"
    # A dangling symlink must refuse, not silently select the reference.
    return read_operator_record(path) if path.exists() or path.is_symlink() else reference_identity()


def docker(argv):
    process = subprocess.run(["docker", *argv], capture_output=True, text=True)
    if process.returncode:
        raise ImageRefusal(f"docker {argv[0]} failed: {process.stderr.strip()}")
    return process.stdout


def inspect_image(image):
    rows = json.loads(docker(["image", "inspect", image]))
    if len(rows) != 1 or rows[0].get("Architecture") != "arm64" or rows[0].get("Os") != "linux":
        raise ImageRefusal("expected one linux/arm64 image")
    return rows[0]


def verify_instanttensor(image):
    """Hash files copied from a stopped container; no image code or GPU runs."""
    container = docker(["create", "--network", "none", "--entrypoint", "/bin/true", image]).strip()
    try:
        with tempfile.TemporaryDirectory(prefix="jspark3-image-check-") as directory:
            for name, expected in build_policy()["instanttensor_files"].items():
                destination = Path(directory) / "file"
                docker(["cp", f"{container}:/usr/local/lib/python3.12/dist-packages/{name}", str(destination)])
                if destination.is_symlink() or not destination.is_file() or sha(destination) != expected:
                    raise ImageRefusal(f"InstantTensor hash mismatch: {name}")
                destination.unlink()
    finally:
        docker(["rm", "-v", container])


def verify_local_image(identity):
    item = inspect_image(identity["config_digest"])
    if (item.get("Id") != identity["config_digest"] or
            item.get("RootFS", {}).get("Layers") != identity["diff_ids"]):
        raise ImageRefusal("local image config/layer identity drift")
    verify_instanttensor(identity["config_digest"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    identity = read_operator_record(args.receipt) if args.receipt else selected_identity()
    if args.receipt and identity["source_recipe_sha256"] != sha(RECIPE / "SHA256SUMS"):
        raise ImageRefusal("operator image was built for a different source recipe")
    print(identity["config_digest"])
