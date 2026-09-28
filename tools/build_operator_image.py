#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned local image and record its verified identity. Never pushes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/scripts"))
from _image_identity import (ImageRefusal, build_policy, canonical, inspect_image,
                             read_operator_record, sha, verify_local_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new receipt outside the source tree")
    parser.add_argument("--builder", help="optional Docker Buildx builder name")
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink() or args.output.resolve().is_relative_to(ROOT):
            raise ImageRefusal("receipt output must be new and outside the source tree")
        from validate_release import verify
        if verify(ROOT)["failed"]:
            raise ImageRefusal("source export failed validation")
        policy = build_policy()
        context = ROOT / "docker/stock-v13"
        if (sha(context / "Dockerfile") != policy["dockerfile_sha256"] or
                json.loads((context / "instanttensor-files.json").read_text()) != policy["instanttensor_files"]):
            raise ImageRefusal("image build inputs differ from policy")
        # Snapshot only the two reviewed inputs. No caller Dockerfile, build args,
        # context additions, registry output or tag selection are accepted.
        with tempfile.TemporaryDirectory(prefix="jspark3-image-build-") as directory:
            work = Path(directory)
            for name in ("Dockerfile", "instanttensor-files.json"):
                shutil.copyfile(context / name, work / name)
            metadata = work / "metadata.json"
            command = ["docker", "buildx", "build", "--platform", "linux/arm64",
                       "--pull", "--no-cache", "--provenance=false", "--load",
                       "--metadata-file", str(metadata), "--progress", "plain"]
            if args.builder:
                command += ["--builder", args.builder]
            subprocess.run([*command, str(work)], check=True)
            built = json.loads(metadata.read_text())
            config = built["containerimage.config.digest"]
            item = inspect_image(config)
            record = {"schema_version": 1,
                      "verification": "fixed-dockerfile-build-and-instanttensor-check",
                      "manifest_digest": built["containerimage.digest"],
                      "config_digest": config, "diff_ids": item["RootFS"]["Layers"],
                      "build_policy": policy, "source_recipe_sha256": sha(ROOT / "recipe/SHA256SUMS")}
            verify_local_image(record)
            record["payload_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
            checked = work / "receipt.json"
            checked.write_bytes(canonical(record))
            read_operator_record(checked)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Write then link in the destination directory: no partial receipt or overwrite.
            fd, temporary = tempfile.mkstemp(dir=args.output.parent, prefix=".operator-image-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(canonical(record)); stream.flush(); os.fsync(stream.fileno())
                os.link(temporary, args.output)
            finally:
                os.unlink(temporary)
        print(f"PASS verified local image {config}; receipt {args.output}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
