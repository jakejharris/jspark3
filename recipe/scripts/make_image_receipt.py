#!/usr/bin/env python3
"""Create the self-hashed identity receipt consumed by source transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

MANIFEST = "sha256:a15b3e6056828219cabe19662fd239245e61fb2b45defe3a8e2712b75b925b3f"
CONFIG = "sha256:de01da91a1eefc7b2dee9df78c2e1b7abca64fbb9e19862a7872cc5338b6ce86"
SHA_RE = re.compile(r"[0-9a-f]{64}")


class Refusal(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--recipe-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        for label, value in (("container ID", args.container_id),
                             ("preflight SHA-256", args.preflight_sha256),
                             ("recipe manifest SHA-256", args.recipe_manifest_sha256)):
            if SHA_RE.fullmatch(value) is None:
                raise Refusal(f"invalid {label}")
        if args.output.exists() or args.output.is_symlink():
            raise Refusal("image receipt output already exists")
        value = {
            "schema_version": 2,
            "manifest_digest": MANIFEST,
            "config_digest": CONFIG,
            "verification": "host-observed-inspect-bound-create",
            "container_id": args.container_id,
            "rank": args.rank,
            "preflight_sha256": args.preflight_sha256,
            "recipe_manifest_sha256": args.recipe_manifest_sha256,
        }
        value["payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".image-receipt.", dir=args.output.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical(value))
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, args.output)
            parent = os.open(args.output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return 0
    except (OSError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
