#!/usr/bin/env python3
"""Create hash-bound sibling runtime views without changing downloaded trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

TARGET_NATIVE = "4f5341e048984459471bfb9c894e6bf87e69b9c67402672af901631d1349f265"
TARGET_RUNTIME = "55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd"
DRAFT_NATIVE = "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573"
DRAFT_RUNTIME = "c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f"


class Refusal(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(kind: str, source: bytes) -> bytes:
    expected = TARGET_NATIVE if kind == "target" else DRAFT_NATIVE
    if digest(source) != expected:
        raise Refusal(f"{kind} native config hash drift")
    value = json.loads(source)
    if kind == "target":
        text = value.get("text_config")
        if not isinstance(text, dict):
            raise Refusal("target text_config missing")
        if (text.get("num_attention_heads"), text.get("num_key_value_heads")) != (64, 64):
            raise Refusal("target native head geometry drift")
        linear = text.get("linear_attn_config")
        if not isinstance(linear, dict) or linear.get("num_heads") != 64:
            raise Refusal("target linear head geometry drift")
        text["num_attention_heads"] = 66
        text["num_key_value_heads"] = 66
        text["linear_num_heads"] = 66
        linear["num_heads"] = 66
        expected_out = TARGET_RUNTIME
    else:
        if (value.get("num_attention_heads"), value.get("num_key_value_heads")) != (32, 8):
            raise Refusal("draft native GQA geometry drift")
        if value.get("dflash_config", {}).get("target_layer_ids") != [5, 14, 24, 33, 42]:
            raise Refusal("draft zero-based target-layer taps drift")
        if value.get("is_causal") is not False:
            raise Refusal("draft is_causal drift")
        value["num_attention_heads"] = 36
        value["num_key_value_heads"] = 9
        expected_out = DRAFT_RUNTIME
    output = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if digest(output) != expected_out:
        raise Refusal(f"{kind} derived config hash mismatch")
    return output


def validate_existing(source: Path, destination: Path, expected_config: bytes) -> bool:
    if destination.is_symlink():
        raise Refusal(f"destination must not be a symlink: {destination}")
    if not destination.exists():
        return False
    if not destination.is_dir() or (destination / "config.json").read_bytes() != expected_config:
        raise Refusal(f"existing destination drift: {destination}")
    source_names = {p.name for p in source.iterdir() if p.name != "config.json"}
    destination_names = {p.name for p in destination.iterdir() if p.name != "config.json"}
    if source_names != destination_names:
        raise Refusal(f"existing runtime-view inventory drift: {destination}")
    for name in source_names:
        link = destination / name
        if not link.is_symlink() or link.resolve() != (source / name).resolve():
            raise Refusal(f"existing runtime-view link drift: {link}")
    return True


def build(kind: str, source: Path, destination: Path) -> str:
    source = source.resolve()
    destination = destination.absolute()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise Refusal(f"missing {kind} source tree")
    if destination.parent.resolve() != source.parent.resolve():
        raise Refusal("runtime view must be a sibling of its immutable source")
    if destination.resolve() == source:
        raise Refusal("source and destination must differ")
    derived = transform(kind, (source / "config.json").read_bytes())
    if validate_existing(source, destination, derived):
        return "ALREADY_VALID"

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for item in source.iterdir():
            if item.name != "config.json":
                # The model parent is mounted at a different absolute path in
                # the container. Relative sibling links survive that rebase;
                # host-absolute links do not.
                target = os.path.relpath(item.absolute(), start=destination)
                os.symlink(target, temporary / item.name)
        (temporary / "config.json").write_bytes(derived)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "CREATED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--target-view", type=Path, required=True)
    parser.add_argument("--draft-source", type=Path, required=True)
    parser.add_argument("--draft-view", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {
            "target": build("target", args.target_source, args.target_view),
            "draft": build("draft", args.draft_source, args.draft_view),
            "target_runtime_config_sha256": TARGET_RUNTIME,
            "draft_runtime_config_sha256": DRAFT_RUNTIME,
        }
    except (OSError, ValueError, json.JSONDecodeError, Refusal) as exc:
        print(f"REFUSE: {exc}")
        return 9
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
