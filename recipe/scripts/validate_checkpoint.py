#!/usr/bin/env python3
"""Validate exact JSpark3 v1 serving bytes; never claim redistribution completeness."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

LEDGER_SHA = "cb0da1f97a53aebc3fbc5478f19c82b25586b0bf8533c99fb4ed5321a48f5342"
INDEX_SHA = "2f64d21c67c90bbafeb36c4e9b2f06f54063ed439e9f7cf95962d425a1d8515d"
TARGET_NATIVE = "4f5341e048984459471bfb9c894e6bf87e69b9c67402672af901631d1349f265"
TARGET_RUNTIME = "55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd"
DRAFT_NATIVE = "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573"
DRAFT_RUNTIME = "c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f"
DRAFT_MODEL = "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
TOKENIZER = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_CONFIG = "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"
INDEXED_BYTES = 175_622_979_576
PHYSICAL_BYTES = 175_642_157_752
SHARDS = 120
ROWS = 328
SHARD_RE = re.compile(r"model-(\d{5})-of-00120\.safetensors\Z")


class Refusal(RuntimeError):
    pass


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ledger(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise Refusal("target SHA256SUMS must be a regular file")
    if sha(path) != LEDGER_SHA:
        raise Refusal("target SHA256SUMS hash drift")
    rows: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise Refusal(f"malformed ledger row {number}") from exc
        pure = PurePosixPath(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or pure.is_absolute() or ".." in pure.parts:
            raise Refusal(f"unsafe ledger row {number}")
        if relative in rows:
            raise Refusal(f"duplicate ledger path {relative}")
        rows[relative] = expected
    if len(rows) != ROWS:
        raise Refusal(f"ledger has {len(rows)} rows, expected {ROWS}")
    return rows


def validate_runtime_view(source: Path, runtime: Path, expected_config: str) -> int:
    if runtime.is_symlink() or not runtime.is_dir() or runtime.parent.resolve() != source.parent.resolve():
        raise Refusal("runtime view must be a real sibling directory")
    config = runtime / "config.json"
    if not config.is_file() or config.is_symlink() or sha(config) != expected_config:
        raise Refusal("runtime config drift")
    source_names = {path.name for path in source.iterdir() if path.name != "config.json"}
    runtime_names = {path.name for path in runtime.iterdir() if path.name != "config.json"}
    if runtime_names != source_names:
        raise Refusal("runtime-view inventory drift")
    for name in source_names:
        link = runtime / name
        if not link.is_symlink() or link.resolve(strict=True) != (source / name).resolve(strict=True):
            raise Refusal(f"runtime-view link drift: {name}")
    return len(source_names)


def validate_target(root: Path, runtime: Path, workers: int) -> dict[str, object]:
    if root.is_symlink():
        raise Refusal("target native root must not be a symlink")
    rows = ledger(root / "SHA256SUMS")
    materialization = {
        f".materialization/shards/model-{index:05d}-of-00120.safetensors.json"
        for index in range(1, SHARDS + 1)
    }
    runtime_rows = {name for name in rows if name.startswith("runtime/")}
    if len(runtime_rows) != 72:
        raise Refusal("target runtime omission inventory drift")
    allowed_missing = materialization | runtime_rows
    missing = {name for name in rows if not (root / name).is_file()}
    if missing != allowed_missing:
        raise Refusal("target publication-only omission inventory drift")
    present = sorted(rows.keys() - missing)
    if any((root / name).is_symlink() for name in present):
        raise Refusal("target native serving files must not be symlinks")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        observed = dict(zip(present, pool.map(lambda name: sha(root / name), present)))
    mismatches = {name for name in present if observed[name] != rows[name]}
    if mismatches != {"LICENSE", "README.md"}:
        raise Refusal("target checksum mismatch inventory drift")
    critical = {
        "config.json": TARGET_NATIVE,
        "model.safetensors.index.json": INDEX_SHA,
        "tokenizer.json": TOKENIZER,
        "tokenizer_config.json": TOKENIZER_CONFIG,
    }
    for name, expected in critical.items():
        if observed.get(name) != expected:
            raise Refusal(f"target serving-critical drift: {name}")
    expected_shards = {f"model-{i:05d}-of-00120.safetensors" for i in range(1, SHARDS + 1)}
    if {p.name for p in root.glob("model-*.safetensors") if p.is_file()} != expected_shards:
        raise Refusal("target shard inventory drift")
    if {name for name in rows if SHARD_RE.fullmatch(name)} != expected_shards:
        raise Refusal("target ledger shard inventory drift")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if index.get("metadata", {}).get("total_size") != INDEXED_BYTES:
        raise Refusal("target indexed tensor-byte total drift")
    if set(index.get("weight_map", {}).values()) != expected_shards:
        raise Refusal("target index mapping drift")
    if sum((root / name).stat().st_size for name in expected_shards) != PHYSICAL_BYTES:
        raise Refusal("target physical safetensors-byte total drift")
    links = validate_runtime_view(root, runtime, TARGET_RUNTIME)
    return {"shards": SHARDS, "indexed_tensor_bytes": INDEXED_BYTES,
            "physical_bytes": PHYSICAL_BYTES, "runtime_links": links}


def validate_draft(root: Path, runtime: Path) -> dict[str, object]:
    if root.is_symlink():
        raise Refusal("draft native root must not be a symlink")
    expected = {"config.json": DRAFT_NATIVE, "model.safetensors": DRAFT_MODEL}
    for name, value in expected.items():
        if not (root / name).is_file() or (root / name).is_symlink() or sha(root / name) != value:
            raise Refusal(f"draft serving-critical drift: {name}")
    if (root / "model.safetensors").stat().st_size != 2_342_169_800:
        raise Refusal("draft model byte-size drift")
    links = validate_runtime_view(root, runtime, DRAFT_RUNTIME)
    config = json.loads((runtime / "config.json").read_text(encoding="utf-8"))
    if (config.get("num_attention_heads"), config.get("num_key_value_heads")) != (36, 9):
        raise Refusal("draft runtime GQA drift")
    if config.get("dflash_config", {}).get("target_layer_ids") != [5, 14, 24, 33, 42]:
        raise Refusal("draft zero-based tap drift")
    if config.get("is_causal") is not False:
        raise Refusal("draft causality drift")
    return {"shards": 1, "model_bytes": 2_342_169_800,
            "runtime_links": links, "target_layer_indices_zero_based": [5, 14, 24, 33, 42]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-runtime", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--draft-runtime", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        print("REFUSE: workers must be in [1,16]")
        return 9
    try:
        result = {
            "schema_version": 1,
            "grade": "ENGINEERING-EVIDENCE",
            "serving_checkpoint_pass": True,
            "publication_ledger_complete": False,
            "target": validate_target(args.target_root.absolute(), args.target_runtime.absolute(), args.workers),
            "draft": validate_draft(args.draft_root.absolute(), args.draft_runtime.absolute()),
        }
    except (OSError, ValueError, json.JSONDecodeError, Refusal) as exc:
        print(f"REFUSE: serving checkpoint validation failed: {exc}")
        return 9
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
