#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build ARM64 display artifacts (optionally coop) and receipt verified inputs/outputs."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe/scripts"))
import _diagnostics as diagnostics
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/scripts"))
from _image_identity import ImageRefusal, canonical, read_operator_record, sha, verify_local_image

DISPLAY = "recipe/overlays/v14/display_kv"
COOP = "recipe/overlays/v16/coop"
OUTPUTS = {"display": {name: f"{DISPLAY}/{name}" for name in
                       ("libglm53_display_kv.so", "display_kv_probe")},
           "coop": {"out/cooperative_moe.so": f"{COOP}/bundle/cooperative_moe.so"}}


@contextmanager
def workspace(parent):
    """Publish only on success; preserve private build stages on any refusal."""
    work = Path(tempfile.mkdtemp(prefix="jspark3-native-private-", dir=parent))
    try:
        yield work
    except BaseException:
        print("Private native artifacts (do not share): " + str(work), file=sys.stderr)
        raise
    else:
        shutil.rmtree(work)


def compare_builds(kind, first, second):
    expected = set(OUTPUTS[kind].values())
    if (set(first) != expected or set(second) != expected
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                   for row in (first, second) for value in row.values())):
        raise ImageRefusal("native comparison inventory or digest malformed")
    changed = False
    for artifact in OUTPUTS[kind].values():
        if first[artifact] != second[artifact]:
            changed = True
            print(json.dumps({'status': 'REFUSE', 'reason': 'reproducibility mismatch',
                  'artifact': artifact, 'first_sha256': first[artifact], 'second_sha256': second[artifact]},
                  sort_keys=True), file=sys.stderr)
    if changed:
        raise ImageRefusal(kind + ": two native builds differ")


def build_inputs():
    paths = [ROOT / "tools/build_native.py", ROOT / DISPLAY / "display_kv.c",
             ROOT / DISPLAY / "probe_main.cu", ROOT / DISPLAY / "build_repro.sh", ROOT / COOP / "build_repro.sh",
             ROOT / COOP / "SOURCE_MANIFEST.json", *(ROOT / COOP / "source").rglob("*")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def read_native_record(path, image):
    if path.is_symlink() or not path.is_file():
        raise ImageRefusal("operator native receipt missing or symlinked")
    record = json.loads(path.read_text())
    keys = {"schema_version", "verification", "source_recipe_sha256", "image_receipt_sha256",
            "build_inputs", "reproducibility", "binary_sha256", "hardware_qualified", "payload_sha256"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ImageRefusal("operator native receipt schema drift")
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    if record["payload_sha256"] != hashlib.sha256(canonical(payload)).hexdigest():
        raise ImageRefusal("operator native receipt payload hash mismatch")
    if (record["schema_version"] != 1 or record["verification"] != "fixed-native-build-v1"
            or record["hardware_qualified"] is not False
            or record["reproducibility"] != {"runs": 2, "comparison": "bit-identical"}
            or record["build_inputs"] != build_inputs()):
        raise ImageRefusal("operator native build inputs/recipe drift")
    if (record["source_recipe_sha256"] != sha(ROOT / "recipe/SHA256SUMS")
            or record["source_recipe_sha256"] != image["source_recipe_sha256"]
            or record["image_receipt_sha256"] != image["payload_sha256"]):
        raise ImageRefusal("operator native source/image binding drift")
    outputs = record["binary_sha256"]
    display = set(OUTPUTS["display"].values())
    with_coop = display | set(OUTPUTS["coop"].values())
    if (not isinstance(outputs, dict) or set(outputs) not in (display, with_coop)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in outputs.values())):
        raise ImageRefusal("operator native output inventory drift")
    return record


def build(kind, stage, image, *, execute=None):
    stage.mkdir()
    if kind == "display":
        for name in ("display_kv.c", "probe_main.cu", "build_repro.sh"):
            shutil.copyfile(ROOT / DISPLAY / name, stage / name)
        command = ["/w/build_repro.sh"]
    else:
        for name in ("build_repro.sh", "SOURCE_MANIFEST.json"):
            shutil.copyfile(ROOT / COOP / name, stage / name)
        shutil.copytree(ROOT / COOP / "source", stage / "source")
        command = ["/w/build_repro.sh", "/w/out"]
    (execute or diagnostics.run_private)(["docker", "run", "--rm", "--platform", "linux/arm64",
                    "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{stage}:/w", "-w", "/w", "--entrypoint", "bash",
                    image, *command], check=True)
    return {relative: sha(stage / name) for name, relative in OUTPUTS[kind].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new BINARY_ROOT outside the source tree")
    parser.add_argument("--with-coop", action="store_true",
                        help="also attempt experimental coop builds; may fail byte identity and does not enable coop")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists() or args.output.is_symlink() or output.is_relative_to(ROOT):
            raise ImageRefusal("output must be new and outside the source tree")
        from validate_release import verify
        if verify(ROOT)["failed"]:
            raise ImageRefusal("source export failed validation")
        image = read_operator_record(args.image_receipt)
        if image["source_recipe_sha256"] != sha(ROOT / "recipe/SHA256SUMS"):
            raise ImageRefusal("image receipt belongs to a different source recipe")
        verify_local_image(image)
        inputs = build_inputs()
        output.parent.mkdir(parents=True, exist_ok=True)
        with workspace(output.parent) as work:
            built = work / "artifacts"
            observed = {}
            for kind in (["display", "coop"] if args.with_coop else ["display"]):
                first = build(kind, work / f"{kind}-a", image["config_digest"])
                second = build(kind, work / f"{kind}-b", image["config_digest"])
                compare_builds(kind, first, second)
                for name, relative in OUTPUTS[kind].items():
                    target = built / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(work / f"{kind}-a" / name, target)
                    target.chmod(0o755)
                observed.update(first)
            if build_inputs() != inputs:
                raise ImageRefusal("native build inputs changed during build")
            record = {"schema_version": 1, "verification": "fixed-native-build-v1",
                      "source_recipe_sha256": image["source_recipe_sha256"],
                      "image_receipt_sha256": image["payload_sha256"], "build_inputs": inputs,
                      "reproducibility": {"runs": 2, "comparison": "bit-identical"},
                      "binary_sha256": observed, "hardware_qualified": False}
            record["payload_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
            receipt = built / "native-build-receipt.json"
            receipt.write_bytes(canonical(record))
            read_native_record(receipt, image)
            built.rename(output)
        print(f"PASS verified native builds; BINARY_ROOT={output}; hardware qualification remains required")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
