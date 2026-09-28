#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build ARM64 display and pinned coop artifacts by default and receipt verified inputs/outputs."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/scripts"))
from _coop_qualification import TARGET_NATIVE, verify_builder_host
from _image_identity import ImageRefusal, canonical, read_operator_record, sha, verify_local_image

# Same commands and /w paths as tools/v14/build_display_kv.sh. Do not run the
# resulting probe here: executing its kernels requires the target hardware.
DISPLAY_BUILD = """set -e
gcc -O2 -fPIC -shared -Wall -I/usr/local/cuda/include -o libglm53_display_kv.so display_kv.c -L/usr/local/cuda/lib64/stubs -lcuda
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 -o display_kv_probe probe_main.cu display_kv.c -lcuda
{ gcc --version | head -1; /usr/local/cuda/bin/nvcc --version | tail -1; } > toolchain.txt
"""
DISPLAY = "recipe/overlays/v14/display_kv"
COOP = "recipe/overlays/v16/coop"
OUTPUTS = {"display": {name: f"{DISPLAY}/{name}" for name in
                       ("libglm53_display_kv.so", "display_kv_probe")},
           "coop": {"out/cooperative_moe.so": f"{COOP}/bundle/cooperative_moe.so"}}


def builder_host():
    machine = Path('/etc/machine-id')
    if not machine.is_file() or not machine.read_text().strip():
        raise ImageRefusal('builder machine identity is unavailable')
    architecture = platform.machine()
    physical = None
    if architecture == 'aarch64':
        # Read-only NVML inventory, without creating a CUDA context or exposing
        # any GPU to the compiler containers. One integrated GB10 per board.
        query = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=10)
        rows = [line.strip().split(',') for line in query.stdout.splitlines() if line.strip()]
        if (len(rows) != 1 or len(rows[0]) != 2 or rows[0][0].strip() != 'NVIDIA GB10'
                or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', rows[0][1].strip())):
            raise ImageRefusal('native ARM64 builder requires exactly one physical GB10 UUID')
        physical = {'kind': 'gb10-gpu-uuid',
                    'uuid_sha256': hashlib.sha256(rows[0][1].strip().lower().encode()).hexdigest()}
    return {'architecture': architecture, 'machine_id_sha256': sha(machine), 'physical_identity': physical}


def local_docker():
    # The receipt must describe the host executing the build, not an SSH/TCP
    # client's hardware. Rootless local Unix sockets are also supported.
    endpoint = os.environ.get('DOCKER_HOST') if not os.environ.get('DOCKER_CONTEXT') else None
    if not endpoint:
        endpoint = subprocess.run(['docker', 'context', 'inspect', '--format', '{{.Endpoints.docker.Host}}'],
                                  check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not endpoint.startswith('unix:///'):
        raise ImageRefusal('native receipts require a local Docker Unix socket')
    return endpoint


def build_inputs():
    paths = [ROOT / "tools/build_native.py", ROOT / "recipe/scripts/_coop_qualification.py", ROOT / DISPLAY / "display_kv.c",
             ROOT / DISPLAY / "probe_main.cu", ROOT / COOP / "build_repro.sh",
             ROOT / COOP / "SOURCE_MANIFEST.json", *(ROOT / COOP / "source").rglob("*")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def read_native_record(path, image):
    from _coop_qualification import regular
    if path.is_symlink() or not path.is_file():
        raise ImageRefusal("operator native receipt missing or symlinked")
    regular(path.parent, path.name)
    record = json.loads(path.read_text())
    keys = {"schema_version", "verification", "source_recipe_sha256", "image_receipt_sha256",
            "build_inputs", "builder_host", "reproducibility", "binary_sha256", "hardware_qualified", "payload_sha256"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ImageRefusal("operator native receipt schema drift")
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    if record["payload_sha256"] != hashlib.sha256(canonical(payload)).hexdigest():
        raise ImageRefusal("operator native receipt payload hash mismatch")
    if (record["schema_version"] != 2 or record["verification"] != "fixed-native-build-v2"
            or record["hardware_qualified"] is not False
            or record["reproducibility"] != {"runs": 2, "comparison": "bit-identical"}
            or record["build_inputs"] != build_inputs()):
        raise ImageRefusal("operator native build inputs/recipe drift")
    if (record["source_recipe_sha256"] != sha(ROOT / "recipe/SHA256SUMS")
            or record["source_recipe_sha256"] != image["source_recipe_sha256"]
            or record["image_receipt_sha256"] != image["payload_sha256"]):
        raise ImageRefusal("operator native source/image binding drift")
    verify_builder_host(record['builder_host'])
    outputs = record["binary_sha256"]
    display = set(OUTPUTS["display"].values())
    with_coop = display | set(OUTPUTS["coop"].values())
    if (not isinstance(outputs, dict) or set(outputs) not in (display, with_coop)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in outputs.values())):
        raise ImageRefusal("operator native output inventory drift")
    if set(OUTPUTS["coop"].values()) <= set(outputs):
        if outputs[next(iter(OUTPUTS["coop"].values()))] != TARGET_NATIVE:
            raise ImageRefusal("coop native differs from the qualification candidate pin")
    return record


def build(kind, stage, image):
    stage.mkdir()
    if kind == "display":
        for name in ("display_kv.c", "probe_main.cu"):
            shutil.copyfile(ROOT / DISPLAY / name, stage / name)
        command = ["-c", DISPLAY_BUILD]
    else:
        for name in ("build_repro.sh", "SOURCE_MANIFEST.json"):
            shutil.copyfile(ROOT / COOP / name, stage / name)
        shutil.copytree(ROOT / COOP / "source", stage / "source")
        command = ["/w/build_repro.sh", "/w/out"]
    subprocess.run(["docker", "run", "--rm", "--platform", "linux/arm64",
                    "--network", "none", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
                    "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{stage}:/w", "-w", "/w", "--entrypoint", "bash",
                    image, *command], check=True)
    return {relative: sha(stage / name) for name, relative in OUTPUTS[kind].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new BINARY_ROOT outside the source tree")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--with-coop", action="store_true", help="compatibility alias for the default")
    mode.add_argument("--display-only", action="store_true", help="diagnostic only; cannot prepare coop-on")
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
        endpoint = local_docker()
        host = builder_host()
        verify_builder_host(host)
        inputs = build_inputs()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jspark3-native-", dir=output.parent) as directory:
            work = Path(directory)
            built = work / "artifacts"
            observed = {}
            for kind in (["display"] if args.display_only else ["display", "coop"]):
                first = build(kind, work / f"{kind}-a", image["config_digest"])
                second = build(kind, work / f"{kind}-b", image["config_digest"])
                if first != second:
                    raise ImageRefusal(f"{kind}: two native builds differ: {first} versus {second}")
                if kind == "coop" and first != {next(iter(OUTPUTS["coop"].values())): TARGET_NATIVE}:
                    raise ImageRefusal("matching coop builds differ from the candidate pin")
                for name, relative in OUTPUTS[kind].items():
                    target = built / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(work / f"{kind}-a" / name, target)
                    if target.is_symlink() or sha(target) != first[relative]:
                        raise ImageRefusal("copied native artifact differs from measured build")
                    target.chmod(0o755)
                observed.update(first)
            if build_inputs() != inputs:
                raise ImageRefusal("native build inputs changed during build")
            if local_docker() != endpoint or builder_host() != host:
                raise ImageRefusal('builder identity changed during build')
            record = {"schema_version": 2, "verification": "fixed-native-build-v2",
                      "source_recipe_sha256": image["source_recipe_sha256"],
                      "image_receipt_sha256": image["payload_sha256"], "build_inputs": inputs,
                      "builder_host": host,
                      "reproducibility": {"runs": 2, "comparison": "bit-identical"},
                      "binary_sha256": observed, "hardware_qualified": False}
            record["payload_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
            receipt = built / "native-build-receipt.json"
            receipt.write_bytes(canonical(record))
            read_native_record(receipt, image)
            for stage in work.glob("*-?"):
                if stage.is_dir():
                    shutil.move(str(stage), str(built / stage.name))
            built.rename(output)
        print(f"PASS verified native builds; BINARY_ROOT={output}; hardware qualification remains required")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
