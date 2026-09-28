#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build ARM64 display and pinned coop artifacts by default and receipt verified inputs/outputs."""
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe/scripts"))
import _diagnostics as diagnostics
import shutil
import re
import socket
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/scripts"))
from _coop_qualification import TARGET_NATIVE, verify_builder_host
from _image_identity import ImageRefusal, canonical, read_operator_record, sha, verify_local_image

DISPLAY = "recipe/overlays/v14/display_kv"
COOP = "recipe/overlays/v16/coop"
OUTPUTS = {"display": {name: f"{DISPLAY}/{name}" for name in
                       ("libglm53_display_kv.so", "display_kv_probe")},
           "coop": {"out/cooperative_moe.so": f"{COOP}/bundle/cooperative_moe.so"}}

# Linux initial namespace inode numbers (include/linux/proc_ns.h). Unlike
# /proc/1/ns, these self links are readable by an ordinary operator. Unknown
# layouts refuse collection; never infer host execution from machine()/uname.
INITIAL_NAMESPACES = {'ipc': 0xEFFFFFFF, 'uts': 0xEFFFFFFE, 'user': 0xEFFFFFFD,
                      'pid': 0xEFFFFFFC, 'pid_for_children': 0xEFFFFFFC,
                      'cgroup': 0xEFFFFFFB, 'time': 0xEFFFFFFA, 'time_for_children': 0xEFFFFFFA}
IDENTITY_ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C'}


def require_host():
    """Collection hygiene on the supported Linux host, not hardware attestation."""
    if any(Path(name).exists() for name in ('/.dockerenv', '/run/.containerenv', '/run/systemd/container')):
        raise ImageRefusal('physical identity collection requires the host, not a container')
    for name, inode in INITIAL_NAMESPACES.items():
        if Path('/proc/self/ns/' + name).stat().st_ino != inode:
            raise ImageRefusal('physical identity collection requires the initial ' + name + ' namespace')
    # Mount namespaces have no stable initial inode on the supported kernels.
    # Cloning one allocates new mount IDs; compare the root mount with PID 1.
    roots = []
    for process in ('1', 'self'):
        rows = [line.split()[:5] for line in Path('/proc/' + process + '/mountinfo').read_text().splitlines()]
        roots.append([row for row in rows if len(row) == 5 and row[4] == '/'])
    if len(roots[0]) != 1 or roots[0] != roots[1]:
        raise ImageRefusal('physical identity collection requires the host mount namespace/root')
    # Linux initializes init_net first, with cookie 1. Reading SO_NETNS_COOKIE
    # on an unbound socket emits no traffic and needs no access to PID 1's ns.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        cookie = int.from_bytes(probe.getsockopt(1, 71, 8), sys.byteorder)  # Linux socket level, SO_NETNS_COOKIE
    if cookie != 1:
        raise ImageRefusal('physical identity collection requires the initial network namespace')


def trusted_nvidia_smi():
    executable = Path('/usr/bin/nvidia-smi')
    resolved = executable.resolve(strict=True)
    for path in {executable, resolved, *executable.parents, *resolved.parents}:
        info = path.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ImageRefusal('physical identity executable/path must be root-owned and not group/world writable')
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
        raise ImageRefusal('physical identity executable is not a regular executable')
    return str(executable)


def builder_host():
    machine = Path('/etc/machine-id')
    if not machine.is_file() or not machine.read_text().strip():
        raise ImageRefusal('builder machine identity is unavailable')
    architecture = platform.machine()
    physical = None
    if architecture == 'aarch64':
        require_host()
        # Read-only NVML inventory, without creating a CUDA context or exposing
        # any GPU to the compiler containers. One integrated GB10 per board.
        query = subprocess.run([trusted_nvidia_smi(), '--query-gpu=name,uuid', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=10,
                               env=IDENTITY_ENV, cwd='/', stdin=subprocess.DEVNULL)
        diagnostics.retain(query.stdout + (query.stderr or ''))
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
    paths = [ROOT / "tools/build_native.py", ROOT / "recipe/scripts/_coop_qualification.py", ROOT / DISPLAY / "display_kv.c",
             ROOT / DISPLAY / "probe_main.cu", ROOT / DISPLAY / "build_repro.sh", ROOT / COOP / "build_repro.sh",
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
        with workspace(output.parent) as work:
            built = work / "artifacts"
            observed = {}
            for kind in (["display"] if args.display_only else ["display", "coop"]):
                first = build(kind, work / f"{kind}-a", image["config_digest"])
                second = build(kind, work / f"{kind}-b", image["config_digest"])
                compare_builds(kind, first, second)
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
        print("PASS verified native builds; hardware qualification remains required")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
