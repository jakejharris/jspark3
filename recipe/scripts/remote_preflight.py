#!/usr/bin/env python3
"""Read-only per-rank admission checks for the JSpark3 v1 recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import ipaddress
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import re

IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58"
IMAGE_CONFIG = "sha256:ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263"
TARGET_NATIVE = "Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
TARGET_RUNTIME = TARGET_NATIVE + "-tp3-runtime"
DRAFT_NATIVE = "incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native"
DRAFT_RUNTIME = DRAFT_NATIVE + "-tp3-runtime"
MIN_AVAILABLE_MEMORY = 72 * 1024**3
ALLOWED_LOCAL_STATE = {".env", "preflight.json", "jspark3-release-manifest.json", "verify.json", "verify-rank0.log"}
sys.dont_write_bytecode = True


class Refusal(RuntimeError):
    pass


def run(argv: list[str]) -> str:
    process = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    if process.returncode:
        detail = process.stderr.strip().splitlines()[-1:] or ["no detail"]
        raise Refusal(f"command failed: {argv[0]}: {detail[0]}")
    return process.stdout


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(values: list[str], label: str) -> str:
    if len(values) != 1:
        raise Refusal(f"{label}: expected one value")
    return values[0]


def exact_gpu_inventory(text: str) -> tuple[str, str]:
    rows = [[field.strip() for field in row] for row in csv.reader(text.splitlines())
            if any(field.strip() for field in row)]
    row = one(rows, "GPU inventory")
    if len(row) != 2:
        raise Refusal("GPU inventory row must contain exact name and compute capability")
    name = re.sub(r"\s+", " ", row[0]).strip()
    capability = row[1].strip()
    if name != "NVIDIA GB10" or capability != "12.1":
        raise Refusal("exactly one NVIDIA GB10 with compute capability 12.1 is required")
    return name, capability


def exact_system_product(values: list[str]) -> str:
    normalized = {re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()
                  for value in values if value.strip("\x00 \t\r\n")}
    aliases = {"NVIDIA DGX Spark", "DGX Spark"}
    if not normalized or not normalized <= aliases:
        raise Refusal("system product is not exactly NVIDIA DGX Spark")
    return "NVIDIA DGX Spark"


def system_product() -> str:
    values = []
    for path in (Path("/sys/devices/virtual/dmi/id/product_name"),
                 Path("/proc/device-tree/model")):
        try:
            values.append(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, UnicodeError):
            continue
    return exact_system_product(values)


def safe_component(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value) or value in {".", ".."}:
        raise Refusal(f"unsafe {label} name")


def verify_manifest(recipe: Path) -> str:
    manifest = recipe / "SHA256SUMS"
    if not manifest.is_file():
        raise Refusal("recipe SHA256SUMS missing")
    previous = ""
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, marker, name = line.partition("  ")
        if (marker != "  " or not re.fullmatch(r"[0-9a-f]{64}", digest) or
                Path(name).is_absolute() or ".." in Path(name).parts or
                name <= previous or name in seen):
            raise Refusal("invalid recipe manifest entry")
        seen.add(name)
        previous = name
        path = recipe / name
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise Refusal(f"recipe manifest mismatch: {name}")
    actual = set()
    for path in recipe.rglob("*"):
        relative = path.relative_to(recipe).as_posix()
        if path.is_symlink():
            raise Refusal(f"recipe symlink forbidden: {relative}")
        if path.is_dir():
            if path.name in {"__pycache__", ".git", ".cache"}:
                raise Refusal(f"generated/private recipe directory: {relative}")
            continue
        if not path.is_file():
            raise Refusal(f"non-regular recipe entry: {relative}")
        if relative != "SHA256SUMS" and relative not in ALLOWED_LOCAL_STATE:
            actual.add(relative)
    if actual != seen:
        raise Refusal(f"recipe inventory mismatch missing={sorted(seen - actual)} extra={sorted(actual - seen)}")
    return sha(manifest)


def available_memory() -> int:
    rows = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, marker, value = line.partition(":")
        if marker:
            rows[key] = value.strip()
    match = re.fullmatch(r"(\d+) kB", rows.get("MemAvailable", ""))
    if match is None:
        raise Refusal("MemAvailable is absent or malformed")
    return int(match.group(1)) * 1024


def verify_fly_sources(recipe: Path, fly: Path) -> None:
    sys.path.insert(0, str(recipe / "scripts"))
    try:
        contract = json.loads((recipe / "config/patch-contract.json").read_text(encoding="utf-8"))
        for module_name, section_name in (
            ("apply_tp3_overlay", "apply_tp3_overlay.py"),
            ("apply_image_glm_dflash", "apply_image_glm_dflash.py"),
        ):
            module = importlib.import_module(module_name)
            sources = contract["transforms"][section_name]["sources"]
            for name, relative in module.SOURCE_PATHS.items():
                path = fly / relative
                if not path.is_file() or sha(path) != sources[name]:
                    raise Refusal(f"pinned Fly source mismatch: {name}")
    finally:
        sys.path.pop(0)


def verify_network(ifaces: list[str], cidrs: list[str], hcas: list[str],
                   gid_index: int, socket_iface: str, management_addr: str,
                   peers: list[str]) -> None:
    if (len(ifaces) != 2 or len(hcas) != 2 or len(cidrs) != 2 or
            len(set(ifaces)) != 2 or len(set(hcas)) != 2):
        raise Refusal("two distinct fabric interfaces/HCAs are required")
    for value in ifaces:
        safe_component(value, "interface")
    for value in hcas:
        safe_component(value, "HCA")
    safe_component(socket_iface, "socket interface")
    if len(peers) != 2 or len(set(peers)) != 2:
        raise Refusal("two distinct peer management addresses are required")
    for peer in peers:
        if ipaddress.ip_address(peer).version != 4:
            raise Refusal("peer management addresses must be IPv4")
    for iface, cidr, hca in zip(ifaces, cidrs, hcas):
        iface_path = Path("/sys/class/net") / iface
        if not iface_path.is_dir() or (iface_path / "mtu").read_text().strip() != "9000":
            raise Refusal("fabric interface missing or MTU is not 9000")
        declared = ipaddress.ip_interface(cidr)
        if declared.version != 4:
            raise Refusal("fabric addresses must be IPv4")
        addresses = json.loads(run(["ip", "-j", "-4", "address", "show", "dev", iface]))
        observed = {
            ipaddress.ip_address(item["local"])
            for link in addresses for item in link.get("addr_info", [])
            if item.get("family") == "inet"
        }
        if declared.ip not in observed:
            raise Refusal("declared fabric address is not assigned")
        port = Path("/sys/class/infiniband") / hca / "ports/1"
        if (port / "state").read_text().split(":", 1)[-1].strip().upper() != "ACTIVE":
            raise Refusal("pinned HCA port is not ACTIVE")
        gid = ipaddress.ip_address((port / f"gids/{gid_index}").read_text().strip())
        gid_type = (port / f"gid_attrs/types/{gid_index}").read_text().strip().lower()
        ndev = (port / f"gid_attrs/ndevs/{gid_index}").read_text().strip()
        if gid.is_unspecified or "v2" not in gid_type or ndev != iface:
            raise Refusal("GID is not the declared active RoCE-v2 interface")
        mapped = getattr(gid, "ipv4_mapped", None)
        if mapped is None or mapped != declared.ip:
            raise Refusal("RoCE-v2 GID does not encode the declared IPv4 address")
    if not (Path("/sys/class/net") / socket_iface).is_dir():
        raise Refusal("management/socket interface missing")
    management = ipaddress.ip_address(management_addr)
    if management.version != 4:
        raise Refusal("management address must be IPv4")
    socket_addresses = json.loads(run(["ip", "-j", "-4", "address", "show", "dev", socket_iface]))
    assigned = {
        ipaddress.ip_address(item["local"])
        for link in socket_addresses for item in link.get("addr_info", [])
        if item.get("family") == "inet"
    }
    if management not in assigned:
        raise Refusal("declared management address is not assigned to the socket interface")
    for peer in peers:
        routes = json.loads(run(["ip", "-j", "route", "get", peer]))
        if not routes or routes[0].get("dev") != socket_iface:
            raise Refusal("peer management route leaves the wrong interface")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-only", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1, 2))
    parser.add_argument("--ifaces")
    parser.add_argument("--cidrs")
    parser.add_argument("--hcas")
    parser.add_argument("--gid-index", type=int)
    parser.add_argument("--socket-iface")
    parser.add_argument("--management-addr")
    parser.add_argument("--peer-addrs")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--expected-recipe-manifest-sha256")
    parser.add_argument("--fly-root", type=Path)
    args = parser.parse_args()
    try:
        recipe = args.recipe_root.resolve(strict=True)
        if args.recipe_root.is_symlink() or recipe != args.recipe_root:
            raise Refusal("recipe root must be an exact canonical directory")
        if args.recipe_only:
            recipe_sha = verify_manifest(recipe)
            if (args.expected_recipe_manifest_sha256 is not None and
                    (not re.fullmatch(r"[0-9a-f]{64}", args.expected_recipe_manifest_sha256) or
                     recipe_sha != args.expected_recipe_manifest_sha256)):
                raise Refusal("recipe manifest does not match the expected admission identity")
            print(json.dumps({"recipe_manifest_sha256": recipe_sha, "status": "PASS"},
                             sort_keys=True))
            return 0
        required = (args.rank, args.ifaces, args.cidrs, args.hcas, args.gid_index,
                    args.socket_iface, args.management_addr, args.peer_addrs,
                    args.model_root, args.work_root, args.fly_root)
        if any(value is None for value in required):
            raise Refusal("full preflight arguments are incomplete")
        if platform.machine() != "aarch64":
            raise Refusal("host architecture is not aarch64")
        system = system_product()
        gpu_name, gpu_capability = exact_gpu_inventory(run([
            "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"
        ]))
        image = json.loads(run(["docker", "image", "inspect", IMAGE]))[0]
        if (image.get("Id") != IMAGE_CONFIG or image.get("Architecture") != "arm64" or
                IMAGE not in (image.get("RepoDigests") or [])):
            raise Refusal("OCI manifest/config identity drift")
        name = f"jspark3-rank{args.rank}"
        absent = subprocess.run(["docker", "container", "inspect", name],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        if not absent:
            raise Refusal("deterministic release container name already exists")
        fly = args.fly_root.resolve(strict=True)
        models = args.model_root.resolve(strict=True)
        work_parent = args.work_root.parent.resolve(strict=True)
        if (args.fly_root.is_symlink() or fly != args.fly_root or
                args.model_root.is_symlink() or models != args.model_root or
                work_parent != args.work_root.parent or
                (args.work_root.exists() and args.work_root.is_symlink())):
            raise Refusal("model, work, or Fly root is not an exact canonical path")
        if shutil.disk_usage(models).free < 8 * 1024**3 or shutil.disk_usage(work_parent).free < 8 * 1024**3:
            raise Refusal("less than 8 GiB free at model/work path")
        recipe_manifest_sha256 = verify_manifest(recipe)
        verify_fly_sources(recipe, fly)
        verify_network(args.ifaces.split(","), args.cidrs.split(","),
                       args.hcas.split(","), args.gid_index,
                       args.socket_iface, args.management_addr,
                       args.peer_addrs.split(","))
        validation = subprocess.run([
            sys.executable, str(recipe / "scripts/validate_checkpoint.py"),
            "--target-root", str(models / TARGET_NATIVE),
            "--target-runtime", str(models / TARGET_RUNTIME),
            "--draft-root", str(models / DRAFT_NATIVE),
            "--draft-runtime", str(models / DRAFT_RUNTIME),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if validation.returncode:
            raise Refusal("checkpoint serving-byte gate failed")
        if available_memory() < MIN_AVAILABLE_MEMORY:
            raise Refusal("less than 72 GiB host memory available")
        print(json.dumps({
            "schema_version": 1,
            "rank": args.rank,
            "system": system,
            "architecture": "aarch64",
            "gpu": f"{gpu_name} / SM{gpu_capability.replace('.', '')}",
            "image": "PASS",
            "image_manifest": IMAGE.split("@", 1)[1],
            "image_config": IMAGE_CONFIG,
            "fabric_legs": 2,
            "gid_index": args.gid_index,
            "gid_entries": ["PASS", "PASS"],
            "management_routes": "PASS",
            "checkpoint": "PASS",
            "recipe_manifest": "PASS",
            "recipe_manifest_sha256": recipe_manifest_sha256,
            "free_memory": "PASS_GE_72_GIB",
            "free_disk": "PASS_GE_8_GIB_MODEL_AND_WORK",
            "release_name_absent": True,
            "status": "PASS",
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
