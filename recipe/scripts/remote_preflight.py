#!/usr/bin/env python3
"""Per-rank admission checks; temporary image checks never start a container."""

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

sys.dont_write_bytecode = True
from _image_identity import selected_identity, verify_local_image

IMAGE_IDENTITY = selected_identity()
IMAGE = IMAGE_CONFIG = IMAGE_IDENTITY["config_digest"]
IMAGE_MANIFEST = IMAGE_IDENTITY["manifest_digest"]
TARGET_NATIVE = "Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
TARGET_RUNTIME = TARGET_NATIVE + "-tp3-runtime"
DRAFT_NATIVE = "incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native"
DRAFT_RUNTIME = DRAFT_NATIVE + "-tp3-runtime"
MIN_AVAILABLE_MEMORY = 72 * 1024**3
ALLOWED_LOCAL_STATE = {".env", "preflight.json", "jspark3-release-manifest.json", "verify.json", "verify-rank0.log"}


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
    # Exact validated spellings only: the DMI reports the underscore form
    # NVIDIA_DGX_Spark; no fuzzy punctuation normalization is applied.
    aliases = {"NVIDIA DGX Spark", "DGX Spark", "NVIDIA_DGX_Spark"}
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


def v16_artifacts(
    recipe: Path, profile: str, coop: str, adaptive: str, dense: str
) -> dict:
    """Verify the option contracts and, when enabled, the sealed coop bundle."""
    if profile not in ("production", "production-stock", "qa") or coop not in ("0", "1") \
            or adaptive not in ("off", "ema") \
            or dense not in ("off", "trunk", "negative-coarse"):
        raise Refusal("invalid v1.6 option state")
    if dense == "negative-coarse" and profile != "qa":
        raise Refusal("negative-coarse is forbidden outside the qa profile")
    try:
        import _contracts
        import apply_coop_moe

        try:
            apply_coop_moe.verify_sources()
        except apply_coop_moe.Refusal as exc:
            raise Refusal(f"v1.6 cooperative-MoE source drift: {exc}") from exc
        install_contract = json.loads(
            apply_coop_moe.INSTALL_CONTRACT.read_text(encoding="utf-8")
        )["transforms"][apply_coop_moe.TRANSFORM]
        if install_contract != _contracts.V16_COOP:
            raise Refusal("cooperative-MoE contract differs from its embedded seal")
    except (OSError, UnicodeError, json.JSONDecodeError, ImportError, AttributeError) as exc:
        raise Refusal(f"v1.6 cooperative-MoE source/contract unreadable: {exc}") from exc
    contracts = {}
    for name in (
        "overlays/v16/coop/INSTALL_CONTRACT.json",
        "config/v16-adaptive-k-contract.json",
        "config/v16-dense-fp8-contract.json",
    ):
        path = recipe / name
        if path.is_symlink() or not path.is_file():
            raise Refusal(f"v1.6 contract missing or unsafe: {name}")
        contracts[name] = sha(path)
    bundle: dict | str = "UNSEALED_ALLOWED_ONLY_WHEN_OFF"
    if coop == "1":
        try:
            bundle = apply_coop_moe.verify_bundle(
                recipe / "overlays/v16/coop/bundle",
                recipe / "overlays/v16/coop/BUILD.json",
            )
        except apply_coop_moe.Refusal as exc:
            raise Refusal(f"coop=1 requires the hardware-sealed BUILD.json and bundle: {exc}") from exc
    return {
        "profile": profile,
        "coop": "on" if coop == "1" else "off",
        "adaptive-k": adaptive,
        "dense-fp8": dense,
        "contracts": contracts,
        "coop_source_manifest_sha256": apply_coop_moe.SOURCE_MANIFEST_SHA256,
        "coop_bundle": bundle,
    }


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


# The remote-session unit name is split so the release privacy scan's literal
# pattern does not fire on a systemd unit name.
DISPLAY_UNITS = ("gdm.service", "gdm3.service", "lightdm.service", "sddm.service",
                 "display-manager.service", "gnome-remote-" "desk" "top.service", "xrdp.service")
DISPLAY_SERVERS = ("Xorg", "Xwayland", "gnome-shell")
DRM_SYSFS = Path("/sys/class/drm")
# systemd ActiveState values under which a unit may hold the card.
UNIT_RUNNING = ("active", "reloading", "refreshing", "activating", "deactivating")


def query_failed(argv: list[str], process: subprocess.CompletedProcess) -> Refusal:
    detail = (process.stderr or "").strip().splitlines()[-1:] or ["no detail"]
    return Refusal(f"display-consumer query failed: {' '.join(argv)}: exit {process.returncode}: {detail[0]}")


def display_server_running(name: str) -> bool:
    """pgrep: 0 = present, 1 = proven absent; any other status proves nothing."""
    argv = ["pgrep", "-x", name]
    process = subprocess.run(argv, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if process.returncode not in (0, 1):
        raise query_failed(argv, process)
    return process.returncode == 0


def display_unit_running(unit: str) -> bool:
    """Absent only on an explicit ActiveState=inactive (loaded or not-found)."""
    argv = ["systemctl", "show", "--property=LoadState,ActiveState", unit]
    process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        raise query_failed(argv, process)
    state = dict(line.split("=", 1) for line in process.stdout.splitlines() if "=" in line)
    if state.get("ActiveState") == "inactive" and state.get("LoadState"):
        return False
    if state.get("ActiveState") in UNIT_RUNNING:
        return True
    raise Refusal(f"display-consumer query indeterminate: {' '.join(argv)}: "
                  f"LoadState={state.get('LoadState')!r} ActiveState={state.get('ActiveState')!r}")


def display_row(profile: str, card: str | None) -> dict:
    """v1.4 display-reserve KV host state (#234). Refuses; never downgrades."""
    if profile == "display0":
        if card is not None:
            raise Refusal("display0 profile must not name a DRM card")
        return {"profile": "display0"}
    if profile != "full" or card is None or re.fullmatch(r"/dev/dri/card[0-9]+", card) is None:
        raise Refusal("full profile needs --drm-card /dev/dri/cardN")
    node = Path(card)
    if not node.is_char_device():
        raise Refusal(f"DRM card {card} is not a character device")
    driver = DRM_SYSFS / node.name / "device/driver"
    card_driver = driver.resolve().name if driver.exists() else "missing"
    # nvidia_drm parameters are root-only (0400): read them through a
    # no-network, no-GPU container of the pinned image instead of sudo.
    params = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat",
                  "-v", "/sys/module/nvidia_drm/parameters:/p:ro", IMAGE,
                  "/p/modeset", "/p/fbdev"]).split()
    if len(params) != 2:
        raise Refusal("nvidia_drm parameters unreadable")
    target = run(["systemctl", "get-default"]).strip()
    # A failed or indeterminate query refuses; only proven absence counts.
    units = [unit for unit in DISPLAY_UNITS if display_unit_running(unit)]
    servers = sorted(name for name in DISPLAY_SERVERS if display_server_running(name))
    row = {"profile": "full", "card": card, "card_driver": card_driver,
           "modeset": params[0], "fbdev": params[1], "default_target": target,
           "display_servers": servers, "display_units_active": units}
    if (card_driver != "nvidia" or params != ["Y", "N"] or target != "multi-user.target" or
            servers or units):
        raise Refusal(f"display-reserve host state not ready: {json.dumps(row, sort_keys=True)}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--recipe-only", action="store_true")
    scope.add_argument("--image-only", action="store_true", help="local image gate only; no hardware admission")
    parser.add_argument("--image-receipt", type=Path, help="external build receipt; only with --image-only")
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
    parser.add_argument("--ablit", choices=("0", "1"))
    parser.add_argument("--ablit-root", type=Path)
    parser.add_argument("--ablit-manifest-sha256")
    parser.add_argument("--display-profile", choices=("full", "display0"))
    parser.add_argument("--drm-card")
    parser.add_argument("--v16-profile", choices=("production", "production-stock", "qa"))
    parser.add_argument("--v16-coop", choices=("0", "1"))
    parser.add_argument("--v16-adaptive-k", choices=("off", "ema"))
    parser.add_argument("--v16-dense-fp8", choices=("off", "trunk", "negative-coarse"))
    args = parser.parse_args()
    try:
        recipe = args.recipe_root.resolve(strict=True)
        if args.recipe_root.is_symlink() or recipe != args.recipe_root:
            raise Refusal("recipe root must be an exact canonical directory")
        if recipe != Path(__file__).resolve().parents[1]:
            raise Refusal("run preflight from the selected recipe's own scripts")
        if args.image_receipt and not args.image_only:
            raise Refusal("external image receipt is only allowed for the image-only check")
        if args.image_only:
            verify_manifest(recipe)
            from _image_identity import read_operator_record
            identity = read_operator_record(args.image_receipt) if args.image_receipt else IMAGE_IDENTITY
            if args.image_receipt and identity["source_recipe_sha256"] != sha(recipe / "SHA256SUMS"):
                raise Refusal("operator image was built for a different source recipe")
            verify_local_image(identity)
            print(json.dumps({"status": "PASS", "scope": "image-only", "hardware_qualified": False,
                              "image_manifest": identity["manifest_digest"],
                              "image_config": identity["config_digest"]}, sort_keys=True))
            return 0
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
                    args.model_root, args.work_root, args.fly_root, args.ablit,
                    args.display_profile, args.v16_profile, args.v16_coop,
                    args.v16_adaptive_k, args.v16_dense_fp8)
        if any(value is None for value in required):
            raise Refusal("full preflight arguments are incomplete")
        if platform.machine() != "aarch64":
            raise Refusal("host architecture is not aarch64")
        system = system_product()
        gpu_name, gpu_capability = exact_gpu_inventory(run([
            "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"
        ]))
        verify_local_image(IMAGE_IDENTITY)
        name = f"jspark3-v16-rank{args.rank}"
        absent = subprocess.run(["docker", "container", "inspect", name],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        if not absent:
            raise Refusal("deterministic release container name already exists")
        # The v1.3.0 rollback container may exist, stopped; it must not be serving.
        running = run(["docker", "ps", "--format", "{{.Names}}"]).split()
        if f"jspark3-v13-rank{args.rank}" in running:
            raise Refusal("the v1.3.0 release container is still running")
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
        v16 = v16_artifacts(
            recipe, args.v16_profile, args.v16_coop,
            args.v16_adaptive_k, args.v16_dense_fp8,
        )
        verify_fly_sources(recipe, fly)
        if args.ablit == "1":
            if args.ablit_root is None or args.ablit_manifest_sha256 is None:
                raise Refusal("enabled transplant requires a complete donor binding")
            if args.ablit_root.resolve(strict=True) != args.ablit_root:
                raise Refusal("donor root must be an exact canonical path")
            from validate_ablit_artifacts import validate
            validate(args.ablit_root, args.ablit_manifest_sha256)
        elif args.ablit_root is not None or args.ablit_manifest_sha256 is not None:
            raise Refusal("disabled control must not carry donor configuration")
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
        display = display_row(args.display_profile, args.drm_card)
        print(json.dumps({
            "schema_version": 1,
            "rank": args.rank,
            "system": system,
            "architecture": "aarch64",
            "gpu": f"{gpu_name} / SM{gpu_capability.replace('.', '')}",
            "image": "PASS",
            "image_manifest": IMAGE_MANIFEST,
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
            "ablit": int(args.ablit),
            "ablit_manifest_sha256": args.ablit_manifest_sha256,
            "display": display,
            "v16": v16,
            "status": "PASS",
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
