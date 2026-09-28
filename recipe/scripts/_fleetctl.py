#!/usr/bin/env python3
"""Fail-closed three-node lifecycle controller for the JSpark3 v1.6 recipe."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

import production_stock

from _image_identity import selected_identity

IMAGE_IDENTITY = selected_identity()
IMAGE = IMAGE_CONFIG = IMAGE_IDENTITY["config_digest"]
IMAGE_MANIFEST = IMAGE_IDENTITY["manifest_digest"]
RELEASE_LABEL = "v1.6.0"
CONTAINER_PREFIX = "jspark3-v16-rank"
V14_PROFILES = ("full", "display0")
DISPLAY_KV_LIB = "/recipe/overlays/v14/display_kv/libglm53_display_kv.so"
DRM_CARD_RE = re.compile(r"/dev/dri/card[0-9]+")
# The measured Cadence B4+B5 layer, pinned in config/cadence-contract.json.
B45_ENV = {
    "B45_COMBINED": "1",
    "B5_PREFIX_VERIFY": "1",
    "B5_CALIB_REPLAYS": "20",
    "B5_T4_SEED_MS": "74.30",
    "B5_T7_SEED_MS": "92.53",
    "B5_OUT": "/tmp/b45",
    "JSPARK3_KDA_QKV_SHADOW": "1",
    "B4_CAPTURE_ORDER": '["bf16_0","int8_0"]',
}
B45_MODULES = {
    "b45_bootstrap.py": "a19fd73e683be553a5880e12f1cce1bd0e8ac793217646e9c47573008b3439e3",
    "b45_graphs.py": "d8b165b0c63775cd25c1962a8c2ad2d6ab726a449065e33ea362bf437da6e251",
    "b5_controller.py": "1f7cea92cbaba7309074f813681988c7f7dc4a2e5260383540700ca76e6df11d",
    "b5_prefix_verify.py": "778fa7119158b0d1cdf23964e55bdbb7f65b6cf4d9a060281eb5f5490a465e78",
    "kda_mixed_output_blocks.py": "1628a3c1581c36f9d2dd1453b2169a0556c36938343cd0a0d0705c2f3e19160a",
}
B45_PTH_SHA256 = "eea018d5bfee8fdc28e6470f650b8f4adaa5ec4cc4f8ac86b3e8849368c9fdeb"
B45_KDA_ORIGINAL_SHA256 = "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5"
B5_RECEIPT_MARKER = "B5_PREFIX_VERIFY_RECEIPT rank=0 T4_seed_ms=74.300 T7_seed_ms=92.530 narrow_width=3"
V15_CUDAGRAPH_CAPTURE_SIZES = (8, 16, 24, 32, 48)
ADAPTIVE_K_CUDAGRAPH_CAPTURE_SIZES = (4, 8, 12, 16, 20, 24, 28, 32, 48)
MEMORY = 68719476736
SHM = 34359738368
TARGET_RUNTIME = "/models/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime"
DRAFT_RUNTIME = "/models/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime"
TARGET_NATIVE_NAME = "Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
DRAFT_NATIVE_NAME = "incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native"
FORBIDDEN_NCCL = ("NCCL_PROTO", "NCCL_ALGO", "NCCL_IB_ADDR_RANGE")
REQUIRED_ENV = (
    "JSPARK_RANK0_HOST", "JSPARK_RANK1_HOST", "JSPARK_RANK2_HOST",
    "JSPARK_RANK0_ADDR", "JSPARK_RANK1_ADDR", "JSPARK_RANK2_ADDR",
    "JSPARK_MASTER_ADDR", "JSPARK_FABRIC_IFACES_0", "JSPARK_FABRIC_IFACES_1",
    "JSPARK_FABRIC_IFACES_2", "JSPARK_FABRIC_ADDRS_0", "JSPARK_FABRIC_ADDRS_1",
    "JSPARK_FABRIC_ADDRS_2", "JSPARK_HCAS_0", "JSPARK_HCAS_1", "JSPARK_HCAS_2",
    "JSPARK_SOCKET_IFNAME_0", "JSPARK_SOCKET_IFNAME_1", "JSPARK_SOCKET_IFNAME_2",
    "JSPARK_IB_GID_INDEX", "JSPARK_MODEL_ROOT", "JSPARK_WORK_ROOT",
    "JSPARK_RECIPE_ROOT", "JSPARK_FLY_ROOT", "JSPARK_MASTER_PORT",
    "JSPARK_API_BIND", "JSPARK_API_PORT", "ABLIT", "JSPARK_GPU_MEMORY_UTILIZATION",
    "JSPARK3_V14_PROFILE",
)
OPTIONAL_ENV = ("ABLIT_METHOD", "ABLIT_LAYERS", "ABLIT_INCLUDE_MTP", "JSPARK_ABLIT_ROOT", "JSPARK_ABLIT_MANIFEST_SHA256")
V16_ENV = (
    "JSPARK3_V16_PROFILE", "JSPARK3_V16_COOP", "GLM53_ADAPTIVE_K",
    "JSPARK3_V16_DENSE_FP8", "JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL",
    "JSPARK3_V16_APC_LRU",
)
# Host DRM node per rank (resolved at preflight); required exactly under the full profile.
DRM_CARD_ENV = ("JSPARK_DRM_CARD_0", "JSPARK_DRM_CARD_1", "JSPARK_DRM_CARD_2")
SHA_RE = re.compile(r"[0-9a-f]{64}")
RECIPE_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ROW_KEYS = {
    "schema_version", "rank", "system", "architecture", "gpu", "image",
    "image_manifest", "image_config", "fabric_legs", "gid_index",
    "gid_entries", "management_routes", "checkpoint", "recipe_manifest",
    "recipe_manifest_sha256", "free_memory", "free_disk",
    "release_name_absent", "ablit", "ablit_manifest_sha256", "display", "v16", "status",
}


class Refusal(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Refusal(f"cannot read env file: {exc}") from exc
    values: dict[str, str] = {}
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, mark, value = line.partition("=")
        if not mark or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise Refusal(f"invalid env syntax at line {number}")
        if key in values:
            raise Refusal(f"duplicate env key: {key}")
        if (len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"):
            value = value[1:-1]
        if "$" in value or "`" in value or "\n" in value:
            raise Refusal(f"shell expansion is not supported: {key}")
        values[key] = value
    missing = [key for key in REQUIRED_ENV if not values.get(key)]
    if missing:
        raise Refusal(f"missing env keys: {','.join(missing)}")
    unknown = sorted(set(values) - set(REQUIRED_ENV) - set(OPTIONAL_ENV) - set(DRM_CARD_ENV) - set(V16_ENV))
    if unknown:
        raise Refusal(f"unknown env keys: {','.join(unknown)}")
    return values


def container_name(rank: int) -> str:
    return f"{CONTAINER_PREFIX}{rank}"


def v14_expectation(profile: str) -> str:
    """The sealed identity expectation, exactly as container_entry.sh reads it."""
    if profile not in V14_PROFILES:
        raise Refusal("JSPARK3_V14_PROFILE must be full or display0")
    path = RECIPE_ROOT / "config" / f"v14-expect-{profile}.json"
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"sealed expectation missing: {path.name}")
    text = path.read_text(encoding="utf-8").rstrip("\n")
    if "\n" in text or not isinstance(json.loads(text), dict):
        raise Refusal(f"sealed expectation must be one JSON object line: {path.name}")
    return text


def v16_options(values: dict[str, str]) -> dict[str, str]:
    """Return the strict public v1.6 profile, defaulting legacy callers all-off."""
    return {
        "profile": values.get("JSPARK3_V16_PROFILE", "production"),
        "coop": values.get("JSPARK3_V16_COOP", "0"),
        "adaptive-k": values.get("GLM53_ADAPTIVE_K", "off"),
        "dense-fp8": values.get("JSPARK3_V16_DENSE_FP8", "off"),
        "apc-lru": values.get("JSPARK3_V16_APC_LRU", "0"),
    }


def cudagraph_capture_sizes(values: dict[str, str]) -> list[int]:
    """Keep off at v1.5 bytes; ema adds exact query-4 C1..C8 totals."""
    adaptive_k = v16_options(values)["adaptive-k"]
    if adaptive_k == "off":
        return list(V15_CUDAGRAPH_CAPTURE_SIZES)
    if adaptive_k == "ema":
        return list(ADAPTIVE_K_CUDAGRAPH_CAPTURE_SIZES)
    raise Refusal("GLM53_ADAPTIVE_K must be off or ema")


def v16_expectation(values: dict[str, str]) -> str:
    """Extend the sealed v1.4 identity with the selected v1.6 state."""
    options = v16_options(values)
    value = json.loads(v14_expectation(values["JSPARK3_V14_PROFILE"]))
    value["coop"] = "on" if options["coop"] == "1" else "off"
    value["adaptive-k"] = options["adaptive-k"]
    value["dense-fp8"] = options["dense-fp8"]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def split2(value: str, label: str) -> list[str]:
    result = [part.strip() for part in value.split(",")]
    if (len(result) != 2 or any(not part for part in result) or len(set(result)) != 2 or
            value != ",".join(result)):
        raise Refusal(f"{label} must contain two distinct comma-separated values")
    return result


def validate_env(values: dict[str, str]) -> None:
    if any(key in os.environ for key in FORBIDDEN_NCCL) or any(key in values for key in FORBIDDEN_NCCL):
        raise Refusal("forbidden inherited NCCL override")
    for key in (
        "GLM53_EXL3_MOE_FAST", "GLM53_COOP_GEOMETRY", "HAREM_KDA_FLASHKDA",
        "GLM53_KDA_BF16_LARGE_M", "GLM53_COOPERATIVE_MOE",
        "GLM53_COOP_QUALIFICATION", "GLM53_COOP_MAINTENANCE_TEST",
        "JSPARK3_V16_COOP_MAINTENANCE", "GLM53_DENSE_FP8", "GLM53_APC_DRAFT_LRU",
    ):
        if key in os.environ:
            raise Refusal(f"unsupported inherited TP3 override: {key}")
    if "ABLIT" in os.environ and os.environ["ABLIT"] != values["ABLIT"]:
        raise Refusal("inherited ABLIT conflicts with the declared profile")
    hosts = [values[f"JSPARK_RANK{rank}_HOST"] for rank in range(3)]
    addresses = [values[f"JSPARK_RANK{rank}_ADDR"] for rank in range(3)]
    if len(set(hosts)) != 3 or len(set(addresses)) != 3:
        raise Refusal("three distinct SSH hosts and management addresses are required")
    if any(re.fullmatch(r"[A-Za-z0-9_.:@-]{1,255}", item) is None for item in hosts):
        raise Refusal("SSH host labels must use a safe non-shell form")
    if values["JSPARK_MASTER_ADDR"] != addresses[0]:
        raise Refusal("master address must equal rank 0 management address")
    for address in addresses:
        if ipaddress.ip_address(address).version != 4:
            raise Refusal("management addresses must be IPv4")
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    all_ips = []
    for rank in range(3):
        split2(values[f"JSPARK_FABRIC_IFACES_{rank}"], f"rank{rank} fabric interfaces")
        split2(values[f"JSPARK_HCAS_{rank}"], f"rank{rank} HCAs")
        cidrs = split2(values[f"JSPARK_FABRIC_ADDRS_{rank}"], f"rank{rank} fabric addresses")
        interfaces = [ipaddress.ip_interface(item) for item in cidrs]
        if any(item.version != 4 for item in interfaces):
            raise Refusal("fabric addresses must be IPv4")
        if interfaces[0].network == interfaces[1].network:
            raise Refusal(f"rank{rank} fabric legs share one network")
        networks.extend(item.network for item in interfaces)
        all_ips.extend(item.ip for item in interfaces)
    if len(set(all_ips)) != 6:
        raise Refusal("fabric address reused")
    network_counts = {network: networks.count(network) for network in set(networks)}
    if len(network_counts) != 3 or set(network_counts.values()) != {2}:
        raise Refusal("fabric CIDRs do not form a three-edge pairwise triangle")
    profile = values["JSPARK3_V14_PROFILE"]
    if profile not in V14_PROFILES:
        raise Refusal("JSPARK3_V14_PROFILE must be full or display0")
    cards = [values.get(key) for key in DRM_CARD_ENV]
    if profile == "full":
        if any(card is None or DRM_CARD_RE.fullmatch(card) is None for card in cards):
            raise Refusal("full profile needs JSPARK_DRM_CARD_0..2 as /dev/dri/cardN")
    elif any(card is not None for card in cards):
        raise Refusal("display0 profile must not declare DRM cards")
    v14_expectation(profile)
    options = v16_options(values)
    if options["profile"] not in ("production", "production-stock", "qa"):
        raise Refusal("JSPARK3_V16_PROFILE must be production, production-stock or qa")
    if options["profile"] == production_stock.PROFILE:
        try:
            production_stock.environment(values)
            if production_stock.DONOR_KEYS.intersection(os.environ):
                raise ValueError("production-stock: inherited donor keys must be absent")
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
    if options["coop"] not in ("0", "1"):
        raise Refusal("JSPARK3_V16_COOP must be 0 or 1")
    if options["adaptive-k"] not in ("off", "ema"):
        raise Refusal("GLM53_ADAPTIVE_K must be off or ema")
    if options["dense-fp8"] not in ("off", "trunk", "negative-coarse"):
        raise Refusal("JSPARK3_V16_DENSE_FP8 must be off, trunk or negative-coarse")
    if options["apc-lru"] not in ("0", "1"):
        raise Refusal("JSPARK3_V16_APC_LRU must be 0 or 1")
    acknowledgement = values.get("JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL")
    if options["dense-fp8"] == "negative-coarse":
        if options["profile"] != "qa" or acknowledgement != "I_UNDERSTAND_TEST_ONLY":
            raise Refusal("negative-coarse requires the qa profile and test-only acknowledgement")
    elif acknowledgement is not None:
        raise Refusal("dense-FP8 test acknowledgement is valid only for negative-coarse")
    v16_expectation(values)
    if values["ABLIT"] not in ("0", "1"):
        raise Refusal("v1.6 requires explicit ABLIT=0 or ABLIT=1")
    if values["ABLIT"] == "1":
        if (values.get("ABLIT_METHOD") != "transplant" or values.get("ABLIT_LAYERS") != "15-45" or
                values.get("ABLIT_INCLUDE_MTP") != "1" or
                SHA_RE.fullmatch(values.get("JSPARK_ABLIT_MANIFEST_SHA256", "")) is None):
            raise Refusal("ablation method/range/manifest contract drift")
        donor = Path(values.get("JSPARK_ABLIT_ROOT", ""))
        if not donor.is_absolute() or ".." in donor.parts or len(donor.parts) < 3:
            raise Refusal("donor root must be an absolute normalized path")
    elif set(values) & set(OPTIONAL_ENV):
        raise Refusal("disabled control must not declare donor artifacts or ablation method")
    if values["JSPARK_GPU_MEMORY_UTILIZATION"] not in ("0.76", "0.78", "0.80", "0.83"):
        raise Refusal("unreviewed GPU memory utilization")
    gid = values["JSPARK_IB_GID_INDEX"]
    port = values["JSPARK_MASTER_PORT"]
    api_port = values["JSPARK_API_PORT"]
    if not gid.isdigit() or not 0 <= int(gid) <= 255:
        raise Refusal("invalid common GID index")
    if port != "29533" or api_port != "8888" or values["JSPARK_API_BIND"] != "0.0.0.0":
        raise Refusal("JSpark3 v1.6 requires master port 29533 and API bind 0.0.0.0:8888")
    paths = {
        key: Path(values[key])
        for key in ("JSPARK_MODEL_ROOT", "JSPARK_WORK_ROOT", "JSPARK_RECIPE_ROOT", "JSPARK_FLY_ROOT")
    }
    for key, path in paths.items():
        if not path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
            raise Refusal(f"{key} must be an absolute normalized path")
    if len(set(paths.values())) != len(paths):
        raise Refusal("model, work, recipe, and Fly roots must be distinct")
    work = paths["JSPARK_WORK_ROOT"]
    for key, path in paths.items():
        if key != "JSPARK_WORK_ROOT" and (work in path.parents or path in work.parents):
            raise Refusal("work root must not contain or be contained by an immutable root")
    joined = "\n".join(values.values())
    if ".example.invalid" in joined or "198.51.100." in joined or "192.0.2." in joined:
        raise Refusal("documentation placeholder remains in env")


def host(values: dict[str, str], rank: int) -> str:
    return values[f"JSPARK_RANK{rank}_HOST"]


def ssh_argv(values: dict[str, str], rank: int, remote_argv: list[str]) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
            host(values, rank), shlex.join(remote_argv)]


# Every remote Python whose stdout is consumed as JSON runs with -S (no site
# initialization).  The serving image installs two startup hooks into the
# container's dist-packages: glm53_video.pth (apply_image_glm_dflash) prints a
# banner BEFORE any script output, and zzz_b45.pth -> b45_bootstrap ->
# b5_prefix_verify prints calibration receipts AFTER it.  Both fire on every
# site-initializing interpreter, so a plain `python3 -c`/script stdout can
# never be a clean JSON channel (historical startup verify refusal).  All remote Python
# invoked here is stdlib-only (sibling modules resolve via sys.path[0], which
# -S preserves), so skipping site hooks provably changes nothing but the
# noise; the strict parsers stay byte-exact with no banner stripping.
JSON_PYTHON = ("python3", "-S")
REMOTE_STDERR_LIMIT = 16 * 1024  # Characters retained in failure receipts.


def remote(values: dict[str, str], rank: int, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(ssh_argv(values, rank, argv), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and process.returncode:
        detail = process.stderr.strip() or "no detail"
        if len(detail) > REMOTE_STDERR_LIMIT:
            detail = "[stderr truncated; tail follows]\n" + detail[-REMOTE_STDERR_LIMIT:]
        raise Refusal(f"rank{rank} remote command failed (exit {process.returncode}):\n{detail}")
    return process


def rank_env(
    values: dict[str, str],
    rank: int,
    preflight_sha256: str = "0" * 64,
    recipe_manifest_sha256: str | None = None,
) -> list[str]:
    socket = values[f"JSPARK_SOCKET_IFNAME_{rank}"]
    options = v16_options(values)
    fixed = {
        "NODE_RANK": str(rank),
        "ABLIT": values["ABLIT"], "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA": "0",
        # The upstream issue-249 controls stay off.  Only the sealed v1.6
        # switches below are admitted and they are identical on every rank.
        "GLM53_DENSE_FP8": "off",
        "JSPARK3_V16_PROFILE": options["profile"],
        "JSPARK3_V16_COOP": options["coop"],
        "GLM53_ADAPTIVE_K": options["adaptive-k"],
        "GLM53_ADAPTIVE_K_EPOCH_FILE": "/evidence/adaptive-k-epoch.json",
        "JSPARK3_V16_ADAPTIVE_K_EXPECT": options["adaptive-k"],
        "JSPARK3_V16_DENSE_FP8": options["dense-fp8"],
        # Compact DFlash windows at ordinary LRU priority in the prefix cache.
        "GLM53_APC_DRAFT_LRU": options["apc-lru"],
        "EXL3_FAT_GROUPED": "0", "GLM53_INDEXER_WORKSPACE": "stock",
        "NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": values[f"JSPARK_HCAS_{rank}"],
        "NCCL_IB_GID_INDEX": values["JSPARK_IB_GID_INDEX"],
        "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_CROSS_NIC": "0",
        "NCCL_IB_MERGE_NICS": "0", "NCCL_NVLS_ENABLE": "0",
        "NCCL_CUMEM_ENABLE": "0", "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_SOCKET_IFNAME": socket, "GLOO_SOCKET_IFNAME": socket,
        "TP_SOCKET_IFNAME": socket, "MN_IF_NAME": socket,
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "VLLM_HOST_IP": values[f"JSPARK_RANK{rank}_ADDR"],
        "HF_HOME": "/root/.cache/huggingface", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "12.1a", "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1", "EXL3_FUSED_MOE": "1",
        "GLM53_SUPPRESS_STOPS_IN_REASONING": "1",
        # v1.4: cadence-sched v6 cap mode, compact drafter KV, capacity log,
        # display-reserve KV by profile, and the sealed identity expectation.
        "GLM53_MIXED_PREFILL_CHUNK": "cap", "GLM53_MIXED_PREFILL_CAP": "2560",
        "GLM53_SCHED_EPOCH_FILE": "/evidence/sched-epoch.json",
        # Mia #238 launcher rules hold by construction: server_argv pins the
        # dflash speculator (compact pages are DFlash-only; the allocator also
        # refuses otherwise in-container) and never passes --kv-cache-memory-bytes.
        "GLM53_DRAFT_KV_COMPACT": "1", "GLM53_KV_CAPACITY_LOG": "1",
        "GLM53_DISPLAY_KV": "1" if values["JSPARK3_V14_PROFILE"] == "full" else "0",
        "JSPARK3_V14_PROFILE": values["JSPARK3_V14_PROFILE"],
        "JSPARK3_V14_EXPECT": v16_expectation(values),
        # v1.5: stage 8 EXL3 fat-expert path. The bitwise self-check against
        # the legacy loop is off at boot: armed from the environment it would
        # run inside vLLM's memory-profiling forward (an accumulator clone plus
        # the legacy loop), inflating the profiled peak and shrinking the KV
        # budget. It is armed after boot through the epoch file, which also
        # switches the mode at a step boundary.
        "JSPARK3_EXL3_FATPATH": "fused",
        "JSPARK3_EXL3_FATPATH_EPOCH_FILE": "/evidence/fatpath-epoch.json",
        "JSPARK3_EXL3_FATPATH_SELFCHECK": "0",
        "MODEL_PATH": TARGET_RUNTIME, "DRAFT_PATH": DRAFT_RUNTIME,
        "JSPARK_TARGET_RUNTIME": TARGET_RUNTIME,
        "JSPARK_DRAFT_RUNTIME": DRAFT_RUNTIME,
        "JSPARK3_KDA_MIXED_OUTPUT_BLOCKS": "8",
        "JSPARK3_KDA_FG_BATCHED": "1",
        "JSPARK3_TRUNK_W8A16": "1",
        "JSPARK3_TRUNK_W8A16_K704_GROUP": "64",
        **B45_ENV,
        "JSPARK_PREFLIGHT_SHA256": preflight_sha256,
        "JSPARK_RECIPE_MANIFEST_SHA256": recipe_manifest_sha256 or sha_file(RECIPE_ROOT / "SHA256SUMS"),
    }
    if values["JSPARK3_V14_PROFILE"] == "full":
        fixed.update({"GLM53_DISPLAY_KV_MIB": "1792", "GLM53_DISPLAY_KV_CREDIT_MIB": "960",
                      "GLM53_DRM_CARD": "/dev/dri/card0",
                      "GLM53_DISPLAY_KV_LIB": DISPLAY_KV_LIB})
    if values["ABLIT"] == "1":
        fixed.update({key: values[key] for key in OPTIONAL_ENV if key != "JSPARK_ABLIT_ROOT"})
    if options["dense-fp8"] == "negative-coarse":
        fixed["JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL"] = values[
            "JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL"
        ]
    return [f"{key}={value}" for key, value in fixed.items()]


def server_argv(values: dict[str, str], rank: int) -> list[str]:
    speculative = json.dumps({
        "method": "dflash", "model": DRAFT_RUNTIME, "num_speculative_tokens": 7,
        "kv_cache_dtype": "auto", "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard", "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
    }, separators=(",", ":"))
    compilation = json.dumps({
        "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_capture_sizes(values),
    }, separators=(",", ":"))
    overrides = json.dumps({
        "num_attention_heads": 66, "num_key_value_heads": 66, "linear_num_heads": 66,
        "text_config": {"num_attention_heads": 66, "num_key_value_heads": 66, "linear_num_heads": 66},
    }, separators=(",", ":"))
    argv = [
        "--served-model-name", "glm-5.3-flash", "--host", values["JSPARK_API_BIND"],
        "--port", values["JSPARK_API_PORT"], "--trust-remote-code", "--quantization", "exl3",
        "--load-format", "instanttensor",
        "--tensor-parallel-size", "3", "--pipeline-parallel-size", "1",
        "--gpu-memory-utilization", values["JSPARK_GPU_MEMORY_UTILIZATION"], "--max-model-len", "1000000",
        "--max-num-seqs", "32", "--max-num-batched-tokens", "8192",
        "--kv-cache-dtype", "fp8", "--enable-prefix-caching", "--no-enable-flashinfer-autotune",
        "--distributed-executor-backend", "mp", "--nnodes", "3", "--node-rank", str(rank),
        "--master-addr", values["JSPARK_MASTER_ADDR"], "--master-port", values["JSPARK_MASTER_PORT"],
        "--tool-call-parser", "glm47", "--enable-auto-tool-choice", "--reasoning-parser", "glm45",
        "--limit-mm-per-prompt", '{"image":8,"video":0}',
        "--mm-encoder-tp-mode", "data", "--mm-processor-cache-gb", "0", "--enable-expert-parallel",
        "--hf-overrides", overrides, "--speculative-config", speculative,
        "--compilation-config", compilation,
        "--default-chat-template-kwargs", "{\"enable_thinking\":false}",
        "--chat-template", "/sources/fly/files/chat_template.jinja",
        "--max-logprobs", "-1", "--enable-prompt-tokens-details",
        "--jit-monitor-verbose",
    ]
    if rank:
        argv.append("--headless")
    return argv


def container_argv(
    values: dict[str, str],
    rank: int,
    preflight_sha256: str = "0" * 64,
    recipe_manifest_sha256: str | None = None,
) -> list[str]:
    name = container_name(rank)
    work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
    argv = [
        "docker", "create", "--name", name, "--hostname", name, "--restart", "no",
        "--network", "host", "--ipc", "host", "--shm-size", str(SHM),
        "--memory", str(MEMORY), "--memory-swap", str(MEMORY),
        "--ulimit", "memlock=-1:-1", "--ulimit", "nofile=65536:65536", "--cap-add", "IPC_LOCK",
        "--security-opt", "label=disable", "--cgroupns", "private",
        "--device", "/dev/infiniband:/dev/infiniband:rwm", "--gpus", "all",
        "--mount", f"type=bind,src={values['JSPARK_RECIPE_ROOT']},dst=/recipe,readonly",
        "--mount", f"type=bind,src={values['JSPARK_FLY_ROOT']},dst=/sources/fly,readonly",
        "--mount", f"type=bind,src={values['JSPARK_MODEL_ROOT']},dst=/models,readonly",
        "--mount", f"type=bind,src={work}/evidence,dst=/evidence",
        "--mount", f"type=bind,src={work}/cache/vllm,dst=/root/.cache/vllm",
        "--mount", f"type=bind,src={work}/cache/triton,dst=/root/.triton/cache",
        "--mount", f"type=bind,src={work}/cache/tilelang,dst=/root/.tilelang/cache",
        "--label", "org.opencontainers.image.title=jspark3-recipe",
        "--label", f"jspark3.release={RELEASE_LABEL}", "--label", f"jspark3.rank={rank}",
        "--label", "jspark3.grade=engineering-evidence", "--label", "jspark3.b45=cadence-reference",
    ]
    if values["ABLIT"] == "1":
        argv.extend(("--mount", f"type=bind,src={values['JSPARK_ABLIT_ROOT']},dst=/ablit,readonly"))
    if values["JSPARK3_V14_PROFILE"] == "full":
        argv.extend(("--device", f"{values[f'JSPARK_DRM_CARD_{rank}']}:/dev/dri/card0"))
    for item in rank_env(values, rank, preflight_sha256, recipe_manifest_sha256):
        argv.extend(("--env", item))
    argv.extend(("--workdir", "/vllm-workspace", "--entrypoint", "/recipe/scripts/container_entry.sh", IMAGE))
    argv.extend(server_argv(values, rank))
    return argv


def preflight_argv(values: dict[str, str], rank: int) -> list[str]:
    peers = [values[f"JSPARK_RANK{other}_ADDR"] for other in range(3) if other != rank]
    options = v16_options(values)
    argv = [
        *JSON_PYTHON, f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
        "--rank", str(rank), "--ifaces", values[f"JSPARK_FABRIC_IFACES_{rank}"],
        "--cidrs", values[f"JSPARK_FABRIC_ADDRS_{rank}"], "--hcas", values[f"JSPARK_HCAS_{rank}"],
        "--gid-index", values["JSPARK_IB_GID_INDEX"],
        "--socket-iface", values[f"JSPARK_SOCKET_IFNAME_{rank}"],
        "--management-addr", values[f"JSPARK_RANK{rank}_ADDR"],
        "--peer-addrs", ",".join(peers), "--model-root", values["JSPARK_MODEL_ROOT"],
        "--work-root", values["JSPARK_WORK_ROOT"], "--recipe-root", values["JSPARK_RECIPE_ROOT"],
        "--fly-root", values["JSPARK_FLY_ROOT"],
        "--ablit", values["ABLIT"],
        "--display-profile", values["JSPARK3_V14_PROFILE"],
        "--v16-profile", options["profile"],
        "--v16-coop", options["coop"],
        "--v16-adaptive-k", options["adaptive-k"],
        "--v16-dense-fp8", options["dense-fp8"],
    ]
    if values["JSPARK3_V14_PROFILE"] == "full":
        argv.extend(("--drm-card", values[f"JSPARK_DRM_CARD_{rank}"]))
    if values["ABLIT"] == "1":
        argv.extend(("--ablit-root", values["JSPARK_ABLIT_ROOT"],
                     "--ablit-manifest-sha256", values["JSPARK_ABLIT_MANIFEST_SHA256"]))
    return argv


def checkpoint_argv(values: dict[str, str]) -> list[str]:
    root = values["JSPARK_MODEL_ROOT"]
    return [
        *JSON_PYTHON, "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/validate_checkpoint.py",
        "--target-root", f"{root}/{TARGET_NATIVE_NAME}",
        "--target-runtime", f"{root}/{TARGET_NATIVE_NAME}-tp3-runtime",
        "--draft-root", f"{root}/{DRAFT_NATIVE_NAME}",
        "--draft-runtime", f"{root}/{DRAFT_NATIVE_NAME}-tp3-runtime",
    ]


def render_dry_run(
    command: str, values: dict[str, str], *, remove: bool = False,
    preflight_sha: str = "",
) -> None:
    if command == "start":
        if preflight_sha and not SHA_RE.fullmatch(preflight_sha):
            raise Refusal("dry-run preflight SHA-256 is malformed")
        preflight_sha = preflight_sha or "0" * 64
        recipe_sha = recipe_manifest_sha256()
        recipe_check = [
            *JSON_PYTHON, "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
            "--recipe-only", "--recipe-root", values["JSPARK_RECIPE_ROOT"],
            "--expected-recipe-manifest-sha256", recipe_sha,
        ]
        for rank in (0, 1, 2):
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, recipe_check))}")
        for rank in (2, 1, 0):
            remote_command = ["docker", "container", "inspect", container_name(rank)]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")
        for rank in (2, 1, 0):
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, recipe_check))}")
            work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
            commands = [["mkdir", "-p", f"{work}/evidence", f"{work}/cache/vllm",
                         f"{work}/cache/triton", f"{work}/cache/tilelang"],
                        container_argv(values, rank, preflight_sha, recipe_sha)]
            for remote_command in commands:
                print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")
            identity = f"<ACTUAL_CONTAINER_ID_RANK{rank}>"
            inspect = ["docker", "container", "inspect", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, inspect))}")
            receipt = f"<LOCAL_TEMP_IMAGE_RECEIPT_RANK{rank}>"
            mint = [
                sys.executable, str(RECIPE_ROOT / "scripts/make_image_receipt.py"),
                "--output", receipt, "--container-id", identity, "--rank", str(rank),
                "--preflight-sha256", preflight_sha,
                "--recipe-manifest-sha256", recipe_sha,
            ]
            print(f"DRY-RUN controller {shlex.join(mint)}")
            receipt_path = f"{work}/evidence/image-receipt.json"
            install = image_receipt_install_argv(
                receipt_path, f"<HOST_MINTED_RECEIPT_BASE64_RANK{rank}>"
            )
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, install))}")
        for rank in (2, 1, 0):
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, recipe_check))}")
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, checkpoint_argv(values)))}")
            identity = f"<ACTUAL_CONTAINER_ID_RANK{rank}>"
            inspect = ["docker", "container", "inspect", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, inspect))}")
            start = ["docker", "start", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, start))}")
        return
    if command == "status":
        for rank in (0, 1, 2):
            identity = f"<MANIFEST_CONTAINER_ID_RANK{rank}>"
            inspect = ["docker", "container", "inspect", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, inspect))}")
            cgroup = cgroup_argv(f"<INSPECTED_HOST_PID_RANK{rank}>")
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, cgroup))}")
        print("DRY-RUN controller GET /health and GET /v1/models")
        return
    if command == "verify":
        for rank in (0, 1, 2):
            identity = f"<MANIFEST_CONTAINER_ID_RANK{rank}>"
            inspect = ["docker", "container", "inspect", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, inspect))}")
            for remote_command in runtime_identity_argv(identity):
                print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")
        for rank in (0, 1, 2):
            identity = f"<MANIFEST_CONTAINER_ID_RANK{rank}>"
            inspect = ["docker", "container", "inspect", identity]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, inspect))}")
            cgroup = cgroup_argv(f"<INSPECTED_HOST_PID_RANK{rank}>")
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, cgroup))}")
        for rank in (0, 1, 2):
            logs = ["docker", "logs", f"<MANIFEST_CONTAINER_ID_RANK{rank}>"]
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, logs))}")
        print("DRY-RUN controller GET /health and GET /v1/models")
        print("DRY-RUN controller POST arithmetic 323")
        base = f"http://{values['JSPARK_MASTER_ADDR']}:{values['JSPARK_API_PORT']}"
        print(f"DRY-RUN controller {shlex.join(focused_witness_argv(values, base))}")
        print(f"DRY-RUN controller {shlex.join(long_context_witness_argv(values, base))}")
        return
    if command == "stop":
        for rank in (0, 1, 2):
            identity = f"<MANIFEST_CONTAINER_ID_RANK{rank}>"
            for remote_command in (["docker", "container", "inspect", identity],
                                   ["docker", "stop", "--time", "30", identity]):
                print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")
        if remove:
            for rank in (0, 1, 2):
                identity = f"<MANIFEST_CONTAINER_ID_RANK{rank}>"
                for remote_command in (["docker", "container", "inspect", identity],
                                       ["docker", "rm", identity]):
                    print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")
        return
    for rank in (0, 1, 2):
        if command == "preflight":
            commands = [preflight_argv(values, rank)]
        else:
            commands = [["docker", "container", "inspect", container_name(rank)]]
        for remote_command in commands:
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, remote_command))}")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise Refusal(f"refusing symlink output: {path}")
    payload = canonical(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise Refusal(f"refusing symlink output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def configuration_digest(values: dict[str, str]) -> str:
    keys = (*REQUIRED_ENV, *(key for key in DRM_CARD_ENV if key in values))
    bound = {key: values[key] for key in keys}
    bound["v16_options"] = v16_options(values)
    # S9.7's flag joins the identity only when on, so pre-S9.7 manifests and
    # APC_LRU=0 rollbacks keep their configuration digest.
    if bound["v16_options"]["apc-lru"] == "0":
        del bound["v16_options"]["apc-lru"]
    acknowledgement = values.get("JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL")
    if acknowledgement is not None:
        bound["v16_negative_control_ack"] = acknowledgement
    return sha_bytes(canonical(bound))


def strict_loads(text: str) -> object:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Refusal(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value):
        raise Refusal(f"non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def strict_object(text: str, label: str) -> dict:
    value = strict_loads(text)
    if not isinstance(value, dict):
        raise Refusal(f"{label} is not a JSON object")
    return value


def inspect_document(text: str, label: str) -> dict:
    value = strict_loads(text)
    if (not isinstance(value, list) or len(value) != 1 or
            not isinstance(value[0], dict)):
        raise Refusal(f"{label} inspect response schema drift")
    return value[0]


def model_identities(value: dict) -> list[str]:
    rows = value.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise Refusal("served-model response schema drift")
    identities = [row.get("id") for row in rows]
    if any(not isinstance(identity, str) for identity in identities):
        raise Refusal("served-model identity schema drift")
    return identities


def completion_content(value: dict) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise Refusal("completion response choices schema drift")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise Refusal("completion response message schema drift")
    return message["content"]


def recipe_manifest_sha256() -> str:
    manifest = RECIPE_ROOT / "SHA256SUMS"
    if not manifest.is_file() or manifest.is_symlink():
        raise Refusal("local recipe manifest missing or unsafe")
    return sha_file(manifest)


def expected_preflight_row(values: dict[str, str], rank: int, recipe_sha: str) -> dict:
    return {
        "schema_version": 1, "rank": rank, "system": "NVIDIA DGX Spark",
        "architecture": "aarch64",
        "gpu": "NVIDIA GB10 / SM121", "image": "PASS",
        "image_manifest": IMAGE_MANIFEST, "image_config": IMAGE_CONFIG,
        "fabric_legs": 2, "gid_index": int(values["JSPARK_IB_GID_INDEX"]),
        "gid_entries": ["PASS", "PASS"], "management_routes": "PASS",
        "checkpoint": "PASS", "recipe_manifest": "PASS",
        "recipe_manifest_sha256": recipe_sha, "free_memory": "PASS_GE_72_GIB",
        "free_disk": "PASS_GE_8_GIB_MODEL_AND_WORK", "release_name_absent": True,
        "ablit": int(values["ABLIT"]),
        "ablit_manifest_sha256": values.get("JSPARK_ABLIT_MANIFEST_SHA256"),
        "display": expected_display_row(values, rank),
        "v16": expected_v16_row(values),
        "status": "PASS",
    }


def expected_v16_row(values: dict[str, str]) -> dict:
    import remote_preflight

    options = v16_options(values)
    return remote_preflight.v16_artifacts(
        RECIPE_ROOT, options["profile"], options["coop"],
        options["adaptive-k"], options["dense-fp8"],
    )


def expected_display_row(values: dict[str, str], rank: int) -> dict:
    if values["JSPARK3_V14_PROFILE"] != "full":
        return {"profile": "display0"}
    return {"profile": "full", "card": values[f"JSPARK_DRM_CARD_{rank}"], "card_driver": "nvidia",
            "modeset": "Y", "fbdev": "N", "default_target": "multi-user.target",
            "display_servers": [], "display_units_active": []}


def validate_preflight_receipt(
    receipt: dict, values: dict[str, str], preflight_sha: str
) -> tuple[list[dict], str]:
    expected_keys = {
        "schema_version", "candidate", "grade", "configuration_sha256",
        "image_manifest", "image_config", "recipe_manifest_sha256",
        "ranks", "status", "payload_sha256",
    }
    rows = receipt.get("ranks")
    recipe_sha = recipe_manifest_sha256()
    if (set(receipt) != expected_keys or receipt.get("schema_version") != 1 or
            receipt.get("candidate") != "jspark3" or
            receipt.get("grade") != "ENGINEERING-EVIDENCE" or
            receipt.get("status") != "PREFLIGHT_PASS" or
            receipt.get("configuration_sha256") != configuration_digest(values) or
            receipt.get("image_manifest") != IMAGE_MANIFEST or
            receipt.get("image_config") != IMAGE_CONFIG or
            receipt.get("recipe_manifest_sha256") != recipe_sha or
            not isinstance(rows, list) or len(rows) != 3 or
            rows != [expected_preflight_row(values, rank, recipe_sha) for rank in range(3)] or
            not SHA_RE.fullmatch(preflight_sha)):
        raise Refusal("preflight receipt does not bind this exact configuration")
    return rows, recipe_sha


def remote_recipe_sha(values: dict[str, str], rank: int, expected: str | None = None) -> str:
    argv = [
        *JSON_PYTHON, "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
        "--recipe-only", "--recipe-root", values["JSPARK_RECIPE_ROOT"],
    ]
    if expected is not None:
        argv.extend(("--expected-recipe-manifest-sha256", expected))
    process = remote(values, rank, argv)
    value = strict_loads(process.stdout)
    if (not isinstance(value, dict) or set(value) != {"recipe_manifest_sha256", "status"} or
            value.get("status") != "PASS" or not SHA_RE.fullmatch(str(value.get("recipe_manifest_sha256", "")))):
        raise Refusal(f"rank{rank} recipe-only verification schema drift")
    return str(value["recipe_manifest_sha256"])


def verify_remote_recipe(values: dict[str, str], rank: int, expected: str) -> None:
    if remote_recipe_sha(values, rank, expected) != expected:
        raise Refusal(f"rank{rank} recipe changed after preflight")


def verify_remote_checkpoint(values: dict[str, str], rank: int) -> None:
    value = strict_loads(remote(values, rank, checkpoint_argv(values)).stdout)
    if (not isinstance(value, dict) or value.get("schema_version") != 1 or
            value.get("grade") != "ENGINEERING-EVIDENCE" or
            value.get("serving_checkpoint_pass") is not True or
            value.get("publication_ledger_complete") is not False or
            (value.get("target") or {}).get("shards") != 120 or
            (value.get("draft") or {}).get("shards") != 1):
        raise Refusal(f"rank{rank} serving-byte revalidation schema drift")


def mint_image_receipt(
    *, container_id: str, rank: int, preflight_sha: str, recipe_sha: str
) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "image-receipt.json"
        process = subprocess.run([
            sys.executable, str(RECIPE_ROOT / "scripts/make_image_receipt.py"),
            "--output", str(output), "--container-id", container_id,
            "--rank", str(rank), "--preflight-sha256", preflight_sha,
            "--recipe-manifest-sha256", recipe_sha,
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode:
            raise Refusal(f"rank{rank} image-receipt mint failed: {process.stderr.strip()}")
        return output.read_bytes()


def image_receipt_install_argv(path: str, encoded: str) -> list[str]:
    code = (
        "import base64,os,sys; p=sys.argv[1]; d=base64.b64decode(sys.argv[2],validate=True); "
        "f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); "
        "h=os.fdopen(f,'wb'); h.write(d)==len(d) or (_ for _ in ()).throw(OSError('short write')); "
        "h.flush(); os.fsync(h.fileno()); h.close(); q=os.open(os.path.dirname(p),os.O_RDONLY|os.O_DIRECTORY); "
        "os.fsync(q); os.close(q)"
    )
    return [*JSON_PYTHON, "-c", code, path, encoded]


def install_image_receipt(values: dict[str, str], rank: int, data: bytes) -> None:
    path = f"{values['JSPARK_WORK_ROOT']}/rank{rank}/evidence/image-receipt.json"
    remote(values, rank, image_receipt_install_argv(path, base64.b64encode(data).decode("ascii")))


def cmd_preflight(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("preflight", values)
        return
    validate_env(values)
    rows = []
    for rank in range(3):
        process = remote(values, rank, preflight_argv(values, rank))
        try:
            row = strict_loads(process.stdout)
        except (json.JSONDecodeError, Refusal) as exc:
            raise Refusal(f"rank{rank} invalid preflight response") from exc
        recipe_sha = recipe_manifest_sha256()
        if row != expected_preflight_row(values, rank, recipe_sha):
            raise Refusal(f"rank{rank} preflight did not pass")
        rows.append(row)
    receipt = {
        "schema_version": 1, "candidate": "jspark3",
        "grade": "ENGINEERING-EVIDENCE", "configuration_sha256": configuration_digest(values),
        "image_manifest": IMAGE_MANIFEST, "image_config": IMAGE_CONFIG,
        "recipe_manifest_sha256": recipe_manifest_sha256(), "ranks": rows,
        "status": "PREFLIGHT_PASS",
    }
    receipt["payload_sha256"] = sha_bytes(canonical(receipt))
    atomic_json(args.output, receipt)
    print(f"PASS preflight receipt={args.output} sha256={sha_file(args.output)}")


def read_receipt(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Refusal("receipt missing or symlinked")
    try:
        value = strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, Refusal) as exc:
        raise Refusal(f"invalid JSON receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal("receipt must be an object")
    claimed = value.get("payload_sha256")
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    if claimed != sha_bytes(canonical(payload)):
        raise Refusal("receipt payload hash mismatch")
    return value


def validate_container_contract(
    values: dict[str, str], rank: int, identity: str, item: dict,
    preflight_sha: str, recipe_sha: str,
) -> None:
    name = container_name(rank)
    config = item.get("Config") or {}
    host_config = item.get("HostConfig") or {}
    validate_bound_identity(rank, identity, item)
    if (config.get("Hostname") != name or config.get("WorkingDir") != "/vllm-workspace" or
            config.get("Entrypoint") != ["/recipe/scripts/container_entry.sh"] or
            config.get("Cmd") != server_argv(values, rank)):
        raise Refusal(f"rank{rank} container identity/argv contract drift")

    environment = config.get("Env") or []
    parsed_env = {}
    for entry in environment:
        key, marker, value = str(entry).partition("=")
        if not marker or key in parsed_env:
            raise Refusal(f"rank{rank} container environment schema drift")
        parsed_env[key] = value
    expected_env = dict(item.split("=", 1) for item in
                        rank_env(values, rank, preflight_sha, recipe_sha))
    if any(parsed_env.get(key) != value for key, value in expected_env.items()):
        raise Refusal(f"rank{rank} required environment drift")
    if any(key in parsed_env for key in FORBIDDEN_NCCL):
        raise Refusal(f"rank{rank} forbidden environment present")

    labels = config.get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.title": "jspark3-recipe",
        "jspark3.release": RELEASE_LABEL, "jspark3.rank": str(rank),
        "jspark3.grade": "engineering-evidence", "jspark3.b45": "cadence-reference",
    }
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise Refusal(f"rank{rank} release label drift")

    if (host_config.get("NetworkMode") != "host" or host_config.get("IpcMode") != "host" or
            host_config.get("ShmSize") != SHM or host_config.get("Memory") != MEMORY or
            host_config.get("MemorySwap") != MEMORY or
            (host_config.get("RestartPolicy") or {}).get("Name") != "no" or
            host_config.get("CgroupnsMode") != "private"):
        raise Refusal(f"rank{rank} namespace/resource contract drift")

    work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
    expected_mounts = {
        "/recipe": (values["JSPARK_RECIPE_ROOT"], False),
        "/sources/fly": (values["JSPARK_FLY_ROOT"], False),
        "/models": (values["JSPARK_MODEL_ROOT"], False),
        "/evidence": (f"{work}/evidence", True),
        "/root/.cache/vllm": (f"{work}/cache/vllm", True),
        "/root/.triton/cache": (f"{work}/cache/triton", True),
        "/root/.tilelang/cache": (f"{work}/cache/tilelang", True),
    }
    if values["ABLIT"] == "1":
        expected_mounts["/ablit"] = (values["JSPARK_ABLIT_ROOT"], False)
    mounts = item.get("Mounts") or []
    observed_mounts = {}
    for mount in mounts:
        destination = mount.get("Destination")
        if destination in observed_mounts:
            raise Refusal(f"rank{rank} duplicate mount destination")
        observed_mounts[destination] = (mount.get("Source"), mount.get("RW"), mount.get("Type"))
    if observed_mounts != {dst: (source, writable, "bind")
                           for dst, (source, writable) in expected_mounts.items()}:
        raise Refusal(f"rank{rank} mount contract drift")

    devices = host_config.get("Devices") or []
    expected_devices = [{"PathOnHost": "/dev/infiniband", "PathInContainer": "/dev/infiniband",
                         "CgroupPermissions": "rwm"}]
    if values["JSPARK3_V14_PROFILE"] == "full":
        expected_devices.append({"PathOnHost": values[f"JSPARK_DRM_CARD_{rank}"],
                                 "PathInContainer": "/dev/dri/card0", "CgroupPermissions": "rwm"})
    if devices != expected_devices:
        raise Refusal(f"rank{rank} InfiniBand/DRM device contract drift")
    requests = host_config.get("DeviceRequests") or []
    if requests != [{"Driver": "", "Count": -1, "DeviceIDs": None,
                     "Capabilities": [["gpu"]], "Options": {}}]:
        raise Refusal(f"rank{rank} GPU request contract drift")
    ulimits = host_config.get("Ulimits") or []
    if sorted(ulimits, key=lambda row: row['Name']) != [
            {"Name": "memlock", "Soft": -1, "Hard": -1},
            {"Name": "nofile", "Soft": 65536, "Hard": 65536}]:
        raise Refusal(f"rank{rank} memlock/nofile contract drift")
    if (host_config.get("CapAdd") != ["CAP_IPC_LOCK"] or
            host_config.get("SecurityOpt") != ["label=disable"]):
        raise Refusal(f"rank{rank} capability/security contract drift")


def validate_admission_state(rank: int, item: dict) -> None:
    state = item.get("State") or {}
    if state.get("OOMKilled") is not False or item.get("RestartCount") != 0:
        raise Refusal(f"rank{rank} container admission state drift")


def validate_bound_identity(rank: int, identity: str, item: dict) -> None:
    name = container_name(rank)
    config = item.get("Config") or {}
    labels = config.get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.title": "jspark3-recipe",
        "jspark3.release": RELEASE_LABEL, "jspark3.rank": str(rank),
        "jspark3.grade": "engineering-evidence", "jspark3.b45": "cadence-reference",
    }
    if (item.get("Id") != identity or item.get("Name") != f"/{name}" or
            item.get("Image") != IMAGE_CONFIG or config.get("Image") != IMAGE or
            any(labels.get(key) != value for key, value in expected_labels.items())):
        raise Refusal(f"rank{rank} manifest-bound container identity drift")


def stop_created(values: dict[str, str], created: list[dict]) -> bool:
    confirmed = True
    by_rank = {row["rank"]: row for row in created}
    for rank in (0, 1, 2):
        row = by_rank.get(rank)
        if row is None:
            continue
        identity = row["container_id"]
        probe = remote(values, rank, ["docker", "container", "inspect", identity], check=False)
        if probe.returncode:
            confirmed = False
            continue
        try:
            inspected = inspect_document(probe.stdout, f"rank{rank} cleanup")
        except (TypeError, json.JSONDecodeError, Refusal):
            confirmed = False
            continue
        if (inspected.get("State") or {}).get("Running"):
            remote(values, rank, ["docker", "stop", "--time", "30", identity], check=False)
        final = remote(values, rank, ["docker", "container", "inspect", identity], check=False)
        try:
            final_item = inspect_document(final.stdout, f"rank{rank} cleanup-final")
            stopped = final.returncode == 0 and not (final_item.get("State") or {}).get("Running")
        except (TypeError, json.JSONDecodeError, Refusal):
            stopped = False
        confirmed = confirmed and stopped
    return confirmed


def cmd_start(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("start", values, preflight_sha=args.preflight_sha256)
        return
    validate_env(values)
    if args.confirm != "START-JSPARK3":
        raise Refusal("start confirmation must be START-JSPARK3")
    if not SHA_RE.fullmatch(args.preflight_sha256 or "") or sha_file(args.preflight) != args.preflight_sha256:
        raise Refusal("preflight file SHA-256 mismatch")
    receipt = read_receipt(args.preflight)
    _, recipe_sha = validate_preflight_receipt(receipt, values, args.preflight_sha256)
    if args.manifest.exists():
        raise Refusal("release manifest already exists")
    created = []
    base_manifest = {
        "schema_version": 1, "candidate": "jspark3",
        "grade": "ENGINEERING-EVIDENCE", "configuration_sha256": configuration_digest(values),
        "preflight_sha256": args.preflight_sha256, "image_manifest": IMAGE_MANIFEST,
        "image_config": IMAGE_CONFIG, "recipe_manifest_sha256": recipe_sha,
        "start_order": [2, 1, 0],
    }

    def save(status: str) -> None:
        manifest = dict(base_manifest)
        manifest.update({"containers": sorted(created, key=lambda row: row["rank"]), "status": status})
        manifest["payload_sha256"] = sha_bytes(canonical(manifest))
        atomic_json(args.manifest, manifest)

    failure_status = "CREATE_FAILED_STATE_UNCONFIRMED"
    try:
        for rank in range(3):
            verify_remote_recipe(values, rank, recipe_sha)
        for rank in (2, 1, 0):
            if remote(values, rank, ["docker", "container", "inspect", container_name(rank)], check=False).returncode == 0:
                raise Refusal(f"rank{rank} deterministic release name already exists")
        for rank in (2, 1, 0):
            verify_remote_recipe(values, rank, recipe_sha)
            work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
            remote(values, rank, ["mkdir", "-p", f"{work}/evidence", f"{work}/cache/vllm",
                                  f"{work}/cache/triton", f"{work}/cache/tilelang"])
            candidate = remote(values, rank, container_argv(
                values, rank, args.preflight_sha256, recipe_sha)).stdout.strip()
            if not SHA_RE.fullmatch(candidate):
                probe = remote(values, rank, ["docker", "container", "inspect", container_name(rank)], check=False)
                if probe.returncode == 0:
                    recovered = inspect_document(
                        probe.stdout, f"rank{rank} create-recovery"
                    ).get("Id", "")
                    if SHA_RE.fullmatch(recovered):
                        created.append({"rank": rank, "container_id": recovered,
                                        "name": container_name(rank), "image_config": IMAGE_CONFIG,
                                        "image_receipt_sha256": None})
                raise Refusal(f"rank{rank} docker create returned an unsafe identity")
            binding = {"rank": rank, "container_id": candidate,
                       "name": container_name(rank), "image_config": IMAGE_CONFIG,
                       "image_receipt_sha256": None}
            created.append(binding)
            inspected = inspect_document(remote(
                values, rank, ["docker", "container", "inspect", candidate]).stdout,
                f"rank{rank} post-create")
            validate_container_contract(values, rank, candidate, inspected,
                                        args.preflight_sha256, recipe_sha)
            validate_admission_state(rank, inspected)
            if (inspected.get("State") or {}).get("Running"):
                raise Refusal(f"rank{rank} created container started unexpectedly")
            image_receipt = mint_image_receipt(
                container_id=candidate, rank=rank,
                preflight_sha=args.preflight_sha256, recipe_sha=recipe_sha)
            install_image_receipt(values, rank, image_receipt)
            binding["image_receipt_sha256"] = sha_bytes(image_receipt)
        save("CREATED")
        failure_status = "START_FAILED_STATE_UNCONFIRMED"
        for rank in (2, 1, 0):
            verify_remote_recipe(values, rank, recipe_sha)
            verify_remote_checkpoint(values, rank)
            binding = next(row for row in created if row["rank"] == rank)
            inspected = inspect_document(remote(
                values, rank, ["docker", "container", "inspect", binding["container_id"]]).stdout,
                f"rank{rank} pre-start")
            validate_container_contract(values, rank, binding["container_id"], inspected,
                                        args.preflight_sha256, recipe_sha)
            validate_admission_state(rank, inspected)
            if (inspected.get("State") or {}).get("Running"):
                raise Refusal(f"rank{rank} container running before ordered start")
            remote(values, rank, ["docker", "start", binding["container_id"]])
        save("STARTED")
        print(f"PASS started ranks=2,1,0 manifest={args.manifest} sha256={sha_file(args.manifest)}")
    except BaseException:
        if created:
            stopped = stop_created(values, created)
            status = failure_status.replace("STATE_UNCONFIRMED", "STOPPED_PRESERVED") if stopped else failure_status
            try:
                save(status)
            except BaseException as save_exc:
                print(f"START REFUSED; local failure manifest write also failed: {save_exc}", file=sys.stderr)
            if stopped:
                print("START REFUSED; all manifest-bound created containers are inspect-confirmed stopped/preserved", file=sys.stderr)
            else:
                print("START REFUSED; at least one created container state is unconfirmed; inspect exact manifest IDs", file=sys.stderr)
        else:
            print("START REFUSED before any container identity was created", file=sys.stderr)
        raise


def bound_manifest(
    path: Path, values: dict[str, str], *, require_all: bool = False,
    require_started: bool = False, candidate_recipe_sha256: str | None = None,
) -> dict:
    value = read_receipt(path)
    if candidate_recipe_sha256 is not None and not SHA_RE.fullmatch(candidate_recipe_sha256):
        raise Refusal("candidate recipe manifest SHA-256 is malformed")
    # Zero-boot verification: a patched verifier recipe must be able to verify
    # a fleet started from an earlier recipe.  The candidate identity then
    # comes from the operator-declared (SHA-256) candidate recipe manifest and
    # is re-proven on every rank (verify_remote_recipe), while the verifier's
    # own recipe identity is reported separately in the receipt.  Without the
    # flag the historical self-identity check is unchanged.
    expected_recipe_sha = candidate_recipe_sha256 or recipe_manifest_sha256()
    expected_keys = {
        "schema_version", "candidate", "grade", "configuration_sha256",
        "preflight_sha256", "recipe_manifest_sha256", "image_manifest",
        "image_config", "start_order", "containers", "status", "payload_sha256",
    }
    allowed_status = {
        "CREATED", "STARTED", "CREATE_FAILED_STOPPED_PRESERVED",
        "CREATE_FAILED_STATE_UNCONFIRMED", "START_FAILED_STOPPED_PRESERVED",
        "START_FAILED_STATE_UNCONFIRMED",
    }
    if (set(value) != expected_keys or value.get("schema_version") != 1 or
            value.get("candidate") != "jspark3" or
            value.get("grade") != "ENGINEERING-EVIDENCE" or
            value.get("configuration_sha256") != configuration_digest(values) or
            value.get("image_manifest") != IMAGE_MANIFEST or
            value.get("image_config") != IMAGE_CONFIG or
            value.get("recipe_manifest_sha256") != expected_recipe_sha or
            not SHA_RE.fullmatch(str(value.get("preflight_sha256", ""))) or
            value.get("start_order") != [2, 1, 0] or value.get("status") not in allowed_status):
        raise Refusal("manifest does not bind this environment/image")
    if require_started and value.get("status") != "STARTED":
        raise Refusal("verification requires a STARTED manifest")
    rows = value.get("containers")
    ranks = [row.get("rank") for row in rows] if isinstance(rows, list) else []
    if (not isinstance(rows, list) or not rows or ranks != sorted(set(ranks)) or
            any(rank not in (0, 1, 2) for rank in ranks) or (require_all and ranks != [0, 1, 2])):
        raise Refusal("manifest rank set drift")
    for row in rows:
        if (set(row) != {"rank", "container_id", "name", "image_config", "image_receipt_sha256"} or
                not SHA_RE.fullmatch(str(row.get("container_id", ""))) or
                row.get("name") != container_name(row['rank']) or
                row.get("image_config") != IMAGE_CONFIG or
                (row.get("image_receipt_sha256") is not None and
                 not SHA_RE.fullmatch(str(row.get("image_receipt_sha256"))))):
            raise Refusal("manifest container binding drift")
        if value.get("status") in {"CREATED", "STARTED"} and row.get("image_receipt_sha256") is None:
            raise Refusal("successful manifest lacks image receipt binding")
    return value


def inspect_bound(values: dict[str, str], row: dict, manifest: dict) -> dict:
    inspected = inspect_identity(values, row)
    validate_container_contract(values, row["rank"], row["container_id"], inspected,
                                manifest["preflight_sha256"], manifest["recipe_manifest_sha256"])
    return inspected


def inspect_identity(values: dict[str, str], row: dict) -> dict:
    rank, identity = row["rank"], row["container_id"]
    process = remote(values, rank, ["docker", "container", "inspect", identity], check=False)
    if process.returncode:
        raise Refusal(f"rank{rank} manifest-bound container missing")
    inspected = inspect_document(process.stdout, f"rank{rank} bound")
    validate_bound_identity(rank, identity, inspected)
    return inspected


def cgroup_argv(pid: int | str) -> list[str]:
    code = """import json,pathlib,sys
pid=int(sys.argv[1])
rows=(pathlib.Path('/proc')/str(pid)/'cgroup').read_text().splitlines()
sep=chr(58)*2
matches=[line.split(sep,1)[1] for line in rows if line.startswith('0'+sep)]
if len(matches)!=1:
    raise SystemExit(9)
root=pathlib.Path('/sys/fs/cgroup').resolve()
p=(root/matches[0].lstrip('/')).resolve(strict=True)
if p!=root and root not in p.parents:
    raise SystemExit(9)
events=dict(line.split() for line in (p/'memory.events').read_text().splitlines())
print(json.dumps({'memory_max':(p/'memory.max').read_text().strip(),'swap_max':(p/'memory.swap.max').read_text().strip(),'swap_current':int((p/'memory.swap.current').read_text()),'events':events},sort_keys=True))
"""
    return [*JSON_PYTHON, "-c", code, str(pid)]


def cgroup_state(values: dict[str, str], rank: int, pid: int) -> dict:
    output = remote(values, rank, cgroup_argv(pid)).stdout
    value = strict_loads(output)
    if not isinstance(value, dict) or set(value) != {"memory_max", "swap_max", "swap_current", "events"}:
        raise Refusal(f"rank{rank} cgroup status schema drift")
    return value


def collect_status(values: dict[str, str], manifest: dict) -> list[dict]:
    rows = []
    for binding in manifest["containers"]:
        rank = binding["rank"]
        item = inspect_identity(values, binding)
        state = item.get("State") or {}
        hc = item.get("HostConfig") or {}
        row = {
            "rank": rank, "name": binding["name"], "running": state.get("Running") is True,
            "oom_killed": state.get("OOMKilled") is True, "restart_count": item.get("RestartCount"),
            "exit_code": state.get("ExitCode"), "actual_image_config": item.get("Image"),
            "image_config_match": item.get("Image") == IMAGE_CONFIG,
            "image_reference_match": (item.get("Config") or {}).get("Image") == IMAGE,
            "memory_bytes": hc.get("Memory"), "memory_swap_bytes": hc.get("MemorySwap"),
            "restart_policy": (hc.get("RestartPolicy") or {}).get("Name"),
        }
        if row["running"]:
            pid = state.get("Pid")
            if type(pid) is not int or pid <= 0:
                raise Refusal(f"rank{rank} running container has no host PID")
            row["cgroup"] = cgroup_state(values, rank, pid)
        rows.append(row)
    return rows


def b45_identity_argv(identity: str) -> list[str]:
    # Inventory hashes the pinned source modules only. A real generated
    # __pycache__ directory is skipped; any other non-file, symlink, or
    # extra name refuses. Host still compares the modules dict to B45_MODULES.
    pinned = "{" + ", ".join(repr(name) for name in sorted(B45_MODULES)) + "}"
    code = """import hashlib,json,pathlib,sys
h=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
b45=pathlib.Path('/opt/b45')
site=pathlib.Path('/usr/local/lib/python3.12/dist-packages')
pinned=PINNED
mods={}
for p in (b45.iterdir() if b45.is_dir() else []):
    if p.name=='__pycache__' and not p.is_symlink() and p.is_dir():
        continue
    if p.is_symlink() or not p.is_file() or p.name not in pinned:
        sys.stderr.write('unexpected /opt/b45 entry: '+p.name+'\\n')
        raise SystemExit(9)
    mods[p.name]=h(p)
out=pathlib.Path('/tmp/b45')
out_entries=sorted(p.relative_to(out).as_posix() for p in out.rglob('*')) if out.is_dir() else []
graphs=out/'graphs'
rows=[json.loads(line) for p in sorted(graphs.glob('captures-*.jsonl')) for line in p.read_text().splitlines() if line.strip()] if graphs.is_dir() else []
dots_intact=all(isinstance(r.get('dot'),str) and (graphs/r['dot']).is_file() and h(graphs/r['dot'])==r.get('dot_sha256') for r in rows)
serving=sum(1 for p in graphs.glob('graph-serving-*.dot') if p.is_file() and p.stat().st_size>0) if graphs.is_dir() else 0
print(json.dumps({'modules':mods,'pth_sha256':h(site/'zzz_b45.pth'),'kda_original_sha256':h(site/'vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py'),'b45_out_entries':out_entries,'capture_receipts':len(rows),'capture_dots_intact':bool(dots_intact),'serving_graph_dumps':serving},sort_keys=True))
""".replace("PINNED", pinned)
    return ["docker", "exec", identity, *JSON_PYTHON, "-c", code]


def b45_identity(values: dict[str, str], binding: dict) -> dict:
    rank = binding["rank"]
    identity = binding["container_id"]
    value = strict_object(remote(values, rank, b45_identity_argv(identity)).stdout,
                          f"rank{rank} cadence module identity")
    if (set(value) != CADENCE_IDENTITY_KEYS or
            value["modules"] != B45_MODULES or value["pth_sha256"] != B45_PTH_SHA256 or
            value["kda_original_sha256"] != B45_KDA_ORIGINAL_SHA256):
        raise Refusal(f"rank{rank} cadence module install drift")
    receipts = value["b45_out_entries"]
    # The pinned B4 writer emits under B5_OUT/graphs (write() -> OUT = /tmp/b45/graphs),
    # so entries are 'graphs/activation-<pid>.jsonl'; an unqualified prefix can never match.
    if not receipts or not any(name.startswith("graphs/activation-") for name in receipts):
        raise Refusal(f"rank{rank} cadence execution receipts absent under B5_OUT: "
                      "module file presence alone is not execution proof")
    if not cadence_capture_evidence_ok(value):
        raise Refusal(f"rank{rank} cadence capture evidence missing or drifted: the byte-pinned "
                      "B4 capture must leave hash-matching capture receipts for the full bank "
                      f"set (>= {CADENCE_CAPTURE_RECEIPTS_MIN}) and >= {CADENCE_SERVING_DUMPS_MIN} "
                      "non-empty serving graph dumps")
    return {"modules_verified": len(value["modules"]),
            "kda_original_untouched": True,
            "execution_receipts": len(receipts),
            "capture_receipts": value["capture_receipts"],
            "capture_dots_intact": value["capture_dots_intact"],
            "serving_graph_dumps": value["serving_graph_dumps"]}


def runtime_identity(values: dict[str, str], binding: dict, manifest: dict) -> dict:
    rank, identity = binding["rank"], binding["container_id"]
    config_argv, pipeline_argv = runtime_identity_argv(identity)
    configs = strict_object(remote(values, rank, config_argv).stdout,
                            f"rank{rank} runtime identity")
    if set(configs) != {
        "target_runtime_config", "draft_runtime_config",
        "image_receipt_sha256", "image_receipt",
    }:
        raise Refusal(f"rank{rank} runtime identity schema drift")
    image_receipt = configs.pop("image_receipt")
    claimed = image_receipt.get("payload_sha256") if isinstance(image_receipt, dict) else None
    payload = ({key: item for key, item in image_receipt.items() if key != "payload_sha256"}
               if isinstance(image_receipt, dict) else {})
    if (not isinstance(image_receipt, dict) or
            set(image_receipt) != {"schema_version", "manifest_digest", "config_digest", "verification",
                                   "container_id", "rank", "preflight_sha256",
                                   "recipe_manifest_sha256", "payload_sha256"} or
            claimed != sha_bytes(canonical(payload)) or image_receipt.get("schema_version") != 2 or
            image_receipt.get("manifest_digest") != IMAGE_MANIFEST or
            image_receipt.get("config_digest") != IMAGE_CONFIG or
            image_receipt.get("verification") != "host-observed-inspect-bound-create" or
            image_receipt.get("container_id") != identity or image_receipt.get("rank") != rank or
            image_receipt.get("preflight_sha256") != manifest["preflight_sha256"] or
            image_receipt.get("recipe_manifest_sha256") != manifest["recipe_manifest_sha256"] or
            configs.get("image_receipt_sha256") != binding["image_receipt_sha256"]):
        raise Refusal(f"rank{rank} host-minted image receipt drift")
    pipeline = remote(values, rank, pipeline_argv)
    pipeline_value = strict_object(pipeline.stdout, f"rank{rank} transform pipeline")
    stock = strict_object(remote(values, rank, ["docker", "exec", identity, *JSON_PYTHON,
                           "/recipe/scripts/verify_stock.py"]).stdout, f"rank{rank} stock identity")
    if stock.get("status") != "PASS" or stock.get("rank") != rank or stock.get("ablit") != int(values["ABLIT"]):
        raise Refusal(f"rank{rank} stock construction drift")
    if v16_options(values)["profile"] == production_stock.PROFILE:
        try:
            production_stock.environment(values)
            production_stock.stock_identity(stock, rank)
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
    cadence = b45_identity(values, binding)
    result = {**configs, "transform_pipeline_state": pipeline_value.get("state"),
              "transform_target_set_sha256": pipeline_value.get("target_set_sha256"),
              "image_receipt_bound": True, "cadence_b45": cadence, "stock_v13": stock}
    expected = {
        "target_runtime_config": "55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd",
        "draft_runtime_config": "c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f",
        "transform_pipeline_state": "ALREADY_APPLIED",
        # SHA-256 of canonical final targets in patch-contract.json: the five
        # frozen v1.3.0 stages (1f3beb88...) plus v1.4 stages 6 (core) and 7
        # (display KV) and v1.5 stage 8 (EXL3 fat path); regenerate with
        # tools/v15/build_v15_contracts.py.
        "transform_target_set_sha256": "e675e27d5d28fa8d864b689a8b2faa04bf728b5ab6ef99ee4034f379ef6821ec",
    }
    drift = [f"{key}: expected {value!r}, actual {result[key]!r}"
             for key, value in expected.items() if result[key] != value]
    if drift:
        raise Refusal(f"rank{rank} runtime-view/transform identity drift: " + "; ".join(drift))
    return result


def runtime_identity_argv(identity: str) -> tuple[list[str], list[str]]:
    code = (
        "import hashlib,json,pathlib; "
        "h=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); "
        "print(json.dumps({'target_runtime_config':h('" + TARGET_RUNTIME + "/config.json'),"
        "'draft_runtime_config':h('" + DRAFT_RUNTIME + "/config.json'),"
        "'image_receipt_sha256':h('/evidence/image-receipt.json'),"
        "'image_receipt':json.loads(pathlib.Path('/evidence/image-receipt.json').read_text())},sort_keys=True))"
    )
    return ["docker", "exec", identity, *JSON_PYTHON, "-c", code], [
        "docker", "exec", identity, *JSON_PYTHON, "/recipe/scripts/apply_base_pipeline.py",
        "--vllm-root", "/usr/local/lib/python3.12/dist-packages/vllm",
        "--source-root", "/sources/fly", "--asset-root", "/opt/glm53",
        "--contract", "/recipe/config/patch-contract.json",
        "--image-receipt", "/evidence/image-receipt.json", "--check",
    ]


def cmd_status(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("status", values)
        return
    validate_env(values)
    manifest = bound_manifest(args.manifest, values)
    rows = collect_status(values, manifest)
    endpoint = {"health_http_200": False, "served_model_match": False}
    if rows and next((row for row in rows if row["rank"] == 0), {}).get("running"):
        base = f"http://{values['JSPARK_MASTER_ADDR']}:{values['JSPARK_API_PORT']}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as response:
                endpoint["health_http_200"] = response.status == 200
            code, models = http_json(base + "/v1/models", timeout=10)
            endpoint["served_model_match"] = (
                code == 200 and model_identities(models) == ["glm-5.3-flash"]
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
                urllib.error.URLError, Refusal):
            pass
    print(json.dumps({"grade": "ENGINEERING-EVIDENCE",
                      "ranks": rows, "endpoint": endpoint}, indent=2, sort_keys=True))


def cmd_stop(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("stop", values, remove=args.remove)
        return
    validate_env(values)
    if args.confirm != "STOP-JSPARK3":
        raise Refusal("stop confirmation must be STOP-JSPARK3")
    if args.remove and args.remove_confirm != "REMOVE-JSPARK3":
        raise Refusal("removal needs --remove-confirm REMOVE-JSPARK3")
    manifest = bound_manifest(args.manifest, values)
    by_rank = {row["rank"]: row for row in manifest["containers"]}
    stop_order = [rank for rank in (0, 1, 2) if rank in by_rank]
    for rank in stop_order:
        inspected = inspect_identity(values, by_rank[rank])
        if (inspected.get("State") or {}).get("Running"):
            remote(values, rank, ["docker", "stop", "--time", "30", by_rank[rank]["container_id"]])
    if args.remove:
        for rank in stop_order:
            inspected = inspect_identity(values, by_rank[rank])
            if (inspected.get("State") or {}).get("Running"):
                raise Refusal(f"rank{rank} still running; refusing removal")
            remote(values, rank, ["docker", "rm", by_rank[rank]["container_id"]])
    print(f"PASS stopped ranks={','.join(map(str, stop_order))} preserved={str(not args.remove).lower()}")


def http_json(url: str, payload: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = strict_object(response.read().decode("utf-8"), "HTTP response")
            return response.status, value
    except urllib.error.HTTPError as exc:
        raise Refusal(f"HTTP {exc.code} from API") from exc


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def progress_complete(logs: str, label: str, count: int) -> bool:
    fraction = re.compile(rf"(?<!\d){count}/{count}(?!\d)")
    return any(label in line and "100%" in line and fraction.search(line) is not None
               for line in ANSI_RE.sub("", logs).splitlines())


def preserve_rank0_logs(
    values: dict[str, str], manifest: dict, output: Path
) -> None:
    rank0 = next((row for row in manifest["containers"] if row["rank"] == 0), None)
    if rank0 is None:
        raise Refusal("manifest lacks rank0 for log preservation")
    process = remote(values, 0, ["docker", "logs", rank0["container_id"]], check=False)
    logs = process.stdout + process.stderr
    if not logs:
        logs = f"docker logs returned exit {process.returncode} without output\n"
    atomic_text(output, logs)


def focused_witness_argv(values: dict[str, str], base: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve().with_name("focused_witness.py")),
        "--base-url", base,
        *sum((["--counter-host", f"rank{rank}={host(values, rank)}"] for rank in range(3)), []),
        *sum((["--counter-hcas", f"rank{rank}={values[f'JSPARK_HCAS_{rank}']}"]
              for rank in range(3)), []),
    ]


def long_context_witness_argv(values: dict[str, str], base: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve().with_name("long_context_witness.py")),
        "--base-url", base,
    ]


# The pinned B4 module replaces ModelCudaGraphManager.capture with custom banked
# capture that renders no stock progress bar: the archived measured-arm startup
# log (second serving start, rank0; same six module bytes and image digest)
# contains the dflash2 draft bar completing 5/5 and the shard/startup/receipt
# lines, but zero occurrences of 'Capturing CUDA graphs (FULL)'. The stock bar
# gate was therefore a guaranteed refusal for a healthy Cadence server. Capture
# proof now comes from what the pinned path actually writes under B5_OUT/graphs:
# captures-*.jsonl receipts whose recorded graph-dump hashes match the
# on-container dump bytes, plus non-empty graph-serving-*.dot dumps for the
# full bank set, required on every rank (minimums below the CADENCE_IDENTITY_KEYS).
CADENCE_IDENTITY_KEYS = {"modules", "pth_sha256", "kda_original_sha256", "b45_out_entries",
                         "capture_receipts", "capture_dots_intact", "serving_graph_dumps"}

# Full-bank capture completeness of the byte-pinned path, per rank, at the moment
# verify runs (startup capture only; nothing else writes captures-*.jsonl):
# the pinned profile.json capture sizes give 4 base FULL family descriptors
# (16/24/32/48 tokens) plus the mandatory (4,4,1)/(8,8,1) pair captured into BOTH
# banks -> 8 serving-phase captures, each with its non-empty graph-serving-*.dot
# dump, and the same set again in the profile phase -> 16 receipt rows.
# Measured ground truth: the cadence reference pair measured-arm startup receipts per rank
# are exactly 16 rows (8 serving + 8 profile) and 8 non-empty serving dumps.
CADENCE_CAPTURE_RECEIPTS_MIN = 16
CADENCE_SERVING_DUMPS_MIN = 8


def cadence_capture_evidence_ok(value: dict) -> bool:
    return (value.get("capture_dots_intact") is True and
            isinstance(value.get("capture_receipts"), int) and
            value["capture_receipts"] >= CADENCE_CAPTURE_RECEIPTS_MIN and
            isinstance(value.get("serving_graph_dumps"), int) and
            value["serving_graph_dumps"] >= CADENCE_SERVING_DUMPS_MIN)


def cadence_capture_evidence(runtime: list[dict]) -> bool:
    return bool(runtime) and all(
        cadence_capture_evidence_ok(row.get("cadence_b45", {})) for row in runtime)


def load_gate(logs: str, runtime: list[dict]) -> dict:
    """Rank0 load/graph/cadence receipt gate for verify.

    The stock 'Capturing CUDA graphs (FULL) ... 5/5' target bar is deliberately
    not required: the byte-pinned B4 capture path cannot render it. It is
    replaced by per-rank Cadence capture evidence; the dflash2 draft bar, the
    shard-progress bars, the startup line, and the B5 calibration receipt are
    all still produced by the pinned path and stay required.
    """
    return {
        "target_loader_complete": len(runtime) == 3 and all(
            row.get("stock_v13", {}).get("loaders", {}).get("target", {}).get("state") == "COMPLETE"
            for row in runtime),
        "draft_loader_complete": len(runtime) == 3 and all(
            row.get("stock_v13", {}).get("loaders", {}).get("draft", {}).get("state") == "COMPLETE"
            for row in runtime),
        "cross_rank_loader_census": len(runtime) == 3 and all(len({
            row.get("stock_v13", {}).get("loaders", {}).get(kind, {}).get("ordered_headers_sha256")
            for row in runtime}) == 1 for kind in ("target", "draft")),
        "cadence_capture_evidence": cadence_capture_evidence(runtime),
        "draft_graphs_5": progress_complete(logs, "Capturing dflash2 CUDA graphs (FULL)", 5),
        "startup_complete": "Application startup complete." in logs,
        "b5_controller_calibrated": B5_RECEIPT_MARKER in ANSI_RE.sub("", logs),
    }


# S7 sealed ordinary caps: floor(v1.3's 18.99 / 19.24 / 19.30 GiB * 2**30), both profiles (ADVISORY Q9).
V14_ORDINARY_CAP_BYTES = {0: 20_390_357_237, 1: 20_658_792_693, 2: 20_723_217_203}
V14_IDENTITY_RE = re.compile(r"JSPARK3_V14_IDENTITY rank=(\S+) ok=(\d) (?:expect=unset )?(\{.*\})")
V16_IDENTITY_RE = re.compile(
    r"\[jspark3-v16:(coop|adaptive-k|dense-fp8)\] rank=([^\s]+) state=([^\s]+)"
)
APC_PRIORITY_RE = re.compile(
    r"\[glm53-apc-per-group\] retention_by_group=\S+ \(global=\S+ swa_env=0 "
    r"eagle_min_exempt=\[([0-9, ]*)\] low_priority=\[([0-9, ]*)\] "
)


def v16_dense_fp8_receipt_ok(text: str, rank: int, mode: str) -> bool:
    """Bind enabled FP8 to its completed census and audited B45 ownership.

    The finalizer emits the same receipt through both logging and stdout.
    Accept those identical copies, but refuse any malformed or conflicting one.
    Off retains the baseline identity and has no FP8 finalization receipt.
    """
    marker = "JSPARK3_V16_DENSE_FP8_RECEIPT="
    payloads = [line.split(marker, 1)[1] for line in text.splitlines() if marker in line]
    if mode == "off":
        return not payloads
    if mode not in ("trunk", "negative-coarse") or not payloads:
        return False
    expected = {
        "status": "JSPARK3_V16_DENSE_FP8_FINALIZE_PASS",
        "rank": rank,
        "mode": mode,
        "scheme": ("e4m3fn_per_output_channel" if mode == "trunk"
                   else "e4m3fn_per_module_scale"),
        "activation_dtype": "bfloat16",
        "w8a8": False,
        "runtime_modules": 169,
        "logical_tensors": 225,
        "category_counts": {
            "kda_o": 34, "mla_fused_qkv_a": 11, "mla_q_b": 11,
            "mla_kv_b": 11, "mla_o": 11, "shared_gate_up": 42,
            "shared_down": 42, "dense_gate_up": 3, "dense_down": 3, "lm_head": 1,
        },
        "b45_kda_composite_owner": "B45_UNCHANGED",
        "b45_kda_composite_converted": False,
        "b45_kda_input_modules": 34,
        "b45_kda_fg_modules": 34,
        "b45_kda_ownership_audited": True,
    }
    try:
        receipts = [strict_object(payload, "dense FP8 post-load receipt") for payload in payloads]
        # Canonical JSON distinguishes booleans/floats from integer counts.
        return all(canonical({key: row.get(key) for key in expected}) == canonical(expected)
                   and canonical(row) == canonical(receipts[0]) for row in receipts)
    except (Refusal, ValueError, TypeError):
        return False


def v14_identity_ok(text: str, rank: int, profile: str) -> bool:
    """One or more ok=1 worker identity lines for this rank, none ok=0."""
    rows = V14_IDENTITY_RE.findall(text)
    if not rows or any(ok != "1" for _, ok, _ in rows):
        return False
    pool = 1879048192 if profile == "full" else 0
    credit = 1006632960 if profile == "full" else 0  # S6 bounded credit: 960 MiB of the 1792 MiB pool
    cap = V14_ORDINARY_CAP_BYTES[rank]  # S7: E = min(P, cap) on both profiles
    for line_rank, _, payload in rows:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return False
        if (line_rank != str(rank) or value.get("rank") != str(rank) or
                value.get("profile") != profile or value.get("draft_block") != 640 or
                value.get("mla_block") != 2560 or value.get("pool_open_bytes") != pool or
                value.get("credited_bytes") != credit or
                (profile == "full" and not credit <= value.get("carved_bytes", 0) <= pool) or
                not v14_budget_ok(value, rank, cap, credit) or not v14_placement_ok(value, profile)):
            return False
    return True


def v14_budget_ok(value: dict, rank: int, cap: int, credit: int) -> bool:
    """S7: exactly one budget receipt, E = min(P, cap), total = E + C, bound into the identity."""
    receipt = value.get("budget") or {}
    profiled = value.get("ordinary_profiled_bytes")
    if not isinstance(profiled, int) or isinstance(profiled, bool) or value.get("budget_calls") != 1:
        return False
    effective = min(profiled, cap)
    return (value.get("ordinary_cap_bytes") == cap and value.get("ordinary_budget_bytes") == effective and
            receipt.get("rank") == rank and receipt.get("profiled_bytes") == profiled and
            receipt.get("cap_bytes") == cap and receipt.get("ordinary_bytes") == effective and
            receipt.get("clamped_bytes") == profiled - effective and receipt.get("credited_bytes") == credit and
            receipt.get("total_bytes") == effective + credit)


def v14_placement_ok(value: dict, profile: str) -> bool:
    """S7: the named placement map covers the plan at the final N (empty under display0)."""
    rows, plan = value.get("placement"), value.get("plan") or {}
    blocks = value.get("num_blocks")
    if not isinstance(rows, list):
        return False
    if profile != "full":
        return rows == []
    return (bool(rows) and isinstance(blocks, int) and blocks > 0 and
            [r.get("index") for r in rows] == plan.get("from_pool") and
            [r.get("bytes") for r in rows] == plan.get("pool_sizes") and
            all(isinstance(r.get("bytes"), int) and r["bytes"] % blocks == 0 and r.get("owners") and
                all(o.get("group") is not None and o.get("layer") for o in r["owners"]) for r in rows))


def v14_identity_gate(values: dict[str, str], manifest: dict, rank0_logs: str) -> dict:
    """Fail-closed v1.4 boot identity: every rank's worker line, rank0's scheduler lines."""
    profile = values["JSPARK3_V14_PROFILE"]
    gate: dict[str, bool] = {}
    for binding in manifest["containers"]:
        rank = binding["rank"]
        if rank == 0:
            text = rank0_logs
        else:
            process = remote(values, rank, ["docker", "logs", binding["container_id"]])
            text = process.stdout + process.stderr
        text = ANSI_RE.sub("", text)
        gate[f"rank{rank}_profile_line"] = f"JSPARK3_V14_PROFILE rank={rank} profile={profile} " in text
        gate[f"rank{rank}_worker_identity"] = v14_identity_ok(text, rank, profile)
    rank0 = ANSI_RE.sub("", rank0_logs)
    gate["sched_mode_cap_2560"] = "[cadence-sched:v6] mode=cap cap=2560 " in rank0
    gate["sched_identity_ok"] = "[cadence-sched:v6] identity ok=1 " in rank0
    gate["sched_kv_identity_ok"] = "[cadence-sched:v6] kv identity ok=1 " in rank0
    return gate


def v16_identity_gate(values: dict[str, str], manifest: dict, rank0_logs: str) -> dict[str, bool]:
    """Require exactly one selected-state identity per option on every rank."""
    options = v16_options(values)
    expected = {
        "coop": "on" if options["coop"] == "1" else "off",
        "adaptive-k": options["adaptive-k"],
        "dense-fp8": options["dense-fp8"],
    }
    ranks = [binding["rank"] for binding in manifest["containers"]]
    gate: dict[str, bool] = {
        "tp3_rank_census": len(ranks) == 3 and all(type(rank) is int for rank in ranks)
        and set(ranks) == {0, 1, 2},
    }
    if not gate["tp3_rank_census"]:
        return gate
    for binding in manifest["containers"]:
        rank = binding["rank"]
        if rank == 0:
            text = rank0_logs
        else:
            process = remote(values, rank, ["docker", "logs", binding["container_id"]])
            text = process.stdout + process.stderr
        text = ANSI_RE.sub("", text)
        rows = V16_IDENTITY_RE.findall(text)
        for option, state in expected.items():
            selected = [(line_rank, observed) for name, line_rank, observed in rows if name == option]
            gate[f"rank{rank}_{option}"] = selected == [(str(rank), state)]
        gate[f"rank{rank}_dense_fp8_post_load"] = v16_dense_fp8_receipt_ok(
            text, rank, options["dense-fp8"]
        )
    gate["rank0_apc_lru"] = v16_apc_lru_ok(ANSI_RE.sub("", rank0_logs), options["apc-lru"])
    return gate


def v16_apc_lru_ok(text: str, apc_lru: str) -> bool:
    """The engine's one coordinator line shows the selected drafter eviction priority."""
    rows = APC_PRIORITY_RE.findall(text)
    if len(rows) != 1:
        return False
    exempt, low_priority = rows[0]
    if apc_lru == "1":
        return low_priority == "" and exempt != ""
    return low_priority == exempt != ""


def _verify_bound(
    args: argparse.Namespace, values: dict[str, str], manifest: dict,
    candidate_recipe_sha256: str | None = None,
) -> None:
    if candidate_recipe_sha256 is not None:
        # Candidate mode: re-prove on every bound rank that the on-host recipe
        # bytes still hash to the declared candidate manifest before trusting
        # any runtime identity read from it.
        for binding in manifest["containers"]:
            verify_remote_recipe(values, binding["rank"], candidate_recipe_sha256)
    runtime = []
    for binding in manifest["containers"]:
        inspect_bound(values, binding, manifest)
        runtime.append({"rank": binding["rank"], **runtime_identity(values, binding, manifest)})
    stock_production = None
    if v16_options(values)["profile"] == production_stock.PROFILE:
        try:
            stock_production = production_stock.all_ranks(values, runtime)
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
    statuses = collect_status(values, manifest)
    for row in statuses:
        cgroup = row.get("cgroup") or {}
        events = cgroup.get("events") or {}
        if not (row["running"] and not row["oom_killed"] and row["restart_count"] == 0 and
                row["image_config_match"] and row["image_reference_match"] and row["memory_bytes"] == MEMORY and
                row["memory_swap_bytes"] == MEMORY and row["restart_policy"] == "no" and
                cgroup.get("memory_max") == str(MEMORY) and cgroup.get("swap_max") == "0" and
                cgroup.get("swap_current") == 0 and int(events.get("oom", -1)) == 0 and
                int(events.get("oom_kill", -1)) == 0 and int(events.get("oom_group_kill", -1)) == 0):
            raise Refusal(f"rank{row['rank']} safety/image gate failed")
    rank0 = manifest["containers"][0]
    log_process = remote(values, 0, ["docker", "logs", rank0["container_id"]])
    logs = log_process.stdout + log_process.stderr
    atomic_text(args.log_output, logs)
    load = load_gate(logs, runtime)
    if not all(load.values()):
        raise Refusal("load/graph/cadence receipt gate failed")
    v14_identity = v14_identity_gate(values, manifest, logs)
    if not all(v14_identity.values()):
        failed = sorted(key for key, ok in v14_identity.items() if not ok)
        raise Refusal(f"v1.4 identity gate failed: {','.join(failed)}")
    v16_identity = v16_identity_gate(values, manifest, logs)
    if not all(v16_identity.values()):
        failed = sorted(key for key, ok in v16_identity.items() if not ok)
        raise Refusal(f"v1.6 option identity gate failed: {','.join(failed)}")
    base = f"http://{values['JSPARK_MASTER_ADDR']}:{values['JSPARK_API_PORT']}"
    with urllib.request.urlopen(base + "/health", timeout=10) as health:
        if health.status != 200:
            raise Refusal("health endpoint is not HTTP 200")
    code, models = http_json(base + "/v1/models")
    identities = model_identities(models)
    if code != 200 or identities != ["glm-5.3-flash"]:
        raise Refusal("served-model identity drift")
    arithmetic = {
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Return only the integer: 117 + 206"}],
        "temperature": 0, "max_tokens": 8, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    code, result = http_json(base + "/v1/chat/completions", arithmetic, timeout=60)
    if code != 200 or completion_content(result).strip() != "323":
        raise Refusal("arithmetic gate failed")
    witness = subprocess.run(focused_witness_argv(values, base), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if witness.returncode:
        raise Refusal(f"focused witness failed: {witness.stderr.strip()}")
    witness_value = strict_object(witness.stdout, "focused witness response")
    long_context = subprocess.run(long_context_witness_argv(values, base), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if long_context.returncode:
        raise Refusal(f"long-context witness failed: {long_context.stderr.strip()}")
    long_context_value = strict_object(long_context.stdout, "long-context witness response")
    receipt = {
        "schema_version": 1, "grade": "ENGINEERING-EVIDENCE",
        "manifest_sha256": sha_file(args.manifest), "runtime_identity": runtime, "image_and_safety": statuses,
        "load": load, "v14_identity": v14_identity, "v16_identity": v16_identity,
        "health_http": 200, "served_model": "glm-5.3-flash",
        "arithmetic": 323, "focused_witness": witness_value,
        "long_context_witness": long_context_value, "status": "VERIFY_PASS",
    }
    if stock_production is not None:
        receipt["production_stock"] = stock_production
    if candidate_recipe_sha256 is not None:
        receipt["candidate_recipe_manifest_sha256"] = candidate_recipe_sha256
        receipt["verifier_recipe_manifest_sha256"] = recipe_manifest_sha256()
    receipt["payload_sha256"] = sha_bytes(canonical(receipt))
    atomic_json(args.output, receipt)
    print(f"PASS verify receipt={args.output} sha256={sha_file(args.output)}")


def cmd_verify(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("verify", values)
        return
    validate_env(values)
    candidate = args.candidate_recipe_manifest_sha256 or None
    manifest = bound_manifest(args.manifest, values, require_all=True, require_started=True,
                              candidate_recipe_sha256=candidate)
    try:
        _verify_bound(args, values, manifest, candidate)
    except Exception:
        try:
            preserve_rank0_logs(values, manifest, args.log_output)
        except Exception as log_exc:
            print(f"VERIFY LOG PRESERVATION REFUSED: {log_exc}", file=sys.stderr)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "start", "status", "stop", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        command.add_argument("--dry-run", action="store_true")
        if name in ("start", "status", "stop", "verify"):
            command.add_argument("--manifest", type=Path, default=Path("jspark3-release-manifest.json"))
        if name == "preflight":
            command.add_argument("--output", type=Path, default=Path("preflight.json"))
        elif name == "start":
            command.add_argument("--preflight", type=Path, default=Path("preflight.json"))
            command.add_argument("--preflight-sha256", default="")
            command.add_argument("--confirm", default="")
        elif name == "stop":
            command.add_argument("--confirm", default="")
            command.add_argument("--remove", action="store_true")
            command.add_argument("--remove-confirm", default="")
        elif name == "verify":
            command.add_argument("--output", type=Path, default=Path("verify.json"))
            command.add_argument("--log-output", type=Path, default=Path("verify-rank0.log"))
            command.add_argument("--candidate-recipe-manifest-sha256", default="",
                                 help="verify a fleet started from an earlier recipe: the declared "
                                      "candidate recipe manifest SHA-256 replaces the verifier self-identity "
                                      "check and is re-proven on every rank before any runtime read")
    return result


def main() -> int:
    args = parser().parse_args()
    values: dict[str, str] | None = None
    try:
        values = load_env(args.env_file)
        functions = {"preflight": cmd_preflight, "start": cmd_start, "status": cmd_status,
                     "stop": cmd_stop, "verify": cmd_verify}
        functions[args.command](args, values)
        return 0
    except Exception as exc:
        if (getattr(args, "command", None) == "verify" and
                not getattr(args, "dry_run", False) and values is not None):
            failure = {
                "schema_version": 1, "grade": "ENGINEERING-EVIDENCE",
                "manifest_sha256": (sha_file(args.manifest) if args.manifest.is_file() else None),
                "rank0_log_path": (str(args.log_output) if args.log_output.is_file() else None),
                "reason": str(exc), "status": "VERIFY_REFUSED",
                "operator_direction": "Inspect the bound IDs, then use stop.sh with the exact manifest if safety requires shutdown.",
            }
            failure["payload_sha256"] = sha_bytes(canonical(failure))
            try:
                atomic_json(args.output, failure)
            except OSError as write_exc:
                print(f"REFUSE: failure receipt write failed: {write_exc}", file=sys.stderr)
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_triar
v18_triar.install(globals())

if __name__ == "__main__":
    raise SystemExit(main())
