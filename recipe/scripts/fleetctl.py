#!/usr/bin/env python3
"""Fail-closed three-node lifecycle controller for the JSpark3 v1 recipe."""

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

IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58"
IMAGE_CONFIG = "sha256:ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263"
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
    "JSPARK_API_BIND", "JSPARK_API_PORT",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
RECIPE_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ROW_KEYS = {
    "schema_version", "rank", "system", "architecture", "gpu", "image",
    "image_manifest", "image_config", "fabric_legs", "gid_index",
    "gid_entries", "management_routes", "checkpoint", "recipe_manifest",
    "recipe_manifest_sha256", "free_memory", "free_disk",
    "release_name_absent", "status",
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
    unknown = sorted(set(values) - set(REQUIRED_ENV))
    if unknown:
        raise Refusal(f"unknown env keys: {','.join(unknown)}")
    return values


def split2(value: str, label: str) -> list[str]:
    result = [part.strip() for part in value.split(",")]
    if (len(result) != 2 or any(not part for part in result) or len(set(result)) != 2 or
            value != ",".join(result)):
        raise Refusal(f"{label} must contain two distinct comma-separated values")
    return result


def validate_env(values: dict[str, str]) -> None:
    if any(key in os.environ for key in FORBIDDEN_NCCL) or any(key in values for key in FORBIDDEN_NCCL):
        raise Refusal("forbidden inherited NCCL override")
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
    gid = values["JSPARK_IB_GID_INDEX"]
    port = values["JSPARK_MASTER_PORT"]
    api_port = values["JSPARK_API_PORT"]
    if not gid.isdigit() or not 0 <= int(gid) <= 255:
        raise Refusal("invalid common GID index")
    if port != "29533" or api_port != "8000" or values["JSPARK_API_BIND"] != "0.0.0.0":
        raise Refusal("JSpark3 v1 requires master port 29533 and API bind 0.0.0.0:8000")
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


def remote(values: dict[str, str], rank: int, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(ssh_argv(values, rank, argv), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and process.returncode:
        detail = process.stderr.strip().splitlines()[-1:] or ["no detail"]
        raise Refusal(f"rank{rank} remote command failed: {detail[0]}")
    return process


def rank_env(
    values: dict[str, str],
    rank: int,
    preflight_sha256: str = "0" * 64,
    recipe_manifest_sha256: str | None = None,
) -> list[str]:
    socket = values[f"JSPARK_SOCKET_IFNAME_{rank}"]
    fixed = {
        "NODE_RANK": str(rank),
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
        "GLM53_SUPPRESS_STOPS_IN_REASONING": "1", "GLM53_MIXED_PREFILL_CHUNK": "skip",
        "MODEL_PATH": TARGET_RUNTIME, "DRAFT_PATH": DRAFT_RUNTIME,
        "JSPARK_TARGET_RUNTIME": TARGET_RUNTIME,
        "JSPARK_DRAFT_RUNTIME": DRAFT_RUNTIME,
        "JSPARK3_KDA_MIXED_OUTPUT_BLOCKS": "8",
        "JSPARK3_KDA_FG_BATCHED": "1",
        "JSPARK3_TRUNK_W8A16": "1",
        "JSPARK3_TRUNK_W8A16_K704_GROUP": "64",
        "JSPARK_PREFLIGHT_SHA256": preflight_sha256,
        "JSPARK_RECIPE_MANIFEST_SHA256": recipe_manifest_sha256 or sha_file(RECIPE_ROOT / "SHA256SUMS"),
    }
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
        "cudagraph_capture_sizes": [8, 16, 24, 32, 48],
    }, separators=(",", ":"))
    overrides = json.dumps({
        "num_attention_heads": 66, "num_key_value_heads": 66, "linear_num_heads": 66,
        "text_config": {"num_attention_heads": 66, "num_key_value_heads": 66, "linear_num_heads": 66},
    }, separators=(",", ":"))
    argv = [
        "--served-model-name", "glm-5.3-flash", "--host", values["JSPARK_API_BIND"],
        "--port", values["JSPARK_API_PORT"], "--trust-remote-code", "--quantization", "exl3",
        "--tensor-parallel-size", "3", "--pipeline-parallel-size", "1",
        "--gpu-memory-utilization", "0.83", "--max-model-len", "1000000",
        "--max-num-seqs", "32", "--max-num-batched-tokens", "8192",
        "--kv-cache-dtype", "fp8", "--enable-prefix-caching", "--no-enable-flashinfer-autotune",
        "--distributed-executor-backend", "mp", "--nnodes", "3", "--node-rank", str(rank),
        "--master-addr", values["JSPARK_MASTER_ADDR"], "--master-port", values["JSPARK_MASTER_PORT"],
        "--tool-call-parser", "glm47", "--enable-auto-tool-choice", "--reasoning-parser", "glm45",
        "--language-model-only", "--mm-encoder-tp-mode", "data", "--enable-expert-parallel",
        "--hf-overrides", overrides, "--speculative-config", speculative,
        "--compilation-config", compilation,
        "--default-chat-template-kwargs", "{\"enable_thinking\":false}",
        "--chat-template", "/sources/fly/files/chat_template.jinja",
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
    name = f"jspark3-rank{rank}"
    work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
    argv = [
        "docker", "create", "--name", name, "--hostname", name, "--restart", "no",
        "--network", "host", "--ipc", "host", "--shm-size", str(SHM),
        "--memory", str(MEMORY), "--memory-swap", str(MEMORY),
        "--ulimit", "memlock=-1:-1", "--cap-add", "IPC_LOCK",
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
        "--label", "jspark3.release=v1.0.0", "--label", f"jspark3.rank={rank}",
        "--label", "jspark3.grade=engineering-evidence",
    ]
    for item in rank_env(values, rank, preflight_sha256, recipe_manifest_sha256):
        argv.extend(("--env", item))
    argv.extend(("--workdir", "/vllm-workspace", "--entrypoint", "/recipe/scripts/container_entry.sh", IMAGE))
    argv.extend(server_argv(values, rank))
    return argv


def preflight_argv(values: dict[str, str], rank: int) -> list[str]:
    peers = [values[f"JSPARK_RANK{other}_ADDR"] for other in range(3) if other != rank]
    return [
        "python3", f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
        "--rank", str(rank), "--ifaces", values[f"JSPARK_FABRIC_IFACES_{rank}"],
        "--cidrs", values[f"JSPARK_FABRIC_ADDRS_{rank}"], "--hcas", values[f"JSPARK_HCAS_{rank}"],
        "--gid-index", values["JSPARK_IB_GID_INDEX"],
        "--socket-iface", values[f"JSPARK_SOCKET_IFNAME_{rank}"],
        "--management-addr", values[f"JSPARK_RANK{rank}_ADDR"],
        "--peer-addrs", ",".join(peers), "--model-root", values["JSPARK_MODEL_ROOT"],
        "--work-root", values["JSPARK_WORK_ROOT"], "--recipe-root", values["JSPARK_RECIPE_ROOT"],
        "--fly-root", values["JSPARK_FLY_ROOT"],
    ]


def checkpoint_argv(values: dict[str, str]) -> list[str]:
    root = values["JSPARK_MODEL_ROOT"]
    return [
        "python3", "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/validate_checkpoint.py",
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
            "python3", "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
            "--recipe-only", "--recipe-root", values["JSPARK_RECIPE_ROOT"],
            "--expected-recipe-manifest-sha256", recipe_sha,
        ]
        for rank in (0, 1, 2):
            print(f"DRY-RUN rank{rank} {shlex.join(ssh_argv(values, rank, recipe_check))}")
        for rank in (2, 1, 0):
            remote_command = ["docker", "container", "inspect", f"jspark3-rank{rank}"]
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
        logs = ["docker", "logs", "<MANIFEST_CONTAINER_ID_RANK0>"]
        print(f"DRY-RUN rank0 {shlex.join(ssh_argv(values, 0, logs))}")
        print("DRY-RUN controller GET /health and GET /v1/models")
        print("DRY-RUN controller POST arithmetic 323")
        base = f"http://{values['JSPARK_MASTER_ADDR']}:{values['JSPARK_API_PORT']}"
        print(f"DRY-RUN controller {shlex.join(focused_witness_argv(values, base))}")
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
            commands = [["docker", "container", "inspect", f"jspark3-rank{rank}"]]
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
    return sha_bytes(canonical({key: values[key] for key in REQUIRED_ENV}))


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
        "image_manifest": IMAGE.split("@", 1)[1], "image_config": IMAGE_CONFIG,
        "fabric_legs": 2, "gid_index": int(values["JSPARK_IB_GID_INDEX"]),
        "gid_entries": ["PASS", "PASS"], "management_routes": "PASS",
        "checkpoint": "PASS", "recipe_manifest": "PASS",
        "recipe_manifest_sha256": recipe_sha, "free_memory": "PASS_GE_72_GIB",
        "free_disk": "PASS_GE_8_GIB_MODEL_AND_WORK", "release_name_absent": True,
        "status": "PASS",
    }


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
            receipt.get("image_manifest") != IMAGE.split("@", 1)[1] or
            receipt.get("image_config") != IMAGE_CONFIG or
            receipt.get("recipe_manifest_sha256") != recipe_sha or
            not isinstance(rows, list) or len(rows) != 3 or
            rows != [expected_preflight_row(values, rank, recipe_sha) for rank in range(3)] or
            not SHA_RE.fullmatch(preflight_sha)):
        raise Refusal("preflight receipt does not bind this exact configuration")
    return rows, recipe_sha


def remote_recipe_sha(values: dict[str, str], rank: int, expected: str | None = None) -> str:
    argv = [
        "python3", "-B", f"{values['JSPARK_RECIPE_ROOT']}/scripts/remote_preflight.py",
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
    return ["python3", "-c", code, path, encoded]


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
        "image_manifest": IMAGE.split("@", 1)[1], "image_config": IMAGE_CONFIG,
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
    name = f"jspark3-rank{rank}"
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
        "jspark3.release": "v1.0.0", "jspark3.rank": str(rank),
        "jspark3.grade": "engineering-evidence",
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
    if devices != [{"PathOnHost": "/dev/infiniband", "PathInContainer": "/dev/infiniband",
                    "CgroupPermissions": "rwm"}]:
        raise Refusal(f"rank{rank} InfiniBand device contract drift")
    requests = host_config.get("DeviceRequests") or []
    if requests != [{"Driver": "", "Count": -1, "DeviceIDs": None,
                     "Capabilities": [["gpu"]], "Options": {}}]:
        raise Refusal(f"rank{rank} GPU request contract drift")
    ulimits = host_config.get("Ulimits") or []
    if ulimits != [{"Name": "memlock", "Soft": -1, "Hard": -1}]:
        raise Refusal(f"rank{rank} memlock contract drift")
    if (host_config.get("CapAdd") != ["CAP_IPC_LOCK"] or
            host_config.get("SecurityOpt") != ["label=disable"]):
        raise Refusal(f"rank{rank} capability/security contract drift")


def validate_admission_state(rank: int, item: dict) -> None:
    state = item.get("State") or {}
    if state.get("OOMKilled") is not False or item.get("RestartCount") != 0:
        raise Refusal(f"rank{rank} container admission state drift")


def validate_bound_identity(rank: int, identity: str, item: dict) -> None:
    name = f"jspark3-rank{rank}"
    config = item.get("Config") or {}
    labels = config.get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.title": "jspark3-recipe",
        "jspark3.release": "v1.0.0", "jspark3.rank": str(rank),
        "jspark3.grade": "engineering-evidence",
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
        "preflight_sha256": args.preflight_sha256, "image_manifest": IMAGE.split("@", 1)[1],
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
            if remote(values, rank, ["docker", "container", "inspect", f"jspark3-rank{rank}"], check=False).returncode == 0:
                raise Refusal(f"rank{rank} deterministic release name already exists")
        for rank in (2, 1, 0):
            verify_remote_recipe(values, rank, recipe_sha)
            work = f"{values['JSPARK_WORK_ROOT']}/rank{rank}"
            remote(values, rank, ["mkdir", "-p", f"{work}/evidence", f"{work}/cache/vllm",
                                  f"{work}/cache/triton", f"{work}/cache/tilelang"])
            candidate = remote(values, rank, container_argv(
                values, rank, args.preflight_sha256, recipe_sha)).stdout.strip()
            if not SHA_RE.fullmatch(candidate):
                probe = remote(values, rank, ["docker", "container", "inspect", f"jspark3-rank{rank}"], check=False)
                if probe.returncode == 0:
                    recovered = inspect_document(
                        probe.stdout, f"rank{rank} create-recovery"
                    ).get("Id", "")
                    if SHA_RE.fullmatch(recovered):
                        created.append({"rank": rank, "container_id": recovered,
                                        "name": f"jspark3-rank{rank}", "image_config": IMAGE_CONFIG,
                                        "image_receipt_sha256": None})
                raise Refusal(f"rank{rank} docker create returned an unsafe identity")
            binding = {"rank": rank, "container_id": candidate,
                       "name": f"jspark3-rank{rank}", "image_config": IMAGE_CONFIG,
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
    require_started: bool = False,
) -> dict:
    value = read_receipt(path)
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
            value.get("image_manifest") != IMAGE.split("@", 1)[1] or
            value.get("image_config") != IMAGE_CONFIG or
            value.get("recipe_manifest_sha256") != recipe_manifest_sha256() or
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
                row.get("name") != f"jspark3-rank{row['rank']}" or
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
    return ["python3", "-c", code, str(pid)]


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
            image_receipt.get("manifest_digest") != IMAGE.split("@", 1)[1] or
            image_receipt.get("config_digest") != IMAGE_CONFIG or
            image_receipt.get("verification") != "host-observed-inspect-bound-create" or
            image_receipt.get("container_id") != identity or image_receipt.get("rank") != rank or
            image_receipt.get("preflight_sha256") != manifest["preflight_sha256"] or
            image_receipt.get("recipe_manifest_sha256") != manifest["recipe_manifest_sha256"] or
            configs.get("image_receipt_sha256") != binding["image_receipt_sha256"]):
        raise Refusal(f"rank{rank} host-minted image receipt drift")
    pipeline = remote(values, rank, pipeline_argv)
    pipeline_value = strict_object(pipeline.stdout, f"rank{rank} transform pipeline")
    result = {**configs, "transform_pipeline_state": pipeline_value.get("state"),
              "transform_target_set_sha256": pipeline_value.get("target_set_sha256"),
              "image_receipt_bound": True}
    if (result["target_runtime_config"] != "55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd" or
            result["draft_runtime_config"] != "c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f" or
            result["transform_pipeline_state"] != "ALREADY_APPLIED" or
            result["transform_target_set_sha256"] != "ed7b0092e5a5a1d2aeb6dd2cbe9780783df89d70f733dff019dd05aa8cdd08bd"):
        raise Refusal(f"rank{rank} runtime-view/transform identity drift")
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
    return ["docker", "exec", identity, "python3", "-c", code], [
        "docker", "exec", identity, "python3", "/recipe/scripts/apply_base_pipeline.py",
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


def _verify_bound(
    args: argparse.Namespace, values: dict[str, str], manifest: dict
) -> None:
    runtime = []
    for binding in manifest["containers"]:
        inspect_bound(values, binding, manifest)
        runtime.append({"rank": binding["rank"], **runtime_identity(values, binding, manifest)})
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
    load = {
        "target_shards_120": progress_complete(logs, "Loading safetensors checkpoint shards", 120),
        "draft_shards_1": progress_complete(logs, "Loading safetensors checkpoint shards", 1),
        "target_graphs_5": progress_complete(logs, "Capturing CUDA graphs (FULL)", 5),
        "draft_graphs_5": progress_complete(logs, "Capturing dflash2 CUDA graphs (FULL)", 5),
        "startup_complete": "Application startup complete." in logs,
    }
    if not all(load.values()):
        raise Refusal("rank0 load/graph receipt gate failed")
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
    receipt = {
        "schema_version": 1, "grade": "ENGINEERING-EVIDENCE",
        "manifest_sha256": sha_file(args.manifest), "runtime_identity": runtime, "image_and_safety": statuses,
        "load": load, "health_http": 200, "served_model": "glm-5.3-flash",
        "arithmetic": 323, "focused_witness": witness_value, "status": "VERIFY_PASS",
    }
    receipt["payload_sha256"] = sha_bytes(canonical(receipt))
    atomic_json(args.output, receipt)
    print(f"PASS verify receipt={args.output} sha256={sha_file(args.output)}")


def cmd_verify(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.dry_run:
        render_dry_run("verify", values)
        return
    validate_env(values)
    manifest = bound_manifest(args.manifest, values, require_all=True, require_started=True)
    try:
        _verify_bound(args, values, manifest)
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


if __name__ == "__main__":
    raise SystemExit(main())
