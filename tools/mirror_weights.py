#!/usr/bin/env python3
"""Safely fetch, verify, and stage the authorized JSpark3 weight mirror.

The transfer has two resumable phases: an exact pinned download into one durable
directory, then an allowlisted upload to an existing Hugging Face pull-request
revision. Dry-run is the default for both network-writing phases. This tool
never creates or merges a pull request and never uploads the repository card or
other metadata with the weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator

FETCH_CONFIRM_TOKEN = "FETCH-JSPARK3-MIRROR"
UPLOAD_CONFIRM_TOKEN = "UPLOAD-JSPARK3-MIRROR"
MIN_FREE_BYTES = 200_000_000_000
EXPECTED_FILES = 144
EXPECTED_BYTES = 175_715_854_754
EXPECTED_LFS_FILES = 123
EXPECTED_LFS_BYTES = 175_715_659_341
EXPECTED_SHARDS = 120
CACHE_PREFIX = (".cache", "huggingface")
LOCK_NAME = "jspark3-transfer.lock"
PR_REF_RE = re.compile(r"refs/pr/([1-9][0-9]*)")
COMPLETION_PATH = "jspark3/MIRROR-COMPLETION.json"
AUTHORIZED_MIRROR_STATUSES = frozenset({
    "authorized, transfer not started",
    "authorized, transfer in progress on a separate review branch; not merged",
})


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_manifest(root: Path) -> dict:
    return json.loads((root / "huggingface/jspark3/WEIGHTS-MANIFEST.json").read_text(encoding="utf-8"))


def load_release(root: Path) -> dict:
    return json.loads((root / "manifests/release.json").read_text(encoding="utf-8"))


def upstream_repo(manifest: dict) -> tuple[str, str]:
    return manifest["upstream_repository"], manifest["upstream_revision"]


def target_repo(release: dict) -> str:
    url = release["intended_destinations"]["huggingface"]
    return url.split("huggingface.co/", 1)[1].strip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_cache_path(relative: Path) -> bool:
    return relative.parts[:2] == CACHE_PREFIX


def scan_payload(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """Inventory regular payload files without traversing Hub resume metadata."""
    found: dict[str, Path] = {}
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        current = Path(dirpath)
        kept_dirs = []
        for name in dirnames:
            path = current / name
            relative = path.relative_to(directory)
            if path.is_symlink():
                problems.append(f"symlink outside Hub cache metadata: {relative.as_posix()}")
                continue
            if is_cache_path(relative):
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in filenames:
            path = current / name
            relative = path.relative_to(directory)
            if is_cache_path(relative):
                continue
            label = relative.as_posix()
            if path.is_symlink():
                problems.append(f"symlink outside Hub cache metadata: {label}")
            elif not path.is_file():
                problems.append(f"non-regular entry outside Hub cache metadata: {label}")
            else:
                found[label] = path
    return found, problems


def manifest_contract(manifest: dict) -> list[str]:
    problems: list[str] = []
    entries = manifest.get("entries", [])
    paths = [entry.get("path") for entry in entries]
    lfs = [entry for entry in entries if entry.get("lfs") is True]
    shards = {entry["path"] for entry in lfs if entry["path"].endswith(".safetensors")}
    expected_shards = {f"model-{index:05d}-of-00120.safetensors" for index in range(1, 121)}
    expected_other = {"model.safetensors.index.json", "quantization_config.json", "tokenizer.json"}
    if len(entries) != EXPECTED_FILES or manifest.get("files") != EXPECTED_FILES:
        problems.append(f"manifest must contain {EXPECTED_FILES} entries")
    if len(paths) != len(set(paths)):
        problems.append("manifest paths are not unique")
    if sum(int(entry.get("size", -1)) for entry in entries) != EXPECTED_BYTES:
        problems.append(f"manifest must total {EXPECTED_BYTES} bytes")
    if len(lfs) != EXPECTED_LFS_FILES or manifest.get("lfs_files") != EXPECTED_LFS_FILES:
        problems.append(f"manifest must identify exactly {EXPECTED_LFS_FILES} LFS payloads")
    if sum(int(entry.get("size", -1)) for entry in lfs) != EXPECTED_LFS_BYTES:
        problems.append(f"LFS payloads must total {EXPECTED_LFS_BYTES} bytes")
    if shards != expected_shards or len(shards) != EXPECTED_SHARDS:
        problems.append("LFS shard allowlist is not exactly model-00001..00120-of-00120.safetensors")
    if {entry["path"] for entry in lfs} - shards != expected_other:
        problems.append("non-shard LFS allowlist is not the three pinned JSON files")
    for entry in entries:
        if "dflash" in str(entry.get("path", "")).lower():
            problems.append(f"DFlash2 path is forbidden: {entry.get('path')}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            problems.append(f"invalid SHA-256 in manifest: {entry.get('path')}")
    return problems


def lfs_entries(manifest: dict) -> list[dict]:
    return sorted((entry for entry in manifest["entries"] if entry.get("lfs") is True),
                  key=lambda entry: entry["path"])


def prepare_directory(path: Path, *, create: bool) -> Path:
    if path.is_symlink():
        raise ValueError(f"working directory itself must not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"working path is not a directory: {directory}")
    cache = directory / CACHE_PREFIX[0]
    hub_cache = cache / CACHE_PREFIX[1]
    for candidate in (cache, hub_cache):
        if candidate.is_symlink():
            raise ValueError(f"Hub cache boundary must not be a symlink: {candidate}")
    return directory


def preflight(root: Path, directory: Path, min_free_bytes: int) -> list[str]:
    manifest = load_manifest(root)
    problems = manifest_contract(manifest)
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    found, scan_problems = scan_payload(directory)
    problems.extend(scan_problems)
    for relative, path in found.items():
        entry = expected.get(relative)
        if entry is None:
            problems.append(f"extra payload file not in manifest: {relative}")
        elif path.stat().st_size != entry["size"]:
            problems.append(f"completed payload has wrong size: {relative}")
    free = shutil.disk_usage(directory).free
    if free < min_free_bytes:
        problems.append(f"only {free} bytes free; require at least {min_free_bytes}")
    checks = (
        (["hf", "--version"], None),
        (["hf", "download", "--help"], "--local-dir"),
        (["hf", "upload-large-folder", "--help"], "--revision"),
    )
    for command, required_text in checks:
        try:
            result = subprocess.run(command, check=False, text=True, capture_output=True)
        except FileNotFoundError:
            problems.append("installed `hf` CLI not found")
            break
        output = result.stdout + result.stderr
        if result.returncode or not output.strip() or (required_text and required_text not in output):
            problems.append(f"installed CLI does not support: {shlex.join(command)}")
    if not problems:
        print(f"PREFLIGHT PASS dir={directory} free={free} present={len(found)}/{EXPECTED_FILES} "
              f"lfs={EXPECTED_LFS_FILES}; no network call made")
    return problems


def report_problems(label: str, problems: list[str]) -> int:
    print(f"{label} FAIL ({len(problems)} problems)", file=sys.stderr)
    for problem in problems[:30]:
        print("  " + problem, file=sys.stderr)
    return 9


@contextmanager
def transfer_lock(directory: Path) -> Iterator[None]:
    cache = directory / CACHE_PREFIX[0] / CACHE_PREFIX[1]
    cache.mkdir(parents=True, exist_ok=True)
    if cache.is_symlink():
        raise ValueError(f"Hub cache boundary must not be a symlink: {cache}")
    lock_path = cache / LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another fetch or upload already holds {lock_path}") from exc
        yield


def verify(root: Path, directory: Path, report: bool = True) -> list[str]:
    manifest = load_manifest(root)
    problems = manifest_contract(manifest)
    found, scan_problems = scan_payload(directory)
    problems.extend(scan_problems)
    checked = 0
    for entry in manifest["entries"]:
        relative = entry["path"]
        path = found.pop(relative, None)
        if path is None:
            problems.append(f"missing: {relative}")
            continue
        size = path.stat().st_size
        if size != entry["size"]:
            problems.append(f"size drift: {relative} ({size} != {entry['size']})")
        else:
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                problems.append(f"hash drift: {relative} ({digest} != {entry['sha256']})")
            else:
                checked += 1
    problems.extend(f"extra payload file not in manifest: {path}" for path in sorted(found))
    if report:
        if problems:
            report_problems("VERIFY", problems)
        else:
            print(f"VERIFY PASS files={checked} bytes={manifest['bytes']} "
                  f"lfs_files={EXPECTED_LFS_FILES} lfs_bytes={EXPECTED_LFS_BYTES}")
    return problems


def upload_command(destination: str, directory: Path, pr_ref: str, manifest: dict,
                   workers: int) -> list[str]:
    command = ["hf", "upload-large-folder", destination, str(directory),
               "--repo-type", "model", "--revision", pr_ref, "--num-workers", str(workers)]
    for entry in lfs_entries(manifest):
        command.extend(["--include", entry["path"]])
    return command


def validate_pr_ref(value: str) -> str:
    if not PR_REF_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("must be an existing pull-request ref like refs/pr/7")
    return value


def publication_problems(release: dict) -> list[str]:
    problems = []
    mirror = release.get("weights_mirror", {})
    if release.get("publication_authorized") is not True:
        problems.append("publication_authorized is not true")
    if mirror.get("gate") is not None:
        problems.append(f"weights mirror gate is not clear: {mirror.get('gate')}")
    if mirror.get("status") not in AUTHORIZED_MIRROR_STATUSES:
        problems.append("weights mirror state is not an authorized pre-merge transfer state")
    return problems


def cmd_preflight(args: argparse.Namespace) -> int:
    try:
        directory = prepare_directory(args.dir, create=True)
    except ValueError as exc:
        return report_problems("PREFLIGHT", [str(exc)])
    problems = preflight(repo_root(), directory, args.min_free_bytes)
    return report_problems("PREFLIGHT", problems) if problems else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    root = repo_root()
    try:
        directory = prepare_directory(args.dir, create=True)
    except ValueError as exc:
        return report_problems("FETCH", [str(exc)])
    problems = preflight(root, directory, args.min_free_bytes)
    if problems:
        return report_problems("FETCH PREFLIGHT", problems)
    manifest = load_manifest(root)
    repository, revision = upstream_repo(manifest)
    command = ["hf", "download", repository, "--revision", revision, "--repo-type", "model",
               "--local-dir", str(directory), "--max-workers", str(args.workers)]
    if args.dry_run:
        print("DRY-RUN fetch command:", shlex.join(command))
        print(f"DRY-RUN: reuse this exact directory; preserve {directory / '.cache/huggingface'}")
        return 0
    if args.confirm != FETCH_CONFIRM_TOKEN:
        return report_problems("FETCH", [f"confirmation must be {FETCH_CONFIRM_TOKEN}"])
    try:
        with transfer_lock(directory):
            process = subprocess.run(command, check=False)
    except ValueError as exc:
        return report_problems("FETCH", [str(exc)])
    if process.returncode:
        return report_problems("FETCH", [f"download exited {process.returncode}; resume with the same command and directory"])
    print("FETCH PROCESS EXITED 0; this is not completion. Run offline verify next.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        directory = prepare_directory(args.dir, create=False)
    except ValueError as exc:
        return report_problems("VERIFY", [str(exc)])
    return 1 if verify(repo_root(), directory) else 0


def cmd_upload(args: argparse.Namespace) -> int:
    root = repo_root()
    try:
        directory = prepare_directory(args.dir, create=False)
    except ValueError as exc:
        return report_problems("UPLOAD", [str(exc)])
    if verify(root, directory):
        return report_problems("UPLOAD", ["offline verification failed; nothing uploaded"])
    manifest = load_manifest(root)
    release = load_release(root)
    problems = publication_problems(release)
    if problems:
        return report_problems("UPLOAD", problems)
    destination = target_repo(release)
    command = upload_command(destination, directory, args.pr_ref, manifest, args.workers)
    if args.dry_run:
        print(f"DRY-RUN: exact allowlist contains {EXPECTED_LFS_FILES} files and {EXPECTED_LFS_BYTES} bytes")
        print("DRY-RUN upload command:", shlex.join(command))
        print("DRY-RUN: no network call made; metadata and DFlash2 are excluded")
        return 0
    if args.confirm != UPLOAD_CONFIRM_TOKEN:
        return report_problems("UPLOAD", [f"confirmation must be {UPLOAD_CONFIRM_TOKEN}"])
    try:
        with transfer_lock(directory):
            process = subprocess.run(command, check=False)
    except ValueError as exc:
        return report_problems("UPLOAD", [str(exc)])
    if process.returncode:
        return report_problems("UPLOAD", [f"uploader exited {process.returncode}; resume with the same directory, PR ref, and command"])
    print("UPLOAD PROCESS EXITED 0; this is not completion. Run remote-verify against the PR ref.")
    return 0


def batches(values: list[str], size: int = 50) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def remote_infos(api, repository: str, revision: str, paths: set[str]) -> dict[str, object]:
    found = {}
    for batch in batches(sorted(paths)):
        for info in api.get_paths_info(repository, batch, revision=revision, repo_type="model", expand=True):
            found[info.path] = info
    return found


def audit_remote(root: Path, pr_ref: str, receipt: Path | None) -> tuple[dict, list[str]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        return {}, [f"huggingface_hub is unavailable: {exc}"]
    manifest = load_manifest(root)
    release = load_release(root)
    repository = target_repo(release)
    base_revision = release["weights_mirror"]["hf_revision"]
    expected = {entry["path"]: entry for entry in lfs_entries(manifest)}
    allowed_additions = set(expected)
    if receipt is not None:
        allowed_additions.add(COMPLETION_PATH)
    api = HfApi()
    problems: list[str] = []
    try:
        base_paths = set(api.list_repo_files(repository, revision=base_revision, repo_type="model"))
        pr_paths = set(api.list_repo_files(repository, revision=pr_ref, repo_type="model"))
        additions = pr_paths - base_paths
        missing_base = base_paths - pr_paths
        unexpected = additions - allowed_additions
        missing = set(expected) - additions
        if missing_base:
            problems.append(f"PR deletes {len(missing_base)} metadata paths")
        if unexpected:
            problems.append("unexpected PR additions: " + ", ".join(sorted(unexpected)[:10]))
        if any("dflash" in path.lower() for path in pr_paths):
            problems.append("DFlash2 content appears in the PR revision")
        base_info = remote_infos(api, repository, base_revision, base_paths)
        pr_base_info = remote_infos(api, repository, pr_ref, base_paths & pr_paths)
        changed = [path for path in sorted(base_paths & pr_paths)
                   if (base_info[path].size, base_info[path].blob_id) !=
                   (pr_base_info[path].size, pr_base_info[path].blob_id)]
        if changed:
            problems.append("PR changes live metadata before completion: " + ", ".join(changed[:10]))
        present = set(expected) & additions
        weight_info = remote_infos(api, repository, pr_ref, present)
        bad_weights = []
        measured_bytes = 0
        for path in sorted(present):
            entry = expected[path]
            info = weight_info.get(path)
            if info is None:
                bad_weights.append(f"{path}: no path metadata")
                continue
            measured_bytes += info.size
            remote_sha = info.lfs.sha256 if info.lfs is not None else None
            if info.size != entry["size"] or remote_sha != entry["sha256"]:
                bad_weights.append(f"{path}: size/SHA-256 mismatch")
        if bad_weights:
            problems.append("remote payload mismatch: " + "; ".join(bad_weights[:10]))
        if receipt is not None:
            if not receipt.is_file() or receipt.is_symlink():
                problems.append("local completion receipt must be one regular file")
            elif COMPLETION_PATH not in additions:
                problems.append("completion receipt is absent from PR revision")
            else:
                downloaded = Path(hf_hub_download(repository, COMPLETION_PATH, revision=pr_ref,
                                                   repo_type="model"))
                if sha256_file(downloaded) != sha256_file(receipt):
                    problems.append("remote completion receipt differs from the local proven receipt")
        result = {
            "repository": repository,
            "pr_ref": pr_ref,
            "metadata_parent_revision": base_revision,
            "base_paths": len(base_paths),
            "pr_additions": len(additions),
            "lfs_present": len(present),
            "lfs_missing": len(missing),
            "lfs_bytes_present": measured_bytes,
            "unexpected_additions": len(unexpected),
            "metadata_changed": len(changed),
            "completion_receipt_present": COMPLETION_PATH in additions,
        }
        if missing:
            problems.append(f"remote mirror incomplete: {len(missing)} allowlisted LFS paths missing")
        return result, problems
    except Exception as exc:
        return {}, [f"remote audit failed: {type(exc).__name__}: {exc}"]


def cmd_remote_status(args: argparse.Namespace) -> int:
    result, problems = audit_remote(repo_root(), args.pr_ref, args.receipt)
    if result:
        print(json.dumps(result, indent=2, sort_keys=True))
    hard = [problem for problem in problems if not problem.startswith("remote mirror incomplete:")]
    if hard:
        return report_problems("REMOTE STATUS", hard)
    print("REMOTE STATUS: reachable and structurally safe; process liveness is not completion")
    return 0


def cmd_remote_verify(args: argparse.Namespace) -> int:
    result, problems = audit_remote(repo_root(), args.pr_ref, args.receipt)
    if result:
        print(json.dumps(result, indent=2, sort_keys=True))
    if problems:
        return report_problems("REMOTE VERIFY", problems)
    print(f"REMOTE VERIFY PASS: exact {EXPECTED_LFS_FILES} payloads, {EXPECTED_LFS_BYTES} bytes, "
          "all LFS SHA-256 values match, live metadata unchanged, no DFlash2")
    return 0


def cmd_completion_receipt(args: argparse.Namespace) -> int:
    result, problems = audit_remote(repo_root(), args.pr_ref, None)
    if problems:
        return report_problems("COMPLETION RECEIPT", problems)
    manifest = load_manifest(repo_root())
    payload = {
        "schema_version": 1,
        "status": "remote-weight-pr-verified",
        "verified_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "target_repository": result["repository"],
        "pull_request_ref": args.pr_ref,
        "metadata_parent_revision": result["metadata_parent_revision"],
        "upstream_repository": manifest["upstream_repository"],
        "upstream_revision": manifest["upstream_revision"],
        "lfs_files": EXPECTED_LFS_FILES,
        "lfs_bytes": EXPECTED_LFS_BYTES,
        "weights_manifest_sha256": sha256_file(repo_root() / "huggingface/jspark3/WEIGHTS-MANIFEST.json"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.is_symlink():
        return report_problems("COMPLETION RECEIPT", ["output path is a symlink"])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"COMPLETION RECEIPT WRITTEN {args.out} sha256={sha256_file(args.out)}")
    return 0


def cmd_self_test(_args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = load_manifest(root)
    problems = manifest_contract(manifest)
    release = load_release(root)
    if current_problems := publication_problems(release):
        problems.append("current release manifest fails publication gate: " + "; ".join(current_problems))
    for status in AUTHORIZED_MIRROR_STATUSES:
        candidate = {**release, "publication_authorized": True,
                     "weights_mirror": {**release.get("weights_mirror", {}),
                                        "status": status, "gate": None}}
        if publication_problems(candidate):
            problems.append(f"authorized pre-merge status was rejected: {status}")
    for status in (None, "", "authorized, transfer not started ",
                   "authorized, transfer complete"):
        candidate = {**release, "publication_authorized": True,
                     "weights_mirror": {**release.get("weights_mirror", {}),
                                        "status": status, "gate": None}}
        if not publication_problems(candidate):
            problems.append(f"unauthorized mirror status was accepted: {status!r}")
    for authorization in (False, None, 1):
        candidate = {**release, "publication_authorized": authorization,
                     "weights_mirror": {**release.get("weights_mirror", {}),
                                        "status": "authorized, transfer not started", "gate": None}}
        if not publication_problems(candidate):
            problems.append(f"non-true publication authorization was accepted: {authorization!r}")
    for gate in ("license review", "", False, 0):
        candidate = {**release, "publication_authorized": True,
                     "weights_mirror": {**release.get("weights_mirror", {}),
                                        "status": "authorized, transfer not started", "gate": gate}}
        if not publication_problems(candidate):
            problems.append(f"non-null mirror gate was accepted: {gate!r}")
    command = upload_command("owner/repo", Path("/large/mirror"), "refs/pr/7", manifest, 4)
    includes = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--include"]
    expected = [entry["path"] for entry in lfs_entries(manifest)]
    if includes != expected or len(includes) != EXPECTED_LFS_FILES:
        problems.append("upload command is not the exact manifest-derived LFS allowlist")
    if any(path not in expected for path in includes):
        problems.append("upload command includes non-LFS metadata")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / ".cache/huggingface").mkdir(parents=True)
        (directory / ".cache/huggingface/allowed-link").symlink_to(directory)
        _, cache_problems = scan_payload(directory)
        if cache_problems:
            problems.append("Hub cache metadata symlinks were not isolated")
        (directory / "bad-link").symlink_to(directory)
        _, outside_problems = scan_payload(directory)
        if not any("bad-link" in problem for problem in outside_problems):
            problems.append("outside-cache symlink was not rejected")
    if problems:
        return report_problems("SELF-TEST", problems)
    print("SELF-TEST PASS manifest-contract publication-gate exact-allowlist cache-boundary symlink-rejection")
    return 0


def add_dir_free_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="offline CLI, disk, manifest, and directory safety checks")
    add_dir_free_options(check)
    check.set_defaults(func=cmd_preflight)
    fetch = sub.add_parser("fetch", help="resumably download the exact pinned source; dry-run by default")
    add_dir_free_options(fetch)
    fetch.add_argument("--workers", type=int, default=4)
    fetch.add_argument("--confirm", default="")
    fetch.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    fetch.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    fetch.set_defaults(func=cmd_fetch)
    verify_parser = sub.add_parser("verify", help="offline size and SHA-256 verification of all 144 files")
    verify_parser.add_argument("dir", type=Path)
    verify_parser.set_defaults(func=cmd_verify)
    upload = sub.add_parser("upload", help="resumably upload only 123 LFS payloads to a PR; dry-run by default")
    upload.add_argument("dir", type=Path)
    upload.add_argument("--pr-ref", required=True, type=validate_pr_ref)
    upload.add_argument("--workers", type=int, default=4)
    upload.add_argument("--confirm", default="")
    upload.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    upload.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    upload.set_defaults(func=cmd_upload)
    status = sub.add_parser("remote-status", help="read-only PR progress/liveness check; not completion")
    status.add_argument("--pr-ref", required=True, type=validate_pr_ref)
    status.add_argument("--receipt", type=Path)
    status.set_defaults(func=cmd_remote_status)
    remote = sub.add_parser("remote-verify", help="prove exact remote PR completion by path, size, and LFS SHA-256")
    remote.add_argument("--pr-ref", required=True, type=validate_pr_ref)
    remote.add_argument("--receipt", type=Path)
    remote.set_defaults(func=cmd_remote_verify)
    receipt = sub.add_parser("completion-receipt", help="write metadata only after exact remote proof")
    receipt.add_argument("--pr-ref", required=True, type=validate_pr_ref)
    receipt.add_argument("--out", required=True, type=Path)
    receipt.set_defaults(func=cmd_completion_receipt)
    self_test = sub.add_parser("self-test", help="offline focused safety assertions")
    self_test.set_defaults(func=cmd_self_test)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
