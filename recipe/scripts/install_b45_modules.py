#!/usr/bin/env python3
"""Fail-closed installer for the measured Cadence B4+B5 module layer.

Copies the six sealed byte-exact modules to /opt/b45 and installs zzz_b45.pth
as the single import owner in site-packages, exactly as the measured cadence reference pair
construction did through `docker cp`. The on-disk original KDA module is never
modified; the installer refuses unless its bytes still match the pinned hash
the bootstrap asserts on. States:

  fresh    - targets absent, installed from verified sources (0444)
  already  - every target already carries the exact expected bytes
  REFUSE   - any source drift, target drift, foreign content, environment
             drift, KDA-target drift, or non-empty output directory

Refusals exit 9 with a REFUSE line on stderr, matching the house style.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


class Refusal(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"cadence contract missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"cadence contract unreadable: {exc}") from exc
    for key in ("modules", "import_owner", "kda_import_target", "environment", "output"):
        if key not in value:
            raise Refusal(f"cadence contract lacks {key}")
    return value


def check_environment(contract: dict, environ: dict[str, str]) -> None:
    for key, expected in contract["environment"].items():
        if environ.get(key) != expected:
            raise Refusal(f"Cadence environment drift: {key}={environ.get(key)!r} expected {expected!r}")


def verify_sources(modules_dir: Path, contract: dict) -> None:
    if modules_dir.is_symlink() or not modules_dir.is_dir():
        raise Refusal(f"module source directory missing or unsafe: {modules_dir}")
    expected = dict(contract["modules"])
    expected[contract["import_owner"]["file"]] = contract["import_owner"]["sha256"]
    observed = {path.name for path in modules_dir.iterdir()}
    missing = sorted(set(expected) - observed)
    if missing:
        raise Refusal(f"sealed module source missing: {','.join(missing)}")
    for name, digest in expected.items():
        if sha_file(modules_dir / name) != digest:
            raise Refusal(f"sealed module source hash drift: {name}")


def install_target(source: Path, target: Path) -> str:
    """Install one verified source to target; return fresh or already."""
    data = source.read_bytes()
    if target.is_symlink():
        raise Refusal(f"refusing symlinked target: {target}")
    if target.exists():
        if target.is_file() and sha_file(target) == hashlib.sha256(data).hexdigest():
            return "already"
        raise Refusal(f"target exists with drifted bytes: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "fresh"


def prepare_output(output: dict) -> None:
    out_dir = Path(output["out_dir"])
    graphs_dir = Path(output["graphs_dir"])
    if out_dir.is_symlink():
        raise Refusal(f"refusing symlinked output directory: {out_dir}")
    if out_dir.exists():
        entries = [path for path in out_dir.iterdir() if path != graphs_dir]
        if entries or (graphs_dir.is_dir() and any(graphs_dir.iterdir())):
            raise Refusal(
                f"output directory {out_dir} is not empty; a candidate with stale "
                "receipts must not serve again - construct a fresh container")
        if not graphs_dir.is_dir():
            raise Refusal(f"output directory lacks {graphs_dir}")
        return
    graphs_dir.mkdir(parents=True, exist_ok=True)


def check_kda_original(site: Path, target: dict) -> None:
    path = site / target["site_relative_path"]
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"KDA import target missing or unsafe: {path}")
    observed = sha_file(path)
    if observed != target["required_sha256"]:
        raise Refusal(
            f"KDA import target hash drift: {path} is {observed[:16]}, "
            f"expected {target['required_sha256'][:16]}; the bootstrap would refuse to shadow it")


def run(contract_path: Path, modules_dir: Path, b45_root: Path, site: Path,
        environ: dict[str, str]) -> tuple[str, int, str]:
    contract = load_contract(contract_path)
    check_environment(contract, environ)
    verify_sources(modules_dir, contract)
    check_kda_original(site, contract["kda_import_target"])
    prepare_output(contract["output"])

    if b45_root.is_symlink():
        raise Refusal(f"refusing symlinked module root: {b45_root}")
    if b45_root.exists():
        foreign = sorted(path.name for path in b45_root.iterdir()
                         if path.name not in contract["modules"])
        if foreign:
            raise Refusal(f"foreign content in {b45_root}: {','.join(foreign)}")
    states = []
    for name in sorted(contract["modules"]):
        states.append(install_target(modules_dir / name, b45_root / name))
    owner = contract["import_owner"]
    states.append(install_target(modules_dir / owner["file"], site / owner["file"]))
    state = "fresh" if "fresh" in states else "already"
    return state, len(contract["modules"]), contract["output"]["out_dir"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--b45-root", type=Path, default=Path("/opt/b45"))
    parser.add_argument("--site", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages"))
    args = parser.parse_args()
    try:
        contract_path = args.recipe_root / "config" / "cadence-contract.json"
        state, module_count, out_dir = run(
            contract_path, args.recipe_root / "modules", args.b45_root,
            args.site, dict(os.environ))
        rank = os.environ.get("NODE_RANK", "?")
        print(f"B45_MODULE_INSTALL_PASS rank={rank} state={state} modules={module_count} "
              f"pth=1 out={out_dir}", flush=True)
        return 0
    except Refusal as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == "__main__":
    diagnostics.install_exception_hook()
    raise SystemExit(main())
