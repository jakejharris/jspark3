#!/usr/bin/env python3
"""Offline core checks for the v1.1 Cadence wiring and the long-context witness.

Covers the core pass (measured module bytes, the fail-closed installer states
(fresh / already-applied / refusals), the launch payload wiring in fleetctl and
container_entry.sh, and the long-context witness gates) plus the F-1 verifier
capture-gate fix (Cadence capture evidence replaces the absent stock bar).
No hosts, no containers, no network beyond a guaranteed-refused localhost probe.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "recipe" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fleetctl  # noqa: E402
import install_b45_modules as installer  # noqa: E402
import long_context_witness as witness  # noqa: E402

# Independent copy of the measured release bundle's module-hashes.json.
MEASURED = {
    "b45_bootstrap.py": "fa72748f0e7c914d6486e2c2728e1f10c6b457cc1f215081d935bc8bbede3d07",
    "b45_graphs.py": "dce824588bba64a462cf9d2f04a80e1c39a60fe7826dd311157873cbcacaba3c",
    "b5_controller.py": "3d4ded0f4d03b6707f7b6e6f5df0374d6350100148719c8c11c91d4a2c25d5b3",
    "b5_prefix_verify.py": "da8ea1a779fad08459632a9ec141de540b1f21a255a5af3e522f804108dcb0a0",
    "kda_mixed_output_blocks.py": "db6d60f0ac99d3cc23d5d0b6194a557779b82f2b34132c4dba11f28098fda61f",
    "zzz_b45.pth": "eea018d5bfee8fdc28e6470f650b8f4adaa5ec4cc4f8ac86b3e8849368c9fdeb",
}
KDA_ORIGINAL = "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_module_bytes() -> None:
    contract = json.loads((ROOT / "recipe/config/cadence-contract.json").read_text())
    modules_dir = ROOT / "recipe/modules"
    for name, digest in MEASURED.items():
        path = modules_dir / name
        expect(path.is_file(), f"missing packaged module {name}")
        observed = sha(path)
        expect(observed == digest, f"{name} bytes differ from the measured release")
        if name == "zzz_b45.pth":
            expect(contract["import_owner"]["sha256"] == digest, "contract pth hash drift")
        else:
            expect(contract["modules"].get(name) == digest, f"contract hash drift: {name}")
    expect(contract["kda_import_target"]["required_sha256"] == KDA_ORIGINAL,
           "contract KDA import target drift")
    expect(set(contract["modules"]) | {"zzz_b45.pth"} == set(MEASURED),
           "packaged module set differs from the measured six")
    print("PASS module-bytes six packaged files equal the measured release bytes")


def build_fake_tree(base: Path) -> dict:
    """A self-consistent contract + sources; same code paths as production."""
    modules = base / "modules"
    modules.mkdir(parents=True)
    hashes = {}
    for name in ("mod_a.py", "mod_b.py"):
        (modules / name).write_bytes(f"# sealed {name}\n".encode())
        hashes[name] = sha(modules / name)
    (modules / "owner.pth").write_bytes(b"import sys; sys.path.insert(0, '/opt/x')\n")
    pth = sha(modules / "owner.pth")
    kda_rel = "pkg/kda_file.py"
    kda_path = base / "site" / kda_rel
    kda_path.parent.mkdir(parents=True)
    kda_path.write_bytes(b"# original kda bytes\n")
    contract = {
        "modules": hashes,
        "import_owner": {"file": "owner.pth", "sha256": pth, "site": str(base / "site")},
        "kda_import_target": {"site_relative_path": kda_rel, "required_sha256": sha(kda_path)},
        "environment": {"B45_COMBINED": "1", "B4_CAPTURE_ORDER": '["bf16_0","int8_0"]'},
        "output": {"out_dir": str(base / "out"), "graphs_dir": str(base / "out" / "graphs")},
    }
    contract_path = base / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return {"contract": contract_path, "modules": modules, "site": base / "site",
            "b45": base / "b45", "env": {"B45_COMBINED": "1", "B4_CAPTURE_ORDER": '["bf16_0","int8_0"]'}}


def run_installer(tree: dict, env: dict | None = None):
    return installer.run(tree["contract"], tree["modules"], tree["b45"], tree["site"],
                         env if env is not None else tree["env"])


def expect_refusal(tree: dict, message: str, **kwargs) -> None:
    try:
        run_installer(tree, **kwargs)
    except installer.Refusal as exc:
        expect(message in str(exc), f"refusal lacked expected detail {message!r}: {exc}")
    else:
        raise AssertionError(f"expected refusal containing {message!r}")


def check_installer_states() -> None:
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        tree = build_fake_tree(base)
        state, count, out_dir = run_installer(tree)
        expect(state == "fresh" and count == 2 and out_dir == str(base / "out"),
               f"fresh install returned {state}/{count}/{out_dir}")
        expect((tree["b45"] / "mod_a.py").stat().st_mode & 0o222 == 0,
               "installed module is writable")
        state, _, _ = run_installer(tree)
        expect(state == "already", f"re-install returned {state}")
        # Drifted installed target must refuse (targets install read-only).
        (tree["b45"] / "mod_b.py").chmod(0o644)
        (tree["b45"] / "mod_b.py").write_bytes(b"# tampered\n")
        expect_refusal(tree, "drifted bytes")
        (tree["b45"] / "mod_b.py").unlink()
        # Foreign content must refuse.
        (tree["b45"] / "rogue.py").write_bytes(b"x")
        expect_refusal(tree, "foreign content")
        (tree["b45"] / "rogue.py").unlink()
        # Drifted source must refuse.
        (tree["modules"] / "mod_a.py").write_bytes(b"# mutated source\n")
        expect_refusal(tree, "source hash drift")
        (tree["modules"] / "mod_a.py").write_bytes(f"# sealed mod_a.py\n".encode())
        # Wrong KDA original must refuse.
        (tree["site"] / "pkg/kda_file.py").write_bytes(b"# different build\n")
        expect_refusal(tree, "KDA import target hash drift")
        (tree["site"] / "pkg/kda_file.py").write_bytes(b"# original kda bytes\n")
        # Environment drift must refuse.
        expect_refusal(tree, "environment drift", env={"B45_COMBINED": "0"})
        # Non-empty output (stale receipts) must refuse.
        (base / "out/graphs/steps-1.jsonl").write_text("{}\n")
        expect_refusal(tree, "must not serve again")
        print("PASS installer-states fresh/already/refusals (drift, foreign, source, KDA, env, stale output)")


def check_fleetctl_wiring() -> None:
    contract = json.loads((ROOT / "recipe/config/cadence-contract.json").read_text())
    expect(fleetctl.B45_MODULES == contract["modules"], "fleetctl module pins differ from contract")
    expect(fleetctl.B45_PTH_SHA256 == contract["import_owner"]["sha256"], "fleetctl pth pin drift")
    expect(fleetctl.B45_KDA_ORIGINAL_SHA256 == contract["kda_import_target"]["required_sha256"],
           "fleetctl KDA pin drift")
    expect(fleetctl.B45_ENV == contract["environment"], "fleetctl env differs from contract")
    values = {}
    for rank in range(3):
        values.update({
            f"JSPARK_SOCKET_IFNAME_{rank}": "mgmt0",
            f"JSPARK_HCAS_{rank}": "roce0,roce1",
            # TEST-NET-1 documentation addresses stand in for rank addresses.
            f"JSPARK_RANK{rank}_ADDR": f"192.0.2.{rank + 1}",
        })
    values.update({
        "JSPARK_IB_GID_INDEX": "3", "JSPARK_API_BIND": "0.0.0.0", "JSPARK_API_PORT": "8000",
        "JSPARK_MASTER_ADDR": "192.0.2.1", "JSPARK_MASTER_PORT": "29533",
        "JSPARK_WORK_ROOT": "/srv/work", "JSPARK_RECIPE_ROOT": "/srv/recipe",
        "JSPARK_FLY_ROOT": "/srv/fly", "JSPARK_MODEL_ROOT": "/srv/models",
    })
    env_list = fleetctl.rank_env(values, 0)
    for key, value in fleetctl.B45_ENV.items():
        expect(f"{key}={value}" in env_list, f"rank0 env lacks {key}={value}")
    argv = fleetctl.server_argv(values, 0)
    expect(["--max-logprobs", "-1"] == argv[-2:], f"server argv lacks trailing --max-logprobs -1: {argv[-4:]}")
    argv = fleetctl.container_argv(values, 1)
    joined = " ".join(argv)
    expect("jspark3.release=v1.1.0" in joined, "create payload lacks the v1.1.0 release label")
    expect("jspark3.b45=boot41" in joined, "create payload lacks the measured provenance label")
    print("PASS fleetctl-wiring env/flags/labels/contract pins in the create payload")


def check_entry_script() -> None:
    text = " ".join((SCRIPTS / "container_entry.sh").read_text().split())
    for key, value in fleetctl.B45_ENV.items():
        unquoted = "${" + key + ":-} != " + value
        quoted = "${" + key + ":-} != '" + value + "'"
        expect(unquoted in text or quoted in text, f"entry script lacks refusal literal for {key}")
    expect('"$recipe/scripts/install_b45_modules.py" --recipe-root "$recipe"' in text,
           "entry script never installs the cadence modules")
    subprocess.run(["bash", "-n", str(SCRIPTS / "container_entry.sh")], check=True)
    print("PASS entry-script B4+B5 environment drift refusals and installer invocation")


def check_witness() -> None:
    body_a, body_b = witness.payload_bytes(witness.build_payload()), witness.payload_bytes(witness.build_payload())
    expect(body_a == body_b, "witness payload generator is not deterministic")
    expect(hashlib.sha256(body_a).hexdigest() == witness.PINNED_PAYLOAD_SHA256, "pinned payload hash drift")
    needle = witness.CODE_WORD
    good = witness.evaluate(33000, 21, "stop", f"{needle}\n1", 32768)
    expect(good["pass"] and good["code_word_verbatim"], f"good case refused: {good}")
    expect(not witness.evaluate(32768, 21, "stop", needle, 32768)["pass"], "boundary 32768 must refuse")
    expect(not witness.evaluate(33000, 2, "stop", needle, 32768)["pass"], "two-token decode must refuse")
    expect(not witness.evaluate(33000, 21, "abort", needle, 32768)["pass"], "abort finish must refuse")
    expect(not witness.evaluate(33000, 21, "stop", "JSPARK3-WRONG-0000", 32768)["pass"], "wrong code word must refuse")
    expect(not witness.evaluate(33000, 21, "stop", "", 32768)["pass"], "empty content must refuse")
    raised = subprocess.run(
        [sys.executable, str(SCRIPTS / "long_context_witness.py"),
         "--base-url", "http://127.0.0.1:9", "--timeout", "2"],
        capture_output=True, text=True)
    expect(raised.returncode == 9 and raised.stdout == "" and raised.stderr.startswith("REFUSE:"),
           f"unreachable endpoint must refuse visibly: {raised.returncode} {raised.stdout!r}")
    weak = subprocess.run(
        [sys.executable, str(SCRIPTS / "long_context_witness.py"),
         "--base-url", "http://127.0.0.1:9", "--min-prompt-tokens", "1024"],
        capture_output=True, text=True)
    expect(weak.returncode == 9 and "hard 32768 floor" in weak.stderr,
           "floor-lowering must refuse")
    print("PASS witness determinism, pinned payload, boundary/multi-decode/needle gates, visible refusal")


def check_verify_capture_gate() -> None:
    """F-1 regression: verify's capture gate must accept a valid Cadence startup
    without the stock 'Capturing CUDA graphs (FULL) 5/5' bar (the byte-pinned B4
    capture cannot render it), while still refusing missing/drifted Cadence
    capture evidence and keeping every unrelated gate literal enforced.
    Log fixture lines replicate the archived measured-arm startup log
    (second serving start, rank0): dflash2 bar completes 5/5, the stock target
    bar never appears, shard bars and the B5 receipt are present. The evidence
    thresholds equal the pinned path's full-bank construction (16 receipts,
    8 serving dumps per rank, measured in the boot41/43 measured-arm startup
    receipts); a partial bank set or partial serving dumps must be refused.
    """
    logs = "\n".join([
        "(EngineCore pid=1) Loading safetensors checkpoint shards: 100%|120/120 [01:10<00:00, 12.1it/s]",
        "(EngineCore pid=1) Loading safetensors checkpoint shards: 100%|1/1 [00:00<00:00, 3.2it/s]",
        "(Worker_TP0_EP0 pid=1866) Capturing dflash2 CUDA graphs (FULL):   0%|  0/5 [00:00<?, ?it/s]",
        "(Worker_TP0_EP0 pid=1866) Capturing dflash2 CUDA graphs (FULL): 100%|5/5 [00:32<00:00, 6.47s/it]",
        "Application startup complete.",
        "B5_PREFIX_VERIFY_RECEIPT rank=0 T4_seed_ms=74.300 T7_seed_ms=92.530 narrow_width=3",
    ])
    expect("Capturing CUDA graphs (FULL)" not in logs, "fixture must be a valid Cadence log without the stock target bar")
    expect(not fleetctl.progress_complete(logs, "Capturing CUDA graphs (FULL)", 5),
           "the absent stock bar must not satisfy anything")

    def rank_row(capture_ok=True, receipts=16, serving=8):
        return {"rank": 0, "cadence_b45": {"modules_verified": 5, "kda_original_untouched": True,
                "execution_receipts": 9, "capture_receipts": receipts if capture_ok else 0,
                "capture_dots_intact": capture_ok, "serving_graph_dumps": serving}}

    valid = fleetctl.load_gate(logs, [rank_row() for _ in range(3)])
    expect(all(valid.values()) and valid["cadence_capture_evidence"] and valid["draft_graphs_5"],
           f"a valid Cadence startup must pass the full load gate: {valid}")

    exec_view = {"modules": {}, "pth_sha256": "x", "kda_original_sha256": "y",
                 "b45_out_entries": ["graphs/activation-1.jsonl"],
                 "capture_receipts": 16, "capture_dots_intact": True, "serving_graph_dumps": 8}
    expect(fleetctl.cadence_capture_evidence_ok(exec_view), "a healthy in-container capture report must pass")
    for drift in ({"capture_receipts": 0}, {"capture_dots_intact": False},
                  {"serving_graph_dumps": 0}, {"capture_receipts": "12"},
                  {"capture_dots_intact": None}, {"capture_receipts": 15},
                  {"serving_graph_dumps": 7}):
        expect(not fleetctl.cadence_capture_evidence_ok({**exec_view, **drift}),
               f"drifted capture evidence must fail: {drift}")

    for name, rows in (("missing receipts", [rank_row(capture_ok=False)] * 3),
                       ("hash-drifted dumps", [rank_row(capture_ok=False, receipts=16)] * 3),
                       ("no serving dumps", [rank_row(serving=0)] * 3),
                       ("incomplete receipts 15/16", [rank_row(receipts=15)] * 3),
                       ("incomplete serving dumps 7/8", [rank_row(serving=7)] * 3),
                       ("cadence block absent", [{"rank": r} for r in range(3)]),
                       ("empty runtime", [])):
        gate = fleetctl.load_gate(logs, rows)
        expect(not gate["cadence_capture_evidence"] and not all(gate.values()),
               f"{name} must fail the capture gate")

    for needle, key in (("Application startup complete.", "startup_complete"),
                        ("Capturing dflash2 CUDA graphs (FULL)", "draft_graphs_5"),
                        ("B5_PREFIX_VERIFY_RECEIPT", "b5_controller_calibrated"),
                        ("120/120", "target_shards_120"),
                        ("1/1", "draft_shards_1")):
        reduced = fleetctl.load_gate(logs.replace(needle, ""), [rank_row() for _ in range(3)])
        expect(reduced[key] is False, f"removing {needle!r} must fail {key}")
        expect(reduced["cadence_capture_evidence"] is True,
               f"removing {needle!r} must not touch the capture evidence condition")
    expect(not fleetctl.progress_complete("Capturing dflash2 CUDA graphs (FULL):  60%|3/5", "Capturing dflash2 CUDA graphs (FULL)", 5),
           "an incomplete progress bar must stay refused")
    print("PASS verify-capture-gate Cadence evidence replaces the absent stock bar; drift and unrelated gates still refused")


def check_dry_runs() -> None:
    for script, args, needles in (
        ("start", ["--dry-run"], ["B45_COMBINED=1", "jspark3.b45=boot41", "--max-logprobs -1"]),
        ("verify", ["--dry-run"], ["long_context_witness.py"]),
    ):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "fleetctl.py"), script,
             "--env-file", str(ROOT / "recipe/.env.example"), *args],
            capture_output=True, text=True)
        expect(result.returncode == 0, f"{script} --dry-run failed: {result.stderr}")
        for needle in needles:
            expect(needle in result.stdout, f"{script} dry-run lacks {needle!r}")
    for source in ("fleetctl.py", "install_b45_modules.py", "long_context_witness.py"):
        py_compile.compile(str(SCRIPTS / source), doraise=True)
    print("PASS dry-runs render cadence env/label/installer and the long-context witness")


def check_validator_boundaries() -> None:
    """Focused checks for the finalization-time validator logic changes.

    The .git administrative pointer exclusion must be exactly that (a pointer
    file), the .pth payload policy must admit only small text path files, and
    the measured-construction boot labels must survive only as the exact
    permitted literals in their known files.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    import validate_release as vr  # noqa: E402

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        (base / "src.py").write_text("# source\n")
        (base / ".git").write_text("gitdir: /elsewhere/repo/.git/worktrees/x\n")
        names = {p.name for p in vr.tree_files(base)}
        expect(".git" not in names and "src.py" in names,
               f"git administrative pointer not excluded: {names}")
        (base / ".git").write_text("not a pointer\n")
        names = {p.name for p in vr.tree_files(base)}
        expect(".git" in names, "a non-pointer file named .git must stay in the walk and be scanned")

    real_pth = ROOT / "recipe/modules/zzz_b45.pth"
    expect(vr.pth_payload_allowed(real_pth), "shipped path-configuration file must be allowed")
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        binary = base / "weights.pth"
        binary.write_bytes(b"\x00" * 16)
        expect(not vr.pth_payload_allowed(binary), "binary .pth payload must refuse")
        oversized = base / "big.pth"
        oversized.write_text("x" * 4097)
        expect(not vr.pth_payload_allowed(oversized), "oversized .pth payload must refuse")

    permitted_files = {
        "recipe/config/cadence-contract.json",
        "recipe/config/profile.json",
        "recipe/modules/b5_prefix_verify.py",
        "recipe/scripts/fleetctl.py",
        "recipe/scripts/install_b45_modules.py",
        "tools/test_v11_offline.py",
        "tools/validate_release.py",
    }
    boot = re.compile(rb"(?i)(?<![a-z0-9])boot\d")
    seen: dict[str, int] = {}
    for path in vr.tree_files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        spans: list[tuple[int, int]] = []
        for literal in vr.PERMITTED_BOOT_LITERALS:
            start = 0
            while True:
                index = data.find(literal, start)
                if index < 0:
                    break
                spans.append((index, index + len(literal)))
                start = index + len(literal)
        for match in boot.finditer(data):
            expect(any(lo <= match.start() < hi for lo, hi in spans),
                   f"boot label outside the permitted literals in {rel}: {match.group(0)!r}")
            seen[rel] = seen.get(rel, 0) + 1
    expect(set(seen) <= permitted_files,
           f"boot labels leaked into unexpected files: {sorted(set(seen) - permitted_files)}")
    print("PASS validator-boundaries git-pointer exclusion, .pth policy, narrow boot-label permission")


def main() -> int:
    checks = (check_module_bytes, check_installer_states, check_fleetctl_wiring,
              check_entry_script, check_witness, check_verify_capture_gate, check_dry_runs,
              check_validator_boundaries)
    try:
        for check in checks:
            check()
    except AssertionError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 9
    print(f"PASS all core checks ({len(checks)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
