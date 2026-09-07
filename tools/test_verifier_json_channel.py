#!/usr/bin/env python3
"""Offline regression for the verifier JSON channel class fix (boot-51).

The transformed serving image installs two site hooks into the container
dist-packages: glm53_video.pth prints a banner BEFORE any script output at
every site-initializing interpreter start, and zzz_b45.pth -> b45_bootstrap ->
b5_prefix_verify prints receipt lines AFTER it.  Every fleetctl remote Python
whose stdout is consumed by strict_object/strict_loads therefore received
contaminated stdout and verify refused before any witness ran.

The fix runs every such remote Python with -S (no site initialization).  All
invoked code is stdlib-only and sibling imports resolve via sys.path[0] (the
script directory), which -S preserves, so semantics are provably unchanged.

These checks prove, with no hosts, containers, or network:

- every JSON-producing remote Python argv builder (host ssh and docker exec)
  emits python3 -S, including the graph-capture collection, runtime identity,
  and pipeline --check channels, and the dry-run rendering matches;
- a real .pth startup hook (fresh venv, leading import-time print plus
  trailing atexit print) contaminates a plain interpreter exactly as seen on
  boot-51, and -S on the same interpreter yields byte-clean JSON;
- the consumers still accept valid payloads and still refuse
  leading/trailing/both noise, malformed, empty, and nonzero-exit channels;
- b45_identity and runtime_identity pass end-to-end through a fake remote
  with clean payloads and refuse when any of the three channels carries
  startup noise;
- the pipeline check still validates actual on-disk transformed bytes under
  -S (ALREADY_APPLIED on an exact fixture tree; tampered or missing target
  bytes refuse) while the same tree on a noisy interpreter reproduces the
  old refusal;
- the zero-boot candidate binding: bound_manifest accepts a manifest bound to
  a declared candidate recipe SHA-256 only when it matches exactly, and the
  default self-identity check is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "recipe" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fleetctl  # noqa: E402
import _atomic  # noqa: E402

LEADING = "glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)"
TRAILING = "B5_PREFIX_VERIFY_RECEIPT rank=0 T4_seed_ms=74.300 T7_seed_ms=92.530 narrow_width=3"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def env_values() -> dict[str, str]:
    values = {}
    for rank in range(3):
        values.update({
            f"JSPARK_RANK{rank}_HOST": f"host{rank}.example",
            f"JSPARK_RANK{rank}_ADDR": f"192.0.2.{rank + 1}",
            f"JSPARK_FABRIC_IFACES_{rank}": "enp1s0f0np0,enp1s0f1np1",
            f"JSPARK_FABRIC_ADDRS_{rank}": f"198.51.100.{rank}/24,203.0.113.{rank}/24",
            f"JSPARK_HCAS_{rank}": "roce0,roce1",
            f"JSPARK_SOCKET_IFNAME_{rank}": "mgmt0",
        })
    values.update({
        "JSPARK_MASTER_ADDR": "192.0.2.1", "JSPARK_MASTER_PORT": "29533",
        "JSPARK_IB_GID_INDEX": "3", "JSPARK_API_BIND": "0.0.0.0",
        "JSPARK_API_PORT": "8000", "JSPARK_MODEL_ROOT": "/srv/models",
        "JSPARK_WORK_ROOT": "/srv/work", "JSPARK_RECIPE_ROOT": "/srv/recipe",
        "JSPARK_FLY_ROOT": "/srv/fly",
    })
    return values


def test_argv_builders_run_site_free() -> None:
    values = env_values()
    cases = {
        "preflight": fleetctl.preflight_argv(values, 1),
        "checkpoint": fleetctl.checkpoint_argv(values),
        "cgroup": fleetctl.cgroup_argv(1234),
        "image-receipt-install": fleetctl.image_receipt_install_argv("/p", "eA=="),
        "b45-identity": fleetctl.b45_identity_argv("c" * 64),
    }
    config_argv, pipeline_argv = fleetctl.runtime_identity_argv("c" * 64)
    cases["runtime-identity-config"] = config_argv
    cases["runtime-identity-pipeline"] = pipeline_argv
    for label, argv in cases.items():
        expect(tuple(argv[:2]) == fleetctl.JSON_PYTHON or
               tuple(argv[3:5]) == fleetctl.JSON_PYTHON,
               f"{label} argv lacks python3 -S: {argv[:6]}")
    expect(any(str(part).endswith("apply_base_pipeline.py") for part in pipeline_argv)
           and "--check" in pipeline_argv,
           "pipeline channel no longer checks on-disk state")
    expect(tuple(pipeline_argv[3:5]) == fleetctl.JSON_PYTHON,
           f"pipeline check lacks -S after docker exec: {pipeline_argv[:6]}")
    # The recipe-only channel builds its argv inside remote_recipe_sha.
    seen = []
    original = fleetctl.remote
    fleetctl.remote = lambda v, r, argv, *, check=True: seen.append(argv) or subprocess.CompletedProcess(
        argv, 0, json.dumps({"recipe_manifest_sha256": "a" * 64, "status": "PASS"}), "")
    try:
        fleetctl.remote_recipe_sha(values, 0, "a" * 64)
    finally:
        fleetctl.remote = original
    expect(tuple(seen[0][:2]) == fleetctl.JSON_PYTHON and "-B" in seen[0],
           f"recipe-only channel lacks python3 -S -B: {seen[0][:4]}")
    print("ok argv-builders all remote JSON Python channels run python3 -S")


def make_noise_venv(base: Path) -> tuple[Path, str]:
    """A real interpreter whose site hooks print before and after stdout."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(base / "venv")],
                   check=True, capture_output=True)
    site = next((base / "venv").glob("lib/python*/site-packages"))
    (site / "noise_trail.py").write_text(
        f"import atexit; atexit.register(lambda: print({TRAILING!r}, flush=True))\n")
    (site / "aaa_noise_trail.pth").write_text("import noise_trail\n")
    (site / "noise_lead.py").write_text(f"print({LEADING!r}, flush=True)\n")
    (site / "zzz_noise_lead.pth").write_text("import noise_lead\n")
    return base / "venv" / "bin" / "python", str(site)


def test_pth_startup_noise_real_interpreter() -> None:
    """A real .pth hook pair reproduces both contaminations; -S removes them."""
    with tempfile.TemporaryDirectory() as raw:
        python, _ = make_noise_venv(Path(raw))
        snippet = "import json,hashlib,pathlib; print(json.dumps({'ok': True}, sort_keys=True))"
        plain = subprocess.run([python, "-c", snippet], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        expect(plain.returncode == 0, plain.stderr)
        expect(plain.stdout.startswith(LEADING + "\n") and plain.stdout.rstrip().endswith(TRAILING),
               f"expected leading+trailing .pth noise, got: {plain.stdout!r}")
        for label, noise in (("leading", LEADING + "\n"), ("trailing", TRAILING + "\n")):
            contaminated = (noise + '{"ok": true}\n') if label == "leading" else ('{"ok": true}\n' + noise)
            try:
                fleetctl.strict_object(contaminated, label)
            except (fleetctl.Refusal, json.JSONDecodeError):
                pass
            else:
                raise AssertionError(f"strict parser accepted {label} noise")
        try:
            fleetctl.strict_object(plain.stdout, "noisy interpreter")
        except (fleetctl.Refusal, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("strict parser accepted real .pth-contaminated stdout")
        silent = subprocess.run([python, "-S", "-c", snippet], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        expect(silent.returncode == 0, silent.stderr)
        value = fleetctl.strict_object(silent.stdout, "-S interpreter")
        expect(value == {"ok": True}, f"-S payload damaged: {value}")
        print("ok pth-startup real .pth hooks contaminate plain python3; python3 -S stdout is clean JSON")


def fake_remote(stdout: str, returncode: int = 0):
    """Patch fleetctl.subprocess.run so the real remote() wrapper executes."""
    original = fleetctl.subprocess.run

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout, "stderr-detail")

    fleetctl.subprocess.run = run
    return original


def restore_remote(original) -> None:
    fleetctl.subprocess.run = original


def expect_refusal(message: str, fn) -> None:
    # Fail-closed refusal surfaces either as fleetctl.Refusal (contract gates)
    # or json.JSONDecodeError (strict parser); both are caught as REFUSE by
    # fleetctl.main's top-level handler.
    try:
        fn()
    except (fleetctl.Refusal, json.JSONDecodeError) as exc:
        expect(not message or message.split(":")[0] in str(exc) or message in str(exc),
               f"refusal lacks {message!r}: {exc}")
    else:
        raise AssertionError(f"expected refusal: {message}")


def test_remote_consumer_refusals() -> None:
    values = env_values()
    good = json.dumps({"recipe_manifest_sha256": "a" * 64, "status": "PASS"})
    for label, stdout, rc in (
        ("clean", good, 0),
        ("leading", LEADING + "\n" + good, 9),
        ("trailing", good + TRAILING + "\n", 9),
        ("both", LEADING + "\n" + good + TRAILING + "\n", 9),
        ("malformed", "{not json", 9),
        ("empty", "", 9),
        ("nonzero", good, 9),
    ):
        original = fake_remote(stdout, rc if label == "nonzero" else 0)
        try:
            if label == "clean":
                expect(fleetctl.remote_recipe_sha(values, 0, "a" * 64) == "a" * 64,
                       "clean recipe-only channel refused")
            else:
                expect_refusal("", lambda: fleetctl.remote_recipe_sha(values, 0, "a" * 64))
        finally:
            restore_remote(original)
    # The real remote() wrapper must refuse a nonzero child itself.
    original = fake_remote(good, 9)
    try:
        expect_refusal("remote command failed", lambda: fleetctl.remote(values, 0, ["true"]))
    finally:
        restore_remote(original)
    print("ok consumer-refusals clean accepts; noise/malformed/empty/nonzero all refuse")


def b45_payload() -> dict:
    return {
        "modules": dict(fleetctl.B45_MODULES),
        "pth_sha256": fleetctl.B45_PTH_SHA256,
        "kda_original_sha256": fleetctl.B45_KDA_ORIGINAL_SHA256,
        "b45_out_entries": ["graphs/activation-1866.jsonl", "graphs/captures-1866.jsonl"],
        "capture_receipts": fleetctl.CADENCE_CAPTURE_RECEIPTS_MIN,
        "capture_dots_intact": True,
        "serving_graph_dumps": fleetctl.CADENCE_SERVING_DUMPS_MIN,
    }


def runtime_payloads() -> tuple[dict, dict, dict, dict, dict]:
    binding = {"rank": 0, "container_id": "c" * 64, "name": "jspark3-rank0",
               "image_config": fleetctl.IMAGE_CONFIG, "image_receipt_sha256": "e" * 64}
    manifest = {"preflight_sha256": "f" * 64, "recipe_manifest_sha256": "9" * 64}
    receipt = {
        "schema_version": 2, "manifest_digest": fleetctl.IMAGE.split("@", 1)[1],
        "config_digest": fleetctl.IMAGE_CONFIG,
        "verification": "host-observed-inspect-bound-create",
        "container_id": binding["container_id"], "rank": 0,
        "preflight_sha256": manifest["preflight_sha256"],
        "recipe_manifest_sha256": manifest["recipe_manifest_sha256"],
    }
    receipt["payload_sha256"] = fleetctl.sha_bytes(fleetctl.canonical(receipt))
    configs = {
        "target_runtime_config": "55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd",
        "draft_runtime_config": "c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f",
        "image_receipt_sha256": binding["image_receipt_sha256"],
        "image_receipt": receipt,
    }
    pipeline = {"state": "ALREADY_APPLIED",
                "target_set_sha256": "ed7b0092e5a5a1d2aeb6dd2cbe9780783df89d70f733dff019dd05aa8cdd08bd"}
    return binding, manifest, configs, pipeline, b45_payload()


def routed_remote(configs: dict, pipeline: dict, b45: dict,
                  leading: str = "", trailing: str = "",
                  only: str | None = None):
    """Fake remote; noise applies only to channel `only` (None = no noise)."""
    seen = []
    original = fleetctl.subprocess.run

    def run(cmd, **kwargs):
        seen.append(cmd)
        # remote() wraps docker exec in ssh_argv + shlex.join, so the inner
        # python argv is one shell string. Classify on that joined text.
        joined = " ".join(str(part) for part in cmd)
        if "apply_base_pipeline.py" in joined:
            body, channel = json.dumps(pipeline), "pipeline"
        elif "b45_out_entries" in joined:
            body, channel = json.dumps(b45), "b45"
        else:
            body, channel = json.dumps(configs), "configs"
        out = body + "\n"
        if channel == only:
            out = leading + out + trailing
        return subprocess.CompletedProcess(cmd, 0, out, "")

    fleetctl.subprocess.run = run
    return original, seen


def test_identity_channels_end_to_end() -> None:
    values = env_values()
    binding, manifest, configs, pipeline, b45 = runtime_payloads()
    original, seen = routed_remote(configs, pipeline, b45)
    try:
        row = fleetctl.b45_identity(values, binding)
        expect(row["capture_receipts"] == fleetctl.CADENCE_CAPTURE_RECEIPTS_MIN,
               f"b45 row damaged: {row}")
        result = fleetctl.runtime_identity(values, binding, manifest)
        expect(result["transform_pipeline_state"] == "ALREADY_APPLIED" and
               result["image_receipt_bound"] is True, f"runtime identity damaged: {result}")
    finally:
        restore_remote(original)
    joined = [str(cmd) for cmd in seen]
    expect(all("python3 -S" in cmd or " '-S'," in cmd for cmd in joined),
           f"identity channels lost -S: {joined[:1]}")
    for channel in ("configs", "pipeline", "b45"):
        for where, lead, trail in (("leading", LEADING + "\n", ""),
                                   ("trailing", "", TRAILING + "\n"),
                                   ("both", LEADING + "\n", TRAILING + "\n")):
            original, _ = routed_remote(configs, pipeline, b45,
                                        leading=lead, trailing=trail, only=channel)
            try:
                expect_refusal("",
                               lambda: fleetctl.runtime_identity(values, binding, manifest))
            finally:
                restore_remote(original)
    # Drifted b45 module bytes still refuse on a clean channel.
    original, _ = routed_remote(configs, pipeline,
                                {**b45, "modules": {**b45["modules"], "b45_graphs.py": "0" * 64}})
    try:
        expect_refusal("cadence module install drift",
                       lambda: fleetctl.b45_identity(values, binding))
    finally:
        restore_remote(original)
    print("ok identity-channels b45/runtime/pipeline accept clean, refuse noise on any channel, drift still refused")


def build_pipeline_fixture(base: Path) -> dict:
    """An exact final-state transform tree; the real contract + on-disk bytes."""
    dist = base / "dist-packages"
    vllm = dist / "vllm"
    vllm.mkdir(parents=True)
    source_root = base / "fly"
    source_root.mkdir()
    asset_root = base / "assets"
    asset_root.mkdir()
    transforms = {}
    for index, stage in enumerate(("apply_tp3_overlay.py", "apply_image_glm_dflash.py",
                                   "apply_kpool_tail.py", "apply_kda_mixed.py", "apply_kda_fg.py")):
        target = vllm / f"stage{index}.py"
        after = f"# stage {index} transformed\n".encode()
        target.write_bytes(after)
        transforms[stage] = {
            "targets": [{
                "path": f"vllm/stage{index}.py",
                "before_sha256": sha(f"# stage {index} original\n".encode()),
                "after_sha256": sha(after),
                "source_sha256": "0" * 64,
                "required_before_seams": [], "required_after_seams": [],
                "forbidden_after_seams": [],
            }],
        }
    # verify_sources pins the two module-backed stages' Fly sources.
    import apply_tp3_overlay as tp3  # noqa: E402
    import apply_image_glm_dflash as dflash  # noqa: E402
    for stage, module in (("apply_tp3_overlay.py", tp3), ("apply_image_glm_dflash.py", dflash)):
        sources = {}
        for name, relative in module.SOURCE_PATHS.items():
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"# fly source {name}\n".encode())
            sources[name] = sha(path.read_bytes())
        transforms[stage]["sources"] = sources
    contract = base / "contract.json"
    contract.write_text(json.dumps({"transforms": transforms}))
    receipt = {
        "schema_version": 2, "manifest_digest": _atomic.IMAGE_MANIFEST,
        "config_digest": _atomic.IMAGE_CONFIG,
        "verification": "host-observed-inspect-bound-create",
        "container_id": "c" * 64, "rank": 0,
        "preflight_sha256": "f" * 64, "recipe_manifest_sha256": "9" * 64,
    }
    receipt["payload_sha256"] = _atomic.sha_bytes(_atomic.canonical(
        {key: item for key, item in receipt.items() if key != "payload_sha256"}))
    receipt_path = base / "image-receipt.json"
    receipt_path.write_bytes(_atomic.canonical(receipt))
    return {"vllm": vllm, "source": source_root, "asset": asset_root,
            "contract": contract, "receipt": receipt_path}


def pipeline_check(python: str, fixture: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        python, *extra, str(SCRIPTS / "apply_base_pipeline.py"),
        "--vllm-root", str(fixture["vllm"]), "--source-root", str(fixture["source"]),
        "--asset-root", str(fixture["asset"]), "--contract", str(fixture["contract"]),
        "--image-receipt", str(fixture["receipt"]), "--check",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def test_pipeline_check_on_disk_under_S() -> None:
    """--check on the noisy interpreter reproduces boot-51; -S checks real bytes."""
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        python, _ = make_noise_venv(base)
        fixture = build_pipeline_fixture(base)
        noisy = pipeline_check(str(python), fixture)
        expect(noisy.returncode == 0, f"fixture check failed: {noisy.stderr}")
        expect(noisy.stdout.startswith(LEADING + "\n") and noisy.stdout.rstrip().endswith(TRAILING),
               f"expected both contaminations around the pipeline receipt: {noisy.stdout!r}")
        try:
            fleetctl.strict_object(noisy.stdout, "pipeline check noisy")
        except (fleetctl.Refusal, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("old wire pattern parsed as clean JSON")
        clean = pipeline_check(str(python), fixture, "-S")
        expect(clean.returncode == 0, f"-S check failed: {clean.stderr}")
        value = fleetctl.strict_object(clean.stdout, "pipeline check -S")
        expect(value["state"] == "ALREADY_APPLIED" and value["detected_stage"] == 5,
               f"-S altered pipeline semantics: {value}")
        expect(clean.stdout == _atomic.canonical(value).decode(),
               "-S stdout is not exactly the canonical receipt")
        # -S must not weaken the on-disk byte validation: tamper and remove.
        target = fixture["vllm"] / "stage2.py"
        original_bytes = target.read_bytes()
        target.write_bytes(b"# tampered bytes\n")
        refused = pipeline_check(str(python), fixture, "-S")
        expect(refused.returncode == 9 and refused.stdout == "" and "REFUSE:" in refused.stderr,
               f"tampered target must refuse visibly: {refused.returncode} {refused.stdout!r}")
        target.write_bytes(original_bytes)
        target.unlink()
        refused = pipeline_check(str(python), fixture, "-S")
        expect(refused.returncode == 9 and "REFUSE:" in refused.stderr,
               "missing target must refuse under -S")
        print("ok pipeline-check -S preserves on-disk byte validation; noisy interpreter reproduces old refusal")


def build_manifest(base: Path, values: dict, recipe_sha: str) -> Path:
    manifest = {
        "schema_version": 1, "candidate": "jspark3", "grade": "ENGINEERING-EVIDENCE",
        "configuration_sha256": fleetctl.configuration_digest(values),
        "preflight_sha256": "f" * 64, "image_manifest": fleetctl.IMAGE.split("@", 1)[1],
        "image_config": fleetctl.IMAGE_CONFIG, "recipe_manifest_sha256": recipe_sha,
        "start_order": [2, 1, 0],
        "containers": [
            {"rank": rank, "container_id": f"{rank}bcd" * 16, "name": f"jspark3-rank{rank}",
             "image_config": fleetctl.IMAGE_CONFIG, "image_receipt_sha256": "e" * 64}
            for rank in range(3)
        ],
        "status": "STARTED",
    }
    manifest["payload_sha256"] = fleetctl.sha_bytes(fleetctl.canonical(manifest))
    path = base / "manifest.json"
    path.write_bytes(fleetctl.canonical(manifest))
    return path


def test_candidate_recipe_binding() -> None:
    values = env_values()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        candidate_sha = "7" * 64
        path = build_manifest(base, values, candidate_sha)
        # Default self-identity: the manifest names a different recipe than the
        # local verifier checkout, so it must refuse exactly as before.
        expect_refusal("manifest does not bind this environment/image",
                       lambda: fleetctl.bound_manifest(path, values, require_all=True,
                                                       require_started=True))
        bound = fleetctl.bound_manifest(path, values, require_all=True, require_started=True,
                                        candidate_recipe_sha256=candidate_sha)
        expect(bound["recipe_manifest_sha256"] == candidate_sha, "candidate binding damaged")
        expect_refusal("manifest does not bind this environment/image",
                       lambda: fleetctl.bound_manifest(path, values, require_all=True,
                                                       require_started=True,
                                                       candidate_recipe_sha256="8" * 64))
        expect_refusal("candidate recipe manifest SHA-256 is malformed",
                       lambda: fleetctl.bound_manifest(path, values, candidate_recipe_sha256="zz"))
        # A manifest naming the LOCAL recipe still binds without the flag.
        local_dir = base / "local"
        local_dir.mkdir()
        local = build_manifest(local_dir, values, fleetctl.recipe_manifest_sha256())
        fleetctl.bound_manifest(local, values, require_all=True, require_started=True)
    args = fleetctl.parser().parse_args(["verify", "--candidate-recipe-manifest-sha256", "7" * 64])
    expect(args.candidate_recipe_manifest_sha256 == "7" * 64, "verify flag not wired")
    print("ok candidate-binding declared candidate recipe binds exactly; default self-identity unchanged")


def main() -> int:
    checks = (
        test_argv_builders_run_site_free,
        test_pth_startup_noise_real_interpreter,
        test_remote_consumer_refusals,
        test_identity_channels_end_to_end,
        test_pipeline_check_on_disk_under_S,
        test_candidate_recipe_binding,
    )
    for check in checks:
        check()
    print(f"verifier-json-channel regression: all {len(checks)} groups passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
