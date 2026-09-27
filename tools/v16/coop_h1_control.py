#!/usr/bin/env python3
"""Maintenance-only H1 raw-output sensitivity control for cooperative MoE.

This file is intentionally outside the recipe and native bundle.  It loads the
byte-pinned CUDA integration test, and in perturb mode replaces only that test's
comparison function.  The candidate tensor is cloned after the kernel returns;
the clone is perturbed immediately before the frozen tolerance comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


SCHEMA_VERSION = 1
PERTURB_TOLERANCE_MULTIPLIER = 8.0
TEST_FILE = "test_cuda_integration.py"
SERVING_ENV_KEYS = (
    "NODE_RANK",
    "JSPARK3_V16_COOP",
    "JSPARK3_V16_PROFILE",
    "JSPARK3_V14_EXPECT",
    "JSPARK_TARGET_RUNTIME",
    "JSPARK_DRAFT_RUNTIME",
    "MODEL_PATH",
    "DRAFT_PATH",
    "VLLM_HOST_IP",
    "VLLM_ENGINE_READY_TIMEOUT_S",
)


class ControlRefusal(RuntimeError):
    """The maintenance control is not in an exact, safe configuration."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def perturb_candidate(candidate, reference_peak: float, peak_tolerance: float):
    """Return a perturbed clone and metadata without changing ``candidate``.

    The injected delta is eight times the frozen global peak tolerance.  A
    passing unmodified output can differ from stock by at most one tolerance,
    leaving at least seven tolerances of separation even in the opposing-error
    direction at the selected element.
    """
    if not math.isfinite(reference_peak) or reference_peak < 0.0:
        raise ControlRefusal("reference peak must be finite and nonnegative")
    if not math.isfinite(peak_tolerance) or not 0.0 < peak_tolerance < 1.0:
        raise ControlRefusal("peak tolerance must be finite and in (0,1)")
    clone = getattr(candidate, "clone", None)
    if not callable(clone):
        raise ControlRefusal("candidate tensor does not provide clone()")
    perturbed = clone()
    view = getattr(perturbed, "view", None)
    if not callable(view):
        raise ControlRefusal("candidate tensor does not provide view()")
    flat = view(-1)
    if len(flat) < 1:
        raise ControlRefusal("candidate tensor is empty")
    # Match compare()'s clamped FP32 denominator for deliberately empty-local
    # outputs too. Zero-reference cases are part of real96/mixed at rows 1..4.
    reference_scale = max(reference_peak, 1e-30)
    delta = reference_scale * peak_tolerance * PERTURB_TOLERANCE_MULTIPLIER
    flat[0] = flat[0] + delta
    return perturbed, {
        "element": 0,
        "delta": delta,
        "reference_scale": reference_scale,
        "delta_over_reference_scale": delta / reference_scale,
        "tolerance_multiplier": PERTURB_TOLERANCE_MULTIPLIER,
    }


def verify_test_source(test_source: Path, source_manifest: Path) -> str:
    if test_source.is_symlink() or not test_source.is_file():
        raise ControlRefusal("CUDA integration test must be a regular non-symlink file")
    if source_manifest.is_symlink() or not source_manifest.is_file():
        raise ControlRefusal("source manifest must be a regular non-symlink file")
    manifest = json.loads(source_manifest.read_text())
    expected = manifest.get("files", {}).get(TEST_FILE)
    observed = digest(test_source)
    if not isinstance(expected, str) or observed != expected:
        raise ControlRefusal("CUDA integration test does not match SOURCE_MANIFEST.json")
    return observed


def verify_bundle(bundle: Path) -> dict[str, str]:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ControlRefusal("bundle must be a regular non-symlink directory")
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlRefusal("bundle manifest must be a regular non-symlink file")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    required = {"cooperative_moe.so", "runtime.py", "dispatch_policy.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ControlRefusal("bundle manifest is incomplete")
    root = bundle.resolve()
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ControlRefusal("bundle manifest contains malformed file identity")
        candidate = bundle / relative
        path = candidate.resolve()
        if candidate.is_symlink() or not path.is_relative_to(root) or not path.is_file():
            raise ControlRefusal(f"unsafe or missing bundle file: {relative}")
        if digest(path) != expected:
            raise ControlRefusal(f"bundle digest drift: {relative}")
    return {
        "manifest_sha256": digest(manifest_path),
        "native_sha256": files["cooperative_moe.so"],
        "policy_sha256": files["dispatch_policy.json"],
    }


def add_payload_hash(payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return result


def verify_payload_hash(record: dict) -> dict:
    payload = dict(record)
    claimed = payload.pop("payload_sha256", None)
    if claimed != hashlib.sha256(canonical(payload)).hexdigest():
        raise ControlRefusal("receipt payload hash mismatch")
    return payload


def write_receipt(path: Path, payload: dict) -> dict:
    if path.exists() or path.is_symlink():
        raise ControlRefusal(f"receipt already exists: {path}")
    parent = path.parent.resolve()
    if not parent.is_dir():
        raise ControlRefusal(f"receipt directory is missing: {parent}")
    record = add_payload_hash(payload)
    raw = canonical(record)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
    return record


def verify_baseline_receipt(path: Path, identity: dict) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ControlRefusal("perturb mode requires a regular baseline receipt")
    payload = verify_payload_hash(json.loads(path.read_text()))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "control": "coop-h1-raw-output",
        "mode": "baseline",
        "status": "PASS",
        "ep_rank": identity["ep_rank"],
        "qualification_geometry": identity["qualification_geometry"],
        "test_source_sha256": identity["test_source_sha256"],
        "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
        "native_sha256": identity["native_sha256"],
        "policy_sha256": identity["policy_sha256"],
        "thresholds": identity["thresholds"],
        "control_sha256": identity["control_sha256"],
        "exl3_sha256": identity["exl3_sha256"],
        "fatpath_sha256": identity["fatpath_sha256"],
        "detector_rejections": 0,
        "detector_outcome": "unmodified-pass",
        "serving_path_reachable": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ControlRefusal(f"baseline receipt identity mismatch: {key}")
    if payload.get("real_weight_comparisons") != identity["expected_real_weight_comparisons"]:
        raise ControlRefusal("baseline receipt has an incomplete real-weight comparison matrix")
    return payload


def refuse_serving_environment() -> None:
    if os.environ.get("GLM53_COOP_MAINTENANCE_TEST") != "1":
        raise ControlRefusal("GLM53_COOP_MAINTENANCE_TEST=1 is required")
    if os.environ.get("GLM53_COOP_QUALIFICATION") not in (None, ""):
        raise ControlRefusal("H1 control owns the qualification flag; external override refused")
    if os.environ.get("GLM53_COOP_GEOMETRY") not in (None, ""):
        raise ControlRefusal("H1 control refuses a forced geometry")
    present = [key for key in SERVING_ENV_KEYS if os.environ.get(key) not in (None, "")]
    if present:
        raise ControlRefusal("H1 control refuses serving environment keys: " + ",".join(present))


def load_gate(test_source: Path):
    spec = importlib.util.spec_from_file_location("jspark3_coop_h1_cuda_gate", test_source)
    if spec is None or spec.loader is None:
        raise ControlRefusal("could not load CUDA integration test")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict:
    refuse_serving_environment()
    source_sha = verify_test_source(args.test_source, args.source_manifest)
    bundle_identity = verify_bundle(args.bundle)
    os.environ["GLM53_COOP_BUNDLE"] = str(args.bundle.resolve())
    os.environ["GLM53_COOP_EP_RANK"] = str(args.rank)
    os.environ["GLM53_COOP_QUALIFICATION"] = "1"
    os.environ["GLM53_COOP_GEOMETRY"] = str(args.geometry)
    gate = load_gate(args.test_source)
    thresholds = {
        "peak": gate.PEAK_TOL,
        "row_peak": gate.ROW_PEAK_TOL,
        "rel_l2": gate.REL_L2_TOL,
    }
    if thresholds != {"peak": .003, "row_peak": .05, "rel_l2": .05}:
        raise ControlRefusal("CUDA gate changed the frozen numerical tolerances")
    if gate.SANITIZER:
        raise ControlRefusal("H1 control requires the full integration matrix")
    # Two patterns per row, initial output plus two changed-input comparisons.
    expected_comparisons = len(gate.ROWS) * 2 * 3
    identity = {
        "ep_rank": args.rank,
        "qualification_geometry": args.geometry,
        "test_source_sha256": source_sha,
        "bundle_manifest_sha256": bundle_identity["manifest_sha256"],
        "native_sha256": bundle_identity["native_sha256"],
        "policy_sha256": bundle_identity["policy_sha256"],
        "thresholds": thresholds,
        "control_sha256": digest(Path(__file__)),
        "exl3_sha256": gate.EXL3_SHA256,
        "fatpath_sha256": gate.FATPATH_SHA256,
        "expected_real_weight_comparisons": expected_comparisons,
    }
    observed = []
    if args.mode == "perturb":
        if args.baseline_receipt is None:
            raise ControlRefusal("perturb mode requires --baseline-receipt")
        verify_baseline_receipt(args.baseline_receipt, identity)
        original_compare = gate.compare

        def sensitivity_compare(actual, reference, label):
            if not label.startswith("real96/"):
                return original_compare(actual, reference, label)
            # A preceding boot's baseline receipt cannot certify this invocation.
            # Require its unmodified tensor to pass before testing the clone.
            original_compare(actual, reference, label)
            reference_peak = float(reference.detach().abs().max().item())
            perturbed, injection = perturb_candidate(
                actual, reference_peak, gate.PEAK_TOL
            )
            before = len(gate.records)
            control_label = label + "/h1-perturb"
            try:
                original_compare(perturbed, reference, control_label)
            except AssertionError:
                if len(gate.records) != before + 1:
                    raise ControlRefusal("perturbation failed outside the tolerance detector")
                result = gate.records[-1]
                if (
                    result.get("label") != control_label
                    or result.get("pass") is not False
                    or not math.isfinite(result.get("peak_rel", float("nan")))
                    or result["peak_rel"] <= gate.PEAK_TOL
                ):
                    raise ControlRefusal("perturbation did not fail the peak-relative tolerance")
                observed.append({
                    "label": control_label,
                    "peak_rel": result["peak_rel"],
                    "row_rel": result["row_rel"],
                    "rel_l2": result["rel_l2"],
                    **injection,
                })
                return None
            raise ControlRefusal("perturbed candidate escaped the frozen tolerance detector")

        gate.compare = sensitivity_compare
    elif args.baseline_receipt is not None:
        raise ControlRefusal("--baseline-receipt is valid only in perturb mode")

    gate.main()
    real_weight_records = [
        record for record in gate.records
        if str(record.get("label", "")).startswith("real96/")
        and not str(record["label"]).endswith("/h1-perturb")
    ]
    if len(real_weight_records) != expected_comparisons or any(
        record.get("pass") is not True for record in real_weight_records
    ):
        raise ControlRefusal("unmodified real-weight comparison matrix is incomplete or failed")
    if args.mode == "baseline":
        outcome = {
            "real_weight_comparisons": len(real_weight_records),
            "detector_rejections": 0,
            "detector_outcome": "unmodified-pass",
        }
    else:
        if not observed or len(observed) != len(real_weight_records):
            raise ControlRefusal("not every real-weight comparison exercised the perturbation")
        outcome = {
            "real_weight_comparisons": len(real_weight_records),
            "detector_rejections": len(observed),
            "detector_outcome": "expected-peak-tolerance-failure-observed",
            "minimum_observed_peak_rel": min(item["peak_rel"] for item in observed),
            "maximum_observed_peak_rel": max(item["peak_rel"] for item in observed),
            "injection": {
                "element": 0,
                "tolerance_multiplier": PERTURB_TOLERANCE_MULTIPLIER,
            },
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "control": "coop-h1-raw-output",
        "mode": args.mode,
        "status": "PASS",
        **identity,
        **outcome,
        "serving_path_reachable": False,
    }
    return write_receipt(args.receipt, payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("baseline", "perturb"), required=True)
    value.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--geometry", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--test-source", type=Path, required=True)
    value.add_argument("--source-manifest", type=Path, required=True)
    value.add_argument("--bundle", type=Path, required=True)
    value.add_argument("--receipt", type=Path, required=True)
    value.add_argument("--baseline-receipt", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        record = run(args)
    except (ControlRefusal, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
