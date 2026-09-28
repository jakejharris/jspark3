#!/usr/bin/env python3
"""Gate model admission on matching first-prompt and finalization receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from v16_common import QAError, json_safe, load_json, sha256_json, write_json

SCHEMA = "jspark3-v16-admission-gate/1"
FIRST_SCHEMA = "jspark3-v16-first-prompt/1"
FINAL_SCHEMA = "jspark3-v16-finalize-receipt/1"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(first: dict, final: dict, *, first_path: str = "first", final_path: str = "final",
             first_sha256: str | None = None) -> dict:
    findings = []
    if first.get("schema") != FIRST_SCHEMA or first.get("verdict") != "PASS":
        findings.append("first-prompt receipt is not PASS")
    if final.get("schema") != FINAL_SCHEMA or final.get("verdict") != "PASS":
        findings.append("finalization receipt is not PASS")
    first_identity = first.get("identity_config")
    final_identity = final.get("identity_config")
    valid_identity = (
        isinstance(first_identity, dict)
        and set(first_identity) == {"coop", "adaptive-k", "dense-fp8"}
        and first_identity.get("coop") in ("off", "on")
        and first_identity.get("adaptive-k") in ("off", "ema")
        and first_identity.get("dense-fp8") in ("off", "trunk", "negative-coarse")
    )
    if not valid_identity or first_identity != final_identity:
        findings.append("first-prompt and finalization config identities differ")
    if final.get("payload_sha256") != sha256_json({key: value for key, value in final.items() if key != "payload_sha256"}):
        findings.append("finalization payload SHA mismatch")
    if first_sha256 is None or not any(
            row.get("sha256") == first_sha256 and row.get("schema") == FIRST_SCHEMA
            for row in final.get("input_hashes", {}).get("client_evidence", [])):
        findings.append("first-prompt receipt was not reconciled in this finalization session")
    if (final.get("producer") == "qualify_runtime.py" or first.get("producer") == "qualify_runtime.py"
            or valid_identity and first_identity.get("coop") == "on"):
        if (first.get("producer") != "qualify_runtime.py" or final.get("producer") != "qualify_runtime.py"
                or not first.get("boot")
                or first.get("boot") != final.get("boot")
                or first.get("manifest_sha256") != final.get("manifest_sha256")):
            findings.append("operator receipts belong to different boots")
        expected_stock = {'ABLIT': '0', 'profile': 'production-stock', 'APC': '1'}
        if first.get('stock_profile') != expected_stock or final.get('stock_profile') != expected_stock:
            findings.append('public operator admission requires unchanged stock-only settings')
        component = first.get('component_qualification')
        if component != final.get('component_qualification'):
            findings.append('component qualification changed during admission')
        if first_identity and first_identity.get('coop') == 'on':
            import re
            fields = {'native_sha256', 'policy_sha256', 'component_seal_sha256', 'gate_index_sha256', 'operator_image_config'}
            if (not isinstance(component, dict) or set(component) != fields
                    or any(not re.fullmatch('[0-9a-f]{64}', str(component.get(k, '')).removeprefix('sha256:'))
                           for k in fields)):
                findings.append('operator coop-on lacks complete component/native/policy/image identity')
    if first_identity and first_identity.get("dense-fp8") == "negative-coarse":
        findings.append("test-only negative-coarse cannot open admission")
    return {"schema": SCHEMA, "verdict": "PASS" if not findings else "FAIL",
            "identity_config": first_identity if not findings else None,
            "inputs": {"first_prompt": first_path, "finalize": final_path},
            "findings": findings}


def self_check() -> dict:
    identity = {"coop": "off", "adaptive-k": "ema", "dense-fp8": "off"}
    first = {"schema": FIRST_SCHEMA, "verdict": "PASS", "identity_config": identity}
    final = {"schema": FINAL_SCHEMA, "verdict": "PASS", "identity_config": identity,
             "input_hashes": {"client_evidence": [{"sha256": "first-fixture", "schema": FIRST_SCHEMA}]}}
    final["payload_sha256"] = sha256_json(final)
    healthy = evaluate(first, final, first_sha256="first-fixture")
    mismatch = evaluate(first, {**final, "identity_config": {**identity, "dense-fp8": "trunk"}}, first_sha256="first-fixture")
    passed = healthy["verdict"] == "PASS" and mismatch["verdict"] == "FAIL"
    return {"schema": "jspark3-v16-admission-self-check/1",
            "verdict": "PASS" if passed else "FAIL", "healthy": healthy,
            "identity_mismatch_negative_control": mismatch}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-prompt", type=Path)
    parser.add_argument("--finalize", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        report = self_check()
    elif args.first_prompt is None or args.finalize is None:
        report = {"schema": SCHEMA, "verdict": "FAIL",
                  "findings": ["--first-prompt and --finalize are required"]}
    else:
        try:
            report = evaluate(load_json(args.first_prompt), load_json(args.finalize),
                              first_path=str(args.first_prompt), final_path=str(args.finalize),
                              first_sha256=_sha(args.first_prompt))
            final = load_json(args.finalize)
            if final.get('producer') == 'qualify_runtime.py':
                evidence = final.get('evidence_sha256', {})
                if not evidence or 'first-prompt.json' not in evidence:
                    raise QAError('operator evidence inventory is missing')
                for name, expected in evidence.items():
                    path = args.finalize.parent / name
                    if Path(name).name != name or path.is_symlink() or _sha(path) != expected:
                        raise QAError('operator evidence changed: ' + name)
            report["input_sha256"] = {"first_prompt": _sha(args.first_prompt),
                                      "finalize": _sha(args.finalize)}
        except (OSError, ValueError, QAError) as exc:
            report = {"schema": SCHEMA, "verdict": "FAIL", "findings": [str(exc)]}
    write_json(args.out, report)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
