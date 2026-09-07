#!/usr/bin/env python3
"""Offline regression for the boot-49 receipt-channel fix.

The stage-2 .pth side effect makes every later python3 print a startup line to
stdout, so the stage receipt can no longer travel on stdout (boot-49:
`Extra data: line 2 column 1 (char 1045)`). These checks prove, with no hosts,
containers, or network, that:

- the legacy wire pattern (`json\\nnoise\\n`) breaks json.loads exactly as seen
  in boot-49, including a real site .pth import path;
- `print_receipt` honors JSPARK3_RECEIPT_OUT (fresh file, canonical bytes,
  stdout untouched) and keeps stdout as the default for standalone callers;
- `run_stage` returns the receipt unchanged while stdout carries noise, and
  refuses missing / malformed / non-object / wrong-transform receipts and a
  failed child even when a receipt file exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "recipe" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import apply_base_pipeline as pipeline  # noqa: E402
from _atomic import canonical, print_receipt  # noqa: E402

RECEIPT = {
    "schema_version": 1,
    "transform": "apply_kpool_tail.py",
    "state": "APPLIED",
    "script_sha256": "0" * 64,
    "contract_sha256": "0" * 64,
    "image_manifest": "m" * 64,
    "image_config": "c" * 64,
    "targets": [],
}
NOISE = "glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)"

# Child snippets, executed with `python3 -c` so no fixture tree is needed.
CHILD = """
import os, sys
sys.path.insert(0, {scripts!r})
mode, stage = sys.argv[1], sys.argv[2]
receipt = dict({receipt!r}, transform=stage)
if mode == "legacy":
    from _atomic import canonical
    print({noise!r})
    sys.stdout.buffer.write(canonical(receipt))
elif mode == "receipt-file":
    from _atomic import print_receipt
    print({noise!r})
    print_receipt(receipt)
elif mode == "pth-startup":
    import site
    site.addsitedir(os.environ["NOISE_SITE"])
    from _atomic import print_receipt
    print_receipt(receipt)
elif mode == "exit-with-receipt":
    from _atomic import print_receipt
    print_receipt(receipt)
    sys.exit(9)
elif mode == "exit-silent":
    sys.exit(0)
elif mode == "malformed":
    with open(os.environ["JSPARK3_RECEIPT_OUT"], "wb") as handle:
        handle.write(b"not json at all\\n")
elif mode == "non-object":
    from _atomic import canonical
    with open(os.environ["JSPARK3_RECEIPT_OUT"], "wb") as handle:
        handle.write(canonical([{{"schema_version": 1}}]))
elif mode == "wrong-transform":
    from _atomic import print_receipt
    print_receipt(dict(receipt, transform="apply_kda_fg.py"))
"""

CHILD = CHILD.format(scripts=str(SCRIPTS), noise=NOISE, receipt=RECEIPT)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def child_process(mode: str, stage: str = "apply_kpool_tail.py", **environ: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(environ)
    return subprocess.run([sys.executable, "-c", CHILD, mode, stage], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, env=env)


def pth_site(directory: Path) -> None:
    site = directory / "noise-site"
    site.mkdir()
    (site / "zzz_noise.pth").write_text("import zzz_noise\n")
    (site / "zzz_noise.py").write_text(f"print({NOISE!r})\n")
    return site


def test_legacy_wire_pattern_breaks_jsonloads() -> None:
    process = child_process("legacy")
    expect(process.returncode == 0, process.stderr)
    expect(process.stdout == canonical(RECEIPT).decode() + f"{NOISE}\n",
           f"unexpected wire order: {process.stdout!r}")
    try:
        json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        expect("Extra data" in str(exc), str(exc))
        expect(exc.pos == len(canonical(RECEIPT).decode()),
               f"junk starts at {exc.pos}, not right after the receipt")
    else:
        raise AssertionError("legacy stdout pattern parsed as clean JSON")


def test_pth_startup_produces_same_wire_pattern() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        site = pth_site(Path(scratch))
        process = child_process("pth-startup", NOISE_SITE=str(site))
        expect(process.returncode == 0, process.stderr)
        expect(process.stdout == canonical(RECEIPT).decode() + f"{NOISE}\n",
               f".pth startup did not contaminate stdout: {process.stdout!r}")
        try:
            json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            expect("Extra data" in str(exc), str(exc))
        else:
            raise AssertionError(".pth-contaminated stdout parsed as clean JSON")


def test_print_receipt_honors_receipt_out_channel() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        receipt_path = Path(scratch) / "stage.receipt.json"
        expect(not receipt_path.exists(), "receipt path pre-existed")
        process = child_process("receipt-file", JSPARK3_RECEIPT_OUT=str(receipt_path))
        expect(process.returncode == 0, process.stderr)
        expect(receipt_path.read_bytes() == canonical(RECEIPT),
               "receipt file is not exactly the canonical receipt")
        expect(json.loads(receipt_path.read_text()) == RECEIPT, "receipt file damaged")
        expect(NOISE in process.stdout and canonical(RECEIPT).decode() not in process.stdout,
               "child stdout should stay dirty; the receipt must only travel on the file")


def test_print_receipt_stdout_default_preserved() -> None:
    process = child_process("receipt-file")
    expect(process.returncode == 0, process.stderr)
    expect(process.stdout == canonical(RECEIPT).decode() + f"{NOISE}\n",
           "standalone caller without JSPARK3_RECEIPT_OUT lost stdout receipt")


def stage_refused(message: str, mode: str, stage: str = "apply_kpool_tail.py") -> None:
    args = argparse.Namespace()
    original = pipeline.command
    pipeline.command = lambda a, n: [sys.executable, "-c", CHILD, mode, stage]
    try:
        try:
            pipeline.run_stage(args, stage, "apply")
        except pipeline.Refusal as exc:
            expect(message in str(exc), f"refusal does not name the fault: {exc}")
        else:
            raise AssertionError(f"{mode}: run_stage accepted a bad stage result")
    finally:
        pipeline.command = original


def test_run_stage_parses_receipt_despite_stdout_noise() -> None:
    args = argparse.Namespace()
    original = pipeline.command
    pipeline.command = lambda a, n: [sys.executable, "-c", CHILD, "receipt-file", n]
    try:
        receipt = pipeline.run_stage(args, "apply_kpool_tail.py", "apply")
    finally:
        pipeline.command = original
    expect(receipt == RECEIPT, "run_stage altered the receipt")


def test_run_stage_refuses_bad_channels() -> None:
    stage_refused("apply_kpool_tail.py: stage receipt unavailable on receipt channel",
                  "exit-silent")  # clean exit, receipt file never written
    stage_refused("apply_kpool_tail.py", "exit-with-receipt")  # failed child, file exists
    stage_refused("apply_kpool_tail.py", "malformed")
    stage_refused("apply_kpool_tail.py", "non-object")
    stage_refused("apply_kpool_tail.py", "wrong-transform")


def stage_command(a: argparse.Namespace, n: str) -> list[str]:
    return [sys.executable, "-c", CHILD, "receipt-file", n]


def test_no_stale_receipts_across_invocations() -> None:
    args = argparse.Namespace()
    seen: list[str] = []
    original_run = pipeline.subprocess.run
    original_command = pipeline.command

    def spy_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        seen.append(kwargs["env"]["JSPARK3_RECEIPT_OUT"])
        return original_run(cmd, **kwargs)

    pipeline.subprocess.run = spy_run
    pipeline.command = stage_command
    try:
        pipeline.run_stage(args, "apply_kpool_tail.py", "apply")
        pipeline.run_stage(args, "apply_kpool_tail.py", "recovery")
    finally:
        pipeline.subprocess.run = original_run
        pipeline.command = original_command
    expect(len(seen) == 2 and seen[0] != seen[1],
           "receipt channel was not fresh per invocation")
    for path in seen:
        expect(not Path(path).exists(), f"temp receipt survived cleanup: {path}")


def test_recover_journals_uses_helper_with_partial_continuation() -> None:
    args = argparse.Namespace()
    asked: list[str] = []
    original = pipeline.command

    def command(a: argparse.Namespace, n: str) -> list[str]:
        asked.append(n)
        return stage_command(a, n)

    pipeline.command = command
    try:
        receipts = pipeline.recover_journals(args, Path("/unused"),
                                             ["apply_tp3_overlay.py", "apply_image_glm_dflash.py"])
    finally:
        pipeline.command = original
    expect(asked == ["apply_tp3_overlay.py", "apply_image_glm_dflash.py"],
           f"recovery ran the wrong stages: {asked}")
    expect(all(row == dict(RECEIPT, transform=name) for row, name in zip(receipts, asked)),
           "recovery receipts damaged")


def main() -> int:
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok {name}")
    print("receipt-channel regression: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
