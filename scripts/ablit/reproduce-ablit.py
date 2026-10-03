#!/usr/bin/env python3
"""Fetch, convert and split refusal-removed weights into three ranks.

Run with Python 3.12, NumPy 2.1.0 and the release's installed TensorFold wheel.
Accept the source's Hugging Face terms yourself and set HF_TOKEN to fetch it.
The caller must reserve sufficient CPU, memory and disk capacity for this job.
Existing verified source/base snapshots may be supplied to avoid downloads.
The work directory must be new. Outputs include licenses and rank manifests.
"""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE_API = "https://huggingface.co/api/models/orcarouter/GLM-5.3-Flash-Uncensored-MLX/revision/c02a5f6fa06f0aa444877b44d19fd5c96390329f?blobs=true"
LOCK = None
METRICS = None
SEQUENCE = 0


def step(name, command, heavy=True):
    global SEQUENCE
    SEQUENCE += 1
    tag = f"{SEQUENCE:02d}-{name}"
    start = time.monotonic()
    if METRICS:
        (METRICS / ".current-step.json").write_text(json.dumps({"step": tag, "state": "waiting"}) + "\n")
    with contextlib.ExitStack() as stack:
        if heavy and LOCK:
            fd = stack.enter_context(LOCK.open("r+"))
            fcntl.flock(fd, fcntl.LOCK_EX)
        acquired = time.monotonic()
        if METRICS:
            (METRICS / ".current-step.json").write_text(json.dumps({"step": tag, "state": "running"}) + "\n")
            command = [sys.executable, str(HERE / "measure-step.py"), str(METRICS / (tag + ".time.json")), *command]
        result = subprocess.run(command)
        ended = time.monotonic()
    if METRICS:
        (METRICS / (tag + ".json")).write_text(json.dumps({"step": name, "lock_wait_seconds": acquired - start, "wall_seconds": ended - acquired, "exit_code": result.returncode}, indent=2) + "\n")
        (METRICS / ".current-step.json").write_text(json.dumps({"step": tag, "state": "done"}) + "\n")
    result.check_returncode()


def run(script, *args, heavy=True):
    step(Path(script).stem, [sys.executable, str(HERE / script), *map(str, args)], heavy)


def main():
    global LOCK, METRICS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--vontra", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--mem-floor-gib", type=float, default=10)
    p.add_argument("--source-api", type=Path, help="Saved anonymous API response at the pinned revision, for offline use")
    p.add_argument("--lock", type=Path, help="Existing shared lock file; released after every CPU/disk step")
    p.add_argument("--metrics", type=Path, help="Write per-step wall time and Linux process resource receipts")
    a = p.parse_args()
    import numpy
    if numpy.__version__ != "2.1.0":
        raise SystemExit("Use the pinned build environment with NumPy 2.1.0")
    if importlib.util.find_spec("tensorfold") is None:
        raise SystemExit("Install the release's pinned TensorFold wheel first")
    if not a.source and not os.environ.get("HF_TOKEN"):
        raise SystemExit("Accept the source repository's terms and set HF_TOKEN before downloading")
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    LOCK = a.lock
    METRICS = a.metrics.resolve() if a.metrics else work / "receipts"
    METRICS.mkdir(parents=True, exist_ok=False)
    source = a.source.resolve() if a.source else work / "source"
    base = a.vontra.resolve() if a.vontra else work / "base"
    if not a.source:
        run("fetch-source.py", source, heavy=False)
    os.environ.pop("HF_TOKEN", None)
    if not a.vontra:
        run("fetch-source.py", base, "--variant", "base", heavy=False)
    if a.source_api:
        api = a.source_api.read_bytes()
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(SOURCE_API, timeout=60) as response:
            api = response.read()
    (work / "source-api.json").write_bytes(api)
    run("verify-source.py", source, work / "source-api.json")
    run("prepare-inputs.py", base, work / "source-api.json", work / "prepared")
    run("build-template.py", work / "template.jinja", "--source", base / "chat_template.jinja")
    converted = work / "converted"
    run("convert-weights.py", "run", "--orca", source, "--vontra", base,
        "--target", work / "prepared/target.json", "--out", converted,
        "--template", work / "template.jinja", "--small-sha", work / "prepared/small.sha256",
        "--source-sums", work / "prepared/source-FILES.tsv", "--workers", a.workers,
        "--mem-floor-gib", a.mem_floor_gib)
    identity = (converted / "CONVERSION-ID").read_text().strip()
    for rank in range(3):
        out = converted / f"rank{rank}"
        step(f"split-rank{rank}", [sys.executable, "-m", "tensorfold.families.glm5_next.cuda.split",
                                   str(converted / "weights"), "--rank", str(rank), "--world", "3", str(out)])
        (out / ".complete").write_text(f"{identity} rank={rank} world=3\n")
        run("rank-manifest.py", out)
    notices = work / "input-notices"
    notices.mkdir()
    for name, directory in (("refusal-removed", source), ("base", base)):
        for file in ("LICENSE", "README.md"):
            shutil.copyfile(directory / file, notices / f"{name}-{file}")
    print("Complete: three rank directories and their manifests are under", converted)


if __name__ == "__main__":
    main()
