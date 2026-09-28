"""Shared checks extracted from the established release validator."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.dom.minidom
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe/scripts"))
import _diagnostics as diagnostics


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": "PASS", "detail": detail})
        print(f"PASS {name}" + (f": {detail}" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        record = diagnostics.record_failure(detail if isinstance(detail, BaseException) else ValueError(detail))
        self.checks.append({"check": name, "status": "FAIL", **record})
        print(f"FAIL {name}: " + record["reason"])

def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

def git_pointer(path: Path) -> bool:
    """A linked-worktree `.git` administrative pointer file, never a release source file."""
    try:
        with path.open("rb") as stream:
            return stream.read(8) == b"gitdir: "
    except OSError:
        return False

def pth_payload_allowed(path: Path) -> bool:
    """A `.pth` file is a Python path-configuration file: small UTF-8 text.

    Torch checkpoints also use the `.pth` suffix, so the payload policy refuses
    anything binary or oversized rather than whitelisting the suffix.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) > 4096 or b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True

def check_syntax(root: Path, files: list[Path], report: Report) -> None:
    problems = []
    counts = {"json": 0, "jsonl": 0, "yaml": 0, "sh": 0, "py": 0, "svg": 0}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        yaml = None
    for path in files:
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                counts["json"] += 1
            elif suffix == ".jsonl":
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.strip():
                        json.loads(line)
                counts["jsonl"] += 1
            elif suffix in (".yml", ".yaml", ".cff"):
                if yaml is None:
                    problems.append(f"PyYAML unavailable for {rel}")
                else:
                    yaml.safe_load(path.read_text(encoding="utf-8"))
                counts["yaml"] += 1
            elif suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                counts["sh"] += 1
            elif suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
                counts["py"] += 1
            elif suffix == ".svg":
                xml.dom.minidom.parse(str(path))
                counts["svg"] += 1
        except Exception as exc:  # noqa: BLE001
            diagnostics.retain(__import__("traceback").format_exc())
            problems.append(f"{rel}: syntax refused")
    if problems:
        report.fail("syntax", "; ".join(problems[:10]))
    else:
        report.ok("syntax", ", ".join(f"{k}={v}" for k, v in counts.items()))

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")

def upstream_verbatim(root: Path) -> set[str]:
    """Files carried byte-for-byte from the mirrored upstream repository.

    Their contents are fixed by contract, so link and prose rules that govern this
    project's own writing do not apply to them; the weights-mirror check hashes them
    instead. The set is derived from the manifest rather than hand-listed.
    """
    manifest_path = root / "huggingface/jspark3/WEIGHTS-MANIFEST.json"
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"huggingface/" + entry.get("mirror_path", entry["path"])
            for entry in manifest["entries"] if entry.get("present_in_tree")}

def check_links(root: Path, files: list[Path], report: Report) -> None:
    problems = []
    count = 0
    verbatim = upstream_verbatim(root)
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        if path.relative_to(root).as_posix() in verbatim:
            continue
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            count += 1
            clean = target.split("#", 1)[0]
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                problems.append(f"{path.relative_to(root).as_posix()} -> {target}")
    if problems:
        report.fail("markdown-links", "; ".join(problems[:10]))
    else:
        report.ok("markdown-links", f"{count} local links resolve ({len(verbatim)} upstream-verbatim files exempt)")

def check_dry_runs(root: Path, report: Report) -> None:
    recipe = root / "recipe"
    env = recipe / ".env.example"
    problems = []
    ran = 0
    for command in ("preflight", "start", "status", "stop", "verify"):
        process = subprocess.run([sys.executable, "-B", str(recipe / "scripts/fleetctl.py"), command,
                                  "--env-file", str(env), "--dry-run"], cwd=recipe, capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        diagnostics.retain(process.stdout + process.stderr)
        ran += 1
        if process.returncode or "DRY-RUN" not in process.stdout:
            problems.append(f"{command}: rc={process.returncode} {process.stderr.strip()[-120:]}")
    for wrapper in ("clean-room-setup.sh", "preflight.sh", "start.sh", "health.sh", "status.sh", "verify.sh", "stop.sh", "rollback.sh"):
        process = subprocess.run([str(recipe / "scripts" / wrapper), "--env-file", str(env), "--dry-run"],
                                 cwd=recipe, capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        diagnostics.retain(process.stdout + process.stderr)
        ran += 1
        if process.returncode or "DRY-RUN" not in process.stdout:
            problems.append(f"{wrapper}: rc={process.returncode} {process.stderr.strip()[-120:]}")
    if problems:
        report.fail("lifecycle-dry-runs", "; ".join(problems[:6]))
    else:
        report.ok("lifecycle-dry-runs", f"{ran} controller and wrapper dry-runs rendered without host contact")
