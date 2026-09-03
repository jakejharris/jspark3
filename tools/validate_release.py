#!/usr/bin/env python3
"""Validate the JSpark3 v1 release tree.

Checks, in order: inventory and payload policy; required files; a
binary-aware scan for internal labels, private paths, hosts, addresses, and
secrets; the GitHub, container-registry, and package-URL owner identity;
syntax of every JSON, JSONL, YAML, shell, Python, and SVG file; local
Markdown links; identity contracts between the dependency manifest and the
recipe's pinned constants; byte-identity of license and attribution copies;
the Hugging Face card front matter; the release manifest shape; the mirrored
target-weight payload, its manifest, and its gate; the
machine-readable results, the comparison taxonomy, and the claim reconciliation
of every number quoted in public prose (no cross-class percentage, no bare label); the SBOM; both SHA256SUMS manifests; and the lifecycle
dry-runs. Exit status is non-zero if any check fails.

    python3 tools/validate_release.py . --report validation.json
    python3 tools/validate_release.py . --write-sums   # regenerate SHA256SUMS
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.dom.minidom

SKIP_DIRS = {".git", "dist", "__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".pcap", ".pcapng", ".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt",
                      ".key", ".pem", ".p12", ".pfx", ".log"}
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", "PI-SESSION.jsonl", "pi-pane.log"}
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_TREE_BYTES = 64 * 1024 * 1024

REQUIRED = [
    "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "REQUIRED_ATTRIBUTION.md",
    "CITATION.cff", "CITATION.bib", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "SHA256SUMS",
    "docs/ARCHITECTURE.md", "docs/TECHNICAL-REPORT.md", "docs/BENCHMARKS.md", "docs/INSTALL.md",
    "docs/OPERATIONS.md", "docs/LIMITATIONS.md", "docs/REPRODUCIBILITY.md", "docs/LICENSING.md",
    "docs/diagrams/architecture.svg", "docs/diagrams/architecture.mmd",
    "manifests/dependencies.json", "manifests/release.json", "manifests/derivation.json",
    "manifests/sbom.cdx.json",
    "huggingface/README.md", "huggingface/RESULTS.json", "huggingface/LICENSE",
    "huggingface/UPSTREAM_MODEL_CARD.md", "huggingface/THIRD_PARTY_NOTICES.md",
    "huggingface/MANIFEST.json", "huggingface/MIRROR.json", "huggingface/PROVENANCE.md",
    "huggingface/ORIGINAL_MODEL_CARD.md", "huggingface/SHA256SUMS", "huggingface/config.json",
    "huggingface/jspark3/PROVENANCE.md", "huggingface/jspark3/WEIGHTS-MANIFEST.json",
    "huggingface/jspark3/MIRROR-COMPLETION.json",
    "huggingface/jspark3/RECIPE-LICENSE", "huggingface/jspark3/UPLOAD.md",
    "huggingface/jspark3/THIRD_PARTY_NOTICES.md", "huggingface/jspark3/REQUIRED_ATTRIBUTION.md",
    "results/results.json", "results/SUMMARY.md",
    "results/evidence/reference/published-references.json",
    "results/evidence/reference/mia-tp2-historical-0e2e78f/agent-summary.json",
    "results/evidence/reference/mia-tp2-current-c190db1a-adapted/agent-summary.json",
    "results/evidence/reference/mia-tp2-current-c190db1a-adapted/rapid-screen/MATRIX.json",
    "results/evidence/reference/mia-tp2-current-exact-attempt/finding.json",
    "results/evidence/reference/fly-derived-9093765c-adapted/agent-summary.json",
    "recipe/README.md", "recipe/SHA256SUMS", "recipe/.env.example", "recipe/.gitignore",
    "recipe/LICENSE", "recipe/THIRD_PARTY_NOTICES.md", "recipe/REQUIRED_ATTRIBUTION.md",
    "recipe/config/profile.json", "recipe/config/checkpoint-contract.json",
    "recipe/config/patch-contract.json", "recipe/scripts/fleetctl.py",
    "recipe/scripts/container_entry.sh", "recipe/overlays/trunk_w8a16.py",
    "recipe/overlays/patch_base_loader_hook.py", "recipe/transforms/README.md",
    "recipe/docs/REPRODUCIBILITY.md", "recipe/docs/LIMITATIONS.md",
    "docker/Dockerfile", "docker/build.sh", "docker/README.md",
    ".github/workflows/validate.yml", ".github/workflows/release.yml", ".github/workflows/image.yml",
    "release/RELEASE-NOTES.md", "release/ANNOUNCEMENT-BLOG.md", "release/ANNOUNCEMENT-SOCIAL.md",
    "tools/validate_release.py", "tools/build_release_assets.sh", "tools/build_sbom.py",
    "tools/mirror_weights.py",
    "tools/build_hashes.py", "tools/analyze_tail.py", "tools/analyze_sse_pcap.py",
]

# Internal labels, private locations, and machine identity that must not appear anywhere.
# Literals are split so this file does not trip its own scan.
_INTERNAL = b"|".join([b"day" b"break", b"rag" b"ged-tp3", b"single-own" b"er-closer",
                       b"autonomous-mul" b"tiboot", b"jspark3-" b"r33"])
LEAK_PATTERNS = [
    ("internal label", re.compile(rb"(?i)(?<![a-z0-9])boot\d")),
    ("internal program name", re.compile(rb"(?i)" + _INTERNAL)),
    ("home path", re.compile(rb"/home/[A-Za-z]")),
    ("windows path", re.compile(rb"(?i)users/[a-z]|/mn" rb"t/c/|desk" rb"top")),
    ("host label", re.compile(rb"(?i)(?<![a-z0-9])spark[123](?![a-z0-9])")),
    ("private address", re.compile(rb"(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})(?![\d.])")),
    ("bearer token", re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("api key", re.compile(rb"(?<![A-Za-z0-9])(?:sk|hf)_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("owner mailbox", re.compile(rb"(?i)jjh" rb"digital")),
]
# The intended namespace is allowed only in these forms. Source forge, container
# registry, Hugging Face, and package URLs all carry the maintainer's confirmed
# account. Literals stay split so this file does not trip its own scan.
_ACCOUNT = b"ja" b"ke" b"jh" b"arris"
_SUPERSEDED = b"ja" b"ke" b"jh"
_IDENTITY_PREFIXES = rb"(?:github\.com/|ghcr\.io/|huggingface\.co/|pkg:github/|pkg:huggingface/)"
# A forge, registry, Hub, or purl identity may never carry the superseded
# account as a complete owner segment. The delimiter requirement prevents the
# shorter old account from matching the confirmed account as a prefix.
SUPERSEDED_OWNER_FORMS = re.compile(
    rb"(?i)" + _IDENTITY_PREFIXES + _SUPERSEDED + rb"(?=/|[?#]|$)"
)
DOTTED_QUAD = re.compile(rb"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
ALLOWED_QUADS = re.compile(rb"^(?:0\.0\.0\.0|127\.0\.0\.1|192\.0\.2\.\d+|198\.51\.100\.\d+|203\.0\.113\.\d+)$")

# Public prose whose numbers must reconcile with results.json or the structural allowlist.
PROSE = [
    "README.md", "CHANGELOG.md", "docs/ARCHITECTURE.md", "docs/TECHNICAL-REPORT.md",
    "docs/BENCHMARKS.md", "docs/INSTALL.md", "docs/OPERATIONS.md", "docs/LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md", "docs/LICENSING.md", "huggingface/README.md",
    "release/RELEASE-NOTES.md", "release/ANNOUNCEMENT-BLOG.md", "release/ANNOUNCEMENT-SOCIAL.md",
    "recipe/README.md", "recipe/docs/REPRODUCIBILITY.md", "recipe/docs/LIMITATIONS.md",
    "docker/README.md", "results/SUMMARY.md", "FINAL-RELEASE-INDEX.md", "RELEASE-GATE.md",
]
# Configuration, identity, and version literals that are not measurements.
STRUCTURAL_NUMBERS = {
    "1.0.0", "1.0", "1.2.0", "1.5", "2.0", "4.0", "5.3", "3.9", "3.11", "3.12", "12.1",
    "0.83", "1,000,000", "8,192", "1,595,392,320", "4.46", "154,880", "154,944",
    "2,048", "2,112", "175,622,979,576", "175,642,157,752", "175,715,854,754",
    "2,342,169,800", "9,000",
    "1.1", "0.4", "1.9",
}
# The mirrored target weights. These are measured facts about someone else's
# repository at a pinned revision; the validator refuses if any of them drifts.
MIRROR_UPSTREAM = "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
MIRROR_REVISION = "25a44fdbf16862a46b7cc9921142c6c81350af2f"
MIRROR_ORIGIN = "brandonmusic/GLM-5.3-Flash-tr3-4bpw"
MIRROR_ORIGIN_REVISION = "5ab363a8dcf6405955fd5f99671e01a1c9fb124b"
MIRROR_FILES = 144
MIRROR_BYTES = 175715854754
MIRROR_SHARDS = 120
MIRROR_LICENSE_SHA256 = "9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a"
HF_MAIN_REVISION = "e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc"
HF_METADATA_PARENT_REVISION = "e9cbbafaf9ae4ab64f385c2f68e7fe2f06d78676"
RELEASE_STATUS = "v1.0.0-released-hf-public"
RELEASE_DATE = "2026-09-02"
RELEASE_URL = "https://github.com/jakejharris/jspark3/releases/tag/v1.0.0"
MIRROR_STATUS = "published and remotely verified on public main"
MIRROR_RECEIPT_SHA256 = "97c4ff5d715cd30186de7f092ee50d0ccbaeb560a720a88dc5d3d49f2412c8a9"
MIRROR_MANIFEST_SHA256 = "ef4cc59538e7c4056cd7f0b8228561987ab8bd94c51e0efeadf6ca4fd0e20b59"

NUMBER_TOKEN = re.compile(r"(?<![\w.])[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.,])[+-]?\d+\.\d+%?|(?<![\w.,])[+-]?\d+%")
# The comparison taxonomy: the bare word is banned because it collapses three different
# evidence classes into one, and a percentage may never be computed against an external row.
BARE_BASELINE = re.compile(r"(?i)(?<![\w-])baselines?(?![\w-])")
EXTERNAL_CLASSES = {"published_reference", "local_reproduction"}


def prose_units(text: str):
    """Split public prose into comparison units: one Markdown table row, or one sentence."""
    def sentences(buffer: list[str], start: int):
        paragraph = " ".join(line.strip() for line in buffer)
        for piece in re.split(r"(?<=[.!?])\s+", paragraph):
            if piece.strip():
                yield start, piece
    buffer: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("|"):
            if buffer:
                yield from sentences(buffer, start)
                buffer = []
            yield number, line
        elif not stripped:
            if buffer:
                yield from sentences(buffer, start)
                buffer = []
        else:
            if not buffer:
                start = number
            buffer.append(line)
    if buffer:
        yield from sentences(buffer, start)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": "PASS", "detail": detail})
        print(f"PASS {name}" + (f": {detail}" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        self.checks.append({"check": name, "status": "FAIL", "detail": detail})
        print(f"FAIL {name}: {detail}")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def tree_files(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            files.append(Path(dirpath) / name)
    return files


def check_inventory(root: Path, files: list[Path], report: Report) -> None:
    problems = []
    total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        if path.is_symlink():
            problems.append(f"symlink: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            problems.append(f"forbidden payload: {rel}")
        if size > MAX_FILE_BYTES:
            problems.append(f"file over {MAX_FILE_BYTES} bytes: {rel} ({size})")
    if total > MAX_TREE_BYTES:
        problems.append(f"tree over {MAX_TREE_BYTES} bytes: {total}")
    # Compiled caches are never shipped and are invisible to the walk above, so look for
    # them directly: a stray .pyc can carry the source text of a private tool into the tree.
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            if name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                problems.append(f"compiled cache directory: {Path(dirpath, name).relative_to(root).as_posix()}")
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                problems.append(f"compiled cache file: {Path(dirpath, name).relative_to(root).as_posix()}")
    if problems:
        report.fail("inventory", "; ".join(problems[:10]))
    else:
        report.ok("inventory", f"{len(files)} files, {total} bytes")
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        report.fail("required-files", "missing " + ", ".join(missing))
    else:
        report.ok("required-files", f"{len(REQUIRED)} present")


def check_leaks(root: Path, files: list[Path], report: Report) -> None:
    problems = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        for label, pattern in LEAK_PATTERNS:
            match = pattern.search(data)
            if match:
                problems.append(f"{label} in {rel}: {match.group(0)[:40]!r}")
                break
        match = SUPERSEDED_OWNER_FORMS.search(data)
        if match:
            problems.append(
                f"superseded owner segment in {rel}: "
                f"{match.group(0).decode('utf-8', 'replace')}"
            )
        for quad in DOTTED_QUAD.findall(data):
            if not ALLOWED_QUADS.match(quad):
                problems.append(f"non-documentation address in {rel}: {quad.decode()}")
                break
    if problems:
        report.fail("privacy-scan", "; ".join(problems[:12]))
    else:
        report.ok("privacy-scan", f"{len(files)} files clean")


def check_owner_identity(root: Path, files: list[Path], report: Report) -> None:
    """Refuse a forge, registry, Hub, or package identity under the superseded account.

    The confirmed maintainer account is pinned in ``manifests/release.json``; an earlier
    spelling inferred from the personal domain is refused wherever it appears.
    """
    problems = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        match = SUPERSEDED_OWNER_FORMS.search(path.read_bytes())
        if match:
            problems.append(f"superseded owner in {rel}: {match.group(0).decode('utf-8', 'replace')}")
    destinations = json.loads((root / "manifests/release.json").read_text(encoding="utf-8")).get(
        "intended_destinations", {})
    account = "/" + _ACCOUNT.decode() + "/"
    for key in ("github", "huggingface"):
        if account not in destinations.get(key, ""):
            problems.append(f"intended_destinations.{key} does not name the maintainer's account")
    if destinations.get("ghcr") is not None:
        problems.append("intended_destinations.ghcr must be null for v1.0.0")
    if problems:
        report.fail("owner-identity", "; ".join(problems[:12]))
    else:
        report.ok("owner-identity",
                  f"{len(files)} files free of the superseded owner; public destinations "
                  "name the confirmed maintainer account and GHCR is excluded")


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
            problems.append(f"{rel}: {type(exc).__name__}: {str(exc)[:80]}")
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


def constant(text: str, name: str) -> str | None:
    match = re.search(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else None


def check_identity(root: Path, report: Report) -> None:
    deps = json.loads((root / "manifests/dependencies.json").read_text(encoding="utf-8"))
    image = deps["container_image"]
    reference, digest, config = image["reference"], image["manifest_digest"], image["config_digest"]
    target, draft = deps["target_checkpoint"], deps["draft_checkpoint"]
    problems = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    expect(reference.endswith("@" + digest), "image reference does not end with its manifest digest")
    scripts = root / "recipe/scripts"
    fleet = (scripts / "fleetctl.py").read_text(encoding="utf-8")
    preflight = (scripts / "remote_preflight.py").read_text(encoding="utf-8")
    receipt = (scripts / "make_image_receipt.py").read_text(encoding="utf-8")
    atomic = (scripts / "_atomic.py").read_text(encoding="utf-8")
    checkpoint = (scripts / "validate_checkpoint.py").read_text(encoding="utf-8")
    views = (scripts / "prepare_runtime_views.py").read_text(encoding="utf-8")
    expect(constant(fleet, "IMAGE") == reference, "fleetctl IMAGE differs from manifest")
    expect(constant(fleet, "IMAGE_CONFIG") == config, "fleetctl IMAGE_CONFIG differs from manifest")
    expect(constant(preflight, "IMAGE") == reference, "remote_preflight IMAGE differs from manifest")
    expect(constant(preflight, "IMAGE_CONFIG") == config, "remote_preflight IMAGE_CONFIG differs")
    expect(constant(receipt, "MANIFEST") == digest, "make_image_receipt MANIFEST differs")
    expect(constant(receipt, "CONFIG") == config, "make_image_receipt CONFIG differs")
    expect(digest.split(":")[1] in atomic and config.split(":")[1] in atomic, "_atomic.py lacks the image digests")
    for name, key in (("TARGET_NATIVE", "native_config_sha256"), ("TARGET_RUNTIME", "runtime_config_sha256"),
                      ("INDEX_SHA", "model_index_sha256"), ("TOKENIZER", "tokenizer_sha256"),
                      ("TOKENIZER_CONFIG", "tokenizer_config_sha256")):
        expect(constant(checkpoint, name) == target[key], f"validate_checkpoint {name} differs from manifest")
    for name, key in (("DRAFT_NATIVE", "native_config_sha256"), ("DRAFT_RUNTIME", "runtime_config_sha256"),
                      ("DRAFT_MODEL", "model_sha256")):
        expect(constant(checkpoint, name) == draft[key], f"validate_checkpoint {name} differs from manifest")
    for name, key in (("TARGET_NATIVE", "native_config_sha256"), ("TARGET_RUNTIME", "runtime_config_sha256")):
        expect(constant(views, name) == target[key], f"prepare_runtime_views {name} differs from manifest")
    profile = json.loads((root / "recipe/config/profile.json").read_text(encoding="utf-8"))
    expect(profile["image"]["reference"] == reference and profile["image"]["config_digest"] == config,
           "profile.json image block differs from manifest")
    patch = json.loads((root / "recipe/config/patch-contract.json").read_text(encoding="utf-8"))
    expect(patch["image"]["manifest"] == digest.split(":")[1] and patch["image"]["config"] == config.split(":")[1],
           "patch-contract.json image block differs from manifest")
    expect(patch["upstreams"]["flycockpit_commit"] == deps["source_revisions"]["flycockpit"]["commit"],
           "FlyCockpit commit differs between patch contract and manifest")
    expect(patch["upstreams"]["vcruz305_commit"] == deps["source_revisions"]["vcruz305"]["commit"],
           "vcruz305 commit differs between patch contract and manifest")
    contract = (root / "recipe/config/checkpoint-contract.json").read_text(encoding="utf-8")
    expect(target["revision"] in contract and draft["revision"] in contract,
           "checkpoint-contract.json lacks a pinned revision")
    dockerfile = (root / "docker/Dockerfile").read_text(encoding="utf-8")
    expect(f"FROM {reference}" in dockerfile, "Dockerfile FROM is not the pinned reference")
    owned = deps.get("owned_runtime_image", {})
    expect(owned.get("intended_reference") is None and owned.get("digest") is None,
           "owned runtime image must have no intended registry reference or digest")
    expect(owned.get("published") is False and owned.get("redistributed_here") is False,
           "owned runtime image must be recorded as unpublished and not redistributed")
    expect(owned.get("runtime_reference") == reference,
           "owned runtime image record must point execution to the exact upstream digest")
    expect(owned.get("publication_policy") == "NO-GO for v1.0.0",
           "owned runtime image must carry the v1.0.0 NO-GO policy")
    build_script = (root / "docker/build.sh").read_text(encoding="utf-8")
    image_workflow = (root / ".github/workflows/image.yml").read_text(encoding="utf-8")
    expect("--push" not in build_script and "ghcr.io/jakejharris/jspark3" not in build_script,
           "docker/build.sh must not expose a registry push path")
    expect("docker/login-action" not in image_workflow and "packages: write" not in image_workflow,
           "image workflow must not have registry login or package-write authority")
    expect("push: false" in image_workflow and "ghcr.io/jakejharris/jspark3" not in image_workflow,
           "image workflow must be a local-only, non-pushing build check")
    licensing = (root / "docs/LICENSING.md").read_text(encoding="utf-8")
    expect("no jspark3 ghcr image is published for v1.0.0" in licensing.lower() and
           "independently satisfying nvidia's terms" in licensing.lower(),
           "licensing page lacks the binding GHCR NO-GO and redistribution boundary")
    for name in ("README.md", "docs/INSTALL.md", "docs/TECHNICAL-REPORT.md", "huggingface/README.md"):
        text = (root / name).read_text(encoding="utf-8")
        expect(target["revision"] in text, f"{name} lacks the target revision")
        expect(draft["revision"] in text, f"{name} lacks the draft revision")
    # Overlay identity: profile pins must equal the shipped file hashes and the entrypoint literals.
    overlay = profile["w8a16_overlay"]
    entry = (root / "recipe/scripts/container_entry.sh").read_text(encoding="utf-8")
    expect(overlay["overlay_sha256"] == sha256(root / "recipe/overlays/trunk_w8a16.py"), "overlay hash drift")
    expect(overlay["loader_patcher_sha256"] == sha256(root / "recipe/overlays/patch_base_loader_hook.py"), "patcher hash drift")
    expect(f"expected_overlay={overlay['overlay_sha256']}" in entry and
           f"expected_patcher={overlay['loader_patcher_sha256']}" in entry and
           f"base_loader_after={overlay['base_loader_after_sha256']}" in entry and
           f"base_loader_before={overlay['base_loader_before_sha256']}" in entry,
           "entrypoint hash literals differ from profile")
    derivation = json.loads((root / "manifests/derivation.json").read_text(encoding="utf-8"))
    expect(derivation["overlay_module"]["public_sha256"] == overlay["overlay_sha256"] and
           derivation["loader_patcher"]["public_sha256"] == overlay["loader_patcher_sha256"] and
           derivation["patched_base_loader"]["public_after_sha256"] == overlay["base_loader_after_sha256"],
           "derivation.json hashes differ from profile")
    for entry_row in derivation["files"]:
        path = root / "recipe" / entry_row["public"]
        expect(path.is_file() and sha256(path) == entry_row["public_sha256"], f"derivation record stale: {entry_row['public']}")
    if problems:
        report.fail("identity-contracts", "; ".join(problems[:10]))
    else:
        report.ok("identity-contracts", "manifest, recipe constants, profile, contracts, Dockerfile, docs agree")


def check_copies(root: Path, report: Report) -> None:
    problems = []
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "REQUIRED_ATTRIBUTION.md"):
        base = (root / name).read_bytes()
        if (root / "recipe" / name).read_bytes() != base:
            problems.append(f"recipe/{name} differs from root copy")
    # The Hugging Face repository root carries the checkpoint's own license and notices,
    # so this project's Apache set lives beside them under huggingface/jspark3/.
    for name, copy in (("LICENSE", "huggingface/jspark3/RECIPE-LICENSE"),
                       ("THIRD_PARTY_NOTICES.md", "huggingface/jspark3/THIRD_PARTY_NOTICES.md"),
                       ("REQUIRED_ATTRIBUTION.md", "huggingface/jspark3/REQUIRED_ATTRIBUTION.md")):
        if (root / copy).read_bytes() != (root / name).read_bytes():
            problems.append(f"{copy} differs from the root {name}")
    if (root / "huggingface/RESULTS.json").read_bytes() != (root / "results/results.json").read_bytes():
        problems.append("huggingface/RESULTS.json differs from results/results.json")
    attribution = (root / "REQUIRED_ATTRIBUTION.md").read_text(encoding="utf-8")
    sentence = next((line for line in attribution.splitlines() if line.startswith("This work includes")), "")
    if not sentence:
        problems.append("attribution sentence missing")
    for name in ("README.md", "huggingface/README.md", "docs/LICENSING.md", "THIRD_PARTY_NOTICES.md"):
        if sentence and sentence not in (root / name).read_text(encoding="utf-8"):
            problems.append(f"{name} lacks the verbatim attribution sentence")
    if problems:
        report.fail("license-copies", "; ".join(problems))
    else:
        report.ok("license-copies", "license set byte-identical; attribution present verbatim")


def check_hf_card(root: Path, report: Report) -> None:
    import yaml  # type: ignore
    text = (root / "huggingface/README.md").read_text(encoding="utf-8")
    problems = []
    if not text.startswith("---\n"):
        problems.append("no front matter")
    else:
        block = text.split("---\n", 2)
        meta = yaml.safe_load(block[1]) or {}
        if meta.get("license") != "other":
            problems.append("license must be 'other'")
        if not meta.get("license_name"):
            problems.append("license_name missing")
        if meta.get("license_link") != "LICENSE":
            problems.append("license_link must be LICENSE")
        # The repository carries the mirrored checkpoint, so the card must declare the
        # checkpoint's own operative terms exactly as the upstream repository declares them.
        if meta.get("license_name") != "shapleymcg-license-1.0":
            problems.append("license_name must be the checkpoint's own license name")
        if meta.get("base_model") != "zai-org/GLM-5.3-Flash":
            problems.append("base_model must name the base model of the mirrored checkpoint")
        if meta.get("base_model_relation") != "quantized":
            problems.append("base_model_relation must be 'quantized'")
        if meta.get("library_name") != "transformers":
            problems.append("library_name must be 'transformers' for the mirrored checkpoint")
        if "tags" not in meta or not isinstance(meta["tags"], list):
            problems.append("tags missing")
        else:
            missing = [tag for tag in ("shapleymcg", "glm", "exl3", "tr3", "vllm", "quantized")
                       if tag not in meta["tags"]]
            if missing:
                problems.append("card drops upstream tags: " + ", ".join(missing))
    if problems:
        report.fail("huggingface-card", "; ".join(problems))
    else:
        report.ok("huggingface-card", "front matter valid")


def check_weights_mirror(root: Path, report: Report) -> None:
    """The mirrored target weights: layout, manifest, in-tree bytes, and the gate.

    Every hash here is measured, never asserted: each in-tree file is hashed and
    compared with its manifest entry, and no LFS object may be committed at all.
    """
    problems = []
    manifest = json.loads((root / "huggingface/jspark3/WEIGHTS-MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("upstream_repository") != MIRROR_UPSTREAM or manifest.get("upstream_revision") != MIRROR_REVISION:
        problems.append("manifest does not pin the upstream repository and revision")
    if manifest.get("origin_repository") != MIRROR_ORIGIN or manifest.get("origin_revision") != MIRROR_ORIGIN_REVISION:
        problems.append("manifest does not record the origin repository and revision")
    entries = manifest.get("entries", [])
    if len(entries) != MIRROR_FILES or manifest.get("files") != MIRROR_FILES:
        problems.append(f"manifest must describe {MIRROR_FILES} files, found {len(entries)}")
    total = sum(entry["size"] for entry in entries)
    if total != MIRROR_BYTES or manifest.get("bytes") != MIRROR_BYTES:
        problems.append(f"manifest byte total must be {MIRROR_BYTES}, found {total}")
    shards = sum(1 for entry in entries if entry["path"].endswith(".safetensors"))
    if shards != MIRROR_SHARDS:
        problems.append(f"manifest must list {MIRROR_SHARDS} safetensors shards, found {shards}")
    lfs_entries = [entry for entry in entries if entry.get("lfs") is True]
    if len(lfs_entries) != 123 or manifest.get("lfs_files") != 123:
        problems.append("manifest must identify exactly 123 upload payloads")
    if sum(entry["size"] for entry in lfs_entries) != 175715659341:
        problems.append("manifest upload payloads must total 175715659341 bytes")
    for entry in entries:
        if "dflash" in entry["path"].lower():
            problems.append(f"the separately pinned draft must not appear in the mirror: {entry['path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            problems.append(f"manifest entry without a SHA-256: {entry['path']}")
        if entry.get("hash_source") not in ("hub-lfs-oid+upstream-sha256sums", "local-sha256-of-fetched-bytes"):
            problems.append(f"manifest entry without a stated hash source: {entry['path']}")
    if manifest.get("draft_checkpoint_included") is not False:
        problems.append("the draft checkpoint must be recorded as not mirrored")
    # In-tree bytes must hash to what the manifest records; no LFS object may be committed.
    hashed = 0
    for entry in entries:
        target = root / "huggingface" / entry.get("mirror_path", entry["path"])
        if entry.get("lfs"):
            if entry.get("present_in_tree") is not False:
                problems.append(f"an LFS object must not be marked present in the tree: {entry['path']}")
            if target.is_file():
                problems.append(f"LFS object committed to the tree: {entry['path']}")
            continue
        if entry.get("present_in_tree") is not True:
            problems.append(f"small upstream file must be present in the tree: {entry['path']}")
            continue
        if not target.is_file():
            problems.append(f"missing upstream file: {entry['path']}")
            continue
        if target.stat().st_size != entry["size"]:
            problems.append(f"size drift against the manifest: {entry['path']}")
        elif sha256(target) != entry["sha256"]:
            problems.append(f"hash drift against the manifest: {entry['path']}")
        else:
            hashed += 1
    # The checkpoint's own license, verbatim, at the mirror root.
    license_entry = next((entry for entry in entries if entry["path"] == "LICENSE"), None)
    if not license_entry or license_entry["sha256"] != MIRROR_LICENSE_SHA256:
        problems.append("the manifest's LICENSE hash differs from the pinned upstream value")
    if sha256(root / "huggingface/LICENSE") != MIRROR_LICENSE_SHA256:
        problems.append("huggingface/LICENSE is not the upstream license file")
    # The attribution notice is a condition of the checkpoint's license, not a courtesy.
    attribution = next((line for line in (root / "REQUIRED_ATTRIBUTION.md").read_text(encoding="utf-8").splitlines()
                        if line.startswith("This work includes")), "")
    if not attribution or attribution not in (root / "huggingface/README.md").read_text(encoding="utf-8"):
        problems.append("the card lacks the verbatim attribution notice the checkpoint license requires")
    # Release metadata: the remotely verified mirror and exact receipt are
    # published on an immutable, public, ungated, enabled main revision.
    release = json.loads((root / "manifests/release.json").read_text(encoding="utf-8"))
    mirror = release.get("weights_mirror", {})
    for field, value in (("upstream_repository", MIRROR_UPSTREAM), ("upstream_revision", MIRROR_REVISION),
                         ("origin_repository", MIRROR_ORIGIN), ("origin_revision", MIRROR_ORIGIN_REVISION),
                         ("files", MIRROR_FILES), ("bytes", MIRROR_BYTES),
                         ("quantization_author", "Brandon M. Music")):
        if mirror.get(field) != value:
            problems.append(f"release manifest weights_mirror.{field} differs from the pinned value")
    if release.get("publication_authorized") is not True:
        problems.append("the mirror publication authorization must be recorded")
    if mirror.get("status") != MIRROR_STATUS:
        problems.append("the mirror must record verified publication on public main")
    if mirror.get("hf_revision") != HF_MAIN_REVISION:
        problems.append("the mirror must record the immutable terminal Hugging Face main revision")
    for field, value in (("repository_public", True), ("repository_gated", False),
                         ("repository_enabled", True)):
        if mirror.get(field) is not value:
            problems.append(f"release manifest weights_mirror.{field} differs from terminal Hub state")
    if mirror.get("gate") is not None:
        problems.append("the published mirror must not retain an outstanding gate")
    transfer_contract = {
        "upload_files": 123,
        "upload_bytes": 175715659341,
        "destination_policy": "existing Hugging Face pull-request revision only; never main",
        "transfer_client": "hf upload-large-folder",
        "resume_cache": ".cache/huggingface",
        "completion_receipt": "jspark3/MIRROR-COMPLETION.json",
        "completion_receipt_sha256": MIRROR_RECEIPT_SHA256,
        "merge_policy": "manual only after exact remote verification",
    }
    for field, value in transfer_contract.items():
        if mirror.get(field) != value:
            problems.append(f"release manifest weights_mirror.{field} differs from the safe-transfer contract")
    deps = json.loads((root / "manifests/dependencies.json").read_text(encoding="utf-8"))
    dep_mirror = deps.get("target_checkpoint", {}).get("mirror", {})
    if dep_mirror.get("repository") != MIRROR_UPSTREAM or dep_mirror.get("revision") != MIRROR_REVISION:
        problems.append("dependencies.json target_checkpoint.mirror differs from the pinned value")
    if dep_mirror.get("status") != mirror.get("status") or dep_mirror.get("gate") is not None:
        problems.append("dependencies.json mirror publication state differs from release.json")
    for field, value in (("hf_revision", HF_MAIN_REVISION), ("repository_public", True),
                         ("repository_gated", False), ("repository_enabled", True)):
        if dep_mirror.get(field) != value:
            problems.append(f"dependencies.json mirror.{field} differs from terminal Hub state")
    for field, value in transfer_contract.items():
        if dep_mirror.get(field) != value:
            problems.append(f"dependencies.json mirror.{field} differs from the safe-transfer contract")
    receipt_path = root / "huggingface/jspark3/MIRROR-COMPLETION.json"
    expected_receipt = {
        "lfs_bytes": 175715659341,
        "lfs_files": 123,
        "metadata_parent_revision": HF_METADATA_PARENT_REVISION,
        "pull_request_ref": "refs/pr/1",
        "schema_version": 1,
        "status": "remote-weight-pr-verified",
        "target_repository": "jakejharris/jspark3",
        "upstream_repository": MIRROR_UPSTREAM,
        "upstream_revision": MIRROR_REVISION,
        "verified_at": "2026-09-03T02:02:55+00:00",
        "weights_manifest_sha256": MIRROR_MANIFEST_SHA256,
    }
    if not receipt_path.is_file():
        problems.append("the exact mirror completion receipt is missing")
    elif sha256(receipt_path) != MIRROR_RECEIPT_SHA256:
        problems.append("the mirror completion receipt bytes differ from the verified receipt")
    else:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"the mirror completion receipt is invalid: {exc}")
        else:
            if receipt != expected_receipt:
                problems.append("the mirror completion receipt fields differ from the verified receipt")
    mirror_tool = root / "tools/mirror_weights.py"
    tool_text = mirror_tool.read_text(encoding="utf-8")
    for forbidden in ("huggingface-cli",):
        if forbidden in tool_text:
            problems.append(f"mirror tool retains obsolete client syntax: {forbidden}")
    for required in ("upload-large-folder", "remote-verify", "completion-receipt", "refs/pr/"):
        if required not in tool_text:
            problems.append(f"mirror tool lacks safe-transfer behavior: {required}")
    self_test = subprocess.run([sys.executable, str(mirror_tool), "self-test"],
                               check=False, text=True, capture_output=True)
    if self_test.returncode:
        problems.append("mirror tool offline self-test failed: " +
                        (self_test.stdout + self_test.stderr).strip()[:300])
    if problems:
        report.fail("weights-mirror", "; ".join(problems[:10]))
    else:
        report.ok("weights-mirror",
                  f"{MIRROR_FILES} files, {MIRROR_BYTES} bytes, {MIRROR_SHARDS} shards; "
                  f"{hashed} in-tree files hash to the manifest, no shard committed, draft not mirrored, "
                  f"exact receipt and public main {HF_MAIN_REVISION} verified")


def check_release_manifest(root: Path, report: Report) -> None:
    release = json.loads((root / "manifests/release.json").read_text(encoding="utf-8"))
    problems = []
    expected = {"github": "https://github.com/jakejharris/jspark3",
                "huggingface": "https://huggingface.co/jakejharris/jspark3",
                "ghcr": None}
    if release.get("name") != "JSpark3 v1" or release.get("slug") != "jspark3" or release.get("tag") != "v1.0.0":
        problems.append("identity differs from the public identity contract")
    if release.get("intended_destinations") != expected:
        problems.append("intended destinations differ from contract")
    live = release.get("live_links", {})
    if release.get("publication_authorized") is not True:
        problems.append("publication_authorized must record the maintainer's approval")
    if release.get("status") != RELEASE_STATUS:
        problems.append("status must record the terminal v1.0.0 release and public Hub mirror")
    if release.get("date_released") != RELEASE_DATE:
        problems.append("date_released must record the v1.0.0 release date")
    if live.get("github") != expected["github"]:
        problems.append("the live GitHub link must equal the intended destination")
    if live.get("huggingface") != expected["huggingface"]:
        problems.append("the live Hugging Face link must equal the intended destination")
    if live.get("ghcr_digest") is not None:
        problems.append("GHCR digest must remain null for v1.0.0")
    if live.get("release_page") != RELEASE_URL:
        problems.append("release-page link must equal the deterministic v1.0.0 release URL")
    container = release.get("container_distribution", {})
    if container.get("runtime_reference") != "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58":
        problems.append("release must use the exact upstream runtime image digest")
    if container.get("jspark3_ghcr_image_published") is not False or container.get("local_build_only") is not True:
        problems.append("release must record no JSpark3 GHCR image and a local-only build")
    if container.get("redistribution_policy") != "do not redistribute a local build without independently satisfying NVIDIA and upstream terms":
        problems.append("release lacks the binding local-build redistribution policy")
    if release.get("maintainer_contact") != "https://github.com/jakejharris":
        problems.append("maintainer_contact must use the confirmed public GitHub profile")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    if 'version: 1.0.0' not in cff:
        problems.append("CITATION.cff version differs")
    if f'date-released: {RELEASE_DATE}' not in cff:
        problems.append("CITATION.cff release date differs")
    if f'url: "{RELEASE_URL}"' not in cff:
        problems.append("CITATION.cff release URL differs")
    terminal_docs = (
        "README.md", "RELEASE-GATE.md", "FINAL-RELEASE-INDEX.md", "CHANGELOG.md",
        "docs/INSTALL.md", "release/RELEASE-NOTES.md",
    )
    for relative in terminal_docs:
        text = (root / relative).read_text(encoding="utf-8")
        if RELEASE_URL not in text:
            problems.append(f"{relative} lacks the terminal v1.0.0 release URL")
        if RELEASE_DATE not in text:
            problems.append(f"{relative} lacks the v1.0.0 release date")
    state_files = terminal_docs + (
        "docs/TECHNICAL-REPORT.md", "docs/LICENSING.md", "huggingface/README.md",
        "release/ANNOUNCEMENT-BLOG.md", "release/ANNOUNCEMENT-SOCIAL.md",
        "manifests/release.json", "manifests/dependencies.json",
    )
    contradictions = {
        "v1.0.0 release pending": "release still described as pending",
        "github-public-hf-metadata-live-release-pending": "obsolete release status retained",
        "public tree, pre-tag": "pre-tag index heading retained",
        "assembled pre-tag release state": "pre-tag index copy retained",
        "tag and release are not created yet": "tag and release still described as absent",
        "no `v1.0.0` tag": "v1.0.0 tag still described as absent",
        "weight transfer has not started": "Hub weight transfer still described as unstarted",
        "weight transfer not started": "Hub weight transfer still described as unstarted",
        "transfer is authorized but has not started": "Hub weight transfer still described as unstarted",
        "transfer remains separate and unstarted": "Hub weight transfer still described as unstarted",
        "mirror is prepared but not uploaded yet": "Hub mirror still described as pre-transfer",
        "weight mirror not merged": "Hub mirror still described as unmerged",
        "transfer in progress on a separate review branch": "Hub mirror still described as in progress",
        "not merged into the public hub main revision": "Hub mirror still described as unmerged",
        "not merged into main": "Hub mirror still described as unmerged",
        "weight transfer is pending": "Hub mirror still described as pending",
    }
    for relative in state_files:
        lowered = (root / relative).read_text(encoding="utf-8").lower()
        for phrase, reason in contradictions.items():
            if phrase in lowered:
                problems.append(f"{relative}: {reason}")
    if problems:
        report.fail("release-manifest", "; ".join(problems))
    else:
        report.ok("release-manifest", "v1.0.0 release URL and date frozen; verified mirror on immutable public Hub main; GHCR excluded")


def fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def check_results(root: Path, report: Report) -> dict:
    results = json.loads((root / "results/results.json").read_text(encoding="utf-8"))
    display = results["display"]
    problems = []
    c1 = results["c1_single_stream"]
    runs = c1["candidate_batteries"]
    for phase in ("c1_code", "c1_count", "c1_prose", "c3", "c6"):
        key = phase.replace("c1_", "c1-") if phase.startswith("c1_") else phase
        values = sorted(runs[r][key] for r in ("r1", "r2", "r3"))
        median = values[1]
        if display.get(f"c1.median.{phase}") != fmt_rate(median):
            problems.append(f"median {phase} display drift")
        if display.get(f"c1.delta.{key}") != fmt_pct(pct(median, c1["internal_ablation_control_product_battery"][key])):
            problems.append(f"delta {key} display drift")
    for key, row in c1["strict_paired_r3"].items():
        if display.get(f"paired.{key}.delta") != fmt_pct(pct(row["candidate"], row["matched_control"])):
            problems.append(f"paired {key} display drift")
    for key, row in results["scheduler_matched"]["windows"].items():
        if display.get(f"sched.{key}.delta") != fmt_pct(row["aggregate_delta_percent_vs_internal_ablation_control"]):
            problems.append(f"scheduler {key} display drift")
        if display.get(f"sched.{key}.candidate") != fmt_rate(row["candidate_aggregate_tok_s"]):
            problems.append(f"scheduler {key} rate drift")
    pre = results["prefill_matched"]
    if display.get("prefill.delta") != fmt_pct(pct(pre["candidate_prefill_tok_s"], pre["internal_ablation_control_prefill_tok_s"])):
        problems.append("prefill delta display drift")
    gate = c1["internal_promotion_gate"]
    if gate.get("verdict") != "REJECT" or display.get("gate.code_floor") != "67.0":
        problems.append("internal promotion gate must remain disclosed as REJECT at the 67.0 floor")
    demo = results["agent_demonstration"]["internal_sustained_pacing_gate"]
    if demo.get("verdict") != "FAIL":
        problems.append("demonstration pacing gate must remain disclosed as FAIL")
    if results.get("evidence_grade") != "ENGINEERING-EVIDENCE":
        problems.append("results evidence grade literal differs")
    # Comparison taxonomy: three labelled classes, each with its admission rule intact.
    published = results.get("published_references", {})
    local = results.get("local_reproductions", {})
    ablation = results.get("internal_ablation", {})
    if published.get("class_label") != "Published reference recipes" or published.get("basis") != "author-reported":
        problems.append("published-reference block missing its class label or author-reported basis")
    if local.get("class_label") != "Local reproduction of a published recipe":
        problems.append("local-reproduction block missing its class label")
    if ablation.get("class_label") != "Internal ablation" or "unreleased internal development build" not in ablation.get("denominator", ""):
        problems.append("internal-ablation block must name an unreleased internal development build as the denominator")
    required_fields = ("nodes", "quantization_lane", "runtime", "context", "tensor_parallel",
                       "speculation_as_loaded", "workload", "thinking_sampling", "concurrency",
                       "sample_count", "estimator", "source_url", "basis")
    for name, row in published.get("references", {}).items():
        for field in required_fields:
            if row.get(field) in (None, "") and field != "expert_parallel":
                problems.append(f"published reference {name} lacks {field}")
        if row.get("basis") != "author-reported":
            problems.append(f"published reference {name} is not marked author-reported")
    for name in ("mia-tp2-historical-0e2e78f", "mia-tp2-current-c190db1a-adapted", "fly-derived-9093765c-adapted"):
        row = local.get(name, {})
        if not row.get("fidelity"):
            problems.append(f"local reproduction {name} lacks its fidelity qualifier")
        if not row.get("source_commit"):
            problems.append(f"local reproduction {name} lacks a pinned source commit")
    attempt = local.get("mia-tp2-current-exact-attempt", {})
    if attempt.get("benchmark_result") is not None or attempt.get("rate_claim_eligible") is not False:
        problems.append("the failed exact-recipe attempt must carry no benchmark number")
    screen = local.get("mia-tp2-current-c190db1a-adapted", {}).get("rapid_screen", {})
    if screen.get("rate_claim_eligible") is not False:
        problems.append("the adapted rapid screen must stay marked rate-claim ineligible")
    if set(results.get("display_class", {})) != set(display):
        problems.append("every display value must carry an evidence class")
    if problems:
        report.fail("results-consistency", "; ".join(problems[:10]))
    else:
        report.ok("results-consistency", f"{len(display)} display values recomputed; failure evidence preserved")
    return results


def check_claims(root: Path, results: dict, report: Report) -> None:
    allowed = set(results["display"].values()) | STRUCTURAL_NUMBERS
    allowed |= {v.lstrip("+") for v in allowed if v.startswith("+")}
    classes: dict[str, set[str]] = {}
    for key, value in results["display"].items():
        kind = results["display_class"][key]
        classes.setdefault(value, set()).add(kind)
        classes.setdefault(value.lstrip("+"), set()).add(kind)

    def kinds(token: str) -> set[str]:
        return classes.get(token, set()) | classes.get(token.lstrip("+"), set())

    problems = []
    checked = 0
    units = 0
    for name in PROSE:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b", " ", text)          # addresses
        text = re.sub(r"\bv?\d+\.\d+\.\d+\b", " ", text)                                   # versions
        text = re.sub(r"\b[0-9a-f]{12,64}\b", " ", text)                                  # digests
        text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*-\d+(?:\.\d+)?\b", " ", text)                # hyphenated names like GLM-5.3
        text = re.sub(r"\b20\d\d-\d\d-\d\d(?:T[\d:]+Z?)?\b", " ", text)                   # dates
        text = re.sub(r"`[^`\n]*`", " ", text)                                             # inline code
        text = re.sub(r"```.*?```", " ", text, flags=re.S)                                 # code blocks
        for line_number, line in enumerate(text.splitlines(), 1):
            for token in NUMBER_TOKEN.findall(line):
                checked += 1
                bare = token.lstrip("+")
                if token in allowed or bare in allowed:
                    continue
                problems.append(f"{name}:{line_number} {token}")
        # A percentage may never share a table row or a sentence with a published-reference
        # or local-reproduction value: that is how a comparison gets manufactured across
        # instruments that do not compare. A percentage that is itself author-reported for
        # that row is allowed, because it is quoted rather than computed.
        for line_number, unit in prose_units(text):
            units += 1
            tokens = NUMBER_TOKEN.findall(unit)
            if not any(kinds(token) & EXTERNAL_CLASSES for token in tokens):
                continue
            for token in tokens:
                if token.endswith("%") and not (kinds(token) & EXTERNAL_CLASSES):
                    problems.append(f"{name}:{line_number} percentage {token} shares a row or sentence with an external-class value")
        # A table that carries an external-class row is a comparison table as a whole:
        # a delta placed in any other row of it still reads as a comparison against that
        # row, so the rule applies to the whole table, not only to the row it sits in.
        block: list[tuple[int, str]] = []
        blocks: list[list[tuple[int, str]]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("|"):
                block.append((line_number, line))
            elif block:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
        for rows in blocks:
            tokens = [(number, token) for number, line in rows for token in NUMBER_TOKEN.findall(line)]
            if not any(kinds(token) & EXTERNAL_CLASSES for _, token in tokens):
                continue
            for number, token in tokens:
                if token.endswith("%") and not (kinds(token) & EXTERNAL_CLASSES):
                    problems.append(f"{name}:{number} percentage {token} sits in a table that carries an external-class row")
        # The bare word collapses the three evidence classes into one; name the class instead.
        raw = (root / name).read_text(encoding="utf-8")
        for line_number, line in enumerate(raw.splitlines(), 1):
            if BARE_BASELINE.search(line):
                problems.append(f"{name}:{line_number} bare word 'baseline' in public prose")
    if problems:
        report.fail("claim-reconciliation", f"{len(problems)} findings: " + "; ".join(problems[:15]))
    else:
        report.ok("claim-reconciliation",
                  f"{checked} numeric tokens reconcile with results.json display values; "
                  f"{units} comparison units and every table carrying an external row are free of a "
                  f"cross-class percentage, and no bare 'baseline' appears")


def check_sbom(root: Path, report: Report) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sbom.cdx.json"
        subprocess.run([sys.executable, str(root / "tools/build_sbom.py"),
                        "--dependencies", str(root / "manifests/dependencies.json"),
                        "--release", str(root / "manifests/release.json"),
                        "--output", str(out)], check=True, capture_output=True)
        if out.read_bytes() != (root / "manifests/sbom.cdx.json").read_bytes():
            report.fail("sbom", "manifests/sbom.cdx.json is stale; rerun tools/build_sbom.py")
            return
    sbom = json.loads((root / "manifests/sbom.cdx.json").read_text(encoding="utf-8"))
    report.ok("sbom", f"CycloneDX {sbom['specVersion']}, {len(sbom['components'])} components, regenerates byte-identical")


def sums(root: Path, base: Path, exclude: set[str]) -> str:
    lines = []
    for path in tree_files(base):
        rel = path.relative_to(base).as_posix()
        if rel in exclude:
            continue
        lines.append(f"{sha256(path)}  {rel}")
    return "\n".join(lines) + "\n"


def check_sums(root: Path, report: Report, write: bool) -> None:
    targets = [
        (root / "recipe", root / "recipe/SHA256SUMS", {"SHA256SUMS"}),
        (root, root / "SHA256SUMS", {"SHA256SUMS"}),
    ]
    problems = []
    for base, manifest, exclude in targets:
        expected = sums(root, base, exclude)
        if write:
            manifest.write_text(expected, encoding="utf-8")
        elif not manifest.is_file() or manifest.read_text(encoding="utf-8") != expected:
            problems.append(f"{manifest.relative_to(root).as_posix()} is stale (run with --write-sums)")
    if problems:
        report.fail("sha256sums", "; ".join(problems))
    else:
        recipe_rows = (root / "recipe/SHA256SUMS").read_text(encoding="utf-8").count("\n")
        root_rows = (root / "SHA256SUMS").read_text(encoding="utf-8").count("\n")
        report.ok("sha256sums", f"recipe {recipe_rows} rows, root {root_rows} rows" + (" (written)" if write else " (verified)"))


def check_dry_runs(root: Path, report: Report) -> None:
    recipe = root / "recipe"
    env = recipe / ".env.example"
    problems = []
    ran = 0
    for command in ("preflight", "start", "status", "stop", "verify"):
        process = subprocess.run([sys.executable, str(recipe / "scripts/fleetctl.py"), command,
                                  "--env-file", str(env), "--dry-run"], cwd=recipe, capture_output=True, text=True)
        ran += 1
        if process.returncode or "DRY-RUN" not in process.stdout:
            problems.append(f"{command}: rc={process.returncode} {process.stderr.strip()[-120:]}")
    for wrapper in ("clean-room-setup.sh", "preflight.sh", "start.sh", "health.sh", "status.sh", "verify.sh", "stop.sh", "rollback.sh"):
        process = subprocess.run([str(recipe / "scripts" / wrapper), "--env-file", str(env), "--dry-run"],
                                 cwd=recipe, capture_output=True, text=True)
        ran += 1
        if process.returncode or "DRY-RUN" not in process.stdout:
            problems.append(f"{wrapper}: rc={process.returncode} {process.stderr.strip()[-120:]}")
    if problems:
        report.fail("lifecycle-dry-runs", "; ".join(problems[:6]))
    else:
        report.ok("lifecycle-dry-runs", f"{ran} controller and wrapper dry-runs rendered without host contact")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--write-sums", action="store_true", help="regenerate recipe/SHA256SUMS and SHA256SUMS")
    args = parser.parse_args()
    root = args.root.resolve()
    report = Report()
    files = tree_files(root)
    check_inventory(root, files, report)
    check_leaks(root, files, report)
    check_owner_identity(root, files, report)
    check_syntax(root, files, report)
    check_links(root, files, report)
    try:
        check_identity(root, report)
    except Exception as exc:  # noqa: BLE001
        report.fail("identity-contracts", f"{type(exc).__name__}: {exc}")
    try:
        check_copies(root, report)
        check_hf_card(root, report)
        check_release_manifest(root, report)
        check_weights_mirror(root, report)
        results = check_results(root, report)
        check_claims(root, results, report)
        check_sbom(root, report)
    except Exception as exc:  # noqa: BLE001
        report.fail("release-shape", f"{type(exc).__name__}: {exc}")
    check_sums(root, report, args.write_sums)
    check_dry_runs(root, report)
    summary = {"root": root.name, "checks": report.checks, "failed": report.failed,
               "verdict": "PASS" if report.failed == 0 else "FAIL"}
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"VERDICT {summary['verdict']} ({len(report.checks)} checks, {report.failed} failed)")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
