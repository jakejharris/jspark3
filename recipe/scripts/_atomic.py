"""Crash-recoverable, fail-closed transactions for base-recipe source transforms."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Callable, Iterator, Mapping

IMAGE_MANIFEST = "9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58"
IMAGE_CONFIG = "ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263"
ABSENT = "ABSENT"


class Refusal(RuntimeError):
    """A contract gate failed before admission."""


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def observed(path: Path) -> str:
    return sha_file(path) if path.is_file() else ABSENT


def compiled(path: Path, data: bytes) -> None:
    if path.suffix == ".py":
        compile(data.decode("utf-8"), str(path), "exec")
    elif path.suffix == ".json":
        json.loads(data)


def read_image_receipt(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Refusal("image receipt missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"invalid image receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal("image receipt must be an object")
    expected_keys = {
        "schema_version", "manifest_digest", "config_digest", "verification",
        "container_id", "rank", "preflight_sha256", "recipe_manifest_sha256",
        "payload_sha256",
    }
    if set(value) != expected_keys or value.get("schema_version") != 2:
        raise Refusal("image receipt schema drift")
    claimed = value.get("payload_sha256")
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    if claimed != sha_bytes(canonical(payload)):
        raise Refusal("image receipt payload hash mismatch")
    manifest = str(value.get("manifest_digest", "")).removeprefix("sha256:")
    config = str(value.get("config_digest", "")).removeprefix("sha256:")
    if manifest != IMAGE_MANIFEST or config != IMAGE_CONFIG:
        raise Refusal("OCI image identity drift")
    for key in ("container_id", "preflight_sha256", "recipe_manifest_sha256"):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
            raise Refusal(f"image receipt {key} drift")
    if (type(value.get("rank")) is not int or value["rank"] not in (0, 1, 2) or
            value.get("verification") != "host-observed-inspect-bound-create"):
        raise Refusal("image receipt host-binding drift")
    return value


def load_contract(
    path: Path, transform: str, expected: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    if path.is_symlink() or not path.is_file():
        raise Refusal("patch contract missing or symlinked")
    try:
        whole = json.loads(path.read_text(encoding="utf-8"))
        section = whole["transforms"][transform]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Refusal(f"invalid patch contract: {exc}") from exc
    if section != expected:
        raise Refusal(f"{transform}: contract content drift")
    if whole.get("image") != {"manifest": IMAGE_MANIFEST, "config": IMAGE_CONFIG}:
        raise Refusal("patch contract OCI identity drift")
    return whole, sha_file(path)


def safe_target(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise Refusal(f"unsafe target path: {relative}")
    root = root.resolve(strict=True)
    current = root
    root_dev = root.stat().st_dev
    for part in rel.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise Refusal(f"symlink target component: {current}")
        if metadata.st_dev != root_dev:
            raise Refusal(f"cross-device target component: {current}")
    parent = current.parent.resolve(strict=True)
    if parent != root and root not in parent.parents:
        raise Refusal(f"target escapes root: {relative}")
    if parent.stat().st_dev != root_dev:
        raise Refusal(f"cross-device target parent: {current.parent}")
    return current


def check_seams(path: Path, data: bytes, records: object, label: str) -> None:
    if not isinstance(records, list):
        raise Refusal(f"{path}: invalid {label} seam contract")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise Refusal(f"{path}: non-UTF-8 target") from exc
    for record in records:
        if not isinstance(record, dict) or set(record) != {"text", "count"}:
            raise Refusal(f"{path}: malformed {label} seam")
        needle, count = record["text"], record["count"]
        if not isinstance(needle, str) or not isinstance(count, int):
            raise Refusal(f"{path}: malformed {label} seam value")
        found = text.count(needle)
        if found != count:
            raise Refusal(f"{path}: {label} seam count {found}, expected {count}")


def check_forbidden(path: Path, data: bytes, records: object) -> None:
    if not isinstance(records, list) or not all(isinstance(item, str) for item in records):
        raise Refusal(f"{path}: malformed forbidden-after seam contract")
    text = data.decode("utf-8")
    for needle in records:
        if needle in text:
            raise Refusal(f"{path}: forbidden after seam remains: {needle!r}")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exact(path: Path, data: bytes, mode: int, uid: int, gid: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while staging transform")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        try:
            os.fchown(descriptor, uid, gid)
        except PermissionError:
            current = os.fstat(descriptor)
            if current.st_uid != uid or current.st_gid != gid:
                raise
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def root_lock(root: Path) -> Iterator[None]:
    # Locking the directory descriptor avoids creating a lock file during --check.
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def metadata(path: Path) -> tuple[int, int, int]:
    item = path.stat() if path.is_file() else path.parent.stat()
    return (
        stat.S_IMODE(item.st_mode) if path.is_file() else 0o644,
        item.st_uid,
        item.st_gid,
    )


def transaction_names(
    root: Path, transform: str, records: list[dict[str, object]]
) -> tuple[Path, dict[Path, tuple[Path, Path]]]:
    journal = root / f".jspark3-{transform}.transaction.json"
    files: dict[Path, tuple[Path, Path]] = {}
    for record in records:
        target = safe_target(root, str(record["path"]))
        token = sha_bytes(str(record["path"]).encode())[:16]
        files[target] = (
            target.parent / f".{target.name}.jspark3-{transform}-{token}.stage",
            target.parent / f".{target.name}.jspark3-{transform}-{token}.backup",
        )
    return journal, files


def transaction_debris(
    root: Path,
    transform: str,
    journal: Path,
    files: Mapping[Path, tuple[Path, Path]],
) -> list[Path]:
    """Return recognized transaction paths and reject look-alike debris."""
    known = {journal}
    for stage, backup in files.values():
        known.update((stage, backup))
    discovered = set()
    for target in files:
        discovered.update(target.parent.glob(f".{target.name}.jspark3-{transform}-*.stage"))
        discovered.update(target.parent.glob(f".{target.name}.jspark3-{transform}-*.backup"))
    unknown = sorted(discovered - known)
    if unknown:
        raise Refusal(f"unknown transaction artifact: {unknown[0]}")
    present = sorted(path for path in known if path.exists() or path.is_symlink())
    if any(path.is_symlink() for path in present):
        raise Refusal("symlink transaction artifact")
    if any(not path.is_file() for path in present):
        raise Refusal("non-regular transaction artifact")
    return present


def remove_artifacts(files: Mapping[Path, tuple[Path, Path]]) -> None:
    for stage, backup in files.values():
        stage.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        fsync_dir(stage.parent)


def finish_transaction(
    journal: Path, files: Mapping[Path, tuple[Path, Path]]
) -> None:
    """Make the terminal state authoritative before idempotent debris removal."""
    journal.unlink(missing_ok=True)
    fsync_dir(journal.parent)
    remove_artifacts(files)


def recover_orphans(
    root: Path,
    transform: str,
    records: list[dict[str, object]],
    files: Mapping[Path, tuple[Path, Path]],
) -> None:
    """Clean a pre-journal staging crash or interrupted terminal cleanup."""
    states = []
    for record in records:
        target = safe_target(root, str(record["path"]))
        before, after = str(record["before_sha256"]), str(record["after_sha256"])
        state = observed(target)
        if state not in {before, after}:
            raise Refusal(f"{transform}: orphan recovery target drift: {target}")
        if before != after:
            states.append("before" if state == before else "after")
    if states and len(set(states)) != 1:
        raise Refusal(f"{transform}: journal-free mixed state")
    # Without a durable journal no rename was authorized.  If every mutable
    # target is uniformly exact-before, any known stage/backup is disposable
    # pre-journal debris, even when power failed halfway through its write.  A
    # uniformly exact-after set is terminal cleanup debris.  Unknown names,
    # symlinks, non-regular artifacts, and mixed target states were refused
    # before this point.
    remove_artifacts(files)


def recover(
    root: Path,
    transform: str,
    records: list[dict[str, object]],
    journal: Path,
    files: Mapping[Path, tuple[Path, Path]],
    contract_sha: str,
) -> None:
    try:
        note = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"unreadable transaction journal: {exc}") from exc
    if (
        set(note) != {"schema_version", "transform", "contract_sha256", "stage", "targets"}
        or note.get("schema_version") != 1
        or note.get("transform") != transform
        or note.get("contract_sha256") != contract_sha
        or note.get("stage") != "PREPARED"
    ):
        raise Refusal("transaction journal identity drift")
    entries = note.get("targets")
    expected_entries = [
        {
            "path": str(record["path"]),
            "before_sha256": str(record["before_sha256"]),
            "after_sha256": str(record["after_sha256"]),
            "stage_sha256": str(record["after_sha256"]),
            "backup_sha256": str(record["before_sha256"]),
        }
        for record in records
    ]
    if entries != expected_entries:
        raise Refusal("transaction journal target drift")
    target_states: dict[Path, str] = {}
    for record in records:
        target = safe_target(root, str(record["path"]))
        stage, backup = files[target]
        before, after = str(record["before_sha256"]), str(record["after_sha256"])
        state = observed(target)
        if state not in {before, after}:
            raise Refusal(f"recovery target drift: {target}")
        target_states[target] = state
        if stage.exists() and observed(stage) != after:
            raise Refusal(f"recovery stage drift: {stage}")
        if backup.exists() and (before == ABSENT or observed(backup) != before):
            raise Refusal(f"recovery backup drift: {backup}")

    if all(target_states[safe_target(root, str(record["path"]))] == str(record["after_sha256"])
           for record in records):
        finish_transaction(journal, files)
        return

    can_complete = all(
        target_states[safe_target(root, str(record["path"]))] == str(record["after_sha256"])
        or observed(files[safe_target(root, str(record["path"]))][0]) == str(record["after_sha256"])
        for record in records
    )
    if can_complete:
        for record in records:
            target = safe_target(root, str(record["path"]))
            stage, _ = files[target]
            if observed(target) == str(record["before_sha256"]):
                os.replace(stage, target)
                fsync_dir(target.parent)
        for record in records:
            target = safe_target(root, str(record["path"]))
            if observed(target) != str(record["after_sha256"]):
                raise Refusal(f"recovery final hash mismatch: {target}")
        finish_transaction(journal, files)
        return

    can_rollback = all(
        target_states[safe_target(root, str(record["path"]))] == str(record["before_sha256"])
        or str(record["before_sha256"]) == ABSENT
        or observed(files[safe_target(root, str(record["path"]))][1]) == str(record["before_sha256"])
        for record in records
    )
    if can_rollback:
        for record in reversed(records):
            target = safe_target(root, str(record["path"]))
            _, backup = files[target]
            before = str(record["before_sha256"])
            if observed(target) != before:
                if before == ABSENT:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
                fsync_dir(target.parent)
        for record in records:
            target = safe_target(root, str(record["path"]))
            if observed(target) != str(record["before_sha256"]):
                raise Refusal(f"recovery rollback hash mismatch: {target}")
        finish_transaction(journal, files)
        return
    raise Refusal("prepared transaction can neither complete nor roll back exactly")


def execute(
    *,
    root: Path,
    contract_path: Path,
    receipt_path: Path,
    transform: str,
    expected_section: Mapping[str, object],
    builder: Callable[[Mapping[Path, bytes]], Mapping[Path, bytes]],
    apply: bool,
    script_path: Path | None = None,
) -> dict[str, object]:
    """Classify, validate, and optionally commit one exact transform set."""
    root = root.resolve(strict=True)
    receipt = read_image_receipt(receipt_path)
    _, contract_sha = load_contract(contract_path, transform, expected_section)
    records_object = expected_section.get("targets")
    if not isinstance(records_object, list) or not records_object:
        raise Refusal(f"{transform}: empty target contract")
    records: list[dict[str, object]] = records_object
    journal, files = transaction_names(root, transform, records)
    with root_lock(root):
        debris = transaction_debris(root, transform, journal, files)
        if journal.exists():
            if not apply:
                raise Refusal("prepared transaction requires --apply recovery")
            recover(root, transform, records, journal, files, contract_sha)
        elif debris:
            if not apply:
                raise Refusal("transaction artifacts require --apply recovery")
            recover_orphans(root, transform, records, files)
        before_state: dict[Path, bytes] = {}
        states: list[str] = []
        for record in records:
            target = safe_target(root, str(record["path"]))
            state = observed(target)
            before, after = str(record["before_sha256"]), str(record["after_sha256"])
            if before == after and state == before:
                data = target.read_bytes()
                compiled(target, data)
                check_seams(target, data, record.get("required_before_seams", []), "before")
                check_seams(target, data, record.get("required_after_seams", []), "after")
                check_forbidden(target, data, record.get("forbidden_after_seams", []))
                before_state[target] = data
            elif state == before:
                states.append("before")
                if before != ABSENT:
                    data = target.read_bytes()
                    compiled(target, data)
                    check_seams(target, data, record.get("required_before_seams", []), "before")
                    before_state[target] = data
            elif state == after:
                states.append("after")
                data = target.read_bytes()
                compiled(target, data)
                check_seams(target, data, record.get("required_after_seams", []), "after")
                check_forbidden(target, data, record.get("forbidden_after_seams", []))
            else:
                raise Refusal(f"{transform}: unknown target state {target}: {state}")
        if not states:
            raise Refusal(f"{transform}: no mutable targets")
        if len(set(states)) != 1:
            raise Refusal(f"{transform}: mixed before/after state")
        if states[0] == "after":
            state_name = "ALREADY_APPLIED"
        elif not apply:
            state_name = "CHECKED"
        else:
            outputs = dict(builder(before_state))
            if set(outputs) != set(files):
                raise Refusal(f"{transform}: builder target-set drift")
            for record in records:
                target = safe_target(root, str(record["path"]))
                data = outputs[target]
                compiled(target, data)
                if sha_bytes(data) != str(record["after_sha256"]):
                    raise Refusal(f"{transform}: generated after hash mismatch: {target}")
                check_seams(target, data, record.get("required_after_seams", []), "after")
                check_forbidden(target, data, record.get("forbidden_after_seams", []))
            for path, (stage, backup) in files.items():
                if stage.exists() or backup.exists():
                    raise Refusal(f"stale transaction artifact beside {path}")
            entries: list[dict[str, str]] = []
            try:
                for record in records:
                    target = safe_target(root, str(record["path"]))
                    stage, backup = files[target]
                    mode, uid, gid = metadata(target)
                    write_exact(stage, outputs[target], mode, uid, gid)
                    before = str(record["before_sha256"])
                    if before != ABSENT:
                        write_exact(backup, target.read_bytes(), mode, uid, gid)
                    fsync_dir(target.parent)
                    entries.append({
                        "path": str(record["path"]),
                        "before_sha256": before,
                        "after_sha256": str(record["after_sha256"]),
                        "stage_sha256": str(record["after_sha256"]),
                        "backup_sha256": before,
                    })
                atomic_json(journal, {
                    "schema_version": 1,
                    "transform": transform,
                    "contract_sha256": contract_sha,
                    "stage": "PREPARED",
                    "targets": entries,
                })
            except BaseException:
                if not journal.exists():
                    remove_artifacts(files)
                raise
            try:
                for record in records:
                    target = safe_target(root, str(record["path"]))
                    stage, _ = files[target]
                    os.replace(stage, target)
                    fsync_dir(target.parent)
                for record in records:
                    target = safe_target(root, str(record["path"]))
                    if observed(target) != str(record["after_sha256"]):
                        raise Refusal(f"post-commit hash mismatch: {target}")
            except BaseException:
                # Keep the exact journal/backups: the next --apply either
                # completes or rolls back deterministically; --check refuses.
                raise
            finish_transaction(journal, files)
            state_name = "APPLIED"
        script = (script_path or Path(sys.argv[0])).resolve()
        return {
            "schema_version": 1,
            "transform": transform,
            "state": state_name,
            "script_sha256": sha_file(script),
            "contract_sha256": contract_sha,
            "image_manifest": receipt["manifest_digest"],
            "image_config": receipt["config_digest"],
            "targets": [
                {
                    "path": str(record["path"]),
                    "before_sha256": str(record["before_sha256"]),
                    "after_sha256": str(record["after_sha256"]),
                    "observed_sha256": observed(safe_target(root, str(record["path"]))),
                }
                for record in records
            ],
        }


def print_receipt(value: object) -> None:
    sys.stdout.buffer.write(canonical(value))
