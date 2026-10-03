#!/usr/bin/env bash
# Download the pinned PyPI wheels (wheels.lock: the engine's dependencies, the same files the measured boot
# installed) into wheels/ and check each file's sha256. Run on every box.
#
#   scripts/fetch-wheels.sh [--verify-only] [--dry-run]
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; }
VERIFY_ONLY=0
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
if [[ $DRY_RUN == 1 ]]; then
  run python3 - wheels.lock wheels "$VERIFY_ONLY"
  exit 0
fi
mkdir -p "$HERE/wheels"
log_open
python3 - "$HERE/wheels.lock" "$HERE/wheels" "$VERIFY_ONLY" 2>>"$LOG" <<'PY' || die "the wheels in wheels/ do not match wheels.lock (names above); delete those files and run this again"
import hashlib, json, os, sys, urllib.request
lock, out, verify_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
bad = []
for line in open(lock):
    if not line.strip() or line.startswith("#"):
        continue
    name, want = line.split()
    want = want.split(":", 1)[1]
    path = os.path.join(out, name)
    if not os.path.exists(path):
        if verify_only:
            bad.append(name + " (missing)")
            continue
        project, version = name.split("-")[:2]
        release = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{project}/{version}/json", timeout=60))
        url = next(u["url"] for u in release["urls"] if u["filename"] == name)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != want:
        bad.append(name)
for name in bad:
    print("fetch-wheels.sh: does not match wheels.lock: " + name, file=sys.stderr)
sys.exit(1 if bad else 0)
PY
info "every wheel in wheels.lock is in wheels/ and matches its sha256"
