#!/usr/bin/env bash
# Reproducibly build, qualify, and seal JSpark3 v1.6's TP3 ABI2 cooperative
# MoE bundle inside the pinned serving image on one Spark.
#
#   build_coop_moe.sh HOST build CAMPAIGN
#   build_coop_moe.sh HOST seal CAMPAIGN PROFILE_LOG_DIR [--write|--check]
#   build_coop_moe.sh HOST cleanup CAMPAIGN
#
# build compiles twice at the identical /w path in fresh --network=none
# containers and requires byte-identical .so files. seal accepts only the nine
# complete rank{0..2}-geo{0..2}.jsonl profiles, binds a newly selected policy to
# that binary, and then writes/checks the local recipe artifact record.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

host=${1:?ssh host required}
action=${2:?build, seal, or cleanup required}
campaign=${3:?campaign id required}
if [[ ! $campaign =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]]; then
  echo "REFUSE: campaign must match [a-z0-9][a-z0-9-]{0,47}" >&2
  exit 9
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
coop=$root/recipe/overlays/v16/coop
identity_args=()
if [[ -n ${JSPARK_IMAGE_RECEIPT:-} ]]; then identity_args=(--receipt "$JSPARK_IMAGE_RECEIPT"); fi
image=$(python3 -B "$root/recipe/scripts/_image_identity.py" "${identity_args[@]}")
remote=jspark3-v16-coop-$campaign

build_remote() {
  ssh -o BatchMode=yes "$host" "mkdir ~/$remote"
  scp -q "$root/tools/v16/coop_h1_control.py" "$host:$remote/"
  for run in a b; do
    ssh -o BatchMode=yes "$host" "mkdir ~/$remote/$run"
    scp -q "$coop/build_repro.sh" "$coop/SOURCE_MANIFEST.json" "$host:$remote/$run/"
    scp -qr "$coop/source" "$host:$remote/$run/"
  done
  ssh -o BatchMode=yes "$host" bash -s -- "$remote" "$image" <<'REMOTE'
set -euo pipefail
work=$1
image=$2
cd ~/$work
for run in a b; do
  docker run --rm --network none --user "$(id -u):$(id -g)" \
    -v "$PWD/$run:/w" -w /w --entrypoint bash "$image" \
    /w/build_repro.sh /w/out
done
a=a/out/cooperative_moe.so
b=b/out/cooperative_moe.so
if ! cmp -s "$a" "$b"; then
  echo "REFUSE: cooperative_moe.so is not bit reproducible" >&2
  sha256sum "$a" "$b" >&2
  cmp -l "$a" "$b" | head -40 >&2 || true
  exit 9
fi
sha256sum "$a"
REMOTE
  printf 'PASS remote_candidate=~/%s/a/out image=%s builds=2 comparison=bit-identical\n' "$remote" "$image"
}

seal_remote() {
  logs=${4:?profile log directory required}
  mode=${5:---check}
  if [[ $mode != --write && $mode != --check ]]; then
    echo "REFUSE: seal mode must be --write or --check" >&2
    exit 9
  fi
  if [[ ! -d $logs || -L $logs ]]; then
    echo "REFUSE: profile log directory missing or symlinked" >&2
    exit 9
  fi
  expected=()
  for rank in 0 1 2; do
    for geometry in 0 1 2; do
      name=rank${rank}-geo${geometry}.jsonl
      path=$logs/$name
      if [[ ! -f $path || -L $path ]]; then
        echo "REFUSE: missing regular profile $name" >&2
        exit 9
      fi
      expected+=("$path")
    done
  done
  if [[ $(find "$logs" -maxdepth 1 -type f -name '*.jsonl' | wc -l) != 9 ]]; then
    echo "REFUSE: profile directory must contain exactly nine JSONL files" >&2
    exit 9
  fi

  source_sha=$(sha256sum "$coop/SOURCE_MANIFEST.json" | cut -d ' ' -f1)
  builder_sha=$(sha256sum "$coop/build_repro.sh" | cut -d ' ' -f1)
  ssh -o BatchMode=yes "$host" "mkdir -p ~/$remote/profiles"
  scp -q "${expected[@]}" "$host:$remote/profiles/"
  ssh -o BatchMode=yes "$host" bash -s -- "$remote" "$image" "$source_sha" "$builder_sha" <<'REMOTE'
set -euo pipefail
work=$1
image=$2
source_sha=$3
builder_sha=$4
cd ~/$work
# Recheck both artifacts at seal time; a stale/partial campaign cannot claim
# two-build identity just because it has a usable a/out directory.
for run in a b; do
  test -f "$run/out/cooperative_moe.so" && test ! -L "$run/out/cooperative_moe.so"
  test "$(sha256sum "$run/SOURCE_MANIFEST.json" | cut -d ' ' -f1)" = "$source_sha"
  test "$(sha256sum "$run/build_repro.sh" | cut -d ' ' -f1)" = "$builder_sha"
done
cmp -s a/out/cooperative_moe.so b/out/cooperative_moe.so || {
  echo "REFUSE: seal requires two matching native builds" >&2
  exit 9
}
profiles=()
for rank in 0 1 2; do
  for geometry in 0 1 2; do
    profiles+=("/profiles/rank${rank}-geo${geometry}.jsonl")
  done
done
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/a:/w" -v "$PWD/profiles:/profiles:ro" -w /w --entrypoint python3 "$image" \
  /w/source/select_policy.py --bundle /w/out "${profiles[@]}"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/a:/w" -w /w --entrypoint python3 "$image" \
  /w/source/manifest.py verify-artifacts /w/out
REMOTE

  scratch=$(mktemp -d)
  trap 'rm -rf "$scratch"' EXIT
  scp -qr "$host:$remote/a/out" "$scratch/bundle"
  PYTHONDONTWRITEBYTECODE=1 python3 - "$mode" "$coop" "$scratch/bundle" "$logs" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

mode, coop, bundle, logs = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((bundle / "manifest.json").read_text())
for relative, expected in manifest["files"].items():
    path = bundle / relative
    if not path.resolve().is_relative_to(bundle.resolve()) or path.is_symlink() or sha(path) != expected:
        raise SystemExit(f"REFUSE: fetched bundle drift: {relative}")
native = manifest["files"]["cooperative_moe.so"]
policy = json.loads((bundle / "dispatch_policy.json").read_text())
if policy.get("native_sha256") != native:
    raise SystemExit("REFUSE: selected row policy is not bound to the rebuilt binary")
profiles = {
    path.name: sha(path)
    for path in sorted(logs.glob("*.jsonl"))
}
if len(profiles) != 9:
    raise SystemExit("REFUSE: expected exactly nine profile hashes")
sys.path.insert(0, str(coop.parents[2] / "scripts"))
from _image_identity import read_operator_record, selected_identity
identity = read_operator_record(Path(os.environ["JSPARK_IMAGE_RECEIPT"])) if os.environ.get("JSPARK_IMAGE_RECEIPT") else selected_identity()
build_image = {"manifest": identity["manifest_digest"].removeprefix("sha256:"),
               "config": identity["config_digest"].removeprefix("sha256:")}
record = {
    "schema_version": 1,
    "image": build_image,
    "source_manifest_sha256": sha(coop / "SOURCE_MANIFEST.json"),
    "reproducibility": {
        "runs": 2,
        "comparison": "bit-identical",
        "binary_sha256": native,
    },
    "qualification": {
        "ep_ranks": [0, 1, 2],
        "geometries": [0, 1, 2],
        "cases": 468,
        "profile_log_sha256": profiles,
    },
    "bundle": {
        "manifest_sha256": sha(bundle / "manifest.json"),
        "native_sha256": native,
        "runtime_sha256": manifest["files"]["runtime.py"],
        "dispatch_policy_sha256": manifest["files"]["dispatch_policy.json"],
    },
}
record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
# Use the same fail-closed validator as boot before any local seal mutation.
candidate_record = bundle.parent / "BUILD.json"
candidate_record.write_bytes(record_bytes)
sys.path.insert(0, str(coop.parents[2] / "scripts"))
from apply_coop_moe import verify_bundle
verify_bundle(bundle, candidate_record, build_image=build_image)
target_bundle = coop / "bundle"
target_record = coop / "BUILD.json"

def tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }

if mode == "--write":
    if target_record.is_symlink() or (target_record.exists() and (
        not target_record.is_file() or target_record.read_bytes() != record_bytes
    )):
        raise SystemExit("REFUSE: existing BUILD.json differs; remove it explicitly before repinning")
    if target_bundle.exists():
        if not target_bundle.is_dir() or target_bundle.is_symlink() or tree_hashes(target_bundle) != tree_hashes(bundle):
            raise SystemExit("REFUSE: existing sealed bundle differs; remove it explicitly before repinning")
    else:
        shutil.copytree(bundle, target_bundle)
    target_record.write_bytes(record_bytes)
else:
    if not target_bundle.is_dir() or target_bundle.is_symlink() or tree_hashes(target_bundle) != tree_hashes(bundle):
        raise SystemExit("REFUSE: rebuilt bundle does not match the sealed recipe bundle")
    if not target_record.is_file() or target_record.is_symlink():
        raise SystemExit("REFUSE: missing or unsafe BUILD.json")
    expected_record = json.loads(target_record.read_text())
    if {k: v for k, v in expected_record.items() if k != "image"} != {k: v for k, v in record.items() if k != "image"}:
        raise SystemExit("REFUSE: rebuilt record does not match BUILD.json")
print(json.dumps({"status": "PASS", "mode": mode, "native_sha256": native, "image": build_image}, sort_keys=True))
PY
}

cleanup_remote() {
  ssh -o BatchMode=yes "$host" bash -s -- "$remote" <<'REMOTE'
set -euo pipefail
work=$1
case "$work" in
  jspark3-v16-coop-[a-z0-9]*) ;;
  *) echo "REFUSE: unsafe cleanup path" >&2; exit 9 ;;
esac
test -d ~/$work
rm -rf ~/$work
REMOTE
}

case $action in
  build) build_remote ;;
  seal) seal_remote "$@" ;;
  cleanup) cleanup_remote ;;
  *) echo "REFUSE: action must be build, seal, or cleanup" >&2; exit 9 ;;
esac
