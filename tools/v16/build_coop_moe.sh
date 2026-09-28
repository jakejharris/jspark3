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
# complete campaign through qualify_coop.py; profile-only seals are refused.
set -euo pipefail
if [[ ${JSPARK_PRIVATE_BUILD_LOG:-0} != 1 ]]; then
  diagnostic_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  exec python3 -B "$diagnostic_root/recipe/scripts/_diagnostics.py" "$0" "$@"
fi
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
  scp -q "$root/tools/v16/coop_h1_control.py" "$root/recipe/scripts/_diagnostics.py" "$host:$remote/"
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
  echo "REFUSE: profile-only legacy seals are retired; run qualify_coop.py --seal COMPLETED_CAMPAIGN --output NEW_DIRECTORY on the qualification host" >&2
  exit 9
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
