#!/usr/bin/env bash
# Build the engine wheel from this tree's engine/ source inside the pinned image (no network; nothing is compiled
# on the host: CUDA kernels compile at first start) and print its content digest. The wheel must match
# WHEEL_CONTENT_SHA256 in pins.env, the content of the wheel the published numbers were measured with.
# Run on every box, or build once and copy wheels/ to the others.
#
#   scripts/build-wheel.sh [--dry-run]
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done

wheel=$HERE/wheels/$ENGINE_WHEEL
if [[ $DRY_RUN == 0 ]]; then
  [[ -f $HERE/engine/pyproject.toml ]] || die "this tree has no engine/ source; use a complete release archive"
  image_ok || die "pull the pinned image first: scripts/pull-image.sh"
  if [[ -f $wheel ]] && wheel_ok 2>/dev/null; then
    info "wheels/$ENGINE_WHEEL is already the pinned build"
    exit 0
  fi
  mkdir -p "$HERE/wheels"
  rm -f "$wheel"
fi
warn_if_serving
info "building the engine wheel inside the pinned image (no network)"
run docker run --rm --network none -v "$HERE/engine":/src:ro -v "$HERE/wheels":/out "$IMAGE" bash -c \
  "cp -r /src /tmp/src && pip wheel -q --no-deps --no-build-isolation --no-index -w /out /tmp/src && chown $(id -u):$(id -g) /out/$ENGINE_WHEEL" \
  || die "the wheel build failed"
[[ $DRY_RUN == 1 ]] && exit 0
[[ -f $wheel ]] || die "the build did not produce wheels/$ENGINE_WHEEL"
digest=$(python3 "$HERE/scripts/wheel-content.py" "$wheel" | awk '{print $1}')
info "wheels/$ENGINE_WHEEL content sha256 $digest"
[[ $WHEEL_CONTENT_SHA256 != PENDING ]] || die "pins.env WHEEL_CONTENT_SHA256 is PENDING: this tree has no pinned engine build to compare with"
[[ $digest == "$WHEEL_CONTENT_SHA256" ]] \
  || die "the built wheel's content is not WHEEL_CONTENT_SHA256 in pins.env: do not serve it. Check engine/ against SHA256SUMS (python3 tools/payload.py verify)"
info "the content matches the pinned build"
