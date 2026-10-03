#!/usr/bin/env bash
# Pull the pinned NVIDIA PyTorch image by digest (anonymous; no NGC login needed) and check its image ID.
# Run on every box. The image is used under NVIDIA's terms (THIRD_PARTY_NOTICES.md).
#
#   scripts/pull-image.sh [--dry-run]
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; }
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done

if [[ $DRY_RUN == 0 ]] && image_ok 2>/dev/null; then
  info "the pinned image is already here (ID ${IMAGE_ID:7:12})"
  exit 0
fi
info "pulling ${IMAGE%@*} at ${IMAGE#*@} (progress in the log)"
run docker pull "$IMAGE" || die "docker could not pull the image; check this box's internet access and docker login state"
[[ $DRY_RUN == 1 ]] && exit 0
image_ok || die "the pulled image's ID is not IMAGE_ID in pins.env; do not use it (docker image rm it and pull again)"
info "image pulled and its ID matches pins.env"
