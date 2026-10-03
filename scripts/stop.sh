#!/usr/bin/env bash
# Stop and remove this box's rank container (<CONTAINER_PREFIX>-rank<R>, started by serve.sh). Containers this recipe
# did not start are never touched, nor are weights, caches or the session tier. Order between boxes doesn't matter;
# if one rank fails, stop all three.
#
#   scripts/stop.sh [--keep] [--container-prefix NAME] [--dry-run]
#
# --keep stops the container but keeps it, with its logs. serve.sh will not reuse that name: remove it later with
# scripts/stop.sh, or start the next run under another prefix (--container-prefix, or CONTAINER_PREFIX).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }
KEEP=0
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --keep) KEEP=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
load_cluster

name=$(container_name "$RANK")
if [[ $DRY_RUN == 0 ]]; then
  read -r state label <<<"$(container_state "$name")"
  if [[ $state == absent ]]; then
    info "$name is not on this box; nothing to stop"
    exit 0
  fi
  [[ -n ${label:-} ]] || die "$name was not started by this recipe; it is left alone (CONTAINER_PREFIX in cluster.env names this recipe's containers)"
fi
# Act on the full container ID read just now, so nothing else that takes the name meanwhile is touched.
id=$name
[[ $DRY_RUN == 1 ]] || id=$(docker inspect --format '{{.Id}}' "$name")
if [[ $KEEP == 1 ]]; then
  if [[ $DRY_RUN == 1 || $state == running || $state == restarting || $state == paused ]]; then
    run docker stop "$id" || die "docker could not stop $name"
  fi
  [[ $DRY_RUN == 1 ]] && exit 0
  info "$name stopped and kept (with its logs); start the next run under another prefix, or remove it with scripts/stop.sh"
  exit 0
fi
run docker rm -f "$id" || die "docker could not remove $name"
[[ $DRY_RUN == 1 ]] && exit 0
info "$name stopped and removed"
