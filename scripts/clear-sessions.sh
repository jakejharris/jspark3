#!/usr/bin/env bash
# Delete the conversation state the session tier keeps on this box's disk. Stop this box's rank first; clear all
# three boxes while all three are stopped, since each rank keeps its own part.
#
#   scripts/clear-sessions.sh [--weights base|ablit] [--drafter dflash2|none] [--all] [--dry-run]
#
# Without --all: the current weights and drafter combination, $DATA/sessions/<weights>-<drafter>/<engine build>.
# --all: everything under $DATA/sessions (every combination and every earlier engine build).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }
ALL=0
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --all) ALL=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
load_cluster

name=$(container_name "$RANK")
if [[ $DRY_RUN == 0 && $(container_state "$name") != absent ]]; then
  die "$name exists on this box: stop it first (scripts/stop.sh), on all three boxes"
fi
if [[ $ALL == 1 ]]; then
  target=$DATA/sessions
else
  target=$(session_dir)
fi
if [[ $DRY_RUN == 0 && ! -d $target ]]; then
  info "nothing to clear at ${target/#"$DATA"/\$DATA}"
  exit 0
fi
size=$([[ -d $target ]] && du -sh "$target" 2>/dev/null | cut -f1 || echo 0)
# The engine writes as root inside the container, so delete through the same image rather than needing sudo.
run docker run --rm --network none -v "$(dirname "$target")":/parent "$IMAGE" rm -rf "/parent/$(basename "$target")" \
  || die "could not delete the session files"
[[ $DRY_RUN == 1 ]] && exit 0
info "cleared ${target/#"$DATA"/\$DATA} ($size)"
