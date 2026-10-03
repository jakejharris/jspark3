#!/usr/bin/env bash
# Put this box's memory in the state the published numbers were measured from. Needs sudo. Run it once after each
# boot, after building and splitting and before the first start (running it again before another start is
# harmless). Persists nothing: a reboot undoes it.
#
#   scripts/host-prep.sh [--dry-run]
#
# It turns swap off, sets vm.compaction_proactiveness=0, flushes and drops the page cache, and compacts free memory.
# The compaction setting is the part that matters for stable serving: with background compaction on, a box that
# has just built or split can stall the server (preflight warns when it is not 0). The rest only matches the
# measured host state: the engine sizes its memory from what the kernel reports as available, which counts
# reclaimable page cache, and reads conversation state with direct IO, so a warm page cache does not change what
# the server does.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
sudo=(sudo)
[[ $(id -u) == 0 ]] && sudo=()
run "${sudo[@]}" swapoff -a || die "swapoff failed"
run "${sudo[@]}" sysctl -q -w vm.compaction_proactiveness=0 || die "could not set vm.compaction_proactiveness"
run sync
run "${sudo[@]}" sh -c 'echo 3 > /proc/sys/vm/drop_caches' || die "could not drop the page cache"
run "${sudo[@]}" sh -c 'echo 1 > /proc/sys/vm/compact_memory' || die "could not compact memory"
[[ $DRY_RUN == 1 ]] && exit 0
info "done: swap off, proactive compaction off, page cache dropped, memory compacted (until the next reboot)"
