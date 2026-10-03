#!/usr/bin/env bash
# Run every script's help, dry run and safe refusal paths with sentinel settings (a fake token, addresses and data
# folder) and fail if any terminal output contains one of them, this machine's home folder, the recipe's folder or
# the user name. Also checks that the private logs are mode 0600. Needs no data; starts and removes no containers.
#
#   tests/check-leaks.sh
set -uo pipefail
# shellcheck source=clean-env.sh
source "$(dirname "$0")/clean-env.sh"
HERE=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

data=$tmp/sentinel-data-q7k
sed -e "s|^DATA=[^ ]*|DATA=$data|" -e 's|^MASTER_ADDR=[^ ]*|MASTER_ADDR=198.51.100.73|' \
    -e 's|^PREV_PEER=|PREV_PEER=203.0.113.18|' -e 's|^NEXT_PEER=|NEXT_PEER=203.0.113.19|' \
    "$HERE/cluster.env.example" >"$tmp/rank0.env"
sed 's|^RANK=0 |RANK=1 |' "$tmp/rank0.env" >"$tmp/rank1.env"
export HF_TOKEN=sentinel-token-not-real-q7k
sentinels=("$HF_TOKEN" "$data" sentinel-data-q7k 198.51.100.73 203.0.113.18 203.0.113.19 "$HOME" "$HERE")
words=("${USER:-$(id -un)}")

out=$tmp/out.txt
: >"$out"
run() {  # env-file command...: record everything it prints
  local env=$1; shift
  local shown="$*"
  { echo "### ${shown//"$HERE"/<recipe>}"; CLUSTER_ENV=$tmp/$env "$@" 2>&1; echo "### rc=$?"; } >>"$out"
}
S=$HERE/scripts
for s in "$S"/*.sh; do
  [[ $s == "$S/lib.sh" ]] && continue
  run rank0.env "$s" --help
  [[ $s == "$S/host-prep.sh" ]] || run rank0.env "$s" --dry-run
done
run rank0.env "$S/serve.sh" --dry-run --drafter none --session-tier off
run rank1.env "$S/serve.sh" --dry-run
run rank0.env "$S/fetch-weights.sh" --verify-only
run rank0.env "$S/fetch-weights.sh" --weights ablit --verify-only
run rank0.env "$S/split.sh"
run rank0.env "$S/split.sh" --verify-only
run rank1.env "$S/wait-ready.sh" --timeout 1
run rank0.env "$S/status.sh"
run rank0.env "$S/clear-sessions.sh"
run rank0.env "$S/serve.sh" --bogus-option
run rank0.env python3 "$S/preflight.py" --for fetch
run rank1.env python3 "$S/preflight.py" --for split

bad=0
for value in "${sentinels[@]}"; do
  [[ ${#value} -ge 4 ]] || continue
  if grep -nF -- "$value" "$out" >/dev/null; then
    echo "FAIL  output contains a sentinel or local value (first line: $(grep -nF -- "$value" "$out" | head -1 | cut -d: -f1))"
    bad=1
  fi
done
for value in "${words[@]}"; do
  [[ ${#value} -ge 3 ]] || continue
  if grep -nw -- "$value" "$out" >/dev/null; then
    echo "FAIL  output contains the user name (first line: $(grep -nw -- "$value" "$out" | head -1 | cut -d: -f1))"
    bad=1
  fi
done
# Control: the sentinel values did reach the scripts' messages, in their redacted form.
# shellcheck disable=SC2016  # literal names: the redacted form, not an expansion
for name in '$DATA' '$MASTER_ADDR'; do
  grep -qF -- "$name" "$out" || { echo "FAIL  no output names $name, so this test exercised nothing"; bad=1; }
done
[[ $bad == 0 ]] && echo "PASS  no token, address, user name, home or recipe path in $(grep -c '^### rc=' "$out") script runs"
for log in "$HERE"/logs/*.log; do
  [[ -e $log ]] || continue
  mode=$(stat -c %a "$log")
  [[ $mode == 600 ]] || { echo "FAIL  logs/${log##*/} has mode $mode, not 600"; bad=1; }
done
[[ $bad == 0 ]] && echo "PASS  every private log is mode 0600"
exit $bad
