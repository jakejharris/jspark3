#!/usr/bin/env bash
# Wait until the cluster serves: on rank 0, until the API lists the model and answers a one-token request, which
# runs through all three ranks. Gives up after --timeout seconds (default 2400; the first start compiles kernels).
#
#   scripts/wait-ready.sh [--timeout SECONDS] [--dry-run]
#
# On rank 1 or 2 it only checks that this box's container is running: readiness is decided on rank 0.
# The engine compares the ranks' settings at start and refuses a mix of DRAFTER or SESSION_TIER values; this names
# the ranks that differ. It cannot see a WEIGHTS mix (same shapes): scripts/status.sh on each box shows each one's.
# Exit 0: ready. Exit 2: a container stopped, or the timeout passed; the message says what to do.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }
TIMEOUT=2400
parse_common "$@"
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --timeout) [[ ${2:-} =~ ^[0-9]+$ ]] || die "--timeout needs a number of seconds"; TIMEOUT=$2; shift 2 ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done
load_cluster

name=$(container_name "$RANK")
host=$API_HOST
[[ $host == 0.0.0.0 || $host == :: ]] && host=127.0.0.1
url=http://$host:$API_PORT

if [[ $DRY_RUN == 1 ]]; then
  info "dry run: would wait up to $TIMEOUT s for $name$([[ $RANK == 0 ]] && echo " and the API on port $API_PORT")"
  exit 0
fi

# The container must still be running; if it stopped, keep its last output in the private log and say so.
alive() {
  local state
  state=$(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$name" 2>/dev/null) \
    || die "$name is not on this box; start it with scripts/serve.sh"
  [[ $state == running* ]] && return 0
  log_open
  local tail
  tail=$(docker logs --tail 300 "$name" 2>&1) || true
  { echo "container state: $state"; printf '%s\n' "$tail"; } >>"$LOG"
  # The engine compares every rank's settings at start (draft model, session tier, context, ...) and refuses a mix.
  local mixed
  mixed=$(printf '%s\n' "$tail" | python3 -c '
import re, sys
for line in sys.stdin:
    if "the ranks were started with different settings" in line:
        rows = dict((int(r), row) for r, row in re.findall(r"rank (\d) (\[[^]]*\])", line))
        print(" ".join(str(r) for r in sorted(rows) if rows[r] != rows.get(0)) or "?")
        break
' 2>/dev/null) || mixed=''
  if [[ -n $mixed ]]; then
    die "the three ranks were started with different settings; rank(s) $mixed differ from rank 0. WEIGHTS, DRAFTER and SESSION_TIER must be the same in cluster.env on every box (scripts/status.sh shows each box's). Stop all three, fix cluster.env, start again"
  fi
  die "$name stopped (exit code ${state#* }). Its last output is in the log. Stop all three ranks (scripts/stop.sh on each box), then start ranks 2 and 1, then 0"
}

if [[ $RANK != 0 ]]; then
  alive
  info "$name is running; readiness is checked on rank 0"
  info "$(tier_line "$name")"
  exit 0
fi

probe() {  # 0: lists the model and answers one token; 1: not yet
  python3 - "$url" "$SERVE_NAME" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
url, name = sys.argv[1:]
models = json.load(urllib.request.urlopen(url + "/v1/models", timeout=5))
assert any(m.get("id") == name for m in models.get("data", []))
body = {"model": name, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 1, "temperature": 0}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
reply = json.load(urllib.request.urlopen(req, timeout=600))
assert reply["choices"][0]["message"] is not None
PY
}

start=$SECONDS
last=0
info "waiting for all three ranks to load and answer (up to $((TIMEOUT / 60)) min; the first start compiles kernels)"
while :; do
  alive
  probe && break
  elapsed=$((SECONDS - start))
  ((elapsed < TIMEOUT)) || die "not ready after $((TIMEOUT / 60)) min. On each box, scripts/status.sh shows its container; read 'docker logs $(container_name R)' there for the first error. Then stop all three and start again (followers first), or wait longer with scripts/wait-ready.sh --timeout"
  if ((elapsed - last >= 60)); then
    info "still loading, $((elapsed / 60)) min"
    last=$elapsed
  fi
  sleep 10
done
info "ready after $(((SECONDS - start) / 60)) min: the API on port $API_PORT answered through all three ranks"
info "$(tier_line "$name")"
