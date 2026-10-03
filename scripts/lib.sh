# Shared by the scripts in this directory: configuration, the data layout, output and small helpers.
# Source it; don't run it.
#
# Output rule: what a script prints says what failed and what to do, in terms of setting names and paths under
# $DATA, never the values that identify you (your paths, user name, addresses, tokens). Command output and other
# free-form detail go to a private log, logs/<script>.log in the recipe directory (mode 0600), which the message
# names. Share terminal output freely; read a log before you share it.
# shellcheck shell=bash

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../pins.env
source "$HERE/pins.env"
# shellcheck source=../config/defaults.env
source "$HERE/config/defaults.env"
# shellcheck source=../config/serve.conf
source "$HERE/config/serve.conf"

DRY_RUN=${DRY_RUN:-0}
OPT_WEIGHTS=''
OPT_DRAFTER=''
OPT_SESSION_TIER=''
OPT_CONTAINER_PREFIX=''
SCRIPT=${0##*/}
LOG_REL=logs/${SCRIPT%.*}.log
LOG=$HERE/$LOG_REL

# Replace identifying values with the names of the settings that hold them.
redact() {
  local text=$*
  local name value
  for name in HF_TOKEN DATA HERE HOME MASTER_ADDR PREV_PEER NEXT_PEER USER; do
    value=${!name:-}
    [[ ${#value} -ge 2 ]] || continue
    case $name in
      HERE) text=${text//"$value"/<recipe>} ;;
      HOME) text=${text//"$value"/'~'} ;;
      USER) text=${text//"$value"/<user>} ;;
      *) text=${text//"$value"/\$$name} ;;
    esac
  done
  printf '%s' "$text"
}

log_open() {
  [[ -n ${LOG_OPENED:-} ]] && return 0
  (umask 077 && mkdir -p "$HERE/logs" && touch "$LOG") || return 0
  chmod 600 "$LOG" 2>/dev/null || true
  printf '\n== %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SCRIPT" >>"$LOG"
  LOG_OPENED=1
}

info() { printf '%s: %s\n' "$SCRIPT" "$(redact "$*")"; }
warn() { printf '%s: warning: %s\n' "$SCRIPT" "$(redact "$*")" >&2; }
die() {
  printf '%s: %s\n' "$SCRIPT" "$(redact "$*")" >&2
  [[ -n ${LOG_OPENED:-} ]] && printf '%s: details: %s\n' "$SCRIPT" "$LOG_REL" >&2
  exit 2
}

# Run a command. Under --dry-run, print it (redacted) instead. Otherwise its output goes to the private log.
run() {
  if [[ $DRY_RUN == 1 ]]; then
    local quoted
    quoted=$(printf ' %q' "$@")
    printf '+%s\n' "$(redact "$quoted")"
    return 0
  fi
  log_open
  printf '+ %s\n' "$(printf '%q ' "$@")" >>"$LOG"
  "$@" >>"$LOG" 2>&1
}

# Like run, but prints a progress line every 30 seconds with how much DIR holds so far (of TOTAL bytes, if known).
run_watch() {
  local label=$1 dir=$2 total=$3
  shift 3
  if [[ $DRY_RUN == 1 ]]; then run "$@"; return; fi
  log_open
  printf '+ %s\n' "$(printf '%q ' "$@")" >>"$LOG"
  "$@" >>"$LOG" 2>&1 &
  local pid=$! start=$SECONDS waited
  while kill -0 "$pid" 2>/dev/null; do
    waited=0
    while ((waited < 30)) && kill -0 "$pid" 2>/dev/null; do sleep 1; waited=$((waited + 1)); done
    kill -0 "$pid" 2>/dev/null || break
    local have
    have=$(du -sb "$dir" 2>/dev/null | cut -f1) || have=0
    if [[ $total -gt 0 ]]; then
      printf '%s: %s: %.1f of %.1f GB, %d min\n' "$SCRIPT" "$label" "$(bc_div "$have")" "$(bc_div "$total")" $(((SECONDS - start) / 60))
    else
      printf '%s: %s: %.1f GB so far, %d min\n' "$SCRIPT" "$label" "$(bc_div "$have")" $(((SECONDS - start) / 60))
    fi
  done
  wait "$pid"
}
bc_div() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1e9 }'; }

# Options every script accepts. Call with "$@"; sets REST to whatever it didn't consume.
parse_common() {
  REST=()
  while (($#)); do
    case $1 in
      --weights) [[ $# -ge 2 ]] || die '--weights needs base or ablit'; OPT_WEIGHTS=$2; shift 2 ;;
      --weights=*) OPT_WEIGHTS=${1#*=}; shift ;;
      --drafter) [[ $# -ge 2 ]] || die '--drafter needs dflash2 or none'; OPT_DRAFTER=$2; shift 2 ;;
      --drafter=*) OPT_DRAFTER=${1#*=}; shift ;;
      --session-tier) [[ $# -ge 2 ]] || die '--session-tier needs disk or off'; OPT_SESSION_TIER=$2; shift 2 ;;
      --session-tier=*) OPT_SESSION_TIER=${1#*=}; shift ;;
      --container-prefix) [[ $# -ge 2 ]] || die '--container-prefix needs a name'; OPT_CONTAINER_PREFIX=$2; shift 2 ;;
      --container-prefix=*) OPT_CONTAINER_PREFIX=${1#*=}; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      *) REST+=("$1"); shift ;;
    esac
  done
}

# Read cluster.env (cluster.env.example under --dry-run when cluster.env doesn't exist yet), then apply the
# command-line options and the defaults. Validates every value the scripts use.
# --verify-third defers ablit availability to split.sh's per-rank check, after it revokes the old marker.
load_cluster() {
  local file=${CLUSTER_ENV:-$HERE/cluster.env}  # CLUSTER_ENV: another file, for tests
  if [[ ! -f $file ]]; then
    [[ $DRY_RUN == 1 ]] || die "no cluster.env: cp cluster.env.example cluster.env and fill it in (INSTALL.md)"
    file=$HERE/cluster.env.example
    info "dry run: no cluster.env, using cluster.env.example"
  fi
  # shellcheck source=../cluster.env.example
  source "$file"
  WEIGHTS=${OPT_WEIGHTS:-${WEIGHTS:-$WEIGHTS_DEFAULT}}
  DRAFTER=${OPT_DRAFTER:-${DRAFTER:-$DRAFTER_DEFAULT}}
  SESSION_TIER=${OPT_SESSION_TIER:-${SESSION_TIER:-$SESSION_TIER_DEFAULT}}
  CONTAINER_PREFIX=${OPT_CONTAINER_PREFIX:-${CONTAINER_PREFIX:-$CONTAINER_PREFIX_DEFAULT}}
  MASTER_PORT=${MASTER_PORT:-$SERVE_MASTER_PORT}
  API_PORT=${API_PORT:-$SERVE_API_PORT}
  API_HOST=${API_HOST:-$SERVE_API_HOST}
  [[ $WEIGHTS == base || $WEIGHTS == ablit ]] || die "WEIGHTS must be base or ablit"
  [[ $DRAFTER == dflash2 || $DRAFTER == none ]] || die "DRAFTER must be dflash2 or none"
  [[ $SESSION_TIER == disk || $SESSION_TIER == off ]] || die "SESSION_TIER must be disk or off"
  [[ $CONTAINER_PREFIX =~ ^[a-z0-9][a-z0-9_.-]{0,40}$ ]] || die "CONTAINER_PREFIX must be lower-case letters, digits and _.- (at most 41)"
  [[ ${RANK:-} =~ ^[012]$ ]] || die "RANK must be 0, 1 or 2"
  [[ -n ${DATA:-} && $DATA == /* ]] || die "DATA must be an absolute path"
  local name
  for name in DATA MASTER_ADDR MASTER_PORT LAN_IFACE PREV_IFACE NEXT_IFACE API_PORT API_HOST; do
    [[ ${!name:-} =~ ^[A-Za-z0-9_./:-]+$ ]] || die "$name in cluster.env is empty or has characters the scripts don't accept"
  done
  for name in PREV_PEER NEXT_PEER; do
    [[ -z ${!name:-} || ${!name} =~ ^[A-Za-z0-9_.:-]+$ ]] || die "$name in cluster.env has characters the scripts don't accept"
  done
  if [[ ${1:-} != --verify-third && $WEIGHTS == ablit && ! -f $HERE/manifests/ablit/rank0.sha256 ]]; then
    die "--weights ablit is not available in this tree (no manifests/ablit); use base"
  fi
}

# Downloading and converting the ablit weights needs the pinned converter and the input manifests. Checking and
# serving an ablit third needs only manifests/ablit/rank<R>.sha256.
ablit_ready() {
  [[ $ABLIT_SOURCE_REPO != PENDING && $ABLIT_SOURCE_REV != PENDING && $ABLIT_CONVERTER_SHA256 != PENDING
     && $ABLIT_CONVERSION_ID != PENDING
     && -f $HERE/manifests/inputs/ablit-source.sha256 && -f $HERE/manifests/inputs/ablit-weights.sha256 ]]
}
# shellcheck disable=SC2034  # used by the scripts that source this file
ABLIT_NOT_READY="the ablit download and conversion are not pinned in this tree yet. A verified ablit third you already have can be checked with scripts/split.sh --weights ablit --verify-only"


# The data layout under $DATA (INSTALL.md, "Where things live").
variant_dir() { echo "$DATA/${1:-$WEIGHTS}"; }
snapshot_dir() { echo "$DATA/${1:-$WEIGHTS}/weights"; }
rank_dir() { echo "$DATA/${2:-$WEIGHTS}/rank$1"; }
drafter_dir() { echo "$DATA/drafter"; }
session_dir() { echo "$DATA/sessions/$WEIGHTS-$DRAFTER/$SERVE_SESSION_NAMESPACE"; }
container_name() { echo "$CONTAINER_PREFIX-rank$1"; }
# The per-rank output manifest a split must reproduce.
rank_manifest() { echo "$HERE/manifests/${2:-$WEIGHTS}/rank$1.sha256"; }

# The tuned settings for the chosen weights (INSTALL.md, "Tuned settings per weights"): config/profiles/<weights>.env.
profile_rel() { echo "config/profiles/${1:-$WEIGHTS}.env"; }
# Reads and checks it: it holds exactly SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY, which this sets. Anything else
# stops here, before a container starts: every engine setting has one source, config/serve.env, so a profile cannot
# change the engine's numerics. These rules also describe the installed profile files.
load_profile() {
  local rel file line key value digits n=0 seen=' '
  rel=$(profile_rel)
  file=$HERE/$rel
  [[ -f $file ]] || die "$rel is missing from this tree; restore it: git checkout config/profiles/"
  # shellcheck disable=SC2034  # read by serve.sh
  SERVE_DRAFT_POLICY='' NO_DRAFTER_POLICY=''
  while IFS= read -r line || [[ -n $line ]]; do
    n=$((n + 1))
    [[ -z $line || $line == '#'* ]] && continue
    [[ $line =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.:-]*)$ ]] || die "$rel line $n is not KEY=VALUE"
    key=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
    [[ $seen != *" $key "* ]] || die "$rel sets $key twice"
    seen+="$key "
    case $key in
      SERVE_DRAFT_POLICY) [[ $value =~ ^f[qc]7:0\.([0-9]+)$ ]] || die "$rel: $key must be fq7:<c> or fc7:<c>, c from 0.1 to 0.9" ;;
      NO_DRAFTER_POLICY) [[ $value =~ ^c7:0\.([0-9]+)$ ]] || die "$rel: $key must be c7:<c>, c from 0.1 to 0.9" ;;
      *) die "$rel sets $key: a profile holds SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only; every other setting is in config/serve.env" ;;
    esac
    digits=${BASH_REMATCH[1]}  # 0.<digits>: from 0.1 to 0.9
    [[ $digits =~ ^[1-8] || $digits =~ ^90*$ ]] || die "$rel: the $key confidence must be from 0.1 to 0.9"
    printf -v "$key" '%s' "$value"
  done <"$file"
  for key in SERVE_DRAFT_POLICY NO_DRAFTER_POLICY; do
    [[ $seen == *" $key "* ]] || die "$rel does not set $key; restore it: git checkout config/profiles/"
  done
  if grep -qx 'TF_GLM_REPLY_PREFILL=1' "$HERE/config/serve.env" && ! grep -qx 'TF_GLM_FAIR_SCHED=1' "$HERE/config/serve.env"; then
    die "config/serve.env: TF_GLM_REPLY_PREFILL=1 needs TF_GLM_FAIR_SCHED=1 (the engine refuses to start otherwise)"
  fi
}

# The sha256 SHA256SUMS records for a file of this tree, or "none".
shipped_sum() {
  local sum=''
  [[ -f $HERE/SHA256SUMS ]] && sum=$(awk -v p="$1" '$2 == p { print $1 }' "$HERE/SHA256SUMS")
  echo "${sum:-none}"
}

# What a running container says about its session tier (from the label serve.sh sets), as one line.
tier_line() {
  local tier
  tier=$(docker inspect --format '{{index .Config.Labels "jspark3.session-tier"}}' "$1" 2>/dev/null) || tier=''
  case $tier in
    disk) echo "session tier: disk (conversation state is kept under \$DATA/sessions on each box, up to ${SERVE_SESSION_GIB} GiB per box)" ;;
    off) echo "session tier: off (no conversation state is kept on disk)" ;;
    *) echo "session tier: unknown (this container was not started by this recipe's serve.sh)" ;;
  esac
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# What this recipe tree is: the sha256 of its SHA256SUMS, plus the git commit when it is a git checkout (with
# "-modified" when tracked files differ from it). Recorded on each container as a label.
recipe_identity() {
  local id commit
  id=sums:$([[ -f $HERE/SHA256SUMS ]] && sha256_of "$HERE/SHA256SUMS" || echo none)
  if commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null); then
    [[ -z $(git -C "$HERE" status --porcelain --untracked-files=no 2>/dev/null) ]] || commit=$commit-modified
    id="$id,commit:$commit"
  fi
  echo "$id"
}

# DIR.verified records the digest of the list DIR was last checked against; true if that is LIST's digest now.
verified() { [[ -f $1.verified && $(cat "$1.verified") == "$(sha256_of "$2")" ]]; }

# A container's state and whether this recipe started it, as "<state> <release label>" ("absent" when there is none).
# The scripts only ever stop or remove containers that carry the recipe's label.
container_state() {
  local line
  line=$(docker inspect --format '{{.State.Status}} {{index .Config.Labels "jspark3.release"}}' "$1" 2>/dev/null) || line=absent
  echo "$line"
}

# Building or splitting fragments memory; on a box that is serving, background compaction can then stall the server.
warn_if_serving() {
  [[ $DRY_RUN == 1 ]] && return 0
  if [[ -n $(docker ps -q --filter label=jspark3.release 2>/dev/null) ]]; then
    warn "a server from this recipe is running on this box: building or splitting now can stall it. Prefer to run this before serve.sh (INSTALL.md), with vm.compaction_proactiveness=0"
  fi
}

# The local docker image for $IMAGE, checked by ID.
image_ok() {
  local id
  id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || { warn "the pinned image is not pulled (scripts/pull-image.sh)"; return 1; }
  [[ $id == "$IMAGE_ID" ]] || { warn "the local image ID is not the pinned IMAGE_ID (scripts/pull-image.sh)"; return 1; }
}

# The engine wheel in wheels/, checked against the pinned content digest.
wheel_ok() {
  local wheel=$HERE/wheels/$ENGINE_WHEEL
  [[ -f $wheel ]] || { warn "no engine wheel in wheels/ (scripts/build-wheel.sh)"; return 1; }
  [[ $WHEEL_CONTENT_SHA256 != PENDING ]] || { warn "pins.env WHEEL_CONTENT_SHA256 is PENDING: this tree has no pinned engine wheel yet"; return 1; }
  python3 "$HERE/scripts/wheel-content.py" --expect "$WHEEL_CONTENT_SHA256" "$wheel" >/dev/null 2>&1 \
    || { warn "wheels/$ENGINE_WHEEL is not the pinned engine build (scripts/build-wheel.sh)"; return 1; }
}
