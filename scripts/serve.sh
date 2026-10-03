#!/usr/bin/env bash
# Start this box's rank of the measured configuration. Start ranks 2 and 1 first, then 0. On rank 0 this waits until
# the model answers a request through all three ranks (scripts/wait-ready.sh).
#
#   scripts/serve.sh [--weights base|ablit] [--drafter dflash2|none] [--session-tier disk|off] [--no-wait]
#                    [--timeout SECONDS] [--container-prefix NAME] [--dry-run] [-- extra engine flags]
#
# The API listens on API_HOST:API_PORT on rank 0 (default 127.0.0.1:8002). It has no authentication and no CORS
# policy: reach it through an ssh tunnel or an authenticating proxy, never by exposing the port (INSTALL.md).
# Every box must use the same --weights, --drafter and --session-tier. --drafter none runs without the DFlash2
# drafter, drafting from the model's own prediction head instead (up to 7 tokens at confidence 0.3).
# Engine settings come from config/serve.env and config/serve.conf (shared) and the draft policies from
# config/profiles/<weights>.env (tuned per weights); extra engine flags after --
# change the configuration from the one measured.
# The session tier keeps conversation state, including the drafter's cache state, on this box's disk under
# $DATA/sessions (INSTALL.md, "Session tier"). --session-tier off (or SESSION_TIER=off) keeps none on disk.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }
WAIT=1
TIMEOUT=()
EXTRA=()
parse_common "$@"
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --no-wait) WAIT=0; shift ;;
    --timeout) [[ ${2:-} =~ ^[0-9]+$ ]] || die "--timeout needs a number of seconds"; TIMEOUT=(--timeout "$2"); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown option $1 (--help; engine flags go after --)" ;;
  esac
done
load_cluster
load_profile
profile=$(profile_rel)
if [[ $(sha256_of "$HERE/$profile") != "$(shipped_sum "$profile")" ]]; then
  warn "$profile differs from the file this release shipped (SHA256SUMS): this is not a measured configuration"
fi

name=$(container_name "$RANK")
model=$SERVE_MODEL_DIR.rank${RANK}of3
third=$(rank_dir "$RANK")
manifest=$(rank_manifest "$RANK")
template=$HERE/template/chat-template.jinja
sessions=$(session_dir)

if [[ $DRY_RUN == 0 ]]; then
  command -v docker >/dev/null || die "docker is not installed (INSTALL.md, 'What you need')"
  # Never reuse or remove an existing container: a name clash is refused with what to do.
  read -r state label <<<"$(container_state "$name")"
  if [[ $state != absent ]]; then
    [[ -n ${label:-} ]] || die "a container named $name exists on this box and was not started by this recipe; it is left alone. Set CONTAINER_PREFIX in cluster.env to another name, the same on all three boxes"
    case $state in
      running|restarting|paused) die "$name is already running on this box; stop it first: scripts/stop.sh" ;;
      *) die "$name from an earlier start still exists ($state); remove it with scripts/stop.sh, then start again" ;;
    esac
  fi
  image_ok || die "pull the pinned image first: scripts/pull-image.sh"
  wheel_ok || die "build the engine wheel first: scripts/build-wheel.sh (and scripts/fetch-wheels.sh)"
  { [[ -d $third ]] && verified "$third" "$manifest"; } \
    || die "the $WEIGHTS rank $RANK third is not split and verified on this box: scripts/split.sh${OPT_WEIGHTS:+ --weights $WEIGHTS}"
  if [[ $DRAFTER == dflash2 ]]; then
    { [[ -d $(drafter_dir) ]] && verified "$(drafter_dir)" "$HERE/manifests/inputs/drafter.sha256"; } \
      || die "the drafter is not downloaded and verified on this box: scripts/fetch-weights.sh (or serve with --drafter none on every box)"
  fi
  [[ -f $template && $(sha256_of "$template") == "$TEMPLATE_SHA256" ]] \
    || die "template/chat-template.jinja does not match TEMPLATE_SHA256 in pins.env; restore it: git checkout template/"
  mkdir -p "$DATA/kernel-cache/tensorfold"
  if [[ $SESSION_TIER == disk ]]; then
    mkdir -p "$sessions"
    # The session tier writes only while it leaves 150 GiB free on its disk (an engine rule); say so before it matters.
    free=$(df --output=avail -B1 "$sessions" | tail -1)
    if ((free < (150 + SERVE_SESSION_GIB) * 1073741824)); then
      warn "the DATA disk has $(bc_div "$free") GB free; the session tier stops writing below 150 GiB free and uses up to ${SERVE_SESSION_GIB} GiB, so conversations may not be kept on disk"
    fi
  fi
  # Every start is cold: no prompt snapshots from an earlier boot.
  rm -rf "$DATA/kernel-cache/tensorfold/prefix-snapshots"
fi

# The serve command, in the measured order. Only rank 0 listens for requests.
if [[ $DRAFTER == dflash2 ]]; then
  draft=(--drafter /drafter --draft-policy "$SERVE_DRAFT_POLICY")
  drafter_mount=(-v "$(drafter_dir)":/drafter:ro)
else
  draft=(--drafter none --draft-policy "$NO_DRAFTER_POLICY")
  drafter_mount=()
fi
cmd=(python3 /bundle/fabric/launch.py tensorfold serve "$model" --name "$SERVE_NAME" --no-update-check --tp 3
     --rank "$RANK" --master "$MASTER_ADDR" --master-port "$MASTER_PORT" --context "$SERVE_CONTEXT"
     --max-tokens "$SERVE_MAX_TOKENS" "${draft[@]}")
[[ $RANK == 0 ]] && cmd+=(--host "$API_HOST" --port "$API_PORT")
cmd+=(--parallel "$SERVE_PARALLEL" "${EXTRA[@]}")
for word in "${cmd[@]}"; do
  [[ $word =~ ^[A-Za-z0-9_./:=,+-]+$ ]] || die "an engine flag has characters serve.sh doesn't pass through: use letters, digits and _./:=,+-"
done

# Container: GPU, host IPC and network, the RDMA devices and locked memory for NCCL over the cables.
container=(--name "$name" --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864
           --device /dev/infiniband --cap-add IPC_LOCK
           --label "jspark3.release=$RELEASE" --label "jspark3.weights=$WEIGHTS" --label "jspark3.drafter=$DRAFTER"
           --label "jspark3.session-tier=$SESSION_TIER" --label "jspark3.recipe=$(recipe_identity)"
           --label "jspark3.profile=$(sha256_of "$HERE/$profile")"
           --label "jspark3.rank-manifest=$( [[ -f $manifest ]] && sha256_of "$manifest" || echo none)")
# Measured engine environment, plus this box's interfaces: the LAN for bootstrap, the two cable ports for the ring.
environment=(--env-file "$HERE/config/serve.env"
             -e RING_PREV_IFACE="$PREV_IFACE" -e RING_NEXT_IFACE="$NEXT_IFACE"
             -e NCCL_SOCKET_IFNAME="$LAN_IFACE" -e GLOO_SOCKET_IFNAME="$LAN_IFACE")
# This rank's third and the chat template (read-only), the drafter, wheels and launcher, kernel caches, sessions.
# (The measured boot also mounted an idle /capture folder; nothing in this configuration writes there, so it is left out.)
mounts=(-v "$third":"$model":ro -v "$template":"$model"/chat_template.jinja:ro "${drafter_mount[@]}"
        -v "$HERE/wheels":/bundle/wheels:ro -v "$HERE/scripts/fabric":/bundle/fabric:ro
        -v "$DATA/kernel-cache":/kernel-cache -v "$DATA/kernel-cache/tensorfold":/root/.cache/tensorfold)
if [[ $SESSION_TIER == disk ]]; then
  mounts+=(-v "$sessions":"$SERVE_SESSION_DIR")
else
  # Off: no session persistence, and an empty disk folder so the engine's older prompt store stays off as well.
  environment+=(-e TF_GLM_SESSION_DISK=0 -e TF_GLM_DISK_DIR=)
fi

info "starting rank $RANK ($WEIGHTS weights with $profile, drafter $DRAFTER, session tier $SESSION_TIER) as container $name"
run docker run -d "${container[@]}" "${environment[@]}" "${mounts[@]}" "$IMAGE" \
  bash -c "pip install -q --no-index --find-links /bundle/wheels tensorfold pytest && exec ${cmd[*]}" \
  || die "docker could not start $name"

if [[ $RANK != 0 ]]; then
  info "rank $RANK started. Start the other follower, then rank 0, which waits until all three answer"
  exit 0
fi
if [[ $WAIT == 0 ]]; then
  info "rank 0 started; check readiness with scripts/wait-ready.sh"
  exit 0
fi
dry=()
[[ $DRY_RUN == 1 ]] && dry=(--dry-run)
exec "$HERE/scripts/wait-ready.sh" --container-prefix "$CONTAINER_PREFIX" "${TIMEOUT[@]}" "${dry[@]}"
