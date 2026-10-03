#!/usr/bin/env bash
# Show this box's rank: whether its container runs, what it serves and its session tier. On rank 0, also whether the
# API answers. Read-only.
#
#   scripts/status.sh [--identity]
#
# --identity also reads, from inside the running container, what it actually runs, and compares each with this
# tree's pins: the image ID, the engine wheel's sha256 and content digest, the installed engine version, the chat
# template, the fabric launcher, the per-rank data manifest, the weights profile (config/profiles/<weights>.env, against
# SHA256SUMS) and the recipe tree (SHA256SUMS digest and git commit) recorded when it started. Exit 1 if anything
# differs.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }
IDENTITY=0
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --identity) IDENTITY=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
load_cluster
[[ $DRY_RUN == 1 ]] && { info "dry run: would show $(container_name "$RANK")"; exit 0; }

name=$(container_name "$RANK")
label() { docker inspect --format "{{index .Config.Labels \"jspark3.$1\"}}" "$name"; }
read -r state started <<<"$(docker inspect --format '{{.State.Status}} {{.State.StartedAt}}' "$name" 2>/dev/null || echo absent)"
if [[ $state == absent ]]; then
  info "rank $RANK: no container named $name (start it with scripts/serve.sh)"
  exit 1
fi
[[ -n $(label release) ]] || die "$name was not started by this recipe (CONTAINER_PREFIX in cluster.env names this recipe's containers)"
info "rank $RANK: $name $state since ${started%%.*}Z; release $(label release), $(label weights) weights, drafter $(label drafter)"
info "$(tier_line "$name")"
[[ $state == running ]] || exit 1

bad=0
if [[ $IDENTITY == 1 ]]; then
  compare() {  # what got want
    if [[ $2 == "$3" ]]; then info "identity: $1 $2 (as pinned)"; else info "identity: $1 $2 DIFFERS from $3"; bad=1; fi
  }
  model=$SERVE_MODEL_DIR.rank${RANK}of3
  compare "image ID" "$(docker inspect --format '{{.Image}}' "$name")" "$IMAGE_ID"
  wheel_file=$(docker exec "$name" sha256sum "/bundle/wheels/$ENGINE_WHEEL" 2>/dev/null | cut -d' ' -f1) || wheel_file=unreadable
  info "identity: engine wheel file sha256 $wheel_file"
  content=$(docker exec -i "$name" python3 - "/bundle/wheels/$ENGINE_WHEEL" <"$HERE/scripts/wheel-content.py" 2>/dev/null | cut -d' ' -f1) \
    || content=unreadable
  compare "engine wheel content" "$content" "$WHEEL_CONTENT_SHA256"
  installed=$(docker exec "$name" python3 -c 'import importlib.metadata as m; print(m.version("tensorfold"))' 2>/dev/null) \
    || installed=unreadable
  version=${ENGINE_WHEEL#tensorfold-}
  compare "installed engine version" "$installed" "${version%%-*}"
  compare "chat template" "$(docker exec "$name" sha256sum "$model/chat_template.jinja" 2>/dev/null | cut -d' ' -f1)" "$TEMPLATE_SHA256"
  compare "fabric launcher" "$(docker exec "$name" sha256sum /bundle/fabric/launch.py 2>/dev/null | cut -d' ' -f1)" \
    "$(sha256_of "$HERE/scripts/fabric/launch.py")"
  manifest=$(rank_manifest "$RANK" "$(label weights)")
  compare "rank data manifest" "$(label rank-manifest)" "$([[ -f $manifest ]] && sha256_of "$manifest" || echo none)"
  profile=$(profile_rel "$(label weights)")
  compare "settings profile $profile" "$(label profile)" "$(shipped_sum "$profile")"
  compare "recipe tree" "$(label recipe)" "$(recipe_identity)"
fi

if [[ $RANK == 0 ]]; then
  host=$API_HOST
  [[ $host == 0.0.0.0 || $host == :: ]] && host=127.0.0.1
  if python3 - "http://$host:$API_PORT" "$SERVE_NAME" <<'PY' >/dev/null 2>&1; then
import json, sys, urllib.request
url, name = sys.argv[1:]
models = json.load(urllib.request.urlopen(url + "/v1/models", timeout=5))
assert any(m.get("id") == name for m in models.get("data", []))
PY
    info "API on port $API_PORT lists the model (scripts/smoke.sh runs real requests)"
  else
    info "API on port $API_PORT does not answer yet (still loading, or see scripts/wait-ready.sh)"
    exit 1
  fi
fi
exit $bad
