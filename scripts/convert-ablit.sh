#!/usr/bin/env bash
# Build the ablit weights and all three thirds from the refusal-removed source and the base snapshot, inside the pinned
# image with no network, and check every output file before anything is marked verified.
#
#   scripts/convert-ablit.sh [--verify-only] [--dry-run]
#
# Inputs (scripts/fetch-weights.sh --weights ablit): $DATA/base/weights and $DATA/ablit/source, both verified.
# Runs scripts/ablit/reproduce-ablit.py (converter scripts/ablit/convert-weights.py, pinned by ABLIT_CONVERTER_SHA256)
# in $DATA/ablit/work. Outputs: $DATA/ablit/weights (manifests/inputs/ablit-weights.sha256) and $DATA/ablit/rank0-2
# (manifests/ablit/rank<R>.sha256), each marked verified only when every file matches; split.sh then has nothing to do.
# To save time, convert on one box and copy each follower's third to it (INSTALL.md).
# --verify-only  re-check $DATA/ablit/weights without converting
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }
VERIFY_ONLY=0
parse_common --weights ablit "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
load_cluster

converter=$HERE/scripts/ablit/convert-weights.py
out=$(snapshot_dir ablit)
work=$(variant_dir ablit)/work
manifest=$HERE/manifests/inputs/ablit-weights.sha256

check() {  # dir manifest: compare, name the files that differ
  local dir=$1 list=$2 res rc=0
  rm -f "$dir.verified" || return 1
  res=$(python3 "$HERE/scripts/verify-split.py" check "$dir" "$list" 2>&1) || rc=$?
  log_open
  printf '%s\n' "$res" >>"$LOG"
  [[ $rc == 0 ]] || { printf '%s\n' "$res" | head -60 >&2; return 1; }
}
mark() { rm -f "$1.verified"; sha256_of "$2" >"$1.verified"; }  # dir manifest

if [[ $VERIFY_ONLY == 1 ]]; then
  [[ $DRY_RUN == 1 ]] && { run python3 scripts/verify-split.py check "$out" manifests/inputs/ablit-weights.sha256; exit 0; }
  check "$out" "$manifest" || die "the ablit weights do not match their manifest (files listed above); run scripts/convert-ablit.sh again"
  mark "$out" "$manifest"
  info "ablit weights verified"
  exit 0
fi

ablit_ready || die "$ABLIT_NOT_READY"
if [[ $DRY_RUN == 0 ]]; then
  [[ -f $converter && $(sha256_of "$converter") == "$ABLIT_CONVERTER_SHA256" ]] \
    || die "scripts/ablit/convert-weights.py does not match ABLIT_CONVERTER_SHA256 in pins.env; restore it: git checkout scripts/ablit/"
  verified "$(snapshot_dir base)" "$HERE/manifests/inputs/base-weights.sha256" \
    || die "the base snapshot is not downloaded and verified: scripts/fetch-weights.sh --weights ablit"
  verified "$(variant_dir ablit)/source" "$HERE/manifests/inputs/ablit-source.sha256" \
    || die "the refusal-removed source is not downloaded and verified: scripts/fetch-weights.sh --weights ablit"
  image_ok || die "pull the pinned image first: scripts/pull-image.sh"
  wheel_ok || die "build the engine wheel first: scripts/build-wheel.sh"
  done_all=1
  for rank in 0 1 2; do verified "$(rank_dir "$rank")" "$(rank_manifest "$rank")" || done_all=0; done
  if [[ $done_all == 1 ]] && verified "$out" "$manifest"; then
    info "ablit weights and all three thirds already built and verified"
    exit 0
  fi
  # The converted weights plus three thirds: a little under twice the source (-L: the source may be a link).
  need=$(du -sbL "$(variant_dir ablit)/source" | python3 -c 'import sys; print(int(sys.stdin.read().split()[0]) * 2)')
  free=$(df --output=avail -B1 "$DATA" | tail -1)
  ((need <= free)) || die "the conversion needs about $(bc_div "$need") GB but the DATA disk has $(bc_div "$free") GB free"
  rm -rf "$work" "$out" "$out.verified"
  for rank in 0 1 2; do rm -rf "$(rank_dir "$rank")" "$(rank_dir "$rank").partial" "$(rank_dir "$rank").verified"; done
fi
warn_if_serving
info "converting (CPU and disk only; progress every 30 s)"
# The same steps, environment and offline inventory the conversion was validated with; no Hugging Face token is passed.
run_watch "ablit conversion" "$work" 0 docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e PIP_NO_CACHE_DIR=1 -e XDG_CACHE_HOME=/tmp/cache -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -v "$DATA":/data -v "$HERE/scripts/ablit":/convert:ro -v "$HERE/wheels":/wheels:ro \
  "$IMAGE" bash -c \
  "python -m pip install -q --no-deps --no-index --target /tmp/engine /wheels/$ENGINE_WHEEL && PYTHONPATH=/tmp/engine python3 -u /convert/reproduce-ablit.py /data/ablit/work --source /data/ablit/source --vontra /data/base/weights --source-api /convert/ablit-source-api.json" \
  || die "the conversion failed (logs/convert-ablit.log); nothing was marked verified"
[[ $DRY_RUN == 1 ]] && exit 0

converted=$work/converted
info "checking every converted file against manifests/inputs/ablit-weights.sha256"
check "$converted/weights" "$manifest" \
  || die "the converted weights do not match manifests/inputs/ablit-weights.sha256 (files listed above); kept in \$DATA/ablit/work, nothing marked verified"
id=$(cat "$converted/CONVERSION-ID" 2>/dev/null || true)
[[ $id == "$ABLIT_CONVERSION_ID" ]] \
  || die "the conversion id is ${id:-missing}, not ABLIT_CONVERSION_ID in pins.env; kept in \$DATA/ablit/work, nothing marked verified"
for rank in 0 1 2; do
  info "checking the rank $rank third against manifests/ablit/rank$rank.sha256"
  check "$converted/rank$rank" "$(rank_manifest "$rank")" \
    || die "the rank $rank third does not match manifests/ablit/rank$rank.sha256 (files listed above); kept in \$DATA/ablit/work, nothing marked verified"
done
mv "$converted/weights" "$out"
mark "$out" "$manifest"
for rank in 0 1 2; do
  mv "$converted/rank$rank" "$(rank_dir "$rank")"
  mark "$(rank_dir "$rank")" "$(rank_manifest "$rank")"
done
cp "$HERE/scripts/ablit/WEIGHTS-LICENSE.txt" "$HERE/scripts/ablit/TEMPLATE-ADDITIONS-LICENSE.txt" "$(variant_dir ablit)/"
info "ablit weights and all three thirds built and verified ($(du -sh "$(variant_dir ablit)" | cut -f1) in \$DATA/ablit)"
info "next: copy rank 1 and rank 2 to their boxes (INSTALL.md), or run scripts/split.sh --weights ablit --verify-only on each"
