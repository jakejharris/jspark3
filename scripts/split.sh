#!/usr/bin/env bash
# Write this box's third of the weights and check it against the shipped per-rank manifest. CPU and disk only.
#
#   scripts/split.sh [--rank R | --all] [--weights base|ablit] [--verify-only] [--dry-run]
#
# Reads the verified snapshot in $DATA/<weights>/weights (scripts/fetch-weights.sh, or scripts/convert-ablit.sh for
# ablit) and writes $DATA/<weights>/rank<R>, using the engine's own split (TP=3: heads, KDA heads and experts that
# don't divide by three are zero-padded). The result must equal manifests/<weights>/rank<R>.sha256 file for file;
# any difference fails and names the files. Only a matching third is moved into place and marked verified.
#
# --rank R       split rank R instead of cluster.env's RANK (for example to split all thirds on one box)
# --all          split ranks 0, 1 and 2 here (then copy each follower's third to it; INSTALL.md)
# --verify-only  only check an existing third (for example one copied from another box) and mark it verified
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
VERIFY_ONLY=0
RANKS=()
parse_common "$@"
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --rank) [[ ${2:-} =~ ^[012]$ ]] || die "--rank needs 0, 1 or 2"; RANKS+=("$2"); shift 2 ;;
    --all) RANKS=(0 1 2); shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done
# A real recheck must revoke its marker before a missing ablit manifest can refuse it.
# Dry runs keep the ordinary availability check and never change markers.
if [[ $VERIFY_ONLY == 1 && $DRY_RUN == 0 ]]; then
  load_cluster --verify-third
else
  load_cluster
fi
((${#RANKS[@]})) || RANKS=("$RANK")

snapshot=$(snapshot_dir)
case $WEIGHTS in
  base) inputs=$HERE/manifests/inputs/base-weights.sha256 ;;
  ablit) inputs=$HERE/manifests/inputs/ablit-weights.sha256 ;;
esac

check_rank() {  # dir manifest rank -> 0 match, 1 mismatch (files listed)
  local dir=$1 manifest=$2 rank=$3 out rc=0
  rm -f "$dir.verified" || return 1
  out=$(python3 "$HERE/scripts/verify-split.py" check "$dir" "$manifest" 2>&1) || rc=$?
  log_open
  printf '%s\n' "$out" >>"$LOG"
  if [[ $rc != 0 ]]; then
    printf '%s\n' "$out" | head -60 >&2
    return 1
  fi
}

mark() {  # dir manifest
  rm -f "$1.verified"
  sha256_of "$2" >"$1.verified"
}

for rank in "${RANKS[@]}"; do
  out=$(rank_dir "$rank")
  manifest=$(rank_manifest "$rank")
  what="$WEIGHTS rank $rank third"
  if [[ $VERIFY_ONLY == 1 && $DRY_RUN == 0 ]]; then
    rm -f "$out.verified" "$out.partial.verified"
  fi
  if [[ ! -f $manifest ]]; then
    die "this tree has no manifest for the $what (manifests/$WEIGHTS/rank$rank.sha256), so it cannot be verified"
  fi
  if [[ $VERIFY_ONLY == 0 && $DRY_RUN == 0 && -d $out ]] && verified "$out" "$manifest"; then
    info "$what: already split and verified"
    continue
  fi

  if [[ $VERIFY_ONLY == 1 ]]; then
    [[ $DRY_RUN == 1 ]] && { run python3 scripts/verify-split.py check "$out" "${manifest#"$HERE"/}"; continue; }
    src=$out
    [[ -d $src ]] || src=$out.partial
    [[ -d $src ]] || die "$what: nothing to verify at $out"
    info "$what: checking every file against manifests/$WEIGHTS/rank$rank.sha256"
    check_rank "$src" "$manifest" "$rank" || die "$what does not match its manifest (files listed above). Split it again: scripts/split.sh --rank $rank"
    [[ $src == "$out" ]] || mv "$src" "$out"
    mark "$out" "$manifest"
    info "$what: verified"
    continue
  fi

  [[ $WEIGHTS == base ]] || ablit_ready || die "$ABLIT_NOT_READY"
  if [[ $DRY_RUN == 0 ]]; then
    { [[ -d $snapshot ]] && verified "$snapshot" "$inputs"; } \
      || die "the $WEIGHTS snapshot is not downloaded and verified yet (scripts/fetch-weights.sh${WEIGHTS/base/}${WEIGHTS/ablit/ and scripts/convert-ablit.sh})"
    image_ok || die "pull the pinned image first: scripts/pull-image.sh"
    wheel_ok || die "build the engine wheel first: scripts/build-wheel.sh"
    # One third holds about a third of the snapshot's tensors, plus padding.
    need=$(find "$snapshot" -maxdepth 1 -name '*.safetensors' -printf '%s\n' | python3 -c 'import sys; print((sum(map(int, sys.stdin)) * 11 + 29) // 30)')
    free=$(df --output=avail -B1 "$DATA" | tail -1)
    ((need <= free)) || die "the $what needs about $(bc_div "$need") GB but the DATA disk has $(bc_div "$free") GB free"
    rm -rf "$out.partial" "$out" "$out.verified" "$out.partial.verified"
  fi
  warn_if_serving
  info "$what: splitting (CPU and disk only; progress every 30 s)"
  # A base third's .complete marker is empty. An ablit third's names the conversion and is followed by the third's own
  # MANIFEST.sha256, as on the thirds the ablit numbers were measured on (manifests/README.md).
  marker="touch /data/rank$rank.partial/.complete && "
  [[ $WEIGHTS == ablit ]] && marker=''
  run_watch "$what" "$out.partial" 0 docker run --rm --network none \
    -v "$(variant_dir)":/data -v "$HERE/wheels":/wheels:ro "$IMAGE" bash -c \
    "pip install -q --no-index --find-links /wheels tensorfold && python -m tensorfold.families.glm5_next.cuda.split /data/weights --rank $rank --world 3 /data/rank$rank.partial && ${marker}chown -R $(id -u):$(id -g) /data/rank$rank.partial" \
    || die "$what: the split failed"
  if [[ $WEIGHTS == ablit ]]; then
    sealer=$HERE/scripts/verify-split.py
    [[ $DRY_RUN == 1 ]] && sealer=scripts/verify-split.py
    run python3 "$sealer" seal "$out.partial" --rank "$rank" --conversion-id "$ABLIT_CONVERSION_ID" \
      || die "$what: could not write its .complete marker and MANIFEST.sha256 (logs/split.log)"
  fi
  [[ $DRY_RUN == 1 ]] && continue
  info "$what: checking every file against manifests/$WEIGHTS/rank$rank.sha256"
  if ! check_rank "$out.partial" "$manifest" "$rank"; then
    die "$what does not match its manifest (files listed above). It is kept at $out.partial for inspection; nothing was marked verified"
  fi
  mv "$out.partial" "$out"
  mark "$out" "$manifest"
  info "$what: written and verified ($(du -sh "$out" | cut -f1))"
done
