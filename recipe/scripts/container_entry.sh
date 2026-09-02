#!/usr/bin/env bash
set -euo pipefail

# These reads use shell builtins and happen before Python, CUDA, or model allocation.
IFS= read -r memory_max </sys/fs/cgroup/memory.max
IFS= read -r swap_max </sys/fs/cgroup/memory.swap.max
if [[ $memory_max != 68719476736 || $swap_max != 0 ]]; then
  echo "REFUSE: cgroup requires memory.max=68719476736 and memory.swap.max=0" >&2
  exit 9
fi
IFS= read -r swap_current </sys/fs/cgroup/memory.swap.current
oom=missing
oom_kill=missing
oom_group_kill=missing
while read -r event value; do
  case $event in
    oom) oom=$value ;;
    oom_kill) oom_kill=$value ;;
    oom_group_kill) oom_group_kill=$value ;;
  esac
done </sys/fs/cgroup/memory.events
if [[ $swap_current != 0 || $oom != 0 || $oom_kill != 0 || $oom_group_kill != 0 ]]; then
  echo "REFUSE: entry cgroup already has swap or OOM events" >&2
  exit 9
fi
for forbidden in NCCL_PROTO NCCL_ALGO NCCL_IB_ADDR_RANGE; do
  if [[ -v $forbidden ]]; then
    echo "REFUSE: forbidden fabric override $forbidden" >&2
    exit 9
  fi
done
if [[ ${JSPARK3_KDA_MIXED_OUTPUT_BLOCKS:-} != 8 || ${JSPARK3_KDA_FG_BATCHED:-} != 1 ]]; then
  echo "REFUSE: combined KDA environment drift" >&2
  exit 9
fi
if [[ ${JSPARK3_TRUNK_W8A16:-} != 1 || ${JSPARK3_TRUNK_W8A16_K704_GROUP:-} != 64 ]]; then
  echo "REFUSE: JSpark3 W8A16 environment drift" >&2
  exit 9
fi

recipe=/recipe
vllm=/usr/local/lib/python3.12/dist-packages/vllm
receipt=/evidence/image-receipt.json
contract=$recipe/config/patch-contract.json

if [[ ! -f $receipt || -L $receipt ]]; then
  echo "REFUSE: host-minted image receipt is missing or unsafe" >&2
  exit 9
fi
if [[ ! ${NODE_RANK:-} =~ ^[012]$ || ! ${JSPARK_PREFLIGHT_SHA256:-} =~ ^[0-9a-f]{64}$ || ! ${JSPARK_RECIPE_MANIFEST_SHA256:-} =~ ^[0-9a-f]{64}$ ]]; then
  echo "REFUSE: host receipt binding environment is incomplete" >&2
  exit 9
fi
python3 "$recipe/scripts/remote_preflight.py" --recipe-only --recipe-root "$recipe" --expected-recipe-manifest-sha256 "$JSPARK_RECIPE_MANIFEST_SHA256"
PYTHONPATH="$recipe/scripts" python3 -c 'import pathlib,sys; from _atomic import read_image_receipt; v=read_image_receipt(pathlib.Path(sys.argv[1])); ok=(v["rank"]==int(sys.argv[2]) and v["preflight_sha256"]==sys.argv[3] and v["recipe_manifest_sha256"]==sys.argv[4]); raise SystemExit(0 if ok else 9)' "$receipt" "$NODE_RANK" "$JSPARK_PREFLIGHT_SHA256" "$JSPARK_RECIPE_MANIFEST_SHA256"
python3 "$recipe/scripts/apply_base_pipeline.py" --vllm-root "$vllm" --source-root /sources/fly --asset-root /opt/glm53 --contract "$contract" --image-receipt "$receipt" --apply

overlay_source=$recipe/overlays/trunk_w8a16.py
overlay_target=$vllm/model_executor/layers/quantization/trunk_w8a16.py
patcher=$recipe/overlays/patch_base_loader_hook.py
base_loader=$vllm/model_executor/model_loader/base_loader.py
expected_overlay=5aeff0cf92e715094d737faded2bf35000f7ce586213c495431b5a4805f7307d
expected_patcher=c84bdfbf69f7b1d3841155d35f73a06a601f2bcb33ae9e1d8423178dc31139b4
base_loader_before=a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08
base_loader_after=3205bff77aac34785167f5b21306048b9dc916b2c0691bf774bb3d9202bbd8da

[[ $(sha256sum "$overlay_source" | awk '{print $1}') == "$expected_overlay" ]] || {
  echo "REFUSE: JSpark3 overlay hash drift" >&2
  exit 9
}
[[ $(sha256sum "$patcher" | awk '{print $1}') == "$expected_patcher" ]] || {
  echo "REFUSE: JSpark3 loader patcher hash drift" >&2
  exit 9
}
install -m 0444 "$overlay_source" "$overlay_target"
python3 "$patcher" \
  --target "$base_loader" \
  --expected-before-sha256 "$base_loader_before" \
  --expected-after-sha256 "$base_loader_after"
[[ $(sha256sum "$overlay_target" | awk '{print $1}') == "$expected_overlay" ]] || {
  echo "REFUSE: installed JSpark3 overlay hash drift" >&2
  exit 9
}
printf 'JSPARK3_STARTUP_PATCH_PASS rank=%s overlay_sha256=%s group704=64 runtime_modules=169 logical_tensors=225\n' \
  "$NODE_RANK" "$expected_overlay"

if [[ ${JSPARK_TARGET_RUNTIME:-} != /models/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime ]]; then
  echo "REFUSE: target runtime path drift" >&2
  exit 9
fi
if [[ ${JSPARK_DRAFT_RUNTIME:-} != /models/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime ]]; then
  echo "REFUSE: draft runtime path drift" >&2
  exit 9
fi
exec vllm serve "$JSPARK_TARGET_RUNTIME" "$@"
