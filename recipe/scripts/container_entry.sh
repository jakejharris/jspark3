#!/usr/bin/env bash
set -euo pipefail

if [[ $(ulimit -Sn) != 65536 || $(ulimit -Hn) != 65536 ]]; then
  echo "REFUSE: InstantTensor distributed loader requires nofile=65536:65536" >&2
  exit 9
fi

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
if [[ ${B45_COMBINED:-} != 1 || ${B5_PREFIX_VERIFY:-} != 1 || ${B5_CALIB_REPLAYS:-} != 20 ||
      ${B5_T4_SEED_MS:-} != 74.30 || ${B5_T7_SEED_MS:-} != 92.53 || ${B5_OUT:-} != /tmp/b45 ||
      ${JSPARK3_KDA_QKV_SHADOW:-} != 1 || ${B4_CAPTURE_ORDER:-} != '["bf16_0","int8_0"]' ]]; then
  echo "REFUSE: Cadence B4+B5 environment drift" >&2
  exit 9
fi

if [[ ${ABLIT:-} != 0 && ${ABLIT:-} != 1 ]]; then
  echo "REFUSE: this derivative requires explicit ABLIT=0 or ABLIT=1" >&2
  exit 9
fi
for forbidden in GLM53_EXL3_MOE_FAST GLM53_COOP_GEOMETRY HAREM_KDA_FLASHKDA \
  GLM53_KDA_BF16_LARGE_M GLM53_COOPERATIVE_MOE GLM53_COOP_QUALIFICATION \
  GLM53_COOP_MAINTENANCE_TEST JSPARK3_V16_COOP_MAINTENANCE \
  GLM53_COOP_SANITIZER GLM53_COOP_EP_RANK GLM53_COOP_TEST_HELPERS GLM53_COOP_BUNDLE; do
  if [[ -v $forbidden ]]; then
    echo "REFUSE: unsupported TP3 override $forbidden" >&2
    exit 9
  fi
done
if [[ ${GLM53_DENSE_FP8:-} != off || ${EXL3_FAT_GROUPED:-} != 0 ||
      ${GLM53_INDEXER_WORKSPACE:-} != stock ||
      ${VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA:-} != 0 ]]; then
  echo "REFUSE: legacy option drift (GLM53_DENSE_FP8 must stay off)" >&2
  exit 9
fi
# v1.6: independent, fail-closed decode options.  negative-coarse is a
# QA-only sensitivity control and can never enter a production profile.
case ${JSPARK3_V16_PROFILE:-} in
  production|qa) ;;
  production-stock)
    if [[ ${ABLIT:-} != 0 || ${JSPARK3_ABLIT_SWAP:-0} != 0 ]]; then
      echo "REFUSE: production-stock requires ABLIT=0 and swap disabled" >&2; exit 9
    fi
    for donor in ABLIT_METHOD ABLIT_LAYERS ABLIT_INCLUDE_MTP JSPARK_ABLIT_ROOT JSPARK_ABLIT_MANIFEST_SHA256; do
      if [[ -v $donor ]]; then
        echo "REFUSE: production-stock donor key present: $donor" >&2; exit 9
      fi
    done
    ;;
  *) echo "REFUSE: JSPARK3_V16_PROFILE must be production, production-stock or qa" >&2; exit 9 ;;
esac
case ${JSPARK3_V16_COOP:-} in
  0|1) ;;
  *) echo "REFUSE: JSPARK3_V16_COOP must be 0 or 1" >&2; exit 9 ;;
esac
case ${GLM53_ADAPTIVE_K:-} in
  off|ema) ;;
  *) echo "REFUSE: GLM53_ADAPTIVE_K must be off or ema" >&2; exit 9 ;;
esac
if [[ ${GLM53_ADAPTIVE_K_EPOCH_FILE:-} != /evidence/adaptive-k-epoch.json ||
      ${JSPARK3_V16_ADAPTIVE_K_EXPECT:-} != "${GLM53_ADAPTIVE_K}" ]]; then
  echo "REFUSE: adaptive-k epoch/expectation drift" >&2
  exit 9
fi
case ${JSPARK3_V16_DENSE_FP8:-} in
  off|trunk)
    if [[ -v JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL ]]; then
      echo "REFUSE: dense-FP8 test acknowledgement is valid only for negative-coarse" >&2
      exit 9
    fi
    ;;
  negative-coarse)
    if [[ ${JSPARK3_V16_PROFILE} != qa ||
          ${JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL:-} != I_UNDERSTAND_TEST_ONLY ]]; then
      echo "REFUSE: negative-coarse requires the qa profile and explicit test-only acknowledgement" >&2
      exit 9
    fi
    ;;
  *) echo "REFUSE: JSPARK3_V16_DENSE_FP8 must be off, trunk or negative-coarse" >&2; exit 9 ;;
esac
case ${GLM53_APC_DRAFT_LRU:-} in
  0|1) ;;
  *) echo "REFUSE: GLM53_APC_DRAFT_LRU must be 0 or 1" >&2; exit 9 ;;
esac
# v1.5: EXL3 fat-expert path (base stage 8) under an explicit mode, an explicit
# self-check count (0 at boot; armed later through the epoch file), and the
# sealed epoch file.
case ${JSPARK3_EXL3_FATPATH:-} in
  legacy|syncfree|fused) ;;
  *) echo "REFUSE: JSPARK3_EXL3_FATPATH must be legacy, syncfree or fused" >&2; exit 9 ;;
esac
if [[ ! ${JSPARK3_EXL3_FATPATH_SELFCHECK:-} =~ ^(0|[1-9][0-9]*)$ ||
      ${JSPARK3_EXL3_FATPATH_EPOCH_FILE:-} != /evidence/fatpath-epoch.json ]]; then
  echo "REFUSE: v1.5 EXL3 fat-path environment drift" >&2
  exit 9
fi
# v1.4: compact drafter KV, cadence-sched v6 cap mode, capacity log, and the
# display-reserve KV pool (#234) under an explicit, sealed profile.
case ${JSPARK3_V14_PROFILE:-} in
  full) want_display=1 ;;
  display0) want_display=0 ;;
  *) echo "REFUSE: JSPARK3_V14_PROFILE must be full or display0" >&2; exit 9 ;;
esac
if [[ ${GLM53_DRAFT_KV_COMPACT:-} != 1 || ${GLM53_MIXED_PREFILL_CHUNK:-} != cap ||
      ${GLM53_MIXED_PREFILL_CAP:-} != 2560 || ${GLM53_KV_CAPACITY_LOG:-} != 1 ||
      ${GLM53_SCHED_EPOCH_FILE:-} != /evidence/sched-epoch.json ||
      ${GLM53_DISPLAY_KV:-} != "$want_display" ]]; then
  echo "REFUSE: v1.4 KV/scheduler environment drift" >&2
  exit 9
fi
display_lib=/recipe/overlays/v14/display_kv/libglm53_display_kv.so
if [[ $want_display == 1 ]]; then
  if [[ ${GLM53_DISPLAY_KV_MIB:-} != 1792 || ${GLM53_DISPLAY_KV_CREDIT_MIB:-} != 960 ||
        ${GLM53_DRM_CARD:-} != /dev/dri/card0 || ! -c /dev/dri/card0 ||
        ${GLM53_DISPLAY_KV_LIB:-} != "$display_lib" || ! -f $display_lib || -L $display_lib ]] ||
     ! grep -qxE '[0-9a-f]{64}  overlays/v14/display_kv/libglm53_display_kv\.so' /recipe/SHA256SUMS; then
    echo "REFUSE: display-reserve KV needs MiB=1792, credit 960 MiB, /dev/dri/card0 and the sealed library" >&2
    exit 9
  fi
elif [[ -v GLM53_DISPLAY_KV_LIB || -v GLM53_DISPLAY_KV_CREDIT_MIB ]]; then
  # The NVIDIA runtime exposes /dev/dri/card0 to every GPU container, so only
  # the library knob distinguishes the profiles here; GLM53_DISPLAY_KV=0
  # never opens it.
  echo "REFUSE: display0 profile must not carry the display library" >&2
  exit 9
fi
expect_file=/recipe/config/v14-expect-${JSPARK3_V14_PROFILE}.json
if [[ ! -f $expect_file || -L $expect_file ]] || ! python3 -S - "$expect_file" <<'PY'
import json
import os
import sys

base = json.loads(open(sys.argv[1], encoding="utf-8").read())
base["coop"] = "on" if os.environ["JSPARK3_V16_COOP"] == "1" else "off"
base["adaptive-k"] = os.environ["GLM53_ADAPTIVE_K"]
base["dense-fp8"] = os.environ["JSPARK3_V16_DENSE_FP8"]
expected = json.dumps(base, sort_keys=True, separators=(",", ":"))
raise SystemExit(0 if os.environ.get("JSPARK3_V14_EXPECT") == expected else 9)
PY
then
  echo "REFUSE: identity expectation drift" >&2
  exit 9
fi
printf 'JSPARK3_V14_PROFILE rank=%s profile=%s compact=1 sched=cap/2560 display=%s/1792 ABLIT=%s trunk=W8A16 controller=B5 coop=%s adaptive-k=%s dense-fp8=%s loader=instanttensor swa_retention=0 cross_nic=%s\n' \
  "$NODE_RANK" "$JSPARK3_V14_PROFILE" "$want_display" "$ABLIT" \
  "$JSPARK3_V16_COOP" "$GLM53_ADAPTIVE_K" "$JSPARK3_V16_DENSE_FP8" "${NCCL_CROSS_NIC:-}"

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
python3 -S "$recipe/scripts/apply_grammar_fsm.py" --vllm-root "$vllm" \
  --contract "$recipe/config/v16-grammar-fsm-contract.json" \
  --image-receipt "$receipt" --apply
python3 -S "$recipe/scripts/apply_warmjit.py" --vllm-root "$vllm" \
  --contract "$recipe/config/v16-warmjit-contract.json" \
  --image-receipt "$receipt" --apply
python3 -S "$recipe/scripts/apply_coop_moe.py" --vllm-root "$vllm" \
  --contract "$recipe/overlays/v16/coop/INSTALL_CONTRACT.json" \
  --image-receipt "$receipt" --apply
python3 -S "$recipe/scripts/apply_adaptive_k.py" --vllm-root "$vllm" \
  --contract "$recipe/config/v16-adaptive-k-contract.json" \
  --image-receipt "$receipt" --apply

overlay_source=$recipe/overlays/trunk_w8a16.py
overlay_target=$vllm/model_executor/layers/quantization/trunk_w8a16.py
patcher=$recipe/overlays/patch_base_loader_hook.py
base_loader=$vllm/model_executor/model_loader/base_loader.py
expected_overlay=5aeff0cf92e715094d737faded2bf35000f7ce586213c495431b5a4805f7307d
expected_patcher=8a2f74597dc9252a35e7050bbe15c7ea4864c9bfff27ccf50135b5b28048bd9c
base_loader_before=a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08
base_loader_after=8969780b8fa0c7a866bfa74fe16f3fd1133de9c3fb8192f27aa048cf70904802

[[ $(sha256sum "$overlay_source" | awk '{print $1}') == "$expected_overlay" ]] || {
  echo "REFUSE: JSpark3 overlay hash drift" >&2
  exit 9
}
[[ $(sha256sum "$patcher" | awk '{print $1}') == "$expected_patcher" ]] || {
  echo "REFUSE: JSpark3 loader patcher hash drift" >&2
  exit 9
}
if [[ $ABLIT == 1 ]]; then
  [[ ${ABLIT_METHOD:-} == transplant && ${ABLIT_LAYERS:-} == 15-45 && ${ABLIT_INCLUDE_MTP:-} == 1 ]] || {
    echo "REFUSE: ablation method/range drift" >&2; exit 9;
  }
  python3 -S "$recipe/scripts/validate_ablit_artifacts.py" --root /ablit --manifest-sha256 "$JSPARK_ABLIT_MANIFEST_SHA256"
fi
install -m 0444 "$recipe/overlays/ablit_transplant.py" "$vllm/model_executor/layers/quantization/ablit_transplant.py"
install -m 0444 "$recipe/scripts/validate_ablit_artifacts.py" "$vllm/model_executor/layers/quantization/ablit_artifacts.py"
install -m 0444 "$recipe/overlays/instanttensor_audit.py" "$vllm/model_executor/layers/quantization/instanttensor_audit.py"
python3 -S "$recipe/scripts/patch_loader_audit.py" "$recipe" "$vllm/model_executor/model_loader/default_loader.py"
install -m 0444 "$overlay_source" "$overlay_target"
python3 "$patcher" \
  --target "$base_loader" \
  --expected-before-sha256 "$base_loader_before" \
  --expected-after-sha256 "$base_loader_after"
[[ $(sha256sum "$overlay_target" | awk '{print $1}') == "$expected_overlay" ]] || {
  echo "REFUSE: installed JSpark3 overlay hash drift" >&2
  exit 9
}
python3 -S "$recipe/scripts/apply_swa.py" --package-root "${vllm%/vllm}" --recipe-root "$recipe" --receipt /evidence/swa-receipt.json
python3 "$recipe/scripts/install_b45_modules.py" --recipe-root "$recipe"
if [[ $JSPARK3_V16_DENSE_FP8 == off ]]; then
  printf '[jspark3-v16:dense-fp8] rank=%s state=off\n' "$NODE_RANK"
else
  python3 -S "$recipe/scripts/apply_dense_fp8.py" --vllm-root "$vllm" \
    --contract "$recipe/config/v16-dense-fp8-contract.json" \
    --image-receipt "$receipt" --apply
fi
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
