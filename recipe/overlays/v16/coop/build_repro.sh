#!/usr/bin/env bash
# Build the pinned TP3 ABI2 cooperative-MoE bundle at a stable in-container
# path. tools/v16/build_coop_moe.sh runs this twice in fresh containers and
# refuses unless the final shared objects are bit-identical.
set -euo pipefail

output=${1:?usage: build_repro.sh EMPTY_OUTPUT_DIRECTORY}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$here/source
source_manifest=$here/SOURCE_MANIFEST.json

mkdir -p "$output"
output=$(cd "$output" && pwd)
if [[ -n $(find "$output" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "REFUSE: cooperative-MoE output directory is not empty" >&2
  exit 9
fi

export LC_ALL=C
export TZ=UTC
export SOURCE_DATE_EPOCH=1789774362
export PYTHONDONTWRITEBYTECODE=1

python3 - "$source_root" "$source_manifest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, manifest_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
if manifest.get("schema_version") != 1:
    raise SystemExit("REFUSE: source manifest schema drift")
expected = manifest.get("files")
observed = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*")) if path.is_file()
}
if expected != observed:
    raise SystemExit("REFUSE: pinned cooperative-MoE source drift")
PY
python3 "$source_root/manifest.py" verify-sources "$source_root"

mkdir -p "$output/headers/quant" "$output/source"
cp -a "$source_root/vendor/exllamav3_ext/." "$output/headers/"
cp "$source_root/native/cooperative_moe_kernel.cuh" "$output/headers/quant/glm53_coop_kernel.cuh"
cp "$source_root/native/exl3_moe_coop.cuh" "$output/headers/quant/"
cp -a "$source_root/native" "$output/source/"
cp "$source_root/dispatch_policy.json" "$source_root/runtime.py" "$source_root/PROVENANCE.json" \
  "$source_root/LICENSE.MIT" "$source_root/LICENSE.upstream-AGPL-3.0" "$output/"
cp "$source_root/vendor/LICENSE.exllamav3" "$output/"
# -lineinfo records each compiled file's mtime in the fatbin line table; pin
# the staged copies so the embedded timestamps do not depend on copy time.
find "$output/headers" "$output/source" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +

nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
seed=jspark3-v16-coop-357fce7
flags=(
  -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr
  --cudart=shared
  -gencode arch=compute_121a,code=sm_121a
  -Xcompiler=-fPIC
  -Xcompiler=-ffile-prefix-map=/w=.
  -Xcompiler=-fdebug-prefix-map=/w=.
  --ptxas-options=-v
  -I "$output/headers"
)
{
  "$nvcc" --version
  printf 'source_date_epoch=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'flags='; printf '%q ' "${flags[@]}"; printf '\n'
  printf 'per_unit_seed=%s-{rows32,rows64,dispatch}\n' "$seed"
  printf 'intermediates=--keep --keep-dir OUTPUT/keep/{rows32,rows64,dispatch}\n'
} > "$output/toolchain.txt"

# --keep gives host stubs stable basenames instead of PID-derived tmpxft names.
# Separate directories and seeds distinguish the two capacity translation units.
# Keep the intermediates for diagnosis; do not rely on container PID allocation
# or strip/normalize the finished binary to claim reproducibility.
for rows in 32 64; do
  mkdir -p "$output/keep/rows$rows"
  if ! "$nvcc" "${flags[@]}" --frandom-seed "$seed-rows$rows" \
    --keep --keep-dir "$output/keep/rows$rows" -v -DGLM53_ROWS_MAX="$rows" -c \
    "$output/source/native/cooperative_moe.cu" -o "$output/rows$rows.o" \
    > "$output/build$rows.log" 2>&1; then
    cat "$output/build$rows.log" >&2
    exit 9
  fi
done

mkdir -p "$output/keep/dispatch"
"$nvcc" "${flags[@]}" --frandom-seed "$seed-dispatch" \
  --keep --keep-dir "$output/keep/dispatch" -v -shared -Xlinker=--build-id=sha1 \
  "$output/source/native/dispatch.cu" "$output/rows32.o" "$output/rows64.o" \
  -o "$output/cooperative_moe.so" > "$output/link.log" 2>&1
rm "$output/rows32.o" "$output/rows64.o"
python3 "$source_root/manifest.py" create "$output"
sha256sum "$output/cooperative_moe.so" "$output/runtime.py" "$output/manifest.json"
