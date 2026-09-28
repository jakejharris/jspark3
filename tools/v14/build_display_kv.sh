#!/usr/bin/env bash
# Build the v1.4 display-reserve KV library and its standalone probe inside the
# pinned serving image on one Spark, twice, and require bit-identical outputs.
#
#   tools/v14/build_display_kv.sh <ssh-host> [--write|--check]
#
# --check (default) fetches both outputs into recipe/overlays/v14/display_kv/
# only if they match BUILD.json. --write records new hashes (sources changed).
# Both outputs stay out of git (binary); recipe/SHA256SUMS lists them.
set -euo pipefail
host=${1:?ssh host}
mode=${2:---check}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dir=$here/recipe/overlays/v14/display_kv
build_json=$dir/BUILD.json
identity_args=()
if [[ -n ${JSPARK_IMAGE_RECEIPT:-} ]]; then identity_args=(--receipt "$JSPARK_IMAGE_RECEIPT"); fi
image=$(python3 -B "$here/recipe/scripts/_image_identity.py" "${identity_args[@]}")
sources=(display_kv.c probe_main.cu)
outputs=(libglm53_display_kv.so display_kv_probe)
remote=jspark3-v14-display-kv-build-$(date -u +%Y%m%dT%H%M%SZ)

ssh -o BatchMode=yes "$host" "mkdir -p ~/$remote/a ~/$remote/b"
for x in a b; do
    scp -q "${sources[@]/#/$dir/}" "$host:$remote/$x/"
done
ssh -o BatchMode=yes "$host" bash -s -- "$remote" "$image" <<'REMOTE'
set -euo pipefail
cd ~/"$1"
for x in a b; do
    docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD/$x:/w" -w /w \
        --entrypoint bash "$2" -c '
set -e
gcc -O2 -fPIC -shared -Wall -I/usr/local/cuda/include -o libglm53_display_kv.so display_kv.c \
    -L/usr/local/cuda/lib64/stubs -lcuda
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 -o display_kv_probe probe_main.cu display_kv.c -lcuda
{ gcc --version | head -1; /usr/local/cuda/bin/nvcc --version | tail -1; } > toolchain.txt'
done
for f in libglm53_display_kv.so display_kv_probe; do
    cmp -s "a/$f" "b/$f" || { echo "REFUSE: $f is not reproducible" >&2; exit 9; }
done
REMOTE

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
scp -q "$host:$remote/a/libglm53_display_kv.so" "$host:$remote/a/display_kv_probe" \
    "$host:$remote/a/toolchain.txt" "$scratch/"
python3 - "$mode" "$build_json" "$scratch" "$image" "$dir" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
mode, build_json, scratch, image, src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "image": image,
    "network": "none",
    "reproducible": "built twice, bit-identical",
    "toolchain": (scratch / "toolchain.txt").read_text().strip().splitlines(),
    "sources": {n: sha(src / n) for n in ("display_kv.c", "probe_main.cu")},
    "outputs": {n: sha(scratch / n) for n in ("libglm53_display_kv.so", "display_kv_probe")},
}
if mode == "--write":
    build_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
else:
    expected = json.loads(build_json.read_text())
    # Image metadata may differ under the verified operator policy. All build
    # inputs, toolchain lines and output bytes must still match the reference.
    if {k: v for k, v in expected.items() if k != "image"} != {k: v for k, v in record.items() if k != "image"}:
        sys.exit("REFUSE: build does not match BUILD.json")
for name in record["outputs"]:
    shutil.copyfile(scratch / name, src / name)
    (src / name).chmod(0o755)
print("PASS " + json.dumps(record, sort_keys=True))
PY
ssh -o BatchMode=yes "$host" "rm -rf ~/$remote"
