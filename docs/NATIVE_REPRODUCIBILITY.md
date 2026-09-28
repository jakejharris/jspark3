# Native build reproducibility experiment

This experiment builds all three compared artifacts three times per host:
`libglm53_display_kv.so`, `display_kv_probe`, and `cooperative_moe.so`. Each build
uses a fresh GPU-hidden ARM64 container, the production builder commands, and
deliberately different pre-compiler process counts (0, 7, 19). All nine pairwise
ELF comparisons are retained, including section/symbol differences on failure.
It never executes a probe or kernel and does not produce a native admission
receipt, component qualification, or permission to change serving.

## Source transfer

For an unpublished candidate, create a Git bundle on the controller containing
the reviewed branch. Set `REPO` to the existing repository and `REV` to the full
reviewed commit from the implementer's result. Use a new bundle destination:

```sh
: "${REPO:?existing repository containing the reviewed branch}"
: "${REV:?full reviewed commit}"
: "${BUNDLE:?new absolute bundle filename outside the repository}"
test ! -e "$BUNDLE"
test "$(git -C "$REPO" rev-parse feat/v184-coop-on-v183)" = "$REV"
git -C "$REPO" bundle create "$BUNDLE" feat/v184-coop-on-v183
git -C "$REPO" bundle verify "$BUNDLE"
```

Transfer that file to each builder through the existing approved controller
transport. On each host, choose a new repository destination and preserve it:

```sh
: "${LOCAL_BUNDLE:?absolute transferred bundle filename}"
: "${REPO:?new repository destination on this builder}"
test ! -e "$REPO"
git clone --no-checkout "$LOCAL_BUNDLE" "$REPO"
export REPO
```

An existing repository containing `REV` works as well. No publish, builder-to-builder
SSH, remote branch availability, or default `$HOME/jspark3` path is assumed.

## One round on each Spark

Run in the ordinary host shell, one host at a time, under the existing one-second
memory and endpoint-health guard. Apply the full [handoff resource policy](COOP_QUALIFICATION_HANDOFF.md#memory-and-health-budget):
12 GiB `MemAvailable` before image/native work; abort the owned operation below
4 GiB or on deteriorating serving health. Compiler containers have 4 CPUs, 8 GiB
RAM and 8 GiB total memory+swap, no network and hidden GPUs. SIGINT/SIGTERM/SIGHUP
stop only the current experiment container by its exact ID and retain its files.

Set `REV` to the **same full reviewed commit on both hosts**, `STAGING` to a new
absolute directory outside the repository, and `IMAGE_BUILDER` to a fresh local
Buildx builder name. No checkpoint, helper, sanitizer or serving changes are needed.

```sh
set -euo pipefail
: "${REPO:?existing repository containing REV}"
: "${REV:?full reviewed commit}"
: "${STAGING:?new absolute private staging directory}"
: "${IMAGE_BUILDER:?fresh local Buildx builder name}"
test "$(git -C "$REPO" rev-parse --verify "$REV^{commit}")" = "$REV"
test ! -e "$STAGING"
mkdir -m 700 "$STAGING"
git -C "$REPO" archive --format=tar --output="$STAGING/source.tar" "$REV"
sha256sum "$STAGING/source.tar"
mkdir "$STAGING/source"
tar -xf "$STAGING/source.tar" -C "$STAGING/source"
cd "$STAGING/source"
python3 -B tools/validate_release.py .
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
docker buildx create --name "$IMAGE_BUILDER" --driver docker-container \
  --driver-opt memory=8g,memory-swap=8g,cpu-quota=400000,cpu-period=100000
trap 'docker buildx stop "$IMAGE_BUILDER"' EXIT
python3 -B tools/build_operator_image.py --builder "$IMAGE_BUILDER" \
  --output "$STAGING/operator-image.json"
docker buildx stop "$IMAGE_BUILDER"
trap - EXIT
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
python3 -B tools/v16/experiment_native_build.py \
  --image-receipt "$STAGING/operator-image.json" --output "$STAGING/native-repro"
```

Require `PASS` on both hosts, three runs and nine identical comparisons in each
`native-repro/report.json`. Retain the entire mode-0700 experiment directory
privately on each host, even on failure. Compiler output and raw ELF detail are
in private diagnostic sidecars; projected JSON files contain hashes and fixed
structural facts. Share the two `report.json` files through the controller, then
run this command from the same reviewed source with their actual local paths:

```sh
python3 -B tools/v16/experiment_native_build.py --compare \
  /absolute/first/report.json /absolute/second/report.json
```

Require `PASS`, three matching artifact hash pairs and
`independent_native_builders: true`. The comparison checks exact source identities,
complete passing reports and distinct GB10 identity hashes. Identity remains
operator-attested; report self-hashes detect drift, not malicious editing. Image
configuration digests may differ. `hardware_qualified` remains false: deterministic
compilation does not establish kernel correctness or serving performance.

If any build differs, share its projected comparisons and report; preserve both
original binaries and full private compiler/ELF diagnostics for local inspection.
Do not strip or normalize the output to force equality. Repeated process IDs can
hide the original defect, which is why the experiment varies process counts.
`--allow-emulated` is available solely for local CPU rehearsal; it cannot produce
an independent native-builder result.

## Display fix and scope

The legacy one-shot nvcc link embedded a PID-derived
`tmpxft_…_probe_main.cudafe1.cpp` file symbol in `.strtab`. Local retained failures
showed that section alone changing. The fixed builder keeps the CUDA host stub
under a stable filename, compiles named objects with distinct per-translation-unit
seeds, and links without modifying the resulting ELF. NVIDIA documents
[`--frandom-seed`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#frandom-seed-frandom-seed)
and forwarding the seed to supported host compilers. Both display entry points
use that script. The coop builder already uses stable intermediates and seeds;
its reviewed target and compilation inputs are unchanged.

The same default display comparison exists in v1.8.3. Its fix is inherited here.
Historical display pins remain historical; fresh operator image/native receipts
bind the current source and observed deterministic bytes. Do not edit or reuse
old receipts as evidence for this source.
