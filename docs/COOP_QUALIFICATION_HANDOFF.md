# v1.8.4 component qualification handoff

Run this document from the exact reviewed commit, alongside its source. It stages
component qualification and produces a separate reviewable seal. It does not
schedule or authorize serving changes. Final release validation intentionally
refuses until measured component and serving evidence is integrated.

## 1. Exact source and two independent builders

Use Bash on each builder, as the ordinary operator with local Docker access.
All example outputs live below `$HOME`; no root-owned staging directory is needed.
Use the existing serving fleet's original model directory as `TARGET_SNAPSHOT`.
Only **read access** is needed: the commands authenticate and copy a subset into
a separate fixture, and never change the serving model or load it onto a GPU.

Set `REPO` to an existing repository containing the reviewed tag/commit and `REV`
to that tag or full commit. Export with `git archive` so ignored local material
cannot enter the source tree. Resolve the tag once and use that exact full commit
on both builders. The archive and resolved revision are retained in staging;
there is no dependency on a temporary tarball supplied by an implementer.

On the first builder:

```sh
set -euo pipefail
export REPO="$HOME/jspark3"
: "${REV:?set REV to the reviewed tag or full commit}"
export REV="$(git -C "$REPO" rev-parse --verify "$REV^{commit}")"
export STAGING="$HOME/jspark3-v184-qualify"
test ! -e "$STAGING"
mkdir "$STAGING"
printf '%s\n' "$REV" > "$STAGING/revision.txt"
git -C "$REPO" archive --format=tar --output="$STAGING/source.tar" "$REV"
sha256sum "$STAGING/source.tar"
export SOURCE="$STAGING/source"
mkdir "$SOURCE"
tar -xf "$STAGING/source.tar" -C "$SOURCE"
export CAMPAIGN="$STAGING/campaign"
export BUILD="$CAMPAIGN/build-first"
export SECOND_BUILD="$CAMPAIGN/build-second"
export QUAL="$CAMPAIGN/component"
export SEALED_OUTPUT="$CAMPAIGN/sealed-coop"
export QUAL_FLY="$CAMPAIGN/fly"
export QUAL_HELPERS="$CAMPAIGN/helpers"
export QUAL_MODEL="$CAMPAIGN/fixtures/25a44fdbf16862a46b7cc9921142c6c81350af2f"
export QUAL_SANITIZER="$CAMPAIGN/sanitizer/usr/local/cuda-13.4/compute-sanitizer"
: "${TARGET_SNAPSHOT:?export the absolute existing original serving model directory}"
mkdir -p "$BUILD" "$SECOND_BUILD" "$QUAL_HELPERS" "$QUAL_MODEL"
cd "$SOURCE"
python3 -B tools/validate_release.py .
```

### Memory and health budget

Use host `/proc/meminfo` **MemAvailable**, not GPU free-memory reporting. The
owner's pre-fix rehearsal had about 13.7–15.3 GiB available beside serving. Run
only one staging/build/check operation per host at a time, with the owner's
existing one-second memory and endpoint-health guard running throughout.

| Operation | Container CPU / memory / total memory+swap limit | Minimum host MemAvailable before starting |
| --- | --- | --- |
| Image construction, dedicated builder below | 4 / 8 GiB / 8 GiB | 12 GiB (12,582,912 KiB) |
| Native compilation | 4 / 8 GiB / 8 GiB | 12 GiB (12,582,912 KiB) |
| CPU environment / check-only | 4 / 4 GiB / 4 GiB | 8 GiB (8,388,608 KiB) |
| Exclusive GPU gates | 4 / 16 GiB / 16 GiB | 20 GiB (20,971,520 KiB), after owner stops competing serving work |

The difference is a 4 GiB host reserve. Stop the current owned operation if
MemAvailable drops below **4 GiB (4,194,304 KiB)** or serving health deteriorates;
do not stop another container to make room. These start checks do not replace the
continuous owner guard. The limits prohibit extra container swap; they do not
prove that the host has no pre-existing swap activity. CPU check thread pools
are also limited to one thread. The earlier 16 GiB check-only cap is now 4 GiB;
the rehearsal's sampled check peak was 684 MiB, with no GPU exposed.

Image construction uses Buildx and has no default cap in the tool. The commands
below create a **new local docker-container builder** with 4 CPUs, 8 GiB RAM and
8 GiB total memory+swap using Docker's [documented driver options](https://docs.docker.com/build/builders/drivers/docker-container/).
Do not replace it with an unbounded existing builder beside serving. Stop this
owned builder after image creation, before native compilation. If image building
fails or is interrupted, stop that same named builder before retrying. Keep the
12 GiB start/4 GiB abort guard; if unavailable, defer image creation to maintenance.
The prior 4–12 second image-build observations came from warm dependencies, not a
cold-build resource guarantee. Fresh builders may take substantially longer.
Image receipts must be freshly generated for this source; do not edit old receipts.
Defer staging copies if the guard trips. The fixture copy is about 4.43 GB on disk.

Run image construction under that policy, followed by native compilation:

```sh
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
export IMAGE_BUILDER=jspark3-v184-first
# Use a fresh name; create refuses a pre-existing builder without --append.
docker buildx create --name "$IMAGE_BUILDER" --driver docker-container \
  --driver-opt memory=8g,memory-swap=8g,cpu-quota=400000,cpu-period=100000
python3 -B tools/build_operator_image.py --builder "$IMAGE_BUILDER" --output "$BUILD/operator-image.json"
docker buildx stop "$IMAGE_BUILDER"
python3 -B recipe/scripts/remote_preflight.py --recipe-root "$SOURCE/recipe" \
  --image-only --image-receipt "$BUILD/operator-image.json"
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
python3 -B tools/build_native.py --image-receipt "$BUILD/operator-image.json" \
  --output "$BUILD/native"
sha256sum "$BUILD/native/recipe/overlays/v16/coop/bundle/cooperative_moe.so"
```

Require exactly
`3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`.
Display and coop are each built twice; retain all raw output. Compiler containers
hide GPUs. The builder reads
`/usr/bin/nvidia-smi --query-gpu=name,uuid --format=csv,noheader` on the host
before and after compiling, without launching GPU work. It requires a root-owned,
non-writable-by-group/others executable and parent directories (including resolved
symlink targets), supplies only a fixed system PATH and `LC_ALL=C`, and uses `/`
as its working directory. Inherited PATH shims, loader variables and GPU overrides
are not used for this query. NVIDIA
[documents GPU UUIDs as immutable identifiers](https://docs.nvidia.com/deploy/nvidia-smi/).
The integrated GB10 identifies the physical Spark board. The owner independently
observed distinct UUIDs despite identical cloned `/etc/machine-id` values.

Schema-2 native receipts hash the normalized UUID, identify its GB10 origin,
and retain machine-id only as diagnostic metadata. Exactly one GB10 and native
ARM64 execution are required for independent qualification. Use a local Docker
Unix socket, including a local rootless socket; remote Docker contexts are
refused so a client's hardware cannot stand in for the actual build host. x86
emulation remains diagnostic and cannot satisfy the independent-builder check.
Run the collector directly in the ordinary host shell, outside containers,
`unshare`/`nsenter` sessions, chroots and service sandboxes. It requires the initial
Linux PID/user/IPC/UTS/cgroup/time namespaces, PID 1's root mount view and the
initial network namespace. Missing, unreadable or unsupported namespace evidence
refuses collection rather than issuing an identity. These are collection hygiene
checks on the supported Spark Linux environment; no root escalation is needed.
Both hosts must rebuild receipts from this reviewed source, even if their previous
receipts already used schema 2. Do not upgrade evidence by editing it.

Builder identity is **operator-attested**. The trusted operator runs the two-board
determinism check on their own hardware and reports that each compiled the same
pinned bytes. The checks catch accidental duplicate identity, PATH substitution
and collection from an isolated environment. Receipt self-hashes detect drift;
they cannot authenticate an edited receipt or establish physical provenance
against a malicious operator. No hardware attestation is claimed. Anyone can
independently rebuild the source and compare its output with the pinned native
SHA-256 above. See the [seal field descriptions](COOP_REPRODUCIBILITY.md#builder-evidence-fields)
for the precise scope of that evidence.

On a **second physical Spark**, use the same full `REV` from `revision.txt` and a
repository containing it. Apply the same image-build and headroom policy there:

```sh
set -euo pipefail
export REPO="$HOME/jspark3"
: "${REV:?set the exact full reviewed commit used on the first builder}"
export SECOND_STAGING="$HOME/jspark3-v184-second"
test ! -e "$SECOND_STAGING"
mkdir "$SECOND_STAGING"
git -C "$REPO" archive --format=tar --output="$SECOND_STAGING/source.tar" "$REV"
sha256sum "$SECOND_STAGING/source.tar"
export SOURCE2="$SECOND_STAGING/source"
export BUILD2="$SECOND_STAGING/build"
mkdir "$SOURCE2" "$BUILD2"
tar -xf "$SECOND_STAGING/source.tar" -C "$SOURCE2"
cd "$SOURCE2"
python3 -B tools/validate_release.py .
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
export IMAGE_BUILDER=jspark3-v184-second
docker buildx create --name "$IMAGE_BUILDER" --driver docker-container \
  --driver-opt memory=8g,memory-swap=8g,cpu-quota=400000,cpu-period=100000
python3 -B tools/build_operator_image.py --builder "$IMAGE_BUILDER" --output "$BUILD2/operator-image.json"
docker buildx stop "$IMAGE_BUILDER"
python3 -B recipe/scripts/remote_preflight.py --recipe-root "$SOURCE2/recipe" \
  --image-only --image-receipt "$BUILD2/operator-image.json"
awk '/^MemAvailable:/ {exit ($2 < 12582912)}' /proc/meminfo
python3 -B tools/build_native.py --image-receipt "$BUILD2/operator-image.json" --output "$BUILD2/native"
sha256sum "$BUILD2/native/recipe/overlays/v16/coop/bundle/cooperative_moe.so"
```

Require matching source archive hashes and the same native target. Independent
eligible image config digests may differ. Do not copy the first binary as a build.

### Exact transfer through the controller

Run this on the existing SSH controller, where both known host aliases work;
no builder-to-builder SSH setup is assumed. Set the four variables to the actual
aliases and the absolute output paths printed/expanded on the builders. The first
host's destination is the empty `SECOND_BUILD` directory created above. `rsync`
must be installed on the controller and builders. Keep the entire native tree.

```sh
set -euo pipefail
: "${FIRST_HOST:?existing first-builder SSH alias}"
: "${SECOND_HOST:?existing second-builder SSH alias}"
: "${SECOND_REMOTE_BUILD:?absolute BUILD2 path on the second builder}"
: "${FIRST_REMOTE_SECOND_BUILD:?absolute SECOND_BUILD path on the first builder}"
TRANSFER="$(mktemp -d "$HOME/jspark3-v184-transfer.XXXXXX")"
rsync -a -s -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  "$SECOND_HOST:$SECOND_REMOTE_BUILD/" "$TRANSFER/"
rsync -a -s -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  "$TRANSFER/" "$FIRST_HOST:$FIRST_REMOTE_SECOND_BUILD/"
```

Return to the first builder's shell with its original exports. Check-only below
validates the transferred source/image/native bindings and rejects the same GB10
twice **before starting any container**. Full campaign and seal repeat this check;
sealing also requires the same second-build evidence retained at campaign start.

Fetch the upstream helper and its original license at the exact pinned revision:

```sh
curl -fL --retry 2 \
  https://raw.githubusercontent.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/2fb09425ab644f30c6b61fe3da2d567b99b36464/tests/test_exl3_overlay.py \
  -o "$QUAL_HELPERS/test_exl3_overlay.py"
curl -fL --retry 2 \
  https://raw.githubusercontent.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/2fb09425ab644f30c6b61fe3da2d567b99b36464/LICENSE \
  -o "$QUAL_HELPERS/LICENSE"
python3 -B - <<'PY'
import hashlib,json,os,pathlib
pin=json.loads(pathlib.Path('recipe/config/coop-helper.json').read_text())
root=pathlib.Path(os.environ['QUAL_HELPERS'])
for name,expected in [('test_exl3_overlay.py',pin['sha256']),('LICENSE',pin['license_sha256'])]:
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print('PASS exact helper and AGPL-3.0-only license')
PY
git clone https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks "$QUAL_FLY"
git -C "$QUAL_FLY" checkout 9093765c757bd1976372196e44af84a67cf86bad
curl -fL --retry 2 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-sanitizer-13-4_13.4.92-1_arm64.deb \
  -o "$CAMPAIGN/sanitizer.deb"
python3 -B - <<'PY'
import hashlib,json,os,pathlib
pin=json.loads(pathlib.Path('recipe/config/coop-sanitizer.json').read_text())
assert hashlib.sha256((pathlib.Path(os.environ['CAMPAIGN'])/'sanitizer.deb').read_bytes()).hexdigest()==pin['package_sha256']
print('PASS NVIDIA package hash')
PY
dpkg-deb -x "$CAMPAIGN/sanitizer.deb" "$CAMPAIGN/sanitizer"
```

Helper SHA-256:
`5814b64fbac04b04a770e78b6d3efd4601a0e6dc9be686fa0c3f00697be6b837`.
License SHA-256:
`8d56b405468aad11f87ab5763f901e276e08d9646ff5c8481b1762b6b789e9ed`.
The isolated helper directory must contain only those two files. Dependencies
are stdlib, torch, the exact stage-8 vLLM EXL3/fatpath, exllamav3_ext and safetensors.
The pinned NVIDIA tool is mounted read-only; it is not baked into a different
qualification image or redistributed in the release.

Materialize the required existing shards with ordinary files, not HF symlinks.
This copies the exact original index and only shards containing layer-3 tensors
for all 288 experts. Fetch and authenticate the original publication ledger
before maintenance; the fixture retains it, and every shard is checked against
its trusted digest both before and after copying:

```sh
curl -fL --retry 2 \
  https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/25a44fdbf16862a46b7cc9921142c6c81350af2f/SHA256SUMS \
  -o "$CAMPAIGN/checkpoint-SHA256SUMS"
python3 -B - <<'PY'
import json,os,pathlib,subprocess,sys
sys.path.insert(0,'recipe/scripts')
import _coop_checkpoint as fixture
source=pathlib.Path(os.environ['TARGET_SNAPSHOT'])
out=pathlib.Path(os.environ['QUAL_MODEL'])
publication=pathlib.Path(os.environ['CAMPAIGN'])/'checkpoint-SHA256SUMS'
ledger=fixture.checkpoint.ledger(publication)
pin=fixture.authority()
index=source/'model.safetensors.index.json'
assert fixture.checkpoint.sha(index)==pin['files']['model.safetensors.index.json']
weights=json.loads(index.read_text())['weight_map']
names={'model.safetensors.index.json'}
for e in range(288):
 for projection in ('gate_proj','up_proj','down_proj'):
  for suffix in ('trellis','suh','svh','mcg'):
   names.add(weights[f'model.language_model.layers.3.mlp.experts.{e}.{projection}.{suffix}'])
assert names==set(pin['files'])
for name in sorted(names):
 assert pathlib.Path(name).name==name and not (out/name).exists()
 assert ledger[name]==pin['files'][name]
 assert fixture.checkpoint.sha(source/name)==ledger[name],name
 subprocess.run(['cp','--reflink=auto','-L',str(source/name),str(out/name)],check=True)
assert not (out/'SHA256SUMS').exists()
subprocess.run(['cp',str(publication),str(out/'SHA256SUMS')],check=True)
assert fixture.authenticate(out)==pin
fixture.verify_shapes(out)
print('PASS publication-authenticated index/shards and exact CPU tensor metadata')
PY
```

Ledger SHA-256 is `cb0da1f97a53aebc3fbc5478f19c82b25586b0bf8533c99fb4ed5321a48f5342`.
The exact shard subset is pinned in `recipe/config/coop-checkpoint.json`; its
repository, revision, publication authority and expected file hashes enter the
seal. A self-consistent observed hash cannot replace this authority. Trellis
tensors must be I16 with all three dimensions, suh/svh F16 vectors of the exact
projection widths, and mcg I32 `[1]`.

## 2. Check-only first; no GPU reservation or endpoint change

```sh
cd "$SOURCE"
awk '/^MemAvailable:/ {exit ($2 < 8388608)}' /proc/meminfo
python3 -B tools/v16/qualify_coop.py \
  --image-receipt "$BUILD/operator-image.json" --build-root "$BUILD/native" \
  --model-root "$QUAL_MODEL" --fly-root "$QUAL_FLY" --helpers-root "$QUAL_HELPERS" \
  --sanitizer-root "$QUAL_SANITIZER" --output "$QUAL" --check-only \
  --independent-build-root "$SECOND_BUILD/native" \
  --independent-image-receipt "$SECOND_BUILD/operator-image.json"
```

Require PASS and retain `$QUAL.check/plan.json`, `check.json`,
`container-environment/environment.json`, `environment.log` and cleanup evidence.
The tool checks current source/build/image receipts, helper/license/package
hashes, all fixture shards, complete CPU tensor shapes, exact stage-8 transforms,
imports, shared cudart loading and sanitizer launch-evidence options. It hides
GPUs explicitly. Every later GPU command is printed and retained, but not run.
A new check attempt requires a new QUAL name. Failed checks preserve diagnostics.
Retain `independent-build.json` and `plan-estimates.json` alongside the commands.
The pre-fix owner rehearsal passed with real fixtures and stage-8 imports in
17 seconds. That is historical CPU evidence: rerun this revised check-only with
fresh schema-2 receipts and the reduced cap before reserving hardware.

`plan-estimates.json` gives a seconds range for each of the 56 planned steps,
including the CPU environment step. GPU ranges are explicitly **unmeasured,
low-confidence planning allowances**, allocated to the owner's 1.5–4 hour
estimate; their sum is 5,417–14,430 seconds (about 1.5–4 hours). They are not
gate timeouts or performance claims. Container execution receipts record actual
start/end timestamps for replacing these estimates after the first campaign.

Historical staging wall times from one owner rehearsal (not guarantees):

| Step | Observed wall seconds |
| --- | --- |
| Archive staging / source validation | 1 / 2 |
| Image build, warm dependencies, first / second host | 4 / 12 |
| Image preflight, each host | 1 |
| Display+coop, twice each, first / second host | 13 / 14 |
| Native hash / helper download / sanitizer staging | 1 / 1 / 2 |
| Checkpoint authentication and subset copy | 10 |
| CPU check-only / second-builder transfer | 17 / 2 |

Cold image/download timing is unknown. The GPU estimate excludes staging, drain,
seal/review, fleet restore and subsequent serving qualification. Budget those
separately; a serving restore can dominate the window. No GPU gate was timed in
the check-only rehearsal.

## 3. Exclusive component campaign, then a new-directory seal

Only the hardware owner drains/closes traffic and stops the bound fleet through
its old controller, preserving exact old container IDs and all evidence. Confirm
no competing workload on GPU 0. The component runner never performs those actions.
All nine logical EP range/geometry pairs run serially on one reserved GB10.

```sh
cd "$SOURCE"
awk '/^MemAvailable:/ {exit ($2 < 20971520)}' /proc/meminfo
python3 -B tools/v16/qualify_coop.py \
  --image-receipt "$BUILD/operator-image.json" --build-root "$BUILD/native" \
  --model-root "$QUAL_MODEL" --fly-root "$QUAL_FLY" --helpers-root "$QUAL_HELPERS" \
  --sanitizer-root "$QUAL_SANITIZER" --output "$QUAL" \
  --independent-build-root "$SECOND_BUILD/native" \
  --independent-image-receipt "$SECOND_BUILD/operator-image.json"
python3 -B tools/v16/qualify_coop.py --seal "$QUAL" --output "$SEALED_OUTPUT" \
  --independent-build-root "$SECOND_BUILD/native" \
  --independent-image-receipt "$SECOND_BUILD/operator-image.json"
python3 -B tools/v16/qualify_coop.py --check-seal "$SEALED_OUTPUT"
```

Success requires 18 H1 invocations (nine baseline/perturb pairs), nine profiles
with 468 complete cases, 18 smoke sanitizer runs, six geometry-2 sanitizer runs,
and three production-policy checks. The selector uses the frozen worst-case
candidate/stock threshold ≤0.97; unmeasured/nonwinning rows, including row 64 if
it loses, stay stock. The numerical and graph/allocation tolerances are unchanged.
The runner validates every gate receipt/log/environment/execution, selected policy
and profile hash before sealing. Each policy run binds its exact selected bundle;
selection may alter only `dispatch_policy.json` and that entry in the manifest.
The runtime, native, headers, source/license/provenance files and toolchain record
must remain identical to the original qualified bundle. Source copies and builder
are checked against the trusted recipe, not their own replacement hashes.
A sanitizer summary with no instrumented candidate
launches, contradictory summaries, unknown output format or any unexplained error
refuses. Never hand-edit a PASS or weaken a detector to advance.

SIGINT/SIGTERM/SIGHUP cleanup uses only the exact ID from that command's cidfile.
Every attempt retains timestamps, logs, source/image/helper/checkpoint/toolchain
and GPU identity. Keep the full campaign, both native build trees, independent
image receipts and `$SEALED_OUTPUT` for review. No raw build is overwritten.
The legacy `build_coop_moe.sh ... seal` now refuses profile-only sealing.

## 4. Pass 2 after independent evidence review

1. Integrate only publishable source metadata/evidence. Never distribute `.so`,
   raw compiler intermediates, model shards, NVIDIA package contents or private logs.
   Copy the measured policy into both source and bundle; refresh the source
   manifest, embedded/INSTALL pins and inventories. Preserve exact stage-8 and
   footer hashes when those bytes are unchanged.
2. Set BUILD's final source-manifest hash while preserving its separate original
   qualification source-manifest identity. Verify its compiled-input map remains
   exact. Rebuild the final source twice on the first machine and independently
   on the second; every native result must retain the target digest. Keep the
   original component campaign as pre-policy evidence, not a claim it ran later bytes.
3. Pin the actual reviewed BUILD digest, gate index, policy and native in
   `recipe/config/coop-release.json`, `manifests/binaries.json` and the distinct
   v1.8.4 delivery binding/catalog. No null, dummy profile hash or old delivery
   record may be promoted. The prepared branch presently refuses final release.
4. Finish and review the owner's edited-mode admission handoff and pin/adapt the
   retained decode benchmark tooling before scheduling the two serving starts.
   The owner's retained epoch-oriented decode harness must be reviewed for
   read-only default-state evidence without runtime overrides. This serving and
   measurement work remains a separate prerequisite for the serving window.
5. Use the real public image→native→prepare→fleet→stock qualify_runtime commands
   from an ordinary clone of the final candidate on real Sparks. Compare all
   prepared files with the independent archive path. Require coop=1, stock ABLIT=0,
   APC=1, full native stock/capture/loader proofs, unchanged per-rank identities,
   post-hygiene ≥1100, no swap/OOM/late compilation, and inactive TRIAR/thirds off.
6. Record the two complete stock decode sweeps, eight post-hygiene prefill turns
   and full-model quality witnesses in the separate v1.8.4 result record. Stop
   and retain that fleet; create a new edited ABLIT=1 boot with the same qualified
   native/policy and a separately reviewed edited admission. Leave that newly
   admitted edited endpoint serving. Never flip coop or ABLIT under captured graphs.
7. Final source/clone/archive/inventory/privacy/publication review, then the release owner's
   explicit release go. This task does not push or publish.

Frozen `release/results.json`, `release/results-v1.8.0.json` and
`release/RELEASE-NUMBERS.md` must remain byte-identical to base 4a5348c.
