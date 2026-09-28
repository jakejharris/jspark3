# Install JSpark3 v1.8.3

JSpark3 serves GLM-5.3 Flash with EXL3/TR3 quantization across three DGX Sparks.
The target's routed experts use 4 bits per weight; "stock" means unedited
(`ABLIT=0`), not full precision. The default is `production-stock`, coop off,
adaptive-K `ema`, dense FP8 `trunk`, and display profile `full`.
The required DFlash2 draft model is non-commercial; read [licensing](LICENSING.md)
before choosing this stack.

## Requirements

- Three DGX Sparks (GB10, ARM64), NVIDIA container GPU support, Python 3.10+
  and Docker with Buildx. Docker must use cgroup v2 and the systemd cgroup driver.
- A pairwise RoCE triangle: two fabric interfaces per rank, each pair sharing
  a distinct IPv4 subnet, MTU 9000, and one common IPv4 RoCE-v2 GID index across
  all six HCAs. A single shared fabric subnet is refused. Direct links are the
  measured topology; switched/VLAN and NFS deployments are not qualified here.
- A management LAN for SSH, API and NCCL/Gloo bootstrap. Discover interface/HCA
  pairs with `ibdev2netdev`, GIDs with `show_gids`, and addresses with `ip -br addr`.
  Management routes must use the declared socket interface.
- Non-interactive SSH from the controller to all three ranks, including itself
  if rank 0 is the controller. Its SSH user must run Docker and `systemctl show`
  without sudo. Test `ssh -o BatchMode=yes HOST docker info` on each host.
- At least 72 GiB `MemAvailable` before start and 8 GiB free on each model/work
  filesystem. Each container has a 64 GiB memory limit and effective zero swap.
  Stop the previous serving fleet before preflight; see [upgrades](OPERATIONS.md#stop-restart-and-upgrade).

Plan for roughly 164 GiB of target shards, 2.34 GB of draft weights and a
21 GB serving image per Spark, plus base-image/build layers, transfer archives
and caches. Keep additional free space for those temporary copies. The two
base images share layers (about 9.9 GB compressed in the audited manifests).
One clean-room run measured 4m48s for full preflight and 8m05s from start to API
health. Cold downloads and cold image-build duration were not measured.

The controller is the machine running `fleetctl.py`: rank 0 or another Linux
host with SSH to all ranks and HTTP access to rank 0. The recipe, model, Fly
source and work roots must each have the **same absolute path on every Spark**.
Use new, dedicated recipe/work directories for each installation.

## 1. Obtain and validate the source

Download the recipe tarball and its distribution `SHA256SUMS` from the
[release page](https://github.com/jakejharris/jspark3/releases). Choose the
v1.8.3 assets when published; this document describes that source version.
In the download directory, verify the tarball before extracting it:

```sh
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf jspark3-recipe-v1.8.3.tar.gz
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py . --require-final
export JSPARK_SRC="$PWD"
export JSPARK_RUNTIME="$(realpath -m ../jspark3-runtime-v1.8.3)"
```

The exported tarball, `git archive`, and a normal clone checked out at the release
tag all work. Once published, the Git alternative is
`git clone --branch v1.8.3 https://github.com/jakejharris/jspark3.git`.
Validation scans the current file names and contents, including untracked files;
Git metadata and history are outside the source-export privacy guarantee.
Keep builds and evidence outside the source directory. The following controller
blocks use one shell.

## 2. Build one image, then transfer that image

Run on **one of your Sparks**, with Docker access, from the validated source:

```sh
python3 -B tools/build_operator_image.py --output ../operator-image.json
python3 -B recipe/scripts/remote_preflight.py --recipe-root "$JSPARK_SRC/recipe" \
  --image-only --image-receipt ../operator-image.json
export JSPARK_IMAGE_RECEIPT="$(realpath ../operator-image.json)"
JSPARK_IMAGE_ID="$(python3 -B recipe/scripts/_image_identity.py --receipt "$JSPARK_IMAGE_RECEIPT")"
docker save "$JSPARK_IMAGE_ID" -o ../operator-image.tar
```

Copy that tar to each other Spark, then run `docker load -i operator-image.tar`
there. Use the same image on all ranks, subject to upstream terms. Independent
image builds can differ in metadata. There is no JSpark3 image to pull from
GHCR; the Dockerfile pulls two public digest-pinned upstream bases. Ignore the
weight card's separate image instructions. The unchanged Dockerfile's labels
say v1.3.0; the operator receipt identifies the actual image.

## 3. Build native artifacts and prepare the runtime

Run on that same Spark, using its verified image. GCC/CUDA run inside ARM64
containers; a host CUDA compiler and GPU access are not needed for compilation.

```sh
python3 -B tools/build_native.py --image-receipt "$JSPARK_IMAGE_RECEIPT" --output ../native-build
python3 -B tools/prepare_runtime.py --binary-root ../native-build \
  --native-receipt ../native-build/native-build-receipt.json \
  --image-receipt "$JSPARK_IMAGE_RECEIPT" --output "$JSPARK_RUNTIME"
python3 -B "$JSPARK_RUNTIME/recipe/scripts/remote_preflight.py" \
  --recipe-root "$JSPARK_RUNTIME/recipe" --recipe-only
cp "$JSPARK_RUNTIME/recipe/.env.example" "$JSPARK_RUNTIME/operator.env"
```

The display library and probe must be byte-identical across two fresh builds.
Their receipt records the output hashes. Coop is neither built nor required;
the prepared example explicitly sets `JSPARK3_V16_COOP=0`. Keep it off.
Copy the **prepared** example above, not the source's historical coop-on example.

## 4. Stage the recipe, weights and runtime views on every Spark

Edit `operator.env` with your real hosts, addresses, paths, interfaces, GID and
DRM nodes. Use the same paths on all three. Leave `JSPARK3_TRIAR=1` and
`JSPARK3_TRIAR_P2P=auto`: the pinned Cadence path keeps TRIAR resident but
inactive, and admission checks that path. Do not create runtime epoch files
or enable active TRIAR. See [operations](OPERATIONS.md#triar-and-runtime-settings).

Copy `$JSPARK_RUNTIME/recipe/` to the declared `JSPARK_RECIPE_ROOT` on each host,
for example `rsync -a "$JSPARK_RUNTIME/recipe/" HOST:/srv/jspark3-recipe-v1.8.3/`.
Create the model, Fly parent and work directories as operator-writable paths
before the next commands. Do not merge the recipe into an old installation.

Run the following on **each Spark**. Set these paths to the same values used in
`operator.env`. Install the [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#install-with-pip)
in a virtual environment outside the source and recipe, if needed.

```sh
export JSPARK_MODEL_ROOT=/srv/models
export JSPARK_FLY_ROOT=/srv/sources/FlyCockpit-GLM-5.3-Flash-EXL3-3x-DGX-Sparks
export JSPARK_RECIPE_ROOT=/srv/jspark3-recipe-v1.8.3
export HF_HUB_DISABLE_XET=1
hf download Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw \
  --revision 25a44fdbf16862a46b7cc9921142c6c81350af2f \
  --local-dir "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native"
git clone https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks "$JSPARK_FLY_ROOT"
git -C "$JSPARK_FLY_ROOT" checkout 9093765c757bd1976372196e44af84a67cf86bad
M="$JSPARK_MODEL_ROOT"
cd "$JSPARK_RECIPE_ROOT"
python3 -B scripts/prepare_runtime_views.py \
  --target-source "$M/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb" \
  --target-view "$M/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime" \
  --draft-source "$M/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native" \
  --draft-view "$M/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime"
python3 -B scripts/validate_checkpoint.py \
  --target-root "$M/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb" \
  --target-runtime "$M/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime" \
  --draft-root "$M/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native" \
  --draft-runtime "$M/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime"
```

Every rank needs the complete checkpoint as real files; the validator hashes
all 120 target shards. Runtime views are sibling directories of relative
symlinks plus derived configs (target heads 64→66; draft heads 32/8→36/9).
Do not edit the downloads. Success includes `"serving_checkpoint_pass": true`.
Use the pinned Mia-AiLab revision above: the current JSpark3 HF mirror's ledger
can differ after card edits even when its weight shards are unchanged.

## 5. Select display configuration and isolate the endpoint

The shipped `full` profile adds a 1.75 GiB display-backed pool per rank. It
requires an NVIDIA-bound DRM card, `nvidia_drm` loaded with `modeset=1 fbdev=0`,
`multi-user.target`, and no active display manager, X/Wayland or remote desktop.
For a dedicated headless host, arrange the module parameters in your OS boot
configuration, run `sudo systemctl set-default multi-user.target`, and reboot
with console/SSH recovery available. Preflight checks the effective state.

If keeping a desktop, select `JSPARK3_V14_PROFILE=display0` and remove **all
three** `JSPARK_DRM_CARD_n` lines. That changes memory layout and needs its own
admission run; it is not the measured full-profile configuration.

API bind `0.0.0.0:8888` and master port `29533` are fixed. GPU memory utilization
must be 0.76, 0.78, 0.80 or 0.83. The API has no built-in authentication.
Firewall port 8888 to the controller while qualifying; later expose it only to
trusted clients or your authenticated gateway. An admission receipt does not
change firewall or routing rules.

## 6. Preflight, start, verify and admit

Back on the controller, keep ordinary traffic blocked and use the prepared
controller. All commands below retain receipts outside `recipe/`.

```sh
cd "$JSPARK_RUNTIME/recipe"
python3 -B scripts/fleetctl.py preflight --env-file ../operator.env --dry-run
python3 -B scripts/fleetctl.py preflight --env-file ../operator.env --output ../preflight.json
python3 -B scripts/fleetctl.py start --env-file ../operator.env \
  --preflight ../preflight.json \
  --preflight-sha256 "$(sha256sum ../preflight.json | cut -d ' ' -f1)" \
  --manifest ../service.json --confirm START-JSPARK3
python3 -B scripts/fleetctl.py verify --env-file ../operator.env \
  --manifest ../service.json --output ../verify.json --log-output ../verify-rank0.log
python3 -B "$JSPARK_SRC/tools/v16/qualify_runtime.py" \
  --recipe "$JSPARK_RUNTIME/recipe" --env-file "$JSPARK_RUNTIME/operator.env" \
  --manifest "$JSPARK_RUNTIME/service.json" --output "$JSPARK_RUNTIME/qualification"
```

`verify` waits up to 20 minutes for health and the expected model; use
`--ready-timeout SECONDS` to change the bound. A stopped/restarted rank or wrong
model refuses immediately. `status --env-file ../operator.env --manifest
../service.json` reports progress. Verify includes native stock checks inside
each container, correctness, long-context and effective memory/no-swap checks.

Qualification records the first prefill pass, applies page-cache hygiene on
all ranks, then requires post-hygiene prefill ≥1100 tok/s, the `finehit` cache
gate, zero compilation after warmup, native TRIAR-inactive proof and final
verification of the same boot. It produces the first-prompt and finalization
receipts consumed by `admission_gate.py`. Require `qualification/admission.json`
to say `PASS` before opening your routing. See [operations](OPERATIONS.md) for
failure handling and independent receipt rechecking.

## 7. First request

Use your rank 0 management address:

```sh
curl --fail-with-body http://RANK0_ADDR:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Write a haiku about three small computers."}],"max_tokens":200}'
python3 -B "$JSPARK_RUNTIME/recipe/scripts/api_smoke.py" --base-url http://RANK0_ADDR:8888
```

The OpenAI-compatible base URL is `http://RANK0_ADDR:8888/v1`. Thinking is off
by default; a request can set `"chat_template_kwargs":{"enable_thinking":true}`.
Configured limits are 1,000,000 context tokens, 32 sequences and 8 images per
prompt; these are limits, not throughput or quality guarantees. Bare JSON
clients should use `response_format` with `json_schema`.

## Receipt and coop policy

Image/native receipts bind known source hashes, build recipes and the actual
operator outputs. They detect drift; they are local records from a trusted
builder, not signed third-party attestations. Keep the validated source,
receipts, exact image and native files together. Without a native receipt,
`prepare_runtime.py` still requires the historical output hashes.

The earlier coop build differed across two native Spark builds. Stable build
intermediates/shared cudart are under [experiment](COOP_REPRODUCIBILITY.md),
not a promise of reproducibility. `build_native.py --with-coop` remains
experimental and does not enable serving. Operator `coop=1` needs future
runtime support: even a fresh BUILD.json seal for an operator image is refused
by the historical build-image gate. Do not edit old seals or profile hashes.

Edited mode (`ABLIT=1`, profile `production`) replaces attention output
projections in layers 15–45 using an operator-supplied donor checkpoint. This
release does not supply the donor or its download location. Its manifest,
revision, tensors and shapes remain checked. The admission command above is
for unedited production-stock only. Stock mode requires all donor keys absent,
including inherited `ABLIT_METHOD`, `ABLIT_LAYERS`, `ABLIT_INCLUDE_MTP`,
`JSPARK_ABLIT_ROOT` and `JSPARK_ABLIT_MANIFEST_SHA256`.
