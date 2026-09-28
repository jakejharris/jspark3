# Installation

Validate the source export before creating a separate private runtime copy.
The export has no native binaries, model weights, container layers or donor
weights. Hardware admission remains unqualified; see the local image procedure below.

1. Run `sha256sum -c SHA256SUMS` and the validator in the root README.
2. Build and verify a local image using the command below. Obtain the exact model
   and source dependencies identified in `recipe/config/`. Image/module
   redistribution is outside this source package's license posture.
3. On **one of your own DGX Sparks**, build the native artifacts using your
   verified image (Python/Docker on the host; GCC/CUDA inside the ARM64 image):
   `python3 -B tools/build_native.py --image-receipt ../operator-image.json --output ../native-build`.
   No host CUDA compiler or GPU access is needed for compilation. Another ARM64
   Docker host or Docker ARM64 emulation can also build them. It builds the display
   library and display probe twice in fresh network-disabled
   containers at `/w`, checks byte identity, and writes a local receipt binding
   the exact source files, build recipe, image receipt and output hashes.
   Coop is not built or required by default.
4. Run `python3 -B tools/prepare_runtime.py --binary-root ../native-build --native-receipt ../native-build/native-build-receipt.json --image-receipt ../operator-image.json --output RUNTIME_DIR`.
   It verifies the source export, native receipt and actual output hashes,
   copies `recipe/` and writes its runtime checksum inventory.
   Keep its runtime-build receipt. It installs the image receipt as
   `recipe/config/operator-image.json` and includes it in runtime checksums.
   The native receipt is installed as `recipe/config/operator-native.json`.
   The source export stays unchanged. Run the controller from this prepared
   runtime's `recipe/scripts/`, and copy this same recipe to all three hosts.
   Without `--native-receipt`, the original historical output pins remain mandatory.
   With a native receipt, preparation explicitly sets `JSPARK3_V16_COOP=0` in
   the prepared `recipe/.env.example`. Keep that setting; operator coop-on
   support requires a future implementation change.
   Check the prepared recipe before configuring the fleet:
   `python3 -B RUNTIME_DIR/recipe/scripts/remote_preflight.py --recipe-root "$(realpath RUNTIME_DIR/recipe)" --recipe-only`.
   This verifies runtime checksums without contacting GPUs or other hosts;
   it does not constitute full preflight or admission.
5. Copy the **prepared runtime recipe's** `.env.example` to a private environment
   file, preserving `JSPARK3_V16_COOP=0` for operator native builds. Do not copy
   the source export's example, which describes the historical coop-on setup.
   Replace documentation hosts, addresses, roots, interfaces and device paths.
   Point `JSPARK_RECIPE_ROOT` at the prepared runtime recipe on each host.
   Inspect `preflight`, `start`, `verify`, `status` and `stop` through
   `recipe/scripts/fleetctl.py COMMAND --env-file ENV_FILE --dry-run`.
6. Follow the qualification protocol in [operations](OPERATIONS.md) with
   admission closed. A dry-run, artifact hash match or source test cannot
   establish hardware correctness or permission to admit requests.

The example API port is `8888`, GPU memory utilization is `0.76`, and the
preflight requires at least 72 GiB available host memory. Container limits are
64 GiB with swap disabled. These are configuration requirements, not new
capacity measurements. The example uses `JSPARK3_V14_PROFILE=full`; operators
without the headless display setup can choose `display0` and omit all three
`JSPARK_DRM_CARD_n` keys. The available KV pool then differs.

For display-memory backing, the headless host configuration requires
`nvidia_drm modeset=1 fbdev=0`. Record and verify that host configuration and
the selected DRM device at preflight.

Stock mode uses `ABLIT=0`, `JSPARK3_V16_PROFILE=production-stock`, swap disabled,
and all five donor keys absent, including inherited environment keys:
`ABLIT_METHOD`, `ABLIT_LAYERS`, `ABLIT_INCLUDE_MTP`, `JSPARK_ABLIT_ROOT`, and
`JSPARK_ABLIT_MANIFEST_SHA256`. Each
rank must supply the exact native DISABLED receipt. Mode-0 production admission
requires fresh hardware qualification of the exact shipped configuration.
Historical stock measurements under the validation profile do not qualify production.

Edited mode is an explicit opt-in: `ABLIT=1`, profile `production`, donor root
and manifest digest supplied by the operator. The donor repository comes from
the operator's hash-pinned `MANIFEST.json`; revision, index, layer range, dtype,
shapes and tensor hashes remain checked. No repository name or donor payload
is embedded in this export. Mode changes require a restart and prefix
recomputation until a separately sealed and qualified switching implementation
is included.

## Image prerequisite and local preparation

Use Python 3 and Docker with Buildx on an ARM64 host (or an ARM64-emulating
builder). Allow space for both large upstream images and the resulting image.
From a clean source export, after step 1:

```sh
python3 -B tools/build_operator_image.py --output ../operator-image.json
python3 -B recipe/scripts/remote_preflight.py --recipe-root "$PWD/recipe" \
  --image-only --image-receipt ../operator-image.json
export JSPARK_IMAGE_RECEIPT="$(realpath ../operator-image.json)"
```

The build command accepts an optional `--builder NAME`. It builds only
`docker/stock-v13/Dockerfile`, with its two digest-pinned public GHCR bases,
without build cache, and loads the result locally. It never pushes. It checks
the Dockerfile against `recipe/config/image-build-policy.json`, verifies all
six InstantTensor file hashes by copying them from a stopped container, and
records the build's manifest digest, local config ID, layer identities and
source recipe checksum. No GPU is required for this image check. An
`image-only` PASS does not constitute fleet preflight or hardware admission.

Keep the receipt private and unedited. Receipts from a different source recipe,
unknown build inputs, changed layers or wrong InstantTensor bytes are refused.
Self-hashes detect drift; these receipts are local operator records, not signed
third-party attestations. Use a trusted local Docker daemon and builder.

Use that exact local image on all three hosts. Build once and transfer it within
your own fleet, subject to the upstream terms; for example, obtain its ID with
`python3 -B recipe/scripts/_image_identity.py --receipt ../operator-image.json`,
then use `docker save IMAGE_ID` and `docker load` on your other hosts. Do not
independently rebuild three times: metadata may differ. Preflight refuses a
different image on any rank. The native build helpers read `JSPARK_IMAGE_RECEIPT`;
their source, toolchain and output hash checks still apply.

The display artifacts must match across two fresh local builds. Their operator
receipt records the actual output hashes without rewriting published pins or
claiming that different binaries inherit historical GPU results. It is a self-hashed local
build record, with the same trust boundary as the image receipt, not a signature.
Receipt edits, source/build changes, a different image receipt and binary hash
mismatches are refused. Keep both receipts and the source export together.

Coop-MoE did **not** reproduce identical bytes across two fresh builds on a real
DGX Spark. The default build and prepared runtime therefore omit its binary.
`tools/build_native.py --with-coop` is an experimental opt-in that still requires
two-build byte identity and may fail; it neither enables coop nor qualifies it.
Use the default command above for installation.

Operator-native preparation selects `production-stock` with coop **off** and
display profile `full`; adaptive-K `ema` and dense FP8 `trunk` remain unchanged.
The source export's historical example has coop on, so use the prepared example.
Its environment and native outputs are all bound by the runtime checksum inventory.

The historical BUILD.json, bundle manifest and measured row policy remain intact.
`coop=1` still requires the historical sealed artifact and is refused for a missing
or changed operator binary. Enabling coop for operator builds requires a **future implementation
change**. The current `tools/v16/build_coop_moe.sh` seal procedure is not an
operator-image enablement route: runtime rejects a BUILD.json carrying the operator
image with `cooperative-MoE build image drift`, even after fresh profiling. Do not
rebind old profile hashes, edit the image identity, or enable coop to bypass that gate.

Build/prepare once and copy the exact image and prepared recipe to all three
Sparks. Full hardware, display/DRM, checkpoint, fabric, memory/no-swap and fresh
production qualification requirements still apply. Coop-off performance is not
established by the historical coop-on measurements.

The prepared runtime selects `config/operator-image.json` everywhere: controller,
remote preflight, per-container receipts and patch installers. Without that
file, the original `config/image-oci.json` selects the historical reference
for existing installations and source-only dry-runs. Its digest is not a GHCR
artifact. Historical native build records continue to describe their original
build image; they are not new qualification of the operator image. No image is
distributed by this source export.

After native preparation, use the prepared runtime controller's `preflight`,
`start` and `verify` workflow in [operations](OPERATIONS.md). Retain the normal
checkpoint, hardware, memory, fabric and qualification gates.

After editing the private environment file and staging the exact image, recipe,
models and sources on each host, run on the controller with traffic blocked:

```sh
cd RUNTIME_DIR/recipe
python3 -B scripts/fleetctl.py preflight --env-file ../operator.env --output ../preflight.json
python3 -B scripts/fleetctl.py start --env-file ../operator.env \
  --preflight ../preflight.json \
  --preflight-sha256 "$(sha256sum ../preflight.json | cut -d ' ' -f1)" \
  --manifest ../service.json --confirm START-JSPARK3
python3 -B scripts/fleetctl.py verify --env-file ../operator.env \
  --manifest ../service.json --output ../verify.json --log-output ../verify-rank0.log
```

Keep traffic blocked until the remaining qualification protocol passes.

### Notes

Reproducible OCI metadata, signed build attestations and portable cache/hygiene
tooling are follow-ups. This policy does not claim bit-identical image rebuilds
or new performance measurements.

The qualification protocol also requires user-level page-cache hygiene and
verified kernel-cache preparation. This package does not ship portable
`host-hygiene.py` or `kernel_cache.py` commands. Use independently reviewed
operator tooling and retain its receipts; the draft command names are not
installation commands for this checkpoint. On a new fleet with no cache
snapshot, complete warmup and gates with admission closed before taking a
snapshot. Bind every later seed to the same image, recipe and kernel sources.
