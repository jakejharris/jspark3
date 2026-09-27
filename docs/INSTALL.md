# Installation

Validate the source export before creating a separate private runtime copy.
The export has no native binaries, model weights, container layers or donor
weights. Hardware admission remains unqualified. A clean installation from these source
files alone has not been demonstrated; see the image prerequisite below.

1. Run `sha256sum -c SHA256SUMS` and the validator in the root README.
2. Obtain the exact model, image and source dependencies identified in
   `recipe/config/`. The local image recipe is `docker/stock-v13/Dockerfile`;
   its InstantTensor hash manifest accompanies it. Image/module redistribution
   is outside this source package's license posture.
3. Build the native artifacts in a separate working copy using the pinned
   image/toolchain. `tools/v14/build_display_kv.sh HOST --check` builds the
   display library and probe twice and compares their expected record.
   `tools/v16/build_coop_moe.sh HOST build BUILD_ID` builds the cooperative
   kernel twice. Its native build entry is
   `recipe/overlays/v16/coop/build_repro.sh`. A selected layout may require an
   additional native build listed in `manifests/binaries.json`. Collect all outputs
   under their exact paths in `manifests/binaries.json` beneath BINARY_ROOT.
   Historical build records specify expected outputs; this export did not
   rebuild or requalify them. A mismatch requires a reviewed source/build
   update and fresh qualification, never an automatic acceptance of new hashes.
4. Run `python3 -B tools/prepare_runtime.py --binary-root BINARY_ROOT --output RUNTIME_DIR`.
   It verifies the source export and all local binary pins, runs native
   artifact checks, copies `recipe/` and writes its runtime checksum inventory.
   Keep its runtime-build receipt. The source export stays unchanged.
5. Copy the runtime recipe's `.env.example` to a private environment file.
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

The Dockerfile copies InstantTensor from a pinned upstream image and verifies
its file hashes. It does not establish reproducible image-config or layer
identity. The runtime still requires the exact image identity recorded in
`recipe/config/image-oci.json`; a locally built image with different metadata
will be refused even if the InstantTensor files match. Do not replace those
pins merely to pass preflight. A reproducible image build or a separately
reviewed identity policy is required before this can be presented as a complete
outside-operator installation. No image is distributed by this source export.

The qualification protocol also requires user-level page-cache hygiene and
verified kernel-cache preparation. This package does not ship portable
`host-hygiene.py` or `kernel_cache.py` commands. Use independently reviewed
operator tooling and retain its receipts; the draft command names are not
installation commands for this checkpoint. On a new fleet with no cache
snapshot, complete warmup and gates with admission closed before taking a
snapshot. Bind every later seed to the same image, recipe and kernel sources.
