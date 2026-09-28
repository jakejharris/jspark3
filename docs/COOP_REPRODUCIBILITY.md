# Cooperative MoE: reproducible builds and component qualification

The unchanged deterministic builder targets
`3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`.
It uses fixed `/w` paths, source timestamps, per-unit seeds and retained compiler
intermediates, with shared cudart. It never strips or normalizes completed ELFs.
The earlier build experiment remains separate evidence; repeating its old-builder
controls is not a release prerequisite. New GPU qualification remains pending.

`tools/build_native.py` builds display and coop twice in fresh, network-disabled
containers. Both runs must agree and coop must hit the exact candidate hash.
It retains raw build directories and a receipt bound to the current source and
operator image. `--with-coop` is a compatibility alias. A failed coop build cannot
silently yield a display-only result; `--display-only` is diagnostic.

## Owner component workflow

Use a clean candidate source export or ordinary clone with no private working
files. Run `python3 -B tools/validate_release.py .` for this pending candidate;
`--require-final` must refuse until all component, admission and measurement pins
are integrated. Build the operator image and native outputs as documented in
[installation](INSTALL.md), omitting only that premature final-release assertion.

Stage these immutable inputs before reserving a GPU:

- An eligible image receipt and the complete `build_native.py` output, including
  both `coop-a` and `coop-b`. Independently repeat the build on a second physical
  Spark and retain its image/build receipts and machine identity evidence.
- The exact target checkpoint revision directory from
  `recipe/config/checkpoint-contract.json`. It must contain the original pinned
  index and the shards holding layer-3 weights for all 288 experts. The runner
  mounts only this one snapshot read-only at the HF-cache path expected by the
  helper, hashes the shards, and checks all three EP ranges and tensor shapes.
- Fly sources at the commit in `recipe/config/patch-contract.json`.
- `test_exl3_overlay.py` and `LICENSE` from the exact repository/revision/file
  hashes in `recipe/config/coop-helper.json`. Dependencies come from the verified
  image: torch, vLLM, exllamav3_ext and safetensors. The helper is AGPL-3.0-only,
  downloaded directly by the operator; no private helper path is required.
- NVIDIA's exact package in `recipe/config/coop-sanitizer.json`. Verify its
  SHA-256, extract it with `dpkg-deb -x`, and point `--sanitizer-root` at the listed
  directory. The runner verifies its executable/support-file hashes and mounts
  them read-only. No sanitizer package or NVIDIA binary is distributed here.

All variables below are absolute paths, with `QUAL` a new directory outside the
source checkout. Run check-only while the existing fleet remains serving:

```sh
python3 -B tools/v16/qualify_coop.py \
  --image-receipt "$BUILD/operator-image.json" --build-root "$BUILD/native" \
  --model-root "$QUAL_MODEL" --fly-root "$QUAL_FLY" --helpers-root "$QUAL_HELPERS" \
  --sanitizer-root "$QUAL_SANITIZER" --output "$QUAL" --check-only
```

Check-only writes `QUAL.check`, retains the complete container command plan,
and runs CPU-only stage-8 transforms, dependency imports, host load/linkage,
fixture-shape and sanitizer-option checks in a disposable container. It exposes
no GPU and starts no service. Use a fresh output name for another check.

Only after the owner drains and stops the bound fleet and reserves one GB10,
repeat the same command without `--check-only`. The runner serially performs:

- Nine full integration/H1 baselines plus nine bound perturbation controls.
- Nine complete profiles: 468 timed and numerically checked cases, preserving
  median/min/max; the source fixes five warmups and 25 samples per implementation.
- Eighteen smoke sanitizer runs and six geometry-2 runs, under memcheck and
  racecheck. The candidate-only filter is `kns=exl3_moe_coop_`. Retain all output;
  require clean summaries and nonzero unfiltered candidate launch evidence from
  `--dump-kernel-launches`. Unknown formats or unexplained errors refuse sealing.
  See [NVIDIA's option documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
- Policy selection on a separate bundle and three production-policy GPU checks
  without qualification or geometry overrides. Every row 1–64 is explicit;
  unmeasured/nonwinning rows stay stock and the worst-case ratio must be ≤0.97.

Each test starts from exact base transforms in a fresh maintenance container.
No live serving container is copied. Container commands, timestamps, actual GPU/
driver/toolchain, stage-8 identities, image/source/helper/checkpoint identities,
proof receipts and logs are retained. Cancellation removes only the exact
container created by that command, using the tested experiment cleanup routine.
The runner has no fleet controller or endpoint mutation operation.

## Seal, review, then integrate

```sh
python3 -B tools/v16/qualify_coop.py --seal "$QUAL" --output "$SEALED_OUTPUT" \
  --independent-build-root "$SECOND_BUILD/native" \
  --independent-image-receipt "$SECOND_BUILD/operator-image.json"
python3 -B tools/v16/qualify_coop.py --check-seal "$SEALED_OUTPUT"
```

The seal operation revalidates every gate and profile and regenerates the policy
from the raw logs before writing new BUILD/bundle/index files. Raw builds and
campaign outputs are preserved. The legacy shell helper's profile-only seal
operation is retired. A check-seal PASS is component evidence, not release
approval or fleet admission.

Pass 2 independently reviews the evidence, integrates the measured source and
bundle policies together, refreshes source/install pins and records both the
pre-policy qualification source identity and final source identity. Verify the
compiled-input map is unchanged, then rebuild twice from the final source and
repeat on the second physical machine. Only reviewed actual hashes enter
`coop-release.json`, the binary inventory and the distinct v1.8.4 delivery.
Until then default preparation and final-release validation refuse.

The schema records the actual qualification image separately from operator image
eligibility. Every operator still passes the fixed Dockerfile/InstantTensor policy
and local image inspection. Independent eligible config digests may differ;
self-hashed image/native receipts never grant GPU qualification. Historical
schema-1 records retain their strict historical image check.

Run the public clone-based coop-on stock installation/admission and compare the
prepared archive path. Keep `hardware_qualified=false` on local build receipts.
The stock-only producer binds the seal/native/policy and actual operator image
through first and final evidence; ABLIT=1 needs a separate owner admission.
Record new measurements in [the v1.8.4 record](../release/results-v1.8.4.json).
Historical throughput and quality do not transfer to the new native bytes.

## Source-only distribution

The original AGPL/MIT notices and source-offer obligations remain in place.
Shared cudart is not new authorization to distribute CUDA, native libraries or
combined images. Operators compile locally and obtain NVIDIA tools directly.
See [licensing](LICENSING.md) for retained obligations.
