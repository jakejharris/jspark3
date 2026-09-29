# Cooperative MoE: reproducible builds and component qualification

The deterministic builder must reproduce the exact GPU-qualified native bytes
identified by the [release pin](../recipe/config/coop-release.json). The
[BUILD record](../recipe/overlays/v16/coop/BUILD.json) binds the component seal
and measured policy; the release notes display their filled-in identities.
It uses fixed `/w` paths, source timestamps, per-unit seeds and retained compiler
intermediates, with shared cudart. It never strips or normalizes completed ELFs.
The earlier build experiment remains separate evidence; repeating its old-builder
controls is not a release prerequisite. The release includes the completed
component seal. Default preparation selects coop on; `prepare_runtime.py --coop-off`
and `build_native.py --display-only` are explicit diagnostic opt-outs.

`tools/build_native.py` builds display and coop twice in fresh, network-disabled
containers. Both runs must agree and coop must hit the exact qualified hash.
It retains raw build directories and a receipt bound to the current source and
operator image. `--with-coop` is a compatibility alias. A failed coop build cannot
silently yield a display-only result; `--display-only` is diagnostic.

## Owner component workflow

Follow the tracked [operator handoff](COOP_QUALIFICATION_HANDOFF.md) for exact
source archives, writable paths, second-builder transfer, numeric memory guards
and timing allowances. `TARGET_SNAPSHOT` can be the existing serving model
directory; only read access is required. Native builders identify their integrated
GB10 by a read-only GPU UUID query from a checked host environment and fixed
root-owned executable. Builder identity is operator-attested: cloned machine-ids
are accepted; duplicate collected board identities are refused before check-only/campaign execution and at sealing.
Old native receipts must be rebuilt, not edited. Remote Docker builders and
emulated x86 builders cannot satisfy independent native qualification.

The workflow below is for maintainers qualifying a component before release.
Use a clean prepared candidate export or ordinary clone with no private working
files. Run `python3 -B tools/validate_release.py .` during that preparation;
`--require-final` rejects a candidate without integrated component, admission
and measurement evidence. Users installing published v1.8.4 follow
[installation](INSTALL.md), including its required `--require-final` check.

Stage these immutable inputs before reserving a GPU:

- An eligible image receipt and the complete `build_native.py` output, including
  both `coop-a` and `coop-b`. Independently repeat the build on a second physical
  Spark and retain its image/build receipts and machine identity evidence.
- The exact target checkpoint revision directory from
  `recipe/config/checkpoint-contract.json`. It must contain the original pinned
  index, original publication `SHA256SUMS`, and shards holding layer-3 weights
  for all 288 experts. The ledger must match the existing checkpoint contract;
  `recipe/config/coop-checkpoint.json` pins the exact required shard subset.
  The runner compares each complete shard with those trusted digests; merely
  recording an observed digest never authenticates a fixture. The runner
  mounts only this one snapshot read-only at the HF-cache path expected by the
  helper and checks all three EP ranges, full tensor shapes and exact dtypes.
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
  --sanitizer-root "$QUAL_SANITIZER" --output "$QUAL" --check-only \
  --independent-build-root "$SECOND_BUILD/native" \
  --independent-image-receipt "$SECOND_BUILD/operator-image.json"
```

Check-only writes `QUAL.check`, retains the complete container command plan,
and runs CPU-only stage-8 transforms, dependency imports, host load/linkage,
fixture-shape and sanitizer-option checks in a disposable container. It exposes
no GPU and starts no service. Its memory and total memory+swap limits are 4 GiB;
native compilation retains 8 GiB and exclusive GPU gates retain 16 GiB. Use a fresh output name for another check.

Only after the owner drains and stops the bound fleet and reserves one GB10,
first run the detector controls with the same operator image and pinned tool:

```sh
python3 -B tools/v16/check_sanitizer.py \
  --image-receipt "$BUILD/operator-image.json" \
  --sanitizer-root "$QUAL_SANITIZER" --output "$CAMPAIGN/sanitizer-controls"
```

The output directory must be new. Compilation hides all GPUs; the controls use
only GPU 0 in a capped, network-disabled maintenance container. The clean legacy
command must reproduce launch-record counting, the fixed memcheck and racecheck
commands must pass with candidate coverage, and a deliberate out-of-bounds CUDA
write must produce an invalid-global-write report and sanitizer exit 9. The
negative application deliberately returns zero, so its exit cannot fake detector
sensitivity. Any mismatch stops the control runner. `RESULT.json` binds the
image, tool package, driver/GPU fingerprint, source, runner and executable hashes.
Raw reports and saved records remain private. This is detector validation, not
candidate qualification. Use this same image, tool and GPU/driver for the campaign.

After these controls pass, repeat the qualification command without `--check-only`.
The runner serially performs:

- Nine full integration/H1 baselines plus nine bound perturbation controls.
- Nine complete profiles: 468 timed and numerically checked cases, preserving
  median/min/max; the source fixes five warmups and 25 samples per implementation.
- Eighteen smoke sanitizer runs and six geometry-2 runs, under memcheck and
  racecheck. The candidate-only filter is `kns=exl3_moe_coop_`. Retain all output;
  require clean summaries and nonzero unfiltered candidate launch evidence.
  The worker keeps `--error-exitcode 9` and explicitly sets `--print-level warn`:
  the pinned frontend otherwise counts informational launch records as errors.
  `--save` retains the instrumented execution; a CPU-only `--read` at info level
  recovers `--dump-kernel-launches` coverage from that same execution. Readback
  occurs only after the live tool exits zero and its summary is clean. The
  informational readback summary is not the error gate. Missing records, failed
  readback, absent candidate coverage or any live failure refuse sealing.
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
campaign outputs are preserved. Selection may change only the policy and its
manifest entry. Every policy gate binds its selected bundle, and the seal binds
both the original and selected identities. The runtime, headers, native bytes,
licenses and provenance must match trusted source pins; copied source and
builder files cannot supply their own authority. The checkpoint publication
authority and expected shard digests remain explicit in the seal.
The legacy shell helper's profile-only seal
operation is retired. A check-seal PASS is component evidence, not release
approval or fleet admission.

Pass 2 independently reviews the evidence, integrates the measured source and
bundle policies together, refreshes source/install pins and records both the
pre-policy qualification source identity and final source identity. Verify the
compiled-input map is unchanged, then rebuild twice from the final source and
repeat on the second physical machine. Only reviewed actual hashes enter
`coop-release.json`, the binary inventory and the distinct v1.8.4 delivery.
For an unintegrated candidate, default preparation refuses. A verified
component-qualified candidate permits private serving; final-release validation
also requires serving measurements and admission. The published release uses
the completed binding, while each operator still admits their own boot.

For v1.8.4 the final-source rebuild receipts are indexed in the
[final-source rebuild summary](../release/final-rebuilds-v1.8.4.json): two
bit-identical builds on a first machine and two on an independent second machine,
each accepted by `build_native.py` against the release source and producing the
pinned native. The builders are identified by GB10 GPU UUID; both share one
machine-id from a cloned OS image. The first machine's image record is the copy
installed in the locked prepared runtime of the measured release boot, bound by
payload hash to its native receipt; the original on-host file was not fetched.

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

## Builder evidence fields

The seal's `QUALIFICATION.json` contains `qualification_build.builder_host` and
`independent_build.native_receipt.builder_host`. Both are **operator-attested**
observations used for a trusted operator's two-board determinism check:

| Field | Meaning and authority |
| --- | --- |
| `architecture` | Locally observed architecture; independent qualification requires native `aarch64`. |
| `machine_id_sha256` | Hash of `/etc/machine-id`, retained as diagnostic metadata. Cloned OS identities may match. |
| `physical_identity.kind` | `gb10-gpu-uuid`: the observation concerns the single integrated GB10. |
| `physical_identity.uuid_sha256` | SHA-256 of the trimmed, lower-case GPU UUID collected on the host before and after building. The two collected values must differ. |
| Native receipt `payload_sha256` | Integrity check over canonical receipt contents. It is not a signature or independent authority for those contents. |

With honest collection, the record answers whether two different physical boards
compiled the same pinned bytes. It does not authenticate receipt edits, prove
execution against a malicious operator, or supply cryptographic hardware
attestation. Anyone consuming the artifact can independently reproduce and
compare the pinned native hash; editable builder identity is not that artifact's
integrity authority. GPU correctness and measured performance still require the
separate component gates.

Collection uses `/usr/bin/nvidia-smi`, validates root ownership and write
permissions along the executable path, and clears the query's inherited
environment. It refuses containers and non-host namespace/root views, retaining
before/after sampling. These checks prevent accidental collection mistakes; the
host operating system and operator remain trusted. The collector uses Linux's
[initial namespace identifiers](https://github.com/torvalds/linux/blob/v6.8/include/linux/proc_ns.h),
PID 1's root mount ID, and the initial network namespace's cookie, assigned first
by [Linux network namespace initialization](https://github.com/torvalds/linux/blob/v6.8/net/core/net_namespace.c).
Unsupported or unavailable host evidence refuses. No controller, SSH, signing or
lifecycle operation is added to the component runner.

## Source-only distribution

The original AGPL/MIT notices and source-offer obligations remain in place.
Shared cudart is not new authorization to distribute CUDA, native libraries or
combined images. Operators compile locally and obtain NVIDIA tools directly.
See [licensing](LICENSING.md) for retained obligations.
