# Cooperative MoE build experiment

This is a candidate build recipe for future coop-on qualification. The operator
default remains coop off. A matching build hash does not confer GPU qualification
or permit bypassing the existing historical seal.

## Build change and evidence boundary

The old recipe embedded PID-derived `tmpxft` host-stub filenames in ELF FILE
symbols. Serial compilation in a fresh container did not guarantee stable PIDs.
The new recipe uses nvcc `--keep --keep-dir` with separate, stable directories and
random seeds for rows32, rows64 and dispatch. Source mtimes and `/w` paths remain
fixed. Intermediates and verbose compiler commands are retained for diagnosis;
finished binaries are neither stripped nor normalized to force a match.
These are supported [nvcc options](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-compiler-driver-nvcc/index.html#keeping-intermediate-phase-files).

Linking now explicitly uses `--cudart=shared`. The verified operator image has
`libcudart.so.13`; the experiment checks the ELF dependency, resolution with
`ldd`, and host library loading without exposing GPUs. GPU behavior still needs
the qualification below. Two emulated ARM64 full builds matched with deliberately
different preceding process counts; native ARM64 confirmation remains required.

## One native Spark round

The hardware owner runs this on their own ARM64 Spark, from a clean source
export, after choosing the build window. Python 3 and Docker are required on the
host. Compilers, readelf and cuobjdump come from the verified image.

```sh
python3 -B tools/build_operator_image.py --output ../coop-image.json
python3 -B tools/v16/experiment_coop_build.py \
  --image-receipt ../coop-image.json --output ../coop-experiment --runs 3
```

For a before/after control in the same round, add
`--baseline-builder ../build_repro-7977795.sh`, a saved copy of that revision's
build script. This runs three baseline containers as well as three candidate
containers; it does not modify the baseline script or sources. The report records
both builder hashes. A baseline failure does not suppress candidate diagnostics.

Every container is fresh, network-disabled, runs as the invoking UID without GPU exposure,
and capped at four CPUs and 8 GiB RAM with no extra swap allowance. All builds
mount their separate host directories at `/w`. The runs deliberately start after
0, 37 and 74 extra processes to challenge the previous PID assumption. No serving
container is started, stopped or edited. The experiment retains failed builds.

Ctrl-C (SIGINT), SIGTERM or a session hangup (SIGHUP) cancels the experiment and force-removes its current
container by the exact ID retained in that run's `container.cid`. Creation
finishes before cancellation cleanup, without starting the compile; repeated
signals cannot interrupt cleanup. The runner exits 130, 143 or 129, respectively, after removal.
The bind-mounted build directory, partial `console.log` and `cleanup.log` survive.
Completed and failed runs also remove their containers before the next run starts.

Keep the complete output directory. `experiment.json` records image/source/build
identity, host and Docker architecture, return codes, output hashes, linkage and
comparisons. A PASS requires three identical candidate ELFs, successful host
loads, resolved shared cudart, and no PID-derived FILE symbols or the checked
defined cudart runtime entry points. A compile, comparison or linkage failure
returns exit 9. The optional baseline result is diagnostic only.

`candidate-1-vs-2.json` and `candidate-1-vs-3.json` locate differing file offsets,
ELF sections and symbols (including symbol-content hashes). CUDA ELF dumps decode
fatbin line tables; corresponding `.cuda-elf.diff` files show their differences.
Each run also retains `console.log`, `out/build32.log`, `out/build64.log`,
`out/link.log`, `out/keep/`, `out/elf.txt`, `out/ldd.txt` and `out/load.txt`.
For any two retained binaries, the standalone diagnostic needs only Python:

```sh
python3 -B tools/v16/diff_native_elf.py FIRST.so SECOND.so --output difference.json
```

Its exit code is 0 for identical bytes, 1 for differences, and 2 for invalid input.
It reads the ELFs without executing them. Report examples are bounded; total
counts and full section hashes remain available.

## After native reproducibility passes

1. Record the common binary hash and complete experiment evidence. Independently
   rebuild with the same public image recipe on another Spark and require the
   same hash. Keep the raw build bundles unchanged; profile a separate copy.
2. In an exclusively reserved GPU maintenance environment, use the exact new
   binary and verified operator image with the pinned stage-8 EXL3/fatpath and
   real checkpoint/helper fixtures required by `test_cuda_integration.py`.
   The unpatched base image alone is insufficient. For every EP rank 0, 1, 2 and
   geometry 0, 1, 2, set `GLM53_COOP_MAINTENANCE_TEST=1`,
   `GLM53_COOP_QUALIFICATION=1`, `GLM53_COOP_EP_RANK`, `GLM53_COOP_GEOMETRY`,
   `GLM53_COOP_BUNDLE` and `GLM53_COOP_TEST_HELPERS`. Run
   `test_cuda_integration.py` and `profile_shapes.py` from the pinned coop source.
   Save exactly nine `rankN-geoN.jsonl` profile logs. Require all 468 numerical
   and timing cases, graph/eager parity and allocation checks. Run the candidate
   memcheck/racecheck gates, including `test_geometry2_sanitizer.py`, plus the
   existing `coop_h1_control.py` baseline/perturb sensitivity control. The
   [source instructions](../recipe/overlays/v16/coop/source/README.md) describe
   the maintenance fixtures and candidate-focused sanitizer scope.
3. Run `source/select_policy.py --bundle QUALIFIED_COPY` with the nine logs;
   then `source/manifest.py verify-artifacts QUALIFIED_COPY`. Run
   `test_policy_gpu.py` for all EP ranks without the qualification override.
   Slower or unmeasured rows remain stock. None of the historical profile logs
   may be reused for the new binary.
4. Seal the new binary, selected row policy, manifest, source identity, verified
   build image and nine profile hashes into a reviewed new BUILD record. The
   existing `tools/v16/build_coop_moe.sh` provides the record schema and checks,
   but its build-image override is **not runtime enablement**: runtime currently
   requires the historical image in `apply_coop_moe.verify_bundle`. A follow-up
   release change must pin the newly qualified build-image identity and artifacts
   explicitly, update binary/recipe/source inventories and test this runtime gate.
   An operator receipt alone must never grant hardware qualification.
5. Only after that seal and the normal full-model, fabric, memory, no-swap,
   correctness and fresh stock-production admission gates pass, make coop the
   prepared operator default again and require outsider builds to equal the new
   qualified pin. Add a clean-install coop-on test and a changed-byte refusal.
   Frozen v1.8.0 results remain unchanged; performance claims for this candidate
   require new measurements.

## Licensing with shared cudart

The source export still carries AGPL-3.0-only coop derivatives and MIT headers,
with their original notices and attribution. Shared cudart removes the static
host CUDA runtime from the candidate ELF; it does not make the whole serving
stack wholly AGPL or clear binary redistribution. The CUDA dependency retains
its [NVIDIA terms](https://docs.nvidia.com/cuda/archive/13.0.1/eula/index.html).
Dynamic linking alone does not resolve copyleft compatibility; the
[GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#GPLStaticVsDynamic)
treats static and dynamic linking alike for that question. A compatible permission
or applicable exception still needs to be established before distributing a
combined binary. This change distributes source only, with operators compiling
locally; no CUDA library, coop ELF or image is included. The existing AGPL network
source-offer, InstantTensor, DFlash2 and attribution obligations in
[licensing](LICENSING.md) remain in force.
