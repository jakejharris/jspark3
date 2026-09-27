# Opt-in cooperative MoE for TP3 / EP

This adaptation of the existing TP2 extension targets GB10 SM121a, K4 MCG,
H=4096, local intermediate=2048, top-k=8, up to 96 local experts, and
SwiGLU limit 10. Native ABI 2 builds separate 32-row and 64-row translation
units (256/512 routed slots). Nine-bit slot ordering avoids collisions above
32 rows. Existing TP2 code and defaults are unchanged.

The adapter shares scratch per device across layers. Calls must be ordered,
nonoverlapping, and single-GPU per rank; ubatching/dual-batch overlap are
rejected. Returned output aliases staging memory. Invalid inputs in the
supported envelope fail; unsupported shapes and activation limits use stock
EXL3. E3 grouped prefill remains stock. There is no retry after native launch.

## Build

Run inside the compatible CUDA 13 ARM64 environment with the existing
runtime dependencies. This compiles code only; it downloads nothing and does
not start or stop engines:

```sh
bash extensions/cooperative_moe/tp3/build.sh /tmp/coop-tp3-build
```

Use a fresh, empty output directory. All nine pinned ExLlamaV3 headers are
vendored with their original license and verified hashes. Keep every file
listed in `manifest.json`; runtime verifies source, adapter, policy and binary
hashes before loading ABI 2. Never substitute the TP2 binary.

The included dispatch table records the historical measured artifact. A new
build normally has a new binary hash and is deliberately refused for serving
until reprofiled. Do not bypass that check or blindly repin the old table.

## Validation and policy

CPU checks (contract tests need PyTorch but no GPU):

```sh
python3 extensions/cooperative_moe/tp3/test_dispatch.py
python3 extensions/cooperative_moe/tp3/test_contract.py
python3 -O extensions/cooperative_moe/tp3/test_contract.py
```

GPU checks require an explicitly reserved GPU, the compatible runtime,
existing checkpoint cache, and upstream `tests/test_exl3_overlay.py` helpers.
Set `GLM53_COOP_BUNDLE` to the build, `GLM53_COOP_TEST_HELPERS` to the tests
directory, and `GLM53_COOP_MAINTENANCE_TEST=1`. For a new native binary, use
`GLM53_COOP_QUALIFICATION=1` and `GLM53_COOP_GEOMETRY=0`, `1`, or `2` only
inside the maintenance process. Run `test_cuda_integration.py` and
`profile_shapes.py` for EP ranks 0, 1, 2 (`GLM53_COOP_EP_RANK`). Profile stdout
is JSONL. Qualify numerical parity, graph replay, no eager allocation and
candidate memcheck/racecheck before serving. `sanitizer_smoke.py` and
`test_geometry2_sanitizer.py` provide candidate-focused checks; they do not
certify the stock kernel as race-free. Do not run these alongside production.

`select_policy.py` reduces the nine complete rank/geometry profile logs into
an explicitly selected build's policy; it requires every numerical comparison
and profile case. Then run `test_policy_gpu.py` (without qualification override)
on all three EP ranges. Unmeasured/slower shapes stay stock. This profiling
receipt is not full-model quality qualification.

The historical policy selects geometry 2 at rows 2,3,4,5,10,18,24; geometry 1
at 1,6,7,8,9,12,15,16,20,25,30,32,35,40,48,56; stock otherwise. In particular,
64 rows are supported but did not meet the conservative 3% worst-case gain
threshold and therefore use stock. Never describe every 64-row call as faster.

See `docs/tp3-throughput-results.md` for historical measurements and the
separate validation status of this public branch. `PROVENANCE.json` records
Mia's source and ExLlamaV3 lineage; original license notices are preserved.
