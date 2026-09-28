# JSpark3 v1.8.3

**Build on your own Sparks. Check your own boot before admitting traffic.**
v1.8.3 carries operators from source to a receipt-bound three-rank runtime:
build the serving image and display natives, prepare the recipe, then qualify
the running fleet. No unpublished image or historical native binary is needed
for the default coop-off path.

- **Display builds that repeat.** The native fix in
  [34fa931](https://github.com/jakejharris/jspark3/commit/34fa931668d82861b7b24dd13f1d35deae8ab05b)
  removes the nvcc PID-dependent `tmpxft` filename from `display_kv_probe` by
  using stable compiler intermediates and distinct translation-unit seeds.
  Both validation DGX Sparks produced byte-identical probe and library outputs.
  `build_native.py` still requires two fresh builds to match, without stripping
  or rewriting finished binaries. A mismatch refuses a receipt and retains
  both builds privately for diagnosis. See the [build recipe](../recipe/overlays/v14/display_kv/build_repro.sh).
- **Shared evidence contains structure, not raw diagnostics.** Shared JSON,
  logs and console output carry fixed labels, validated facts and hashes.
  Arbitrary tool output and exceptions stay in private mode-0600 sidecars.
  The enforced shared-output audit makes release validation refuse new or
  changed executable files until their output paths are reviewed. Share the
  qualification receipts with private sidecars excluded; the
  [output policy](../docs/SHARED_OUTPUT.md) spells out that boundary.
- **Qualification follows the actual boot.** Readiness waiting, warmup,
  page-cache hygiene, verification and admission now form one documented flow.
  Prefix-cache gates use the runtime's fine-hit policy. Native attestation
  checks that TRIAR is inactive in the actual Cadence graph banks. Post-hygiene
  canaries, prefill, correctness, memory/no-swap and zero compilation after
  warmup remain blocking. Source validation is not hardware admission:
  `hardware_qualified=false` stays in the source, and your receipts apply to
  your exact boot.
- **Installation reaches the first request.** The guide covers dependency
  downloads, derived runtime views, host requirements, start, stop and upgrade.
  Normal clones validate the checked-out files, including untracked files,
  without scanning unrelated Git history. Exported-file privacy checks remain
  enforced. Start with [installation](../docs/INSTALL.md) and
  [operations](../docs/OPERATIONS.md).

**Coop stays off for operator builds** (`JSPARK3_V16_COOP=0`). The experimental
coop builder now cleans up its exact container on interrupt, termination or SSH
hangup. Reproducible bytes alone do not qualify a coop kernel for serving;
operator coop-on support remains future work and the historical hardware seal
is unchanged.

**The performance record stays frozen.** The v1.8.0 build measured
**136.9 to 141.7 tok/s code decode at 4 streams and 174.0 to 179.7 tok/s at 8**,
with stock weights and **coop on**. Those are aggregate rates across two sweeps
on one serving start, with each stream forced to 512 output tokens at
temperature 0, thinking off. They are not new v1.8.3 measurements or a prediction
for coop-off builds. The [frozen results](results.json),
[numbers](RELEASE-NUMBERS.md) and [benchmark definitions](../docs/BENCHMARKS.md)
are unchanged.

A separate [v1.8.2 clean-room observation](operator-cleanroom-v1.8.2.json)
recorded **49.05 tok/s median single-stream code decode**, with coop off:
three repetitions, 512 output tokens, temperature 0. That fleet did not run
full admission. This is one observation, not a controlled coop comparison or
a qualification of other installations.

Recipe code is Apache-2.0; the running stack also includes AGPL-3.0-only
components and the DFlash2 draft model, licensed for non-commercial research
and evaluation only. Read [licensing](../docs/LICENSING.md).
EXL3/TR3 attribution remains in [REQUIRED_ATTRIBUTION](../REQUIRED_ATTRIBUTION.md).
The original v1.8.0 release notes remain [archived](RELEASE-NOTES-v1.8.0.md).
