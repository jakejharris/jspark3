# JSpark3 v1.8.3

Operators build their own image and display natives on a DGX Spark, prepare the
receipt-bound recipe, and qualify their own three-rank boot. Coop stays off;
no historical native binary or unpublished serving image is required.

This release adds readiness waiting, credential-redacted failure diagnostics,
page-cache hygiene and a same-boot admission producer. Prefix-cache gates default
to the runtime's fine-hit policy. Native TRIAR-inactive attestation checks the
actual Cadence graph banks. Installation includes exact dependency downloads,
runtime views, host requirements, first requests and stop/upgrade commands.

The experimental coop build runner now cleans up its exact build container on
cancellation. Coop reproducibility/GPU qualification and operator coop-on support
remain separate work; the historical coop-on gate is unchanged.

One v1.8.2 clean-room fleet passed verify, post-hygiene prefill and fine-hit APC,
with single-stream code decode at 49.05 tok/s median (three repetitions, 512
tokens, temperature 0, coop off). It did not run full admission. This is a single
observation, not a qualification of every installation or a controlled coop comparison.

The [frozen v1.8.0 results](results-v1.8.0.json) and [numbers](RELEASE-NUMBERS.md)
are unchanged. Their headline uses coop on. Historical notes remain
[archived](RELEASE-NOTES-v1.8.0.md); follow the current [installation](../docs/INSTALL.md),
[operations](../docs/OPERATIONS.md), [benchmarks](../docs/BENCHMARKS.md) and
[licensing](../docs/LICENSING.md). EXL3/TR3 attribution remains in
[REQUIRED_ATTRIBUTION](../REQUIRED_ATTRIBUTION.md).
