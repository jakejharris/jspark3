# Limitations

What JSpark3 v1 does not do, does not prove, or does worse. Read this before
the benchmarks, not after.

## Scope

- **Not a model.** JSpark3 v1 trains nothing and quantizes nothing. It serves
  the pinned upstream checkpoints. Quality is the quality of those checkpoints
  under this runtime; no public accuracy benchmark was run for this release.
- **Exactly three DGX Sparks.** No two-node, four-node, or mixed-hardware
  variant. The recipe refuses any other count.
- **Pinned everything.** A different checkpoint revision, draft, image
  digest, or vLLM build is refused, not adapted. That is the point, and it
  also means upstream improvements need a new release here.
- **One serving envelope.** The 1,000,000-token configured context, 32
  sequences, 8,192 batched tokens, and graph sizes are fixed by the profile
  and hash-bound. Tuning them is a new, unmeasured configuration.

## Measured regressions and misses

- **Long prefill is slower.** The 113,908-token matched prefill proxy fell
  from 1277.443 to 1234.246 tok/s (-3.38%) and time to first token rose from
  89.169 s to 92.290 s (+3.50%).
- **Three-stream waves were variable.** The C3 per-stream median was 69.634,
  72.421, and 51.382 tok/s across the three candidate batteries; in the strict
  same-day pairing it lost to the control, 65.208 to 51.382 tok/s (-21.20%).
- **Fairness under concurrency did not improve.** Slowest-over-fastest stream
  ratios were 0.303, 0.188, and 0.148 at C12, C24, and C48.
- **Time to first token at 48 streams is long.** The p90 was 96.722 s in the
  C48 wave even though aggregate throughput rose. Interactive workloads need
  an admission layer or a lower concurrency cap.
- **Internal promotion gates were missed.** The campaign code median of
  66.257 tok/s fell short of the project's 67.0 tok/s floor by 0.743 tok/s
  (1.11%), and the agent demonstration's longest uncompensated interior slow
  run was 14 against a limit below 5. The single-stream batteries themselves
  passed all pacing gates. Neither miss is a correctness or stability failure.
- **The older C6 comparison overstates.** Against the earlier control
  battery the C6 per-stream median rose +46.01%, but the same-day matched
  controls put the credible C6 delta at +2.91%.

## Comparison limits

- **No literal reproduction of any published recipe exists here.** The three
  local reproductions are adapted: `mia-tp2-historical-0e2e78f` is
  site/safety-adapted, `mia-tp2-current-c190db1a-adapted` is
  compatibility-adapted and owes its runnability to the single
  `GLM53_INDEXER_WORKSPACE=rightsize` repair, and
  `fly-derived-9093765c-adapted` is minimal-correctness/safety-adapted rather
  than the published launcher. None of them may be described as exact.
- **The exact current-Mia attempt produced no number.** It did not reach HTTP
  under this safety envelope and sent no request, so no throughput, latency,
  prefill, quality, or agent figure is attributable to it.
- **No FlyCockpit run and no jetnet run.** There is no literal FlyCockpit
  reproduction on this fleet, and jetnet was never run here at all; it was
  read and studied statically. Its numbers appear only as author-reported
  context.
- **Published reference numbers are not comparable to ours.** They come from
  their authors' own hardware and harnesses, with different node counts,
  quantization lanes, speculation, contexts, clocking, and estimators, which
  is why this release computes no percentage, delta, or ranking against any
  of them.
- **The adapted current-Mia rapid screen is not a rate claim.** Its own plan
  omits the broader quality battery and marks rate-claim eligibility false; it
  appears only in the technical report, labelled as such.
- **The agent reproductions are product evidence.** They share a prompt but
  not a trajectory, so their aggregate decode rates describe individual runs
  and must not be read as a ranking between recipes.

## Evidence limits

- **The overlay A/B compares against an unreleased internal build.** The
  denominator is the matched three-Spark control (same recipe, overlay
  disabled). It was created during private development, was never released,
  and is not a market comparison; it is a causal control for one variable.
- **Single fleet, single operator.** All evidence comes from one three-Spark
  fleet operated by the project. No third-party or clean-room reproduction has
  been performed; that is an open release gate.
- **Sample sizes are small.** Three candidate batteries, three matched
  controls, one earlier control battery, one wave per concurrency level, one
  scored prefill prompt, one demonstration run per arm.
- **The demonstration is not a benchmark.** The two agent runs shared a
  prompt but not a trajectory. Their decode rates (47.377 and 44.583 tok/s)
  must not be read as a ranking.
- **Semantic correctness of scheduler-wave outputs was not evaluated.** The
  wave harness checked HTTP status, stream completeness, and non-empty
  output only.
- **Rates are not guaranteed.** Temperature, clocks, firmware, fabric
  placement, cache state, and scheduler state all move them. The recipe
  reproduces the construction, not the numbers.

## Operational limits

- Privileged containers (host network, host IPC, `/dev/infiniband`,
  `IPC_LOCK`), 64 GiB pinned host memory per rank with swap disabled.
- No authentication or TLS on the endpoint.
- No JSpark3 GHCR image is published for v1.0.0. The release pins and launches
  the exact upstream image by digest. `docker/` is retained only for local
  reproducibility; do not redistribute a local build without independently
  satisfying NVIDIA and upstream terms.

## Legal limits

- The target checkpoint is attribution-required (ShapleyMcg License v1.0) and
  the DFlash2 draft is CC BY-NC-ND 4.0 for research and evaluation use with
  commercial use requiring permission from Inco AI. The assembled endpoint is
  neither unrestricted open source nor commercial-ready. The container license
  audit is a NO-GO for publishing the prepared derivative image in v1.0.0. See
  [LICENSING.md](LICENSING.md).
