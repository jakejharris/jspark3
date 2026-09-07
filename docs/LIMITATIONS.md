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

## Kernel disable provenance and the 32,768-token single-stream boundary (v1.0.0)

Every benchmark and every measured run of this recipe executed with vLLM's
`persistent_topk` kernel disabled in the sparse-attention indexer, via a
one-line change the upstream launcher applies at container start. The v1.0.0
transform contract pinned the file with the kernel still enabled, so the
construction v1.0.0 shipped had never actually been run past 32,768 tokens
by anyone. v1.0.1 carries the disable in the transform itself: the transform's
emitted sparse-attention kernel file is byte-identical to the file every
measured arm executed. That is the verified target-file transform result;
full equivalence of the assembled public construction to the measured arms
was not exhaustively re-proven and remains the current recipe and review
task. No published number changed:
they all came from the disabled path. Found by a community bug report from
[@BTCXoomer on X](https://x.com/BTCXoomer).

Provenance correction, recorded at v1.1.0: the v1.0.0 release gate described
the public recipe as derived from the measured recipe by identifier renames
only. That was inaccurate for v1.0.0 — the measured sparse-attention kernel
file carried the start-time kernel disable and the v1.0.0 transform contract
did not. The
v1.0.0 release evidence and gate record are preserved unmodified as
historical documents; this note is the correction.

Mechanism, on a GB10 (48 SMs, 101,376 bytes of opt-in shared memory per
block, so the 128 KiB fallback can never apply):

| Decode shape (rows = batch x next_n) | Smem cap | Chunk | CTAs at max_model_len 1,000,000 | Fits 48 SMs? |
|---|---:|---:|---:|---|
| 1 row (speculation off) | 35,968 B | 8,472 | 119 | no, aborts |
| 8 rows (DFlash2 k=7, one stream) | 49,152 B | 11,768 | 85 | no, aborts |
| 16+ rows (two or more streams) | 101,376 B | 24,824 | 41 | yes |

The kernel aborts when the batch's actual longest sequence exceeds 32,768
tokens (the cooperative-launch threshold is on real sequence length) and the
CTA count, which is sized from the configured `max_model_len`, exceeds 48.
That is why only single-stream long context crashes: concurrency moves the
call into the 16+ row shape, and short contexts never arm the cooperative
path at all.

Boundaries implied by the arithmetic, for anyone hand-launching the v1.0.0
construction: `--max-model-len 406656` or lower keeps every shape at or
below 48 CTAs, and `564864` suffices only while speculation stays on. These
ceilings are arithmetic, not measurements; the persistent kernel at 41 to 48
CTAs was never run or benchmarked by this project, and at the boundary the
cooperative launch runs with zero headroom. The disable, not a ceiling, is
the supported answer.

The v1.0.0 verify witness could not have caught any of this: it sends one
warm-up and three short-prompt requests. A release gate for this recipe
needs a single-stream witness above 32,768 prompt tokens with real decode
steps.

Open item, not a defect shown here: upstream later reduced prefill chunks
from 8192 to 7168 citing the same oversubscription on the prefill side. Our
113,908-token prefill at 8192 chunks passed on the measured arm, so there is
no evidence of a prefill-side failure in this construction; recorded so the
next long-prefill investigation starts there.

## v1.1 (Cadence) limits

- **Narrow measured scope.** The paired gains are single-stream decode
  effects on fixed request sets. Batches of two or more requests and prefill
  fall back to the wide path by design, so a busy multi-stream service gets
  no promised gain from the width controller.
- **No replicated code gain.** The paired code effect was positive in the
  first start and spanned zero in the second; no universal code-speed gain is
  claimed.
- **Quality contains candidate-only losses.** In the fixed 62-answer quality
  battery, both candidate arms failed the Caesar-cipher coding task and one
  also failed FizzBuzz, while both reference arms completed both tasks.
  Population-level semantic parity remains inconclusive at every endpoint.
- **Concurrency burst is not capacity certification.** The short-prompt burst
  evidence (up to 24 concurrent streams) certifies neither sustained service,
  per-stream fairness, the 32-sequence envelope, nor capacity near the
  configured maximum context.
- **Long context past 32,768 tokens still awaits assembled-build live
  validation.** The transform-level kernel-file change is pinned and its
  output verified, but a live single-stream witness above 32,768 tokens on
  the assembled public build is a pending release-verification item. The
  configured 1,000,000-token context is a configuration value; it does not
  certify this candidate's operating envelope.
- **Not included.** Separate workspace experiments are not part of this
  construction, and their results do not transfer. The v1.1 candidate retains
  the stock indexer workspace.

## Measured regressions and misses (v1.0.0 evidence)

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
- **Most published reference numbers are context, not direct comparisons.**
  They come from their authors' own hardware and harnesses, with different
  node counts, quantization lanes, speculation, contexts, clocking, and
  estimators. The sparkDash comparison is narrower: JSpark3 ran the same
  pinned author protocol, but on a separate fleet and date, so it is not a
  same-day head-to-head.
- **The adapted current-Mia rapid screen has one battery.** Its original plan
  omits the broader publication-quality battery and marks rate-claim
  eligibility false. That historical flag remains. The maintainer approved
  publishing the 66.257 versus 44.562552 tok/s same-screen comparison with
  the single-battery limitation stated.
- **The agent reproductions are product evidence.** They share the same task
  and prompt but not a trajectory. The 44.583 versus 24.728 tok/s comparison
  reports achieved aggregate decode throughput for those two runs; it does
  not isolate an engine-only effect or predict identical speedups on another
  agent path.

## Evidence limits

- **The overlay A/B compares against an unreleased internal build.** The
  denominator is the matched three-Spark control (same recipe, overlay
  disabled). It was created during private development, was never released,
  and is not a market comparison; it is a causal control for one variable.
- **Single fleet, single operator.** All evidence comes from one three-Spark
  fleet operated by the project. No third-party or clean-room reproduction has
  been performed; that is an open release gate.
- **Sample sizes are small.** Three candidate batteries, three matched
  controls, one earlier control battery, one sparkDash block, one wave per
  concurrency level, one scored prefill prompt, one demonstration run per
  arm.
- **The internal-control demonstration is not a causal benchmark.** The two
  agent runs shared a prompt but not a trajectory. Their decode rates (47.377
  and 44.583 tok/s) describe those runs but do not isolate the overlay's
  effect.
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
