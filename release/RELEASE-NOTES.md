# JSpark3 v1.8.0

Up to 141.7 tok/s code and 90.7 tok/s prose decode at 4 streams on three DGX Sparks, stock weights.

Best of two runs. Full ranges in the results below.


Faster than the published three-Spark reference build on its own benchmark at one and two streams, in every run.

These are protocol-matched stock-weight prose medians from separate fleets and dates. The reference is author-reported, with unknown repetition count and exact harness commit. Individual runs can overlap the reference. No cross-protocol speedup is implied.

| Streams | Our median tok/s | Author-reported tok/s | Source |
|---|---|---|---|
| 1 | 47.57 | 40.1 | [reference](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md) |
| 2 | 64.26 | 56.6 | [reference](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md) |
| 3 | 78.09 | 75.5 | [reference](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md) |
| 4 | 86.44 | 88.4 | [reference](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md) |

At four streams, our median is 86.44 tok/s versus the author-reported 88.4 tok/s.

Four-stream decode, 512 forced tokens, concurrent streams. Full two-run within-start ranges (not confidence intervals):

| Workload | Streams | tok/s range |
|---|---|---|
| code | 4 | 136.9 to 141.7 |
| prose | 4 | 83.3 to 90.7 |

Changes since v1.1.0 include finer prefix reuse and retained replay windows, decode-floor scheduling and aligned prefill chunks, compact drafter KV, optional display-reserve KV, batched/cooperative MoE paths, adaptive draft width, optional dense-trunk FP8, the stock production profile, and grammar/termination fixes. These are source changes; the tables do not assign a speedup to an unmeasured feature.

Versions v1.2 through v1.7.3 were internal iterations.

Thanks to @unsaltedbutter-ai for the contribution in [PR #9](https://github.com/jakejharris/jspark3/pull/9), superseded by the integrated recipe. Upstream credits and license notices remain in the source tree.

Stock weights (`ABLIT=0`) and `production-stock` are the defaults. Choose stock or edited behavior before launch. Changing modes requires a service restart and recomputes conversation prefixes. Edited weights remain an explicit opt-in supplied separately; no edited checkpoint is redistributed.

The historical measured stock cohort used a validation profile for testing. It does not qualify production admission. This sanitized source distribution retains `hardware_qualified=false`; fresh qualification of the exact shipped configuration is required before production mode-0 admission. Measurements describe the separately identified serving configuration, not a hardware test of this source export.

Thirds and active TRIAR are deferred to the next release. No TRIAR speedup is claimed. The selected source retains TRIAR code resident but inactive under the NCCL graph path. Do not enable it; the startup resident flag is distinct from the runtime OFF state. The full positive inactive-path attestation and native qualification are required.

First-pass canaries are recorded only; quick runs are diagnostic. Post-hygiene canaries block admission, together with the 1100 tok/s prefill floor, correctness, memory/no-swap and zero post-readiness compilation requirements.

Stock free-form JSON may arrive in a Markdown fence, and code may be formatted with backticks. Clients needing bare JSON should request structured output with `response_format` and `json_schema`. Grammar-constrained requests retain the full speculative width. No quality-equivalence claim is made between stock and edited weights.

The single-request grammar-width crash is fixed. Its applicability to v1.1.0 is established by source inspection and reproduction on a later build, not a v1.1.0 hardware reproduction. Greedy near-ties still require controlled parity checks. Prefixes beyond retained replay boundaries may safely recompute.

Native binaries, weights and images are obtained separately. The exact pinned image, native builds, cache preparation and operator hygiene remain installation prerequisites. A fully independent image rebuild has not been demonstrated by this export.

Original code and prose are Apache-2.0. Included derivatives retain AGPL-3.0-only and vendored headers retain MIT. The assembled service is not wholly Apache-2.0; per-file SPDX and REUSE records govern. The draft-model dependency retains its non-commercial research/evaluation restriction.

# Benchmark records

Decode is the aggregate rate of 1, 2, 4 or 8 code-writing requests started together (this build serves the Pi coding agent), each forced to 512 output tokens at temperature 0 with thinking off and no prefix-cache reuse, counted from the first to the last streamed token; prefill is the range over eight Pi-shaped coding-agent turns, each extending a cached prefix, measured right after a page-cache hygiene step.

Decode ladder figures are the recorded values to 0.1 tok/s (round half to even); the recorded values are in lo_exact and hi_exact. Every other figure is copied exactly as recorded.

## Release, stock weights, production profile

Build `v1.8.0`; weight mode `0`; configuration `none`. Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1195.0 to 1262.6 |
| `decode_c1` | 68.2 to 73.3 |
| `decode_c2` | 101.0 to 103.9 |
| `decode_c4` | 136.9 to 141.7 |
| `decode_c8` | 174.0 to 179.7 |
| `decode_prose_c1` | 44.3 to 49.1 |
| `decode_prose_c2` | 61.6 to 63.5 |
| `decode_prose_c4` | 83.3 to 90.7 |
| `decode_prose_c8` | 110.1 to 110.8 |

Frozen for release. One production-stock serving start with two within-start sweeps. TRIAR is inactive; neither TRIAR nor thirds performance is claimed. Both sweeps and their ranges are retained; these are not confidence intervals. Structured workloads show substantial within-start variance. Quality observer capture was repaired and both quality panels passed on the same unchanged serving start.

## v1.7.4 base recipe, stock weights, validation profile

Build `v1.7.4`; weight mode `0`; configuration `none`. Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1196.7 to 1263.3 |
| `decode_c1` | 73.0 to 75.7 |
| `decode_c2` | 104.1 to 112.5 |
| `decode_c4` | 137.8 to 141.5 |
| `decode_c8` | 180.1 to 181.0 |
| `decode_prose_c1` | 44.8 to 45.7 |
| `decode_prose_c2` | 65.7 to 66.0 |
| `decode_prose_c4` | 85.9 to 92.6 |
| `decode_prose_c8` | 111.4 to 112.4 |

One validation-profile start with two sweeps. Structured workloads vary: c4 spread is 17.09 percent and c6 is 13.77 percent relative to the first sweep; c6 range relative to the lower value is 15.96 percent. Both individual sweeps and ranges are retained. The production profile and resident collective toggle were not installed in this start.

## v1.7.4 base recipe, no decode levers

Build `v1.7.4`; weight mode `1`; configuration `none`. Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1185.6 to 1266.2 |
| `decode_c1` | 62.7 to 64.4 |
| `decode_c2` | 106.5 to 108.0 |
| `decode_c4` | 126.8 to 136.5 |
| `decode_c8` | 168.3 to 172.5 |
| `decode_prose_c1` | 43.8 to 47.9 |
| `decode_prose_c2` | 58.0 to 60.3 |
| `decode_prose_c4` | 77.1 to 79.8 |
| `decode_prose_c8` | 101.6 to 106.4 |

An automated check held this run on one quality-panel answer that review found correct (the grader rejected an expression wrapped in inline backticks). Every decode wave completed without findings.

## v1.0.0 measured, published unchanged in v1.1.0

Build `v1.0.0`; weight mode `0`; configuration `none`. Serving starts: 1; sweeps/repetitions: 1. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | unavailable: v1.0.0 was not measured on this instrument. |
| `decode_c1` | unavailable: v1.0.0 was not measured on this instrument. |
| `decode_c2` | unavailable: v1.0.0 was not measured on this instrument. |
| `decode_c4` | unavailable: v1.0.0 was not measured on this instrument. |
| `decode_c8` | unavailable: v1.0.0 was not measured on this instrument. |

All other prompt classes, repetitions, author-protocol rows and evidence hashes remain in [frozen results](https://github.com/jakejharris/jspark3/blob/v1.8.0/release/results-v1.8.0.json) and [measurement definitions](https://github.com/jakejharris/jspark3/blob/v1.8.0/release/RELEASE-NUMBERS.md).

Read [installation](https://github.com/jakejharris/jspark3/blob/v1.8.0/docs/INSTALL.md), [operations](https://github.com/jakejharris/jspark3/blob/v1.8.0/docs/OPERATIONS.md), [limitations](https://github.com/jakejharris/jspark3/blob/v1.8.0/docs/LIMITATIONS.md), and [licensing](https://github.com/jakejharris/jspark3/blob/v1.8.0/docs/LICENSING.md). Source archive, SBOM and checksums accompany this release.
