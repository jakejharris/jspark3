# JSpark3 v1.8.1

Up to 141.7 tok/s code and 90.7 tok/s prose decode at 4 streams on three DGX Sparks, unedited EXL3-quantized GLM-5.3 Flash weights.

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

A serving recipe for GLM-5.3 Flash on three DGX Sparks.

The default is GLM-5.3 Flash with EXL3 quantization, unedited (`ABLIT=0`), using `production-stock`. Here, "stock" means the quantized weights have no donor edits; it does not mean the original full-precision weights. Edited weights (`ABLIT=1`) are an explicit opt-in supplied separately; no edited checkpoint is redistributed. Choose the mode before launch. Changing modes requires a service restart and recomputes conversation prefixes.

The published v1.8.0 headline and ranges above were measured with cooperative MoE **on** (`JSPARK3_V16_COOP=1`). Today's operator-built runtime defaults to cooperative MoE **off** (`JSPARK3_V16_COOP=0`) until a qualified coop build ships. These historical numbers do not measure or qualify that default configuration.

The historical measured stock cohort used a validation profile for testing. It does not qualify production admission. This sanitized source distribution retains `hardware_qualified=false`; fresh qualification of the exact shipped configuration is required before production mode-0 admission. Measurements describe the separately identified serving configuration, not a hardware test of this source export.

Thirds and active TRIAR are deferred to the next release. No TRIAR speedup is claimed. The selected source retains TRIAR code resident but inactive under the NCCL graph path. Do not enable it; the startup resident flag is distinct from the runtime OFF state. The full positive inactive-path attestation and native qualification are required.

First-pass canaries are recorded only; quick runs are diagnostic. Post-hygiene canaries block admission, together with the 1100 tok/s prefill floor, correctness, memory/no-swap and zero post-readiness compilation requirements.

Stock free-form JSON may arrive in a Markdown fence, and code may be formatted with backticks. Clients needing bare JSON should request structured output with `response_format` and `json_schema`. Grammar-constrained requests retain the full speculative width. No quality-equivalence claim is made between stock and edited weights.

The single-request grammar-width crash is fixed. Its applicability to v1.1.0 is established by source inspection and reproduction on a later build, not a v1.1.0 hardware reproduction. Greedy near-ties still require controlled parity checks. Prefixes beyond retained replay boundaries may safely recompute.

Native binaries, weights and images are obtained separately. On one of your DGX Sparks, build and verify your own local image with `python3 -B tools/build_operator_image.py --output ../operator-image.json`, then build the ARM64 display library and probe with `python3 -B tools/build_native.py --image-receipt ../operator-image.json --output ../native-build`. Pass both receipts to runtime preparation as described in [installation](docs/INSTALL.md). There is no JSpark3 image to pull from GHCR; the historical reference digest is not a published image. The default build and prepared runtime omit the coop binary. Operator-native preparation explicitly selects coop off in its environment example; keep it off, as operator coop-on support needs a future implementation change. Cache preparation, operator hygiene and hardware qualification remain separate. Performance numbers remain the unchanged v1.8.0 measurements and do not qualify the operator coop-off configuration.

Original code and prose are Apache-2.0. Included derivatives retain AGPL-3.0-only and vendored headers retain MIT. The assembled service is not wholly Apache-2.0; per-file SPDX and REUSE records govern. The draft-model dependency retains its non-commercial research/evaluation restriction.

Read [installation](docs/INSTALL.md), [operations](docs/OPERATIONS.md), [benchmarks](docs/BENCHMARKS.md), [release notes](release/RELEASE-NOTES.md) and [licensing](docs/LICENSING.md). Verify the source with `python3 -B tools/validate_release.py .`.
