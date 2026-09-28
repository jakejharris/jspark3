# JSpark3 v1.8.4 candidate

GLM-5.3 Flash on three NVIDIA DGX Sparks, with tensor parallelism across all three GB10s, expert-parallel MoE and DFlash2 speculative decoding, with per-rank identity checks before launch and measured gates before you admit client traffic.

**Historical v1.8.0 measurement: 136.9 to 141.7 tok/s code decode at 4 streams and 174.0 to 179.7 tok/s at 8, on unedited EXL3-quantized GLM-5.3 Flash weights.**

Measured on one serving start across two sweeps. Ranges span both sweeps.

- Prose decode: 83.3 to 90.7 tok/s at 4 streams, 110.1 to 110.8 tok/s at 8
- Prefill: 1195.0 to 1262.6 tok/s across eight Pi coding-agent turns extending a cached prefix, after page-cache hygiene

These come from the v1.8.0 release build (cooperative-MoE kernel on, one serving start, two sweeps); each range spans both sweeps. Decode is the combined rate of all streams, each forced to 512 output tokens at temperature 0 with thinking off. Every figure and its evidence hash is in [release/results.json](release/results.json), with definitions in [benchmarks](docs/BENCHMARKS.md).

The historical v1.8.0 build was faster than the published three-Spark reference build on its own benchmark at one and two streams, in every run.

MiaAI-Lab publishes three-Spark (TP=3) results for its GLM-5.3 Flash recipe from sparkDash's Decode bench: prose prompts, 512 tokens, thinking off. We ran the same benchmark with those settings, five times per row:

| Streams | JSpark3 median tok/s (range, 5 runs) | MiaAI-Lab, author-reported tok/s |
|---|---|---|
| 1 | **47.57** (43.34 to 48.6) | 40.1 |
| 2 | **64.26** (60.12 to 67.59) | 56.6 |
| 3 | 78.09 (72.66 to 84.27) | 75.5 |
| 4 | 86.44 (83.16 to 87.09) | 88.4 |

At three streams, individual runs overlap the reference. At four streams, our median is 86.44 tok/s versus the author-reported 88.4 tok/s. The reference figures are author-reported, from a different fleet and date, and their repetition count and harness commit aren't published ([reference, 3× Spark section](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md#3x-spark-tp3)).

**Coop.** The default native build now compiles display and coop twice and requires
coop SHA-256 `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`.
This candidate is pending component qualification and new stock-boot measurements.
Default runtime preparation refuses until the measured policy and complete seal
are integrated. The qualified path prepares `JSPARK3_V16_COOP=1`; `--coop-off`
selects an explicit diagnostic path. Build reproducibility does not admit a fleet.
See [installation](docs/INSTALL.md), [component qualification](docs/COOP_REPRODUCIBILITY.md)
and the [pending v1.8.4 record](release/results-v1.8.4.json).


**Weights.** The default is GLM-5.3 Flash with EXL3 quantization, unedited (`ABLIT=0`), using `production-stock`. The routed experts are 4-bit (Brandon M. Music's ShapleyMcg EXL3/TR3 checkpoint), and "stock" means unedited, not full precision. At load time the dense trunk uses INT8 weights and the default trunk FP8 path; the KV cache is FP8. Edited weights (`ABLIT=1`) are an explicit opt-in supplied separately; no edited checkpoint is redistributed. Changing modes requires a service restart and recomputes conversation prefixes.

## What's in it

- **A controller that checks each rank.** `fleetctl` won't start until every rank matches its receipts. It checks the image and native builds you made, all 120 checkpoint shards by hash, the RoCE triangle at MTU 9000, the GID index, and memory and no-swap limits. Once the fleet is running, `verify` checks it end to end, down to retrieving a code word from a prompt longer than 32K tokens.
- **Three-Spark TP3 with expert parallelism,** built on FlyCockpit's TP3 recipe. TP3's head padding lives in derived runtime views, so the weights you download are never modified. MiaAI-Lab [credits JSpark3 and outstandly](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md#3x-spark-tp3) as the route by which that recipe reached its kit.
- **Prefix caching for coding agents.** Our LRU retention for DFlash2 drafter windows builds on MiaAI-Lab’s per-group retention to keep agent prefixes reusable. Upstream fine-grained prefix hits ([plotarmordev, MiaAI-Lab PR #251](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/251)) run with our replay-window fix.
- **Warmup before admission.** GLM and DFlash2 variants compile at boot; the shipped qualification command warms request shapes, applies page-cache hygiene and refuses compilation after warmup.
- **Speculative decoding that keeps its width.** Adaptive verification selects how many DFlash2 draft tokens to verify. Grammar-constrained requests (`response_format` with `json_schema`) keep the full speculative width.
- **A cooperative-MoE decode kernel for TP3,** adapted from MiaAI-Lab's TP2 extension (an ExLlamaV3-derived kernel) to expert-parallel ranks with up to 96 local experts. The historical binary produced the figures above; no performance values are assigned to the new binary yet.
- **A release you can audit and rebuild.** Every file is hash-listed and checked by a 17-check validator, with a CycloneDX SBOM and per-file SPDX licensing. You build the serving image and native libraries on your own Spark, and the default native build is compiled twice and must match byte for byte.

JSpark3 builds on work by Z.AI (GLM-5.3), Brandon M. Music (ShapleyMcg), Inco AI (DFlash2), MiaAI-Lab, FlyCockpit, turboderp (ExLlamaV3), coolbho3k, gabewillen, plotarmordev, the vLLM project and others named in [third-party notices](THIRD_PARTY_NOTICES.md). The ShapleyMcg attribution is in [REQUIRED_ATTRIBUTION.md](REQUIRED_ATTRIBUTION.md). Recipe code is Apache-2.0. A running service also loads AGPL-3.0-only components and always uses the DFlash2 draft model, which is licensed for non-commercial research and evaluation only ([licensing](docs/LICENSING.md)).

Thirds and active TRIAR are deferred. Leave the prepared TRIAR resident flag at
1; admission proves the inactive Cadence/NCCL graph path. First-pass canaries
are recorded only; quick runs are diagnostic. Post-hygiene canaries and the
1100 tok/s prefill floor remain blocking, along with correctness, memory/no-swap
and zero compilation after warmup. The source retains `hardware_qualified=false`;
your passing qualification receipts apply to your exact boot.

Read [installation](docs/INSTALL.md), [operations](docs/OPERATIONS.md),
[benchmarks](docs/BENCHMARKS.md), [limitations](docs/LIMITATIONS.md) and
[release notes](release/RELEASE-NOTES.md). Validate the export with
`python3 -B tools/validate_release.py . --require-final`.
