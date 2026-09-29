# JSpark3 v1.8.4

GLM-5.3 Flash on three NVIDIA DGX Sparks, with tensor parallelism across all three GB10s, expert-parallel MoE and DFlash2 speculative decoding, with per-rank identity checks before launch and measured gates before you admit client traffic.

**Parity with v1.8.0, with cooperative MoE on by default.** Of 12 measured cells, 8 overlap the frozen v1.8.0 ranges, 2 are above and 2 are below; code decode at two streams was not measured. The coop on/off comparison is pending, so no uplift is claimed.

| Cell | v1.8.4 measured (tok/s) | v1.8.0 frozen (tok/s) | Relation |
|---|---|---|---|
| structured c1 | 96.7-97.1 (n=2) | 97.5-98.7 (n=2) | below |
| structured c2 | 120.1-120.5 (n=2) | 94.0-142.9 (n=2) | overlaps |
| structured c4 | 170.9-213.9 (n=2) | 168.6-229.7 (n=2) | overlaps |
| structured c8 | 202.7-220.9 (n=2) | 198.0-198.6 (n=2) | above |
| prose c1 | 44.7-46.1 (n=2) | 44.3-49.1 (n=2) | overlaps |
| prose c2 | 63.7-65.5 (n=2) | 61.6-63.5 (n=2) | above |
| prose c4 | 86.0-88.4 (n=2) | 83.3-90.7 (n=2) | overlaps |
| prose c8 | 105.6-106.4 (n=2) | 110.1-110.8 (n=2) | below |
| code c1 | 65.2-84.7 (n=5) | 68.2-73.3 (n=2) | overlaps |
| code c2 | not measured | 101.0-103.9 (n=2) | n/a |
| code c4 | 136.2-139.5 (n=5) | 136.9-141.7 (n=2) | overlaps |
| code c8 | 164.2-180.9 (n=5) | 174.0-179.7 (n=2) | overlaps |
| prefill (Pi turns, post-hygiene) | 1197.1-1273.4 (n=8) | 1195.0-1262.6 (n=8) | overlaps |

Decode is aggregate tok/s, with 512 forced tokens per stream and streams started
together. Structured and prose ran 2 sweeps; code ran 5 repeats at temperature 0,
and c2 was not run. Prefill is tok/s per post-hygiene Pi turn. All v1.8.4 values
come from a single coop-on boot. Ranges are observed, not confidence intervals.

The v1.8.0 column is the frozen production-stock release cohort in [release/results-v1.8.0.json](release/results-v1.8.0.json). Every v1.8.4 figure and its evidence hash is in [release/results-v1.8.4.json](release/results-v1.8.4.json), with the comparison method in [v1.8.4 measurements](release/MEASUREMENTS-v1.8.4.md) and definitions in [benchmarks](docs/BENCHMARKS.md).

**Historical v1.8.0 reference comparison.** The v1.8.0 build was faster than the published three-Spark reference build on its own benchmark at one and two streams, in every run.

The published reference reports three-Spark (TP=3) results for its GLM-5.3 Flash recipe from sparkDash's Decode bench: prose prompts, 512 tokens, thinking off. We ran the same benchmark with those settings, five times per row:

| Streams | JSpark3 median tok/s (range, 5 runs) | Published reference, author-reported tok/s |
|---|---|---|
| 1 | **47.57** (43.34 to 48.6) | 40.1 |
| 2 | **64.26** (60.12 to 67.59) | 56.6 |
| 3 | 78.09 (72.66 to 84.27) | 75.5 |
| 4 | 86.44 (83.16 to 87.09) | 88.4 |

At three streams, individual runs overlap the reference. At four streams, our median is 86.44 tok/s versus the author-reported 88.4 tok/s. The reference figures are author-reported, from a different fleet and date, and their repetition count and harness commit aren't published ([reference, 3× Spark section](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f4970207e9fb2bdeac40d88b7cbef18c98aea310/README.md#3x-spark-tp3)).

**Coop is on by default.** Operators compile cooperative MoE locally and must
produce byte-identical copies of the library that passed GPU qualification and
was sealed for this release. Its SHA-256 is `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`, the component seal
is `42ea2a36ab5b9de05e7e8b1d690bd1fe7014968ec7381ac4a62516143deb05ec`, and the measured policy is `3623acda4ea998cb53dbeb24c7ac815240a393a31d2e56997df2ce5291bc316b`.
The [component pin](recipe/config/coop-release.json),
[BUILD record](recipe/overlays/v16/coop/BUILD.json) and
[campaign evidence](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/component-v1.8.4.json) bind those identities. The campaign ran
memcheck and racecheck with instrumented candidate-launch evidence and a real
out-of-bounds detector control.

**A/B status:** coop on/off comparison pending; no uplift claimed

The qualified default prepares `JSPARK3_V16_COOP=1`. `prepare_runtime.py --coop-off`
and `build_native.py --display-only` are explicit diagnostic opt-outs from that
default. Changing coop requires a restart. Reproducibility and the component
seal do not admit your fleet: complete the installation and boot
qualification gates before traffic. The kernel uses shared cudart linkage;
AGPL-3.0-only component terms, MIT header notices and the separately licensed
model dependencies remain applicable. See [licensing](docs/LICENSING.md).

The release boot passed same-boot admission. Its receipts and the owner's
[admission attestation](release/v1.8.4-admission/attestation.json) are in
`release/v1.8.4-admission/`; the full admission logs are private and hash-bound
in its finalization receipt.

See [installation](docs/INSTALL.md), [component qualification](docs/COOP_REPRODUCIBILITY.md)
and the [v1.8.4 measurement record](release/results-v1.8.4.json).


**Weights.** The default is GLM-5.3 Flash with EXL3 quantization, unedited (`ABLIT=0`), using `production-stock`. The routed experts are 4-bit (EXL3/TR3 checkpoint), and "stock" means unedited, not full precision. At load time the dense trunk uses INT8 weights and the default trunk FP8 path; the KV cache is FP8. Edited weights (`ABLIT=1`) are an explicit opt-in supplied separately; no edited checkpoint is redistributed. Changing modes requires a service restart and recomputes conversation prefixes.

## What's in it

- **A controller that checks each rank.** `fleetctl` won't start until every rank matches its receipts. It checks the image and native builds you made, all 120 checkpoint shards by hash, the RoCE triangle at MTU 9000, the GID index, and memory and no-swap limits. Once the fleet is running, `verify` checks it end to end, down to retrieving a code word from a prompt longer than 32K tokens.
- **Three-Spark TP3 with expert parallelism.** TP3's head padding lives in derived runtime views, so the weights you download are never modified.
- **Prefix caching for coding agents.** Per-group LRU retention for DFlash2 drafter windows keeps agent prefixes reusable. Fine-grained prefix hits run with our replay-window fix.
- **Warmup before admission.** GLM and DFlash2 variants compile at boot; the shipped qualification command warms request shapes, applies page-cache hygiene and refuses compilation after warmup.
- **Speculative decoding that keeps its width.** Adaptive verification selects how many DFlash2 draft tokens to verify. Grammar-constrained requests (`response_format` with `json_schema`) keep the full speculative width.
- **A cooperative-MoE decode kernel for TP3,** adapted to expert-parallel ranks with at most 96 local experts. The v1.8.4 figures above come from this release's qualified binary; the coop on/off comparison is pending.
- **A release you can audit and rebuild.** Every file is hash-listed and checked by the release validator, with a CycloneDX SBOM and per-file SPDX licensing. You build the serving image and native libraries on your own Spark, and the default native build is compiled twice and must match byte for byte.

Recipe code is Apache-2.0. A running service also loads AGPL-3.0-only components and always uses the DFlash2 draft model, which is licensed for non-commercial research and evaluation only ([licensing](docs/LICENSING.md)).

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

## Credits

JSpark3 builds on work by Z.AI, Inco AI, z-lab, Mia's AI Lab,
FlyCockpit, vcruz305, sfxnz, Tony, turboderp, coolbho3k, gabewillen,
plotarmordev, outstandly, the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component contributions,
source revisions and license notices.

The [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
