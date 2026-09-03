---
license: other
license_name: shapleymcg-license-1.0
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - shapleymcg
  - glm
  - exl3
  - tr3
  - vllm
  - quantized
  - dgx-spark
  - glm-5.3-flash
  - tensor-parallel
  - expert-parallel
  - speculative-decoding
  - serving-recipe
  - reproducibility
---

# JSpark3 v1

JSpark3 turns three NVIDIA DGX Sparks into one fast GLM-5.3 Flash server, with
a reproducible TP3 recipe and public benchmarks. The recipe uses tensor
parallel 3, expert parallel 3, a two-leg RoCE-v2 triangle, EXL3/TR3 4-bpw
target weights, a DFlash2 k=7 draft, FP8 KV cache, prefix caching, a
1,000,000-token configured context, and a selective INT8 (W8A16 Marlin)
overlay for the model trunk.

This public repository is the release home for the card, results, license set,
and attributed target-weight mirror. All 123 allowlisted Git LFS payloads were
verified by size and SHA-256 against the pinned manifest, and
[`jspark3/MIRROR-COMPLETION.json`](jspark3/MIRROR-COMPLETION.json) was verified
byte for byte before merge into the public main revision. The recipe,
documentation, and evidence are in the
[`v1.0.0` GitHub release](https://github.com/jakejharris/jspark3/releases/tag/v1.0.0),
released 2026-09-02.

## Weights

**These weights are not ours.** This repository carries an exact,
hash-verifiable mirror of
[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
at revision `25a44fdbf16862a46b7cc9921142c6c81350af2f`, which is itself a
byte-identical re-host of
[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
at revision `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`. Brandon M. Music is the
quantization author; Z.AI created the base model. JSpark3 trained nothing,
quantized nothing, and modified no weight byte.

The mirror is described file by file in
[`jspark3/WEIGHTS-MANIFEST.json`](jspark3/WEIGHTS-MANIFEST.json): every file at
the pinned revision with its size, SHA-256, and how that hash was obtained. The
chain, the verification method, and one recorded discrepancy in the upstream
checksum file are in [`jspark3/PROVENANCE.md`](jspark3/PROVENANCE.md). The
upstream card is preserved verbatim as
[`UPSTREAM_MODEL_CARD.md`](UPSTREAM_MODEL_CARD.md), and every other upstream
file keeps its exact upstream path so that a checkpoint contract validating a
download from this mirror validates exactly as it does upstream.

The DFlash2 speculative draft is a separate checkpoint from Inco AI under
CC BY-NC-ND 4.0. It is **not** mirrored here; operators fetch it from its own
repository at its own pinned revision.

> **Hub status: all 123 allowlisted Git LFS payloads and
> [`jspark3/MIRROR-COMPLETION.json`](jspark3/MIRROR-COMPLETION.json) were remotely
> verified before merge into the public main revision.** See
> [`jspark3/UPLOAD.md`](jspark3/UPLOAD.md) for the recorded procedure.

## Pinned inputs

| Input | Identity |
|---|---|
| Target checkpoint | [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) at `25a44fdbf16862a46b7cc9921142c6c81350af2f`, declared byte-identical to `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at `5ab363a8dcf6405955fd5f99671e01a1c9fb124b` |
| Draft checkpoint | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Serving image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58` |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (Z.AI) |

## How it compares with what was already public

The comparison that matters for a Spark owner is against recipes that were
publicly available before this release. The reference rows below are
**author-reported**: measured by each recipe's own author, on that author's
hardware, with that author's harness. The JSpark3 v1 row is our own local
measurement, and the Basis column states the conditions of every row.
**No percentage, delta, or ranking is computed between any two rows**: node
counts, quantization lanes, speculation, context, clocking, safety envelope,
and estimators all differ.

| Recipe | Nodes | Lane | Context | Decode (tok/s) | Basis |
|---|---:|---|---:|---|---|
| **JSpark3 v1** `v1.0.0` | 3 | EXL3/TR3 4 bpw, DFlash2, W8A16 trunk overlay | 1,000,000 | structured count 81.962; code 66.257; prose 29.049 | local; frozen 24-request screen, thinking off, temperature 0, 400 max tokens, warm server; medians of three batteries; per-stream estimator |
| FlyCockpit TP3 `9093765c` | 3 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | structured count 69.0 / 68.5 / 71.2; code 52.3 / 58.7 / 58.2 | author-reported |
| [neko-legends TP4](https://github.com/neko-legends/spark-bench/blob/a1d8daffad44ad69d8f9e27e621b5f4afc4157fe/README.md#L81-L188) | 4 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | code 64.5; structured 100.9; math 77.8; prose 23.1; C4 aggregate 253 | author-reported; four DGX Sparks, warm client-wall [`bench_exl3.py`](https://github.com/neko-legends/spark-bench/blob/a1d8daffad44ad69d8f9e27e621b5f4afc4157fe/README.md#L140-L170), thinking off |
| Mia TP2 `c190db1a` | 2 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | sparkDash C1 62.9; lab structured 65.1; lab prose 27.1 | author-reported |
| jetnet TP3 `bfc820ec` | 3 | NVFP4 with Marlin W4A16, MTP-4 | 512K | 35.2; DFlash2 lane, thinking on, 47.2 | author-reported |

Three of those recipes were also run on this fleet, each with a pinned source
revision and every adaptation disclosed. None is an exact reproduction and
none replays a source's own harness, so these are separate evidence rather
than a restatement of the rows above. They are same-prompt agent runs with
independent trajectories, which makes them product evidence, not engine-rate
comparisons.

| Local reproduction | Fidelity | Nodes | Agent aggregate decode (tok/s) |
|---|---|---:|---:|
| `mia-tp2-historical-0e2e78f` | site/safety-adapted | 2 | 24.913 |
| `mia-tp2-current-c190db1a-adapted` | compatibility-adapted | 2 | 24.728 |
| `fly-derived-9093765c-adapted` | minimal-correctness/safety-adapted | 3 | 29.042 |

No literal FlyCockpit run and no jetnet run exists here; jetnet was studied
statically and never run on this fleet.

## What the overlay changed, internally

Separately from the comparison above, the project ran a matched A/B against
**the matched three-Spark control (same recipe, overlay disabled), an
unreleased internal development build**. That control is not a product, was
never published, and is not a market comparison. It is the only comparison in
this release where hardware, topology, checkpoint, draft, image, serving
envelope, workload, estimator, and safety contract are all matched.

Hardware: three DGX Sparks (GB10, SM 12.1), two RoCE-v2 legs per node at MTU
9000. Server warm. Single-stream decode on a frozen 24-request plan (4
warm-up, 20 scored), thinking disabled, temperature 0, top-p 1, fixed seed,
400 max tokens; per-request rate is (completion tokens minus one) over the
interval between the first and last visible streamed token, and the phase
value is the median. All numbers are in `RESULTS.json` with their estimators.

| Phase | Earlier control battery | JSpark3 v1 (median of 3) | Delta | Same-day paired control | JSpark3 v1 r3 | Paired delta |
|---|---:|---:|---:|---:|---:|---:|
| Code | 63.861 | 66.257 | +3.75% | 61.768 | 66.257 | +7.27% |
| Structured count | 77.510 | 81.962 | +5.74% | 76.863 | 81.962 | +6.63% |
| Prose | 28.308 | 29.049 | +2.62% | 26.810 | 29.049 | +8.35% |
| C3 per-stream median | 67.591 | 69.634 | +3.02% | 65.208 | 51.382 | -21.20% |
| C6 per-stream median | 37.460 | 54.694 | +46.01% | 53.149 | 54.694 | +2.91% |

The C6 gain against the older control battery mostly reflects that battery's
state; the paired +2.91% is the credible figure. C3 was variable and lost its
strict pairing.

Token pacing in the paired battery: median inter-token interval 98.645 to
91.912 ms (-6.83%), p99 120.472 to 108.105 ms (-10.27%), worst interval
364.416 to 148.344 ms (-59.29%).

Matched concurrency waves (aggregate service throughput, one wave each, 84
requests, thinking disabled): C12 155.733 to 155.986 tok/s (+0.16%), C24
206.235 to 208.723 (+1.21%), C48 229.966 to 237.946 (+3.47%). DFlash2
acceptance 64.323%, 65.935%, 64.562%. Fairness did not improve; C48 time to
first token at p90 was 96.722 s.

Matched 113,908-token prefill proxy: 1277.443 to 1234.246 tok/s (-3.38%);
time to first token 89.169 to 92.290 s (+3.50%). A measured regression.

Internal promotion gates the measured build missed, kept as disclosed
evidence: campaign code median 66.257 tok/s against a 67.0 floor (short by
0.743 tok/s, 1.11%), and a longest uncompensated interior slow run of 14
against a limit below 5 in the agent demonstration. Neither is a correctness
or stability failure.

Evidence grade for everything measured here: `ENGINEERING-EVIDENCE`, produced
by the project on its own fleet; no third-party reproduction yet.

## Intended use

Serving GLM-5.3 Flash on a three-DGX-Spark fleet you operate, for research,
evaluation, and internal use consistent with the upstream licenses. JSpark3 is
a serving recipe, not a model and not a fine-tune; the weights here are a
mirror of someone else's quantization.

## Limitations

Exactly three DGX Sparks; every input pinned; prefill slower than the matched
control; three-stream waves variable; long time to first token at 48 streams;
single-fleet evidence with small sample sizes; no literal FlyCockpit or jetnet
reproduction and no jetnet run at all; no public accuracy benchmark for this
release; no authentication on the endpoint. The full list is in the GitHub
repository's `docs/LIMITATIONS.md`.

## Licenses

The weights in this repository are licensed under the **ShapleyMcg License
v1.0**, reproduced in full as [`LICENSE`](LICENSE). It is a source-available,
attribution-required license; it is not OSI-approved open source, and the
license text says so itself. Downstream copies of the Work stay under it, and
the license contains a named exclusion, reproduced as written in the license
file. Attribution is a condition of the grant, not a courtesy.

The base model `zai-org/GLM-5.3-Flash` is MIT, Copyright (c) 2026 Z.AI Co.,
Ltd.; keep that notice with any copy of the base work. Third-party notices
carried by the upstream repository are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/B12X-APACHE-2.0.txt).

JSpark3's own recipe code, tooling, and prose are Apache-2.0, in
[`jspark3/RECIPE-LICENSE`](jspark3/RECIPE-LICENSE) with its notices in
[`jspark3/THIRD_PARTY_NOTICES.md`](jspark3/THIRD_PARTY_NOTICES.md). That
license covers none of the weights. The DFlash2 draft is CC BY-NC-ND 4.0 for
research and evaluation use; commercial use requires separate permission from
Inco AI. The assembled endpoint is therefore neither unrestricted open source
nor commercial-ready.

## Attribution

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

DFlash2 is non-commercial research and evaluation use only absent separate
permission from Inco AI. Apache-2.0 covers only this package's own code and
prose.

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

## Citation

```bibtex
@software{jspark3v1_2026,
  author  = {{JSpark3 authors}},
  title   = {JSpark3 v1: a reproducible three-DGX-Spark serving recipe for GLM-5.3 Flash},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/jakejharris/jspark3}
}
```

Cite the upstream works alongside it: Z.AI (GLM-5.3 Flash), Brandon M. Music
(ShapleyMcg), Inco AI (DFlash2), z-lab (DFlash), MiaAI-Lab, FlyCockpit,
vcruz305, vLLM, and ExLlamaV3.
