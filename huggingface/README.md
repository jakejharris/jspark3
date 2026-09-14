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

# JSpark3 v1.1 (Cadence)

**Three DGX Sparks, one OpenAI-compatible GLM-5.3 Flash endpoint, with the
recipe and evidence to rebuild it.**

JSPARK3 is the measured three-Spark path for the EXL3/TR3 checkpoint in this
repository. It combines the exact mirrored weights with a pinned serving
stack and a fail-closed fleet controller. v1.1 — release name **Cadence** —
adds a measured single-stream decode mode (a QKV INT8 decode shadow plus a
request-local speculative width controller that falls back wide for batches
and prefill) and carries the proven long-context kernel fix. The weights here
are unchanged by all of this.

[v1.1.0 release — 2026-09-07](https://github.com/jakejharris/jspark3/releases/tag/v1.1.0)
· [Release guide and verified limits](jspark3/V1.1.0-RELEASE.md).
The GitHub tag selects the release content; this card update is a separate
documentation commit and changes no checkpoint or tokenizer bytes.

## Results

The v1.1 effects were measured inside the candidate against its own disabled
reference route; they are not comparisons against the published v1.0.0
numbers below, which used different request sets and estimators. Two
independent serving starts ran the same paired design; the first start's sham
control failed its predeclared resolution margin, so first-start figures are
diagnostic only, while the second start's predeclared sham passes.

| Paired effect (candidate route vs disabled reference route) | First start (diagnostic — sham failed) | Second start (sham passes) |
|---|---|---|
| Prose decode | **+16.18%** | **+19.17%** (95% CIs exclude zero) |
| Structured-count decode | **+7.66%** | **+7.61%** |
| Code decode | +9.71% | +3.23% with the interval spanning zero — **not a replicated gain** |
| Scope | Single-stream decode with the width controller active; batches of two or more requests and prefill fall back to the wide path |

Conditions, confidence intervals, quality results (including candidate-only
delivered-answer failures), and every caveat are in the
[GitHub benchmarks page](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/BENCHMARKS.md#v11-cadence-evidence).

### v1.0.0 — historical benchmarks, unchanged

| Measured result | JSpark3 v1.0.0 (historical) |
|---|---:|
| Single-stream code decode | **1.49x faster**, 66.3 vs 44.6 tok/s on the two-Spark recipe |
| sparkDash clamp-code time to first token | **391 ms**, down from 719 ms on two Sparks |
| Four-stream aggregate decode | **251 tok/s**, up from 146.5 tok/s on two Sparks |
| Same agent task and prompt | **1.8x the throughput**, 44.6 vs 24.7 tok/s on two Sparks |
| Configured context | **1,000,000 tokens** |

The code-screen comparison and agent task were run on the JSpark3 fleet
against the compatibility-adapted current Mia two-Spark recipe. sparkDash
used the same pinned author protocol as Mia's published figures, on separate
fleets and dates. See the
[machine-readable results](https://github.com/jakejharris/jspark3/blob/v1.1.0/results/results.json)
and [exact figures, receipts, and caveats](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/BENCHMARKS.md#headline-comparisons).

## What you get

- An exact, hash-verifiable mirror of Brandon M. Music's
  [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg) EXL3/TR3
  quantization, as re-hosted by Mia-AiLab, with its provenance intact.
- A TP3 and EP3 serving recipe with DFlash2, FP8 KV cache, prefix caching,
  a selective W8A16 trunk overlay, and — in v1.1 — the measured QKV decode
  shadow and speculative width controller.
- A fail-closed preflight, start, health, and verification path.
- Public measurements with the regressions and missed gates left in.

**[Install and run it with the GitHub recipe](https://github.com/jakejharris/jspark3#quick-start).**

## Known issues in the v1.0.0 recipe (historical)

Two issues affected the v1.0.0 recipe. Both were found by a community bug
report from [@BTCXoomer on X](https://x.com/BTCXoomer), both were fixed since
v1.0.1, and v1.1.0 carries both fixes. The weights in this repository are not
affected and nothing here was re-uploaded.

1. **Single-stream requests past 32,768 tokens crash the v1.0.0 recipe.**
   The v1.0.0 transform contract pinned vLLM's `persistent_topk` kernel
   enabled, and on GB10 that kernel aborts once a single decoding stream
   passes 32,768 tokens of context (85 cooperative CTAs against the 48 the
device holds residently; the fallback needs more shared memory than a GB10
   exposes). Every measured run and every published benchmark executed with
   that kernel disabled, so the v1.0.0 construction had never actually been
   run past 32,768 tokens by anyone. v1.0.1 moved the disable into the
   transform itself, and v1.1.0 carries it in its transform contract; the
   transform emits the exact kernel file the measured arms executed.
   Integrated live verification now passes a pinned single-stream request
   with 48,957 prompt tokens and 51 completion tokens, returning the codeword
   verbatim. Candidate `a729583` and host verifier `456a262` are recorded
   separately in the [release guide](jspark3/V1.1.0-RELEASE.md). The final
   archive has not been cold-boot tested, and maximum-context capacity
   remains unverified. No weight byte or historical benchmark figure changes. Details
   and v1.0.0 workarounds: the
   [install-path warning](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/INSTALL.md#known-issue-in-v100-single-stream-requests-past-32768-tokens)
   and the
   [mechanism and arithmetic](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/LIMITATIONS.md#kernel-disable-provenance-and-the-32768-token-single-stream-boundary-v100).
2. **Containers launched by hand need `NCCL_IB_SUBNET_AWARE_ROUTING=1`.**
   The variable is new in NCCL 2.30.7 and defaults to off; with it off, NCCL
   pairs NICs by index and routes rank 0 to rank 2 over rank 1's leg, which
   breaks a switchless three-node triangle. The lifecycle controller always
   set it; the v1.0.0 documentation never named it, so an operator launching
   containers with their own tooling could not know it existed. The v1.0.1
   install steps list the
   [full fabric environment](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/INSTALL.md#fabric-environment-required-for-hand-launches).

Use the [v1.1.0 release](https://github.com/jakejharris/jspark3/releases/tag/v1.1.0)
from the GitHub repository; install commands select that exact tag. Benchmarks are unchanged across the kernel
fix: every published number was measured with the kernel disable applied,
which is the same file the transform now emits.

> **License and weight provenance:** JSpark3's original recipe code and
> documentation are Apache-2.0. The weights here are Brandon M. Music's exact
> EXL3/TR3 quantization, re-hosted byte-for-byte by Mia-AiLab and mirrored by
> JSpark3 without changing a weight byte. They remain under the
> attribution-required, source-available ShapleyMcg License v1.0, including
> its named exclusion; Z.AI's base model remains MIT. DFlash2 is not mirrored
> and remains a separate CC BY-NC-ND 4.0 dependency.

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
byte for byte before merge into the verified weights-mirror revision. The
recipe, documentation, and evidence are in the
[`v1.0.0` GitHub release](https://github.com/jakejharris/jspark3/releases/tag/v1.0.0),
released 2026-09-02, preserved as history. The current recipe is
[v1.1.0 (Cadence), dated 2026-09-07](https://github.com/jakejharris/jspark3/releases/tag/v1.1.0).

## Weights

Brandon M. Music made this version of GLM-5.3 Flash. Mia's AI Lab hosts a copy with credit, and JSPARK3 keeps a copy of the same model version. The model files are the same, so you only need to download them once. The model cards and other repository files differ.

Use [Mia's copy](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f) or [JSPARK3's copy of this model version](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc).
Follow the [install guide](https://github.com/jakejharris/jspark3/blob/main/docs/INSTALL.md)
for the supporting files and the separate DFlash2 download.

The [file list](jspark3/WEIGHTS-MANIFEST.json) records each file's size and
checksum, an identifier for its contents. The [source note](jspark3/PROVENANCE.md)
explains who made the files, how they were checked, and a difference found in
an older checksum list. The original model card is kept unchanged as
[`UPSTREAM_MODEL_CARD.md`](UPSTREAM_MODEL_CARD.md). Other original files keep
their folder locations so the install guide can check either download source.

DFlash2 is a separate, smaller model from Inco AI that helps generate answers
faster. Download it separately using the install guide. It is not included
in this copy of GLM. Its license is CC BY-NC-ND 4.0.

Before the JSPARK3 copy became public, all 123 large files were checked by
size and checksum. The [completion record](jspark3/MIRROR-COMPLETION.json)
was checked too. The [upload record](jspark3/UPLOAD.md) describes those checks.

## Exact model and software versions

| Download | Exact version |
|---|---|
| GLM model files | [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f), revision [25a44fd](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f). The model files match Brandon M. Music's [version credited by Mia, 5ab363a8](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b), and [1ae6d70](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/1ae6d70430a12d762917786696db06a7b4f9bbae). The model cards and other repository files differ. |
| DFlash2 helper model | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Docker software package | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58` |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (Z.AI) |

## How it compares with what was already public

This section and the one after it are the v1.0.0 historical record, preserved
unchanged; the v1.1 Cadence evidence is never merged with these tables.

The comparison that matters for a Spark owner is against recipes that were
publicly available before this release. The reference rows below are
**author-reported**: measured by each recipe's own author, on that author's
hardware, with that author's harness. The JSpark3 v1 row is our own local
measurement, and the Basis column states the conditions of every row.
Rows without a matched local protocol remain context because node counts,
quantization lanes, speculation, context, clocking, safety envelope, and
estimators differ. The headline exceptions use the same frozen screen, the
same sparkDash author protocol, or the same agent task and prompt, with each
remaining mismatch stated in the linked benchmark page.

| Recipe | Nodes | Lane | Context | Decode (tok/s) | Basis |
|---|---:|---|---:|---|---|
| **JSpark3 v1** `v1.0.0` | 3 | EXL3/TR3 4 bpw, DFlash2, W8A16 trunk overlay | 1,000,000 | structured count 81.962; code 66.257; prose 29.049 | local; frozen 24-request screen, thinking off, temperature 0, 400 max tokens, warm server; medians of three batteries; per-stream estimator |
| FlyCockpit TP3 `9093765c` | 3 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | structured count 69.0 / 68.5 / 71.2; code 52.3 / 58.7 / 58.2 | author-reported |
| [neko-legends TP4](https://github.com/neko-legends/spark-bench/blob/a1d8daffad44ad69d8f9e27e621b5f4afc4157fe/README.md#L81-L188) | 4 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | code 64.5; structured 100.9; math 77.8; prose 23.1; C4 aggregate 253 | author-reported; four DGX Sparks, warm client-wall [`bench_exl3.py`](https://github.com/neko-legends/spark-bench/blob/a1d8daffad44ad69d8f9e27e621b5f4afc4157fe/README.md#L140-L170), thinking off |
| Mia TP2 `c190db1a` | 2 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | sparkDash C1 62.9; lab structured 65.1; lab prose 27.1 | author-reported |
| jetnet TP3 `bfc820ec` | 3 | NVFP4 with Marlin W4A16, MTP-4 | 512K | 35.2; DFlash2 lane, thinking on, 47.2 | author-reported |

Three of those recipes were also run on this fleet, each with a pinned source
revision and every adaptation disclosed. None is an exact reproduction and
none replays a source's own published harness, so these are separate evidence
rather than a restatement of the rows above. Their same-task agent runs used
independent trajectories, so the rates compare achieved product throughput
rather than isolate an engine-only effect.

| Same agent task | Fidelity | Nodes | Agent aggregate decode (tok/s) |
|---|---|---:|---:|
| **JSpark3 v1** | v1.0.0 (historical) | 3 | **44.583** |
| `mia-tp2-historical-0e2e78f` | site/safety-adapted | 2 | 24.913 |
| `mia-tp2-current-c190db1a-adapted` | compatibility-adapted | 2 | 24.728 |
| `fly-derived-9093765c-adapted` | minimal-correctness/safety-adapted | 3 | 29.042 |

On this task, JSpark3 delivered 1.8x the aggregate decode throughput of the
current adapted two-Spark run, 44.583 versus 24.728 tok/s. The agents followed
independent trajectories.

No literal FlyCockpit run and no jetnet run exists here; jetnet was studied
statically and never run on this fleet.

## What the overlay changed, internally

This is v1.0.0-era evidence about the trunk overlay, preserved unchanged; the
v1.1 additions have their own paired evidence in the
[GitHub benchmarks page](https://github.com/jakejharris/jspark3/blob/v1.1.0/docs/BENCHMARKS.md#v11-cadence-evidence).

Separately from the comparison above, the project ran a matched A/B against
**the matched three-Spark control (same recipe, overlay disabled), an
unreleased internal development build**. That control is not a product, was
never published, and is not a market comparison. It is the only comparison
in the v1.0.0 record where hardware, topology, checkpoint, draft, image, serving
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

Exactly three DGX Sparks; every input pinned; the v1.1 gains are single-
stream-scoped, with batches falling back wide and no replicated code gain;
first-start v1.1 figures are diagnostic only because that start's sham
control failed its predeclared resolution margin (the second start's
predeclared sham passes);
quality contains candidate-only delivered-answer failures and inconclusive
population parity; the concurrency burst evidence certifies no sustained
service, fairness, or capacity envelope; prefill slower than the matched
control; three-stream waves variable; long time to first token at 48 streams;
single-fleet evidence with small sample sizes; the sparkDash comparison uses
the same author protocol but separate fleets and dates; the agent comparison
uses independent trajectories; no literal FlyCockpit or jetnet reproduction
and no jetnet run at all; no public accuracy benchmark for this release; no
authentication on the endpoint; the kernel-file transform change is verified
and integrated live verification passed one request with 48,957 prompt
tokens and 51 completion tokens; maximum-context capacity, sustained
concurrency, and final-archive cold-start validation remain unproven.
The focused witness is a fixed admission workload, not a Pi speed claim;
interactive performance remains workload-dependent. The v1.0.0 recipe
carried the two known issues described above, fixed since v1.0.1. The full
list is in the GitHub repository's `docs/LIMITATIONS.md`.

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
@software{jspark3v11_2026,
  author  = {{JSpark3 authors}},
  title   = {JSpark3 v1.1 (Cadence): a reproducible three-DGX-Spark serving recipe for GLM-5.3 Flash},
  version = {1.1.0},
  year    = {2026},
  url     = {https://github.com/jakejharris/jspark3/releases/tag/v1.1.0}
}
```

Cite the upstream works alongside it: Z.AI (GLM-5.3 Flash), Brandon M. Music
(ShapleyMcg), Inco AI (DFlash2), z-lab (DFlash), MiaAI-Lab, FlyCockpit,
vcruz305, vLLM, and ExLlamaV3.
