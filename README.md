# JSpark3 v1.1 (Cadence)

<img width="1920" height="1080" alt="hook" src="https://github.com/user-attachments/assets/e528a9a7-f824-480f-85e2-2a91b322cdcf" />

**Run GLM-5.3 Flash across three NVIDIA DGX Sparks as one OpenAI-compatible
endpoint, with a reproducible TP3 recipe and public benchmarks.**

Cadence adds a measured single-stream decode mode and carries the long-context
kernel fix. It is a serving recipe, not a new model: JSpark3 trains nothing,
quantizes nothing, and changes no checkpoint weight bytes.

[Install](docs/INSTALL.md) · [Benchmarks](docs/BENCHMARKS.md) ·
[Limitations](docs/LIMITATIONS.md) · [Architecture](docs/ARCHITECTURE.md)

> **Before serving:** the target weights require attribution and carry a named
> exclusion. The DFlash2 draft is non-commercial. The assembled endpoint is
> neither unrestricted open source nor commercial-ready. See [License](#license).

## Results

**Cadence's paired effects compare its candidate route with its own disabled
reference route, not with published v1.0.0 benchmarks.** Two independent serving
starts used the same paired design. The first start's sham control failed;
those figures are diagnostic only. The second start's predeclared sham passes.

| Single-stream decode | First start (diagnostic) | Second start |
|---|---:|---:|
| Prose | +16.18% | +19.17% |
| Structured count | +7.66% | +7.61% |
| Code | +9.71% | +3.23%; interval spans zero |

The prose and count intervals exclude zero; **code has no replicated gain**.
Batches and prefill fall back to the wide path. The quality battery includes
candidate-only failures, and neither semantic parity nor sustained concurrency
is certified. Read the [conditions, confidence intervals and quality results](docs/BENCHMARKS.md#v11-cadence-evidence)
and [Cadence limits](docs/LIMITATIONS.md#v11-cadence-limits).

Historical v1.0.0 results remain separate: single-stream code reached
**66.3 tok/s versus 44.6** on the compatibility-adapted two-Spark recipe;
sparkDash four-stream aggregate decode reached **251 tok/s versus 146.5**.
The code screen ran on this fleet; sparkDash used the same pinned author
protocol on separate fleets and dates. These are different workloads and
estimators from the Cadence pairing, not a release-to-release comparison.
[Exact figures, agent-task comparisons and receipts](docs/BENCHMARKS.md#headline-comparisons-v100-historical-record)
include every mismatch. [Historical regressions and missed gates](docs/LIMITATIONS.md#measured-regressions-and-misses-v100-evidence)
remain part of the record.

## Prerequisites

- Exactly three DGX Sparks, each with two RoCE-v2 interfaces, cabled as a
  triangle. Each direct leg needs its own IPv4 network at MTU 9000.
- Docker with the NVIDIA runtime, cgroup v2, `rdma-core`, a management network
  and non-interactive SSH. Allow about 180 GB for the model tree per rank;
  preflight requires 72 GiB available host memory and 8 GiB free work/model
  filesystem space.
- A Linux or macOS controller with Python 3.9+, `ssh`, `rsync` and `sha256sum`.

## Quick start

Get the published release on your controller:

```bash
git clone --branch v1.1.0 https://github.com/jakejharris/jspark3.git
cd jspark3
(cd recipe && sha256sum -c SHA256SUMS)
```

**Cloning is only the first step.** Follow the [installation guide](docs/INSTALL.md#2-get-the-recipe-onto-every-rank)
to copy the recipe to every rank, stage the pinned image, checkpoints and
FlyCockpit source, build and validate TP3 runtime views, and configure the
fabric. Those required downloads and host-specific settings must be complete
before starting the service.

Then [fill every fleet value in `.env`](docs/INSTALL.md#6-configure-env-on-the-controller)
and [run preflight, start, health and verification](docs/INSTALL.md#7-preflight-start-verify).
The lifecycle refuses input or hash drift. It exposes model `glm-5.3-flash`
on rank 0, with thinking off by default and switchable per request.
[Send a first request](docs/INSTALL.md#8-first-request) or consult
[operations and shutdown](docs/OPERATIONS.md).

v1.1.0 includes the kernel and fabric corrections prepared for v1.0.1; no
separate v1.0.1 release was published. For older deployments, read the
[v1.0.0 long-context warning](docs/INSTALL.md#known-issue-in-v100-single-stream-requests-past-32768-tokens)
and [hand-launch fabric requirements](docs/INSTALL.md#fabric-environment-required-for-hand-launches).
The configured 1,000,000-token context is not maximum-capacity certification.

## Explore the recipe

The runtime combines TP3/EP3, EXL3/TR3 4-bpw target weights, DFlash2 speculation,
FP8 KV and prefix caching, and a selective W8A16 trunk overlay. Cadence adds
an INT8 QKV decode shadow and a request-local speculative width controller,
while retaining the stock indexer workspace, drafter and sampler.

| Read about | Start here |
|---|---|
| Topology, transforms and lifecycle | [Architecture](docs/ARCHITECTURE.md) and [technical report](docs/TECHNICAL-REPORT.md) |
| Measured effects, public comparisons and ablations | [Benchmarks](docs/BENCHMARKS.md), [results](results/SUMMARY.md) and [limitations](docs/LIMITATIONS.md) |
| Exact inputs and repeatable measurements | [Reproducibility](docs/REPRODUCIBILITY.md) and [pinned dependencies](manifests/dependencies.json) |
| Source layout and runnable files | [Repository map](docs/ARCHITECTURE.md#repository-map) and [recipe](recipe/README.md) |
| Changes and release provenance | [Changelog](CHANGELOG.md) and [publication record](RELEASE-GATE.md) |
| Contributing or reporting a vulnerability | [Contributing](CONTRIBUTING.md) and [security](SECURITY.md) |

Current release: [v1.1.0 (Cadence), 2026-09-07](https://github.com/jakejharris/jspark3/releases/tag/v1.1.0).
Historical release: [v1.0.0, 2026-09-02](https://github.com/jakejharris/jspark3/releases/tag/v1.0.0).

## Weights

The GitHub recipe and release assets contain no checkpoint weights. The
[public target mirror](https://huggingface.co/jakejharris/jspark3) is an
attributed, byte-identical copy of Brandon M. Music's quantization, re-hosted
by Mia-AiLab. [Provenance and checksum records](huggingface/jspark3/PROVENANCE.md)
include the upstream chain and recorded checksum discrepancy.

Pinned inputs are the
[Mia-AiLab target](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f)
and [Inco AI draft](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The draft is fetched separately and is not mirrored. The recipe verifies
serving bytes regardless of download source.

## License

Original recipe code, tooling and prose are Apache-2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)); that does not relicense dependencies. Target weights remain
under the attribution-required, source-available ShapleyMcg License v1.0,
including its named exclusion; Z.AI's base model remains MIT. DFlash2 is
CC BY-NC-ND 4.0 for research and evaluation; commercial use requires permission
from Inco AI. Read the [full license boundaries](docs/LICENSING.md).

## Credits

Thanks to Z.AI, Brandon M. Music, Mia-AiLab, Inco AI, z-lab, FlyCockpit,
vcruz305, tonyd2wild, sfxnz, vLLM and ExLlamaV3. Their specific contributions
and pinned sources are in [third-party notices](THIRD_PARTY_NOTICES.md).

## Attribution

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Citation

Use [CITATION.cff](CITATION.cff) or [CITATION.bib](CITATION.bib), and cite the
upstream works alongside this recipe.
