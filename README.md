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

| **68.77 tok/s** | **87.67 tok/s** | **223.14 tok/s** | **1,234 tok/s** |
|---|---|---|---|
| Code decode, v1.1 | Structured decode, v1.1 | Four-stream aggregate, v1.1 | Prefill, historical v1.0.0 |
| Single stream | Code 68.77 / prose 34.64 | 56.59 per stream | 113,908-token prompt |

Single-stream figures are descriptive medians of three battery medians;
C4 is the median of three original sparkDash client waves. Prefill is one
historical prompt/TTFT measurement, with no comparable v1.1 rerun.
[Sources, estimators and all repeats](docs/BENCHMARKS.md#headline-stat-bases)
keep these distinct from paired effects and selected demo takes.

Cadence's second-start paired effects were **+19.17% prose** and **+7.61%
structured count** against its own disabled reference route. **Code has no
replicated gain**: +3.23%, with an interval spanning zero. The first start's
sham failed, making its effects diagnostic only; the second start's
predeclared sham passes. Batches and prefill use the wide path. The quality
battery has candidate-only failures; semantic parity and sustained concurrency
remain uncertified. [Paired evidence](docs/BENCHMARKS.md#v11-cadence-evidence)
and [limitations](docs/LIMITATIONS.md#v11-cadence-limits) retain every result.

Historical v1.0.0 code was **1.49× as fast: 66.3 vs 44.6 tok/s** on the
compatibility-adapted two-Spark screen. Its sparkDash C4 result was
**251 tok/s vs 146.5** on separate fleets and dates. These are historical
comparisons, not current speedup claims. [Exact figures and caveats](docs/BENCHMARKS.md#headline-comparisons-v100-historical-record)
and [regressions and missed gates](docs/LIMITATIONS.md#measured-regressions-and-misses-v100-evidence)
remain published.

## Prerequisites

- Exactly three DGX Sparks, each with two RoCE-v2 interfaces, cabled as a
  triangle. Each direct leg needs its own IPv4 network at MTU 9000.
- Docker with the NVIDIA runtime, cgroup v2, `rdma-core`, a management network
  and non-interactive SSH. Allow about 180 GB for the model tree per rank;
  preflight requires 72 GiB available host memory and 8 GiB free work/model
  filesystem space.
- A Linux or macOS controller with Python 3.9+, `ssh`, `rsync` and `sha256sum`.

## Quick start

Download the published release on the computer you will use to manage the three Sparks:

```bash
git clone --branch v1.1.0 https://github.com/jakejharris/jspark3.git
cd jspark3
(cd recipe && sha256sum -c SHA256SUMS)
```

**Cloning is only the first step.** Follow the [installation guide](docs/INSTALL.md#2-get-the-recipe-onto-every-rank)
to copy the setup scripts to each Spark, download the required software and
model files, prepare the files for three Sparks, and set up the network
connections. Complete these steps before starting the server.

Then [fill in every machine setting in `.env`](docs/INSTALL.md#6-configure-env-on-the-controller)
and [check the setup, start the server, and test it](docs/INSTALL.md#7-preflight-start-verify).
The startup scripts refuse files that differ from the required versions.
The first Spark, called rank 0 in the commands, serves requests under the model
name `glm-5.3-flash`. Thinking is off by default and can be enabled per request.
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

Brandon M. Music made this version of GLM-5.3 Flash. Mia's AI Lab hosts a copy with credit, and JSPARK3 keeps a copy of the same model version. The model files are the same, so you only need to download them once. The model cards and other repository files differ.

The install guide downloads [Mia's copy](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f). You can use the
[JSPARK3 copy](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc) instead.
The GitHub repository and release downloads contain the setup software, not the model files.

Cadence also needs the separate [Inco AI DFlash2 draft](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
a smaller model that helps generate answers faster. Download it once as well.
Follow the [install guide](docs/INSTALL.md) for the required supporting files
and setup. [Sources and verification](huggingface/jspark3/PROVENANCE.md)
record the exact model versions and how the files were compared.

## License

Original recipe code, tooling and prose are Apache-2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)); that does not relicense dependencies. Target weights remain
under the attribution-required, source-available ShapleyMcg License v1.0,
including its named exclusion; Z.AI's base model remains MIT. DFlash2 is
CC BY-NC-ND 4.0 for research and evaluation; commercial use requires permission
from Inco AI. Read the [full license boundaries](docs/LICENSING.md).

## Credits

Z.AI created GLM-5.3 Flash. Brandon M. Music made the smaller EXL3/TR3 version
used here. Mia's AI Lab hosts a credited copy and provides the software and
instructions to run it on two Sparks. JSPARK3 supplies its own copy and the
setup for three Sparks. Inco AI provides DFlash2, the smaller helper model.

Thanks to Z.AI, Brandon M. Music, Mia-AiLab, Inco AI, z-lab, FlyCockpit,
vcruz305, tonyd2wild, sfxnz, vLLM and ExLlamaV3. Their specific contributions
and exact source versions are in [third-party notices](THIRD_PARTY_NOTICES.md).

## Attribution

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Citation

Use [CITATION.cff](CITATION.cff) or [CITATION.bib](CITATION.bib), and cite the
upstream works alongside this recipe.
