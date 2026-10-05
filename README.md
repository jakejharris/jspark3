# JSpark3

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks.

**Release candidate: JSpark3 v2.0.2 (GLM-5.3 Flash), ready for review.** This is v2.0.1 plus an image-history checkpoint
fix and GIF frame-zero support. See the [candidate installation guide](INSTALL.md), [upgrade guide](UPGRADING.md),
[release gate](RELEASE-GATE.md) and [root-cause evidence](release/v2.0.2/ROOTCAUSE.md).

The live retry passed at 18:37Z on 2026-10-05 with unchanged engine commits: 8/9/10-image resumes were
`0 -> 68 -> 174`, with both nonzero hits from disk; GIF passed and smoke passed 6/6. The first attempt's
fixture failure and rollback remain recorded in the evidence. Acceptance waited for durable checkpoints;
continuous traffic can still skip optional disk saves. This is a small CUDA acceptance sequence, not a
long-context latency benchmark or a clean installation of the final public recipe. Publication awaits approval.

**Current published release:** [v2.0.1](https://github.com/jakejharris/jspark3/releases/tag/v2.0.1).
The performance and installation observations below are historical v2.0.1 results, sourced from
[the measurements](release/MEASUREMENTS-v2.0.1.md) and [benchmark methods](docs/BENCHMARKS.md).

With base weights and the draft model, eight concurrent requests produced 126.0 tok/s in total on short prompts of 41 to 62 tokens. With refusal-removed (ablit) weights and the draft model, eight concurrent requests on the same short prompts (41 to 62 tokens) produced 121.7 tok/s in total.

v2.0.1 replaces the vLLM engine of v1.8.x with [a fork of TensorFold 0.3.6.2 (MIT)](https://github.com/ashhart/TensorFold), and serves public 4-bit GLM-5.3 Flash weights that the installer downloads, verifies against pinned hashes and splits across your three Sparks. v2.0.1 also saves conversation state to each Spark's disk by default: with base weights and the draft model, a conversation of at least 100,000 tokens that had been pushed out of memory showed its first visible text 1.5 s after it was continued, against 49.9 s to read it from scratch.

**What you need:** three NVIDIA DGX Sparks, connected by a direct high-speed (RDMA) link. Cable the boxes' ConnectX-7 ports in a ring (rank 0 to rank 1, rank 1 to rank 2, rank 2 to rank 0), with each port up, RDMA working and MTU 9000; tensors travel over these cables. Every box also needs a shared LAN on which it can reach rank 0. The engine uses that LAN only to coordinate startup; tensor traffic stays on the cables. Follow the [installation guide](https://github.com/jakejharris/jspark3/blob/v2.0.1/INSTALL.md) to install and run it on three DGX Sparks. The download, build and splitting steps were run from public sources on a Spark that had never run this project, and the packaged release was started and checked on three Sparks.

**Disk on each Spark**, by component:

- Container image: about 25 GB
- Wheels and engine build: under 50 MB
- Download: 181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB)
- Split weights: 63.9 GB for the first third, 62.8 GB for each of the other two
- Kernel cache: about 45 MB per host
- Session cache: up to 64 GiB, written only while 150 GiB stays free
- After splitting: the full download ($DATA/base/weights) is not needed to serve, so you may delete it.

**Before you connect a client:** The server listens on loopback (127.0.0.1) only, with no authentication and no CORS. Reach it through an SSH tunnel or a reverse proxy that adds authentication; don't expose the port.

## Weights

The installer offers two weight variants. The default is `base` (base weights). Choose the variant with `WEIGHTS=ablit` in `cluster.env`, or `--weights ablit` on fetch-weights.sh, split.sh and serve.sh (default `base`).

**`base`**: GLM-5.3 Flash in 4-bit MLX format, with the model's own multi-token prediction head, from [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP/tree/76add2a341a1cd90ad0e86bb69839ea9c35827c6) at revision `76add2a341a1cd90ad0e86bb69839ea9c35827c6` (MIT; formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects). The installer downloads them at the pinned revision, verifies every file against a pinned SHA-256 list, and splits them into three per-host parts, checked against a shipped manifest. This project hosts none of the v2.0.1 weights. Download size: 181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB).

**`ablit`**: refusal-removed (abliterated) weights from [`orcarouter/GLM-5.3-Flash-Uncensored-MLX`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX), an opt-in install for ablit development, red-teaming and refusal research. License: MIT (Copyright (c) 2026 Z.AI Co., Ltd), plus the use conditions on the source model card, quoted below. The installer downloads them with your own token at the pinned revision and converts them on your machine. Ready in this release: `scripts/fetch-weights.sh --weights ablit` downloads the source with your own token, `scripts/convert-ablit.sh` converts it and writes the three per-host parts, and `manifests/ablit/` checks a third you already have. We ran the shipped conversion on one DGX Spark, and its output matched these manifests file for file. On your own machine, `scripts/convert-ablit.sh` runs the pinned conversion scripts in `scripts/ablit/` inside the release's pinned container image, with no network and no GPU, then writes and checks the three per-host parts. The pinned base weights, TensorFold/GLM-5.3-Flash-MLX-4bit-MTP at revision 76add2a341a1cd90ad0e86bb69839ea9c35827c6 (MIT), supply the four-bit tensor layout and the native prediction layer that the conversion restores. The chat template is the base checkpoint's MIT template plus six lines added by this recipe, also under MIT; its reconstructed bytes are hash-checked. Measured on one DGX Spark: the 200.1 GB source download took about an hour on our connection. Converting, splitting and checking then took about 26 minutes of processing, used no GPU and under 4 GiB of process memory, and needed about 371 GB of free disk beyond the downloaded source (about 571 GB in all), on top of the base weights you already installed. `scripts/convert-ablit.sh` checks for about 400 GB free before it starts. To use it:

1. Sign in to a Hugging Face account.
2. Open [the source page](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX) and accept its terms yourself.
3. Give the installer your own token in `HF_TOKEN`.

The source's model card sets these conditions, quoted verbatim:

> - It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
> - **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
>
> By downloading or using this model you acknowledge and accept the above.

You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.

Results for both variants, and for the base weights without the draft model, are in the [release notes](https://github.com/jakejharris/jspark3/releases/tag/v2.0.1).

## Draft model license

The default draft model, Inco AI's [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), is CC BY-NC-ND 4.0: non-commercial research and evaluation use only. The installer downloads it unmodified from Inco AI; this project never redistributes it. For commercial use, run the base weights without the draft model: start with `scripts/serve.sh --drafter none` on all three hosts (or set `DRAFTER=none` in `cluster.env`), and the model drafts with its own multi-token prediction head. That path runs MIT weights on a permissively licensed engine (MIT, with some Apache-2.0 code) and an Apache-2.0 recipe, inside NVIDIA's container under NVIDIA's terms. On the base weights, the draft model is the only non-commercial component.

## Upgrade from v1.8.x and rollback

v2.0.1 is a new installation, not an in-place upgrade. The engine changes from vLLM to a fork of TensorFold 0.3.6.2 (MIT), and the weights change from the v1.8.x EXL3 files to public 4-bit MLX-format weights split across the three hosts. Stop v1.8.x before you start v2.0.1, and keep your v1.8.4 checkout and weights if you might roll back.

Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; v2.0.1 runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.

v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside this release; no target version is assigned.

To roll back, stop v2.0.1 and start v1.8.4 from its tag, following its own installation guide. v1.8.4 is the documented rollback.

In your v1.8.4 checkout:

```bash
git fetch --tags && git checkout v1.8.4
```

Or from a fresh clone:

```bash
git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4
```

See [upgrade and rollback](https://github.com/jakejharris/jspark3/blob/v2.0.1/UPGRADING.md).

## Releases

- **v2.0.2**: candidate ready for review with the two image fixes; [status and gates](RELEASE-GATE.md).

- **v2.0.1**: current; the engine is a fork of TensorFold 0.3.6.2 (MIT). [release](https://github.com/jakejharris/jspark3/releases/tag/v2.0.1) · [installation guide](https://github.com/jakejharris/jspark3/blob/v2.0.1/INSTALL.md)
- **v1.8.4** (previous, vLLM): the documented rollback. [release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.4) · [installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md)
- v2.0.0: internal build, not published.
- v1.8.0: do not install. It requires a container image that was never published.
- v1.1.0 (Cadence): [historical guide](https://github.com/jakejharris/jspark3/blob/v1.1.0/README.md)

This branch contains the v2.0.2 candidate recipe. Published releases remain available at their tags.
Use the published v2.0.1 tag until this candidate receives release approval.

JSpark3 Tempo, a separate named release for DeepSeek-V4.1 Flash with its own version numbers, lives in [jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek).

## Licenses

Recipe files are Apache-2.0. The engine (`engine/`) is MIT: a fork of TensorFold 0.3.6.2 (MIT). It includes third-party code under its own permissive licenses (MIT and Apache-2.0), listed in its THIRD_PARTY_NOTICES, and two lines ported from upstream TensorFold that stay Apache-2.0, credited in its NOTICE. The base weights are MIT. The refusal-removed weights are MIT plus their source card's use conditions. The chat template is Z.AI's MIT template plus six lines added by this recipe, also under MIT. The draft model is CC BY-NC-ND 4.0 (non-commercial); it is downloaded at install and never redistributed here. The NVIDIA container image is pulled from NGC under NVIDIA's terms. See the [licensing guide](https://github.com/jakejharris/jspark3/blob/v2.0.1/NOTICE) and the [third-party notices](https://github.com/jakejharris/jspark3/blob/v2.0.1/THIRD_PARTY_NOTICES.md).

## Credits

v2.0.1 builds on work by Z.AI (GLM-5.3 Flash, the base model, and its chat template); Hugging Face and the transformers contributors (the GLM-5.3 Flash model code the engine's CUDA path implements); Vontra, now TensorFold on Hugging Face (the 4-bit MLX base weights); Ash Hart and the TensorFold contributors (TensorFold 0.3.6.2, the engine release this project forks); Inco AI (the DFlash2 draft model); orcarouter (the refusal-removed source weights); z-lab (DFlash); MiaAI-Lab (the follower doorbell, adapted from upstream TensorFold); Dorian (an upstream TensorFold server fix, ported); turboderp (ExLlamaV3's EXL3 format, which the engine's own decoders read); Apple (MLX); and Prince Canuma and the mlx-vlm contributors (GLM-5.3 Flash code the engine follows and ports).

The [third-party notices](https://github.com/jakejharris/jspark3/blob/v2.0.1/THIRD_PARTY_NOTICES.md) are the complete record: every third-party component of v2.0.1, its authors and its license.

v1.8.x credits, including Brandon M. Music's EXL3/TR3 checkpoint, are in the [v1.8.4 third-party notices](https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md).

The [ShapleyMcg](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
