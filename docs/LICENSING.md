# Licensing

JSpark3 v1 composes several works with different terms. This page separates
them so you can decide what you may do. It is a summary, not legal advice, and
the publisher's completed container audit is recorded in `RELEASE-GATE.md`.

## What this repository licenses

Everything original here, meaning the lifecycle controller, preflight,
entrypoint, transform programs, the W8A16 overlay module, the validators,
builders, evidence tooling, documentation, and diagrams, is released under the
Apache License, Version 2.0 (`LICENSE`, `NOTICE`).

Apache-2.0 covers only that original work. It does not relicense, and cannot
relicense, any checkpoint, container image, or upstream source revision the
recipe depends on.

## What this repository does not contain

- No model weights in the source tree. No `.safetensors` shard, tokenizer
  binary, or model index is committed here; the Hugging Face payload under
  `huggingface/` carries the checkpoint's own metadata files and license, and
  the weight objects themselves are fetched from the Hub.
- No container image layers.
- No patched copy of vLLM. The five runtime transforms are programs that
  reconstruct pinned upstream changes inside a disposable container at start.

## What the Hugging Face repository will redistribute

The Hugging Face side of this release is prepared to mirror the target
checkpoint: an exact, hash-verifiable copy of
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at its pinned revision, which is itself
a byte-identical re-host of the quantization author's repository. The mirror
carries the checkpoint's own `LICENSE`, its notices, and its provenance files
unmodified, and JSpark3's Apache-2.0 set sits beside them under
`huggingface/jspark3/`. The exact 29-file metadata allowlist is public at
`e9cbbafaf9ae4ab64f385c2f68e7fe2f06d78676`. The attributed target-weight
transfer is in progress on a separate review branch and is not merged into
the public Hub main revision. Until remote verification and maintainer merge
complete, main consists of metadata, a manifest, a provenance record, and a
tool.

Operators fetch the pinned upstream revisions themselves. Their terms apply to
that fetch and to serving.

## Third-party terms that apply to a running JSpark3 v1 endpoint

| Component | Source | Terms as published upstream | Practical effect |
|---|---|---|---|
| Target checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (declared byte-identical to `brandonmusic/GLM-5.3-Flash-tr3-4bpw`) | Hugging Face, pinned revision in `manifests/dependencies.json` | ShapleyMcg License v1.0, attribution-required | Source-available, attribution-required, not OSI open source; downstream copies stay under this license; the license contains a named exclusion, reproduced as written. The attribution sentence in `REQUIRED_ATTRIBUTION.md` must stay byte-for-byte intact in copies, model cards, and generated reports, and the license's provenance fields must not be removed from artifact metadata. |
| Base model GLM-5.3 Flash | Z.AI | As published by Z.AI for the base model | Review the base model terms for your use. |
| Draft checkpoint `incoai/GLM-5.3-Flash-DFlash2` | Hugging Face, pinned revision | CC BY-NC-ND 4.0, research and evaluation use | The default serving path is non-commercial. Commercial use requires separate permission from Inco AI. |
| DFlash speculative decoding | `z-lab/dflash` | As published upstream | Referenced technique; review upstream terms. |
| Serving image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks` | MiaAI-Lab, pinned by digest | As published by MiaAI-Lab | v1.0.0 pulls and launches this exact upstream image by digest. The prepared local derivative retains its NVIDIA-derived upstream layers; adding labels and notices did not satisfy the NGC derived-container redistribution grant, so no JSpark3 GHCR image is published for v1.0.0. |
| vLLM | vllm-project | Apache-2.0 | Transforms modify a copy inside the container only. |
| ExLlamaV3 | turboderp-org | As published upstream | Kernel provider for EXL3 weights inside the image. |
| FlyCockpit, vcruz305, sfxnz, tonyd2wild repositories | GitHub, pinned commits | As published by each author | Technique sources and reconstruction targets; credited in `THIRD_PARTY_NOTICES.md`. |

Because the draft checkpoint is non-commercial and the target checkpoint is
attribution-required, the assembled endpoint must not be described as
unrestricted open source or as commercial-ready. That sentence is a truth
contract for every public surface of this release.

The Dockerfile and local build command remain available to reproduce and
inspect the prepared derivative. They do not grant redistribution rights. Do
not push, export, publish, or otherwise redistribute a local build without
independently satisfying NVIDIA's terms and every applicable upstream term.

## Required attribution

The following sentence is reproduced verbatim from the upstream license and
must not be edited:

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Ownership statements the project does not make

The project does not claim to have trained, fine-tuned, or quantized GLM-5.3
Flash, and does not claim ownership of the EXL3/TR3 checkpoint, DFlash2, vLLM,
ExLlamaV3, the MiaAI-Lab image, or any upstream recipe. The contribution is the
three-Spark topology, runtime adaptation, operating envelope, measurement
campaign, and the reproducible, fail-closed serving recipe.
