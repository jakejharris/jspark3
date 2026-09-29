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
  - serving-recipe
---

# JSpark3 v1.8.4

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks, with an attributed mirror of the target weights.

**Current release: v1.8.4.** Start with the [v1.8.4 release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.4) and the [v1.8.4 installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md).

Do not install v1.8.0, including the old archive in this HF repo: it requires a container image that was never published.

Download the [v1.8.4 recipe tarball from GitHub](https://github.com/jakejharris/jspark3/releases/download/v1.8.4/jspark3-recipe-v1.8.4.tar.gz), extract it, and run the checksum and validator commands below from the extracted `jspark3` directory. Or clone the [v1.8.4 source tag](https://github.com/jakejharris/jspark3/tree/v1.8.4):

```sh
git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py . --require-final
```

Then follow the installation guide to build and verify your own local image and native binaries. There is no JSpark3 image to pull from GHCR. Keep the prepared default `JSPARK3_V16_COOP=1` and complete the guide's three-host qualification before serving traffic.

## Weights and measurements

The mirrored target weights are unchanged. They are Brandon M. Music's EXL3/TR3 checkpoint ([source revision](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b)). For serving, use [Mia-AiLab's pinned weights](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f) or [this mirror at its verified revision](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc), as required by the recipe's checksum ledger. DFlash2 is a separate download at revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, described in the installation guide.

v1.8.4 serves at parity with v1.8.0, with cooperative MoE on by default; see the [v1.8.4 measurements](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/MEASUREMENTS-v1.8.4.md). The coop on/off comparison is pending; no uplift is claimed.

## Licenses and attribution

The target weights remain under the ShapleyMcg License v1.0 in [LICENSE](https://huggingface.co/jakejharris/jspark3/blob/main/LICENSE). Z.AI's base model is MIT. JSpark3's original code and prose are Apache-2.0; included derivatives retain AGPL-3.0-only and vendored headers retain MIT. DFlash2 retains its non-commercial research/evaluation restriction. See the [v1.8.4 licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/LICENSING.md) for component boundaries and notices.

## Credits

JSpark3 builds on work by Brandon M. Music, who made the
[EXL3/TR3 4-bpw checkpoint](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b)
that every rank loads, and by Z.AI, Inco AI, z-lab, Mia's AI Lab, FlyCockpit,
Victor Cruz (vcruz305), turboderp, Emi Huang (coolbho3k), Gabriel Willen (gabewillen),
plotarmordev, Ratul Sarna (ratulsarna), nood-co1, Zbigniew Majewski (knapcio),
Lilian Moraru (lilianmoraru), the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md) for component contributions,
source revisions and license notices.

The [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg/tree/4dc85c999983bf46ebdae3821839a5079cc5de17) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
