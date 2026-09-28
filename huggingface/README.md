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
  - glm
  - exl3
  - dgx-spark
  - serving-recipe
---

# JSpark3 v1.8.2

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks, with an attributed mirror of the target weights.

**Current release: v1.8.2.** Start with the [v1.8.2 release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.2) and the [v1.8.2 installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.2/docs/INSTALL.md).

Do not install v1.8.0, including the old archive in this HF repo: it requires a container image that was never published.

Download the [v1.8.2 recipe tarball from GitHub](https://github.com/jakejharris/jspark3/releases/download/v1.8.2/jspark3-recipe-v1.8.2.tar.gz), extract it, and run the checksum and validator commands below from the extracted `jspark3` directory. Or clone the [v1.8.2 source tag](https://github.com/jakejharris/jspark3/tree/v1.8.2):

Use a shallow clone: a full `git clone` fails v1.8.2's privacy scan on old commit history.

```sh
git clone --depth 1 --branch v1.8.2 https://github.com/jakejharris/jspark3.git
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py .
```

Then follow the installation guide to build and verify your own local image and native binaries. There is no JSpark3 image to pull from GHCR. Keep the default `JSPARK3_V16_COOP=0`; cache preparation, operator hygiene and three-host hardware qualification remain required before serving traffic.

## Weights and measurements

The mirrored target weights are unchanged. For serving, use [Mia-AiLab's pinned weights](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f) or [this mirror at its verified revision](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc), as required by the recipe's checksum ledger. DFlash2 is a separate download described in the installation guide.

Historical measurements reached up to 141.7 tok/s code and 90.7 tok/s prose at four streams (best of two runs, stock weights). These unchanged v1.8.0 measurements used `coop=1`; they do not qualify or predict the default operator `coop=0` configuration. See the [measurement definitions and full ranges](https://github.com/jakejharris/jspark3/blob/v1.8.2/release/RELEASE-NUMBERS.md) and [frozen results](https://github.com/jakejharris/jspark3/blob/v1.8.2/release/results-v1.8.0.json).

## Licenses and attribution

The target weights remain under the ShapleyMcg License v1.0 in [LICENSE](https://huggingface.co/jakejharris/jspark3/blob/main/LICENSE). Z.AI's base model is MIT. JSpark3's original code and prose are Apache-2.0; included derivatives retain AGPL-3.0-only and vendored headers retain MIT. DFlash2 retains its non-commercial research/evaluation restriction. See the [v1.8.2 licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.2/docs/LICENSING.md) for component boundaries and notices.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
