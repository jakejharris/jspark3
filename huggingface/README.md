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
  - shapleymcg
---

# JSpark3 v1.8.4

## Next point release (unreleased)

Fresh installs default to [Brandon M. Music's EXL3/TR3 checkpoint](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b).
[Mia-AiLab's pinned copy](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f) and
[the pinned JSpark3 mirror](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc)
remain accepted byte-identical weight re-hosts. Existing copies need no re-download.
Keep the full revision pins and compatibility directory in the install guide.
The selected source retains ShapleyMcg License v1.0; later revisions change the license.

This prepared card change has offline verification only. The published install
links and measurement records below still describe v1.8.4.

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks, with cooperative
MoE enabled by default in the prepared runtime.

Operators compile byte-identical copies of the cooperative library that was
GPU-qualified and sealed for this release. The native SHA-256 is `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`;
the component seal is `42ea2a36ab5b9de05e7e8b1d690bd1fe7014968ec7381ac4a62516143deb05ec`; the measured policy is `3623acda4ea998cb53dbeb24c7ac815240a393a31d2e56997df2ce5291bc316b`.
The [release pin](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/config/coop-release.json),
[BUILD record](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/overlays/v16/coop/BUILD.json)
and [campaign evidence](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/component-v1.8.4.json) provide the identities and evidence.
The campaign ran memcheck and racecheck and included a deliberate invalid CUDA
access to verify the memcheck detector.

**A/B status:** coop on/off comparison pending; no uplift claimed

## Measured throughput

v1.8.4 serves at parity with v1.8.0. Of 12 measured cells from one admitted
coop-on boot, 8 overlap the frozen v1.8.0 ranges, 2 are above and 2 are below.
The coop on/off comparison is pending; no uplift is claimed.

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

The v1.8.0 column is the frozen release cohort in
[results-v1.8.0.json](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/results-v1.8.0.json); see the
[v1.8.4 measurements](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/MEASUREMENTS-v1.8.4.md). The release boot's
full admission logs are private and hash-bound in its
[finalization receipt](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/v1.8.4-admission/finalize.json).

The first requests after a fresh install can be about 1 s slower once, while GPU kernels (DeepGEMM) compile.

## Install and qualify

Use the [release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.4)
or its [recipe archive](https://github.com/jakejharris/jspark3/releases/download/v1.8.4/jspark3-recipe-v1.8.4.tar.gz).
Build the image and native artifacts locally; no JSpark3 container image or
compiled cooperative library is included in the source release.

```sh
git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py . --require-final
```

Follow [installation](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md)
and [operations](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/OPERATIONS.md).
The default is unedited weights (`ABLIT=0`), `production-stock`, coop on.
The v1.8.4 recipe downloads Mia-AiLab's byte-identical re-host of Brandon M. Music's checkpoint. The next-release default is described above.
`prepare_runtime.py --coop-off` and `build_native.py --display-only` are explicit
diagnostic opt-outs. The operator's exact boot must pass admission before traffic. Changes to coop
or weight mode require a restart and prefix recomputation.

## Licensing

The cooperative library builds with shared cudart linkage. Component licenses
remain in force: original recipe code and prose are Apache-2.0, cooperative
derivatives retain AGPL-3.0-only terms, and vendored headers retain MIT notices.
The assembled service is not wholly Apache-2.0. The target weights retain the
ShapleyMcg License v1.0; DFlash2 remains a separate dependency with a
non-commercial research/evaluation restriction. See the
[licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/LICENSING.md).

## Credits

JSpark3 builds on work by Brandon M. Music, who made the
[EXL3/TR3 4-bpw checkpoint](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b)
that every rank loads, and by Z.AI, Inco AI, z-lab, Mia's AI Lab, FlyCockpit,
Victor Cruz (vcruz305), turboderp, Emi Huang (coolbho3k), Gabriel Willen (gabewillen),
plotarmordev, Ratul Sarna (ratulsarna), nood-co1, Zbigniew Majewski (knapcio),
Lilian Moraru (lilianmoraru), the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](https://github.com/jakejharris/jspark3/blob/6c1a786fef9b0771d47ff7affa260cc1baffd716/THIRD_PARTY_NOTICES.md) for component contributions,
source revisions and license notices.

The [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg/tree/4dc85c999983bf46ebdae3821839a5079cc5de17) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

Schedule B citation:

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
