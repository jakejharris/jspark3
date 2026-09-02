---
base_model: zai-org/GLM-5.3-Flash
library_name: transformers
pipeline_tag: image-text-to-text
license: other
license_name: shapleymcg-license-1.0
license_link: https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE
tags:
  - shapleymcg
  - glm
  - exl3
  - tr3
  - vllm
  - quantized
---

# GLM-5.3-Flash TR3 4bpw — mirror (Spark recipe)

**This is not an original quantization.** It is a **byte-identical redistribution** of
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
at Hugging Face revision
[`5ab363a8dcf6405955fd5f99671e01a1c9fb124b`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b)
so that the 2× DGX Spark recipe
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
stays fetchable if the upstream Hub id moves.

Brandon M. Music created the EXL3/TR3 checkpoint (ShapleyMcg). Z.AI created the
base model. Mia's AI Lab only re-hosts this snapshot and serves it on GB10.

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music
> (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under
> the ShapleyMcg License v1.0, an attribution-required license that grants no
> rights to the person known as "0xSero." Use of ShapleyMcg without this
> attribution is unlicensed.

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

## License

- **This checkpoint / ShapleyMcg work:** [ShapleyMCG License 1.0](LICENSE) (source-available; not OSI “open source”). The `LICENSE` file is the upstream file, unmodified.
- **Base model** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): MIT, Copyright (c) 2026 Z.AI Co., Ltd. Keep that notice with any copy of the base work.
- **Spark serve scripts** (separate GitHub repo): MIT.

Do not relicense these shards as MIT. Downstream copies of this Work stay under ShapleyMCG License 1.0.

## What this is

Uniform-K4 EXL3/TR3 routed-expert checkpoint of GLM-5.3-Flash (~164 GiB, 120
safetensor shards). Provenance metadata from the upstream snapshot is kept.

This Hub repo is **not** Brandon’s SM120 B12X / NVFP4-KV / EP2/DCP2 daily driver.
Do not use the `verdictai/glm53-flash-exl3-k4:…` image with a “this is the Spark
recipe” assumption. His original model card is saved as
[`ORIGINAL_MODEL_CARD.md`](ORIGINAL_MODEL_CARD.md).

Spark recipe (2× NVIDIA GB10, TP=2, fp8 MLA KV, DFlash2):

- GitHub: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`) is a **separate** checkpoint under
CC BY-NC-ND 4.0 and is **not** mirrored here.

## Source pin

| Field | Value |
|---|---|
| Upstream | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` |
| Upstream revision | `5ab363a8dcf6405955fd5f99671e01a1c9fb124b` |
| Canonical ShapleyMcg repo | https://github.com/brandonmmusic-max/shapleymcg |
| Canonical Hub (author) | https://huggingface.co/brandonmusic |
