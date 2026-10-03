# Third-party notices

This is the complete record of the third-party components of JSpark3 v2.0.1: each one, its authors and its license.
The Apache-2.0 license in `LICENSE` covers this recipe's own files only; it relicenses nothing below. Review each
component's own terms before you use it.

## The engine, `engine/`

A fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.2 by its contributors, vendored as a single
source snapshot. The engine is MIT: a fork of TensorFold 0.3.6.2 (MIT). It includes third-party code under its own
permissive licenses (MIT and Apache-2.0), each listed with its authors in
[engine/THIRD_PARTY_NOTICES.md](engine/THIRD_PARTY_NOTICES.md), and two lines ported from upstream TensorFold that
stay Apache-2.0, credited in [engine/NOTICE](engine/NOTICE). [engine/LICENSE](engine/LICENSE) and `engine/LICENSES/`
hold the license texts. Those files are the record for everything inside `engine/`.

## Other files in this tree

| Component | Authors | License | Notes |
|---|---|---|---|
| Chat template, `template/chat-template.jinja` | Z.AI (stock template shipped with the base weights); six lines added by this recipe | MIT, Copyright (c) 2026 Z.AI Co., Ltd ([template/LICENSE](template/LICENSE)); the six added lines MIT, Copyright (c) 2026 the JSpark3 v2 contributors ([scripts/ablit/TEMPLATE-ADDITIONS-LICENSE.txt](scripts/ablit/TEMPLATE-ADDITIONS-LICENSE.txt)) | `template/README.md` says which six lines were added |
| Fabric launcher, `scripts/fabric/launch.py` | JSpark3 authors | Apache-2.0 | written for this recipe; no third-party code |

## Downloaded at install time, never redistributed

The scripts fetch each of these from its publisher at a pinned revision or digest and check it by sha256.

| Component | Authors / publisher | Pinned | License | How it is used |
|---|---|---|---|---|
| GLM-5.3 Flash base weights, 4-bit MLX format with the multi-token prediction head: [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP) (formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects) | Z.AI (model); the MLX conversion by Vontra, now TensorFold on Hugging Face | revision `76add2a341a1cd90ad0e86bb69839ea9c35827c6` | MIT, Copyright (c) 2026 Z.AI Co., Ltd | downloaded anonymously; split into three parts on your machines |
| DFlash2 draft model, [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | Inco AI | revision `bf582e4eacc1810f76656d1811693ff6c6737d2a` | CC BY-NC-ND 4.0: non-commercial use only; modified copies may not be shared | downloaded unmodified; requantized in memory at load only; never shared. With the session tier on, its cache state for your conversations is kept on your own disk. Optional: `--drafter none` |
| Refusal-removed (abliterated) source weights, [`orcarouter/GLM-5.3-Flash-Uncensored-MLX`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX) (ablit variant only, opt-in) | orcarouter; derived from Z.AI's GLM-5.3 Flash | revision `c02a5f6fa06f0aa444877b44d19fd5c96390329f` | MIT, Copyright (c) 2026 Z.AI Co., Ltd, plus the use conditions on its model card (quoted below and in `config/ablit-notice.txt`) | gated: fetched with your own Hugging Face token after you accept its terms; converted on your machine; never redistributed |
| NVIDIA PyTorch container, `nvcr.io/nvidia/pytorch` | NVIDIA | digest `sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` | NVIDIA's NGC container terms, and the licenses of the software inside it (including PyTorch, BSD-3-Clause, and Triton, MIT) | pulled anonymously by digest; the engine is built and served inside it |
| Python wheels in `wheels.lock` | listed below | exact files, by sha256 | listed below | downloaded from PyPI, installed into the container |
| Hugging Face CLI (`hf`) | Hugging Face, Inc. | `huggingface_hub` 2.1.1 (pins.env `HF_HUB_VERSION`); you install it, and `fetch-weights.sh` refuses other versions | Apache-2.0 | downloads the weights |

### Python wheels (`wheels.lock`)

Authors and licenses as each project's PyPI metadata gives them at the pinned version (`tools/wheel-licenses.json`).

| Project | Version | Authors | License |
|---|---|---|---|
| `anyio` | 4.15.1 | Alex Grönholm | MIT |
| `certifi` | 2026.7.22 | Kenneth Reitz | MPL-2.0 |
| `click` | 8.5.0 | Pallets | BSD-3-Clause |
| `filelock` | 4.0.6 | Bernát Gábor | MIT |
| `fsspec` | 2026.9.0 | Martin Durant | BSD-3-Clause |
| `h11` | 0.16.0 | Nathaniel J. Smith | MIT |
| `hf-xet` | 1.6.0 | Rajat Arya, Jared Sulzdorf, Di Xiao, Assaf Vayner, Hoyt Koepke | Apache-2.0 |
| `httpcore` | 1.0.9 | Tom Christie | BSD-3-Clause |
| `httpx` | 0.28.1 | Tom Christie | BSD-3-Clause |
| `huggingface-hub` | 1.33.0 | Hugging Face, Inc. | Apache-2.0 |
| `idna` | 3.20 | Kim Davies | BSD-3-Clause |
| `iniconfig` | 2.3.0 | Ronny Pfannschmidt, Holger Krekel | MIT |
| `Jinja2` | 3.1.6 | Pallets | BSD-3-Clause |
| `MarkupSafe` | 3.0.3 | Pallets | BSD-3-Clause |
| `numpy` | 2.5.3 | Travis E. Oliphant et al. | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `packaging` | 26.3 | Donald Stufft | Apache-2.0 OR BSD-2-Clause |
| `pluggy` | 1.6.0 | Holger Krekel | MIT |
| `Pygments` | 2.21.0 | Georg Brandl | BSD-2-Clause |
| `pytest` | 9.1.1 | Holger Krekel, Bruno Oliveira, Ronny Pfannschmidt, Floris Bruynooghe and others | MIT |
| `PyYAML` | 6.0.3 | Kirill Simonov | MIT |
| `safetensors` | 0.8.0 | Nicolas Patry, Luc Georges, Daniël De Kok | Apache-2.0 |
| `tokenizers` | 0.23.2 | Nicolas Patry, Anthony Moi | Apache-2.0 |
| `tqdm` | 4.70.1 | tqdm developers | MPL-2.0 AND MIT |
| `typing-extensions` | 4.16.0 | Guido van Rossum, Jukka Lehtosalo, Łukasz Langa, Michael Lee | PSF-2.0 |

## The ablit source's use conditions

Quoted verbatim from the source's model card:

> - It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
> - **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
>
> By downloading or using this model you acknowledge and accept the above.

You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.
