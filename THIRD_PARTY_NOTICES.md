# Third-party notices

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

The recipe references, but does not redistribute, Z.AI GLM-5.3;
Mia-AiLab/Brandon M. Music's ShapleyMcg EXL3/TR3 quant; Inco AI DFlash2 and
`z-lab/dflash`; vLLM (Apache-2.0); exllamav3; MiaAI-Lab's image/recipe;
FlyCockpit revision `9093765c757bd1976372196e44af84a67cf86bad`; vcruz305
revision `622cb878d66f703c597bd6baaa2423caa1786f99`; sfxnz revision
`f59cb7cc41c3ee32146782900d87133f538b5d30`; and Tony scheduler context
`3eef46632c45ffb6c397de0716c23b3d2d594798`. Review every upstream license.

The DFlash2 checkpoint declares CC BY-NC-ND 4.0 and research/evaluation use.
The default DFlash2 path is non-commercial research/evaluation only;
commercial use requires separate permission from Inco AI. Apache-2.0 in this
tree applies only to original package code and prose and does not relicense any
checkpoint, image, or upstream source.


## v2.0.1 attribution correction (2026-10-03)

The notices above describe the historical v1 recipe and are preserved. The
following supplement credits the code and ideas used by v2.0.1. It does not
change any component's license or the bytes of the v2.0.1 tag.

The complete v2.0.1 component, author, license and pinned-dependency inventories
remain in the [tagged root notices](https://github.com/jakejharris/jspark3/blob/v2.0.1/THIRD_PARTY_NOTICES.md),
[tagged engine notices](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/THIRD_PARTY_NOTICES.md)
and [tagged engine NOTICE](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/NOTICE).
Read those inventories together with this attribution correction; their tagged
copies predate it. All existing authors, license texts and pins are retained.

### Credits

- Z.AI (GLM-5.3 Flash, the base model, and its chat template)
- Hugging Face and the transformers contributors (the GLM-5.3 Flash model code the engine's CUDA path implements)
- Vontra, now TensorFold on Hugging Face (the 4-bit MLX base weights)
- Ash Hart and the TensorFold contributors (TensorFold 0.3.6.2, the engine release this project forks; DFlash ring-snapshot follow-up 47bf822)
- Taus Soe (GLM multi-stream foundation 20dbaba, disk-chain foundation b8a555a/bb16122 and CUDA image input 19680d9, via taussoe/TensorFold)
- FlyCockpit (zero-padding dimensions for three-way splitting, credited since v1.0.0)
- BTCXoomer (reporting the three-Spark NCCL subnet-routing requirement, credited in v1.1.0)
- Inco AI (the DFlash2 draft model)
- orcarouter (the refusal-removed source weights)
- z-lab (DFlash)
- MiaAI-Lab (upstream TensorFold: follower doorbell 358875c, DFlash ring 7c088eb, prefill row-blocking b23c10a and typed-parser hunk fe2b514)
- mikolaj92 (visible-pool and radix-selection optimizations, via upstream TensorFold commits b3b8a39 and f119334)
- Dorian (an upstream TensorFold server fix, ported)
- turboderp (ExLlamaV3's EXL3 format, which the engine's own decoders read)
- QTIP and QuIP# authors (trellis and incoherence-processing foundations used through ExLlamaV3's EXL3 format)
- Apple (MLX)
- Google DeepMind (Gemma 4, supported by the vendored engine's MLX backend)
- Prince Canuma and the mlx-vlm contributors (GLM-5.3 Flash code the engine follows and ports)

### Contribution-specific provenance

In this table, `G/` means `src/tensorfold/families/glm5_next/cuda/` and `T/`
means `src/tensorfold/`, relative to the tagged `engine/` directory. FlyCockpit
is credited for the padding technique, BTCXoomer for reporting the routing
requirement. The QTIP/QuIP# and Google DeepMind entries concern inherited
engine features, not the supported three-Spark GLM recipe's weights or models.
The existing follower-doorbell and Dorian notices remain in the linked engine
inventory. Taus Soe's foundations are distinct from JSpark3's later extensions.

| Contributor / contribution | Source and license-era evidence | Shipped locus and boundary |
|---|---|---|
| FlyCockpit: three-way zero-padding technique | `9093765c757bd1976372196e44af84a67cf86bad`; the v1.8.4 notices identify the source as MIT | `G/split.py`, `G/weights.py`; technique credit, separately implemented in this splitter |
| BTCXoomer: report of the three-Spark NCCL subnet-routing requirement | v1.1.0 README; reporting credit, no new imported source license | Root `scripts/fabric/launch.py`; the recipe remains Apache-2.0 |
| MiaAI-Lab: DFlash ring/window adaptation | Upstream TensorFold `7c088eb`, historical MIT | `G/dflash2.py`, `G/decode.py`, `G/batched.py`, `T/cuda/geometry.py`; local pooled integration around upstream ring logic |
| MiaAI-Lab: row-blocked prefill | Upstream TensorFold `b23c10a`, historical MIT | `G/forward.py`, `G/latent.py`, `G/sparse.py`, `T/cuda/geometry.py`; row/block/scratch adaptation |
| MiaAI-Lab: typed-tool-parameter parsing hunk | Upstream TensorFold `fe2b514`, historical MIT; exact overlap verified, direct derivation inferred | `T/cuda/reply_text.py:109–112` and schema/import context; attribution covers that hunk, not the entire parser/streamer |
| mikolaj92: visible-pool skip and bounded radix selection | Upstream TensorFold `b3b8a39`, `f119334`, historical MIT | `G/sparse.py`; local integration with prefill controls |
| Ash Hart: DFlash saved-ring follow-up | Upstream TensorFold `47bf822`, historical MIT | `G/decode.py`; consulted follow-up to the ring adaptation, not a separate wholesale module port |
| Taus Soe: multi-stream primitives, disk-chain design, CUDA image preprocessing and tower | `taussoe/TensorFold` commits `20dbaba`, `b8a555a`/`bb16122`, `19680d9`, respectively; historical MIT, TensorFold contributors | `G/multi.py`, `G/forward.py`, `G/disk.py`, `G/session_disk.py`, `G/vision.py` and hooks; JSpark3 extends these foundations with its pooled scheduler, session format, direct I/O, request limits and rank integration |
| Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa and the QTIP/QuIP# authors: trellis and incoherence-processing foundations | Intellectual lineage through ExLlamaV3's EXL3 format; no newly verified direct paper/code port or separate license assignment | `G/exl3.py`; existing ExLlamaV3 MIT format/math notice retained; the three-Spark GLM recipe uses MLX weights |
| Google DeepMind: Gemma 4 model authorship | Historical author/model credit; no Gemma weights bundled and no relicensing of external weights | `T/families/gemma4/model.py`; inherited MLX backend using the credited mlx-lm dependency, outside this release's qualified GLM/CUDA recipe |
