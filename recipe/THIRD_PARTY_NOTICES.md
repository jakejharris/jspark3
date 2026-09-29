# Third-party notices

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

The recipe references, but does not redistribute, Z.AI GLM-5.3;
Brandon M. Music's ShapleyMcg EXL3/TR3 quant (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`
revision `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`), fetched from Mia-AiLab's
byte-identical re-host; Inco AI DFlash2 and `z-lab/dflash`; vLLM (Apache-2.0);
exllamav3; MiaAI-Lab's image/recipe and its qualification helper at revision
`2fb09425ab644f30c6b61fe3da2d567b99b36464`; FlyCockpit revision
`9093765c757bd1976372196e44af84a67cf86bad`; sfxnz revision
`f59cb7cc41c3ee32146782900d87133f538b5d30`; and Tony scheduler context
`3eef46632c45ffb6c397de0716c23b3d2d594798`. Review every upstream license.

The DFlash2 checkpoint declares CC BY-NC-ND 4.0 and research/evaluation use.
The default DFlash2 path is non-commercial research/evaluation only;
commercial use requires separate permission from Inco AI. Apache-2.0 in this
tree applies only to original package code and prose and does not relicense any
checkpoint, image, or upstream source.

## Source included under other licenses

Apache-2.0 in this repository applies only to original JSpark3 code and prose. The components below
are included under their own licenses. Per-file terms are recorded by SPDX lines or REUSE annotations, and license texts sit in
`third_party/licenses/` and beside each component. The AGPL-3.0 components load into the same serving
process as the Apache-2.0 code, so the recipe as assembled and run is not represented as wholly
Apache-2.0. Original Apache-2.0 files remain available under Apache-2.0 on their own.

- MiaAI-Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks (AGPL-3.0; contributions made before 2026-09-07 keep
  the MIT notice retained in `LICENSE.MIT`):
  - revision `357fce7a976fb835740916151edf2f9325384482`: DFlash SWA corrections by ratulsarna
    (PR #130) (`recipe/overlays/patch_apc_per_group_retention.py`);
  - revision `bc68f310f8d5e941227ce5c93e95bca43b50fd6c` (PR #202): cooperative MoE for TP3/EP
    (`recipe/overlays/v16/coop/`), adapted for this recipe; the kernel declares ExLlamaV3 origin
    `58d4d732`;
  - commit `77ef2f3f8a911788a599bc5498d2088525ea5a5a` from lilianmoraru's open PR #203: indexer
    warmup range (`recipe/overlays/v14/patch_indexer_warmup_range.py`);
  - revision `7cded8e2ed9d7502f32591cfeeeea4747db3d393`: drafter group, KV capacity log (nood-co1,
    PR #94), Mamba-align state release, and the decode-floor scheduler and prefill chunker that
    this recipe extends, with plotarmordev's changes from PRs #198 and #238
    (`recipe/overlays/patch_mamba_align_state_free.py`, `recipe/overlays/v14/`);
  - revision `f4970207e9fb2bdeac40d88b7cbef18c98aea310` (plotarmordev, PR #251): fine-grained hybrid
    prefix hits (`recipe/overlays/patch_hybrid_prefix_hit.py`), with this recipe's replay-window fix
    (`recipe/overlays/patch_dflash_fine_replay_window.py`);
  - revision `7cded8e2ed9d7502f32591cfeeeea4747db3d393`: the xgrammar termination patch by knapcio
    (PR #21) (`recipe/overlays/v16/grammar_fsm/`), which carries fixes from vLLM pull requests
    #53046 and #52805.
- coolbho3k, DeepSeek-v4.1-Flash-2x-DGX-Spark (AGPL-3.0), revision
  `878e0eecd893fadc69ad2d58b2df0fabb0fae2ee`: display-reserve KV backing
  (`release/runtime/sources/display_kv.c`), as adapted for GLM-5.3 in MiaAI-Lab pull request #234 by
  gabewillen (head `cd504b691816623fc12a840483342a987b30ff73`), with this recipe's changes;
  `recipe/overlays/v14/display_kv/`. The shared library is built from this source by `build.sh`; no
  binary is distributed.
- Turboderp, ExLlamaV3 (MIT): nine headers pinned at `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`,
  vendored unmodified under `recipe/overlays/v16/coop/`. The fat-path epilogue in
  `recipe/overlays/v15/` reproduces ExLlamaV3's output Hadamard arithmetic.
- FlyCockpit (MIT), revision `9093765c757bd1976372196e44af84a67cf86bad`: the TP3 overlay
  transforms (`recipe/scripts/apply_tp3_overlay.py`, `recipe/scripts/apply_image_glm_dflash.py`)
  build this recipe's TP3 path from FlyCockpit's pinned sources, and the first reproduces short
  excerpts of them; the fat-path patch edits FlyCockpit's `exl3.py` in place. FlyCockpit's MIT
  notice is retained alongside Mia's.
- Victor Cruz (vcruz305), GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe (MIT at this revision), revision
  `622cb878d66f703c597bd6baaa2423caa1786f99`: the K-pool tail correction in
  `recipe/scripts/apply_kpool_tail.py` reproduces the replacement text of
  `scripts/patch_kpool_tail_positions.py`. Its MIT notice is retained in
  `third_party/licenses/vcruz305-MIT.txt`.
- vLLM (Apache-2.0): the kpool seed-stride fix reproduces `vllm-project/vllm@db1bfdd4` (PR #57477).
- InstantTensor 0.2.0 (Apache-2.0) is installed into the serving image by
  `docker/stock-v13/Dockerfile`. Its native module statically links libaio (LGPL-2.1) and liburing
  under their own terms. This repository does not distribute that image or module.

If you modify an AGPL-3.0 component and let users interact with the modified service over a network,
section 13 of the AGPL-3.0 requires you to offer those users its Corresponding Source.

## Optional edited-behavior mode

The optional edited-behavior mode (`ABLIT=1`) reads attention output projections from a donor
checkpoint that the operator obtains separately, at immutable revision
`80b6d18d77e3020f2384597081d405f19893f101`. Its model card declares MIT and pins its base-model
license. No donor checkpoint, donor tensor, or stock-weight export is redistributed here.
