# Transform inventory

This recipe ships five original transform programs, not patched vLLM files or
unified diffs. `config/patch-contract.json` is the machine-readable authority
for every full source hash, accepted-before hash, exact seam string and
cardinality, forbidden-after seam, and emitted-after hash. The tables below
make the same boundary reviewable without abbreviating any file identity.

## Transaction and refusal contract

`apply_base_pipeline.py` applies the transforms in the fixed order shown
below. It accepts one exact whole-pipeline stage. A fully final tree returns
`ALREADY_APPLIED`; an exact earlier stage applies only the remaining programs.
Every program checks the host-minted OCI receipt, pinned source files, full
before/after hashes, seam cardinalities, and generated syntax before writing.

Each transform first writes exact temporary siblings and rollback copies, then
fsyncs a `PREPARED` journal before the first rename. A later `--apply` completes
or rolls back a partial transaction from observed hashes. `--check` refuses a
journal or orphan artifact. Terminal state is hash-verified before the journal
is removed; interrupted cleanup is safely repeatable. Unknown bytes, mixed
stages, a missing or duplicate seam, source/image/receipt drift, output hash or
syntax failure, unsafe path/symlink/device crossing, and unrecoverable
transaction state all refuse without admitting the tree.

## Source identity and attribution

| Transform | Pinned source identity | Attribution / role |
|---|---|---|
| `apply_tp3_overlay.py` | FlyCockpit commit `9093765c757bd1976372196e44af84a67cf86bad`; `patch_tp3_glm.py` `707493ae99f75283f3740844c4cea9f3e3ebc9f987fbf93573f81f29f337b876`; `exl3.py` `9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6`; `vocab_parallel_embedding.py` `fb09e464673c8fd4e46dd51690f73585cd75dc275f09d984a891582af617fc05`; `parameter.py` `42291061c57ebd9dff40830fbd8998383a49279092b8d34f2047fffa0f394ead`; FlashInfer source `dbcc86be617cc6cc96ce02d8f5c55c76cc785882e3d74b2a146422c814622eae` | FlyCockpit/MiaAI-Lab TP3 and EXL3 technique transfer; local FlashInfer decode+prefill head-pad correction |
| `apply_image_glm_dflash.py` | FlyCockpit commit `9093765c757bd1976372196e44af84a67cf86bad`; `chat_template.jinja` `96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a`; `dflash2_speculator.py` `d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48`; `qwen3_dflash2.py` `81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b`; patch-data hashes `3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258`, `b027f9485c246957708dbef8492529f7de764c29e562dc09017856da100b4d56`, `1835bfbd64fbb5f063a1c9d5ea2d70cc3558312f45d4470a58d00c2e24b3806e`, `f029f0164c888cafe8e30aa6544e229c23a2e8b8b81283de0ffbbc3cb023e848`, `b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d`, `1c9808f28ae3cbe593e435b053a2deb129f8bcd6a6ac305fe6c65d6cf0ed0314`, `0e117f2c8210d674e79d98a34a26e8b5dc6f956bfb566e3cd3a830b37f6e76de`, `14602ea4350bad1eb8a6e76de3e17e2d5ef1229340bcd199351e20334f5e15d7` | FlyCockpit/MiaAI-Lab GLM/Eagle3/DFlash/video/scheduler/reasoning integration. The program parses pinned patch data and reconstructs exact outputs; it does not execute upstream patchers. |
| `apply_kpool_tail.py` | vcruz305 commit `622cb878d66f703c597bd6baaa2423caa1786f99`; patch evidence `d8845fe7e043263f6ac3f063ad581067985b20dd039544d4de32dc1473b3922e` | vcruz305-inspired K-pool hybrid-position and persistent tail-slot correction |
| `apply_kda_mixed.py` | transform evidence `4955f1a1fb47f700cd4e05380ea61ab53693c64773e72993043a875beb449d38` | JSpark3 mixed-output-block KDA construction |
| `apply_kda_fg.py` | transform evidence `38c7c8c4d4361a2b277c4985e4f0648c07c0687da96c8a109c138f4fc6e12ef1` | JSpark3 batched f/g KDA construction and loader mapping |

## Exact target identities and seam gates

Seam cells list the required unique marker and its post-transform count. Full
multiline strings and required pre-transform counts are exact in
`config/patch-contract.json`; changing even whitespace changes the contract
hash and refuses execution.

| Transform | Target | Accepted before SHA-256 | Emitted after SHA-256 | Required post-transform seam cardinality |
|---|---|---|---|---|
| `apply_tp3_overlay.py` | `vllm/models/glm5next/nvidia/model.py` | `de3ed7f413157596d0e59069d1776b3bf59f9af56aa12809d46a4f528f98a06f` | `fadafb34f2749eedb25c8ce2acc9a4d5ec3550615dd6d7bd1453c6b3e7f5156c` | `TP3-HEAD-PAD 64→66` ×1; `TP3-VOCAB-PAD` ×1; `TP3-SHARED-I` ×1; `TP3-SHARED-MLP` ×1 |
| `apply_tp3_overlay.py` | `vllm/model_executor/layers/quantization/exl3.py` | `7ae401f1e38af7d47d11c10df68a180d4beb4c259197afd2c89ab991dce25778` | `9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6` | `_exl3_expert_map_device` ×2; `def _maybe_shard(` ×1 |
| `apply_tp3_overlay.py` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | `b3e8a07296607153424b4b7ca5f75f00dcec1bce0f49e54b5eff6262fdf80201` | `fb09e464673c8fd4e46dd51690f73585cd75dc275f09d984a891582af617fc05` | `self.padding_size * self.tp_size // gcd` ×1 |
| `apply_tp3_overlay.py` | `vllm/model_executor/parameter.py` | `ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510` | `42291061c57ebd9dff40830fbd8998383a49279092b8d34f2047fffa0f394ead` | `def _pad_then_narrow(` ×1; `loaded_weight = _pad_then_narrow(` ×4 |
| `apply_tp3_overlay.py` | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `d665ef2109b0183d48e3541ecd24e9fa8e1dc3e410983bc29b8d997af9d7cd01` | `e8e7e8f0814513af02d4bd6317ea119c6cc1b6c7081a614080c4f36e9b082526` | `_SM120_KERNEL_HEADS =` ×1; `def _kernel_heads(` ×1 |
| `apply_image_glm_dflash.py` | `vllm/models/glm5next/nvidia/model.py` | `fadafb34f2749eedb25c8ce2acc9a4d5ec3550615dd6d7bd1453c6b3e7f5156c` | `6da99f9f192f617c0f2ab9f5c46b77025c2119ad2ac48f60136eb605aab04120` | `self.aux_hidden_state_layers: tuple[int, ...] = ()` ×2 |
| `apply_image_glm_dflash.py` | `vllm/v1/core/kv_cache_utils.py` | `624ea7b0244972cb6c53044588912dfea54f3d6a91661cbc423af27e3b5c4b86` | `6a6ab115ececb94c54e5b11c810a6d75a5592e566a82b3460b24d16557007d80` | `DFlash2 drafter KV: padded slot-share block=%d` ×1; `compact_block = 64` ×1 |
| `apply_image_glm_dflash.py` | `vllm/v1/core/sched/scheduler.py` | `4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a` | `cd0bd6678c0b74a73e49ae78fe86517adc5ef5b136aa96686e1ee99d4a1b691c` | `def _glm53_mixed_prefill_policy(` ×1; `# [glm53-decode-floor]` ×2 |
| `apply_image_glm_dflash.py` | `vllm/v1/engine/detokenizer.py` | `213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93` | `23327a0b21b0ce53dd680fb272c903fb636260b9de9b4463e360e1714d0fdec0` | `def _maybe_enable_reasoning_stop_guard(` ×1; `# [suppress-stops-in-reasoning]` ×4 |
| `apply_image_glm_dflash.py` | `glm53_video_patch.py` | `ABSENT` | `b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d` | `def _construct_video_placeholder` ×1 |
| `apply_image_glm_dflash.py` | `glm53_video.pth` | `ABSENT` | `debd264515f3a30d81d60a6a1f2c69053476073c989e52b728858484350f3276` | `import glm53_video_patch\n` ×1 |
| `apply_image_glm_dflash.py` | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | `f48c5f93cb3c7d1b1238381761b65c142a1c032023dcef8e449a02f651fb0530` | `f48c5f93cb3c7d1b1238381761b65c142a1c032023dcef8e449a02f651fb0530` | `torch.ops._C.persistent_topk(` ×1 |
| `apply_image_glm_dflash.py` | `vllm/model_executor/models/qwen3_dflash2.py` | `81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b` | `81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b` | `class DFlash2Qwen3ForCausalLM` ×1 |
| `apply_image_glm_dflash.py` | `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | `d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48` | `d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48` | `class DFlash2Speculator` ×1 |
| `apply_image_glm_dflash.py` | `vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py` | `6c6ec57de9f82ef42d82760f144a4571cb16b3cd6de33aaa7aa9fd65e8ab1fa1` | `6c6ec57de9f82ef42d82760f144a4571cb16b3cd6de33aaa7aa9fd65e8ab1fa1` | `SPDX-License-Identifier: Apache-2.0` ×1 |
| `apply_image_glm_dflash.py` | `vllm/model_executor/models/qwen3_dflash.py` | `40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745` | `40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745` | `decoder_layer_cls = DFlashQwen3DecoderLayer` ×1 |
| `apply_image_glm_dflash.py` | `vllm/model_executor/models/registry.py` | `72e2d1b1699726ee4570af28ef3f89b7afdb45bf138f122503298e5850044ad5` | `6ab735761d38b9ac6c0a16463227fa0314755fc2fc02754aa33484ed830bca65` | `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),` ×2 |
| `apply_image_glm_dflash.py` | `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | `94106b29446769d71cd82297c7320715bcdefaea9b97341bc166ca3f14088690` | `94106b29446769d71cd82297c7320715bcdefaea9b97341bc166ca3f14088690` | `draft_kv = speculative_config.kv_cache_dtype` ×1 |
| `apply_image_glm_dflash.py` | `vllm/v1/worker/gpu/spec_decode/__init__.py` | `9d0949eb408f25e16d970c40aef8e921dbdace38dcc1e27d49587c841a63c230` | `9d0949eb408f25e16d970c40aef8e921dbdace38dcc1e27d49587c841a63c230` | `if "DFlash2DraftModel"` ×1 |
| `apply_kpool_tail.py` | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | `3d1d3edc157d87f10aa6fb4862fbd25b435ede51499d7c64e68eb713de85e648` | `cc6382b88cf4f66902516366c64dddc5bcd9575a3da2f3746b2fdbbddc388e17` | `positions=input_batch.positions,` ×1 |
| `apply_kpool_tail.py` | `vllm/v1/attention/backends/mla/indexer.py` | `fe1b106466008c21d0194a59909b19832e127fad255965611c4c7f7a971422a2` | `473003cc30cb4d4e250f549982b87d27e5e69f7ae942901be94dc783302382f5` | `out = slot_mapping` ×1 |
| `apply_kda_mixed.py` | `vllm/models/glm5next/nvidia/kda.py` | `ec090aabecc1a63dacc9694ea677b195e95ce0c63648c418a6daaf34b8196125` | `b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231` | `enable_kda_mixed_output_blocks(self)` ×1 |
| `apply_kda_mixed.py` | `vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py` | `ABSENT` | `b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365` | `def enable_kda_mixed_output_blocks(` ×1 |
| `apply_kda_fg.py` | `vllm/models/glm5next/nvidia/kda.py` | `b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231` | `b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d` | `class _Glm5NextBatchedColumnParallelLinear` ×1; `self.fused_fg_b_proj` ×2 |
| `apply_kda_fg.py` | `vllm/models/glm5next/nvidia/model.py` | `6da99f9f192f617c0f2ab9f5c46b77025c2119ad2ac48f60136eb605aab04120` | `ce456edb8e62df26580d7fd5a844bef52862cbfe01eef766bd43f5032cb60577` | `(".fused_fg_b_proj", ".f_b_proj", 0),` ×1; `(".fused_fg_b_proj", ".g_b_proj", 1),` ×1 |
| `apply_kda_fg.py` | `vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py` | `b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365` | `01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5` | `from .kda_mixed_output_blocks_base import (` ×1 |
| `apply_kda_fg.py` | `vllm/model_executor/layers/quantization/kda_mixed_output_blocks_base.py` | `ABSENT` | `b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365` | `def enable_kda_mixed_output_blocks(` ×1 |

The image transform also refuses unless five verify-only identities match:
`vllm/config/model.py` `051537fb0c01468478eb1d49751bc7964bd5c7452b6248286de534ab5082c123`,
quantization `__init__.py` `c49635a0b75c213e8dbf08622701377e979bb1e330b62e5631dcd8fde8bc4139`,
`vllm/config/compilation.py` `897510f38563db5392480ce2832858a9802ef200ab51f1c099462f20e1a6e6b6`,
`vllm/v1/worker/gpu/cudagraph_utils.py` `c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f`,
and `/opt/glm53/chat_template.jinja`
`96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a`.

Original package code and prose are Apache-2.0. That license does not relicense
vLLM, the model/draft, the OCI image, FlyCockpit, or any upstream technique.
