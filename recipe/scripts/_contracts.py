"""Embedded copy of the exact machine contract each transform must match."""

from __future__ import annotations


def target(
    path: str,
    before: str,
    after: str,
    source: str,
    before_seams: list[tuple[str, int]],
    after_seams: list[tuple[str, int]],
    forbidden: list[str] | None = None,
    **extra: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path,
        "before_sha256": before,
        "after_sha256": after,
        "source_sha256": source,
        "required_before_seams": [
            {"text": text, "count": count} for text, count in before_seams
        ],
        "required_after_seams": [
            {"text": text, "count": count} for text, count in after_seams
        ],
        "forbidden_after_seams": forbidden or [],
    }
    value.update(extra)
    return value


TP3 = {
    "sources": {
        "fly_commit": "9093765c757bd1976372196e44af84a67cf86bad",
        "patch_tp3_glm.py": "707493ae99f75283f3740844c4cea9f3e3ebc9f987fbf93573f81f29f337b876",
        "exl3.py": "9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6",
        "vocab_parallel_embedding.py": "fb09e464673c8fd4e46dd51690f73585cd75dc275f09d984a891582af617fc05",
        "parameter.py": "42291061c57ebd9dff40830fbd8998383a49279092b8d34f2047fffa0f394ead",
        "flashinfer_mla_sparse_sm120.py": "dbcc86be617cc6cc96ce02d8f5c55c76cc785882e3d74b2a146422c814622eae",
    },
    "targets": [
        target(
            "vllm/models/glm5next/nvidia/model.py",
            "de3ed7f413157596d0e59069d1776b3bf59f9af56aa12809d46a4f528f98a06f",
            "fadafb34f2749eedb25c8ce2acc9a4d5ec3550615dd6d7bd1453c6b3e7f5156c",
            "707493ae99f75283f3740844c4cea9f3e3ebc9f987fbf93573f81f29f337b876",
            [("        config = vllm_config.model_config.hf_config\n        self.config = config\n", 1)],
            [
                ("# TP3-HEAD-PAD 64→66", 1),
                ("# TP3-VOCAB-PAD", 1),
                ("# TP3-SHARED-I", 1),
                ("# TP3-SHARED-MLP", 1),
            ],
        ),
        target(
            "vllm/model_executor/layers/quantization/exl3.py",
            "7ae401f1e38af7d47d11c10df68a180d4beb4c259197afd2c89ab991dce25778",
            "9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6",
            "9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6",
            [("layer.expert_map = emap.to(device=device, dtype=torch.long)", 1)],
            [("_exl3_expert_map_device", 2), ("def _maybe_shard(", 1)],
        ),
        target(
            "vllm/model_executor/layers/vocab_parallel_embedding.py",
            "b3e8a07296607153424b4b7ca5f75f00dcec1bce0f49e54b5eff6262fdf80201",
            "fb09e464673c8fd4e46dd51690f73585cd75dc275f09d984a891582af617fc05",
            "fb09e464673c8fd4e46dd51690f73585cd75dc275f09d984a891582af617fc05",
            [("        self.padding_size = padding_size\n", 1)],
            [("self.padding_size * self.tp_size // gcd", 1)],
        ),
        target(
            "vllm/model_executor/parameter.py",
            "ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510",
            "42291061c57ebd9dff40830fbd8998383a49279092b8d34f2047fffa0f394ead",
            "42291061c57ebd9dff40830fbd8998383a49279092b8d34f2047fffa0f394ead",
            [("loaded_weight.narrow", 4)],
            [("def _pad_then_narrow(", 1), ("loaded_weight = _pad_then_narrow(", 4)],
        ),
        target(
            "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
            "d665ef2109b0183d48e3541ecd24e9fa8e1dc3e410983bc29b8d997af9d7cd01",
            "e8e7e8f0814513af02d4bd6317ea119c6cc1b6c7081a614080c4f36e9b082526",
            "dbcc86be617cc6cc96ce02d8f5c55c76cc785882e3d74b2a146422c814622eae",
            [("class FlashInferMLASparseSM120Impl", 1)],
            [("_SM120_KERNEL_HEADS =", 1), ("def _kernel_heads(", 1)],
            ["_SM120_DECODE_MAX_TOKENS", "def _decode_kernel_heads("],
            intermediate_sha256="dbcc86be617cc6cc96ce02d8f5c55c76cc785882e3d74b2a146422c814622eae",
        ),
    ],
}


IMAGE_DFLASH = {
    "sources": {
        "fly_commit": "9093765c757bd1976372196e44af84a67cf86bad",
        "chat_template.jinja": "96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a",
        "dflash2_speculator.py": "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48",
        "qwen3_dflash2.py": "81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b",
        "patch_dflash2.py": "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
        "patch_exl3_ext_aarch64.py": "b027f9485c246957708dbef8492529f7de764c29e562dc09017856da100b4d56",
        "patch_glm5_drafter_group.py": "1835bfbd64fbb5f063a1c9d5ea2d70cc3558312f45d4470a58d00c2e24b3806e",
        "patch_glm_eagle3.py": "f029f0164c888cafe8e30aa6544e229c23a2e8b8b81283de0ffbbc3cb023e848",
        "patch_glm_video_placeholders.py": "b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d",
        "patch_model_overrides.py": "1c9808f28ae3cbe593e435b053a2deb129f8bcd6a6ac305fe6c65d6cf0ed0314",
        "patch_scheduler_decode_floor.py": "0e117f2c8210d674e79d98a34a26e8b5dc6f956bfb566e3cd3a830b37f6e76de",
        "patch_suppress_stops_in_reasoning.py": "14602ea4350bad1eb8a6e76de3e17e2d5ef1229340bcd199351e20334f5e15d7",
    },
    "verify_only": [
        {"path": "vllm/config/model.py", "sha256": "051537fb0c01468478eb1d49751bc7964bd5c7452b6248286de534ab5082c123"},
        {"path": "vllm/model_executor/layers/quantization/__init__.py", "sha256": "c49635a0b75c213e8dbf08622701377e979bb1e330b62e5631dcd8fde8bc4139"},
        {"path": "vllm/config/compilation.py", "sha256": "897510f38563db5392480ce2832858a9802ef200ab51f1c099462f20e1a6e6b6"},
        {"path": "vllm/v1/worker/gpu/cudagraph_utils.py", "sha256": "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f"},
        {"asset_path": "chat_template.jinja", "sha256": "96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a"},
    ],
    "targets": [
        target(
            "vllm/models/glm5next/nvidia/model.py",
            "fadafb34f2749eedb25c8ce2acc9a4d5ec3550615dd6d7bd1453c6b3e7f5156c",
            "6da99f9f192f617c0f2ab9f5c46b77025c2119ad2ac48f60136eb605aab04120",
            "f029f0164c888cafe8e30aa6544e229c23a2e8b8b81283de0ffbbc3cb023e848",
            [("self.aux_hidden_state_layers: tuple[int, ...] = ()", 1)],
            [("self.aux_hidden_state_layers: tuple[int, ...] = ()", 2)],
        ),
        target(
            "vllm/v1/core/kv_cache_utils.py",
            "624ea7b0244972cb6c53044588912dfea54f3d6a91661cbc423af27e3b5c4b86",
            "6a6ab115ececb94c54e5b11c810a6d75a5592e566a82b3460b24d16557007d80",
            "1835bfbd64fbb5f063a1c9d5ea2d70cc3558312f45d4470a58d00c2e24b3806e",
            [("# STANDALONE: the drafter's geometry cannot exactly fill the MLA", 1)],
            [("DFlash2 drafter KV: padded slot-share block=%d", 1), ("compact_block = 64", 1)],
        ),
        target(
            "vllm/v1/core/sched/scheduler.py",
            "4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a",
            "cd0bd6678c0b74a73e49ae78fe86517adc5ef5b136aa96686e1ee99d4a1b691c",
            "0e117f2c8210d674e79d98a34a26e8b5dc6f956bfb566e3cd3a830b37f6e76de",
            [("from vllm.compilation.cuda_graph import CUDAGraphStat", 1)],
            [("def _glm53_mixed_prefill_policy(", 1), ("# [glm53-decode-floor]", 2)],
        ),
        target(
            "vllm/v1/engine/detokenizer.py",
            "213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93",
            "23327a0b21b0ce53dd680fb272c903fb636260b9de9b4463e360e1714d0fdec0",
            "14602ea4350bad1eb8a6e76de3e17e2d5ef1229340bcd199351e20334f5e15d7",
            [("class IncrementalDetokenizer:", 1)],
            [("def _maybe_enable_reasoning_stop_guard(", 1), ("# [suppress-stops-in-reasoning]", 4)],
        ),
        target(
            "glm53_video_patch.py",
            "ABSENT",
            "b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d",
            "b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d",
            [],
            [("def _construct_video_placeholder", 1)],
        ),
        target(
            "glm53_video.pth",
            "ABSENT",
            "debd264515f3a30d81d60a6a1f2c69053476073c989e52b728858484350f3276",
            "b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d",
            [],
            [("import glm53_video_patch\n", 1)],
        ),
        target(
            "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
            "f48c5f93cb3c7d1b1238381761b65c142a1c032023dcef8e449a02f651fb0530",
            "00e32052b781723500987a463814116634c4d00b3b61066915f8cc70780c931e",
            "b41f87832968a63000c9b56ac12948958ad36d1d0f93c031a2969243031aa82d",
            [
                ("torch.ops._C.persistent_topk(", 1),
                ("if current_platform.is_cuda() and select_k in (512, 1024, 2048):", 1),
            ],
            [
                ("torch.ops._C.persistent_topk(", 1),
                ("if False and current_platform.is_cuda() and select_k in (512, 1024, 2048):  # GB10 persistent_topk smem", 1),
            ],
            ["if current_platform.is_cuda() and select_k in (512, 1024, 2048):"],
        ),
        target(
            "vllm/model_executor/models/qwen3_dflash2.py",
            "81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b",
            "81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b",
            "81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b",
            [("class DFlash2Qwen3ForCausalLM", 1)],
            [("class DFlash2Qwen3ForCausalLM", 1)],
        ),
        target(
            "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
            "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48",
            "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48",
            "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48",
            [("class DFlash2Speculator", 1)],
            [("class DFlash2Speculator", 1)],
        ),
        target(
            "vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py",
            "6c6ec57de9f82ef42d82760f144a4571cb16b3cd6de33aaa7aa9fd65e8ab1fa1",
            "6c6ec57de9f82ef42d82760f144a4571cb16b3cd6de33aaa7aa9fd65e8ab1fa1",
            "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
            [("SPDX-License-Identifier: Apache-2.0", 1)],
            [("SPDX-License-Identifier: Apache-2.0", 1)],
        ),
        target(
            "vllm/model_executor/models/qwen3_dflash.py",
            "40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745",
            "40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745",
            "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
            [("decoder_layer_cls = DFlashQwen3DecoderLayer", 1)],
            [("decoder_layer_cls = DFlashQwen3DecoderLayer", 1)],
        ),
        target(
            "vllm/model_executor/models/registry.py",
            "72e2d1b1699726ee4570af28ef3f89b7afdb45bf138f122503298e5850044ad5",
            "6ab735761d38b9ac6c0a16463227fa0314755fc2fc02754aa33484ed830bca65",
            "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
            [("\"DFlash2DraftModel\": (\"qwen3_dflash2\", \"DFlash2Qwen3ForCausalLM\"),", 1)],
            [("\"DFlash2DraftModel\": (\"qwen3_dflash2\", \"DFlash2Qwen3ForCausalLM\"),", 2)],
            intermediate_sha256="72e2d1b1699726ee4570af28ef3f89b7afdb45bf138f122503298e5850044ad5",
        ),
        target(
            "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
            "94106b29446769d71cd82297c7320715bcdefaea9b97341bc166ca3f14088690",
            "94106b29446769d71cd82297c7320715bcdefaea9b97341bc166ca3f14088690",
            "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
            [("draft_kv = speculative_config.kv_cache_dtype", 1)],
            [("draft_kv = speculative_config.kv_cache_dtype", 1)],
        ),
        target(
            "vllm/v1/worker/gpu/spec_decode/__init__.py",
            "9d0949eb408f25e16d970c40aef8e921dbdace38dcc1e27d49587c841a63c230",
            "9d0949eb408f25e16d970c40aef8e921dbdace38dcc1e27d49587c841a63c230",
            "3559f69c9c998873a7adf1064e0636c346bf9516235ac09a0640e725d0aa0258",
            [("if \"DFlash2DraftModel\"", 1)],
            [("if \"DFlash2DraftModel\"", 1)],
        ),
    ],
}


KPOOL = {
    "sources": {
        "vcruz_commit": "622cb878d66f703c597bd6baaa2423caa1786f99",
        "patch_kpool_tail_positions.py": "d8845fe7e043263f6ac3f063ad581067985b20dd039544d4de32dc1473b3922e",
    },
    "targets": [
        target(
            "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
            "3d1d3edc157d87f10aa6fb4862fbd25b435ede51499d7c64e68eb713de85e648",
            "cc6382b88cf4f66902516366c64dddc5bcd9575a3da2f3746b2fdbbddc388e17",
            "d8845fe7e043263f6ac3f063ad581067985b20dd039544d4de32dc1473b3922e",
            [("            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,\n", 1)],
            [("            positions=input_batch.positions,\n", 1)],
        ),
        target(
            "vllm/v1/attention/backends/mla/indexer.py",
            "fe1b106466008c21d0194a59909b19832e127fad255965611c4c7f7a971422a2",
            "473003cc30cb4d4e250f549982b87d27e5e69f7ae942901be94dc783302382f5",
            "d8845fe7e043263f6ac3f063ad581067985b20dd039544d4de32dc1473b3922e",
            [("    out = slot_mapping.clone()\n", 1)],
            [("    out = slot_mapping\n", 1)],
            ["    out = slot_mapping.clone()\n"],
        ),
    ],
}


KDA_MIXED = {
    "sources": {"transform_evidence_sha256": "4955f1a1fb47f700cd4e05380ea61ab53693c64773e72993043a875beb449d38"},
    "targets": [
        target(
            "vllm/models/glm5next/nvidia/kda.py",
            "ec090aabecc1a63dacc9694ea677b195e95ce0c63648c418a6daaf34b8196125",
            "b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231",
            "4955f1a1fb47f700cd4e05380ea61ab53693c64773e72993043a875beb449d38",
            [("from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention\n", 1)],
            [("enable_kda_mixed_output_blocks(self)", 1)],
        ),
        target(
            "vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py",
            "ABSENT",
            "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365",
            "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365",
            [],
            [("def enable_kda_mixed_output_blocks(", 1)],
        ),
    ],
}


KDA_FG = {
    "sources": {"transform_evidence_sha256": "38c7c8c4d4361a2b277c4985e4f0648c07c0687da96c8a109c138f4fc6e12ef1"},
    "targets": [
        target(
            "vllm/models/glm5next/nvidia/kda.py",
            "b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231",
            "b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d",
            "38c7c8c4d4361a2b277c4985e4f0648c07c0687da96c8a109c138f4fc6e12ef1",
            [("from vllm.distributed import divide\n", 1)],
            [("class _Glm5NextBatchedColumnParallelLinear", 1), ("self.fused_fg_b_proj", 2)],
        ),
        target(
            "vllm/models/glm5next/nvidia/model.py",
            "6da99f9f192f617c0f2ab9f5c46b77025c2119ad2ac48f60136eb605aab04120",
            "ce456edb8e62df26580d7fd5a844bef52862cbfe01eef766bd43f5032cb60577",
            "38c7c8c4d4361a2b277c4985e4f0648c07c0687da96c8a109c138f4fc6e12ef1",
            [("(\".in_proj_qkvbfg_a\", \".g_a_proj\", 5),", 1)],
            [("(\".fused_fg_b_proj\", \".f_b_proj\", 0),", 1), ("(\".fused_fg_b_proj\", \".g_b_proj\", 1),", 1)],
        ),
        target(
            "vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py",
            "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365",
            "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5",
            "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5",
            [("def enable_kda_mixed_output_blocks(", 1)],
            [("from .kda_mixed_output_blocks_base import (", 1)],
        ),
        target(
            "vllm/model_executor/layers/quantization/kda_mixed_output_blocks_base.py",
            "ABSENT",
            "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365",
            "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365",
            [],
            [("def enable_kda_mixed_output_blocks(", 1)],
        ),
    ],
}


# --- BEGIN v1.4 sections (generated by tools/v14/build_v14_contracts.py) ---


V14_CORE = {
    "sources": {
        "patch_glm5_drafter_group.py": "504087b2fb6fb615e242d95a5f4e14b7cb8d516148b6bfd056ee822a0daca402",
        "patch_scheduler_decode_floor.py": "5884f118ae45795b85e40069aff71147e102bfed4beb6936c29dbad5d97f0303",
        "patch_mamba_align_chunking.py": "0197ac174d9a6bffcc89ca12f93861cac229143c1c6c9a076297f62e9ce2ee04",
        "patch_kv_capacity_log.py": "3c1587e41f8e94f3420d817e223fcb46d04cb8aebf08dba8f943b94f2fef9f94",
        "patch_indexer_warmup_range.py": "7b413f0436b736acdcca25e16f4aa926621aa92dd2d407f5c13e5be4f327dc3e",
        "patch_kpool_seed_stride.py": "25f1e5c801aa4ad4530d6a11fe761039aa9ec712fb1b5f0d266e4dc5e6cd35c0",
        "mia_commit": "7cded8e2ed9d7502f32591cfeeeea4747db3d393",
        "mia_pr203_head": "77ef2f3f",
        "mia_pr234_head": "cd504b69",
        "mia_pr246_head": "dc702b4f",
        "vllm_pr57477_commit": "db1bfdd4",
    },
    "targets": [
        target(
            "vllm/v1/core/kv_cache_utils.py",
            "6a6ab115ececb94c54e5b11c810a6d75a5592e566a82b3460b24d16557007d80",
            "11c756125b120863aae90220d49188a166063e4eef8858fe02feb5fc51d660b1",
            "3c1587e41f8e94f3420d817e223fcb46d04cb8aebf08dba8f943b94f2fef9f94",
            [("DFlash2 drafter KV: padded slot-share block=%d", 1), ("compact_block = 64", 1)],
            [("def _glm53_draft_block_size(", 1), ("def _glm53_draft_kv_compact(", 1), ("# [glm53-kv-capacity-log] helpers", 1), ("_glm53_log_kv_capacity(vllm_config, kv_cache_config, max_model_len)  # [glm53-kv-capacity-log]", 1)],
            ["compact_block = 64"],
            installers=["patch_glm5_drafter_group.py", "patch_kv_capacity_log.py"],
        ),
        target(
            "vllm/v1/worker/utils.py",
            "3dcd6ad34ee1d1db2875f7f7dd51d90ee0e64041ab282180687770a38b26acb1",
            "6c754c4afe74796010cb730fbc9333c6c7116edf8e66db1ffd32aee3d26727ec",
            "504087b2fb6fb615e242d95a5f4e14b7cb8d516148b6bfd056ee822a0daca402",
            [("    MambaSpec,\n    UniformTypeKVCacheSpecs,", 1)],
            [("    MambaSpec,\n    SlidingWindowSpec,\n    UniformTypeKVCacheSpecs,", 1)],
            [],
            installers=["patch_glm5_drafter_group.py"],
        ),
        target(
            "vllm/v1/core/sched/scheduler.py",
            "cd0bd6678c0b74a73e49ae78fe86517adc5ef5b136aa96686e1ee99d4a1b691c",
            "0b086dd3cc0febc4924202ca1fef8809b8290b9a59ef15fa80004cef4edeb937",
            "0197ac174d9a6bffcc89ca12f93861cac229143c1c6c9a076297f62e9ce2ee04",
            [("def _glm53_mixed_prefill_policy(", 1), ("# [glm53-decode-floor]", 2)],
            [("# [cadence-sched:v6]", 2), ("_GLM53_MIXED.on_scheduler_init(self)  # [cadence-sched:v6]", 1), ("# [glm53-decode-floor:v5]", 16), ("def _glm53_mixed_prefill_policy(", 1), ("# [glm53-mamba-align-chunking-v1]", 2)],
            ["# [glm53-decode-floor]"],
            installers=["patch_scheduler_decode_floor.py", "patch_mamba_align_chunking.py"],
        ),
        target(
            "vllm/v1/attention/backends/mla/indexer.py",
            "473003cc30cb4d4e250f549982b87d27e5e69f7ae942901be94dc783302382f5",
            "4b45bb8573f480e7586d9d3bfab54474f873b700d6754a21987d3910db399c86",
            "7b413f0436b736acdcca25e16f4aa926621aa92dd2d407f5c13e5be4f327dc3e",
            [("            query_slice_start=WarmupIntRange(0, 2),\n", 1)],
            [("# [glm53-indexer-warmup-range] WarmupIntRange stop is exclusive:", 1), ("            query_slice_start=WarmupIntRange(0, 3),\n", 1)],
            ["            query_slice_start=WarmupIntRange(0, 2),\n"],
            installers=["patch_indexer_warmup_range.py"],
        ),
        target(
            "vllm/models/glm5next/nvidia/ops/kpool_compress.py",
            "6f77204e4a3bbe1efa9f0921b93b2d56e47e47bc9111ed9174c18126cbfdcdff",
            "8da850460cfe91a57fbd116985e07be5254778c3069f523ffe75f0dd1152936b",
            "25f1e5c801aa4ad4530d6a11fe761039aa9ec712fb1b5f0d266e4dc5e6cd35c0",
            [("    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n", 1)],
            [("    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n", 1), ("        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n", 2)],
            ["    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n"],
            installers=["patch_kpool_seed_stride.py"],
        ),
    ],
}


V14_DISPLAY = {
    "sources": {
        "patch_display_kv.py": "682cc29c77799f256bd9d672627cf8945c7a80648272e74a18cf1ee1546a5c3b",
        "glm53_display_kv.py": "384c00af49aae77bcdfeafd870b413dce4c285b8673ad781eeeee26aab851545",
    },
    "targets": [
        target(
            "vllm/v1/worker/gpu_worker.py",
            "b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1",
            "c1aa70809364f4cc2fef76baf5b86aa8e9107a48a8d83982769eb0a3ea0b2fab",
            "682cc29c77799f256bd9d672627cf8945c7a80648272e74a18cf1ee1546a5c3b",
            [],
            [("# [glm53-display-kv]", 1)],
            [],
            installers=["patch_display_kv.py"],
        ),
        target(
            "vllm/v1/worker/gpu_model_runner.py",
            "5be64e11d25f9802a1970089e9203ae4b18977b340471a6d0761f0274b54c4b6",
            "c3469e422154de2140e3a3070e5c2b4ef8098dc49e0887d810c4e5a6ec7909a8",
            "682cc29c77799f256bd9d672627cf8945c7a80648272e74a18cf1ee1546a5c3b",
            [],
            [("# [glm53-display-kv]", 3), ("_glm53_dkv.identity(kv_cache_config)", 1)],
            [],
            installers=["patch_display_kv.py"],
        ),
        target(
            "vllm/v1/worker/gpu/attn_utils.py",
            "1dd3dd2826a2cc73005e7baecb71c26de8d56285b35d780716ab11ffe0f8495b",
            "1d976c2b72450060fb7e8c29d9772dad01f5b573a7f6a099d12b5940a3ab232b",
            "682cc29c77799f256bd9d672627cf8945c7a80648272e74a18cf1ee1546a5c3b",
            [],
            [("# [glm53-display-kv]", 1), ("_glm53_dkv.identity(kv_cache_config)", 1)],
            [],
            installers=["patch_display_kv.py"],
        ),
        target(
            "vllm/v1/worker/kv_connector_model_runner_mixin.py",
            "4b3d86125114857a6af0e60b2f37a338f75df497967b8416b719bb0a2af24c09",
            "478b522bea419fd3c2202f5b75f0b1ca2aaf7313ffab395788bc532758e0ecd4",
            "682cc29c77799f256bd9d672627cf8945c7a80648272e74a18cf1ee1546a5c3b",
            [],
            [("# [glm53-display-kv]", 1)],
            [],
            installers=["patch_display_kv.py"],
        ),
        target(
            "vllm/v1/worker/glm53_display_kv.py",
            "ABSENT",
            "384c00af49aae77bcdfeafd870b413dce4c285b8673ad781eeeee26aab851545",
            "384c00af49aae77bcdfeafd870b413dce4c285b8673ad781eeeee26aab851545",
            [],
            [("def identity(kv_cache_config", 1), ("JSPARK3_V14_IDENTITY", 3)],
            [],
            installers=["glm53_display_kv.py"],
        ),
    ],
}


# --- END v1.4 sections ---


# --- BEGIN v1.5 sections (generated by tools/v15/build_v15_contracts.py) ---


V15_FATPATH = {
    "sources": {
        "patch_exl3_fatpath.py": "3fd5ffc1e44566f7060de7598889611eac4cbfaaefb406d905508ffe5a21e9fc",
        "jspark3_exl3_fatpath.py": "69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e",
    },
    "targets": [
        target(
            "vllm/model_executor/layers/quantization/exl3.py",
            "9e823926962d11c410b74a651c21e5b56c4751afa453cb62c637468da98cccb6",
            "71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174",
            "3fd5ffc1e44566f7060de7598889611eac4cbfaaefb406d905508ffe5a21e9fc",
            [("from vllm.model_executor.utils import set_weight_attrs\n", 1), ("    counts = expert_count[:n_exp]\n    fn = exllamav3_ext.exl3_moe\n", 1), ("    if tokens > TEMP_ROWS_FUSED:\n        fat = (counts > TEMP_ROWS_FUSED).nonzero(as_tuple=False).view(-1)\n", 1)],
            [("[jspark3-fatpath]", 3), ("_jspark3_fatpath.dispatch(", 1), ("_jspark3_fatpath.probe(", 1)],
            ["    if tokens > TEMP_ROWS_FUSED:\n"],
            installers=["patch_exl3_fatpath.py"],
        ),
        target(
            "vllm/model_executor/layers/quantization/jspark3_exl3_fatpath.py",
            "ABSENT",
            "69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e",
            "69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e",
            [],
            [("MARKER = \"[jspark3-fatpath]\"", 1), ("def dispatch(", 1), ("def probe(", 1)],
            [],
            installers=["jspark3_exl3_fatpath.py"],
        ),
    ],
}


# --- END v1.5 sections ---


# --- BEGIN v1.6 optional-transform sections ---
# These are deliberately not members of SECTIONS: apply_base_pipeline.py
# remains the exact eight-stage v1.5 pipeline.  The v1.6 entrypoint applies
# these independent, conditional transactions around the existing installs.

# The cooperative contract lives in apply_coop_moe.EXPECTED_SECTION, whose
# source-manifest pin is rebound by the component seal integrator.

V16_ADAPTIVE_K = {
    "sources": {
        "jspark3_adaptive_k.py": "35c06d967579d8314fc8a7230db6129248bc1a80d6ce4be1ded6c210ef8d83b1",
        "patch_adaptive_k.py": "aa3398c4254c4cc9adcf82efe10751c9f400cfd7a910bd7883c107a2c85c7625"
    },
    "targets": [
        {
            "after_sha256": "0d58e688019ddfa5952990be2ea751b71d96bea7a35c4c31c19c672b0890f764",
            "before_sha256": "0b086dd3cc0febc4924202ca1fef8809b8290b9a59ef15fa80004cef4edeb937",
            "forbidden_after_seams": [],
            "installers": [
                "patch_adaptive_k.py"
            ],
            "path": "vllm/v1/core/sched/scheduler.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "# [jspark3-v16:adaptive-k scheduler]"
                },
                {
                    "count": 1,
                    "text": "_GLM53_ADAPTIVE_K.prepare_batch("
                },
                {
                    "count": 1,
                    "text": "_GLM53_ADAPTIVE_K.bind_output("
                },
                {
                    "count": 1,
                    "text": "_GLM53_ADAPTIVE_K.observe_output("
                }
            ],
            "required_before_seams": [
                {
                    "count": 1,
                    "text": "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
                },
                {
                    "count": 1,
                    "text": "        # Check if the scheduling constraints are satisfied.\n"
                }
            ],
            "source_sha256": "aa3398c4254c4cc9adcf82efe10751c9f400cfd7a910bd7883c107a2c85c7625"
        },
        {
            "after_sha256": "6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa",
            "before_sha256": "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f",
            "forbidden_after_seams": [],
            "installers": [
                "patch_adaptive_k.py"
            ],
            "path": "vllm/v1/worker/gpu/cudagraph_utils.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "# [jspark3-v16:adaptive-k target graphs]"
                },
                {
                    "count": 1,
                    "text": "def _glm53_adaptive_query_lens("
                },
                {
                    "count": 1,
                    "text": "decode_query_lens = _glm53_adaptive_query_lens("
                },
                {
                    "count": 1,
                    "text": "def _glm53_adaptive_capture_shape("
                },
                {
                    "count": 1,
                    "text": "if not _glm53_adaptive_capture_shape("
                }
            ],
            "required_before_seams": [
                {
                    "count": 1,
                    "text": "logger = init_logger(__name__)\n\n\n"
                }
            ],
            "source_sha256": "aa3398c4254c4cc9adcf82efe10751c9f400cfd7a910bd7883c107a2c85c7625"
        },
        {
            "after_sha256": "35c06d967579d8314fc8a7230db6129248bc1a80d6ce4be1ded6c210ef8d83b1",
            "before_sha256": "ABSENT",
            "forbidden_after_seams": [],
            "installers": [
                "jspark3_adaptive_k.py"
            ],
            "path": "vllm/v1/core/sched/jspark3_adaptive_k.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "class AdaptiveKPolicy"
                },
                {
                    "count": 1,
                    "text": "POLICY = AdaptiveKPolicy()"
                }
            ],
            "required_before_seams": [],
            "source_sha256": "35c06d967579d8314fc8a7230db6129248bc1a80d6ce4be1ded6c210ef8d83b1"
        }
    ]
}


V16_DENSE_FP8 = {
    "sources": {
        "jspark3_dense_fp8.py": "3c30ce3ea4aaddabef37f6ee3219b6698be319b95324cc8ed80794e292a4203d",
        "jspark3_dense_fp8_reference.py": "f747068d43ea9b7b105049a332fda398d715b535c6cef6fa3b851c71c0869afc",
        "patch_ablit_dense_fp8.py": "0babfbf0ff753db350d58e0f362200ab0727b33340a69aa784eff5f7fdd70e1d"
    },
    "targets": [
        {
            "after_sha256": "81ecbff301d9c5db87869883dab02115906d2f0c249209e127c57fac48c7fad0",
            "before_sha256": "91b28ac17086c857cd8ed960379a82bfbcd02aaf54b639aa97b8101e585ab3d4",
            "forbidden_after_seams": [
                "def finalize_with_ablit(model):\n    from . import trunk_w8a16 as trunk\n    mode = os.environ.get('ABLIT')\n"
            ],
            "installers": [
                "patch_ablit_dense_fp8.py"
            ],
            "path": "vllm/model_executor/layers/quantization/ablit_transplant.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "# [jspark3-v16-dense-fp8]"
                },
                {
                    "count": 1,
                    "text": "jspark3_dense_fp8 as trunk"
                }
            ],
            "required_before_seams": [
                {
                    "count": 1,
                    "text": "def finalize_with_ablit(model):\n    from . import trunk_w8a16 as trunk\n    mode = os.environ.get('ABLIT')\n"
                }
            ],
            "source_sha256": "0babfbf0ff753db350d58e0f362200ab0727b33340a69aa784eff5f7fdd70e1d"
        },
        {
            "after_sha256": "3c30ce3ea4aaddabef37f6ee3219b6698be319b95324cc8ed80794e292a4203d",
            "before_sha256": "ABSENT",
            "forbidden_after_seams": [],
            "installers": [
                "jspark3_dense_fp8.py"
            ],
            "path": "vllm/model_executor/layers/quantization/jspark3_dense_fp8.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "class DenseFp8Method"
                },
                {
                    "count": 1,
                    "text": "def finalize_dense_fp8"
                },
                {
                    "count": 1,
                    "text": "\"w8a8\": False"
                }
            ],
            "required_before_seams": [],
            "source_sha256": "3c30ce3ea4aaddabef37f6ee3219b6698be319b95324cc8ed80794e292a4203d"
        },
        {
            "after_sha256": "f747068d43ea9b7b105049a332fda398d715b535c6cef6fa3b851c71c0869afc",
            "before_sha256": "ABSENT",
            "forbidden_after_seams": [],
            "installers": [
                "jspark3_dense_fp8_reference.py"
            ],
            "path": "vllm/model_executor/layers/quantization/jspark3_dense_fp8_reference.py",
            "required_after_seams": [
                {
                    "count": 1,
                    "text": "CATEGORY_SPECS = {"
                },
                {
                    "count": 1,
                    "text": "def quantize_matrix"
                }
            ],
            "required_before_seams": [],
            "source_sha256": "f747068d43ea9b7b105049a332fda398d715b535c6cef6fa3b851c71c0869afc"
        }
    ]
}


# --- END v1.6 optional-transform sections ---


SECTIONS = {
    "apply_tp3_overlay.py": TP3,
    "apply_image_glm_dflash.py": IMAGE_DFLASH,
    "apply_kpool_tail.py": KPOOL,
    "apply_kda_mixed.py": KDA_MIXED,
    "apply_kda_fg.py": KDA_FG,
    "apply_v14_core.py": V14_CORE,
    "apply_display_kv.py": V14_DISPLAY,
    "apply_exl3_fatpath.py": V15_FATPATH,
}

# Always-on post-base grammar fix; independent of the three performance switches.
V16_GRAMMAR_FSM = {'sources': {'patch_xgrammar_termination.py': 'de84093ea2f51bf08e7d955b9fffbc7ae9ade337506146caebbad40a11d261a3'},
 'targets': [{'after_sha256': '041ab0d1328817102af00a6feec3825658be7480c25cd559efd05783e37de7d2',
              'before_sha256': '3fd606dc2b8e950fe9b49f28cf1c030be78beaaf7c78b457b5942a0909d3457f',
              'forbidden_after_seams': [],
              'installers': ['patch_xgrammar_termination.py'],
              'path': 'vllm/v1/structured_output/backend_xgrammar.py',
              'required_after_seams': [{'count': 1,
                                        'text': '# [glm53-xgrammar-termination] Source-exact vLLM 12f64b39 '
                                                'backport.'}],
              'required_before_seams': [],
              'source_sha256': 'de84093ea2f51bf08e7d955b9fffbc7ae9ade337506146caebbad40a11d261a3'},
             {'after_sha256': '9bc1b09f94902e4d68c1e7a3d4f05c8516d21196ef5773389f76c7bf6337335a',
              'before_sha256': '355f6f1193c15d5d6901a0f567e2e16005e3681f04f70079c6ba11e020b4d33a',
              'forbidden_after_seams': [],
              'installers': ['patch_xgrammar_termination.py'],
              'path': 'vllm/v1/structured_output/__init__.py',
              'required_after_seams': [{'count': 1,
                                        'text': '# [glm53-xgrammar-reasoning] Source-exact vLLM c6e19b3 '
                                                'backport.'}],
              'required_before_seams': [],
              'source_sha256': 'de84093ea2f51bf08e7d955b9fffbc7ae9ade337506146caebbad40a11d261a3'}]}

# S9.11 boot-only compile coverage (post-base, always enabled).
V16_WARMJIT = {'sources': {'warmup_jit.py': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
 'targets': [{'after_sha256': 'e566013a2fe339bcea3e3fd0b961c6731d4ffa8f7aac53b42baeeae36ae0e2b9',
              'before_sha256': 'c1aa70809364f4cc2fef76baf5b86aa8e9107a48a8d83982769eb0a3ea0b2fab',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/gpu_worker.py',
              'required_after_seams': [{'count': 1, 'text': '# [jspark3-s911-warmjit]'}],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a',
              'before_sha256': '659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': 'bd7f4c63d1196cb53bee0a81339aa5651e36938fc38b73c4ce89d978e0176a87',
              'before_sha256': 'bd7f4c63d1196cb53bee0a81339aa5651e36938fc38b73c4ce89d978e0176a87',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '8d04f3068a8dc22de69ec7ec6ab4f4192b8d45b5c3e253dcb7817eaf6c1314cd',
              'before_sha256': '8d04f3068a8dc22de69ec7ec6ab4f4192b8d45b5c3e253dcb7817eaf6c1314cd',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/gpu/sample/gumbel.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '78de8c1e2f338e0c83663217ac50950557aa77c6b52b5b76bd53876c51b89566',
              'before_sha256': '78de8c1e2f338e0c83663217ac50950557aa77c6b52b5b76bd53876c51b89566',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/gpu/sample/logprob.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '3aa60b97cffa744a07757f046d7760b08c029cb29423428c40f5f0bb80ac574e',
              'before_sha256': '3aa60b97cffa744a07757f046d7760b08c029cb29423428c40f5f0bb80ac574e',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/sample/ops/topk_topp_triton.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '520f4513ef4fc97e6b88dc07877cca8e29da0d34e44673fe989e3af002d86381',
              'before_sha256': '520f4513ef4fc97e6b88dc07877cca8e29da0d34e44673fe989e3af002d86381',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/third_party/flash_linear_attention/ops/l2norm.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '8da850460cfe91a57fbd116985e07be5254778c3069f523ffe75f0dd1152936b',
              'before_sha256': '8da850460cfe91a57fbd116985e07be5254778c3069f523ffe75f0dd1152936b',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/models/glm5next/nvidia/ops/kpool_compress.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '044d005cfe59fd0818ed421274e04f3e8dd679b8cdb64fca9f4422f2643484b2',
              'before_sha256': '044d005cfe59fd0818ed421274e04f3e8dd679b8cdb64fca9f4422f2643484b2',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/model_executor/layers/mamba/ops/causal_conv1d.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24',
              'before_sha256': '03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/model_executor/kernels/mhc/tilelang_kernels.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': 'c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce',
              'before_sha256': 'c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/model_executor/kernels/mhc/tilelang.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': 'b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d',
              'before_sha256': 'b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/models/glm5next/nvidia/kda.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': 'f27f887e71a7d092b79f7de55044a303456cfbd1e5eaf3461b394da9b3f21eba',
              'before_sha256': 'f27f887e71a7d092b79f7de55044a303456cfbd1e5eaf3461b394da9b3f21eba',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/attention/backends/gdn_attn.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'},
             {'after_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9',
              'before_sha256': 'ABSENT',
              'forbidden_after_seams': [],
              'installers': ['warmup_jit.py'],
              'path': 'vllm/v1/worker/jspark3_warmup_jit.py',
              'required_after_seams': [],
              'required_before_seams': [],
              'source_sha256': '8e055d6c48e35ffe23a661ab9a8cb6bfab037438294513cb8311d2ea7b5ddbe9'}]}
