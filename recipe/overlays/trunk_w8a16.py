#!/usr/bin/env python3
"""JSpark3 v1: load-time INT8/Marlin conversion for the base recipe's BF16 trunk."""

from __future__ import annotations

import json
import os
import re

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
)
from vllm.scalar_type import scalar_types


logger = init_logger(__name__)
ENV = "JSPARK3_TRUNK_W8A16"
K704_ENV = "JSPARK3_TRUNK_W8A16_K704_GROUP"
EXPECTED_RUNTIME_MODULES = 169
EXPECTED_LOGICAL_TENSORS = 225
WTYPE = scalar_types.uint8b128
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPECTED_CATEGORIES = {
    "kda_o": 34,
    "mla_fused_qkv_a": 11,
    "mla_q_b": 11,
    "mla_kv_b": 11,
    "mla_o": 11,
    "shared_gate_up": 42,
    "shared_down": 42,
    "dense_gate_up": 3,
    "dense_down": 3,
    "lm_head": 1,
}
EXPECTED_SHAPES = {
    "kda_o": (4096, 2816),
    "mla_fused_qkv_a": (2048, 4096),
    "mla_q_b": (5632, 1536),
    "mla_kv_b": (11264, 512),
    "mla_o": (4096, 5632),
    "shared_gate_up": (1408, 4096),
    "shared_down": (4096, 704),
    "dense_gate_up": (8192, 4096),
    "dense_down": (4096, 4096),
    "lm_head": (51648, 4096),
}


class TrunkW8A16Method(QuantizeMethodBase):
    """Post-load method; weight creation remains on the base loader."""

    def __init__(self, input_size: int, output_size: int, suffix: str = "") -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.suffix = suffix

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("JSpark3 W8A16 attaches only after the BF16 loader completes")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply(layer, x, bias, self.input_size, self.output_size, self.suffix)


def _names(suffix: str) -> dict[str, str]:
    tag = f"_{suffix}" if suffix else ""
    return {
        "qweight": f"_trunk_w8a16_qweight{tag}",
        "scales": f"_trunk_w8a16_scales{tag}",
        "empty": f"_trunk_w8a16_empty{tag}",
        "workspace": f"_trunk_w8a16_workspace{tag}",
    }


def _apply(
    layer: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    input_size: int,
    output_size: int,
    suffix: str = "",
) -> torch.Tensor:
    names = _names(suffix)
    empty = getattr(layer, names["empty"])
    return apply_gptq_marlin_linear(
        input=x,
        weight=getattr(layer, names["qweight"]),
        weight_scale=getattr(layer, names["scales"]),
        weight_zp=empty,
        g_idx=empty,
        g_idx_sort_indices=empty,
        workspace=getattr(layer, names["workspace"]),
        wtype=WTYPE,
        output_size_per_partition=output_size,
        input_size_per_partition=input_size,
        is_k_full=True,
        bias=bias,
    )


def _register(layer: nn.Module, name: str, value: torch.Tensor) -> None:
    if hasattr(layer, name):
        raise RuntimeError(f"JSpark3 buffer collision: {name}")
    layer.register_buffer(name, value, persistent=False)


def pack_weight(layer: nn.Module, weight_nk: torch.Tensor, suffix: str = "") -> dict:
    """Pack one already-loaded [N,K] BF16 TP-local tensor and retain no BF16 copy."""
    if weight_nk.dtype != torch.bfloat16 or weight_nk.device.type != "cuda" or weight_nk.ndim != 2:
        raise RuntimeError(
            f"JSpark3 expected CUDA BF16 [N,K], got {weight_nk.dtype} {weight_nk.device} {tuple(weight_nk.shape)}"
        )
    n, k = map(int, weight_nk.shape)
    if k % 128 == 0:
        group_size = 128
    elif k == 704:
        value = os.getenv(K704_ENV)
        if value not in {"64", "32"}:
            raise RuntimeError(f"JSpark3 requires {K704_ENV}=64 or 32, got {value!r}")
        group_size = int(value)
    elif k % 64 == 0:
        group_size = 64
    elif k % 32 == 0:
        group_size = 32
    else:
        raise RuntimeError(f"JSpark3 has no supported group for K={k}")
    if k % group_size:
        raise RuntimeError(f"JSpark3 group {group_size} does not divide K={k}")
    weight_kn = weight_nk.T.contiguous()
    dequantized, qweight, scales, _, _ = gptq_quantize_weights(
        weight_kn, WTYPE, group_size, act_order=False
    )
    packed = gptq_pack(qweight, WTYPE.size_bits, k, n)
    padded_n, padded_k = marlin_padded_nk(n, k, group_size)
    packed = marlin_pad_qweight(packed, n, k, padded_n, padded_k)
    empty_for_repack = torch.empty(0, dtype=torch.int, device=weight_nk.device)
    repacked = ops.gptq_marlin_repack(
        packed, empty_for_repack, padded_k, padded_n, WTYPE.size_bits
    )
    scales = marlin_permute_scales(
        marlin_pad_scales(scales, n, k, padded_n, padded_k, group_size),
        size_k=padded_k,
        size_n=padded_n,
        group_size=group_size,
    )
    workspace = marlin_make_workspace_new(weight_nk.device)
    names = _names(suffix)
    _register(layer, names["qweight"], repacked)
    _register(layer, names["scales"], scales)
    _register(layer, names["empty"], empty_for_repack)
    _register(layer, names["workspace"], workspace)
    original_bytes = weight_nk.numel() * weight_nk.element_size()
    packed_bytes = sum(
        value.numel() * value.element_size()
        for value in (repacked, scales, empty_for_repack, workspace)
    )
    comparison_weight = dequantized
    del qweight, packed, weight_kn
    return {
        "n": n,
        "k": k,
        "padded_n": padded_n,
        "padded_k": padded_k,
        "group_size": group_size,
        "original_bf16_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "dequantized": comparison_weight,
    }


def _layer_index(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _category(name: str, module: nn.Module) -> str | None:
    layer = _layer_index(name)
    if layer is None:
        return "lm_head" if name.endswith("lm_head") else None
    if layer >= 45 or ".experts" in name or "draft" in name.lower():
        return None
    mla = layer % 4 == 3
    if name.endswith(".self_attn.o_proj"):
        return "mla_o" if mla else "kda_o"
    if mla and name.endswith(".self_attn.fused_qkv_a_proj"):
        return "mla_fused_qkv_a"
    if mla and name.endswith(".self_attn.q_b_proj"):
        return "mla_q_b"
    if mla and name.endswith(".self_attn.kv_b_proj"):
        return "mla_kv_b"
    if layer >= 3 and name.endswith(".mlp.shared_experts.gate_up_proj"):
        return "shared_gate_up"
    if layer >= 3 and name.endswith(".mlp.shared_experts.down_proj"):
        return "shared_down"
    if layer < 3 and name.endswith(".mlp.gate_up_proj"):
        return "dense_gate_up"
    if layer < 3 and name.endswith(".mlp.down_proj"):
        return "dense_down"
    return None


def _logical_count(category: str) -> int:
    return {
        "mla_fused_qkv_a": 2,
        "shared_gate_up": 2,
        "dense_gate_up": 2,
    }.get(category, 1)


def finalize_trunk_w8a16(model: nn.Module) -> None:
    value = os.getenv(ENV)
    if value is None:
        return
    if value != "1":
        raise RuntimeError(f"{ENV} must be exactly 1, got {value!r}")
    selected = []
    category_counts = {key: 0 for key in EXPECTED_CATEGORIES}
    for name, module in model.named_modules():
        category = _category(name, module)
        if category is not None:
            selected.append((name, module, category))
            category_counts[category] += 1
    if category_counts != EXPECTED_CATEGORIES or len(selected) != EXPECTED_RUNTIME_MODULES:
        raise RuntimeError(
            "JSpark3 target census drift: "
            + json.dumps({"runtime": len(selected), "categories": category_counts}, sort_keys=True)
        )
    original_bytes = 0
    packed_bytes = 0
    logical_tensors = 0
    non_128_modules = []
    for name, module, category in selected:
        expected_shape = EXPECTED_SHAPES[category]
        weight = getattr(module, "weight", None)
        if weight is None or tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"JSpark3 {category} shape drift at {name}: {None if weight is None else tuple(weight.shape)}"
            )
        method = getattr(module, "quant_method", None)
        if method is None or type(method).__name__ not in {
            "UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"
        }:
            raise RuntimeError(
                f"JSpark3 target is not unquantized at {name}: {type(method).__name__}"
            )
        receipt = pack_weight(module, weight)
        receipts = [receipt]
        module.quant_method = TrunkW8A16Method(receipt["k"], receipt["n"])
        del module._parameters["weight"]
        original_bytes += sum(item["original_bf16_bytes"] for item in receipts)
        packed_bytes += sum(item["packed_bytes"] for item in receipts)
        logical_tensors += _logical_count(category)
        if any(item["group_size"] != 128 for item in receipts):
            non_128_modules.append(name)
        for item in receipts:
            del item["dequantized"]
        torch.cuda.empty_cache()
    if logical_tensors != EXPECTED_LOGICAL_TENSORS:
        raise RuntimeError(f"JSpark3 logical tensor census drift: {logical_tensors}")
    receipt = {
        "status": "JSPARK3_TRUNK_W8A16_FINALIZE_PASS",
        "runtime_modules": len(selected),
        "logical_tensors": logical_tensors,
        "original_bf16_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "net_bytes_freed": original_bytes - packed_bytes,
        "category_counts": category_counts,
        "k704_group": int(os.environ[K704_ENV]),
        "non_128_modules": non_128_modules,
    }
    logger.warning("JSPARK3_TRUNK_W8A16_RECEIPT=%s", json.dumps(receipt, sort_keys=True))
    print("JSPARK3_TRUNK_W8A16_RECEIPT=" + json.dumps(receipt, sort_keys=True), flush=True)
