"""JSpark3 v1.6 post-load dense FP8 replacement for the v1.5 W8A16 trunk.

Production mode ``trunk`` converts exactly the same 169 TP-local runtime
modules / 225 logical tensors as ``trunk_w8a16.py``.  Weights are E4M3FN with
one BF16 scale per output channel and execute through weight-only FP8 Marlin;
activations remain BF16 (W8A16, never W8A8).  The B45 KDA input composite and
its decode-only INT8 QKV shadow retain their existing owners and are excluded.

``negative-coarse`` is a guarded hardware-QA control.  It gives every row of a
module the module-wide maximum scale and must never be selected by a production
profile.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)

from .jspark3_dense_fp8_reference import (
    CATEGORY_SPECS,
    ENV,
    NEGATIVE_MODE,
    PRODUCTION_MODE,
    byte_model,
    resolve_mode,
    validate_composition,
)


logger = init_logger(__name__)
EXPECTED_RUNTIME_MODULES = 169
EXPECTED_LOGICAL_TENSORS = 225
QUANT_CHUNK_ROWS = 512


class DenseFp8Method(QuantizeMethodBase):
    """Post-load method; weight creation and custom loaders remain v1.5."""

    def __init__(self, input_size: int, output_size: int, suffix: str = "") -> None:
        if suffix:
            raise RuntimeError("dense FP8 does not support auxiliary packed banks")
        self.input_size = input_size
        self.output_size = output_size

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("dense FP8 attaches only after the BF16 loader completes")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=self.output_size,
            size_k=self.input_size,
            bias=bias,
        )


# Compatibility with ablit_transplant.py's verified packer wrapper.  In the
# candidate boot that module imports this file as ``trunk``; keeping this tiny
# API means the transplant proof is reused instead of weakened.
TrunkW8A16Method = DenseFp8Method


def _names(suffix: str = "") -> dict[str, str]:
    if suffix:
        raise RuntimeError("dense FP8 does not support auxiliary packed banks")
    return {
        "qweight": "weight",
        "scales": "weight_scale",
        "empty": "_jspark3_dense_fp8_empty",
        "workspace": "workspace",
    }


def _set_parameter(layer: nn.Module, name: str, value: torch.Tensor) -> None:
    parameter = nn.Parameter(value, requires_grad=False)
    if name in layer._parameters:
        layer._parameters[name] = parameter
    else:
        layer.register_parameter(name, parameter)


def _quantize_weight(weight_nk: torch.Tensor, mode: str) -> tuple:
    """Chunked tensor implementation; also executable on CPU for numeric proof."""
    if (weight_nk.dtype != torch.bfloat16 or weight_nk.ndim != 2
            or not all(weight_nk.shape)):
        raise RuntimeError("dense FP8 quantizer requires nonempty BF16 [N,K]")
    if mode not in (PRODUCTION_MODE, NEGATIVE_MODE):
        raise RuntimeError("dense FP8 quantizer requires an enabled mode")
    n = weight_nk.shape[0]
    fp8 = torch.empty(weight_nk.shape, dtype=torch.float8_e4m3fn, device=weight_nk.device)
    stored_scales = torch.empty((n,), dtype=torch.bfloat16, device=weight_nk.device)
    comparison_nk = torch.empty_like(weight_nk)
    coarse_max = None
    if mode == NEGATIVE_MODE:
        for start in range(0, n, QUANT_CHUNK_ROWS):
            stop = min(start + QUANT_CHUNK_ROWS, n)
            block_max = weight_nk[start:stop].float().abs().amax()
            coarse_max = block_max if coarse_max is None else torch.maximum(coarse_max, block_max)
    for start in range(0, n, QUANT_CHUNK_ROWS):
        stop = min(start + QUANT_CHUNK_ROWS, n)
        wf = weight_nk[start:stop].float()
        maxima = wf.abs().amax(dim=1) if coarse_max is None else coarse_max.expand(stop - start)
        if not torch.isfinite(maxima).all():
            raise RuntimeError("dense FP8 refuses non-finite weights")
        scales = maxima.clamp(min=1e-12) / 448.0
        quantized = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        stored = scales.to(torch.bfloat16)
        fp8[start:stop] = quantized
        stored_scales[start:stop] = stored
        comparison_nk[start:stop] = (
            quantized.float() * stored.float()[:, None]
        ).to(torch.bfloat16)
        del wf, maxima, scales, quantized, stored
    return fp8, stored_scales, comparison_nk.T.contiguous()


def pack_weight(layer: nn.Module, weight_nk: torch.Tensor, suffix: str = "") -> dict:
    """Pack an already-loaded CUDA BF16 [N,K] tensor as E4M3FN Marlin."""

    _names(suffix)  # validate before touching the layer
    if (
        weight_nk.dtype != torch.bfloat16
        or weight_nk.device.type != "cuda"
        or weight_nk.ndim != 2
    ):
        raise RuntimeError(
            "JSpark3 dense FP8 expected CUDA BF16 [N,K], got "
            f"{weight_nk.dtype} {weight_nk.device} {tuple(weight_nk.shape)}"
        )
    mode = resolve_mode(os.environ.get(ENV))
    validate_composition(mode, os.environ)
    if mode == "off":
        raise RuntimeError("dense FP8 packer reached with mode=off")
    n, k = map(int, weight_nk.shape)
    for attr, value in (
        ("output_size_per_partition", n),
        ("input_size_per_partition", k),
    ):
        observed = getattr(layer, attr, value)
        if int(observed) != value:
            raise RuntimeError(
                f"dense FP8 layer geometry drift: {attr}={observed}, weight requires {value}"
            )
        setattr(layer, attr, value)

    fp8, stored_scales, dequantized = _quantize_weight(weight_nk, mode)
    original_bytes = weight_nk.numel() * weight_nk.element_size()

    layer.orig_dtype = torch.bfloat16
    _set_parameter(layer, "weight", fp8)
    _set_parameter(layer, "weight_scale", stored_scales)
    layer.weight_block_size = None
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    empty_name = _names()["empty"]
    if hasattr(layer, empty_name):
        raise RuntimeError(f"JSpark3 dense FP8 buffer collision: {empty_name}")
    layer.register_buffer(
        empty_name,
        torch.empty(0, dtype=torch.int, device=weight_nk.device),
        persistent=False,
    )
    packed_bytes = sum(
        value.numel() * value.element_size()
        for value in (
            layer.weight,
            layer.weight_scale,
            getattr(layer, empty_name),
            layer.workspace,
        )
    )
    return {
        "n": n,
        "k": k,
        "padded_n": n,
        "padded_k": k,
        "group_size": -1,
        "original_bf16_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "dequantized": dequantized,
    }


def _baseline_contract():
    from . import trunk_w8a16 as baseline

    expected_counts = {name: row[0] for name, row in CATEGORY_SPECS.items()}
    expected_shapes = {name: (row[1], row[2]) for name, row in CATEGORY_SPECS.items()}
    if (
        baseline.EXPECTED_RUNTIME_MODULES != EXPECTED_RUNTIME_MODULES
        or baseline.EXPECTED_LOGICAL_TENSORS != EXPECTED_LOGICAL_TENSORS
        or baseline.EXPECTED_CATEGORIES != expected_counts
        or baseline.EXPECTED_SHAPES != expected_shapes
    ):
        raise RuntimeError("JSpark3 dense FP8 refuses v1.5 trunk census drift")
    return baseline


def _kda_ownership(model: nn.Module, baseline) -> dict:
    """Audit the loaded B45 composite and separate BF16 F/G owner, without copying."""
    expected_layers = {layer for layer in range(45) if layer % 4 != 3}
    seen = {"input": set(), "fg": set()}
    snapshot = {}
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if (type(method).__name__ == "Glm53DenseFp8Method"
                or hasattr(module, "glm53_bf16_lm_w")
                or hasattr(module, "_glm53_bf16_lm_w")):
            raise RuntimeError(f"dense FP8 refuses legacy KDA/FP8 owner at {name}")
        layer = baseline._layer_index(name)
        if layer not in expected_layers:
            continue
        if name.endswith(".self_attn.in_proj_qkvbfg_a"):
            kind, shape = "input", (8726, 4096)
            weight = getattr(module, "kda_shape_static_exact_weight", None)
            if (type(method).__name__ != "KdaMixedOutputBlocksMethod"
                    or not getattr(module, "_kda_mixed_output_blocks_finalized", False)
                    or getattr(module, "kda_shape_static_exact_m_max", None) != 8
                    or not getattr(module, "kda_qkv_shadow_enabled", False)):
                raise RuntimeError(f"dense FP8 B45 input ownership drift at {name}")
        elif name.endswith(".self_attn.fused_fg_b_proj"):
            kind, shape = "fg", (2, 2816, 128)
            weight = getattr(module, "weight", None)
            if type(module).__name__ != "_Glm5NextBatchedColumnParallelLinear" or method is not None:
                raise RuntimeError(f"dense FP8 B45 F/G ownership drift at {name}")
        else:
            continue
        if (weight is None or tuple(weight.shape) != shape
                or weight.dtype != torch.bfloat16 or weight.device.type != "cuda"):
            raise RuntimeError(f"dense FP8 B45 retained BF16 tensor drift at {name}")
        if layer in seen[kind]:
            raise RuntimeError(f"dense FP8 duplicate B45 {kind} owner at layer {layer}")
        seen[kind].add(layer)
        tensors = (*module.named_parameters(), *module.named_buffers())
        snapshot[name] = (id(module), id(method), tuple(
            (key, id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype)
            for key, value in tensors
        ))
    if any(layers != expected_layers for layers in seen.values()):
        raise RuntimeError(f"dense FP8 B45 ownership census drift: {seen}")
    return snapshot


def finalize_dense_fp8(model: nn.Module) -> None:
    mode = resolve_mode(os.environ.get(ENV))
    validate_composition(mode, os.environ)
    if mode == "off":
        raise RuntimeError("dense FP8 finalizer reached with mode=off")
    rank = os.environ.get("NODE_RANK")
    if rank not in {"0", "1", "2"}:
        raise RuntimeError(f"dense FP8 requires NODE_RANK=0|1|2, got {rank!r}")
    baseline = _baseline_contract()
    selected = []
    category_counts = {key: 0 for key in baseline.EXPECTED_CATEGORIES}
    for name, module in model.named_modules():
        category = baseline._category(name, module)
        if category is not None:
            selected.append((name, module, category))
            category_counts[category] += 1
    if (
        category_counts != baseline.EXPECTED_CATEGORIES
        or len(selected) != EXPECTED_RUNTIME_MODULES
    ):
        raise RuntimeError(
            "JSpark3 dense FP8 target census drift: "
            + json.dumps(
                {"runtime": len(selected), "categories": category_counts},
                sort_keys=True,
            )
        )

    # Validate every destination and excluded owner before the first mutation.
    kda_before = _kda_ownership(model, baseline)
    logical_tensors = sum(baseline._logical_count(category) for _, _, category in selected)
    if logical_tensors != EXPECTED_LOGICAL_TENSORS:
        raise RuntimeError(f"JSpark3 dense FP8 logical tensor census drift: {logical_tensors}")
    for name, module, category in selected:
        weight = getattr(module, "weight", None)
        expected_shape = baseline.EXPECTED_SHAPES[category]
        if weight is None or tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"JSpark3 dense FP8 {category} shape drift at {name}: "
                f"{None if weight is None else tuple(weight.shape)}"
            )
        method_name = type(getattr(module, "quant_method", None)).__name__
        if method_name not in {"UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"}:
            raise RuntimeError(
                f"JSpark3 dense FP8 target is not unquantized at {name}: {method_name}"
            )
        if weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
            raise RuntimeError(f"JSpark3 dense FP8 target is not CUDA BF16 at {name}")
    original_bytes = 0
    packed_bytes = 0
    for name, module, category in selected:
        weight = module.weight
        receipt = pack_weight(module, weight)
        module.quant_method = DenseFp8Method(receipt["k"], receipt["n"])
        original_bytes += receipt["original_bf16_bytes"]
        packed_bytes += receipt["packed_bytes"]
        del receipt["dequantized"]
        torch.cuda.empty_cache()
    if _kda_ownership(model, baseline) != kda_before:
        raise RuntimeError("dense FP8 modified an excluded B45 owner or tensor")

    math_receipt = byte_model()
    receipt = {
        "status": "JSPARK3_V16_DENSE_FP8_FINALIZE_PASS",
        "rank": int(rank),
        "mode": mode,
        "scheme": "e4m3fn_per_output_channel" if mode == PRODUCTION_MODE else "e4m3fn_per_module_scale",
        "activation_dtype": "bfloat16",
        "w8a8": False,
        "runtime_modules": len(selected),
        "logical_tensors": logical_tensors,
        "original_bf16_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "category_counts": category_counts,
        "b45_kda_composite_owner": "B45_UNCHANGED",
        "b45_kda_composite_converted": False,
        "b45_kda_input_modules": 34,
        "b45_kda_fg_modules": 34,
        "b45_kda_ownership_audited": True,
        "ideal_byte_model": math_receipt,
    }
    print(
        f"[jspark3-v16:dense-fp8] rank={rank} state={mode} "
        f"scheme={receipt['scheme']} activation=bf16 modules={len(selected)}",
        flush=True,
    )
    logger.warning(
        "JSPARK3_V16_DENSE_FP8_RECEIPT=%s", json.dumps(receipt, sort_keys=True)
    )
    print(
        "JSPARK3_V16_DENSE_FP8_RECEIPT=" + json.dumps(receipt, sort_keys=True),
        flush=True,
    )


# Compatibility name used by the unchanged branches in ablit_transplant.py.
finalize_trunk_w8a16 = finalize_dense_fp8
