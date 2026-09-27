"""Shape-static wrapper around the frozen R23 mixed KDA projection method,
plus the B4 QKV-only W8A16 shadow used only at explicitly identified pure decode.

B4 (JSPARK3 v2 ladder row B4, retained QKV implementation). Differences from the frozen module 01aa249d...:
  * process_weights_after_loading additionally packs rows [0, 8448) of the retained BF16
    composite (Q/K/V) with the trunk overlay's Marlin uint8b128 group-128 contract
    (pack_weight from trunk_w8a16.py), registered on the layer under the suffix
    "kda_qkv_shadow". The BF16 composite, its 278 control rows (beta, f_a, g_a), the
    row order and the no-bias convention are untouched. Opt-in: JSPARK3_KDA_QKV_SHADOW=1.
  * apply, at flat_m <= 8 only, takes the shadow path iff the forward context carries
    explicit phase metadata for this layer's parent KDA layer saying pure decode
    (num_prefills == 0, num_prefill_tokens == 0, num_decodes + num_spec_decodes > 0).
    Any other case, including missing metadata, a profile run, or a short prefill chunk
    with M <= 8, takes the original exact BF16 path. M is never used as a phase test.
  * The M > 8 path (frozen block-FP8 construction) is unchanged.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from .kda_mixed_output_blocks_base import (
    BLOCK_COLUMNS,
    EXACT_BLOCK_COUNT,
    EXPECTED_INPUT_SIZE,
    EXPECTED_OUTPUT_SIZE,
    KdaMixedOutputBlocksMethod as _BaseKdaMixedOutputBlocksMethod,
)


logger = logging.getLogger(__name__)

ENV_NAME = "JSPARK3_KDA_MIXED_OUTPUT_BLOCKS"
EXACT_M_MAX = 8

SHADOW_ENV = "JSPARK3_KDA_QKV_SHADOW"
SHADOW_QKV_ROWS = 8448
SHADOW_CONTROL_ROWS = EXPECTED_OUTPUT_SIZE - SHADOW_QKV_ROWS  # 278
SHADOW_SUFFIX = "kda_qkv_shadow"
SHADOW_EXPECTED_LAYERS = 34
SHADOW_EXPECTED_GROUP = 128
_SHADOW_RECEIPTS: list[dict] = []


def _shadow_requested() -> bool:
    value = os.getenv(SHADOW_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(f"{SHADOW_ENV} must be exactly 1 when set, got {value!r}")
    return True


def _build_shadow(layer: torch.nn.Module, exact_weight: torch.Tensor) -> None:
    from .trunk_w8a16 import pack_weight

    if tuple(exact_weight.shape) != (EXPECTED_OUTPUT_SIZE, EXPECTED_INPUT_SIZE):
        raise RuntimeError("B4 shadow geometry drift: %s" % (tuple(exact_weight.shape),))
    if exact_weight.dtype != torch.bfloat16 or exact_weight.device.type != "cuda":
        raise RuntimeError("B4 shadow requires the CUDA BF16 composite")
    if getattr(layer, "bias", None) is not None:
        raise RuntimeError("B4 shadow: bias is unsupported on this projection")
    prefix = getattr(layer, "prefix", None)
    if not isinstance(prefix, str) or "." not in prefix:
        raise RuntimeError("B4 shadow requires the layer prefix for phase lookup")
    parent_prefix = prefix.rsplit(".", 1)[0]

    qkv = exact_weight[:SHADOW_QKV_ROWS]
    if not qkv.is_contiguous() or qkv.data_ptr() != exact_weight.data_ptr():
        raise RuntimeError("B4 shadow: QKV slice is not a zero-copy view of the composite")
    receipt = pack_weight(layer, qkv, suffix=SHADOW_SUFFIX)
    if (
        receipt["group_size"] != SHADOW_EXPECTED_GROUP
        or receipt["n"] != SHADOW_QKV_ROWS
        or receipt["k"] != EXPECTED_INPUT_SIZE
        or receipt["padded_n"] != SHADOW_QKV_ROWS
        or receipt["padded_k"] != EXPECTED_INPUT_SIZE
    ):
        raise RuntimeError("B4 shadow packing contract drift: %s" % json.dumps(
            {k: receipt[k] for k in ("group_size", "n", "k", "padded_n", "padded_k")}))
    dequantized = receipt["dequantized"]
    deq_nk = dequantized.T if tuple(dequantized.shape) == (EXPECTED_INPUT_SIZE, SHADOW_QKV_ROWS) else dequantized
    diff = deq_nk.float() - qkv.float()
    rel_fro = float(diff.norm() / qkv.float().norm())
    finite = bool(torch.isfinite(deq_nk.float()).all())
    zero_rows = qkv.abs().sum(dim=1) == 0
    inert_rows = int(zero_rows.sum())
    inert_exact = bool((deq_nk[zero_rows].abs().sum() == 0)) if inert_rows else True
    del receipt["dequantized"], dequantized, deq_nk, diff
    if not finite or not inert_exact:
        raise RuntimeError("B4 shadow: dequantized reference not finite or inert lanes not preserved")

    # control rows stay in the retained BF16 composite; this is a view, not a copy
    layer.kda_qkv_shadow_control_weight = exact_weight[SHADOW_QKV_ROWS:]
    if tuple(layer.kda_qkv_shadow_control_weight.shape) != (SHADOW_CONTROL_ROWS, EXPECTED_INPUT_SIZE):
        raise RuntimeError("B4 shadow control slice drift")
    layer.kda_qkv_shadow_parent_prefix = parent_prefix
    layer.kda_qkv_shadow_enabled = True
    layer._kda_qkv_shadow_seen_eager = False
    layer._kda_qkv_shadow_seen_capture = False
    torch.cuda.empty_cache()

    row = {
        "prefix": prefix,
        "parent_prefix": parent_prefix,
        "n": receipt["n"],
        "k": receipt["k"],
        "group_size": receipt["group_size"],
        "original_qkv_bf16_bytes": receipt["original_bf16_bytes"],
        "packed_bytes": receipt["packed_bytes"],
        "control_rows": SHADOW_CONTROL_ROWS,
        "quant_rel_fro_err": rel_fro,
        "inert_zero_rows": inert_rows,
    }
    _SHADOW_RECEIPTS.append(row)
    logger.info(
        "B4 KDA QKV shadow packed prefix=%s group=%d packed_bytes=%d rel_fro=%.6f inert_rows=%d",
        prefix, receipt["group_size"], receipt["packed_bytes"], rel_fro, inert_rows,
    )
    if len(_SHADOW_RECEIPTS) == SHADOW_EXPECTED_LAYERS:
        summary = {
            "status": "B4_KDA_QKV_SHADOW_PASS",
            "layers": len(_SHADOW_RECEIPTS),
            "group_size": SHADOW_EXPECTED_GROUP,
            "qkv_rows": SHADOW_QKV_ROWS,
            "control_rows": SHADOW_CONTROL_ROWS,
            "packed_bytes_total": sum(r["packed_bytes"] for r in _SHADOW_RECEIPTS),
            "original_qkv_bf16_bytes_total": sum(r["original_qkv_bf16_bytes"] for r in _SHADOW_RECEIPTS),
            "bf16_composite_retained": True,
            "quant_rel_fro_err_max": max(r["quant_rel_fro_err"] for r in _SHADOW_RECEIPTS),
            "inert_zero_rows_total": sum(r["inert_zero_rows"] for r in _SHADOW_RECEIPTS),
            "phase_gate": "num_prefills==0 and num_prefill_tokens==0 and num_decodes+num_spec_decodes>0, M<=8",
        }
        logger.warning("B4_KDA_QKV_SHADOW_RECEIPT=%s", json.dumps(summary, sort_keys=True))
        print("B4_KDA_QKV_SHADOW_RECEIPT=" + json.dumps(summary, sort_keys=True), flush=True)


def _pure_decode(layer: torch.nn.Module) -> bool:
    """Explicit phase metadata for this layer's parent KDA layer, or False."""
    try:
        forward_context = get_forward_context()
    except Exception:  # noqa: BLE001  (no forward context: never the shadow)
        return False
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return False
    metadata = attn_metadata.get(layer.kda_qkv_shadow_parent_prefix)
    if metadata is None:
        return False
    num_prefills = getattr(metadata, "num_prefills", None)
    num_prefill_tokens = getattr(metadata, "num_prefill_tokens", None)
    num_decodes = getattr(metadata, "num_decodes", None)
    num_spec_decodes = getattr(metadata, "num_spec_decodes", None)
    if None in (num_prefills, num_prefill_tokens, num_decodes, num_spec_decodes):
        return False
    return int(num_prefills) == 0 and int(num_prefill_tokens) == 0 and (int(num_decodes) + int(num_spec_decodes)) > 0


def _shadow_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    from .trunk_w8a16 import _apply

    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and not layer._kda_qkv_shadow_seen_capture:
        layer._kda_qkv_shadow_seen_capture = True
        logger.warning("B4_KDA_QKV_SHADOW_ACTIVE prefix=%s m=%d mode=cudagraph_capture",
                       layer.prefix, x.numel() // EXPECTED_INPUT_SIZE)
    elif not capturing and not layer._kda_qkv_shadow_seen_eager:
        layer._kda_qkv_shadow_seen_eager = True
        logger.warning("B4_KDA_QKV_SHADOW_ACTIVE prefix=%s m=%d mode=eager",
                       layer.prefix, x.numel() // EXPECTED_INPUT_SIZE)
    y_qkv = _apply(layer, x, None, EXPECTED_INPUT_SIZE, SHADOW_QKV_ROWS, SHADOW_SUFFIX)
    y_ctl = F.linear(x, layer.kda_qkv_shadow_control_weight)
    return torch.cat((y_qkv, y_ctl), dim=-1)


class KdaMixedOutputBlocksMethod(_BaseKdaMixedOutputBlocksMethod):
    """Use exact BF16 at M<=8 (or the B4 QKV shadow at explicit pure decode) and the frozen R23 method at larger M."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        finalized = getattr(layer, "_kda_mixed_output_blocks_finalized", False)
        exact = getattr(layer, "kda_shape_static_exact_weight", None)
        if finalized:
            if not isinstance(exact, torch.Tensor):
                raise RuntimeError("shape-static KDA finalized without exact weight")
            return

        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("shape-static KDA requires the loaded BF16 weight")
        if tuple(weight.shape) != (EXPECTED_OUTPUT_SIZE, EXPECTED_INPUT_SIZE):
            raise RuntimeError("shape-static KDA exact-weight geometry drift")
        exact_weight = weight.detach().clone()
        super().process_weights_after_loading(layer)
        layer.register_buffer(
            "kda_shape_static_exact_weight", exact_weight, persistent=True
        )
        layer.kda_shape_static_exact_m_max = EXACT_M_MAX
        layer.kda_qkv_shadow_enabled = False
        if _shadow_requested():
            _build_shadow(layer, layer.kda_shape_static_exact_weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise RuntimeError("shape-static KDA bias is unsupported")
        if not getattr(layer, "_kda_mixed_output_blocks_finalized", False):
            raise RuntimeError("shape-static KDA used before finalization")
        if x.shape[-1] != EXPECTED_INPUT_SIZE or x.dtype != torch.bfloat16:
            raise RuntimeError(
                f"unsupported shape-static activation: {tuple(x.shape)}/{x.dtype}"
            )
        flat_m = x.numel() // EXPECTED_INPUT_SIZE
        if flat_m <= EXACT_M_MAX:
            if getattr(layer, "kda_qkv_shadow_enabled", False) and _pure_decode(layer):
                return _shadow_apply(layer, x)
            return F.linear(x, layer.kda_shape_static_exact_weight)
        return super().apply(layer, x, bias=None)


def enable_kda_mixed_output_blocks(layer: torch.nn.Module) -> None:
    """Enable only the frozen eight-block, M8-exact construction."""

    value = os.getenv(ENV_NAME)
    if value is None:
        return
    if value != str(EXACT_BLOCK_COUNT):
        raise RuntimeError(
            f"{ENV_NAME} must be exactly {EXACT_BLOCK_COUNT}, got {value!r}"
        )
    if not isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod):
        raise RuntimeError("shape-static KDA requires the original BF16 method")
    layer.quant_method = KdaMixedOutputBlocksMethod(EXACT_BLOCK_COUNT)
