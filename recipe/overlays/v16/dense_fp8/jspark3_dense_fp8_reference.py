"""Pure-Python reference and contracts for the JSpark3 v1.6 dense-FP8 lane.

This file deliberately has no torch/vLLM dependency.  The runtime packer uses
the same scale policy with torch's ``float8_e4m3fn`` conversion; offline tests
exercise this independent scalar implementation.
"""

from __future__ import annotations

import math
import struct
from typing import Mapping, Sequence


ENV = "JSPARK3_V16_DENSE_FP8"
NEGATIVE_CONTROL_ENV = "JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL"
PRODUCTION_MODE = "trunk"
NEGATIVE_MODE = "negative-coarse"
MODES = ("off", PRODUCTION_MODE, NEGATIVE_MODE)
NEGATIVE_CONTROL_ACK = "I_UNDERSTAND_TEST_ONLY"

# TP3-local runtime tensors.  This is intentionally the exact v1.5
# trunk_w8a16 census; the runtime module cross-checks it before packing.
# category -> (module count, N, K, v1.5 INT8 group size)
CATEGORY_SPECS = {
    "kda_o": (34, 4096, 2816, 128),
    "mla_fused_qkv_a": (11, 2048, 4096, 128),
    "mla_q_b": (11, 5632, 1536, 128),
    "mla_kv_b": (11, 11264, 512, 128),
    "mla_o": (11, 4096, 5632, 128),
    "shared_gate_up": (42, 1408, 4096, 128),
    "shared_down": (42, 4096, 704, 64),
    "dense_gate_up": (3, 8192, 4096, 128),
    "dense_down": (3, 4096, 4096, 128),
    "lm_head": (1, 51648, 4096, 128),
}


def resolve_mode(value: str | None) -> str:
    """Resolve the public boot knob.  No aliases: spelling mistakes refuse."""

    mode = "off" if value is None else value.strip().lower()
    if mode not in MODES:
        raise ValueError(f"{ENV} must be exactly one of {MODES}, got {value!r}")
    return mode


def validate_composition(mode: str, environ: Mapping[str, str]) -> None:
    """Reject combinations that cannot preserve v1.5 ownership contracts."""

    mode = resolve_mode(mode)
    if mode == "off":
        return
    required = {
        "JSPARK3_TRUNK_W8A16": "1",  # keeps the existing post-load hook armed
        "JSPARK3_KDA_MIXED_OUTPUT_BLOCKS": "8",
        "JSPARK3_KDA_FG_BATCHED": "1",
        "JSPARK3_KDA_QKV_SHADOW": "1",
        "GLM53_DENSE_FP8": "off",  # Mia's constructor path conflicts with B45
    }
    bad = {
        key: {"expected": wanted, "observed": environ.get(key)}
        for key, wanted in required.items()
        if environ.get(key) != wanted
    }
    if bad:
        details = ", ".join(
            f"{key}={row['observed']!r} (need {row['expected']!r})"
            for key, row in sorted(bad.items())
        )
        raise ValueError(f"dense-FP8 composition refusal: {details}")
    for key in ("GLM53_KDA_BF16_LARGE_M", "GLM53_COOP_GEOMETRY"):
        if key in environ:
            raise ValueError(f"dense-FP8 composition refusal: unsupported {key}")
    if mode == NEGATIVE_MODE:
        if (environ.get("JSPARK3_V16_PROFILE") != "qa"
                or environ.get(NEGATIVE_CONTROL_ENV) != NEGATIVE_CONTROL_ACK):
            raise ValueError(
                f"{NEGATIVE_MODE} is test-only and requires JSPARK3_V16_PROFILE=qa "
                f"and {NEGATIVE_CONTROL_ENV}={NEGATIVE_CONTROL_ACK}"
            )
    elif NEGATIVE_CONTROL_ENV in environ:
        raise ValueError("dense-FP8 test acknowledgement is valid only for negative-coarse")


def byte_model() -> dict[str, int | float]:
    """Ideal packed-weight byte model, excluding fixed Marlin workspaces."""

    logical_weights = 0
    bf16_bytes = 0
    w8a16_bytes = 0
    fp8_bytes = 0
    for count, n, k, group in CATEGORY_SPECS.values():
        weights = count * n * k
        logical_weights += weights
        bf16_bytes += 2 * weights
        # v1.5: one byte/weight plus one BF16 scale per output/group.
        w8a16_bytes += weights + count * n * (k // group) * 2
        # v1.6 candidate: one byte/weight plus one BF16 scale/output row.
        fp8_bytes += weights + count * n * 2
    delta = w8a16_bytes - fp8_bytes
    return {
        "runtime_modules": sum(row[0] for row in CATEGORY_SPECS.values()),
        "logical_weights": logical_weights,
        "bf16_bytes": bf16_bytes,
        "w8a16_bytes": w8a16_bytes,
        "fp8_bytes": fp8_bytes,
        "w8a16_minus_fp8_bytes": delta,
        "fp8_over_w8a16": fp8_bytes / w8a16_bytes,
        "bandwidth_only_speedup_ceiling": w8a16_bytes / fp8_bytes - 1.0,
    }


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def round_bfloat16(value: float) -> float:
    """Round a Python float through IEEE float32 to BF16, ties-to-even."""

    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    exponent = bits & 0x7F800000
    fraction = bits & 0x007FFFFF
    if exponent == 0x7F800000 and fraction:
        rounded = bits | 0x00400000
    else:
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def _positive_e4m3fn_codebook() -> tuple[tuple[int, float], ...]:
    rows = [(0, 0.0)]
    # Subnormals: mantissa / 8 * 2**(1-bias), bias=7.
    rows.extend((mantissa, mantissa * 2.0**-9) for mantissa in range(1, 8))
    # E4M3FN uses exponent field 15 for finite values through mantissa 6;
    # exponent=15,mantissa=7 is NaN.  Maximum finite value is 448.
    for exponent in range(1, 16):
        max_mantissa = 6 if exponent == 15 else 7
        for mantissa in range(max_mantissa + 1):
            bits = exponent * 8 + mantissa
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
            rows.append((bits, value))
    return tuple(rows)


_E4M3FN_POSITIVE = _positive_e4m3fn_codebook()


def e4m3fn_bits(value: float) -> int:
    """Reference E4M3FN conversion with round-to-nearest, ties-to-even."""

    value = _f32(value)
    if math.isnan(value):
        return 0x7F
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), 448.0)
    # Encoding LSB is the retained significand LSB, so bit parity implements
    # ties-to-even for adjacent finite encodings.
    bits, _ = min(
        _E4M3FN_POSITIVE,
        key=lambda item: (abs(item[1] - magnitude), item[0] & 1),
    )
    return sign | bits


def e4m3fn_value(bits: int) -> float:
    if not isinstance(bits, int) or bits not in range(256):
        raise ValueError("FP8 encoding must be an integer byte")
    sign = -1.0 if bits & 0x80 else 1.0
    code = bits & 0x7F
    if code == 0x7F:
        return math.nan
    exponent, mantissa = divmod(code, 8)
    if exponent == 0:
        value = mantissa * 2.0**-9
    else:
        value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return _f32(sign * value)


def quantize_matrix(
    rows: Sequence[Sequence[float]], mode: str = PRODUCTION_MODE
) -> tuple[list[list[int]], list[float], list[list[float]]]:
    """Reference for load-time quantization and logical dequantized weights."""

    mode = resolve_mode(mode)
    if mode == "off":
        raise ValueError("off has no FP8 quantization")
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be non-empty and rectangular")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("matrix must contain only finite weights")
    maxima = [max(abs(_f32(value)) for value in row) for row in rows]
    if mode == NEGATIVE_MODE:
        maxima = [max(maxima)] * len(maxima)
    scales_f32 = [_f32(max(value, 1e-12) / 448.0) for value in maxima]
    scales_bf16 = [round_bfloat16(value) for value in scales_f32]
    encoded = []
    dequantized = []
    for row, quant_scale, stored_scale in zip(rows, scales_f32, scales_bf16):
        codes = [e4m3fn_bits(_f32(_f32(value) / quant_scale)) for value in row]
        encoded.append(codes)
        dequantized.append(
            [_f32(e4m3fn_value(code) * stored_scale) for code in codes]
        )
    return encoded, scales_bf16, dequantized
