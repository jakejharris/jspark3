"""CPU bit model for three bf16 additions (no torch or GPU imports).

Integers are bf16 encodings. IEEE round-to-nearest-even, gradual underflow,
and NVIDIA's canonical bf16 NaN are explicit; Python float is not the oracle.
"""
from __future__ import annotations

from itertools import permutations

NAN = 0x7FFF
PROBE = (0x3F80, 0x3B80, 0xBBC0)  # 1, 2**-8, -1.5*2**-8
PROBES = tuple(permutations(PROBE))


def units(bits: int) -> int:
    """Finite bf16 as an integer multiple of 2**-133."""
    exponent, fraction = (bits >> 7) & 255, bits & 127
    significand = fraction if exponent == 0 else 128 + fraction
    return (-1 if bits & 0x8000 else 1) * (significand << max(0, exponent - 1))


def rounded(value: int, negative_zero: bool = False) -> int:
    sign = 0x8000 if value < 0 or (value == 0 and negative_zero) else 0
    value = abs(value)
    shift = max(0, value.bit_length() - 8)
    significand, remainder = divmod(value, 1 << shift)
    if shift and (remainder > 1 << (shift - 1) or
                  (remainder == 1 << (shift - 1) and significand & 1)):
        significand += 1
    if significand == 256:
        significand >>= 1
        shift += 1
    exponent = shift + 1 if significand >= 128 else 0
    if exponent >= 255:
        return sign | 0x7F80
    return sign | (exponent << 7) | (significand & 127)


def add(a: int, b: int) -> int:
    a, b = int(a), int(b)
    aa, bb = a & 0x7FFF, b & 0x7FFF
    if aa > 0x7F80 or bb > 0x7F80:
        return NAN
    if aa == 0x7F80 or bb == 0x7F80:
        if aa == bb and (a ^ b) & 0x8000:
            return NAN
        return a if aa == 0x7F80 else b
    return rounded(units(a) + units(b), a == b == 0x8000)


def reduce_bits(values, last: int) -> int:
    a, b = (last + 1) % 3, (last + 2) % 3
    return add(add(values[a], values[b]), values[last])


def fused_bits(values) -> int:
    """Exact ternary sum rounded once, used only as a negative control."""
    return rounded(sum(units(int(v)) for v in values))


SIGNATURES = tuple(tuple(reduce_bits(p, c) for p in PROBES) for c in range(3))


def infer_labels(probe_outputs):
    """Require all six probes to identify exactly one label per element."""
    if len(probe_outputs) != len(PROBES):
        raise ValueError("six independent probe outputs are required")
    size = len(probe_outputs[0])
    if any(len(row) != size for row in probe_outputs):
        raise ValueError("probe count drift")
    labels = []
    for i in range(size):
        signature = tuple(int(row[i]) & 65535 for row in probe_outputs)
        if signature not in SIGNATURES:
            raise ValueError(f"non-pairwise or unstable plan at element {i}")
        labels.append(SIGNATURES.index(signature))
    return bytes(labels)
