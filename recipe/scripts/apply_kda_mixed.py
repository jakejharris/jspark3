#!/usr/bin/env python3
"""Install exact original KDA modules and apply the mixed-output hook atomically."""

from __future__ import annotations

import argparse
import base64
import gzip
from pathlib import Path
import sys

from _atomic import Refusal, execute, print_receipt, safe_target, sha_bytes
from _contracts import KDA_MIXED

KDA_REL = "vllm/models/glm5next/nvidia/kda.py"
MODULE_REL = "vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py"
BASE_REL = "vllm/model_executor/layers/quantization/kda_mixed_output_blocks_base.py"
KDA_BEFORE = "ec090aabecc1a63dacc9694ea677b195e95ce0c63648c418a6daaf34b8196125"
KDA_AFTER = "b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231"
KDA_FINAL = "b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d"
BASE_SHA = "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365"
WRAPPER_SHA = "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5"
# Original package-authored modules are encoded here so this remains a transform,
# not a pre-patched vLLM source payload.
BASE_GZIP_B64 = "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"
WRAPPER_GZIP_B64 = "H4sIAAAAAAACA6VWXU/jOBR9z6/w5CmV0iB2pNGoGlYCpkgspCBoV2hXK8tNnDY7jt3xB5Rh+O97bSckoR+g2by0ia+Pz7XPuddhGN4uyYoOlSa6zNCDJKsVlYhIYXiO9JKiQooflKOb3z6iqlzTHF18PUYrKf6lmS4FRxXVS5EnYRgGAcRWCOPCaCMpxqisVkJqRDgXFl9wFQT1N9H+1UJmy95LwnlSGO4WIAwRhc5q8HvGqqQSOWWYrmlmIDxh5JFKlbCSUyKbNWf8uyFclz9ofukGUsezhkm+5QS7dLAwemU0njORfVN4ThRtIKIAwXNyeXV6gU+vLmfp5DZ2n8Z3x6dT3AzMJtPm8/X4dDr+is8n17Mpvj3/a/xq4Go27Y9c5CS1LK4ciRPHwRO1SeMTYLM7JA4GQRCMJ3/iyXE6Rkco/OP2+vjm4iOGI8Lp+V27ouN6GwaeeYrT4zuI/wzTM0aU2sMjeoPEYOQSgeOfwc7RNck0Ojk7/ISIRumXo89w+Js6qjPUiBG5ALmlXj4WKKeFVVdGlcIPtFwstcKk0FRiJkhe8kWkKCti5A591MolFblhdICGv6OJ4NSzsk9RgoSsDCDhBQUZahm5yTEK8S4ZvEwKY3RGmKKDFzyf4iaWhVLWS9h7CbvAOgeAsaxalLJoibVc6xFwCypVyQGIZzRyQHGd6pRyJeSgP8c+kpRwADcGNF/RsZRCRqHqetv6tt2MhxLOwOg6nZrloIcqKdiYBy/ffNCW1Hfn+CoTH7g3lXelIel3U0qqnLCsLiAhJ7qNPICDNivWLJ04qAH6cISira7cZuJfYei2dVjv2IIKkLx8RLksiy67rkRgX2uOOWxvtowGScZgP6M2XBmozfB9vz/cqbST3Gsi6aJUNmpuigJAehm9qd3ua4yAhLJgXB9NpaEvUK/X3IFa4YqsIdtOLWqtD+2HPbbsnNf7sBueb8fXo56y2oF5SVR/DP10YgUa9sdHuuLRDRp1ZWQxQMtO0v0K825RNBCGw0naFgP1ZcMu/6NG/QIlo8A6c1oISZvi4Bp1n9fa++bv4eE/1jpbPIJgS9dJrh9X1Eb4XZwXoEl9+OlNXhvFrAg7e4R6rEFC5b3jOEJP3ts1vcHzwVPN4TnsQbbJFIyAAuHY1wk3FWXRAB0cbEuoV6n9pC89zY621Ep0Vt9ConW83wTeSy2ven7jcG+DWgKAZZVz5KtrEFifUE7mjO7SRvSu7gg9d+xgkODssdukHbmhw4pR+nnom0QG9zctjbuVtQ37njBjfSRUAtKl/D5qbiQ+O9g/HwLC7/um7i+9INAOrBFt3LA62n5LP0X41DB4RpVRGvTtKxgk+bSB/ByjBdjuya3/QXaU88L/VRd77VB30cT+UtM0wXjXBXR/Im+0OiHLhTWpb3b1ip6mV1uXCpzJnmvd5g4H/wEbtgoiDQwAAA=="
IMPORT_ANCHOR = "from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention\n"
IMPORT_LINE = "from vllm.model_executor.layers.quantization.kda_mixed_output_blocks import enable_kda_mixed_output_blocks\n"
HOOK_ANCHOR = "        super().__init__(input_size, output_sizes, **kwargs)\n"
HOOK_LINE = "        enable_kda_mixed_output_blocks(self)\n"


def unpack(value: str, expected: str) -> bytes:
    data = gzip.decompress(base64.b64decode(value))
    if sha_bytes(data) != expected:
        raise Refusal("embedded original KDA module drift")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        vllm = args.vllm_root.resolve(strict=True)
        root = vllm.parent
        if (root / "vllm").resolve(strict=True) != vllm:
            raise Refusal("--vllm-root must name the vllm package directory")

        def build(before: dict[Path, bytes]) -> dict[Path, bytes]:
            kda = safe_target(root, KDA_REL)
            module = safe_target(root, MODULE_REL)
            text = before[kda].decode("utf-8")
            if text.count(IMPORT_ANCHOR) != 1 or text.count(HOOK_ANCHOR) != 1:
                raise Refusal("mixed-output seam drift")
            text = text.replace(IMPORT_ANCHOR, IMPORT_LINE + IMPORT_ANCHOR, 1)
            text = text.replace(HOOK_ANCHOR, HOOK_ANCHOR + HOOK_LINE, 1)
            return {kda: text.encode(), module: unpack(BASE_GZIP_B64, BASE_SHA)}

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_kda_mixed.py",
            expected_section=KDA_MIXED,
            builder=build,
            apply=args.apply,
            script_path=Path(__file__),
        )
        print_receipt(receipt)
        return 0
    except (OSError, ValueError, UnicodeError, Refusal) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
