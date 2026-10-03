#!/usr/bin/env python3
"""Recreate the modified chat template from the pinned, public MIT original."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP/resolve/76add2a341a1cd90ad0e86bb69839ea9c35827c6/chat_template.jinja"
STOCK_SHA = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
OUTPUT_SHA = "77c01ab2ea2013fb68161c8c0de6bf6c96c95b3e699e518554da384a695ab482"
BLOCK = b"""{#- JSpark3: a request that switches thinking off (enable_thinking or thinking set false) thinks at Low, not Max; this template has no no-think mode. -#}
{%- if effective_reasoning_effort == 'max' and ((enable_thinking is defined and not enable_thinking) or (thinking is defined and not thinking)) -%}
{%- set effective_reasoning_effort = 'low' -%}
{%- endif -%}
{#- JSpark3 (effort-high-default): with no reasoning_effort given and thinking on, answer at High rather than Max. -#}
{%- if reasoning_effort is not defined and effective_reasoning_effort == 'max' -%}{%- set effective_reasoning_effort = 'high' -%}{%- endif -%}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, help="Previously downloaded stock template")
    a = p.parse_args()
    if a.source:
        stock = a.source.read_bytes()
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URL, timeout=45) as response:
            stock = response.read(1024 * 1024)
    if hashlib.sha256(stock).hexdigest() != STOCK_SHA:
        raise SystemExit("Stock template hash does not match the pinned revision")
    lines = stock.splitlines(keepends=True)
    result = b"".join(lines[:2]) + BLOCK + b"".join(lines[2:])
    digest = hashlib.sha256(result).hexdigest()
    if digest != OUTPUT_SHA:
        raise SystemExit("Reconstructed template does not match the required hash")
    with a.output.open("xb") as output:
        output.write(result)
    print(f"{digest}  {a.output.name}")


if __name__ == "__main__":
    main()
