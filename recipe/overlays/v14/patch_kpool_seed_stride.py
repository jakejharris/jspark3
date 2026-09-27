#!/usr/bin/env python3
"""vLLM #57477 (db1bfdd4): the NVIDIA kpool tail seed kernel honors the padded
tail block stride.

The kpool tail cache aliases the indexer cache with the indexer's padded block
stride (38,016 B per block for GLM-5.3-Flash vs a dense 2,048 B tail block).
``_kpool_tail_seed_kernel`` addressed blocks densely
(``(blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM``), so a prefill seed wrote into an
unrelated indexer block and left the request's tail block untouched. The fix
passes ``tail.stride(0)`` / ``tail.stride(1)`` as TAIL_BLOCK_ELEMS / KPOOL_HEAD,
exactly as the other tail kernels in this file already do. Hunks are the
upstream commit's, byte for byte (source: vllm-project/vllm@db1bfdd4).

Target: GLM53_KPOOL_COMPRESS_PY (default: the image's
vllm/models/glm5next/nvidia/ops/kpool_compress.py). Idempotent; a partial or
drifted file fails before writing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KPOOL_COMPRESS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py",
    )
)

HUNKS = (
    ('    tslot_ptr,\n    tail_ptr,\n    n_tokens,\n    HEAD_DIM: tl.constexpr,\n    KPOOL: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n',
     '    tslot_ptr,\n    tail_ptr,\n    n_tokens,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n    KPOOL_HEAD: tl.constexpr,\n    HEAD_DIM: tl.constexpr,\n    KPOOL: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n'),
    ('    ahead belongs to a different tail block (or is past the batch / padding,\n    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is\n    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.\n    """\n    i = tl.program_id(0)\n    t = tl.load(tslot_ptr + i).to(tl.int64)\n',
     '    ahead belongs to a different tail block (or is past the batch / padding,\n    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is\n    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.\n\n    The tail cache aliases the indexer cache with the indexer\'s (padded) block\n    stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /\n    ``KPOOL_HEAD`` (``tail.stride(0)`` / ``tail.stride(1)``), never as a dense\n    ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array.\n    """\n    i = tl.program_id(0)\n    t = tl.load(tslot_ptr + i).to(tl.int64)\n'),
    ('        return\n    offs = tl.arange(0, BLOCK_D)\n    m = offs < HEAD_DIM\n    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)\n    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)\n    tl.store(tail_ptr + base + offs, k, mask=m)\n    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)\n\n\ndef kpool_seed_tail_cache(\n',
     '        return\n    offs = tl.arange(0, BLOCK_D)\n    m = offs < HEAD_DIM\n    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)\n    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)\n    tl.store(tail_ptr + base + offs, k, mask=m)\n    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)\n\n\ndef kpool_seed_tail_cache(\n'),
    (') -> None:\n    """Seed the paged tail cache from a prefill batch (see the kernel)."""\n    assert tail_kv_cache.dtype == torch.bfloat16\n    assert key.dtype == torch.bfloat16\n    n = tslot.shape[0]\n    if n == 0:\n',
     ') -> None:\n    """Seed the paged tail cache from a prefill batch (see the kernel)."""\n    assert tail_kv_cache.dtype == torch.bfloat16\n    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2\n    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == head_dim\n    assert key.dtype == torch.bfloat16\n    n = tslot.shape[0]\n    if n == 0:\n'),
    ('        tslot,\n        tail_kv_cache,\n        n,\n        HEAD_DIM=head_dim,\n        KPOOL=kpool,\n        BLOCK_D=triton.next_power_of_2(head_dim),\n',
     '        tslot,\n        tail_kv_cache,\n        n,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n        KPOOL_HEAD=tail_kv_cache.stride(1),\n        HEAD_DIM=head_dim,\n        KPOOL=kpool,\n        BLOCK_D=triton.next_power_of_2(head_dim),\n'),
)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    new_count = [text.count(new) for _, new in HUNKS]
    if all(n == 1 for n in new_count) and not any(old in text for old, _ in HUNKS):
        print(f"{P.name}: #57477 seed stride already present - verified")
        return 0
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit(f"{P}: #57477 anchor drifted (partial or unknown source)")
        text = text.replace(old, new, 1)
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name} (#57477 seed stride)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
