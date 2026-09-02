#!/usr/bin/env python3
"""Hash- and seam-bound GLM5-Next f_b/g_b batched projection patch."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

from _atomic import Refusal, execute, print_receipt, safe_target, sha_bytes
from _contracts import KDA_FG
from apply_kda_mixed import BASE_GZIP_B64, WRAPPER_GZIP_B64, unpack


KDA_REL = "vllm/models/glm5next/nvidia/kda.py"
MODEL_REL = "vllm/models/glm5next/nvidia/model.py"
MODULE_REL = "vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py"
BASE_REL = "vllm/model_executor/layers/quantization/kda_mixed_output_blocks_base.py"
KDA_BEFORE = "b5efb03327e5b03364a8b9a8019d097bea9ac6383b67e4a2bba8f7d3960b2231"
KDA_AFTER = "b262d0c3668c635fa6045968e1956fa5f0d9029fd52a645c1c75a1b6a412b29d"
MODEL_BEFORE = "6da99f9f192f617c0f2ab9f5c46b77025c2119ad2ac48f60136eb605aab04120"
MODEL_AFTER = "ce456edb8e62df26580d7fd5a844bef52862cbfe01eef766bd43f5032cb60577"
BASE_SHA = "b0a8eefb88d8d649d9729733bb4c7b0050ae69a87934a925e1308b921f360365"
WRAPPER_SHA = "01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5"

KDA_IMPORT_OLD = "from vllm.distributed import divide\n"
KDA_IMPORT_NEW = (
    "from vllm.distributed import divide, get_tensor_model_parallel_rank\n"
)
CLASS_ANCHOR = "\n\n@torch.compile(\n"
CLASS_TEXT = '''

class _Glm5NextBatchedColumnParallelLinear(nn.Module):
    """Two equal BF16 column-parallel projections executed by one bmm."""

    def __init__(
        self,
        batch: int,
        input_size: int,
        output_size: int,
        tp_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if batch != 2 or input_size != 128 or output_size != 8448 or tp_size != 3:
            raise ValueError("GLM5-Next TP3 f/g batched geometry drift")
        self.batch = batch
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.output_size_per_partition = divide(output_size, tp_size)
        self.weight = nn.Parameter(
            torch.empty(
                batch,
                self.output_size_per_partition,
                input_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input_, self.weight.transpose(-1, -2))

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_shard_id not in (0, 1):
            raise ValueError("f/g batched shard id must be 0 or 1")
        if tuple(param.shape) != (2, 2816, 128):
            raise ValueError("f/g batched parameter geometry drift")
        if tuple(loaded_weight.shape) == (8192, 128):
            loaded_weight = torch.cat(
                (loaded_weight, loaded_weight.new_zeros((256, 128))), dim=0
            )
        if tuple(loaded_weight.shape) != (8448, 128):
            raise ValueError("f/g checkpoint geometry drift")
        start = self.tp_rank * self.output_size_per_partition
        param.data[loaded_shard_id].copy_(
            loaded_weight.narrow(0, start, self.output_size_per_partition)
        )
'''

F_CONSTRUCTOR_OLD = '''        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
'''
F_CONSTRUCTOR_NEW = '''        self.fused_fg_b_proj = _Glm5NextBatchedColumnParallelLinear(
            batch=2,
            input_size=self.head_dim,
            output_size=projection_size,
            tp_size=self.tp_size,
            params_dtype=vllm_config.model_config.dtype,
        )
'''
G_CONSTRUCTOR_OLD = '''        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
'''
SPLIT_OLD = '''        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )
'''
SPLIT_NEW = '''        qkv, beta_raw, fg_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                2 * self.head_dim,
            ],
            dim=-1,
        )
'''
FORWARD_OLD = '''        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = self.g_b_proj(g_a)[0]
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)
'''
FORWARD_NEW = '''        g1, g_proj_states = self.fused_fg_b_proj(
            fg_a.view(-1, 2, self.head_dim).transpose(0, 1)
        )
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)
'''

MODEL_MAPPING_ANCHOR = (
    '            (".in_proj_qkvbfg_a", ".g_a_proj", 5),\n'
)
MODEL_MAPPING_LINES = (
    '            (".fused_fg_b_proj", ".f_b_proj", 0),\n'
    '            (".fused_fg_b_proj", ".g_b_proj", 1),\n'
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def patch_kda(path: Path, expected_before: str) -> tuple[str, str, str]:
    source = path.read_bytes()
    before = sha256_bytes(source)
    text = source.decode("utf-8")
    markers = (
        CLASS_TEXT.strip(),
        "self.fused_fg_b_proj = _Glm5NextBatchedColumnParallelLinear(",
        "g1, g_proj_states = self.fused_fg_b_proj(",
    )
    if all(text.count(marker) == 1 for marker in markers):
        compile(text, str(path), "exec")
        return "ALREADY_APPLIED", before, before
    if before != expected_before:
        raise SystemExit(
            "REFUSE: pristine KDA source SHA drift "
            f"expected={expected_before} observed={before}"
        )
    for old in (
        KDA_IMPORT_OLD,
        CLASS_ANCHOR,
        F_CONSTRUCTOR_OLD,
        G_CONSTRUCTOR_OLD,
        SPLIT_OLD,
        FORWARD_OLD,
    ):
        if text.count(old) != 1:
            raise SystemExit("REFUSE: KDA f/g patch seam count drift")
    text = text.replace(KDA_IMPORT_OLD, KDA_IMPORT_NEW, 1)
    text = text.replace(CLASS_ANCHOR, "\n" + CLASS_TEXT + CLASS_ANCHOR, 1)
    text = text.replace(F_CONSTRUCTOR_OLD, F_CONSTRUCTOR_NEW, 1)
    text = text.replace(G_CONSTRUCTOR_OLD, "", 1)
    text = text.replace(SPLIT_OLD, SPLIT_NEW, 1)
    text = text.replace(FORWARD_OLD, FORWARD_NEW, 1)
    if not all(text.count(marker) == 1 for marker in markers):
        raise SystemExit("REFUSE: KDA f/g patch marker mismatch")
    compile(text, str(path), "exec")
    patched = text.encode("utf-8")
    path.write_bytes(patched)
    return "PASS", before, sha256_bytes(patched)


def patch_model(path: Path, expected_before: str) -> tuple[str, str, str]:
    source = path.read_bytes()
    before = sha256_bytes(source)
    text = source.decode("utf-8")
    if text.count(MODEL_MAPPING_LINES) == 1:
        compile(text, str(path), "exec")
        return "ALREADY_APPLIED", before, before
    if before != expected_before:
        raise SystemExit(
            "REFUSE: overlayed model source SHA drift "
            f"expected={expected_before} observed={before}"
        )
    if text.count(MODEL_MAPPING_ANCHOR) != 1 or text.count(MODEL_MAPPING_LINES):
        raise SystemExit("REFUSE: model f/g mapping seam count drift")
    text = text.replace(
        MODEL_MAPPING_ANCHOR,
        MODEL_MAPPING_ANCHOR + MODEL_MAPPING_LINES,
        1,
    )
    if text.count(MODEL_MAPPING_LINES) != 1:
        raise SystemExit("REFUSE: model f/g mapping marker mismatch")
    compile(text, str(path), "exec")
    patched = text.encode("utf-8")
    path.write_bytes(patched)
    return "PASS", before, sha256_bytes(patched)


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
            model = safe_target(root, MODEL_REL)
            module = safe_target(root, MODULE_REL)
            base_module = safe_target(root, BASE_REL)
            outputs = {
                module: unpack(WRAPPER_GZIP_B64, WRAPPER_SHA),
                base_module: unpack(BASE_GZIP_B64, BASE_SHA),
            }
            with tempfile.TemporaryDirectory(prefix=".jspark3-kda-fg-", dir=root) as temporary:
                for source, expected_before, patcher in (
                    (kda, KDA_BEFORE, patch_kda),
                    (model, MODEL_BEFORE, patch_model),
                ):
                    candidate = Path(temporary) / source.name
                    candidate.write_bytes(before[source])
                    status, observed_before, observed_after = patcher(candidate, expected_before)
                    if status != "PASS" or observed_before != expected_before:
                        raise Refusal(f"unexpected f/g candidate state for {source}")
                    outputs[source] = candidate.read_bytes()
                    if observed_after != sha_bytes(outputs[source]):
                        raise Refusal(f"f/g candidate receipt mismatch for {source}")
            return outputs

        receipt = execute(
            root=root,
            contract_path=args.contract,
            receipt_path=args.image_receipt,
            transform="apply_kda_fg.py",
            expected_section=KDA_FG,
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
