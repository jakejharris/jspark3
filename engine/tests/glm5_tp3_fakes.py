"""A tiny GLM-5.3-Flash-shaped MLX 4-bit checkpoint (random words, written without MLX) whose attention heads, KDA
heads, MoE width and vocabulary do not divide by three; the dense MLP does."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfold.families.glm5_next.cuda import split

D, V, H, LH, MOE, DENSE = 128, 1000, 4, 4, 128, 384
TEXT = {
    "model_type": "glm5_next_text", "hidden_size": D, "num_hidden_layers": 2, "vocab_size": V, "rms_norm_eps": 1e-5,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"], "mlp_layer_types": ["dense", "sparse"],
    "first_k_dense_replace": 1, "intermediate_size": DENSE, "moe_intermediate_size": MOE, "n_routed_experts": 4,
    "num_experts_per_tok": 2, "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
    "swiglu_limit": 10.0, "num_attention_heads": H, "q_lora_rank": 64, "kv_lora_rank": 512, "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0, "v_head_dim": 256, "index_n_heads": 2, "index_head_dim": 128, "index_topk": 16,
    "index_kpool": 4,
    "linear_attn_config": {"num_heads": LH, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
    "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20, "num_nextn_predict_layers": 1, "eos_token_id": [5],
}


def tensors(seed: int = 0) -> dict[str, tuple[str, list[int], np.ndarray]]:
    """name -> (safetensors dtype, shape, values: uint32 words or float32 to store as BF16/F32)."""

    rng = np.random.default_rng(seed)
    t: dict = {}
    c = TEXT

    def q(name: str, n: int, k: int) -> None:
        t[name + ".weight"] = ("U32", [n, k // 8], rng.integers(0, 2 ** 32, size=(n, k // 8), dtype=np.uint64)
                               .astype(np.uint32))
        for part in ("scales", "biases"):
            t[f"{name}.{part}"] = ("BF16", [n, k // 64], (0.01 * rng.standard_normal((n, k // 64))).astype(np.float32))

    def bf(name: str, shape: list[int]) -> None:
        t[name] = ("BF16", list(shape), (0.05 * rng.standard_normal(shape)).astype(np.float32))

    def f32(name: str, shape: list[int]) -> None:
        t[name] = ("F32", list(shape), (0.1 * rng.standard_normal(shape)).astype(np.float32))

    def mla(p: str) -> None:
        ql, kl = c["q_lora_rank"], c["kv_lora_rank"]
        q(p + ".q_a_proj", ql, D)
        q(p + ".kv_a_proj_with_mqa", kl, D)
        bf(p + ".q_a_layernorm.weight", [ql])
        bf(p + ".kv_a_layernorm.weight", [kl])
        q(p + ".q_b_proj", H * 256, ql)
        q(p + ".kv_b_proj", H * 512, kl)
        q(p + ".o_proj", D, H * 256)
        q(p + ".indexer.wk", 128, D)
        q(p + ".indexer.weights_proj", 2, D)
        q(p + ".indexer.wq_b", 2 * 128, ql)
        bf(p + ".indexer.k_norm.weight", [128])
        bf(p + ".indexer.k_norm.bias", [128])
        bf(p + ".indexer.index_kpool_compress_gate", [128, D])
        bf(p + ".indexer.index_kpool_compress_ape", [4, 128])

    def kda(p: str) -> None:
        for x in "qkv":
            q(p + f".{x}_proj", LH * 128, D)
            bf(p + f".{x}_conv1d.weight", [LH * 128, 1, 4])
        q(p + ".f_a_proj", 128, D)
        q(p + ".g_a_proj", 128, D)
        q(p + ".b_proj", LH, D)
        q(p + ".f_b_proj", LH * 128, 128)
        q(p + ".g_b_proj", LH * 128, 128)
        f32(p + ".A_log", [LH])
        f32(p + ".dt_bias", [LH * 128])
        bf(p + ".o_norm.weight", [128])
        q(p + ".o_proj", D, LH * 128)

    def mlp(p: str, sparse: bool) -> None:
        if not sparse:
            q(p + ".gate_proj", DENSE, D)
            q(p + ".up_proj", DENSE, D)
            q(p + ".down_proj", D, DENSE)
            return
        bf(p + ".gate.weight", [c["n_routed_experts"], D])
        f32(p + ".gate.e_score_correction_bias", [c["n_routed_experts"]])
        for e in [f"experts.{i}" for i in range(c["n_routed_experts"])] + ["shared_experts"]:
            q(f"{p}.{e}.gate_proj", MOE, D)
            q(f"{p}.{e}.up_proj", MOE, D)
            q(f"{p}.{e}.down_proj", D, MOE)

    pre, n = "model.language_model", c["num_hidden_layers"]
    q(f"{pre}.embed_tokens", V, D)
    q("lm_head", V, D)
    bf(f"{pre}.norm.weight", [D])
    for i in range(n + 1):                      # layer n: the MTP head (MLA + MoE, plain residual)
        p = f"{pre}.layers.{i}"
        (kda if i < n and c["layer_types"][i] == "linear_attention" else mla)(p + ".self_attn")
        mlp(p + ".mlp", i >= n or c["mlp_layer_types"][i] == "sparse")
        bf(p + ".input_layernorm.weight", [D])
        bf(p + ".post_attention_layernorm.weight", [D])
        if i < n:
            for s in ("attn", "ffn"):
                bf(p + f".hc_{s}_fn", [24, 4 * D])
                f32(p + f".hc_{s}_base", [24])
                f32(p + f".hc_{s}_scale", [3])
        else:
            q(p + ".eh_proj", D, 2 * D)
            for norm in ("enorm", "hnorm", "shared_head.norm"):
                bf(p + f".{norm}.weight", [D])
    return t


def stored(dtype: str, values: np.ndarray) -> np.ndarray:
    """Values as the checkpoint stores them (BF16: float32's top half), in bytes."""

    if dtype == "BF16":
        return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1)
    return np.ascontiguousarray(values).view(np.uint8).reshape(-1)


def write(folder: Path, seed: int = 0) -> Path:
    """The checkpoint in two files with its index and config; returns ``folder``."""

    t = tensors(seed)
    names = sorted(t)
    shards = {"model-00001-of-00002.safetensors": names[:len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    folder.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard, keys in shards.items():
        split.write(str(folder / shard), [(k, t[k][0], t[k][1], stored(t[k][0], t[k][2])) for k in keys],
                    {"format": "mlx"})
        weight_map.update({k: shard for k in keys})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": TEXT,
                                                    "quantization": {"bits": 4, "group_size": 64}}))
    return folder
