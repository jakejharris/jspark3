"""S9.11 boot-only compilation of the GLM/DFlash traffic variants.

Triton wrappers run on storage-free meta tensors. Only JITFunction.warmup runs
on CUDA; no Triton kernel is launched and no scheduler, KV, RNG or prefix-cache
state is used. TileLang's dynamic-token mHC kernel runs on tiny private tensors.
Nothing is installed on the inference path. Errors propagate and fail startup.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import time
from types import SimpleNamespace

import torch


MODULES = {
    "rejection": "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
    "dflash": "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
    "gumbel": "vllm.v1.worker.gpu.sample.gumbel",
    "logprob": "vllm.v1.worker.gpu.sample.logprob",
    "topkp": "vllm.v1.sample.ops.topk_topp_triton",
    "norm": "vllm.third_party.flash_linear_attention.ops.l2norm",
    "kpool": "vllm.models.glm5next.nvidia.ops.kpool_compress",
    "conv": "vllm.model_executor.layers.mamba.ops.causal_conv1d",
}
KERNELS = {
    "rejection": ("_compute_local_logits_stats_kernel", "_rejection_kernel",
                  "_resample_kernel", "_insert_resampled_kernel"),
    "dflash": ("_prepare_dflash_inputs_kernel",),
    "gumbel": ("_gumbel_sample_kernel",),
    "logprob": ("_topk_log_softmax_kernel",),
    "topkp": ("_topk_topp_kernel",),
    "norm": ("l2norm_fwd_kernel2",),
    "kpool": ("_kpool_tail_seed_kernel",),
    "conv": ("_causal_conv1d_update_kernel",),
}


def tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def metadata(x):
    """Copy only shape/stride/dtype, never tensor data or its storage."""
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta")


def batch_classes(limit):
    # Triton's default integer specialization: ==1, divisible by 16, other.
    return tuple(n for n in (1, 2, 16) if n <= limit)


@contextmanager
def compile_only(modules, compile_launch):
    """Scope adapters to this synchronous worker boot; restore even on failure."""
    saved = []

    class Launch:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def call(*args, **kwargs):
                return compile_launch(self.kernel, grid, args, kwargs)
            return call

    try:
        for family, names in KERNELS.items():
            module = modules[family]
            for name in names:
                original = getattr(module, name)
                saved.append((module, name, original))
                setattr(module, name, Launch(original))
        # The top-k/p wrapper caches scratch tensors. Its meta cache is private
        # to this scope, never mixed into the serving device's scratch cache.
        for name in ("_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE"):
            saved.append((modules["topkp"], name, getattr(modules["topkp"], name)))
            setattr(modules["topkp"], name, {})
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def triton_cases(g, m):
    """Use the installed wrappers to derive every launch argument and stride."""
    i32, i64, f32 = torch.int32, torch.int64, torch.float32
    for dtype in (g.dtype, f32):
        # Greedy/sampled temperatures are tensor contents, not compile keys.
        # The no-draft pointer, logits dtype and output stride ARE compile keys.
        for spec in range(1, g.spec + 1):
            n = spec + 1
            for draft_dtype in (None, g.dtype, f32):
                m["rejection"].rejection_sample(
                    tensor(n, g.vocab, dtype=dtype),
                    None if draft_dtype is None else tensor(1, spec, g.vocab, dtype=draft_dtype),
                    # DFlash's sampled IDs are int32; live request mappings are
                    # np.intp/int64. Local positions and cumulative counts are int32.
                    tensor(n, dtype=i32), tensor(2, dtype=i32), tensor(n, dtype=i64),
                    tensor(1, dtype=i64), tensor(n, dtype=i64), tensor(n, dtype=i32),
                    tensor(g.max_reqs, dtype=f32), tensor(g.max_reqs, dtype=i64), spec,
                    use_fp64=g.fp64,
                )
        for apply_temperature in (False, True):
            # Target sampling has no cache; DFlash sampling uses per-token cols.
            for cache_cols in (None, (), (2,)):
                # Target mappings are int64; DFlash's sample mapping is int32.
                for mapping_dtype in (i32, i64):
                    m["gumbel"].gumbel_sample(
                        tensor(2, g.vocab, dtype=dtype), tensor(2, dtype=mapping_dtype),
                        tensor(g.max_reqs, dtype=f32), tensor(g.max_reqs, dtype=i64),
                        tensor(2, dtype=i64), apply_temperature,
                        logits_cache=None if cache_cols is None else tensor(g.max_reqs, g.spec, g.vocab, dtype=g.head_dtype),
                        logits_cache_col=None if cache_cols is None else tensor(*cache_cols, dtype=i32),
                        use_fp64=g.fp64,
                    )
        # Includes logprobs=0 (sampled column only), which the old =5 warmup misses.
        # max_logprobs=-1 means unlimited, not an empty range. Cover each capped
        # tile width with its ==1 / divisible-by-16 / other scalar specializations.
        # In particular 17 and the runtime top-20 + sampled column (21) share a key.
        for cols in sorted({1, 2, 3, 4, 5, 8, 9, 16, 17} | {
                2**k + delta for k in range(5, 11) for delta in (0, 1)}):
            m["logprob"].compute_token_logprobs(
                tensor(2, g.vocab, dtype=dtype), tensor(2, cols, dtype=i64))

    for n in batch_classes(g.max_reqs * (g.spec + 1)):
        for topk, topp in ((True, False), (False, True), (True, True)):
            m["topkp"].apply_top_k_top_p_triton(
                tensor(n, g.vocab, dtype=f32),
                tensor(n, dtype=i32) if topk else None,
                tensor(n, dtype=f32) if topp else None)

    # DFlash blocks depend on the longest CONTEXT+QUERY span, not batch size.
    # Enumerate the scheduler's entire token range and deduplicate block widths.
    spans = {}
    for n in range(1, g.max_tokens + 1):
        block = min(256, 1 << (n + g.dflash.query_width - 1).bit_length())
        spans.setdefault(block, n)
    d = g.dflash
    for context in spans.values():
        batch = SimpleNamespace(num_reqs=1,
                                num_scheduled_tokens=torch.tensor([context]),
                                positions=tensor(context, dtype=i64),
                                query_start_loc=tensor(2, dtype=i32),
                                # Live batch: np.intp -> async_copy_to_gpu -> int64.
                                # The OUTPUT sample_idx_mapping remains int32.
                                idx_mapping=tensor(1, dtype=i64))
        for table, block_size in d.tables:
            m["dflash"].prepare_dflash_inputs(
                d.buffers, tensor(d.max_tokens, dtype=i64),
                tensor(d.max_tokens, dtype=i64), tensor(d.max_tokens, dtype=i64),
                tensor(d.max_reqs * g.spec, dtype=i64),
                tensor(d.max_reqs * g.spec, dtype=i64),
                tensor(d.max_reqs * g.spec, dtype=i32),
                tensor(d.max_reqs, dtype=f32), tensor(d.max_reqs, dtype=i64), batch,
                tensor(1, dtype=i32), tensor(1, dtype=i32),
                tensor(d.max_reqs, dtype=i64), tensor(d.max_reqs, dtype=i32),
                tensor(d.max_reqs, dtype=f32), tensor(d.max_reqs, dtype=i64),
                table, block_size, d.mask_id, d.query_width, g.spec,
                d.max_reqs, d.max_tokens, d.max_model_len, False)

    for dim in g.head_dims:
        for n in batch_classes(g.max_tokens):
            for dtype in (g.dtype, f32):
                m["norm"].l2norm_fwd(tensor(n, dim, dtype=dtype))

    for tail in g.tails:
        _, _, pool, dim = tail.shape
        for n in batch_classes(g.max_tokens):
            # Mixed prefill starts after decode rows: int64 slots may be 8-byte
            # aligned while K/score rows remain aligned (head_dim multiple of 8).
            for offset in (0, 1):
                for score_dtype in (g.dtype, f32):
                    m["kpool"].kpool_seed_tail_cache(
                        tail, tensor(n, dim), tensor(n, dim, dtype=score_dtype),
                        tensor(n + offset, dtype=i64)[offset:], pool, dim)

    for c in g.convs:
        dim = c.weight.shape[0]
        # Explicitly enumerate ALL scheduler batch counts and verify widths,
        # including c4 mixed traffic; Triton deduplicates identical compile keys.
        for batch in range(1, g.max_reqs + 1):
            for row_stride in (dim, c.projection_stride):
                for index_stride in g.index_strides:
                    x = tensor(batch, row_stride, dtype=c.state.dtype)[:, :dim]
                    m["conv"].causal_conv1d_update(
                        x, c.state, c.weight, activation="silu",
                        conv_state_indices=tensor(batch, index_stride, dtype=i32)[:, 0])
                for width in range(1, g.spec + 2):
                    for index_stride in sorted(set(g.index_strides) | {width}):
                        x = tensor(batch * width, row_stride, dtype=c.state.dtype)[:, :dim]
                        m["conv"].causal_conv1d_update(
                            x, c.state, c.weight, activation="silu",
                            conv_state_indices=tensor(batch, index_stride, dtype=i32)[:, 0],
                            num_accepted_tokens=tensor(batch, dtype=i32),
                            query_start_loc=tensor(batch + 1, dtype=i32), max_query_len=width)


def geometry(worker):
    runner = worker.model_runner
    cfg = worker.vllm_config
    if not worker.use_v2_model_runner or runner.num_speculative_steps != 7:
        raise RuntimeError("S9.11 requires the sealed V2 DFlash seven-draft profile")
    if cfg.speculative_config.rejection_sample_method in ("synthetic", "block"):
        raise RuntimeError("S9.11 requires standard rejection sampling")
    layers = list(runner.get_model().modules())
    convs, tails, mhcs = {}, {}, {}
    for layer in layers:
        if type(layer).__name__ == "Glm5NextLinearAttention":
            state = layer.kv_cache[0]
            if not layer._conv_state_dim_first:
                state = state.transpose(-1, -2)
            weight = layer._merged_conv_weight
            if weight is None or layer.q_conv1d.bias is not None:
                raise RuntimeError("S9.11: KDA warmup/weight contract missing")
            stride = 3 * layer.local_projection_size + layer.local_num_heads + 2 * layer.head_dim
            key = (state.shape, state.stride(), state.dtype, weight.shape, weight.dtype, stride)
            convs[key] = SimpleNamespace(state=metadata(state), weight=metadata(weight),
                                        projection_stride=stride, head_dim=layer.head_dim)
        if type(layer).__name__ == "SparseAttnIndexerKpool" and layer.tail_cache is not None:
            tail = layer.tail_cache.kv_cache
            tails[(tail.shape, tail.stride(), tail.dtype)] = metadata(tail)
        if type(layer).__name__ == "Glm5NextDecoderLayer" and layer.mhc and not layer.is_mtp_layer:
            key = (layer.hidden_size, layer.n, layer.rms_norm_eps, layer.hc_eps,
                   layer.mhc_post_mult_value, layer.mhc_sinkhorn_iterations,
                   layer.input_layernorm.variance_epsilon)
            mhcs[key] = key
    if not convs or not tails or not mhcs:
        raise RuntimeError("S9.11: expected KDA, kpool and mHC model geometry missing")
    spec = runner.speculator
    dflash = SimpleNamespace(
        buffers=SimpleNamespace(**{name: metadata(getattr(spec.input_buffers, name))
                                  for name in ("input_ids", "positions", "query_start_loc", "seq_lens")}),
        tables=[(metadata(spec.block_tables.input_block_tables[gid]),
                 spec.block_tables.kernel_block_sizes[gid]) for gid in spec.draft_kv_cache_group_ids],
        query_width=spec.num_query_per_req, max_reqs=spec.max_num_reqs,
        max_tokens=spec.max_num_tokens, max_model_len=spec.max_model_len,
        mask_id=spec.parallel_drafting_token_id)
    return SimpleNamespace(
        dtype=cfg.model_config.dtype, head_dtype=cfg.model_config.head_dtype,
        vocab=cfg.model_config.get_vocab_size(),
        max_reqs=cfg.scheduler_config.max_num_seqs,
        max_tokens=cfg.scheduler_config.max_num_batched_tokens,
        max_logprobs=cfg.model_config.max_logprobs,
        spec=runner.num_speculative_steps, fp64=runner.sampler.use_fp64_gumbel,
        convs=list(convs.values()), tails=list(tails.values()), mhcs=list(mhcs),
        head_dims=sorted({c.head_dim for c in convs.values()}), dflash=dflash,
        index_strides=sorted({1, runner.num_speculative_steps + 1} |
                             {t.stride(0) for t in runner.block_tables.input_block_tables}))


def tilelang_cases(g, device):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split, mhc_pre_big_fuse_with_norm_tilelang)
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    count = 0
    for hidden, streams, rms, eps, post_mult, repeats, norm_eps in g.mhcs:
        splits = {compute_num_split(64, streams * hidden, (n + 63) // 64)
                  for n in range(1, g.max_tokens + 1)} if is_deep_gemm_supported() else {1}
        mix = streams * (streams + 2)
        def z(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)
        for split in sorted(splits):
            # num_tokens is T.dynamic in TileLang: one private row compiles the
            # same specialization as all prefill/decode rows at this split count.
            mhc_pre_big_fuse_with_norm_tilelang(
                z(split, 1, mix), z(split, 1), z(3), z(mix),
                z(1, streams, hidden, dtype=torch.bfloat16), z(1, streams),
                z(1, streams * streams), z(1, hidden, dtype=torch.bfloat16),
                z(hidden, dtype=torch.bfloat16), hidden, rms, eps, eps, post_mult,
                repeats, norm_eps, split, streams)
            count += 1
    return count


@torch.inference_mode()
def warmup(worker):
    from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
    from vllm.utils import jit_monitor
    if jit_monitor.is_active() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("S9.11 warmup must run before JIT monitor activation/admission")
    started = time.monotonic()
    g = geometry(worker)
    modules = {name: importlib.import_module(path) for name, path in MODULES.items()}
    counts = {}

    def compile_launch(kernel, grid, args, kwargs):
        def descriptor(value):
            if not isinstance(value, torch.Tensor):
                return value
            if value.device.type != "meta":
                raise RuntimeError("S9.11 compile-only launch received real tensor storage")
            return TritonWarmupTensor(value.dtype,
                                     aligned=(value.storage_offset() * value.element_size()) % 16 == 0)
        kernel.warmup(*(descriptor(a) for a in args), grid=grid, **kwargs)
        name = kernel.fn.__name__
        counts[name] = counts.get(name, 0) + 1

    with compile_only(modules, compile_launch):
        triton_cases(g, modules)
    tile_count = tilelang_cases(g, worker.device)
    torch.cuda.synchronize()
    print("S911_WARMUP_JIT=" + json.dumps({
        "status": "PASS", "rank": worker.rank, "triton_compile_only_calls": counts,
        "tilelang_private_calls": tile_count, "seconds": time.monotonic() - started,
        "prefix_cache_requests": 0, "max_num_seqs": g.max_reqs,
        "verify_widths": list(range(1, g.spec + 2)),
    }, sort_keys=True), flush=True)
