"""GLM-5.3-Flash over three ranks on one GPU, on a tiny synthetic checkpoint whose attention heads, KDA heads, expert
width, vocabulary and drafter heads do not divide by three (so every rank holds zero-padded shares and the last one
a shorter vocabulary slice).

- Rank 0 and rank 2 of three, their all-gathers handing back three copies of their own partials (as
  ``test_glm_engine``'s stand-in): CUDA graphs, MTP and DFlash2 drafts, drafted replies equal serial ones.
- Three real ranks as threads on the one GPU, each all-gather handing every rank all three partials in rank order
  (eager steps): the three-rank model's logits are the one-rank model's within bf16 rounding, and its drafted replies
  equal its serial ones bit for bit."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import split  # noqa: E402

D, V, S = 512, 1024, 4
H = LH = 4                # attention and KDA heads: 6 over three ranks, 2 a rank
MOE, DENSE = 256, 384     # expert width 384 over three ranks (128 a rank); the dense MLP divides (128 a rank)
CONFIG = {
    "model_type": "glm5_next",
    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    "text_config": {
        "hidden_size": D, "num_hidden_layers": 2, "vocab_size": V, "rms_norm_eps": 1e-5,
        "num_attention_heads": H, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": LH, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": MOE, "n_shared_experts": 1,
        "intermediate_size": DENSE, "routed_scaling_factor": 2.5, "norm_topk_prob": True, "hc_mult": S,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": 2, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "swiglu_limit": 10.0, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1,
    },
}
DRAFT = {
    "hidden_size": D, "head_dim": 128, "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 10000.0}, "sliding_window": 2048, "is_causal": False,
    "intermediate_size": 256, "num_hidden_layers": 1,
    "dflash_config": {"mask_token_id": 1001, "conv_group_size": 16, "conv_kernel_size": 2, "block_size": 8,
                      "selector_rank": 16, "selector_top_k": 8, "target_layer_ids": [0, 1]},
}


def _writer(seed: int):
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, str, list[int], np.ndarray]] = []

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    def f32(name: str, shape: list[int], scale: float = 0.1, offset: float = 0.0) -> None:
        x = (rng.standard_normal(shape) * scale + offset).astype(np.float32)
        tensors.append((name, "F32", shape, x.view(np.uint8).reshape(-1)))

    def q4(name: str, n: int, k: int, scale: float = 0.01) -> None:
        words = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32)
        tensors.append((name + ".weight", "U32", [n, k // 8], words.view(np.uint8).reshape(-1)))
        bf16(name + ".scales", [n, k // 64], 0.0, scale)
        bf16(name + ".biases", [n, k // 64], 0.0, -7.5 * scale)

    return tensors, bf16, f32, q4


def _checkpoint(path) -> None:
    """The synthetic model as an MLX 4-bit checkpoint (1 KDA + 1 DSA layer, MoE, the MTP head)."""

    tensors, bf16, f32, q4 = _writer(3)

    def dsa(p: str) -> None:
        q4(p + "self_attn.q_a_proj", 128, D)
        q4(p + "self_attn.kv_a_proj_with_mqa", 128, D)
        bf16(p + "self_attn.q_a_layernorm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.kv_a_layernorm.weight", [128], 0.05, 1.0)
        q4(p + "self_attn.q_b_proj", H * 256, 128)
        q4(p + "self_attn.kv_b_proj", H * 512, 128)
        q4(p + "self_attn.o_proj", D, H * 256)
        q4(p + "self_attn.indexer.wk", 128, D)
        q4(p + "self_attn.indexer.weights_proj", 2, D)
        q4(p + "self_attn.indexer.wq_b", 2 * 128, 128)
        bf16(p + "self_attn.indexer.k_norm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.indexer.k_norm.bias", [128])
        bf16(p + "self_attn.indexer.index_kpool_compress_gate", [128, D])
        bf16(p + "self_attn.indexer.index_kpool_compress_ape", [4, 128])

    def moe(p: str) -> None:
        bf16(p + "mlp.gate.weight", [8, D])
        f32(p + "mlp.gate.e_score_correction_bias", [8], 0.01)
        for e in [f"experts.{i}" for i in range(8)] + ["shared_experts"]:
            q4(p + f"mlp.{e}.gate_proj", MOE, D)
            q4(p + f"mlp.{e}.up_proj", MOE, D)
            q4(p + f"mlp.{e}.down_proj", D, MOE)

    L = "model.language_model."
    q4(L + "embed_tokens", V, D, 0.02)
    bf16(L + "norm.weight", [D], 0.05, 1.0)
    q4("lm_head", V, D)
    for i in (0, 1):
        p = f"{L}layers.{i}."
        bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
        bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
        for site in ("attn", "ffn"):
            bf16(p + f"hc_{site}_fn", [24, S * D], 0.01)
            f32(p + f"hc_{site}_base", [24])
            f32(p + f"hc_{site}_scale", [3], 0.1, 1.0)
    p = L + "layers.0.self_attn."
    for x in "qkv":
        q4(p + f"{x}_proj", LH * 128, D)
        bf16(p + f"{x}_conv1d.weight", [LH * 128, 1, 4], 0.3)
    q4(p + "f_a_proj", 128, D)
    q4(p + "g_a_proj", 128, D)
    q4(p + "b_proj", LH, D)
    q4(p + "f_b_proj", LH * 128, 128)
    q4(p + "g_b_proj", LH * 128, 128)
    f32(p + "A_log", [LH], 0.5)
    f32(p + "dt_bias", [LH * 128], 0.5)
    bf16(p + "o_norm.weight", [128], 0.05, 1.0)
    q4(p + "o_proj", D, LH * 128)
    q4(L + "layers.0.mlp.gate_proj", DENSE, D)
    q4(L + "layers.0.mlp.up_proj", DENSE, D)
    q4(L + "layers.0.mlp.down_proj", D, DENSE)
    dsa(L + "layers.1.")
    moe(L + "layers.1.")
    m = L + "layers.2."
    bf16(m + "enorm.weight", [D], 0.05, 1.0)
    bf16(m + "hnorm.weight", [D], 0.05, 1.0)
    q4(m + "eh_proj", D, 2 * D)
    bf16(m + "shared_head.norm.weight", [D], 0.05, 1.0)
    bf16(m + "input_layernorm.weight", [D], 0.05, 1.0)
    bf16(m + "post_attention_layernorm.weight", [D], 0.05, 1.0)
    dsa(m)
    moe(m)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model-00001-of-00001.safetensors"), tensors, {"format": "mlx"})
    (path / "config.json").write_text(json.dumps(CONFIG))


def _drafter(path) -> None:
    """A one-layer DFlash2 drafter with 8 query / 2 key-value heads: 12 / 3 over three ranks, 4 / 1 a rank."""

    tensors, bf16, _, _ = _writer(4)
    Hq, KV, hd, inter = 8, 2, 128, 256
    bf16("fc.weight", [D, 2 * D], 0.03)
    bf16("hidden_norm.weight", [D], 0.05, 1.0)
    bf16("norm.weight", [D], 0.05, 1.0)
    bf16("candidate_selector.hidden_projection.weight", [16, D], 0.05)
    bf16("candidate_selector.predecessor_codebook", [V, 16], 0.3)
    bf16("candidate_selector.successor_codebook", [V, 16], 0.3)
    p = "layers.0."
    bf16(p + "self_attn.q_proj.weight", [Hq * hd, D], 0.03)
    bf16(p + "self_attn.k_proj.weight", [KV * hd, D], 0.03)
    bf16(p + "self_attn.v_proj.weight", [KV * hd, D], 0.03)
    bf16(p + "self_attn.o_proj.weight", [D, Hq * hd], 0.03)
    bf16(p + "self_attn.q_norm.weight", [hd], 0.05, 1.0)
    bf16(p + "self_attn.k_norm.weight", [hd], 0.05, 1.0)
    bf16(p + "mlp.gate_proj.weight", [inter, D], 0.03)
    bf16(p + "mlp.up_proj.weight", [inter, D], 0.03)
    bf16(p + "mlp.down_proj.weight", [D, inter], 0.03)
    for conv in ("attention_conv", "mlp_conv"):
        bf16(p + conv + ".base_kernel", [2, 2, D], 0.1, 0.5)
        bf16(p + conv + ".kernel_projection.weight", [4 * D // 16, D], 0.02)
    bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
    bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model.safetensors"), tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps(DRAFT))


class _Copies:
    """Rank ``rank`` of three on one GPU: every all-gather returns this rank's input three times."""

    world = 3

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(send.reshape(-1))

    def barrier(self) -> None:
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_tp3")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


def _engine(model, rank: int, drafter: bool, drafter_bits: int = 4):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(model / "model", rank=rank, master="", port=0, world=3, comm=_Copies(rank),
                     drafter=model / "dflash2" if drafter else None, drafter_bits=drafter_bits)


@pytest.fixture(scope="module")
def engine0(model):
    return _engine(model, 0, True)


@pytest.fixture(scope="module")
def engine2(model):
    return _engine(model, 2, False)


def _generate(engine, prompt, sampling, *, draft=True, policy=None, tokens=24):
    out: list[int] = []
    engine.request.policy = policy
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    return out, stats


def test_every_rank_holds_whole_padded_shares(engine0, engine2):
    for e, rank in ((engine0, 0), (engine2, 2)):
        w = e.w
        kda, dsa = w.layers[0].kda, w.layers[1].dsa
        assert (w.world, w.rank) == (3, rank)
        assert kda.heads == dsa.heads == w.mtp.layer.dsa.heads == 2
        assert kda.proj.n % 64 == 0 and w.layers[1].moe.experts.width == 128 and w.layers[0].mlp.width == 128
        assert w.head.n == (384 if rank < 2 else 256) and w.vocab_offset == 384 * rank
    assert (engine0.drafter.heads, engine0.drafter.kvh, engine0.drafter.inter) == (4, 1, 128)


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
@pytest.mark.parametrize("rank", [0, 2])
def test_drafted_replies_equal_serial_at_three_ranks(engine0, engine2, rank, sampling):
    engine = engine0 if rank == 0 else engine2
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=37))
    serial, stats = _generate(engine, prompt, sampling, draft=False)
    assert len(serial) == 24 and stats["drafts"] is False
    for policy in (None, "auto", "1", "2", "3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine, prompt, sampling, policy=policy)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_dflash2_drafts_equal_serial_at_three_ranks(engine0, sampling):
    prompt = list(np.random.default_rng(6).integers(0, 1000, size=41))
    serial, _ = _generate(engine0, prompt, sampling, draft=False, tokens=40)
    for policy in (None, "auto:1:2:0", "auto:1:1:0", "f3", "fc5:0.3", "2", "f7", "fc7:0.3"):
        drafted, stats = _generate(engine0, prompt, sampling, policy=policy, tokens=40)
        assert drafted == serial, policy
        assert len(stats["depths"]) == len(stats["keeps"]) > 0, policy     # each round's drafts and kept tokens
    assert max(_generate(engine0, prompt, sampling, policy="f7", tokens=40)[1]["depths"]) == 7


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_a_bf16_drafter_drafts_exactly_at_three_ranks(model, sampling):
    from tensorfold.families.glm5_next.cuda import qmm

    engine = _engine(model, 0, True, drafter_bits=16)
    assert all(isinstance(q, qmm.B16) for L in engine.drafter.layers for q in (L.qkv, L.kv, L.o, L.gu, L.down))
    prompt = list(np.random.default_rng(6).integers(0, 1000, size=41))
    serial, _ = _generate(engine, prompt, sampling, draft=False, tokens=40)
    for policy in (None, "f7", "fc7:0.3"):
        assert _generate(engine, prompt, sampling, policy=policy, tokens=40)[0] == serial, policy


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
@pytest.mark.parametrize("policy", ["fc7:0.3", "auto"])
def test_dflash2_conversation_resumes_after_an_unrelated_request(model, monkeypatch, sampling, policy):
    from tensorfold.families.glm5_next.cuda import decode

    engine = _engine(model, 0, True)
    rng = np.random.default_rng(22)
    prompt = [int(t) for t in rng.integers(0, 1000, size=137)]
    unrelated = [int(t) for t in rng.integers(0, 1000, size=43)]
    d = engine.drafter
    proposals = []
    propose = d.propose

    def record(*args, **kwargs):
        result = propose(*args, **kwargs)
        proposals.append(list(result))
        return result

    monkeypatch.setattr(d, "propose", record)
    reply, _ = _generate(engine, prompt, sampling, policy=policy)
    after = prompt + reply + [21, 22]
    want = [t[:, :len(prompt)].clone() for t in (*d.kc, *d.vc)]
    restored = []
    restore = decode.restore

    def check_restore(e, snap, drafter=None):
        restore(e, snap, drafter)
        if snap.ids == prompt:
            assert drafter is d and d.context_end == len(prompt) and d.pos_dev.item() == len(prompt)
            for got, expected in zip((*d.kc, *d.vc), want, strict=True):
                assert torch.equal(got[:, :len(prompt)].contiguous().view(torch.uint8),
                                   expected.contiguous().view(torch.uint8))
            restored.append(len(prompt))

    monkeypatch.setattr(decode, "restore", check_restore)
    proposals.clear()
    uninterrupted, stats = _generate(engine, after, sampling, policy=policy)
    assert stats["cached"] == len(prompt)
    uninterrupted_proposals = list(proposals)
    # Recreate A, then hand its live caches to B before A's next turn.
    engine.cache.clear()
    engine.live = []
    assert _generate(engine, prompt, sampling, policy=policy)[0] == reply
    _generate(engine, unrelated, sampling, policy=policy)
    saved = next(s for s in engine.cache if s.ids == prompt)
    assert saved.rows is not None and saved.drafter_end == len(prompt)
    proposals.clear()
    resumed, stats = _generate(engine, after, sampling, policy=policy)
    assert stats["cached"] == len(prompt)
    assert restored == [len(prompt), len(prompt)]
    assert resumed == uninterrupted
    if policy == "fc7:0.3":                         # auto can choose different arms from elapsed timings
        assert proposals == uninterrupted_proposals and proposals
    serial, stats = _generate(engine, after, sampling, draft=False)
    assert stats["cached"] == 0 and resumed == serial


class _Hub:
    """``world`` ranks as threads on one GPU: an all-gather gives every rank all ranks' inputs in rank order."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.sends: list[torch.Tensor | None] = [None] * world
        self.gate = threading.Barrier(world, timeout=120)


class _Member:
    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        hub, n = self.hub, send.numel()
        torch.cuda.synchronize()
        hub.sends[self.rank] = send.reshape(-1).clone()
        torch.cuda.synchronize()
        hub.gate.wait()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(hub.sends[r])
        torch.cuda.synchronize()
        hub.gate.wait()

    def barrier(self) -> None:
        torch.cuda.synchronize()
        self.hub.gate.wait()


def _run_ranks(model, world: int, job) -> list:
    """``job(engine, rank)`` on every rank of ``world`` at once (eager decode engines), results in rank order."""

    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.weights import load

    hub = _Hub(world)
    engines = []
    for rank in range(world):
        w = load(model / "model", rank=rank, world=world)
        w.comm = _Member(hub, rank) if world > 1 else None
        engines.append(Engine(w, capacity=2560, max_rows=8, prefill_rows=64))
    out: list = [None] * world
    errors: list = []

    def run(rank: int) -> None:
        try:
            with torch.no_grad():
                out[rank] = job(engines[rank], rank)
        except BaseException as exc:  # noqa: BLE001 - reported below, other ranks are released
            errors.append(exc)
            hub.gate.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:           # the failing rank's error, not the broken barrier it left the others
        raise next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0])
    return out


def test_three_real_ranks_are_the_one_rank_model_and_draft_exactly(model, engine0):
    """engine0 first: kernels and extensions are built before three threads use them."""

    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy, mtp_decode, prefill, serial_decode
    from tensorfold.families.glm5_next.cuda.forward import forward

    prompt = [int(t) for t in np.random.default_rng(7).integers(0, 1000, size=29)]

    def logits(e, rank):
        prefill(e, prompt[:-1], None)
        return forward(e.w, e.st, e.buf, prompt[-1:])[:1].float().cpu()      # a decode step's row, eager

    one = _run_ranks(model, 1, logits)[0]
    three = torch.cat(_run_ranks(model, 3, logits), dim=1)
    assert three.shape == one.shape == (1, V)
    assert float(torch.nn.functional.cosine_similarity(one, three)) > 0.999

    def decode(e, rank):
        sampling = Sampling(99, 1.0, 20, 0.95)
        runs = []
        for s in (None, sampling):
            serial = serial_decode(e, prefill(e, prompt, s), 24, s).tokens
            drafted = mtp_decode(e, prefill(e, prompt, s), 24, s, policy=DepthPolicy(3, fixed=True)).tokens
            runs.append((serial, drafted))
        return runs

    ranks = _run_ranks(model, 3, decode)
    assert ranks[0] == ranks[1] == ranks[2]                     # every rank samples the same tokens
    for serial, drafted in ranks[0]:
        assert len(serial) == 24 and drafted == serial
