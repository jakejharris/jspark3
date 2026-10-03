"""GLM-5.3-Flash's tensor-parallel forward and commit; kernels keep rows apart, so row r has the serial step's bits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.kernels import prefill_attention, qmm as shared

from . import glue, kda as kda_mod, l2_prefetch, latent, prefill_options as p1, prof, qmm, seqpar, sparse, tp3_probe
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.rows, self.prefill = rows, prefill
        self.l2_prefetch = l2_prefetch.make(w, prefill=prefill)
        head_rows = 1 if prefill else rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        if latent.ENABLED:
            # Dense attention only ever covers contexts up to the dense limit; longer rows go sparse.
            self.attn = None
            self.lat_s = latent.LatentScratch(rows, HL, latent.chunks_for(min(capacity, 2560) + rows), dev,
                                              lw=c.kv_lora,
                                              part_rows=p1.ATTENTION_ROWS if prefill and p1.ATTENTION_TILES else
                                              p1.PROMPT_ATT_ROWS if prefill and p1.PROMPT_SCRATCH else rows)
        else:
            self.attn = AttnScratch(1 if prefill else rows, HL, c.qk_dim, capacity, dev)
        if prefill:
            kda_layers = [l for l in w.layers if l.kind == "kda"]
            width = kda_layers[0].kda.proj.n if kda_layers else 0
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
        self.hin = torch.empty((rows, c.hidden), dtype=torch.bfloat16, device=dev)      # MTP input rows
        self.zero_first = False          # MTP: this step starts at position 0 (its embedding is zeroed)
        self.zero_rows: list[int] = []   # MTP rows at position 0 (their embeddings are zeroed)
        self.overlay = None              # prompt chunks: (rows, image rows) replacing those rows' embeddings (eager)
        self.x = torch.empty((rows, S * D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.xs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.post = torch.empty((rows, S), dtype=f32, device=dev)
        self.comb = torch.empty((rows, S * S), dtype=f32, device=dev)
        self.hcpart = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=f32, device=dev)
        # KDA
        self.ka = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.xs_fa = torch.empty((rows, 2), dtype=f32, device=dev)
        self.xs_ga = torch.empty((rows, 2), dtype=f32, device=dev)
        self.kxs = torch.empty((rows, LL * 128 // 64), dtype=f32, device=dev)
        if not prefill:                  # several streams' rows: the projection rows and chain outputs of all of them
            kl = [l for l in w.layers if l.kind == "kda"]
            self.mproj = torch.zeros((rows, kl[0].kda.proj.n if kl else 0), dtype=bf, device=dev)
            self.kout = torch.zeros((rows, LL * 128), dtype=bf, device=dev)
        # DSA
        self.dp = torch.empty((rows, c.q_lora + c.kv_lora), dtype=bf, device=dev)
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.xs_qr = torch.empty((rows, c.q_lora // 64), dtype=f32, device=dev)
        self.lat = torch.empty((rows, c.kv_lora), dtype=bf, device=dev)
        self.xs_lat = torch.empty((rows, c.kv_lora // 64), dtype=f32, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.kn = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.xs_ao = torch.empty((rows, HL * c.v_dim // 64), dtype=f32, device=dev)
        # DSA indexer (long contexts)
        self.ikr = torch.empty((rows, c.index_dim + c.index_heads), dtype=bf, device=dev)
        self.igr = torch.empty((rows, c.index_dim), dtype=f32, device=dev)
        self.qi = torch.empty((rows, c.index_heads * c.index_dim), dtype=bf, device=dev)
        # dense MLP
        dl = c.dense_width // w.world
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, dl // 64), dtype=f32, device=dev)
        # MoE
        slots = c.top_k + 1
        ml = c.moe_width // w.world
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        self.eact = torch.empty((rows * slots, ml), dtype=bf, device=dev)
        exl3 = c.quant == "exl3"
        self.ey = torch.empty((rows, slots, D), dtype=bf if prefill and not exl3 else f32, device=dev)
        self.plan = grouped.Plan(rows, slots, c.experts + 1, dev, prefill=prefill and not exl3)
        self.exl3 = None
        if c.quant == "exl3":            # EXL3 routed experts, and the shared expert as a BF16 MLP
            from .exl3_mm import Scratch

            sl = c.shared_width // w.world
            self.exl3 = Scratch(rows, slots, D, ml, dev)
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sp: seqpar.Plan | None = None    # TF_GLM_SP: this pass's row shares (set and cleared by compute_streams)
        self.defer = False               # TF_GLM_SP_OVERLAP: a half's view, whose blocks leave b.part unreduced
        self.sk = torch.empty((1 if prefill and not exl3 else 8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mxs = torch.empty((rows, 2 * D // 64), dtype=f32, device=dev)
        self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        self._parents: dict[int, torch.Tensor] = {}
        # DFlash2 taps: the mean of the streams after chosen layers (``set_taps``), filled by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.experts = c.experts
        self.top_k = c.top_k

    # TF_GLM_SP_OVERLAP's half views: these are sliced to the half's rows; the rest is scratch only the main stream's
    # blocks use, one half after the other (tests check every tensor is in one list)
    ROWS = ("ids", "hin", "x", "normed", "xs", "post", "comb", "hcpart", "ka", "kg", "xs_fa", "xs_ga", "kxs", "mproj",
            "kout", "dp", "qr", "xs_qr", "lat", "xs_lat", "q", "kn", "vn", "xs_ao", "ikr", "igr", "qi", "gu", "act",
            "xs_act", "mlog", "pick", "wts", "ey", "sgu", "sact", "sxs", "sy", "part", "hidden", "fnormed", "fxs", "me",
            "mcat", "mxs", "mx")
    SHARED = ("ids_host", "kscratch", "lat_s", "attn", "eact", "plan", "exl3", "sk", "logits", "staged",
              "cofill_kout")         # cofill_kout: several prompt streams only, never a half

    def half(self, a0: int, a1: int) -> "Buffers":
        """Rows a0:a1 as a pass of their own: row views, b.gath from offset world x a0 x D (the halves' reduce
        outputs never overlap), the shared scratch, and blocks that defer their reduce."""

        import copy

        v = copy.copy(self)
        for name in self.ROWS:
            t = getattr(self, name, None)
            if t is not None:
                setattr(v, name, t[a0:a1])
        if hasattr(self, "kproj"):
            v.kproj = self.kproj[:, a0:a1]
        v.taps = [t[a0:a1] for t in self.taps]
        v.gath = self.gath[self.world * a0 * self.part.shape[1]:]
        v.rows, v.sp, v.defer, v._parents = a1 - a0, None, True, {}
        return v

    def set_taps(self, layers: tuple[int, ...], hidden: int) -> None:
        self.tap_at = {}
        for i, layer in enumerate(layers):
            self.tap_at.setdefault(layer, []).append(i)
        self.taps = [torch.empty((self.rows, hidden), dtype=torch.bfloat16, device=self.ids.device) for _ in layers]

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer)."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.conv = torch.zeros((n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch_set = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.scratch = self.scratch_set.views if n else []
        self.latent = latent.ENABLED
        if self.latent:              # one 512-wide latent a token and layer (kc), no separate values (vc)
            self.kc = [torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
            self.vc = [None for _ in dsa_layers]
        else:
            self.kc = [torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
            self.vc = [torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            if self.latent:
                self.mtp_kc = torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)
                self.mtp_vc = None
            else:
                self.mtp_kc = torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev)
                self.mtp_vc = torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev)
        # DSA indexer caches (long contexts only): per layer (and the MTP layer, last) keys, gates, pool keys
        self.index = None
        if w.meta.get("long_context"):
            n_idx = len(dsa_layers) + (1 if w.mtp is not None else 0)
            mk = lambda n: torch.zeros((n, c.index_dim), dtype=torch.bfloat16, device=dev)   # noqa: E731
            self.index = [(mk(capacity), mk(capacity), mk(capacity // 4 + 2)) for _ in range(n_idx)]

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def shifted(self, k: int) -> "State":
        """These caches seen k rows on (TF_GLM_SP_OVERLAP's second half): its own positions, all else shared."""

        import copy

        other = copy.copy(self)
        other.pos, other.pos_dev = self.pos + k, self.pos_dev + k
        return other

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() if x is not None else None for x in self.vc]
        if self.index is not None:
            other.index = [tuple(x.clone() for x in trio) for trio in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone() if self.mtp_vc is not None else None
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor | seqpar.Reduced | None:
    """Every rank's fp32 partial b.part[:R] in rank order: [world, R, D] (summed rank 0 first by the consumer;
    TF_GLM_SP: only this rank's rows, a ``seqpar.Reduced``).

    TF_TP3_REDUCE=nccl (a probe, ``tp3_probe``): NCCL's ring all-reduce sums them instead, [1, R, D]. A half of a
    pipelined pass (TF_GLM_SP_OVERLAP) returns None: its tail reduces on the side stream (``reduced``)."""

    return None if b.defer else reduced(w, b, R)


def reduced(w: Weights, b: Buffers, R: int) -> torch.Tensor | seqpar.Reduced:
    d = b.part.shape[1]
    if w.comm is None:
        return b.part[:R].view(1, R, d)
    rec = tp3_probe.recorder
    pre = b.part[:R].clone() if rec is not None and rec.calls is not None else None
    hint = getattr(b, "l2_prefetch", None)
    parent = hint.start() if hint is not None else None
    try:
        if b.sp is not None:             # also bracket the overlap half's side-stream reduce
            return seqpar.reduce(w, b, R)
        if tp3_probe.REDUCE == "nccl":
            w.comm.all_reduce(b.part[:R])
            out = b.part[:R].view(1, R, d)
        else:
            out = b.gath[:b.world * R * d]
            w.comm.all_gather(b.part[:R].reshape(-1), out)
            out = out.view(b.world, R, d)
    finally:
        if hint is not None:
            hint.finish(parent)
    if pre is not None:
        rec.record(pre, out)
    return out


def mm(b: Buffers, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False) -> torch.Tensor:
    """A projection: 4-bit ones of a prompt chunk on the shared prefill matmul, the rest on ``qmm.matmul``."""

    if b.prefill and isinstance(q, qmm.Q4):
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return qmm.matmul(x, q, xs, out=out, f32=f32, part=b.sk)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int) -> torch.Tensor:
    mm(b, x, q, xs, b.part[:R], f32=True)
    return gather(w, b, R)


def kda_block(layer: LayerW, w: Weights, segs: Sequence, b: Buffers, R: int) -> torch.Tensor:
    """segs: (state, a0, a1) of each stream's rows; one stream's rows project into its state, several into b.mproj."""

    c = w.cfg
    k = layer.kda
    st0 = segs[0][0]
    li = st0.kda_index[layer.index]
    one = len(segs) == 1
    p = b.kproj[0, :R] if b.prefill else st0.proj[li, :R] if one else b.mproj[:R]
    mm(b, b.normed[:R], k.proj, b.xs[:R], p)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    pre = b.prefill
    mm(b, fa, k.fb, None if pre else qmm.group_sums(fa, b.xs_fa[:R]), b.ka[:R])
    mm(b, ga, k.gb, None if pre else qmm.group_sums(ga, b.xs_ga[:R]), b.kg[:R])
    if one:
        st = st0
        cur = st.cur[li]
        out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log, k.dt_bias,
                            k.norm, c.eps, c.lower, R, b.kscratch if pre else st.scratch[li], st.rec[1 - cur, li])
        if pre:                          # a prompt chunk keeps every row: the layer commits now
            st.cur[li] = 1 - cur
            _shift_conv(st.conv[li:li + 1], b.kproj[:, :R], R)
    elif pre:
        # Co-prefill: use each prompt's existing prefill chain/scratch and commit its
        # layer exactly as the one-stream branch does. State.proj only has eight
        # decode rows; it must never receive a prompt segment here.
        for st, a0, a1 in segs:
            n, cur = a1 - a0, st.cur[li]
            o = kda_mod.chain(p[a0:a1], k.b_off, b.ka[a0:a1], b.kg[a0:a1], st.conv[li], k.conv,
                              st.rec[cur, li], k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n,
                              b.kscratch, st.rec[1 - cur, li])
            b.cofill_kout[a0:a1].copy_(o)
            st.cur[li] = 1 - cur
            _shift_conv(st.conv[li:li + 1], b.kproj[:, a0:a1], n)
        out = b.cofill_kout[:R]
    else:
        for st, a0, a1 in segs:          # each stream's chain on its own state; its rows go to its proj for commit
            n = a1 - a0
            st.proj[li, :n].copy_(p[a0:a1])
            cur = st.cur[li]
            o = kda_mod.chain(st.proj[li, :n], k.b_off, b.ka[a0:a1], b.kg[a0:a1], st.conv[li], k.conv, st.rec[cur, li],
                              k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n, st.scratch[li], st.rec[1 - cur, li])
            b.kout[a0:a1].copy_(o)
        out = b.kout[:R]
    return out_proj(w, b, out, k.o, None if pre else qmm.group_sums(out, b.kxs[:R]), R)


def dsa_block(layer: LayerW, w: Weights, caches: Sequence, b: Buffers, R: int, nch: int | None,
              sparse_np: int | None = None) -> torch.Tensor:
    """caches: (latent or key cache, value cache, pos_dev, index caches, host_pos, a0, a1) of each stream's rows; rows past the dense limit attend to their top-512 pools."""

    c = w.cfg
    a = layer.dsa
    mm(b, b.normed[:R], a.proj, b.xs[:R], b.dp[:R])
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    if a.absorb is not None:
        return _dsa_latent(a, w, caches, b, R, nch, sparse_np)
    if sparse_np is not None or len(caches) != 1:
        raise ValueError("sparse CUDA graphs and several streams need the latent cache (TF_GLM_LATENT=1)")
    kc, vc, pos_dev, index, host_pos, _, _ = caches[0]
    mm(b, b.lat[:R], a.kv_k, b.xs_lat[:R], b.kn[:R].view(R, HL * c.qk_dim))
    mm(b, b.lat[:R], a.kv_v, b.xs_lat[:R], b.vn[:R].view(R, HL * c.v_dim))
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
        glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:     # every row sparse: the dense pass is skipped
        o = torch.empty((R, HL, c.v_dim), dtype=torch.bfloat16, device=b.q.device) if b.prefill else b.attn.out[:R]
    elif b.prefill:
        o = prefill_attention.attention(b.q[:R], kc, vc, host_pos, scale=c.qk_dim ** -0.5)
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, None if b.prefill else qmm.group_sums(o, b.xs_ao[:R]), R)


def dense_attention(qa: torch.Tensor, lc: torch.Tensor, pos_dev: torch.Tensor, s: latent.LatentScratch, *,
                    scale: float, nch: int, out: torch.Tensor, hb: int | None = None) -> torch.Tensor:
    """Upstream #128 row blocking, retaining the original pass's head tile and reduction chunk count."""

    R, step = qa.shape[0], s.part_rows
    hb = latent.head_block(R) if hb is None else hb
    if R <= step:
        return latent.attention(qa, lc, pos_dev, s, scale=scale, nch=nch, out=out, hb=hb)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        at = pos_dev if r0 == 0 else pos_dev + r0
        latent.attention(qa[r0:r1], lc, at, s, scale=scale, nch=nch, out=out[r0:r1], hb=hb)
    return out


def _dsa_latent(a, w: Weights, caches: Sequence, b: Buffers, R: int, nch: int | None,
                sparse_np: int | None = None) -> torch.Tensor:
    """DSA on the latent cache: the same indexer and selection, attention over latents with kv_b's key blocks absorbed into the query, each stream on its own cache."""

    from .attention import CHUNK

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    ix = a.index
    scale = c.qk_dim ** -0.5
    indexed = any(v[3] is not None for v in caches)
    if indexed:
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
            glue.router(b.normed[:R], ix.gate, b.igr[:R])
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    ol = s.ol[:R]
    picked = False
    for lc, _, pos_dev, index, host_pos, a0, a1 in caches:
        n = a1 - a0
        with prof.timed("dsa: latent write"):
            latent.latent_write(b.lat[a0:a1], lc, pos_dev)
        # sparse_np: every row is past the dense limit (a captured graph); else the host position decides
        all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
        sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + n - 1 >= c.dense_limit))
        if index is not None:
            ik, ig, pk = index
            with prof.timed("dsa: indexer update"):
                sparse.index_update(b.ikr[a0:a1, :c.index_dim], b.igr[a0:a1], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk,
                                    pos_dev)
        if not all_sparse:
            # rows past the dense limit are recomputed sparsely below, so the dense pass needs only the chunks up to it
            chunks = nch if host_pos is None else -(-(host_pos + n) // CHUNK)
            with prof.timed("dsa: dense attention"):
                if b.prefill and (p1.DENSE_ROWS or p1.PROMPT_SCRATCH or p1.ATTENTION_TILES):
                    dense_n = min(n, c.dense_limit - host_pos) if p1.DENSE_ROWS and sparse_rows else n
                    # Trimming may leave <64 rows: preserve the untrimmed pass's head kernel and chunk count.
                    dense_attention(qa[a0:a0 + dense_n], lc, pos_dev, s, scale=scale,
                                    nch=min(chunks or s.nch, s.nch), out=ol[a0:a0 + dense_n],
                                    hb=latent.head_block(n))
                else:
                    latent.attention(qa[a0:a1], lc, pos_dev, s, scale=scale, nch=min(chunks or s.nch, s.nch),
                                     out=ol[a0:a1])
        if sparse_rows:
            if not picked:
                mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
                picked = True
            if b.prefill and p1.ATTENTION_TILES:
                sparse_attention_tiles(qa[a0:a1], lc, b.qi[a0:a1], b.ikr[a0:a1, c.index_dim:], pk,
                                       host_pos, pos_dev, ol[a0:a1], scale, bucket=sparse_np)
            else:
                with prof.timed("dsa: select tokens"):
                    tokens, counts = sparse.select_tokens(b.qi[a0:a1], b.ikr[a0:a1, c.index_dim:], pk, host_pos, n,
                                                          pk.shape[0] - 2, pos_dev, bucket=sparse_np)
                with prof.timed("dsa: sparse attention"):
                    latent.sparse_attention(qa[a0:a1], lc, tokens, counts, ol[a0:a1], scale)
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R)


def sparse_attention_tiles(qa, cache, qi, wts, pk, host_pos, pos_dev, out, scale, *, bucket=None):
    """Bound selection and partials; retain the original segment's head tile and pool bucket."""
    rows = len(qa)
    hb = latent.head_block(rows)
    bucket = sparse.pool_bucket(host_pos, rows, pk.shape[0] - 2) if bucket is None else bucket
    for start in range(0, rows, p1.ATTENTION_ROWS):
        end = min(rows, start + p1.ATTENTION_ROWS)
        at = pos_dev if start == 0 else pos_dev + start
        pos = None if host_pos is None else host_pos + start
        with prof.timed("dsa: select tokens"):
            tokens, counts = sparse.select_tokens(qi[start:end], wts[start:end], pk, pos, end - start,
                                                  pk.shape[0] - 2, at, bucket=bucket)
        with prof.timed("dsa: sparse attention"):
            latent.sparse_attention(qa[start:end], cache, tokens, counts, out[start:end], scale, hb=hb)
        del tokens, counts


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    mm(b, b.normed[:R], m.gu, b.xs[:R], b.gu[:R])
    glue.swiglu(b.gu[:R], b.act[:R], b.xs_act[:R], w.cfg.limit)
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        if b.plan.prefill:
            b.plan.tile = (128 if p1.EXPERT_PREFILL128 and R >= 4096 and m.experts.gs == 64
                           else grouped.PREFILL_TILE)
        grouped.route(b.pick[:R], b.plan)
    if m.shared is not None:
        # EXL3: the routed slots through the trellis kernels, the shared expert (last slot) through BF16 matmuls
        from . import exl3_mm

        with prof.timed("moe: routed experts"):
            exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)
        with prof.timed("moe: shared expert"):
            s = m.shared
            mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R])
            glue.swiglu(b.sgu[:R], b.sact[:R], b.sxs[:R], c.limit)
            mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True)
            b.ey[:R, c.top_k].copy_(b.sy[:R])
        with prof.timed("moe: combine"):
            glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
        with prof.timed("moe: all-gather"):
            return gather(w, b, R)
    with prof.timed("moe: gate/up"):
        four_warps = (p1.EXPERT_PREFILL64 and b.plan.prefill and b.plan.tile == 64
                      and 1024 <= R < 4096 and m.experts.gs == 64 and m.experts.swiglu)
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R, four_warps=four_warps)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)
    with prof.timed("moe: combine"):
        glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def main_caches(segs: Sequence, di: int, eager: bool) -> list:
    """The main model's DSA layer di cache views of each stream (host positions only in eager steps)."""
    return [(st.kc[di], st.vc[di], st.pos_dev, st.index[di] if st.index is not None else None,
             st.pos if eager else None, a0, a1) for st, a0, a1 in segs]


def attn_block(layer: LayerW, w: Weights, segs: Sequence, b: Buffers, R: int, nch: int | None, eager: bool,
               sparse_np: int | None) -> torch.Tensor | None:
    hint = getattr(b, "l2_prefetch", None)
    if hint is not None:
        hint.select(layer.index, 0)
    if layer.kind == "kda":
        with prof.timed("kda"):
            return kda_block(layer, w, segs, b, R)
    di = segs[0][0].dsa_index[layer.index]
    with prof.timed("dsa (total)"):
        return dsa_block(layer, w, main_caches(segs, di, eager), b, R, nch, sparse_np)


def ffn_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor | None:
    hint = getattr(b, "l2_prefetch", None)
    if hint is not None:
        hint.select(layer.index, 1)
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        return mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)


def hc_pre(w: Weights, h, norm: torch.Tensor, b: Buffers, lo: int, hi: int) -> None:
    """The mixing before a block on rows lo:hi of the streams b.x, then (TF_GLM_SP) every rank's normed rows to all."""

    c = w.cfg
    glue.hc_pre(b.x[lo:hi], h.fn, h.base, h.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi], b.comb[lo:hi],
                b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters)
    seqpar.share(w, b)


def layer_forward(layer: LayerW, w: Weights, segs: Sequence, b: Buffers, R: int, nch: int | None = None,
                  eager: bool = True, sparse_np: int | None = None, *, ffn: bool = True) -> None:
    lo, hi = b.sp.own if b.sp is not None else (0, R)     # the rows this rank mixes (TF_GLM_SP: its share)
    x = b.x[lo:hi]
    hc_pre(w, layer.attn_hc, layer.in_norm, b, lo, hi)
    g = attn_block(layer, w, segs, b, R, nch, eager, sparse_np)
    with prof.timed("hc"):
        glue.hc_post(x, x, g, b.post[lo:hi], b.comb[lo:hi])
        if not ffn:                      # a wholly dead final-layer chunk still updates every attention cache
            return
        hc_pre(w, layer.ffn_hc, layer.post_norm, b, lo, hi)
    g = ffn_block(layer, w, b, R)
    glue.hc_post(x, x, g, b.post[lo:hi], b.comb[lo:hi])


def _half(w: Weights, layers: Sequence[LayerW], st: State, b: Buffers, n: int, nch: int | None, tap0: int, *,
          final: bool, skip_final_ffn: bool, whole_experts: bool = False):
    """One half of a pipelined pass (TF_GLM_SP_OVERLAP) through ``layers``, yielding after each big block it queues:
    the main stream runs the blocks; the side stream runs the tail after each (reduce, hc_post, taps, the next hc_pre
    and share). ``final``: the last tail also writes the final hidden rows (and, with a plan, exchanges them)."""

    lo, hi = b.sp.own if b.sp is not None else (0, n)
    x = b.x[lo:hi]
    t0 = max(lo, tap0)
    segs = [(st, 0, n)]
    side = seqpar.Side(b.x.device)

    def post():
        glue.hc_post(x, x, reduced(w, b, n), b.post[lo:hi], b.comb[lo:hi])

    def taps(layer):
        for slot in b.tap_at.get(layer.index, ()):
            if t0 < hi:
                glue.stream_mean(b.x[t0:hi], b.taps[slot][t0:hi])

    def after_attn(layer, ffn):
        post()
        if ffn:
            hc_pre(w, layer.ffn_hc, layer.post_norm, b, lo, hi)
        else:
            taps(layer)

    def after_ffn(layer, nxt):
        post()
        taps(layer)
        if nxt is not None:
            hc_pre(w, nxt.attn_hc, nxt.in_norm, b, lo, hi)
        elif final:
            glue.stream_mean(b.x[lo:hi], b.hidden[lo:hi])
            if b.sp is not None:
                seqpar.finish(w, b, tap0)

    side.tail(lambda: hc_pre(w, layers[0].attn_hc, layers[0].in_norm, b, lo, hi))
    yield
    for i, layer in enumerate(layers):
        ffn = not (skip_final_ffn and layer is w.layers[-1])
        side.join()
        attn_block(layer, w, segs, b, n, nch, True, None)
        side.tail(lambda: after_attn(layer, ffn))
        yield
        if not ffn:
            continue
        side.join()
        if (whole_experts and layer.mlp is None and b.plan.prefill and layer.moe.shared is None
                and layer.moe.experts.gs == 64 and layer.moe.experts.swiglu):
            # Both halves publish their normed rows and join the side stream
            # before lockstep runs one expert pass into their shared b.part.
            yield layer
        else:
            ffn_block(layer, w, b, n)
        side.tail(lambda: after_ffn(layer, layers[i + 1] if i + 1 < len(layers) else None))
        yield
    side.join()


def lockstep(halves: list, whole_ffn=None) -> None:
    """Run the halves' stages alternately, A then B: every rank queues the same NCCL calls in the same order."""

    done = object()
    live = list(halves)
    while live:
        stages = [(h, next(h, done)) for h in live]
        ready = [stage for _, stage in stages if stage is not None and stage is not done]
        if ready:
            if whole_ffn is None or len(ready) != len(halves) or any(layer is not ready[0] for layer in ready):
                raise RuntimeError("expert pass requires both overlap halves at the same layer")
            whole_ffn(ready[0])
        live = [h for h, stage in stages if stage is not done]


def pipelined(w: Weights, st: State, b: Buffers, R: int, h: int, layers: Sequence[LayerW], nch: int | None, *,
              final: bool, tap_start: int, skip_final_ffn: bool) -> None:
    """TF_GLM_SP_OVERLAP: rows [0, h) and [h, R) of a prompt chunk as two consecutive passes, interleaved by layer."""

    halves = []
    for a0, a1 in ((0, h), (h, R)):
        v = b.half(a0, a1)
        v.sp = seqpar.plan(w, v, a1 - a0)
        s = st if a0 == 0 else st.shifted(a0)
        halves.append(_half(w, layers, s, v, a1 - a0, nch, min(max(tap_start - a0, 0), a1 - a0), final=final,
                            skip_final_ffn=skip_final_ffn, whole_experts=p1.EXPERT_WHOLE_PASS))
    if p1.EXPERT_WHOLE_PASS:
        # The full view reads both published halves. Keep the existing half
        # reduce/hc_post ownership and rank order; never reduce this view itself.
        whole = b.half(0, R)
        lockstep(halves, lambda layer: ffn_block(layer, w, whole, R))
    else:
        lockstep(halves)


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    return stage_streams(w, b, [(st, tokens)])[-1][2]


def stage_streams(w: Weights, b: Buffers, windows: Sequence) -> list:
    """Several streams' windows (state, tokens) into consecutive rows of the static buffer -> segments (state, a0, a1)."""

    segs, ids, a0 = [], [], 0
    for st, tokens in windows:
        check_room(w, st, len(tokens))
        segs.append((st, a0, a0 + len(tokens)))
        ids.extend(tokens)
        a0 += len(tokens)
    R = a0
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    b.staged.synchronize()
    host = b.ids_host[:R].numpy()
    host[:] = ids
    np.copyto(host, w.cfg.image_token, where=host < 0)         # an image's keyed rows embed its placeholder
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return segs


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None,
            layer_start: int = 0, layer_stop: int | None = None,
            tap_start: int = 0, skip_final_ffn: bool = False):
    """Run capturable GPU work on static buffers and device positions; eager long contexts use host_pos (graphs sparse_np) to select sparse attention."""

    return compute_streams(w, [(st, 0, R)], b, logits=logits, nch=nch, eager=host_pos is not None,
                           sparse_np=sparse_np, layer_start=layer_start, layer_stop=layer_stop,
                           tap_start=tap_start, skip_final_ffn=skip_final_ffn)


def compute_streams(w: Weights, segs: Sequence, b: Buffers, *, logits: bool = True, nch: int | None = None,
                    eager: bool = True, sparse_np: int | None = None,
                    layer_start: int = 0, layer_stop: int | None = None,
                    tap_start: int = 0, skip_final_ffn: bool = False):
    """The forward of every segment's rows at once: projections, experts and mixing over all rows, attention and KDA per stream."""

    c = w.cfg
    R = segs[-1][2]
    sliced = layer_stop is not None
    if sliced and (not b.prefill or not eager or not 0 <= layer_start < layer_stop <= len(w.layers)):
        raise ValueError("layer slices require an eager prefill and a valid layer range")
    if not sliced and layer_start:
        raise ValueError("a resumed layer range needs a stop")
    if tap_start or skip_final_ffn:
        if not p1.DEAD_WORK or not b.prefill or len(segs) != 1 or not 0 <= tap_start <= R:
            raise ValueError("Prefill dead work requires one prefill segment and a valid tap suffix")
        if skip_final_ffn and (logits or tap_start != R):
            raise ValueError("Prefill final FFN skip requires no live taps or logits")
    rec = tp3_probe.recorder
    if sliced and rec is not None and rec.calls is not None:
        raise RuntimeError("a sliced prefill cannot interleave a live TP3 probe step")
    if rec is not None and eager and not b.prefill and len(segs) == 1:
        rec.begin_step(segs[0][0].pos, b.ids[:R])
    if layer_start == 0:
        glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
        if b.overlay is not None:
            rows, emb = b.overlay
            b.x[:R].view(R, c.streams, c.hidden)[rows] = emb[:, None, :].expand(-1, c.streams, -1)
    h = seqpar.halves(w, b, segs, R, eager)   # TF_GLM_SP_OVERLAP: a function of R alone, as the plan is
    if h is not None:
        layers = w.layers[layer_start:layer_stop] if sliced else w.layers
        final = not (sliced and layer_stop < len(w.layers)) and not skip_final_ffn
        pipelined(w, segs[0][0], b, R, h, layers, nch, final=final, tap_start=tap_start,
                  skip_final_ffn=skip_final_ffn)
        if not final:
            return None
        return finish_logits(w, segs, b, R, logits)
    b.sp = seqpar.plan(w, b, R)          # a function of R alone, so every layer slice of a chunk gets the same shares
    lo, hi = b.sp.own if b.sp is not None else (0, R)
    t0 = max(lo, tap_start)              # the tap rows this rank writes (the dead-work skip: from tap_start on)
    try:
        for layer in w.layers[layer_start:layer_stop] if sliced else w.layers:
            if skip_final_ffn and layer is w.layers[-1]:
                layer_forward(layer, w, segs, b, R, nch, eager, sparse_np, ffn=False)
            else:
                layer_forward(layer, w, segs, b, R, nch, eager, sparse_np)
            for slot in b.tap_at.get(layer.index, ()):
                if t0 < hi:
                    glue.stream_mean(b.x[t0:hi], b.taps[slot][t0:hi])
        if sliced and layer_stop < len(w.layers):
            return None                 # caller retains this buffer and this stream's partial state
        if rec is not None:
            rec.end_step()
        if skip_final_ffn:
            return None                 # no live rows: nothing to exchange
        glue.stream_mean(b.x[lo:hi], b.hidden[lo:hi])
        if b.sp is not None:             # every rank's final rows and live taps, so b.hidden and b.taps hold them all
            seqpar.finish(w, b, tap_start)
    finally:
        b.sp = None
    return finish_logits(w, segs, b, R, logits)


def finish_logits(w: Weights, segs: Sequence, b: Buffers, R: int, logits: bool) -> torch.Tensor | None:
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, w.cfg.eps, b.fnormed[:R], b.fxs[:R])
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        if len(segs) > 1:
            # Keep the original one-row head kernel per stream, including its
            # rounding; a batched head could choose a different kernel.
            return torch.cat([mm(b, b.fnormed[a1 - 1:a1], w.head, b.fxs[a1 - 1:a1], b.logits[:1]).clone()
                              for _, _, a1 in segs])
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits and hidden buffer views for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@triton.jit
def _row(CONV, PROJ, l, src, c, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr):
    old = tl.load(CONV + l * conv_layer + src * C + c, mask=(src < TAPS) & (c < C), other=0.0)
    new = tl.load(PROJ + l * proj_layer + (src - TAPS) * proj_row + c, mask=(src >= TAPS) & (c < C), other=0.0)
    return tl.where(src < TAPS, old, new)


@triton.jit
def _conv_shift(CONV, PROJ, keep, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program (layer, channel block): the 3 window rows become rows keep .. keep + 2 of [old window; new rows]."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v0 = _row(CONV, PROJ, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v1 = _row(CONV, PROJ, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v2 = _row(CONV, PROJ, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
    tl.store(CONV + l * conv_layer + c, v0, mask=c < C)
    tl.store(CONV + l * conv_layer + C + c, v1, mask=c < C)
    tl.store(CONV + l * conv_layer + 2 * C + c, v2, mask=c < C)


def _shift_conv(conv: torch.Tensor, proj: torch.Tensor, keep: int) -> None:
    """conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [L, R, W >= C])."""

    n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    _conv_shift[(n, triton.cdiv(C, 1024))](conv, proj, keep, conv.stride(0), proj.stride(0), proj.stride(1), C=C,
                                           TAPS=taps, BLOCK=1024, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
