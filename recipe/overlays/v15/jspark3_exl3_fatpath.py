# SPDX-License-Identifier: Apache-2.0
"""JSpark3 v1.5: sync-free, bit-exact fat-expert fallback for EP3 EXL3 MoE.  [jspark3-fatpath]

The served EXL3 MoE (FlyCockpit ``exl3.py``) runs every expert that receives at
most ``TEMP_ROWS_FUSED`` (128) rows in one fused ``exl3_moe`` launch, and hands
every larger ("fat") expert to ``apply_exl3_python_loop``.  On a prefill chunk
or a v1.4 mixed step almost every local expert is fat, and that loop:

* iterates every *global* expert id present in the batch (up to 288) and calls
  ``expert_map[e].item()`` for each one: one GPU->host sync per id, ~288 per
  MoE layer on EP3, with the GPU idle between them (TP2 never had an
  ``expert_map``, so the upstream lineage never paid this);
* runs ``(ids == g).nonzero()`` (another sync) per fat expert;
* moves every fat row through ~12 separate memory-bound fp32 passes around the
  three reconstructed-weight GEMMs (two ``.half()`` copies of the same rows, two
  input Hadamards with the same sign vector, in-place output Hadamards, clamp,
  silu, clamp, multiply, ``.half()``, a routing-scale multiply and
  ``index_add_``).

This module keeps every arithmetic operation of that loop and removes the rest:

``legacy``   the original loop, unchanged (``dispatch`` returns False).
``syncfree`` the same ext kernels, cuBLAS calls and torch ops in the same order,
             on the same rows, into the same accumulator; the host learns the
             per-expert row counts from ONE asynchronous copy issued before the
             fused launch, rows come from a stable sort (identical order to
             ``nonzero``), one fp16 copy serves gate and up, and the input
             Hadamard is computed once per layer when every local expert shares
             the same input sign vector (it does in this checkpoint; checked, not
             assumed).
``fused``    ``syncfree`` plus three Triton epilogues that replay the exllamav3
             fp32 output Hadamard operation by operation (fast-math FTZ
             semantics included) and fuse it with clamp-max, with
             clamp*SiLU-product->fp16, and with routing-scale->scatter-add.
             Transcendentals stay in torch (``F.silu``), so no libm replica is
             needed.

Every mode is designed to produce the same bits as ``legacy``.  A self-check
(``JSPARK3_EXL3_FATPATH_SELFCHECK=N``) runs the legacy loop on a copy of the
accumulator for the first N fat calls of the process and compares bitwise; on a
mismatch the process falls back to ``legacy`` (or raises with
``JSPARK3_EXL3_FATPATH_SELFCHECK_STRICT=1``).  The mode can be switched at a step
boundary without a restart through ``JSPARK3_EXL3_FATPATH_EPOCH_FILE`` (a JSON
file ``{"mode": "legacy|syncfree|fused"}``), which is what makes a same-boot
A/B/A possible.  The same file can re-arm the self-check (``"selfcheck": N``),
clear a demotion (``"reset": true``) and arm the self-check's negative control
(``"fault": "reverse"``: self-checked calls process the fat experts in reverse
order -- same arithmetic, different fp32 accumulation order -- which the
self-check must flag).

Decode is untouched: the fat branch only runs when a step carries more than
``TEMP_ROWS_FUSED`` tokens, which no captured decode graph does (largest 48).
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from typing import Any

import torch
import torch.nn.functional as F

MARKER = "[jspark3-fatpath]"
MODES = ("legacy", "syncfree", "fused")
MODE_ENV = "JSPARK3_EXL3_FATPATH"
EPOCH_ENV = "JSPARK3_EXL3_FATPATH_EPOCH_FILE"
SELFCHECK_ENV = "JSPARK3_EXL3_FATPATH_SELFCHECK"
STRICT_ENV = "JSPARK3_EXL3_FATPATH_SELFCHECK_STRICT"
DEFAULT_MODE = "fused"

# fp32 value of exllamav3's r_scale for scale=1.0: ``scale * 0.088388347648f``
# (hadamard.cu); float32 rounding of 1/sqrt(128).
_R_SCALE_F32 = 0.08838834613561630249023437500
_FLT_MIN = 1.1754943508222875e-38

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "mode": None,            # resolved mode, cached
    "epoch_mtime": None,     # last seen epoch-file mtime_ns
    "forced_legacy": False,  # sticky fallback after a self-check mismatch
    "selfcheck_left": None,  # remaining self-check calls (lazy init)
    "selfcheck_pass": 0,
    "selfcheck_fail": 0,
    "calls": 0,
    "fault": None,           # epoch-file negative control ("reverse"), self-checked calls only
    "logged": set(),
}
_PINNED: dict[str, torch.Tensor] = {}


def _log(msg: str) -> None:
    try:
        from vllm.logger import init_logger

        init_logger(__name__).info("%s %s", MARKER, msg)
    except Exception:  # pragma: no cover - logging must never break the step
        print(f"{MARKER} {msg}", flush=True)


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        _log(msg)


def _env_mode() -> str:
    mode = os.environ.get(MODE_ENV, DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        raise RuntimeError(f"{MARKER} {MODE_ENV}={mode!r} is not one of {MODES}")
    return mode


def _read_epoch(path: str) -> None:
    """Apply a changed epoch file: {"mode": ..., "selfcheck": N, "reset": bool, "fault": "reverse"|null}.

    ``selfcheck`` re-arms N self-checked fat calls; ``reset`` clears a sticky
    self-check demotion; ``fault: "reverse"`` is the negative control for the
    self-check -- it reverses the fat-expert order (same math, different fp32
    accumulation order into the output) on self-checked calls ONLY, so a
    working self-check must report FAIL and demote the process to legacy.
    """
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return
    if mtime == _STATE["epoch_mtime"]:
        return
    with _LOCK:
        try:
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            if not isinstance(spec, dict):
                raise ValueError("epoch file must hold a JSON object")
        except (OSError, ValueError) as exc:
            _log(f"epoch file {path} unreadable ({exc}); keeping mode {_STATE['mode']}")
            _STATE["epoch_mtime"] = mtime
            return
        mode = str(spec.get("mode", "")).strip().lower()
        if mode in MODES and mode != _STATE["mode"]:
            _log(f"epoch mode -> {mode} ({path})")
            _STATE["mode"] = mode
        if spec.get("reset") is True and _STATE["forced_legacy"]:
            _STATE["forced_legacy"] = False
            _log("epoch reset: self-check demotion cleared")
        if isinstance(spec.get("selfcheck"), int) and spec["selfcheck"] >= 0:
            _STATE["selfcheck_left"] = int(spec["selfcheck"])
            _log(f"epoch selfcheck re-armed for {spec['selfcheck']} fat calls")
        fault = spec.get("fault")
        _STATE["fault"] = fault if fault in ("reverse",) else None
        if _STATE["fault"]:
            _log(f"epoch NEGATIVE CONTROL armed: fault={fault} (self-checked calls only)")
        _STATE["epoch_mtime"] = mtime


def current_mode() -> str:
    """Resolve the mode: epoch file (if set and readable) over the environment."""
    path = os.environ.get(EPOCH_ENV, "")
    if path:
        _read_epoch(path)
    if _STATE["mode"] is None:
        _STATE["mode"] = _env_mode()
        _log(f"mode={_STATE['mode']} (env {MODE_ENV})")
    if _STATE["forced_legacy"]:
        return "legacy"
    return _STATE["mode"]


# ----------------------------------------------------------------------------- exllamav3 access
def _exl3_module():
    # The FlyCockpit overlay installs the exllamav3 namespace stubs before any
    # LinearEXL3 exists; by the time a fat call happens the module is loaded.
    return importlib.import_module("exllamav3.modules.quant.exl3")


def _recon_gemm(lin: Any, xh: torch.Tensor, ext: Any) -> torch.Tensor:
    """LinearEXL3.reconstruct_hgemm without its input and output Hadamards.

    Same allocations, same ``ext.reconstruct`` and the same ``ext.hgemm``
    (cublasGemmEx fp16 x fp16 -> fp32) on the same [rows, in] fp16 operand, so
    the returned fp32 tensor is bit-identical to the legacy ``y_`` right before
    its in-place ``had_r_128(y_, y_, None, svh, 1.0)``.
    """
    rows = xh.shape[0]
    y = torch.empty((rows, lin.out_features), dtype=torch.float32, device=xh.device)
    w = torch.empty((lin.in_features, lin.out_features), dtype=torch.half, device=lin.trellis.device)
    ext.reconstruct(w, lin.trellis, lin.K, lin.mcg, lin.mul1)
    ext.hgemm(xh, w, y)
    return y


# ----------------------------------------------------------------------------- Triton epilogues
# Module-level @triton.jit functions (Triton resolves callees through module
# globals, not closures).  Without Triton, "fused" degrades to "syncfree".
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover - the serving image always ships Triton
    triton = None
    tl = None
    _TRITON_OK = False

if _TRITON_OK:
    @triton.jit
    def _ftz(x):
        # PTX .ftz semantics (exllamav3 is built with --use_fast_math): a
        # subnormal operand or result becomes a zero of the same sign.
        return tl.where(tl.abs(x) < 1.1754943508222875e-38, x * 0.0, x)

    @triton.jit
    def _bfly(v, BLOCKS: tl.constexpr, H: tl.constexpr, L: tl.constexpr):
        # One radix-2 stage on element-index bit log2(L) of each 128-vector:
        # lower <- a + b, upper <- a - b (exllamav3 computes (-b) + a, the same
        # IEEE value), both flushed like add.ftz.f32.
        t = tl.reshape(v, [BLOCKS, H, 2, L])
        t = tl.permute(t, [0, 1, 3, 2])
        lo, hi = tl.split(t)
        lo = _ftz(lo)
        hi = _ftz(hi)
        t = tl.join(_ftz(lo + hi), _ftz(lo - hi))
        t = tl.permute(t, [0, 1, 3, 2])
        return tl.reshape(t, [BLOCKS, 128])

    @triton.jit
    def _had_post(v, s, r_scale, BLOCKS: tl.constexpr):
        # had_ff_r_128_inner<pre=false, post=true>: bits 0,1 (in-thread 4-point
        # stage), then warp shuffles over lane bits 0..4 (element bits 2..6),
        # then *= r_scale, then *= float(svh), each a flushed fp32 multiply.
        v = _bfly(v, BLOCKS, 64, 1)
        v = _bfly(v, BLOCKS, 32, 2)
        v = _bfly(v, BLOCKS, 16, 4)
        v = _bfly(v, BLOCKS, 8, 8)
        v = _bfly(v, BLOCKS, 4, 16)
        v = _bfly(v, BLOCKS, 2, 32)
        v = _bfly(v, BLOCKS, 1, 64)
        v = _ftz(v * r_scale)
        v = _ftz(v * _ftz(s))
        return v

    @triton.jit
    def _gate_epilogue_kernel(y_ptr, svh_ptr, t1_ptr, N, r_scale, limit, BLOCKS: tl.constexpr):
        # legacy: had_r_128(y, y, None, svh_gate, 1.0); t1 = y.clamp(max=limit)
        row = tl.program_id(0).to(tl.int64)
        cols = (tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS))[:, None] * 128 + tl.arange(0, 128)[None, :]
        v = tl.load(y_ptr + row * N + cols)
        s = tl.load(svh_ptr + cols).to(tl.float32)
        v = _had_post(v, s, r_scale, BLOCKS)
        v = tl.where(v > limit, limit, v)          # torch clamp_max: NaN passes through
        tl.store(t1_ptr + row * N + cols, v)

    @triton.jit
    def _up_epilogue_kernel(y_ptr, svh_ptr, g_ptr, a_ptr, N, r_scale, limit, BLOCKS: tl.constexpr):
        # legacy: had_r_128(y, y, None, svh_up, 1.0);
        #         act = silu_out * y.clamp(min=-limit, max=limit); a16 = act.half()
        row = tl.program_id(0).to(tl.int64)
        cols = (tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS))[:, None] * 128 + tl.arange(0, 128)[None, :]
        v = tl.load(y_ptr + row * N + cols)
        s = tl.load(svh_ptr + cols).to(tl.float32)
        v = _had_post(v, s, r_scale, BLOCKS)
        v = tl.where(v < -limit, -limit, tl.where(v > limit, limit, v))
        g = tl.load(g_ptr + row * N + cols)
        tl.store(a_ptr + row * N + cols, (g * v).to(tl.float16))

    @triton.jit
    def _scatter_epilogue_kernel(y_ptr, svh_ptr, flat_ptr, w_ptr, out_ptr, N, TOPK, r_scale,
                                 BLOCKS: tl.constexpr):
        # legacy: had_r_128(y, y, None, svh_down, 1.0);
        #         out.index_add_(0, token_idx, y * weights[token_idx, k_pos].float())
        row = tl.program_id(0).to(tl.int64)
        cols = (tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS))[:, None] * 128 + tl.arange(0, 128)[None, :]
        flat = tl.load(flat_ptr + row)
        tok = flat // TOPK
        w = tl.load(w_ptr + flat).to(tl.float32)
        v = tl.load(y_ptr + row * N + cols)
        s = tl.load(svh_ptr + cols).to(tl.float32)
        v = _had_post(v, s, r_scale, BLOCKS)
        tl.atomic_add(out_ptr + tok * N + cols, v * w, sem="relaxed")


def _kernels() -> dict[str, Any]:
    if not _TRITON_OK:
        raise RuntimeError(f"{MARKER} Triton unavailable; fused mode cannot run")
    return {"gate_epilogue": _gate_epilogue_kernel, "up_epilogue": _up_epilogue_kernel,
            "scatter_epilogue": _scatter_epilogue_kernel}


_BLOCKS = 4
# The legacy multiplies and the fp32->fp16 cast are IEEE (torch is not built
# with fast-math) and must not be contracted into an FMA with the following
# atomic add; fp fusion off makes that explicit for the whole kernel.
_LAUNCH = {"num_warps": 4, "enable_fp_fusion": False}


def _gate_epilogue(y: torch.Tensor, svh: torch.Tensor, limit: float) -> torch.Tensor:
    rows, n = y.shape
    t1 = torch.empty_like(y)
    _kernels()["gate_epilogue"][(rows, n // (128 * _BLOCKS))](
        y, svh, t1, n, _R_SCALE_F32, float(limit), BLOCKS=_BLOCKS, **_LAUNCH)
    return t1


def _up_epilogue(y: torch.Tensor, svh: torch.Tensor, g: torch.Tensor, limit: float) -> torch.Tensor:
    rows, n = y.shape
    a16 = torch.empty((rows, n), dtype=torch.float16, device=y.device)
    _kernels()["up_epilogue"][(rows, n // (128 * _BLOCKS))](
        y, svh, g, a16, n, _R_SCALE_F32, float(limit), BLOCKS=_BLOCKS, **_LAUNCH)
    return a16


def _scatter_epilogue(y: torch.Tensor, svh: torch.Tensor, flat: torch.Tensor,
                      weights_flat: torch.Tensor, out: torch.Tensor, topk: int) -> None:
    rows, n = y.shape
    _kernels()["scatter_epilogue"][(rows, n // (128 * _BLOCKS))](
        y, svh, flat, weights_flat, out, n, int(topk), _R_SCALE_F32, BLOCKS=_BLOCKS, **_LAUNCH)


# ----------------------------------------------------------------------------- layer facts (cached)
def _local_to_global(layer: Any, expert_map: torch.Tensor | None, n_local: int) -> list[int]:
    cached = getattr(layer, "_jspark3_fat_l2g", None)
    if cached is not None and len(cached) == n_local:
        return cached
    if expert_map is None:
        l2g = list(range(n_local))
    else:
        emap = [int(v) for v in expert_map.detach().to("cpu").tolist()]   # one-time sync
        l2g = [-1] * n_local
        for g, e in enumerate(emap):
            if 0 <= e < n_local:
                if l2g[e] != -1:
                    raise RuntimeError(f"{MARKER} expert_map is not injective (local {e})")
                l2g[e] = g
        if any(g < 0 for g in l2g):
            raise RuntimeError(f"{MARKER} expert_map leaves a local expert without a global id")
    object.__setattr__(layer, "_jspark3_fat_l2g", l2g)
    return l2g


def _shared_suh(layer: Any, inners: list[dict[str, Any]]) -> torch.Tensor | None:
    """The one input sign vector every local gate/up shares, or None."""
    key = "_jspark3_fat_suh"
    if key in layer.__dict__:
        return layer.__dict__[key]
    first = inners[0]["gate"].suh
    shared = first
    for pack in inners:
        for which in ("gate", "up"):
            s = pack[which].suh
            if s.shape != first.shape or s.dtype != first.dtype or not torch.equal(s, first):
                shared = None
                break
        if shared is None:
            break
    object.__setattr__(layer, key, shared)
    _log_once(f"suh{id(layer)}", f"layer {getattr(layer, 'layer_name', '?')}: "
              f"input sign vector {'shared by all local gate/up' if shared is not None else 'NOT shared; per-expert path'}")
    return shared


# ----------------------------------------------------------------------------- public API
def probe(expert_count: torch.Tensor, n_exp: int, tokens: int, temp_rows: int):
    """Start the one host copy of the per-expert row counts (before the fused launch).

    Returns None when the fat branch cannot run or the legacy loop is selected;
    the caller then keeps the original code path untouched.
    """
    if tokens <= temp_rows:
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    mode = current_mode()
    if mode == "legacy":
        return None
    dev = expert_count.device
    key = str(dev)
    buf = _PINNED.get(key)
    if buf is None or buf.numel() < n_exp:
        buf = torch.empty(max(n_exp, 512), dtype=torch.int64, pin_memory=(dev.type == "cuda"))
        _PINNED[key] = buf
    host = buf[:n_exp]
    host.copy_(expert_count[:n_exp], non_blocking=True)
    event = None
    if dev.type == "cuda":
        event = torch.cuda.Event()
        event.record()
    return (host, event, mode)


def dispatch(handle, x2d: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, layer: Any,
             inners: list[dict[str, Any]], expert_map: torch.Tensor | None, limit: float,
             local: torch.Tensor, out: torch.Tensor, temp_rows: int, legacy_loop) -> bool:
    """Run every fat expert into ``out``; return False to leave it to the legacy loop."""
    if handle is None:
        return False
    host, event, mode = handle
    if event is not None:
        event.synchronize()          # waits for the count copy, not the fused kernel
    counts = host.tolist()
    n_exp = len(counts)
    fat = [e for e in range(n_exp) if counts[e] > temp_rows]
    if not fat:
        return True                  # legacy: `if fat.numel():` -> nothing to do
    l2g = _local_to_global(layer, expert_map, n_exp)
    fat.sort(key=l2g.__getitem__)    # legacy visits torch.unique(ids): ascending global id

    check = _selfcheck_armed()
    ref = out.clone() if check else None
    if check:
        legacy_loop(x2d, ids, weights, inners, expert_map, limit,
                    only_experts=set(fat), out=ref)
        if _STATE["fault"] == "reverse":
            fat = fat[::-1]          # negative control: must be caught below

    _run(mode, fat, counts, x2d, ids, weights, layer, inners, limit, local, out)
    _STATE["calls"] += 1

    if check:
        _selfcheck_result(out, ref, fat, counts, mode)
    return True


def _run(mode, fat, counts, x2d, ids, weights, layer, inners, limit, local, out) -> None:
    exl3 = _exl3_module()
    ext = exl3.ext
    thr = int(exl3.AUTO_RECONSTRUCT_THRESHOLD)
    max_n = int(exl3.MAX_RECONSTRUCT_SLICE_N)
    topk = int(ids.shape[-1])
    fused = mode == "fused" and _TRITON_OK

    # Row lists in nonzero() order: a stable sort keeps ascending flat index
    # (token-major, then top-k slot) inside every expert's segment.
    _, perm = torch.sort(local, stable=True)
    offsets = [0] * (len(counts) + 1)
    for e, c in enumerate(counts):
        offsets[e + 1] = offsets[e] + c
    weights_flat = weights.reshape(-1)

    def rows16(token_idx: torch.Tensor) -> torch.Tensor:
        # legacy: x2d.index_select(0, token_idx) then .contiguous().half()
        return x2d.index_select(0, token_idx).half()

    suh = _shared_suh(layer, inners)
    xh_all = None
    if suh is not None and any(counts[e] > thr for e in fat):
        # had_r_128 is row-independent (grid = rows x 128-blocks), so one pass
        # over every token followed by a row gather equals the legacy
        # gather -> .half() -> had_r_128 per expert and per gate/up.  copy=True:
        # the in-place transform must never alias the hidden states.
        xh_all = x2d.to(torch.half, copy=True)
        ext.had_r_128(xh_all, xh_all, suh, None, 1.0)

    for e in fat:
        n = counts[e]
        flat = perm[offsets[e]:offsets[e] + n]
        token_idx = torch.div(flat, topk, rounding_mode="floor")
        pack = inners[e]
        gate_l, up_l, down_l = pack["gate"], pack["up"], pack["down"]
        recon = n > thr and max(gate_l.out_features, up_l.out_features, down_l.out_features) <= max_n

        if recon:
            # --- gate / up: reconstruct + cuBLAS, shared input Hadamard ---
            if xh_all is not None:
                xh = xh_all.index_select(0, token_idx)
            elif torch.equal(gate_l.suh, up_l.suh):
                xs = rows16(token_idx)
                xh = torch.empty_like(xs)
                ext.had_r_128(xs, xh, gate_l.suh, None, 1.0)
            else:
                xh = None
            if xh is not None:
                y_g = _recon_gemm(gate_l, xh, ext)
                y_u = _recon_gemm(up_l, xh, ext)
                if fused:
                    g = F.silu(_gate_epilogue(y_g, gate_l.svh, limit), inplace=True)
                    a16 = _up_epilogue(y_u, up_l.svh, g, limit)
                else:
                    ext.had_r_128(y_g, y_g, None, gate_l.svh, 1.0)
                    ext.had_r_128(y_u, y_u, None, up_l.svh, 1.0)
                    act = F.silu(y_g.clamp(max=limit)) * y_u.clamp(min=-limit, max=limit)
                    a16 = act.half()
            else:
                xs16 = rows16(token_idx)
                gate = gate_l.forward(xs16, {}, out_dtype=torch.float32)
                up = up_l.forward(xs16, {}, out_dtype=torch.float32)
                a16 = (F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)).half()

            # --- down: reconstruct + cuBLAS, then scale and scatter ---
            ah = torch.empty_like(a16)
            ext.had_r_128(a16, ah, down_l.suh, None, 1.0)
            y_d = _recon_gemm(down_l, ah, ext)
            if fused:
                _scatter_epilogue(y_d, down_l.svh, flat, weights_flat, out, topk)
            else:
                ext.had_r_128(y_d, y_d, None, down_l.svh, 1.0)
                scale = weights_flat.index_select(0, flat).unsqueeze(-1).to(dtype=torch.float32)
                out.index_add_(0, token_idx, y_d * scale)
        else:
            # 129..144 rows: exllamav3 keeps the direct EXL3 GEMM (its own
            # Hadamards inside); replay the legacy ops exactly.
            xs16 = rows16(token_idx)
            gate = gate_l.forward(xs16, {}, out_dtype=torch.float32)
            up = up_l.forward(xs16, {}, out_dtype=torch.float32)
            act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
            down = down_l.forward(act.contiguous().half(), {}, out_dtype=torch.float32)
            scale = weights_flat.index_select(0, flat).unsqueeze(-1).to(dtype=torch.float32)
            out.index_add_(0, token_idx, down * scale)


# ----------------------------------------------------------------------------- self-check
def _selfcheck_armed() -> bool:
    left = _STATE["selfcheck_left"]
    if left is None:
        try:
            left = max(0, int(os.environ.get(SELFCHECK_ENV, "0") or 0))
        except ValueError:
            left = 0
        _STATE["selfcheck_left"] = left
    if left <= 0:
        return False
    _STATE["selfcheck_left"] = left - 1
    return True


def _selfcheck_result(out: torch.Tensor, ref: torch.Tensor, fat: list[int], counts: list[int], mode: str) -> None:
    same = torch.equal(out.view(torch.int32), ref.view(torch.int32))
    rows = sum(counts[e] for e in fat)
    if same:
        _STATE["selfcheck_pass"] += 1
        _log(f"selfcheck PASS mode={mode} fat_experts={len(fat)} fat_rows={rows} "
             f"bitwise=equal pass={_STATE['selfcheck_pass']} fail={_STATE['selfcheck_fail']}")
        return
    _STATE["selfcheck_fail"] += 1
    diff = (out.view(torch.int32) != ref.view(torch.int32))
    both_nan = torch.isnan(out) & torch.isnan(ref)
    real = diff & ~both_nan
    maxabs = float((out - ref).abs().nan_to_num(0.0).max()) if bool(real.any()) else 0.0
    _log(f"selfcheck FAIL mode={mode} fat_experts={len(fat)} fat_rows={rows} "
         f"elements_differ={int(real.sum())} nan_payload_only={int((diff & both_nan).sum())} "
         f"maxabs={maxabs:.3e} -> output replaced by the legacy result; legacy for the rest of this process")
    if not bool(real.any()):
        return                        # only NaN payloads differ: values are equal
    out.copy_(ref)                    # this call's output becomes the legacy result
    _STATE["forced_legacy"] = True
    if os.environ.get(STRICT_ENV, "0") == "1":
        raise RuntimeError(f"{MARKER} self-check mismatch in mode {mode}")


def stats() -> dict[str, Any]:
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in _STATE.items()}
