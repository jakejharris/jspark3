# SPDX-License-Identifier: AGPL-3.0-only
"""Display-reserve KV backing for GB10 (installed as vllm/v1/worker/glm53_display_kv.py).

The UEFI display reservation (~2 GiB per Spark) is invisible to CUDA. With
``nvidia_drm modeset=1 fbdev=0`` on a headless host, a DRM dumb buffer of up to
~2032 MiB can be created there, mmapped and registered with CUDA as
device-mapped I/O memory (``libglm53_display_kv.so``). This module:

* ``budget(p, rank)``     -- the patched ``determine_available_memory`` line:
                             returns E + ``credit()`` with E = min(P, S_r), P
                             vLLM's profiled ordinary KV budget and S_r the
                             sealed per-rank ordinary cap (v1.3's budget);
* ``credit()``            -- opens the pool D once per worker (before the KV
                             budget is computed) and returns the sealed budget
                             credit C <= D;
* ``plan(sizes)``         -- first-fit decreasing over the final KV tensor sizes;
                             refuses before any allocation unless the planned
                             pool payload covers C;
* ``alloc_int8(n, dev)``  -- replaces ``torch.zeros(n, int8, device)`` in the KV
                             tensor allocators: carves planned tensors from the
                             pool (a failed planned carve refuses), everything
                             else allocates ordinarily.

Contract (JSpark3 v1.4 S6, bounded credit): the pool is carved for at least
every credited byte (``C <= carved <= D``), so ordinary KV backing never
exceeds vLLM's profiled ordinary budget. Unplaced pool bytes are an unused
reserve, never a spill.

S7 (ordinary cap, ADVISORY Q9 option 1): vLLM's CUDA-graph memory estimate
moves the profiled ordinary budget P by up to +-0.8 GiB from boot to boot
(historical launch: +0.83 GiB on rank0), and the common block count, with it every rank's
ordinary backing, follows the upward swings into host headroom. The worker
clamps P at the sealed per-rank cap S_r (``display.ordinary_cap_bytes`` in
JSPARK3_V14_EXPECT: floor(18.99 / 19.24 / 19.30 GiB), v1.3's logged budgets)
under both profiles, so only the credit C adds blocks; a low P is never lifted.
``credited_bytes`` stays C (the clamp is not a smaller credit). One exact-byte
budget receipt per worker is bound into the identity, with the named placement
map (every carved tensor's owners by KV cache group). historical launch (S5) credited the whole pool while only 55% of
it could hold whole tensors, which spilled 799 MiB of credited KV into
ordinary memory. A failed pool open with the knob on aborts boot instead of
silently serving with a smaller cache.

Technique: coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark (AGPL-3.0).

JSpark3 v1.4 (fork of MiaAI-Lab #234 cd504b69): the knob is exactly 1 or 0
(``auto`` and every other value are refused, so a missing pool can never
silently shrink the cache); with JSPARK3_V14_EXPECT set, the opened pool size
must equal ``display.pool_open_bytes`` (0 under the display0 profile), and
``identity()`` asserts the KV block sizes this worker received and prints
``JSPARK3_V14_IDENTITY rank=<r> ok=1 <json>`` after the allocation.

Knobs (worker environment):
  GLM53_DISPLAY_KV       1 | 0 (required). ``1`` raises if the pool cannot
                         open; ``0`` never touches DRM.
  GLM53_DISPLAY_KV_MIB   pool size D, multiple of 16, default 1792 (max seen 2032).
  GLM53_DISPLAY_KV_CREDIT_MIB
                         budget credit C, multiple of 16, 0 < C <= D (required
                         when the knob is 1; sealed at 960).
  GLM53_DRM_CARD         DRM node inside the container, default /dev/dri/card0.
  GLM53_DISPLAY_KV_LIB   path of libglm53_display_kv.so
                         (default /usr/local/lib/libglm53_display_kv.so).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

MARK = "[glm53-display-kv]"
ALIGN = 4096
_owner = None
_plan: dict[int, bool] = {}
_state = {"mode": None, "pool_open": 0, "credited": 0, "carved": 0, "tensors": 0, "fallback": 0,
          "profiled": None, "cap": None, "ordinary": None}


def _log(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def _mode() -> str:
    raw = os.environ.get("GLM53_DISPLAY_KV")
    if raw == "1":
        return "on"
    if raw == "0":
        return "off"
    raise ValueError(f"GLM53_DISPLAY_KV={raw!r}: expected exactly 1 or 0 (auto is refused)")


def _expect():
    """JSPARK3_V14_EXPECT as a dict, or None outside a sealed v1.4 container."""
    raw = os.environ.get("JSPARK3_V14_EXPECT")
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"{MARK} REFUSE: JSPARK3_V14_EXPECT unreadable: {exc}") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"{MARK} REFUSE: JSPARK3_V14_EXPECT must be an object")
    return value


def _check_pool(nbytes: int, credit: int) -> None:
    want = _expect()
    if want is None:
        return
    display = want.get("display") or {}
    expected = (display.get("pool_open_bytes"), display.get("credited_bytes"))
    if expected != (nbytes, credit):
        raise RuntimeError(
            f"{MARK} REFUSE: pool_open/credited bytes {nbytes}/{credit} != expected {expected!r}")


def _pool_bytes() -> int:
    mib = int(os.environ.get("GLM53_DISPLAY_KV_MIB", "1792"))
    if mib <= 0 or mib % 16:
        raise ValueError(f"GLM53_DISPLAY_KV_MIB={mib}: must be a positive multiple of 16")
    return mib << 20


def _credit_bytes(pool: int) -> int:
    raw = os.environ.get("GLM53_DISPLAY_KV_CREDIT_MIB")
    mib = int(raw) if raw is not None and raw.strip().isdigit() else -1
    if mib <= 0 or mib % 16 or mib << 20 > pool:
        raise ValueError(f"GLM53_DISPLAY_KV_CREDIT_MIB={raw!r}: must be a positive multiple of 16 "
                         f"no larger than the pool ({pool >> 20} MiB)")
    return mib << 20


def _ordinary_cap(rank: int):
    """The sealed ordinary-budget cap S_r for the worker's global rank, or None when unsealed.

    Inside a sealed container the rank must be the launcher's NODE_RANK and must
    have a cap; nothing defaults to another rank's cap.
    """
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise RuntimeError(f"{MARK} REFUSE: worker rank {rank!r} is not an integer")
    want = _expect()
    if want is None:
        return None
    node = os.environ.get("NODE_RANK")
    if node != str(rank):
        raise RuntimeError(f"{MARK} REFUSE: worker rank {rank} != launcher NODE_RANK {node!r}")
    caps = (want.get("display") or {}).get("ordinary_cap_bytes")
    cap = caps.get(str(rank)) if isinstance(caps, dict) else None
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise RuntimeError(f"{MARK} REFUSE: no sealed ordinary cap for rank {rank} in {caps!r}")
    return cap


class Owner:
    """Process-lifetime owner of one display pool (never destroyed while views live)."""

    def __init__(self, library: str, drm_card: str, nbytes: int):
        import torch

        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA must be initialised before opening the display pool")
        self.lib = ctypes.CDLL(library)
        self.lib.glm53_display_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.glm53_display_create.restype = ctypes.c_void_p
        self.lib.glm53_display_pointer.argtypes = [ctypes.c_void_p]
        self.lib.glm53_display_pointer.restype = ctypes.c_uint64
        self.lib.glm53_display_destroy.argtypes = [ctypes.c_void_p]
        self.lib.glm53_display_destroy.restype = None
        self.lib.glm53_display_error.restype = ctypes.c_char_p
        torch.cuda.current_stream().synchronize()  # ensure a current context
        self.handle = self.lib.glm53_display_create(drm_card.encode(), nbytes)
        if not self.handle:
            raise RuntimeError(self.lib.glm53_display_error().decode())
        self.pointer = int(self.lib.glm53_display_pointer(self.handle))
        self.size = nbytes
        self.used = 0
        self.views: list = []

    def carve(self, nbytes: int):
        """Return an int8 CUDA tensor of exactly ``nbytes`` from the pool or None."""
        import torch

        off = (self.used + ALIGN - 1) // ALIGN * ALIGN
        if off + nbytes > self.size:
            return None
        view = _View(self.pointer + off, nbytes)
        # The pool was registered in the caller's current context; alias it on
        # that device rather than assuming ordinal 0.
        t = torch.as_tensor(view, device=torch.device("cuda", torch.cuda.current_device()))
        if t.data_ptr() != view.ptr or t.dtype != torch.int8 or t.numel() != nbytes:
            raise RuntimeError("CUDA array interface copied or misinterpreted external storage")
        self.used = off + nbytes
        self.views.append(view)  # keep alive for the process lifetime
        return t


class _View:
    def __init__(self, ptr: int, nbytes: int):
        self.ptr = ptr
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "strides": None,
            "typestr": "|i1",
            "data": (ptr, False),
            "version": 3,
        }


def credit() -> int:
    """Open the pool (once) and return the credit C to add to the KV budget."""
    global _owner
    mode = _mode()
    if _state["mode"] is not None:
        return _state["credited"]
    _state["mode"] = mode
    if mode == "off":
        _log("disabled (GLM53_DISPLAY_KV=0)")
        _check_pool(0, 0)
        return 0
    lib = os.environ.get("GLM53_DISPLAY_KV_LIB", "/usr/local/lib/libglm53_display_kv.so")
    card = os.environ.get("GLM53_DRM_CARD", "/dev/dri/card0")
    nbytes = _pool_bytes()
    credited = _credit_bytes(nbytes)
    try:
        if not Path(lib).exists():
            raise RuntimeError(f"{lib} missing (overlay not applied?)")
        if not Path(card).exists():
            raise RuntimeError(f"{card} missing: run the container with --device {card}")
        _owner = Owner(lib, card, nbytes)
    except Exception as exc:  # noqa: BLE001 - policy decision below
        if mode == "on":
            raise RuntimeError(
                f"{MARK} display pool unavailable: {exc}. Need nvidia_drm modeset=1 "
                "fbdev=0 on the host, an idle headless GPU and --device /dev/dri/card0; "
                "set GLM53_DISPLAY_KV=0 to serve without it."
            ) from exc
        _log(f"auto: display pool unavailable ({exc}); serving with ordinary KV only")
        return 0
    _state["pool_open"] = nbytes
    _state["credited"] = credited
    _log(json.dumps(dict(stage="pool_open", bytes=nbytes, mib=nbytes >> 20, credited_bytes=credited,
                         credited_mib=credited >> 20, ptr=hex(_owner.pointer), card=card)))
    _check_pool(nbytes, credited)
    return credited


def budget(profiled: int, rank: int) -> int:
    """The KV budget E + C, with E = min(P, S_r) (replaces ``+= credit()``).

    ``profiled`` is the freshly computed ordinary budget P (requested - non-KV -
    CUDA-graph estimate) and ``rank`` the worker's global rank. Under a sealed
    expectation both profiles clamp (display0 with C = 0); with the pool on,
    an unsealed worker refuses. Each call starts from its own P; ``credit()``
    opens the pool once and never compounds. The identity requires exactly one
    call per worker.
    """
    if isinstance(profiled, bool) or not isinstance(profiled, int):
        raise RuntimeError(f"{MARK} REFUSE: profiled ordinary budget {profiled!r} is not integer bytes")
    cap = _ordinary_cap(rank)
    credited = credit()
    if _state["mode"] == "on" and cap is None:
        raise RuntimeError(f"{MARK} REFUSE: display KV on without a sealed ordinary cap for rank {rank}")
    effective = profiled if cap is None else min(profiled, cap)
    receipt = dict(rank=rank, profiled_bytes=profiled, cap_bytes=cap, ordinary_bytes=effective,
                   clamped_bytes=profiled - effective, credited_bytes=credited,
                   total_bytes=effective + credited, call=_state.get("budget_calls", 0) + 1)
    _state["budget_calls"] = receipt["call"]
    _state.update(profiled=profiled, cap=cap, ordinary=effective, receipt=receipt)
    _log(json.dumps(dict(stage="budget", **receipt)))
    return receipt["total_bytes"]


def plan(sizes: list[int]) -> None:
    """Choose which of the upcoming allocations come from the pool.

    First-fit decreasing over the allocation sizes (index -> in pool). The KV
    tensors are allocated in layer order with mixed sizes, so plain greedy
    could strand a few hundred MiB of pool and spill that onto ordinary
    memory beyond the profiled budget. ``sizes`` are the final
    (post-reconciliation) tensor sizes. The planned payload must cover the
    credit before any tensor is allocated, otherwise this refuses: FFD alone
    cannot cover it when every unplaced tensor is larger than the pool.
    Without a plan ``alloc_int8`` is greedy.
    """
    global _plan
    _plan = {}
    if _owner is None:
        return
    left = _owner.size - _owner.used
    # Stable sort: among equal sizes the lowest tensor index comes first (the same
    # tensor every boot at a given N); the allocator's tensor order is never changed.
    for i in sorted(range(len(sizes)), key=lambda k: -sizes[k]):
        need = (sizes[i] + ALIGN - 1) // ALIGN * ALIGN
        if need <= left:
            _plan[i] = True
            left -= need
    payload = sum(sizes[i] for i in _plan)
    _state["plan"] = dict(tensors=len(sizes), from_pool=sorted(_plan),
                          pool_sizes=[sizes[i] for i in sorted(_plan)], stranded_bytes=left,
                          planned_payload_bytes=payload, credited_bytes=_state["credited"])
    _log(json.dumps(dict(stage="plan", tensors=len(sizes), from_pool=len(_plan),
                         pool_mib=_owner.size >> 20, planned_mib=payload >> 20,
                         credited_mib=_state["credited"] >> 20, stranded_mib=left >> 20)))
    if payload < _state["credited"]:
        raise RuntimeError(
            f"{MARK} REFUSE: planned pool payload {payload} < credited {_state['credited']} bytes; "
            f"the KV budget would spill credited bytes into ordinary memory (sizes {sorted(set(sizes))})")


def alloc_int8(nbytes: int, device, index: int | None = None):
    """Drop-in for torch.zeros(nbytes, dtype=torch.int8, device=device).

    ``index`` is the allocation's position in the list given to ``plan``;
    when a plan exists only planned indices are carved, and a failed planned
    carve refuses instead of falling back (the plan is what covers the credit).
    """
    import torch

    planned = bool(_plan) and index is not None and bool(_plan.get(index))
    if _owner is not None and (not _plan or index is None or planned):
        t = _owner.carve(nbytes)
        if t is not None:
            t.zero_()
            _state["carved"] += nbytes
            _state["tensors"] += 1
            return t
        if planned:
            raise RuntimeError(f"{MARK} REFUSE: planned carve of tensor {index} ({nbytes} bytes) failed; "
                               "refusing the ordinary fallback")
    if _owner is not None:
        _state["fallback"] += 1
    return torch.zeros(nbytes, dtype=torch.int8, device=device)


def report() -> None:
    if _owner is None:
        return
    _log(json.dumps(dict(stage="kv_allocated", pool_mib=_owner.size >> 20, carved_mib=_state["carved"] >> 20,
                         tensors_from_pool=_state["tensors"], tensors_ordinary=_state["fallback"],
                         unused_pool_mib=(_owner.size - _owner.used) >> 20)))


def _rank():
    rank = os.environ.get("NODE_RANK")
    if rank is None:
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else "?"
        except Exception:  # noqa: BLE001 - identity must still print
            rank = "?"
    return rank


def _inner_spec(spec):
    specs = getattr(spec, "kv_cache_specs", None)
    return next(iter(specs.values())) if isinstance(specs, dict) and specs else spec


def _kv_blocks(kv_cache_config):
    """(draft_block, mla_block) from the groups this worker allocated.

    Exact class names: KpoolTailSpec subclasses SlidingWindowSpec and
    HiddenStateCacheSpec subclasses MLAAttentionSpec.
    """
    kinds = [(type(_inner_spec(g.kv_cache_spec)).__name__, int(g.kv_cache_spec.block_size))
             for g in kv_cache_config.kv_cache_groups]
    draft = sorted({b for k, b in kinds if k == "SlidingWindowSpec"})
    mla = sorted({b for k, b in kinds if k == "MLAAttentionSpec"})
    return (draft[0] if len(draft) == 1 else draft,
            mla[0] if len(mla) == 1 else mla,
            [f"{k}:{b}" for k, b in kinds])


def _placement(kv_cache_config) -> dict:
    """The named placement map: every carved tensor's index, bytes, block stride and owners.

    Owners are the tensor's full ``shared_by`` list, each with its KV cache group
    and that group's spec class (an MLA tensor is co-owned by Mamba and drafter
    layers through disjoint block ids). Metadata only: nothing is allocated.
    """
    tensors = list(getattr(kv_cache_config, "kv_cache_tensors", None) or [])
    blocks = getattr(kv_cache_config, "num_blocks", None)
    group_of = {}
    for g, group in enumerate(getattr(kv_cache_config, "kv_cache_groups", None) or []):
        for name in getattr(group, "layer_names", None) or []:
            group_of[name] = (g, type(_inner_spec(group.kv_cache_spec)).__name__)
    rows = []
    for i in (_state.get("plan") or {}).get("from_pool") or []:
        t = tensors[i] if i < len(tensors) else None
        size = getattr(t, "size", None)
        owners = [{"layer": n, "group": group_of.get(n, (None, None))[0], "spec": group_of.get(n, (None, None))[1]}
                  for n in (getattr(t, "shared_by", None) or [])]
        rows.append({"index": i, "bytes": size, "owners": owners,
                     "block_stride": size // blocks if isinstance(size, int) and blocks else None})
    return {"num_blocks": blocks, "tensors": rows}


def _placement_problems(placement: dict, plan) -> list:
    """The map must name every planned tensor at the final N (empty when nothing is carved)."""
    plan = plan or {}
    blocks, rows = placement["num_blocks"], placement["tensors"]
    bad = []
    if [r["index"] for r in rows] != list(plan.get("from_pool") or []):
        bad.append(["indices", [r["index"] for r in rows], plan.get("from_pool")])
    if [r["bytes"] for r in rows] != list(plan.get("pool_sizes") or []):
        bad.append(["bytes", [r["bytes"] for r in rows], plan.get("pool_sizes")])
    for r in rows:
        if not isinstance(blocks, int) or blocks <= 0 or not isinstance(r["bytes"], int) or r["bytes"] % blocks:
            bad.append(["stride", r["index"], r["bytes"], blocks])
        if not r["owners"] or any(o["group"] is None for o in r["owners"]):
            bad.append(["owners", r["index"], r["owners"]])
    return bad


def identity(kv_cache_config) -> None:
    """Assert this worker's KV geometry and pool, then print the identity line.

    The v2 runner's ``profile_cudagraph_memory`` allocates (and frees) a
    minimal KV cache before ``determine_available_memory`` calls ``credit()``.
    That allocation never touches the pool (``plan``/``alloc_int8``/``report``
    are no-ops without an owner), so its identity is deferred to the real
    allocation after ``credit()``. A worker that never credits never prints an
    ok=1 line, which the fleet verify gate refuses.
    """
    if _state.get("identity_printed"):
        return
    if _state["mode"] is None:
        _state["deferred"] = _state.get("deferred", 0) + 1
        _log(json.dumps(dict(stage="identity_deferred", reason="allocation before credit()",
                             deferred=_state["deferred"], tensors_ordinary=_state["fallback"])))
        return
    want = _expect()
    draft, mla, groups = _kv_blocks(kv_cache_config)
    placement = _placement(kv_cache_config)
    live = {
        "rank": _rank(),
        "profile": os.environ.get("JSPARK3_V14_PROFILE"),
        "display": _state["mode"],
        "pool_open_bytes": _state["pool_open"],
        "credited_bytes": _state["credited"],
        "ordinary_profiled_bytes": _state["profiled"],
        "ordinary_cap_bytes": _state["cap"],
        "ordinary_budget_bytes": _state["ordinary"],
        "budget": _state.get("receipt"),
        "budget_calls": _state.get("budget_calls", 0),
        "num_blocks": placement["num_blocks"],
        "draft_block": draft,
        "mla_block": mla,
        "groups": groups,
        "plan": _state.get("plan"),
        "placement": placement["tensors"],
        "carved_bytes": _state["carved"],
        "tensors_from_pool": _state["tensors"],
        "tensors_ordinary": _state["fallback"],
    }
    if want is None:
        print(f"JSPARK3_V14_IDENTITY rank={live['rank']} ok=0 expect=unset {json.dumps(live)}",
              file=sys.stderr, flush=True)
        return
    kv = want.get("kv") or {}
    problems = {
        name: [live[name], kv.get(name)]
        for name in ("draft_block", "mla_block")
        if live[name] != kv.get(name)
    }
    display = want.get("display") or {}
    expected_pool = display.get("pool_open_bytes")
    if live["pool_open_bytes"] != expected_pool:
        problems["pool_open_bytes"] = [live["pool_open_bytes"], expected_pool]
    if live["credited_bytes"] != display.get("credited_bytes"):
        problems["credited_bytes"] = [live["credited_bytes"], display.get("credited_bytes")]
    caps = display.get("ordinary_cap_bytes")
    want_cap = caps.get(str(live["rank"])) if isinstance(caps, dict) else None
    receipt = live["budget"] or {}
    if want_cap is None or live["ordinary_cap_bytes"] != want_cap or receipt.get("cap_bytes") != want_cap:
        problems["ordinary_cap_bytes"] = [live["ordinary_cap_bytes"], want_cap]
    profiled = live["ordinary_profiled_bytes"]
    if (live["budget_calls"] != 1 or str(receipt.get("rank")) != str(live["rank"]) or
            not isinstance(profiled, int) or live["ordinary_budget_bytes"] != min(profiled, want_cap or profiled) or
            receipt.get("credited_bytes") != live["credited_bytes"] or
            receipt.get("total_bytes") != (live["ordinary_budget_bytes"] or 0) + live["credited_bytes"]):
        problems["budget"] = [receipt, f"exactly one receipt with E = min(P, {want_cap}) and total = E + C"]
    bad = _placement_problems(placement, live["plan"])
    if bad:
        problems["placement"] = bad
    # Backing, not packing: every credited byte is carved, and the plan was carved in full.
    planned = (live["plan"] or {}).get("planned_payload_bytes")
    if expected_pool and not (0 < live["credited_bytes"] <= live["carved_bytes"] <= expected_pool):
        problems["carved_bytes"] = [live["carved_bytes"], f">= credited {live['credited_bytes']} and > 0"]
    if planned is not None and live["carved_bytes"] != planned:
        problems["carved_vs_planned"] = [live["carved_bytes"], planned]
    if problems:
        raise RuntimeError(f"{MARK} REFUSE: worker identity mismatch (live, expected) {problems}")
    _state["identity_printed"] = True
    print(f"JSPARK3_V14_IDENTITY rank={live['rank']} ok=1 {json.dumps(live, sort_keys=True)}",
          file=sys.stderr, flush=True)
