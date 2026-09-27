"""Fixed-shape cooperative MoE adapter for the GLM-5.3-Flash TP3 EXL3 overlay.

Decode-sized fused exl3_moe only (tiny M): K4 MCG, H=4096, local I=2048, top-k 8,
1..64 physical rows. Prefill and unsupported shapes stay stock, including E3
grouped fat experts. Allocate/prepare only after weight load, never during graph
capture. Disabled unless explicitly selected.

Shared-scratch contract: install() caches one CoopLaunch per device and attaches
it to every eligible layer. The workspace is reused across layers and invocations.
Kernel A resets B's counters; kernel B resets A's counters for the next *ordered*
call. Overlapping cooperative launches on different CUDA streams are unsupported.
Ubatching and dual-batch overlap are rejected at configuration time. Per-layer
scratch replication is not the preferred fix on this memory-constrained kit.

The DS4.1 cooperative .so is compile-time locked to DeepSeek-V4.1 and must not
be loaded here. This adapter talks to the GLM specialization's glm53_coop_* ABI.
"""

import ctypes as C
import hashlib
import json
import math
import os
import re
from pathlib import Path
import torch

ABI_VERSION = 2
MANIFEST_SCHEMA = 1
PTR_KEYS = (
    "gate_trellis",
    "gate_suh",
    "gate_svh",
    "up_trellis",
    "up_suh",
    "up_svh",
    "down_trellis",
    "down_suh",
    "down_svh",
)
HIDDEN = 4096
INTERMEDIATE_LOCAL = 2048
TOPK = 8
ROWS_MAX = 64
ROW_CAPACITIES = (32, 64)
SLOTS_MAX = TOPK * ROWS_MAX
EXPERTS_MAX = 96
BITS = 4
ACTIVATION_LIMIT = 10.0
# Fixed before preparation/capture; each geometry must be measured on TP3.
DEFAULT_GEOMETRY = 1
GEOMETRY = DEFAULT_GEOMETRY
GEOMETRY_TILES = {
    0: "A-narrow/B-narrow",
    1: "A-wide/B-wide",
    2: "A-wide/B-narrow",
}
def counter_len(capacity):
    return capacity * TOPK * (INTERMEDIATE_LOCAL // 128) + capacity * (HIDDEN // 128) + 2 + (capacity * TOPK + 1) + capacity * TOPK


CTR_LENS = {capacity: counter_len(capacity) for capacity in ROW_CAPACITIES}
CTR_LEN = CTR_LENS[ROWS_MAX]


def row_capacity(rows):
    if not 1 <= rows <= ROWS_MAX:
        raise ValueError(f"unsupported physical row count {rows}")
    return 32 if rows <= 32 else 64


PARAMS_SIZE = 344
_OVERLAP_ENV = (
    "VLLM_UBATCH",
    "VLLM_UBATCHING",
    "VLLM_ENABLE_UBATCHING",
    "VLLM_USE_UBATCHING",
    "VLLM_V1_UBATCH",
    "VLLM_V1_ENABLE_UBATCHING",
    "VLLM_DUAL_BATCH",
    "VLLM_DUAL_BATCH_OVERLAP",
    "VLLM_ENABLE_DUAL_BATCH",
    "VLLM_ENABLE_DBO",
    "VLLM_DBO",
)
_OVERLAP_ARG_RE = re.compile(
    r"(--ubatch(?:ing)?(?:\b|=)|--enable-ubatch(?:ing)?\b|--dual-batch(?:-overlap)?\b|--enable-dbo\b)",
    re.IGNORECASE,
)


class CooperativeMoEError(RuntimeError):
    """Required cooperative-MoE initialization or configuration failure."""


class CoopDiagnostics:
    """Host-side selection log. Capture entries identify graph contents.

    Eager and capture-time counts are not graph-replay counts. CUDA graph
    replay does not re-enter this Python wrapper.
    """

    def __init__(self):
        self.geometry = None
        self.geometry_name = None
        self.native_prepared = False
        self.eligible_layers = 0
        self.ineligible_layers = 0
        self.ineligible_reasons = {}
        self.capture_selection = {}
        self.eager_selection = {}
        self.summary_logged = False
        self.seen_selection = set()
        self.row_policies = {}


def resolve_geometry(raw=None):
    if raw is None:
        raw = os.environ.get("GLM53_COOP_GEOMETRY", "")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return DEFAULT_GEOMETRY
    try:
        geometry = int(text, 10)
    except ValueError as exc:
        raise CooperativeMoEError(
            f"GLM53_COOP_GEOMETRY must be 0, 1, or 2 (got {raw!r})"
        ) from exc
    if geometry not in GEOMETRY_TILES:
        raise CooperativeMoEError(
            f"GLM53_COOP_GEOMETRY must be 0, 1, or 2 (got {geometry})"
        )
    return geometry


def enforce_serialized_execution(extra_args=None):
    """Reject configs that would overlap cooperative launches on shared scratch."""
    for key in _OVERLAP_ENV:
        value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            continue
        lowered = str(value).strip().lower()
        if lowered in ("0", "off", "false", "no"):
            continue
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported {key}={value!r}"
        )
    extra = os.environ.get("EXTRA_ARGS", "") if extra_args is None else extra_args
    if extra and _OVERLAP_ARG_RE.search(str(extra)):
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported EXTRA_ARGS={extra!r}"
        )


def layer_eligible(layer):
    bits = int(getattr(layer, "_exl3_bits", 0) or getattr(layer, "_exl3_k", 0) or 0)
    inners = getattr(layer, "_exl3_inners", None)
    if (
        bits != BITS
        or getattr(layer, "_exl3_hidden_size", None) != HIDDEN
        or getattr(layer, "_exl3_intermediate_local", None) != INTERMEDIATE_LOCAL
        or not inners
        or not 1 <= len(inners) <= EXPERTS_MAX
        or not getattr(layer, "_exl3_ptrs", None)
        or getattr(layer, "_exl3_fused_temps", None) is None
    ):
        return False
    return all(
        int(pack[p].K) == BITS
        and bool(pack[p].mcg)
        and not bool(pack[p].mul1)
        for pack in inners
        for p in ("gate", "up", "down")
    )


def layer_ineligible_reason(layer):
    bits = int(getattr(layer, "_exl3_bits", 0) or getattr(layer, "_exl3_k", 0) or 0)
    inners = getattr(layer, "_exl3_inners", None)
    if bits != BITS:
        return f"bits_{bits}"
    if getattr(layer, "_exl3_hidden_size", None) != HIDDEN:
        return "hidden"
    if getattr(layer, "_exl3_intermediate_local", None) != INTERMEDIATE_LOCAL:
        return "intermediate_local"
    if not inners:
        return "no_inners"
    if not 1 <= len(inners) <= EXPERTS_MAX:
        return f"expert_count_{len(inners)}"
    if not getattr(layer, "_exl3_ptrs", None):
        return "no_ptrs"
    if getattr(layer, "_exl3_fused_temps", None) is None:
        return "no_fused_temps"
    if not layer_eligible(layer):
        return "quant_not_k4_mcg"
    return None


def call_ineligible_reason(x, ids, weights, layer, limit):
    native = getattr(layer, "_glm53_coop_native", None)
    if native is None:
        return "native_unattached"
    if len(getattr(x, "shape", ())) != 2:
        return "input_rank"
    rows = int(x.shape[0])
    if rows < 1:
        return "rows_invalid"
    if rows > ROWS_MAX:
        return "rows_out_of_range"
    if x.shape[1] != HIDDEN:
        return "hidden_mismatch"
    if tuple(ids.shape) != (rows, TOPK) or tuple(weights.shape) != (rows, TOPK):
        return "route_shape"
    if not getattr(x, "is_cuda", False):
        return "input_not_cuda"
    device = getattr(native, "device", None)
    if ids.device != device or weights.device != device or x.device != device:
        return "device_mismatch"
    if not all(t.is_contiguous() for t in (x, ids, weights)):
        return "layout_noncontiguous"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input_dtype"
    if ids.dtype != torch.int64:
        return "ids_dtype"
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "weights_dtype"
    try:
        finite_limit = math.isfinite(float(limit)) and float(limit) >= 0
    except (TypeError, ValueError):
        finite_limit = False
    if not finite_limit:
        return "limit_invalid"
    if float(limit) != ACTIVATION_LIMIT:
        return "activation_limit_unsupported"
    if getattr(native, "row_policy", {}).get(rows) == "stock":
        return "policy_stock"
    return None


def call_eligible(x, ids, weights, layer, limit):
    return call_ineligible_reason(x, ids, weights, layer, limit) is None


def _require(condition, message):
    if not condition:
        raise CooperativeMoEError(message)


def verify_bundle(library_root):
    """Fail closed before dlopen, including under python -O; no old TP2 digest."""
    root = Path(library_root)
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "unsupported build manifest")
    expected = {"abi": ABI_VERSION, "hidden": HIDDEN, "intermediate_local": INTERMEDIATE_LOCAL,
                "topk": TOPK, "experts_max": EXPERTS_MAX, "row_capacities": list(ROW_CAPACITIES),
                "counter_lengths": {str(k): v for k, v in CTR_LENS.items()}, "params_size": PARAMS_SIZE, "activation_limit": ACTIVATION_LIMIT}
    _require(manifest.get("contract") == expected, "build manifest shape/ABI mismatch")
    files = manifest.get("files", {})
    _require("cooperative_moe.so" in files and "runtime.py" in files and "dispatch_policy.json" in files, "incomplete build manifest")
    for relative, expected_hash in files.items():
        path = (root / relative).resolve()
        _require(path.is_relative_to(root.resolve()), "manifest path escapes bundle")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected_hash, f"unvalidated {relative} digest {actual}")
    _require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == files["runtime.py"],
             "loaded adapter does not match build manifest")
    policy = json.loads((root / "dispatch_policy.json").read_text())
    qualification = (os.environ.get("GLM53_COOP_QUALIFICATION") == "1"
                     and os.environ.get("GLM53_COOP_MAINTENANCE_TEST") == "1")
    _require(policy.get("native_sha256") == files["cooperative_moe.so"] or qualification,
             "row policy was not measured for this native binary; qualify and reprofile before production")
    return root / "cooperative_moe.so"


def load_row_policy(root):
    policy = json.loads((Path(root) / "dispatch_policy.json").read_text())
    _require(policy.get("schema") == 1, "unsupported row policy schema")
    rows = policy.get("rows", {})
    _require(set(rows) == {str(n) for n in range(1, ROWS_MAX + 1)}, "incomplete row policy")
    _require(all(value == "stock" or (type(value) is int and value in GEOMETRY_TILES)
                 for value in rows.values()), "invalid row policy geometry")
    return {int(key): value for key, value in rows.items()}


class CoopLaunch:
    def __init__(self, device, library_root, geometry=None):
        self.device = device
        self.geometry = DEFAULT_GEOMETRY if geometry is None else int(geometry)
        _require(
            self.geometry in GEOMETRY_TILES,
            f"unsupported cooperative geometry {self.geometry}",
        )
        capability = torch.cuda.get_device_capability(device)
        _require(
            capability == (12, 1),
            f"cooperative MoE requires SM121, got {capability}",
        )
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "initialize cooperative MoE before graph capture",
        )
        path = verify_bundle(library_root)
        self.row_policy = load_row_policy(library_root)
        if geometry is not None:
            _require(os.environ.get("GLM53_COOP_MAINTENANCE_TEST") == "1",
                     "forced geometry is only available in an explicit maintenance gate")
            self.row_policy = {rows: int(geometry) for rows in range(1, ROWS_MAX + 1)}
        self.library = C.CDLL(str(path))
        abi = int(self.library.glm53_coop_abi())
        _require(abi == ABI_VERSION, f"unexpected glm53_coop_abi {abi}")
        self.launch = self.library.glm53_coop_launch
        self.launch.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.launch.restype = C.c_int
        with torch.cuda.device(device):
            self.library.glm53_coop_info.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int)]
            self.library.glm53_coop_info.restype = C.c_int
            self.resources = {}
            for capacity in ROW_CAPACITIES:
                for prepared_geometry in GEOMETRY_TILES:
                    info = (C.c_int * 18)()
                    status = int(self.library.glm53_coop_info(BITS, prepared_geometry, capacity, info))
                    _require(status == 0, f"glm53_coop_info capacity={capacity} geometry={prepared_geometry} failed status={status}")
                    _require(info[16] == CTR_LENS[capacity] and info[17] == PARAMS_SIZE,
                             f"native layout mismatch capacity={capacity} ctr={info[16]} params={info[17]}")
                    _require(info[4] >= 1 and info[9] >= 1 and info[14] >= 1,
                             "cooperative kernel occupancy < 1")
                    self.resources[f"{capacity}:{prepared_geometry}"] = tuple(int(v) for v in info)
            self.occupancy = tuple(self.resources[f"64:{self.geometry}"][i] for i in (4, 9, 14))
            self.scratch = [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in (
                    ((SLOTS_MAX, HIDDEN), torch.float16),
                    ((SLOTS_MAX, HIDDEN), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, HIDDEN), torch.float32),
                )
            ]
            # Fixed staging buffers: copy_/out= operations avoid device allocations
            # in eager calls and are safe to record during CUDA graph capture.
            self.input = torch.empty((ROWS_MAX, HIDDEN), dtype=torch.float16, device=device)
            self.route_ids = torch.empty(SLOTS_MAX, dtype=torch.int64, device=device)
            self.safe_ids = torch.empty(SLOTS_MAX, dtype=torch.int64, device=device)
            self.route_weights = torch.empty((ROWS_MAX, TOPK), dtype=torch.float16, device=device)
            self.invalid_a = torch.empty(SLOTS_MAX, dtype=torch.bool, device=device)
            self.invalid_b = torch.empty(SLOTS_MAX, dtype=torch.bool, device=device)
            self.output = torch.empty((ROWS_MAX, HIDDEN), dtype=torch.float32, device=device)
            self.counters = {capacity: torch.zeros(CTR_LENS[capacity], dtype=torch.int32, device=device)
                             for capacity in ROW_CAPACITIES}

    def __call__(self, module, x, ids, weights, layer, inners, expert_map, limit):
        _require(
            call_eligible(x, ids, weights, layer, limit),
            "cooperative launch preconditions failed",
        )
        rows = int(x.shape[0])
        experts = len(inners)
        if expert_map is not None:
            _require(expert_map.device == self.device and expert_map.dtype == torch.int64
                     and expert_map.ndim == 1 and expert_map.is_contiguous(),
                     "expert_map must be a contiguous int64 vector on the candidate device")
        slots = rows * TOPK
        xh = self.input[:rows]
        xh.copy_(x)
        local = self.route_ids[:slots]
        safe = self.safe_ids[:slots]
        invalid = self.invalid_a[:slots]
        other = self.invalid_b[:slots]
        flat = ids.view(-1)
        # Preserve the stock mapper's n_local sentinel without temporary tensors.
        n_global = experts if expert_map is None else expert_map.numel()
        torch.lt(flat, 0, out=invalid)
        torch.ge(flat, n_global, out=other)
        torch.logical_or(invalid, other, out=invalid)
        if expert_map is None:
            local.copy_(flat)
        elif n_global:
            torch.clamp(flat, min=0, max=n_global - 1, out=safe)
            torch.index_select(expert_map, 0, safe, out=local)
        else:
            local.fill_(experts)
        torch.lt(local, 0, out=other)
        torch.logical_or(invalid, other, out=invalid)
        torch.ge(local, experts, out=other)
        torch.logical_or(invalid, other, out=invalid)
        local.masked_fill_(invalid, experts)
        rw = self.route_weights[:rows]
        rw.copy_(weights)
        out = self.output[:rows]
        tables = [layer._exl3_ptrs[k] for k in PTR_KEYS]
        _require(
            all(
                t.device == x.device
                and t.dtype == torch.int64
                and t.is_contiguous()
                and tuple(t.shape) == (experts,)
                for t in tables
            ),
            "cooperative pointer tables are not contiguous int64 expert vectors",
        )
        tensors = [xh, local, rw, *tables, *self.scratch, self.counters[row_capacity(rows)], out]
        pointers = (C.c_void_p * 20)(*[t.data_ptr() for t in tensors])
        status = self.launch(
            pointers,
            BITS,
            rows,
            experts,
            float(limit),
            self.row_policy[rows],
            0,
            C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            # Never retry stock after a partially launched CUDA operation.
            raise RuntimeError(f"cooperative MoE kernel CUDA launch status {status}")
        layer._exl3_last_fat_fallback = "none"
        layer._exl3_last_fat_reason = "no_fat_experts"
        return out


def _log_selection(module, diag, rows, selected, reason, capturing, geometry=None):
    kind = "capture" if capturing else "eager"
    key = (kind, int(rows), bool(selected), str(reason), geometry)
    if key in diag.seen_selection:
        return
    diag.seen_selection.add(key)
    record = {"selected": bool(selected), "reason": reason, "kind": kind, "geometry": geometry}
    store = diag.capture_selection if capturing else diag.eager_selection
    store[int(rows)] = record
    module.logger.info(
        "cooperative MoE %s-time selection rows=%s selected=%s reason=%s "
        "geometry=%s (%s); this is not a CUDA-graph replay count"
        % (
            kind,
            rows,
            bool(selected),
            reason,
            geometry,
            GEOMETRY_TILES.get(geometry, "stock"),
        )
    )


def install(module, library_root="/root/.cache/vllm/cooperative_moe", *, enabled=False):
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not enabled and os.environ.get("GLM53_COOPERATIVE_MOE", "0") != "1":
        return False
    if getattr(module, "_glm53_coop_installed", False):
        raise RuntimeError("cooperative MoE already installed")
    enforce_serialized_execution()
    geometry = resolve_geometry()
    qualification = os.environ.get("GLM53_COOP_QUALIFICATION") == "1"
    original_process = module.Exl3MoEMethod.process_weights_after_loading
    original_apply = module.apply_exl3_fused_moe
    kernels = {}
    diag = CoopDiagnostics()
    diag.geometry = geometry
    diag.geometry_name = GEOMETRY_TILES[geometry]

    def process(method, layer):
        result = original_process(method, layer)
        if hasattr(layer, "_glm53_coop_native"):
            delattr(layer, "_glm53_coop_native")
        if layer_eligible(layer):
            device = layer.w13_trellis.device
            key = str(device)
            if key not in kernels:
                kernels[key] = CoopLaunch(device, library_root, geometry=geometry if qualification else None)
                diag.native_prepared = True
                launch = kernels[key]
                diag.row_policies[key] = getattr(launch, "row_policy", {})
                module.logger.info(
                    "cooperative MoE native prepared: policy_mode=%s row_policy=%s "
                    "resources_by_capacity_geometry=%s library=%s"
                    % ("qualification" if qualification else "measured",
                       diag.row_policies[key], getattr(launch, "resources", {}), library_root)
                )
            layer._glm53_coop_native = kernels[key]
            diag.eligible_layers += 1
        else:
            reason = layer_ineligible_reason(layer) or "ineligible"
            diag.ineligible_layers += 1
            diag.ineligible_reasons[reason] = diag.ineligible_reasons.get(reason, 0) + 1
        return result

    def apply(x, ids, weights, layer, inners, expert_map, limit):
        if not diag.summary_logged:
            diag.summary_logged = True
            module.logger.info(
                "cooperative MoE prepared-layer summary: eligible=%s ineligible=%s "
                "ineligible_reasons=%s row_policies=%s native_prepared=%s"
                % (
                    diag.eligible_layers,
                    diag.ineligible_layers,
                    diag.ineligible_reasons or "{}",
                    diag.row_policies,
                    diag.native_prepared,
                )
            )
        capturing = bool(torch.cuda.is_current_stream_capturing())
        reason = call_ineligible_reason(x, ids, weights, layer, limit)
        selected = reason is None
        rows = int(getattr(x, "shape", [0])[0]) if getattr(x, "shape", None) else 0
        _log_selection(
            module,
            diag,
            rows,
            selected,
            "cooperative" if selected else reason,
            capturing,
            getattr(getattr(layer, "_glm53_coop_native", None), "row_policy", {}).get(rows),
        )
        if not selected:
            if reason not in ("native_unattached", "rows_out_of_range", "activation_limit_unsupported", "policy_stock"):
                raise CooperativeMoEError(f"invalid cooperative input: {reason}")
            return original_apply(x, ids, weights, layer, inners, expert_map, limit)
        return layer._glm53_coop_native(
            module, x, ids, weights, layer, inners, expert_map, limit
        )

    module.Exl3MoEMethod.process_weights_after_loading = process
    module.apply_exl3_fused_moe = apply
    module._glm53_coop_original_apply = original_apply
    module._glm53_coop_installed = True
    module._glm53_coop_geometry = geometry if qualification else None
    module._glm53_coop_geometry_name = GEOMETRY_TILES[geometry] if qualification else "measured row policy"
    module._glm53_coop_diag = diag
    module.logger.info(
        "Fixed-shape cooperative MoE wrappers installed: K4 MCG, 1..64 rows, "
        "selection=%s; native prepare runs after weight load; "
        "prefill/E3 and other calls stay stock. Shared scratch must not overlap."
        % (f"qualification geometry {geometry}" if qualification else "measured physical-row policy")
    )
    return True
