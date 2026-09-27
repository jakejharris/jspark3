# B5 prefix-only target verification for the JSPARK3 v1 arm (prefix verification reference), JSPARK3 v2 run, 2026-09-05.
#
# Installed by zzz_b5_prefix_verify.pth ("import b5_prefix_verify") in /usr/local/lib/python3.12/dist-packages, the
# mechanism the image already uses (glm53_video.pth) and B6 used for its instrument. Bind-mount this file, the .pth and
# b5_controller.py read-only into the candidate lookalike; nothing under the arm's paths is written.
#
# ONE named mechanism relative to the parent arm (H5 / P5 of the design of record):
#   keep DFlash2 proposing all seven drafts; when the request-local controller says so, verify only the first three
#   drafts plus the bonus position on a PHYSICAL four-token graph.
#
# How (each point maps to a section of B5-DESIGN-AND-BUILD.md):
#   1. Graph family: CudaGraphManager._init_candidates gains uniform decode descriptors with uniform_token_count=4 at
#      the capture sizes in B5_NARROW_CAPTURE_SIZES (default "4": one graph, 1 request x 4 tokens). The existing
#      8-token family is untouched, so a width-7 step on this arm is byte-for-byte the parent's execution.
#   2. Trim: GPUModelRunner.execute_model receives a shallow copy of the SchedulerOutput in which a narrowed request
#      has num_scheduled_tokens 8 -> 4 and scheduled_spec_decode_tokens 7 -> the first 3. Everything downstream
#      (prepare_inputs, attention and KDA metadata, rejection sampling, post_update, drafter context) then sees a
#      consistent 4-token, 3-draft request. The scheduler's own copy keeps 7; its rollback num_rejected = 7 - accepted
#      lands on the same num_computed_tokens as the runner's 4 - (3 - accepted). Every scheduler-visible outcome of a
#      narrow step (7 scheduled, <= 3 accepted) is one the parent arm already produces.
#   3. Feedback: num_sampled is copied device-to-host per step on a side stream into a two-slot pinned buffer and
#      consumed one step late after an explicit event wait (bounded, the same scheme upstream's adaptive verification
#      uses), so every TP rank consumes identical acceptance history. Step timing is recorded with CUDA events and
#      LOGGED ONLY: it never feeds the decision, because per-rank timing would make ranks decide different widths.
#   4. Calibration: after graph capture, the 4-token and 8-token decode graphs are timed with dummy runs (N replays,
#      median) and rank 0's values are broadcast to all ranks as T4_seed / T7_seed. Receipt printed and written.
#   5. Controller: b5_controller.B5Controller (pure Python, rank-invariant inputs). Fallback to width 7 whenever the
#      batch has >= 2 requests or any prefill, or the request is unknown.
#   6. Switches: B5_FORCE_WIDTH=3|7 bypasses the controller for the whole container (a dedicated DSF panel run);
#      per-request extra_args {"b5": "off"} forces width 7 and {"b5": "force3"} forces width 3 (pure lane) for that
#      request only, so one container serves the off arm, the controller arm and the controller arm (rank-safe pairing).
#   7. Census: B5_CENSUS=1 forces eager dispatch (for the first B5_CENSUS_STEPS decode steps only, if set) and counts
#      per MoE layer per step [distinct LOCAL experts touched, (token, expert) rows routed to this rank, tokens]
#      (device buffer, D2H at step end, one JSONL per rank so the three ranks can be joined by step and layer), plus
#      num_tokens_after_padding, so "four positions' worth of work" is a row count. The per-rank per-layer per-step
#      split also answers the collective-census lane's open question (expert ownership versus step fluctuation).
#      Steps after the census window run on the graphs and give the real step times.
#   8. Hash gate: the module refuses (loudly) unless the target sources match the image bytes this was built against.
#
# Output: /evidence/b5-steps-<hostname>-<pid>.jsonl (one line per step per rank) and /evidence/b5-receipt-<hostname>-<pid>.json.
import atexit
import dataclasses
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import socket
import statistics
import sys
import threading
import time
from collections import deque

_ENABLED = os.environ.get("B5_PREFIX_VERIFY", "0") == "1"
_OUT_DIR = os.environ.get("B5_OUT", "/evidence")
_NARROW = 3                                             # drafts verified on a narrow step (P5: exactly 3)
_NARROW_Q = _NARROW + 1                                 # 4 query positions
_FULL_Q = 8                                             # 1 + num_speculative_tokens on this arm
_NARROW_CAPTURE_SIZES = [int(x) for x in os.environ.get("B5_NARROW_CAPTURE_SIZES", "4").split(",") if x]
_FORCE_WIDTH = os.environ.get("B5_FORCE_WIDTH")         # "3" | "7" | None
_FORCE_WIDTH = int(_FORCE_WIDTH) if _FORCE_WIDTH else None
_CENSUS = os.environ.get("B5_CENSUS", "0") == "1"
_CENSUS_STEPS = int(os.environ.get("B5_CENSUS_STEPS", "0"))   # >0: eager census only for the first N decode steps, then graphs
_CALIB_REPLAYS = int(os.environ.get("B5_CALIB_REPLAYS", "20"))
_T4_SEED_OVERRIDE = os.environ.get("B5_T4_SEED_MS")
_T7_SEED_OVERRIDE = os.environ.get("B5_T7_SEED_MS")

# sha256 of the image files this module was built against (evidence/B5/sources/MANIFEST.txt; A0-MATCH where the A0
# census captured the file). A mismatch means the runner or graph code moved under us: REFUSE, do not guess.
EXPECTED_SHA256 = {
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py": "308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0",
    "vllm/v1/worker/gpu/model_runner.py": "f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415",
    "vllm/v1/worker/gpu/cudagraph_utils.py": "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f",
    "vllm/v1/worker/gpu/input_batch.py": "3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410",
    "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py": "bd7f4c63d1196cb53bee0a81339aa5651e36938fc38b73c4ce89d978e0176a87",
    "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py": "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48",
    "vllm/v1/worker/gpu/model_states/mamba_hybrid.py": "cc6382b88cf4f66902516366c64dddc5bcd9575a3da2f3746b2fdbbddc388e17",
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py": "659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a",
    # v1.4: the composed scheduler (cadence-sched v6 decode floor + mamba align chunking, patch-contract
    # apply_v14_core after_sha256). Its hooks never touch the spec-decode rows or rollback this module narrows.
    "vllm/v1/core/sched/scheduler.py": "0b086dd3cc0febc4924202ca1fef8809b8290b9a59ef15fa80004cef4edeb937",
    "vllm/model_executor/layers/quantization/exl3.py": "71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174",
}

# v1.6 optional transforms run before this import-time gate.  Preserve the
# v1.5 literal above (and its exact key set) as the all-off contract, then
# select only the sealed on-state hashes.  Unknown or partially configured
# states refuse before any vLLM model import.
_V16_COOP = os.environ.get("JSPARK3_V16_COOP", "0")
_V16_ADAPTIVE_K = os.environ.get("GLM53_ADAPTIVE_K", "off")
if _V16_COOP not in ("0", "1"):
    raise RuntimeError("JSPARK3_V16_COOP must be 0 or 1")
if _V16_ADAPTIVE_K not in ("off", "ema"):
    raise RuntimeError("GLM53_ADAPTIVE_K must be off or ema")
if _V16_COOP == "1":
    EXPECTED_SHA256[
        "vllm/model_executor/layers/quantization/exl3.py"
    ] = "89111aaf1d3082dd62c76ecb8c378fabf3f98619bee5124ee4ac3622e1e2eb4f"
if _V16_ADAPTIVE_K == "ema":
    EXPECTED_SHA256[
        "vllm/v1/core/sched/scheduler.py"
    ] = "0d58e688019ddfa5952990be2ea751b71d96bea7a35c4c31c19c672b0890f764"
    EXPECTED_SHA256[
        "vllm/v1/worker/gpu/cudagraph_utils.py"
    ] = "6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa"
_SITE = "/usr/local/lib/python3.12/dist-packages"
_log_lock = threading.Lock()


def _log(msg):
    try:
        sys.stderr.write("[b5_prefix_verify pid=%d] %s\n" % (os.getpid(), msg))
        sys.stderr.flush()
    except Exception:
        pass


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_gate(site=_SITE):
    """Return (ok, report). ok is False on any mismatch or unreadable file."""
    report = {}
    ok = True
    for rel, expected in EXPECTED_SHA256.items():
        path = os.path.join(site, rel)
        try:
            got = _sha256_file(path)
        except Exception as e:
            got = "UNREADABLE:%r" % (e,)
        report[rel] = {"expected": expected, "observed": got, "match": got == expected}
        ok = ok and got == expected
    return ok, report


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.hostname = socket.gethostname()
        self.rank = os.environ.get("NODE_RANK", "?")
        self.fh = None
        self.step = 0
        self.torch = None
        self.ctl = None                   # B5Controller
        self.audit_requests = set()
        self.draft_history = {}
        self.forced_full = set()          # req_ids with extra_args {"b5": "off"}   -> always width 7
        self.forced_lane = set()          # req_ids with extra_args {"b5": "force3"} -> always width 3 (pure lane)
        self.b5_requested = {}            # req_id -> raw extra_args.b5, None when absent; attestation only
        self.cur = None                   # current step record
        self.widths = {}                  # req_id -> draft width used this step (3 or 7)
        self.pending = deque()            # (record, req_ids, widths, event, slot_idx, ev_start, ev_end)
        self.slots = None                 # two pinned CPU buffers
        self.slot_idx = 0
        self.copy_stream = None
        self.copy_events = None
        self.calib = None                 # {"T4_ms":..., "T7_ms":..., "per_rank": ...}
        self.census_buf = None            # device int32 [num_layers, 2] (unique local experts, tokens)
        self.census_layer = 0
        self.capturing = False
        self.active = False
        self.receipt = {}
        self.dispatch_orig = None
        self.adaptive_narrow_steps = 0
        self.adaptive_all_narrow_steps = 0
        self.adaptive_padded_narrow_steps = 0

    def out_path(self, kind):
        return os.path.join(_OUT_DIR, "b5-%s-%s-%d.%s" % (kind, self.hostname, os.getpid(), "jsonl" if kind == "steps" else "json"))

    def write(self, rec):
        try:
            with _log_lock:
                if self.fh is None:
                    os.makedirs(_OUT_DIR, exist_ok=True)
                    self.fh = open(self.out_path("steps"), "a", buffering=1)
                self.fh.write(json.dumps(rec, default=float) + "\n")
        except Exception as e:
            _log("write failed: %r" % (e,))

    def write_receipt(self):
        try:
            os.makedirs(_OUT_DIR, exist_ok=True)
            with open(self.out_path("receipt"), "w") as f:
                json.dump(self.receipt, f, indent=1, default=float)
        except Exception as e:
            _log("receipt write failed: %r" % (e,))


_S = _State()


def _torch():
    if _S.torch is None:
        import torch
        _S.torch = torch
    return _S.torch


def _ev():
    """Timing event on the current stream, or None while capturing."""
    try:
        torch = _torch()
        if torch.cuda.is_current_stream_capturing():
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ graph family
def patch_cudagraph_utils(mod):
    CGM = mod.CudaGraphManager
    Model = mod.ModelCudaGraphManager
    Desc = mod.BatchExecutionDescriptor
    CUDAGraphMode = mod.CUDAGraphMode
    orig_init_candidates = CGM._init_candidates

    def _init_candidates(self):
        orig_init_candidates(self)
        try:
            if not isinstance(self, Model):
                return                      # the drafter's DFlashCudaGraphManager keeps its own 8-query layout
            if self.decode_query_len != _FULL_Q or self.varlen_decode:
                return
            decode_mode = self.cudagraph_mode.decode_mode()
            if decode_mode != CUDAGraphMode.FULL or not self.cudagraph_mode.separate_routine():
                _log("narrow graph family skipped: cudagraph_mode=%s" % (self.cudagraph_mode,))
                return
            added = []
            for size in _NARROW_CAPTURE_SIZES:
                if size % _NARROW_Q != 0 or size // _NARROW_Q > self.max_num_reqs:
                    continue
                desc = Desc(cg_mode=decode_mode, num_tokens=size, num_reqs=size // _NARROW_Q,
                            uniform_token_count=_NARROW_Q, num_active_loras=0)
                descs = self._capture_descs.setdefault(decode_mode, [])
                if desc in descs:
                    continue
                descs.append(desc)
                descs.sort(key=lambda d: d.num_tokens, reverse=True)
                # priority-ordered candidates: put the narrow desc FIRST for every token count it can serve. The
                # existing 8-token descs still serve uniform-8 batches because _is_compatible requires the uniform
                # token count to match exactly; prefill batches (uniform None) match neither and stay eager as today.
                for i in range(1, size + 1):
                    key = (i, 0)
                    lst = self._candidates.get(key, [])
                    if desc not in lst:
                        self._candidates[key] = [desc] + list(lst)
                added.append((size, size // _NARROW_Q))
            _S.receipt["narrow_graph_descs"] = added
            _log("B5 narrow graph family added: %s (num_tokens, num_reqs) uniform_token_count=%d" % (added, _NARROW_Q))
        except Exception as e:
            _log("narrow graph family patch failed: %r" % (e,))
            raise

    CGM._init_candidates = _init_candidates
    orig_run_fullgraph = CGM.run_fullgraph
    def run_fullgraph(self, desc):
        if _S.cur is not None and isinstance(self, mod.ModelCudaGraphManager):
            _S.cur['graph'] = {'num_tokens': desc.num_tokens,
                               'num_reqs': desc.num_reqs,
                               'uniform_token_count': desc.uniform_token_count,
                               'mode': str(desc.cg_mode), 'captured': desc in self.graphs}
        return orig_run_fullgraph(self, desc)
    CGM.run_fullgraph = run_fullgraph


# ------------------------------------------------------------------------------------------------ runner
def _eligible_single_spec_row(so):
    """True iff the batch is exactly one request, scheduled as a full spec-verify row (8 tokens, 7 drafts)."""
    nst = so.num_scheduled_tokens
    if len(nst) != 1 or so.scheduled_new_reqs:
        return False
    rid, n = next(iter(nst.items()))
    spec = so.scheduled_spec_decode_tokens.get(rid)
    return spec is not None and len(spec) == _FULL_Q - 1 and n == _FULL_Q


def rewrite_scheduler_output(so, decide):
    """Return (so_or_copy, widths). decide(req_id, num_reqs, has_prefill) -> (width, rule)."""
    widths, rules = {}, {}
    nst = so.num_scheduled_tokens
    num_reqs = len(nst)
    spec = so.scheduled_spec_decode_tokens or {}
    # a row is a spec-decode row iff exactly one non-draft token is scheduled with its drafts
    has_prefill = bool(so.scheduled_new_reqs) or any(
        nst[r] != 1 + len(spec.get(r, ())) for r in nst)
    for rid in nst:
        if rid not in spec:
            continue
        # Grammar masks describe the scheduler's draft positions plus bonus.
        # A worker-only trim would leave eight masks for four logits. Keep the
        # upstream constrained speculative path, even under a forced B5 width.
        # B5 only narrows single-request batches. Leave mixed-batch controller
        # feedback unchanged; adaptive-k already excludes structured requests.
        if num_reqs == 1 and getattr(so, "has_structured_output_requests", False):
            widths[rid] = _FULL_Q - 1
            rules[rid] = "structured-output"
            continue
        w, rule = decide(rid, num_reqs, has_prefill)
        widths[rid] = w
        rules[rid] = rule
    narrow = [r for r, w in widths.items() if w == _NARROW]
    if not narrow:
        return so, widths, rules
    if not _eligible_single_spec_row(so):
        # the controller only returns 3 for a single-row batch; guard anyway
        for r in narrow:
            widths[r] = _FULL_Q - 1; rules[r] = rules[r] + ";R0-guard"
        return so, widths, rules
    rid = narrow[0]
    new_nst = dict(nst)
    new_spec = dict(spec)
    new_nst[rid] = _NARROW_Q
    new_spec[rid] = list(spec[rid][:_NARROW])
    total = so.total_num_scheduled_tokens - (_FULL_Q - _NARROW_Q)
    so2 = dataclasses.replace(so, num_scheduled_tokens=new_nst, scheduled_spec_decode_tokens=new_spec,
                              total_num_scheduled_tokens=total)
    return so2, widths, rules


def adaptive_narrow_descriptor(so):
    """Describe actual query-4 steps, including B5's c1, on an ema boot."""
    scheduled = so.num_scheduled_tokens
    spec = so.scheduled_spec_decode_tokens or {}
    if _V16_ADAPTIVE_K != "ema" or not scheduled or so.scheduled_new_reqs:
        return None
    if any(scheduled[rid] != _NARROW_Q or len(spec.get(rid, ())) != _NARROW for rid in scheduled):
        return None
    requests = len(scheduled)
    return {'requests': requests, 'query_width': _NARROW_Q,
            'logical_rows': requests * _NARROW_Q}


def adaptive_runtime_receipt(desc, physical_rows, physical_requests, graph):
    """Account physical execution, independently of scheduler decisions."""
    _S.adaptive_all_narrow_steps += 1
    if desc["requests"] >= 2:
        _S.adaptive_narrow_steps += 1
    padded = physical_rows != desc["logical_rows"] or physical_requests != desc["requests"]
    if padded:
        _S.adaptive_padded_narrow_steps += 1
    captured = bool(graph and graph.get("captured")
                    and graph.get("num_tokens") == physical_rows
                    and graph.get("num_reqs") == physical_requests
                    and graph.get("uniform_token_count") == desc["query_width"])
    epoch = getattr(_S.ctl, "epoch_mode", None)
    return {
        "epoch": getattr(epoch, "epoch", "boot"),
        "fault_accept_delta": getattr(epoch, "fault_accept_delta", 0),
        "mode": getattr(_S.ctl, "effective_mode", "unknown"),
        "narrow_steps": _S.adaptive_narrow_steps,
        "all_narrow_steps": _S.adaptive_all_narrow_steps,
        "padded_narrow_steps": _S.adaptive_padded_narrow_steps,
        "physical_rows": physical_rows,
        "physical_requests": physical_requests,
        "query_width": desc["query_width"],
        "rank": _S.rank,
        "requests": desc["requests"],
        "captured": captured,
        "status": "FAIL" if padded or not captured else "PASS",
    }


def _consume_landed():
    """Feed the controller the oldest pending step's accepted counts (explicit event wait: bounded, identical on
    every rank because every rank waits on its own copy of the same values)."""
    torch = _torch()
    while _S.pending and len(_S.pending) >= 1:
        rec, req_ids, widths, ev, slot, ev_start, ev_end = _S.pending[0]
        if len(_S.pending) < 2 and ev is not None:
            # keep exactly one step of latency: consume when a newer step exists or when it has landed already
            if not ev.query():
                return
        _S.pending.popleft()
        if ev is not None:
            ev.synchronize()
            vals = _S.slots[slot][: len(req_ids)].tolist()
        else:
            vals = [None] * len(req_ids)
        gpu_ms = None
        try:
            if ev_start is not None and ev_end is not None and ev_end.query():
                gpu_ms = ev_start.elapsed_time(ev_end)
        except Exception:
            pass
        rec["a"] = dict(zip(req_ids, vals))
        rec["gpu_step_ms"] = gpu_ms
        for rid, a in zip(req_ids, vals):
            if a is not None and rid in widths:
                _S.ctl.record(rid, widths[rid], int(a), None)   # timing never feeds the decision
        _S.write(rec)


def patch_model_runner(mod):
    R = mod.GPUModelRunner
    from b5_controller import B5Controller, FULL, NARROW  # noqa: F401
    torch = _torch()

    # ---- request lifecycle: controller entries and the per-request A/B switch ----
    orig_add = R.add_requests

    def add_requests(self, scheduler_output):
        try:
            for nr in scheduler_output.scheduled_new_reqs:
                _S.ctl.add(nr.req_id)
                sp = getattr(nr, "sampling_params", None)
                xa = getattr(sp, "extra_args", None) if sp is not None else None
                flag = str(xa.get("b5", "")).lower() if isinstance(xa, dict) else ""
                _S.forced_full.discard(nr.req_id); _S.forced_lane.discard(nr.req_id)
                if isinstance(xa, dict) and xa.get('b31_census') == 1:
                    _S.audit_requests.add(nr.req_id)
                if flag == "off":
                    _S.forced_full.add(nr.req_id)
                elif flag == "force3":
                    _S.forced_lane.add(nr.req_id)
        except Exception as e:
            _log("add_requests hook failed: %r" % (e,))
        return orig_add(self, scheduler_output)

    R.add_requests = add_requests

    # ---- the trim ----
    orig_exec = R.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", False) or (len(args) >= 2 and bool(args[1]))
        if dummy or not _S.active:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        rec = None
        try:
            _consume_landed()
            for rid in (scheduler_output.finished_req_ids or ()):
                _S.ctl.remove(rid); _S.forced_full.discard(rid); _S.forced_lane.discard(rid)
                _S.audit_requests.discard(rid); _S.draft_history.pop(rid, None)
                _S.b5_requested.pop(rid, None)
            for rid in (scheduler_output.preempted_req_ids or ()):
                _S.ctl.remove(rid)
            # execute_model builds the record before the runner calls add_requests.
            # Capture new metadata here so the very first prefill attests correctly.
            # Keep it through preemption, like the request-local forced-width sets.
            for nr in scheduler_output.scheduled_new_reqs:
                xa = getattr(getattr(nr, "sampling_params", None), "extra_args", None)
                _S.b5_requested[nr.req_id] = xa.get("b5") if isinstance(xa, dict) else None

            def decide(rid, num_reqs, has_prefill):
                if rid in _S.forced_full:
                    return FULL, "AB-off"
                if rid in _S.forced_lane:
                    # pure controller for this request only; the single-row guard in rewrite_scheduler_output still applies
                    return (NARROW if (num_reqs == 1 and not has_prefill) else FULL), "AB-force3"
                return _S.ctl.decide(rid, num_reqs, has_prefill)

            scheduler_output, widths, rules = rewrite_scheduler_output(scheduler_output, decide)
            adaptive_desc = adaptive_narrow_descriptor(scheduler_output)
            _S.step += 1
            _S.widths = widths
            rec = {"kind": "step", "step": _S.step, "rank": _S.rank, "t_wall": time.time(),
                   "req_ids": list(scheduler_output.num_scheduled_tokens.keys()),
                   "widths": widths, "rules": rules,
                   "b5_requested": {rid: _S.b5_requested[rid] for rid in scheduler_output.num_scheduled_tokens},
                   "num_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
                   "ev_start": _ev()}
            if adaptive_desc is not None:
                rec["adaptive_narrow_descriptor"] = adaptive_desc
            _S.cur = rec
            _S.census_layer = 0
        except Exception as e:
            _log("execute_model pre failed: %r" % (e,))
            _S.cur = None
        return orig_exec(self, scheduler_output, *args, **kwargs)

    R.execute_model = execute_model

    # ---- acceptance feedback and the step record ----
    orig_post = R.postprocess_sampled

    def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
        rec = _S.cur
        if rec is not None and _S.active:
            try:
                if _S.slots is None:
                    _S.slots = [torch.zeros(self.max_num_reqs, dtype=torch.int32, pin_memory=True) for _ in range(2)]
                    _S.copy_stream = torch.cuda.Stream(self.device)
                    _S.copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
                n = int(num_sampled.shape[0])
                slot = _S.slot_idx
                _S.slot_idx ^= 1
                cur = torch.cuda.current_stream(self.device)
                _S.copy_stream.wait_stream(cur)
                with torch.cuda.stream(_S.copy_stream):
                    _S.slots[slot][:n].copy_(num_sampled, non_blocking=True)
                    num_sampled.record_stream(_S.copy_stream)
                    _S.copy_events[slot].record()
                rec["_slot"] = slot
                rec["_ev"] = _S.copy_events[slot]
            except Exception as e:
                _log("postprocess hook failed: %r" % (e,))
        return orig_post(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)

    R.postprocess_sampled = postprocess_sampled

    orig_st = R.sample_tokens

    def sample_tokens(self, *args, **kwargs):
        rec = _S.cur
        req_ids = None
        try:
            if rec is not None and self.execute_model_state is not None:
                ib = self.execute_model_state.input_batch
                req_ids = list(ib.req_ids)
                rec["num_tokens"] = int(ib.num_tokens)
                rec["num_tokens_after_padding"] = int(ib.num_tokens_after_padding)
                rec["num_reqs_after_padding"] = int(ib.num_reqs_after_padding)
                adaptive_desc = rec.get("adaptive_narrow_descriptor")
                if adaptive_desc is not None:
                    runtime = adaptive_runtime_receipt(
                        adaptive_desc, int(ib.num_tokens_after_padding),
                        int(ib.num_reqs_after_padding), rec.get("graph"),
                    )
                    if (runtime["physical_rows"] != adaptive_desc["logical_rows"]
                            or runtime["physical_requests"] != adaptive_desc["requests"]):
                        rec["PADDING_ON_ADAPTIVE_NARROW_STEP"] = True
                    previous_epoch = _S.receipt.get("adaptive_runtime", {}).get("epoch")
                    _S.receipt["adaptive_runtime"] = runtime
                    if (runtime["status"] == "FAIL" or _S.adaptive_all_narrow_steps == 1
                            or _S.adaptive_narrow_steps == 1 or _S.adaptive_all_narrow_steps % 32 == 0
                            or previous_epoch != runtime["epoch"]):
                        print("[jspark3-v16:adaptive-k-runtime] " +
                              json.dumps(runtime, sort_keys=True, separators=(",", ":")), flush=True)
                    if runtime["status"] == "FAIL":
                        _S.write_receipt()
                if any(w == _NARROW for w in _S.widths.values()) and ib.num_tokens_after_padding != ib.num_tokens:
                    rec["PADDING_ON_NARROW_STEP"] = True   # the fake-graph-padding failure: must never be true
                    _log("NARROW STEP PADDED: num_tokens=%d after_padding=%d" % (ib.num_tokens, ib.num_tokens_after_padding))
        except Exception as e:
            _log("sample_tokens pre failed: %r" % (e,))
        try:
            return orig_st(self, *args, **kwargs)
        finally:
            try:
                if rec is not None:
                    ev_end = _ev()
                    if _CENSUS and _S.census_buf is not None:
                        rec["census"] = _S.census_buf[: _S.census_layer].cpu().tolist()   # eager steps only
                    _S.pending.append((rec, req_ids or [], dict(_S.widths), rec.pop("_ev", None), rec.pop("_slot", None),
                                       rec.pop("ev_start", None), ev_end))
                    _S.cur = None
            except Exception as e:
                _log("sample_tokens post failed: %r" % (e,))

    R.sample_tokens = sample_tokens

    # ---- census: force eager dispatch so the MoE hook runs (timing is not scored on census steps) ----
    if _CENSUS:
        _S.dispatch_orig = mod.dispatch_cg_and_sync_dp

        def dispatch_cg_and_sync_dp(*a, **k):
            if _CENSUS_STEPS == 0 or _S.step <= _CENSUS_STEPS:
                k["need_eager"] = True          # census window: eager so the MoE hook runs
            return _S.dispatch_orig(*a, **k)

        mod.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp

    # ---- calibration after capture; activation receipt ----
    orig_capture = R.capture_model

    def capture_model(self):
        out = orig_capture(self)
        try:
            _calibrate(self, mod)
        except Exception as e:
            _log("calibration failed: %r" % (e,))
            _S.receipt["calibration_error"] = repr(e)
        finally:
            _S.write_receipt()
        return out

    R.capture_model = capture_model


def _calibrate(runner, mod):
    """Time the 4-token and 8-token decode graphs with dummy runs (context 0), median of N replays; broadcast rank 0's
    values so every rank seeds the controller identically."""
    torch = _torch()
    SchedulerOutput = mod.SchedulerOutput

    def dummy_so(num_tokens):
        so = SchedulerOutput.make_empty()
        so.total_num_scheduled_tokens = num_tokens
        so.num_scheduled_tokens = {"_b5_calib_0": num_tokens}
        return so

    def time_graph(num_tokens):
        times = []
        for i in range(_CALIB_REPLAYS + 3):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            runner.execute_model(dummy_so(num_tokens), dummy_run=True)
            runner.execute_model_state = None
            e.record(); e.synchronize()
            if i >= 3:
                times.append(s.elapsed_time(e))
        return statistics.median(times), times

    t4, raw4 = time_graph(_NARROW_Q)
    t7, raw7 = time_graph(_FULL_Q)
    per_rank = {"rank": _S.rank, "T4_ms": t4, "T7_ms": t7, "raw4": raw4, "raw7": raw7}
    try:
        from vllm.distributed.parallel_state import get_tp_group
        t4b, t7b = get_tp_group().broadcast_object((t4, t7), src=0)
    except Exception as e:
        _log("broadcast of calibration failed (%r); using local values" % (e,))
        t4b, t7b = t4, t7
    if _T4_SEED_OVERRIDE:
        t4b = float(_T4_SEED_OVERRIDE)
    if _T7_SEED_OVERRIDE:
        t7b = float(_T7_SEED_OVERRIDE)
    _S.calib = {"T4_seed_ms": t4b, "T7_seed_ms": t7b, "local": per_rank, "context_len": 0, "note":
                "dummy runs at context 0: target graph only, excludes drafter and sampler; seeds the entry estimate, "
                "the boot's paired batteries measure the real step times"}
    from b5_controller import B5Controller
    _S.ctl = B5Controller(t4_seed_ms=t4b, t7_seed_ms=t7b, force_width=_FORCE_WIDTH)
    _S.active = True
    _S.receipt.update({"B5_PREFIX_VERIFY_ACTIVE": True, "calibration": _S.calib, "narrow_width": _NARROW,
                       "force_width": _FORCE_WIDTH, "census": _CENSUS, "rank": _S.rank, "hostname": _S.hostname})
    _log("B5_PREFIX_VERIFY_ACTIVE rank=%s T4_seed=%.2f T7_seed=%.2f force_width=%s census=%s" % (
        _S.rank, t4b, t7b, _FORCE_WIDTH, _CENSUS))
    print("B5_PREFIX_VERIFY_RECEIPT rank=%s T4_seed_ms=%.3f T7_seed_ms=%.3f narrow_width=%d force_width=%s census=%s" % (
        _S.rank, t4b, t7b, _NARROW, _FORCE_WIDTH, _CENSUS), flush=True)


# ------------------------------------------------------------------------------------------------ census
def patch_exl3(mod):
    if not _CENSUS:
        return
    orig_apply = mod.apply_exl3_experts
    torch = _torch()

    def apply_exl3_experts(x, topk_ids, topk_weights, layer, **kw):
        try:
            if _S.cur is not None and not torch.cuda.is_current_stream_capturing():
                if _S.census_buf is None:
                    _S.census_buf = torch.zeros(256, 3, dtype=torch.int32, device=x.device)
                tokens = x.shape[-2]
                emap = mod.pin_exl3_expert_map(layer, x.device)
                flat = topk_ids.reshape(-1)
                uniq = torch.unique(flat)
                if emap is not None:
                    hi = int(emap.numel()) - 1
                    local_u = emap[uniq.clamp(min=0, max=hi)]
                    n_local = (local_u >= 0).sum()                      # distinct LOCAL experts touched this layer
                    rows_local = (emap[flat.clamp(min=0, max=hi)] >= 0).sum()   # (token, expert) rows routed to this rank
                else:
                    n_local = uniq.numel(); rows_local = flat.numel()
                li = _S.census_layer
                if li < 256:
                    _S.census_buf[li, 0] = n_local
                    _S.census_buf[li, 1] = rows_local
                    _S.census_buf[li, 2] = tokens
                _S.census_layer += 1
        except Exception as e:
            _log("census hook failed: %r" % (e,))
        return orig_apply(x, topk_ids, topk_weights, layer, **kw)

    mod.apply_exl3_experts = apply_exl3_experts


# Unscored request-scoped proposal census. No tensors copied for ordinary requests.
def _audit_rid():
    if _S.cur is None:
        return None
    ids = _S.cur.get('req_ids', [])
    return ids[0] if len(ids) == 1 and ids[0] in _S.audit_requests else None


def patch_proposal_census(mod):
    orig = mod.DFlash2Speculator._sample_path
    def sample_path(self, candidate_ids, scores, num_reqs):
        out = orig(self, candidate_ids, scores, num_reqs)
        # Retain the actual graph-output tensor during capture. A Python hook inside
        # _sample_path does not execute during replay; read it only AFTER propose.
        if not hasattr(self, '_b31_candidate_buffers'):
            self._b31_candidate_buffers = {}
        self._b31_candidate_buffers[num_reqs] = candidate_ids
        return out
    mod.DFlash2Speculator._sample_path = sample_path
    orig_propose = mod.DFlash2Speculator.propose
    def propose(self, *args, **kwargs):
        out = orig_propose(self, *args, **kwargs)
        rid = _audit_rid()
        if rid is not None:
            # This proposal will be verified at the NEXT target step, not this step.
            _S.draft_history[rid] = {
                'proposal_step': _S.step,
                'candidate_ids': self._b31_candidate_buffers[1][:1].detach().cpu().tolist()[0],
                'selected': out[:1].detach().cpu().tolist()[0],
            }
        return out
    mod.DFlash2Speculator.propose = propose


def patch_rejection_census(mod):
    orig = mod.RejectionSampler._verify
    def verify(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        rid = _audit_rid()
        if rid is not None:
            processed, sampled, num_sampled = out
            history = _S.draft_history.get(rid)
            draft = args[2].detach().cpu().tolist()
            target = processed.argmax(dim=-1).detach().cpu().tolist()
            landed = int(num_sampled.detach().cpu().tolist()[0])
            row = {'kind': 'proposal-census', 'rank': _S.rank, 'req_id': rid,
                   'step': _S.step, 'history': history, 'draft_sampled': draft,
                   'target_greedy': target, 'num_sampled': landed,
                   'sampled': sampled.detach().cpu().tolist(),
                   'first_rejection_index': landed - 1 if landed < len(target) else None}
            if history is not None and landed < len(target):
                j = landed - 1
                selected = history['selected'][j]
                candidates = history['candidate_ids'][j]
                row['first_rejection'] = {'position': j + 1, 'candidate_set': candidates,
                                          'selected': selected, 'target': target[j],
                                          'target_in_set': target[j] in candidates,
                                          'proposal_matches_input': selected == draft[j + 1]}
            _S.write(row)
        return out
    mod.RejectionSampler._verify = verify

# ------------------------------------------------------------------------------------------------ install
TARGETS = {
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": patch_proposal_census,
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler": patch_rejection_census,
    "vllm.v1.worker.gpu.cudagraph_utils": patch_cudagraph_utils,
    "vllm.v1.worker.gpu.model_runner": patch_model_runner,
    "vllm.model_executor.layers.quantization.exl3": patch_exl3,
}
_done = set()


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS or name in _done:
            return None
        _done.add(name)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _gate_once()            # first patched import: the launcher chain has finished by now
            TARGETS[name](module)   # let a failure surface: a half-patched runner must not serve
        loader.exec_module = exec_module
        return spec


_gated = False


def _gate_once():
    """Hash gate, run at the first patched import (i.e. inside the vllm serve processes, after the launcher chain has
    finished patching the writable layer). Refuses loudly: a candidate that silently serves the parent arm is worse
    than no boot."""
    global _gated
    if _gated:
        return
    _gated = True
    ok, report = hash_gate()
    _S.receipt.update({"hash_gate": report, "hash_gate_ok": ok, "pid": os.getpid(), "hostname": _S.hostname, "rank": _S.rank})
    if not ok:
        _S.receipt["B5_PREFIX_VERIFY_ACTIVE"] = False
        _S.write_receipt()
        msg = "B5_PREFIX_VERIFY_REFUSE hash gate failed: %s" % [k for k, v in report.items() if not v["match"]]
        _log(msg)
        print(msg, flush=True)
        raise SystemExit(9)
    _log("hash gate ok (%d files)" % len(report))


def install():
    if not _ENABLED:
        return
    _S.receipt = {"pid": os.getpid(), "hostname": _S.hostname, "rank": _S.rank}
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _Finder())
    for name, fn in TARGETS.items():
        m = sys.modules.get(name)
        if m is not None and name not in _done:
            _done.add(name)
            _gate_once()
            fn(m)
    atexit.register(_S.write_receipt)
    _log("installed; hash gate runs at the first patched import, activation after graph capture")


install()
