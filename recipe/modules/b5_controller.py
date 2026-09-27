from __future__ import annotations
"""B5 request-local width controller. Pure Python, deterministic, hand-executable.

Observed inputs per request (never the prompt text):
  a      : tokens committed by a step (accepted drafts + 1 bonus), known one step late
  width  : draft width the step ran at (7 = full, 8 query positions; 3 = narrow, 4 positions)
  t_ms   : measured wall time of that step (CUDA events, resolved one step late)
Batch inputs: num_reqs, has_prefill.

Phases: STARTUP -> (FULL | NARROW); NARROW runs PROBE_EVERY narrow steps then one full-width
probe, then re-evaluates. PROBE_EVERY is 32: the reference-trace knob study showed the probe cadence is the
main controller tax (8 gave back half the controller gain on the frozen screen) while code protection comes from the
startup evaluation (R2), not from probes. Decisions are taken only at EVALUATION POINTS:
  E1  the first step after STARTUP_FULL_STEPS full-width steps,
  E2  the step after a probe has landed,
  E3  while FULL: every REEVAL_FULL full-width steps.
Between evaluation points the phase is held (no per-step flip-flop).

Rule at an evaluation point (first match wins):
  R0 fallback     num_reqs >= 2 or has_prefill or no timing for width 7 yet           -> 7
  R2 high-accept  r7 >= HIGH_ACCEPT                                                   -> FULL
  R4 enter/stay   gain = (r3/T4) / (r7/T7) - 1 >= HURDLE_ENTER (0.05) when in FULL,
                  or gain >= HURDLE_STAY (0.0) when already NARROW                     -> NARROW
  R5 otherwise                                                                        -> FULL
Outside evaluation points: STARTUP -> 7; FULL -> 7; NARROW -> 3 until PROBE_EVERY narrow steps,
then one probe at 7 (R3).

Estimates (all request-local):
  r7 = mean a over the last W_FULL full-width steps of this request (startup + probes)
  r3 = mean a over the last W_NARROW narrow steps once at least MIN_NARROW have run; before that
       the per-step truncation mean(min(a, 4)) over the same full-width steps (exact for each of
       those steps: the narrow path verifies a prefix of the SAME drafts)
  T7 = EMA of measured full-width step ms; T4 = EMA of measured narrow step ms, seeded from the
       boot-time graph calibration until the request has narrow steps of its own.
HIGH_ACCEPT: narrowing can never pay when r7 > 4 * T7/T4; at the pessimistic T4 = 79 ms that is
4.64, so 4.5 is a conservative "cannot pay" guard (also the empty gap between the reference code
minimum 4.88 and prose maximum 4.00 in qpanel/ADAPTIVE-CLASSIFIER.md, an observation not a rule).
"""
from collections import deque
from dataclasses import dataclass, field
import json
import os
import stat
from pathlib import Path
from typing import Optional

STARTUP_FULL_STEPS = 8
PROBE_EVERY = 32   # FROZEN 2026-09-05 from the reference-trace knob study (was 8 in H5/P5); see CONTROLLER-RULE.md
REEVAL_FULL = 8
HURDLE_ENTER = 0.05
HURDLE_STAY = 0.0
HIGH_ACCEPT = 4.5
W_FULL = 16
W_NARROW = 32
MIN_NARROW = 8
EMA_ALPHA = 0.25

FULL, NARROW = 7, 3
EPOCH_KEYS = frozenset(("epoch", "mode", "fault_accept_delta"))
EPOCH_DEFAULT = "/evidence/adaptive-k-epoch.json"


class AdaptiveEpochMode:
    """Mirror the scheduler policy's atomic epoch semantics for B5 feedback."""

    def __init__(self, epoch_path: Optional[str] = None):
        self.boot_mode = os.environ.get("GLM53_ADAPTIVE_K", "off")
        if self.boot_mode not in ("off", "ema"):
            raise RuntimeError("GLM53_ADAPTIVE_K must be off or ema")
        self.mode = self.boot_mode
        self.epoch = "boot"
        self.fault_accept_delta = 0
        self.path = Path(
            epoch_path
            if epoch_path is not None
            else os.environ.get("GLM53_ADAPTIVE_K_EPOCH_FILE", EPOCH_DEFAULT)
        )
        if not self.path.is_absolute():
            raise RuntimeError("GLM53_ADAPTIVE_K_EPOCH_FILE must be an absolute path")
        self.signature: Optional[tuple[int, int, int]] = None

    def poll(self) -> str:
        if self.boot_mode == "off":
            return "off"
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            signature = None
        else:
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("adaptive-k epoch path must be a regular non-symlink file")
            signature = (info.st_ino, info.st_mtime_ns, info.st_size)
        if signature == self.signature:
            return self.mode
        if signature is None:
            self.signature = None
            return self.mode
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid adaptive-k epoch file: {exc}") from exc
        if not isinstance(value, dict) or set(value) != EPOCH_KEYS:
            raise RuntimeError(
                "adaptive-k epoch must contain exactly epoch, mode, fault_accept_delta"
            )
        epoch = value["epoch"]
        mode = value["mode"]
        fault = value["fault_accept_delta"]
        if not isinstance(epoch, (str, int)) or isinstance(epoch, bool):
            raise RuntimeError("adaptive-k epoch id must be a string or integer")
        if mode not in ("off", "ema"):
            raise RuntimeError("adaptive-k epoch mode must be off or ema")
        if type(fault) is not int or not -7 <= fault <= 7:
            raise RuntimeError("adaptive-k fault_accept_delta must be an integer in [-7, 7]")
        self.mode = mode
        self.epoch = epoch
        self.fault_accept_delta = fault
        self.signature = signature
        return self.mode


@dataclass
class ReqCtl:
    full_a: deque = field(default_factory=lambda: deque(maxlen=W_FULL))
    narrow_a: deque = field(default_factory=lambda: deque(maxlen=W_NARROW))
    n_full: int = 0
    n_narrow: int = 0
    n_narrow_since_probe: int = 0
    full_since_eval: int = 0
    probe_pending: bool = False
    accept_record: bool = True
    t7: Optional[float] = None
    t4: Optional[float] = None
    phase: str = "STARTUP"
    decisions: list = field(default_factory=list)

    def record(self, width: int, a: int, t_ms: Optional[float] = None) -> None:
        if width == FULL:
            self.full_a.append(a); self.n_full += 1; self.full_since_eval += 1
            if t_ms is not None:
                self.t7 = t_ms if self.t7 is None else (1 - EMA_ALPHA) * self.t7 + EMA_ALPHA * t_ms
            if self.phase == "PROBE":
                self.probe_pending = True
        else:
            self.narrow_a.append(a); self.n_narrow += 1; self.n_narrow_since_probe += 1
            if t_ms is not None:
                self.t4 = t_ms if self.t4 is None else (1 - EMA_ALPHA) * self.t4 + EMA_ALPHA * t_ms

    def estimates(self, t4_seed: float):
        r7 = sum(self.full_a) / len(self.full_a)
        if self.n_narrow >= MIN_NARROW:
            r3 = sum(self.narrow_a) / len(self.narrow_a); src = "measured"
        else:
            r3 = sum(min(a, 4) for a in self.full_a) / len(self.full_a); src = "truncated-full"
        T4 = self.t4 if self.t4 is not None else t4_seed
        return r7, r3, src, self.t7, T4

    def evaluate(self, t4_seed: float) -> str:
        """Set self.phase at an evaluation point; return the rule fired."""
        r7, r3, src, T7, T4 = self.estimates(t4_seed)
        self.full_since_eval = 0
        if r7 >= HIGH_ACCEPT:
            self.phase = "FULL"
            return f"R2-high-accept(r7={r7:.2f})"
        gain = (r3 / T4) / (r7 / T7) - 1.0
        hurdle = HURDLE_STAY if self.phase in ("NARROW", "PROBE") else HURDLE_ENTER
        detail = f"gain={gain:+.3f},r7={r7:.2f},r3={r3:.2f}[{src}],T7={T7:.1f},T4={T4:.1f}"
        if gain >= hurdle:
            self.phase = "NARROW"; self.n_narrow_since_probe = 0
            return f"R4-narrow({detail})"
        self.phase = "FULL"
        return f"R5-full({detail})"

    def decide(self, num_reqs: int, has_prefill: bool, t4_seed: float, adaptive_mode: str = "off"):
        if num_reqs >= 2 or has_prefill:
            # v1.6 adaptive-k owns these batches. Its scheduler output may
            # already contain three drafts, while B5's R0 decision remains
            # seven; do not mislabel that feedback as a full-width c1 sample.
            self.accept_record = adaptive_mode != "ema"
            return FULL, "R0-fallback"
        self.accept_record = True
        if self.n_full < STARTUP_FULL_STEPS:
            self.phase = "STARTUP"
            return FULL, "R1-startup"
        if self.t7 is None:
            return FULL, "R0-no-timing"
        # evaluation points
        if self.phase == "STARTUP":
            rule = "E1:" + self.evaluate(t4_seed)
        elif self.probe_pending:
            self.probe_pending = False
            rule = "E2:" + self.evaluate(t4_seed)
        elif self.phase == "FULL" and self.full_since_eval >= REEVAL_FULL:
            rule = "E3:" + self.evaluate(t4_seed)
        else:
            rule = "hold"
        if self.phase == "NARROW":
            if self.n_narrow_since_probe >= PROBE_EVERY:
                self.phase = "PROBE"
                return FULL, rule + ";R3-probe"
            return NARROW, rule + ";narrow"
        if self.phase == "PROBE":   # probe issued, result not yet landed (one-step lag)
            return FULL, rule + ";probe-wait"
        return FULL, rule + ";full"


class B5Controller:
    def __init__(self, t4_seed_ms: float, t7_seed_ms: Optional[float] = None,
                 force_width: Optional[int] = None, epoch_path: Optional[str] = None):
        self.reqs: dict = {}
        self.t4_seed = t4_seed_ms
        self.t7_seed = t7_seed_ms
        self.force_width = force_width
        self.epoch_mode = AdaptiveEpochMode(epoch_path)
        self.effective_mode = self.epoch_mode.mode

    def add(self, req_id): self.reqs[req_id] = ReqCtl()
    def remove(self, req_id): self.reqs.pop(req_id, None)

    def decide(self, req_id, num_reqs, has_prefill):
        self.effective_mode = self.epoch_mode.poll()
        if self.force_width is not None:
            return self.force_width, "FORCED"
        r = self.reqs.get(req_id)
        if r is None:
            return FULL, "R0-unknown"
        if r.t7 is None and self.t7_seed is not None:
            r.t7 = self.t7_seed
        w, rule = r.decide(num_reqs, has_prefill, self.t4_seed, self.effective_mode)
        r.decisions.append((w, rule))
        return w, rule

    def record(self, req_id, width, a, t_ms=None):
        r = self.reqs.get(req_id)
        if r is not None and r.accept_record:
            r.record(width, a, t_ms)
