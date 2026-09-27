"""JSpark3 v1.6 uniform-batch adaptive DFlash verification policy.

The DFlash2 drafter always produces its sealed seven-token continuation.  This
module changes only how many of those tokens the target verifies.  Width three
reuses B5's measured M=4 target shape; width seven is the v1.5 path.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


ENV = "GLM53_ADAPTIVE_K"
EXPECT_ENV = "JSPARK3_V16_ADAPTIVE_K_EXPECT"
EPOCH_ENV = "GLM53_ADAPTIVE_K_EPOCH_FILE"
EPOCH_DEFAULT = "/evidence/adaptive-k-epoch.json"
EPOCH_KEYS = frozenset(("epoch", "mode", "fault_accept_delta"))

FULL_K = 7
NARROW_K = 3
ALPHA = 0.25
STARTUP_FULL_STEPS = 8
PROBE_INTERVAL = 32
MAX_REQUESTS = 8
GAIN_HURDLE = 0.05
HIGH_COMMITTED = 4.5


def _finite_positive(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if not 0.0 < value < float("inf"):
        raise RuntimeError(f"{name} must be a positive finite number")
    return value


class AdaptiveKPolicy:
    """CPU-only, request-local evidence with one safe width per batch."""

    def __init__(self, *, epoch_path: str | None = None) -> None:
        boot = os.environ.get(ENV)
        if boot != "ema":
            raise RuntimeError(f"installed adaptive-k overlay requires {ENV}=ema")
        expected = os.environ.get(EXPECT_ENV)
        if expected is not None and expected != boot:
            raise RuntimeError(
                f"adaptive-k identity mismatch: expected {expected!r}, got {boot!r}"
            )
        self.mode = "ema"
        self.epoch_path = Path(
            epoch_path if epoch_path is not None else os.environ.get(EPOCH_ENV, EPOCH_DEFAULT)
        )
        if not self.epoch_path.is_absolute():
            raise RuntimeError(f"{EPOCH_ENV} must be an absolute path")
        self.t4_ms = _finite_positive("B5_T4_SEED_MS", "74.30")
        self.t7_ms = _finite_positive("B5_T7_SEED_MS", "92.53")
        self.fault_accept_delta = 0
        self.epoch = "boot"
        self._epoch_sig: tuple[int, int, int] | None = None
        self.states: dict[str, dict[str, float | int]] = {}
        self.stats = {"full_batches": 0, "narrow_batches": 0, "epoch_changes": 0}
        self.epoch_stats = {"full_batches": 0, "narrow_batches": 0}
        self._mechanism("boot")

    def _mechanism(self, event: str, query_width: int | None = None) -> None:
        payload = {
            "epoch": self.epoch,
            "event": event,
            "fault_accept_delta": self.fault_accept_delta,
            "full_batches": self.epoch_stats["full_batches"],
            "mode": self.mode,
            "narrow_batches": self.epoch_stats["narrow_batches"],
            "rank": os.environ.get("NODE_RANK", "?"),
        }
        if query_width is not None:
            payload["query_width"] = query_width
        print(
            "[jspark3-v16:adaptive-k-mechanism] "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _record_width(self, width: int) -> int:
        key = "narrow_batches" if width == NARROW_K else "full_batches"
        self.stats[key] += 1
        self.epoch_stats[key] += 1
        total = self.epoch_stats["full_batches"] + self.epoch_stats["narrow_batches"]
        if width == NARROW_K and self.epoch_stats["narrow_batches"] == 1:
            self._mechanism("first-narrow", NARROW_K + 1)
        elif total % 32 == 0:
            self._mechanism("summary", width + 1)
        return width

    @staticmethod
    def _ema(old: float | None, value: float) -> float:
        return value if old is None else ALPHA * value + (1.0 - ALPHA) * old

    def _signature(self) -> tuple[int, int, int] | None:
        try:
            info = self.epoch_path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("adaptive-k epoch path must be a regular non-symlink file")
        return (info.st_ino, info.st_mtime_ns, info.st_size)

    def poll_epoch(self) -> None:
        """Apply a complete epoch document at the next scheduler boundary."""

        signature = self._signature()
        if signature == self._epoch_sig:
            return
        if signature is None:
            self._epoch_sig = None
            return
        try:
            value = json.loads(self.epoch_path.read_text(encoding="utf-8"))
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
        self._mechanism("epoch-end")
        self.mode = mode
        self.epoch = epoch
        self.fault_accept_delta = fault
        self.states.clear()
        self.stats["epoch_changes"] += 1
        self.epoch_stats = {"full_batches": 0, "narrow_batches": 0}
        self._epoch_sig = signature
        self._mechanism("epoch")

    def observe(self, request_id: str, draft_tokens: int, accepted_drafts: int) -> None:
        """Observe only a non-stale, non-first-decode completed verification."""

        if self.mode != "ema" or draft_tokens not in (NARROW_K, FULL_K):
            return
        accepted = max(0, min(draft_tokens, accepted_drafts + self.fault_accept_delta))
        state = self.states.setdefault(
            request_id,
            {"full": None, "prefix": None, "full_steps": 0, "since_full": 0},
        )
        state["prefix"] = self._ema(state["prefix"], float(min(accepted, NARROW_K)))
        if draft_tokens == FULL_K:
            state["full"] = self._ema(state["full"], float(accepted))
            state["full_steps"] = int(state["full_steps"]) + 1
            state["since_full"] = 0
        else:
            state["since_full"] = int(state["since_full"]) + 1

    def request_width(self, request_id: str) -> int:
        state = self.states.get(request_id)
        if state is None or int(state["full_steps"]) < STARTUP_FULL_STEPS:
            return FULL_K
        if int(state["since_full"]) >= PROBE_INTERVAL:
            return FULL_K
        full = float(state["full"])
        prefix = float(state["prefix"])
        gain = ((1.0 + prefix) / self.t4_ms) / ((1.0 + full) / self.t7_ms) - 1.0
        if 1.0 + full < HIGH_COMMITTED and gain >= GAIN_HURDLE:
            return NARROW_K
        return FULL_K

    @staticmethod
    def _is_structured(scheduler: object, request: object) -> bool:
        manager = getattr(scheduler, "structured_output_manager", None)
        return bool(getattr(request, "use_structured_output", False) or
                    (manager is not None and manager.should_advance(request)))

    def _eligible(
        self,
        scheduler: object,
        request_id: str,
        scheduled_tokens: int,
        spec_tokens: object,
    ) -> bool:
        request = scheduler.requests.get(request_id)
        return bool(
            request is not None
            and not request.is_finished()
            and not getattr(request, "is_prefill_chunk", False)
            and getattr(request, "num_output_tokens", 0) > 0
            and isinstance(spec_tokens, list)
            and len(spec_tokens) == FULL_K
            and scheduled_tokens == 1 + len(spec_tokens)
            and not self._is_structured(scheduler, request)
        )

    def schedule_width(
        self,
        scheduler: object,
        default_width: int,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_tokens: dict[str, list[int]],
    ) -> int:
        """Choose the current batch width; any unsafe/full request pins it."""

        self.poll_epoch()
        live = set(scheduler.requests)
        self.states = {key: value for key, value in self.states.items() if key in live}
        if (self.mode != "ema" or default_width != FULL_K
                or not 2 <= len(num_scheduled_tokens) <= MAX_REQUESTS):
            return self._record_width(default_width)
        widths = []
        for request_id, count in num_scheduled_tokens.items():
            spec = scheduled_spec_tokens.get(request_id)
            if not self._eligible(scheduler, request_id, count, spec):
                return self._record_width(FULL_K)
            widths.append(self.request_width(request_id))
        # Sparse MLA/KDA FULL graphs require a uniform query length.  Choosing
        # max (rather than Mia's min) prevents a low-accept prose request from
        # narrowing a high-accept/code request sharing the batch.
        width = max(widths, default=FULL_K)
        return self._record_width(width)

    def prepare_batch(self, scheduler, counts, drafts, new_requests, resumed_requests):
        """Trim only after allocation, before cached data and accounting.

        Both sync draft lists and async placeholders stay seven-wide until this
        point. Allocate at full width, then verify a prefix of that reservation.
        This cannot increase a KV allocation or change the drafter's page layout.
        Never mutate a shared async placeholder list in place.
        """
        excluded = {r.request_id for r in (*new_requests, *resumed_requests)}
        for request_id in excluded:
            self.states.pop(request_id, None)
        width = self.schedule_width(scheduler, scheduler.num_spec_tokens, counts, drafts)
        if width == NARROW_K:
            for request_id in counts:
                counts[request_id] = NARROW_K + 1
                drafts[request_id] = drafts[request_id][:NARROW_K]

    def bind_output(self, scheduler, output):
        """Snapshot first-step/epoch eligibility before async output can land."""
        new_ids = {r.req_id for r in output.scheduled_new_reqs}
        eligible = {
            rid for rid, count in output.num_scheduled_tokens.items()
            if rid not in new_ids
            and (request := scheduler.requests.get(rid)) is not None
            and request.num_output_tokens > 0
            and not getattr(request, "is_prefill_chunk", False)
            and count == 1 + len(output.scheduled_spec_decode_tokens.get(rid, ()))
        } if len(output.num_scheduled_tokens) >= 2 else set()
        # Scheduler-local metadata: workers do not use this attribute.
        output._adaptive_k_observation = (self.stats["epoch_changes"], eligible)

    def observe_output(self, output, request_id, draft_tokens, accepted_drafts):
        revision, eligible = output._adaptive_k_observation
        if revision == self.stats["epoch_changes"] and request_id in eligible:
            self.observe(request_id, draft_tokens, accepted_drafts)


POLICY = AdaptiveKPolicy()
