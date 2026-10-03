"""GLM-5.3-Flash on two or three NCCL ranks; all sample by one keyed rule from the same gathered candidates, so no broadcast."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
DFLASH_POLICY = "fc5:0.3"             # DFlash2 drafts every round: up to 5 while their probability product holds 0.3
EXL3_AUTO = DFLASH_POLICY             # what auto runs on an EXL3 checkpoint with the draft model
GRAPH_ROWS = (1, 2, 3, 4, 5, 6)       # verify windows captured as CUDA graphs
MAX_ROWS = 8                          # the widest verify window (a pending token and up to 7 drafts)
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)


def encode_policy(spec: str) -> list[int]:
    """Encode policy kind, maximum drafts, and two parameters in millionths as four integers, with 10 added to kind for DFlash2."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], cN:P, fpN[:raw|solo|carry], fqN:P, or one of "
                     f"these after f (N from 1 to {MAX_ROWS - 1})")
    try:
        if spec.startswith("fp"):
            parts = spec[2:].split(":")
            most = int(parts[0])
            modes = {"raw": 1, "solo": 2, "carry": 4}
            if not 0 < most < MAX_ROWS or len(parts) > 2 or (len(parts) == 2 and parts[1] not in modes):
                raise bad
            return [16, most, modes.get(parts[1], 0) if len(parts) == 2 else 0, 0]
        if spec.startswith("fq"):
            most_text, conf = spec[2:].split(":")
            most = int(most_text)
            threshold = float(conf)
            if not 0 < most < MAX_ROWS or not 0 <= threshold <= 1:
                raise bad
            return [17, most, int(round(threshold * 1e6)), 0]
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4 if len(parts) == 1 else 5, explore, every, int(round(margin * 1e6))]
        if spec.startswith("f"):
            code = encode_policy(spec[1:])
            return [code[0] + 10] + code[1:] if code[0] else code
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def decode_policy(code: list[int]):
    """Decode a serial, MTP, or automatic policy, retaining exploration, sampling, and margin settings."""

    from .decode import DepthPolicy

    kind, most, a, b = code
    if kind in (16, 17):
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True,
                           confidence=0.0 if kind == 16 else a / 1e6)
    if kind in (4, 5):
        return ("auto", most, a, b / 1e6, kind == 5)
    kind %= 10
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def checkpoint_after(pos: int) -> int:
    """Where a long prompt's prefill also writes its state to disk (``TF_GLM_DISK_DIR``), so a prompt that leaves it
    part way (a client that trimmed or dropped earlier text, an agent sharing only the tool list) resumes from the
    last one before it: every 4,096 tokens up to 32,768, then every 16,384."""

    return (pos // 4096 + 1) * 4096 if pos < 32768 else (pos // 16384 + 1) * 16384


def configured_identical_cache(parallel: int) -> bool:
    value = os.environ.get("TF_GLM_CACHE_LAST_TOKEN", "0")
    if value not in ("0", "1"):
        raise ValueError("TF_GLM_CACHE_LAST_TOKEN must be 0 or 1")
    if value == "1" and parallel != 1:
        raise ValueError("TF_GLM_CACHE_LAST_TOKEN is supported only with one GLM stream")
    return value == "1"


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def configured_prefill_rows(rows: int | None, *, context: int | None, explicit: bool | None,
                            parallel: int) -> int:
    """An explicit constructor value wins; the opt-in environment trial keeps an explicit context intact."""
    from tensorfold.cuda.geometry import PREFILL_ROWS

    if rows is not None:
        return int(rows)
    value = os.environ.get("TF_GLM_PREFILL_ROWS")
    if value is None:
        return PREFILL_ROWS
    if value not in ("2048", "4096", "8192"):
        raise ValueError("TF_GLM_PREFILL_ROWS must be 2048, 4096, or 8192")
    if context is None or context <= 0 or explicit is False:
        raise ValueError("TF_GLM_PREFILL_ROWS needs an explicit positive --context; keep the baseline window, "
                         "and free memory or revert the row knob if admission refuses")
    return int(value)


def configured_prefill_rows_idle(rows: int, *, context: int | None, explicit: bool | None) -> int:
    """Reserve the large workspace at startup; rank zero picks each chunk's active row budget."""
    value = os.environ.get("TF_GLM_PREFILL_ROWS_IDLE", "0")
    if value == "0":
        return 0
    if value != "8192":
        raise ValueError("TF_GLM_PREFILL_ROWS_IDLE must be 0 or 8192")
    from . import prefill_options as p1

    if rows != 4096 or not p1.ATTENTION_TILES:
        raise ValueError("TF_GLM_PREFILL_ROWS_IDLE needs TF_GLM_PREFILL_ROWS=4096 and TF_GLM_A2_ATTENTION_TILES=1")
    if context is None or context <= 0 or explicit is False:
        raise ValueError("TF_GLM_PREFILL_ROWS_IDLE needs an explicit positive --context")
    return 8192


def fit_shared_pool(plan: dict, context: int, requested: int, kept_bytes: int, target, draft, gather):
    """Choose one rank-safe pool size while retaining the startup memory reserve and snapshot budget."""
    def workspace(tokens):
        slots = max(target.minimum_slots, tokens + target.reserve)
        return slots, target.bytes_at(slots) + (draft.bytes_at(slots) if draft is not None else 0)

    def total(tokens):
        return plan["weight_bytes_estimate"] + max(plan["loading_bytes_estimate"], workspace(tokens)[1])

    ceiling = plan["budget_bytes"] - kept_bytes
    low, high = context, requested
    while low < high:
        mid = (low + high + 1) // 2
        if total(mid) <= ceiling:
            low = mid
        else:
            high = mid - 1
    offers = gather([requested, low])
    if any(row[0] != requested for row in offers):
        raise ValueError("TF_GLM_POOL_TOKENS must match on every rank")
    chosen = min(row[1] for row in offers)
    slots, work = workspace(chosen)
    return chosen, slots, work, total(chosen)


class GlmEngine:
    """GLM-5.3-Flash on ``world`` ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    disk = None          # kept prompts on disk (TF_GLM_DISK_DIR)
    tower = None         # the vision tower (rank 0)
    cache_last_token = False

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int | None = None, context_explicit: bool | None = None, serial_only: bool = False, comm=None,
                 prefill_rows: int | None = None, parallel: int = 1, world: int = 2, drafter_bits: int = 4) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between the machines (tests)."""

        import torch

        from tensorfold.cuda.comm import NCCL
        from .decode import Engine
        from .weights import Config, load
        from .split import padded_config, rank_files, rule
        from tensorfold.cuda.capacity import Geometry, admit
        from tensorfold.cuda.geometry import draft_geometry, mla_geometry, split_weights, with_fixed
        from . import vision

        self.model_dir = Path(model_dir)
        encode_policy(policy)                           # a bad default fails here, not in the first request
        if not 1 <= parallel <= 8:
            raise ValueError("GLM --parallel must be between 1 and 8")
        check_prefill_offsets = "TF_GLM_PREFILL_ROWS" in os.environ
        prefill_rows = configured_prefill_rows(prefill_rows, context=context, explicit=context_explicit,
                                               parallel=parallel)
        self.prefill_rows_busy = prefill_rows
        self.prefill_rows_idle = configured_prefill_rows_idle(prefill_rows, context=context,
                                                              explicit=context_explicit)
        prefill_rows = max(prefill_rows, self.prefill_rows_idle)
        check_prefill_offsets = check_prefill_offsets or bool(self.prefill_rows_idle)
        from . import LATENT

        if parallel > 1 and not LATENT:
            raise ValueError("several GLM streams need the latent cache (TF_GLM_LATENT=1)")
        from .session_cache import SessionConfig
        from .reply_prefill import ReplyConfig

        session_config = SessionConfig.from_env(parallel)
        reply_config = ReplyConfig.from_env(parallel)
        from .idle_bell import FollowerDoorbell, configured as configured_doorbell

        doorbell_enabled = configured_doorbell()
        self.reply_prefill, self.reply_prefill_rows = reply_config.enabled, reply_config.rows
        from .copy_drafts import configured_copy_rows
        from . import prefill_options as p1

        verify_rows = configured_copy_rows()
        from .reply_reservations import ReplyConfig as ReservationConfig
        from . import draft_graphs

        self.reply_config = ReservationConfig.from_env(parallel, session_config)
        self.draft_slot_graphs = draft_graphs.configured(parallel, drafter)
        if p1.COFILL_BIG and (parallel == 1 or not p1.ATTENTION_TILES):
            raise ValueError("TF_GLM_COFILL_BIG requires --parallel >=2 and TF_GLM_A2_ATTENTION_TILES=1")
        pool_padding = (parallel - 1) * 64
        geometries = {}

        def target_plan(text):
            g = with_fixed(mla_geometry(padded_config(text, world), world, verify_rows,
                                        minimum_slots=DENSE_CAPACITY, latent=LATENT,
                                        sequences=max(1, parallel), pooled=parallel > 1,
                                        decode_rows=verify_rows,
                                        prefill_rows=prefill_rows,
                                        check_prefill_offsets=check_prefill_offsets),
                           seeing + session_config.reserve + reply_config.reserve + self.reply_config.reserve
                           + ((parallel - 1) * draft_graphs.SLOT_BUDGET if self.draft_slot_graphs else 0))
            geometries["target"] = g
            return g

        def draft_plan(text):
            g = draft_geometry(self._draft_text(text), world, MAX_ROWS, bounded=True, streams=parallel)
            if parallel == 1:
                geometries["draft"] = g
                return g
            result = Geometry(lambda slots: g.bytes_at(slots + pool_padding), g.reserve, g.minimum_slots)
            geometries["draft"] = result
            return result

        if check_prefill_offsets and LATENT:
            # Refuse an unsafe local shape before device selection or distributed rendezvous.
            raw = json.loads((Path(model_dir) / "config.json").read_text())
            text = dict(raw.get("text_config") or raw)
            mla_geometry(padded_config(text, world), world, MAX_ROWS, minimum_slots=DENSE_CAPACITY,
                         latent=True, prefill_rows=prefill_rows, check_prefill_offsets=True)
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank, self.world = rank, world
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        self.quality_score_enabled = os.environ.get("TF_GLM_QUALITY_SCORE", "0") == "1"
        self.cache_last_token = configured_identical_cache(parallel)
        self.own_comm = comm is None                    # NCCL between the machines (else a test's stand-in)
        self.comm = comm if comm is not None else NCCL(rank, world, master, port)
        store = getattr(self.comm, "store", None)
        self.follower_doorbell = (FollowerDoorbell(store, rank, world)
                                  if doorbell_enabled and world > 1 and store is not None else None)
        self.comm.barrier()
        cfg = Config.read(model_dir)
        # Without --context the window stays dense, attending every key without indexer work.
        explicit = context is not None if context_explicit is None else bool(context_explicit)

        # rank 0 encodes images (its folder holds the vision tower): the tower and its largest image's scratch
        sees = rank == 0 and os.environ.get("TF_GLM_VISION", "1") != "0" and vision.available(model_dir)
        seeing = vision.tower_bytes(model_dir) + vision.WORKSPACE if sees else 0
        self.capacity_plan = self._admit(admit, model_dir, context if explicit else cfg.dense_limit, explicit, torch,
                                   target_plan,
                                   split_weights(rule, world, self._pad(model_dir, world)), rank=rank, world=world,
                                   gather=self._gather_ints, files=rank_files(model_dir, rank, world) or None,
                                   draft_dir=drafter,
                                   draft_geometry=draft_plan)
        self.limit = self.capacity_plan["context_window"]
        capacity = self.capacity_plan["cache_slots"]
        self.pool_limit = self.limit
        wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * 2 ** 30)
        pool_text = os.environ.get("TF_GLM_POOL_TOKENS")
        try:
            pool_request = int(pool_text) if pool_text is not None and pool_text.isdecimal() else 0
        except ValueError:
            pool_request = 0
        pool_valid = pool_text is None or (parallel >= 2 and self.limit <= pool_request <= parallel * self.limit)
        pool_settings = self._gather_ints([int(pool_text is not None), int(pool_valid), pool_request])
        if any(not row[1] for row in pool_settings):
            raise ValueError("TF_GLM_POOL_TOKENS needs an integer from --context through parallel times --context")
        if any(row[0] != pool_settings[0][0] or row[2] != pool_settings[0][2] for row in pool_settings):
            raise ValueError("TF_GLM_POOL_TOKENS must match on every rank")
        if pool_text is not None:
            # The ordinary capacity budget already leaves at least a tenth of host RAM
            # available. Keep the requested snapshot store inside that same budget.
            self.pool_limit, capacity, work, total = fit_shared_pool(
                self.capacity_plan, self.limit, pool_request, wanted,
                geometries["target"], geometries.get("draft"), self._gather_ints)
            self.capacity_plan["cache_slots"] = capacity
            self.capacity_plan["cache_workspace_bytes_estimate"] = work
            self.capacity_plan["serving_peak_bytes_estimate"] = self.capacity_plan["weight_bytes_estimate"] + work
            self.capacity_plan["total_bytes_estimate"] = total
            self.capacity_plan["full_mapped_working_set_bytes_estimate"] = (
                self.capacity_plan["total_bytes_estimate"] + self.capacity_plan["mapped_table_bytes"])
            if rank == 0:
                print(f"[tensorfold] GLM shared token pool: {self.pool_limit:,} usable tokens "
                      f"(requested {pool_request:,}); per-request context {self.limit:,}", flush=True)
        long_context = self.limit > cfg.dense_limit
        # both ranks must run the same calls: refuse to start when they were given different settings
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), int(LATENT),
                prefill_rows, int(parallel), int(drafter_bits) if drafter is not None else 0,
                int(self.quality_score_enabled), verify_rows, int(p1.COFILL_BIG), *session_config.agreement(),
                int(reply_config.enabled), reply_config.rows, *self.reply_config.agreement(),
                p1.PREFILL_DECODE_QUANTUM, int(self.draft_slot_graphs),
                int(doorbell_enabled), int(self.follower_doorbell is not None),
                int(p1.EXPERT_WHOLE_PASS), int(p1.EXPERT_PREFILL128), int(p1.EXPERT_PREFILL64),
                self.prefill_rows_idle, self.prefill_rows_busy, int(p1.EXPRESS), p1.EXPRESS_ATOMIC_ROWS,
                int(p1.EXPRESS_COFILL), int(p1.EXACT_REPLAY)]
        # other conversations' kept prompts get what the window leaves, at most TF_GLM_CACHE_GIB, the same on both ranks
        plan = self.capacity_plan
        spare = max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
        both = self._gather_ints(mine + [int(self.cache_last_token), spare >> 20])
        if any(row[:-1] != both[0][:-1] for row in both):
            raise RuntimeError("the ranks were started with different settings (draft model, context, drafts, "
                               "TF_GLM_LATENT): " + ", ".join(f"rank {r} {row[:-1]}" for r, row in enumerate(both)) +
                               "; pull the draft model on every machine (or pass --drafter none to all) and give all "
                               "the same flags")
        self.cache_bytes = min(row[-1] for row in both) << 20
        plan["kept_bytes"] = self.cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + self.cache_bytes
        if rank == 0 and self.cache_bytes < wanted:
            print(f"[tensorfold] other conversations' prompts are kept in {self.cache_bytes / 2 ** 30:.1f} GiB, what "
                  f"the {self.limit}-token window leaves (TF_GLM_CACHE_GIB asks {wanted / 2 ** 30:.1f})", flush=True)
        w = load(model_dir, rank=rank, world=world)
        w.comm = self.comm
        self.comm.barrier()
        if w.mtp is None and drafter is None and not serial_only:
            raise ValueError("this checkpoint has no MTP head and no DFlash2 draft model was given, so every round "
                             "would decode one token: pull the draft model on both machines (--drafter), or pass "
                             "--no-drafts to both for the serial reference")
        self.w = w
        self.tower = vision.Tower(model_dir) if sees else None
        self.disk = None                 # kept prompts on disk (TF_GLM_DISK_DIR, one stream only; set below)
        self.checkpoint_after = checkpoint_after
        if rank == 0:
            print("[tensorfold] GLM-5.3-Flash image input: " + ("on (vision tower on rank 0)" if sees else
                  "off (" + ("TF_GLM_VISION=0" if os.environ.get("TF_GLM_VISION", "1") == "0" else
                             f"no vision tower in {model_dir}") + ")"), flush=True)
        self.drafter = None
        # Multi-stream serving keeps the C1 engine/graphs and partitions its cache arena.
        self.concurrent = parallel > 1
        self.multi = self.scheduler = None
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=capacity + pool_padding, bits=drafter_bits,
                                   streams=parallel)
            if self.concurrent and self.drafter.block > 64:
                raise ValueError("pooled GLM serving supports DFlash2 blocks of at most 64 rows")
            if rank == 0:
                print(f"[tensorfold] DFlash2 drafter: {'bf16' if drafter_bits == 16 else '4-bit'} linears; "
                      f"default draft policy {self.policy}", flush=True)
        from . import tp3_probe

        self.e = Engine(w, capacity=capacity + pool_padding, max_rows=verify_rows, prefill_rows=prefill_rows,
                        graphs=not tp3_probe.CAPTURE, graph_rows=GRAPH_ROWS,
                        long_context=long_context, taps=self.drafter.tap_layers if self.drafter is not None else ())
        if self.drafter is not None:
            self.drafter.capture()
        self.draft_graph_bank = (draft_graphs.capture(self, parallel) if self.draft_slot_graphs else None)
        self.costs = self._calibrate()
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7): {c['timed']}", flush=True)
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) +
                  f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row); "
                  f"DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        # kept conversations (decode.Snapshot, least recently used first) and the live caches' ids; states and saved rows stay within cache_bytes
        self.cache: list = []
        self.live: list[int] = []
        self.cache_entries = int(os.environ.get("TF_GLM_CACHE_ENTRIES", "64" if session_config.disk else "8"))
        if self.reply_prefill and (self.cache_entries <= 0 or self.cache_bytes <= 0):
            raise ValueError("reply prefill requires an enabled in-memory prompt store")
        # TF_GLM_DISK_DIR: every prompt's rows also go to disk (the rows it adds to its longest written prefix), so a
        # conversation that left the device, or one from before a restart, resumes from there; TF_GLM_DISK_GIB caps it
        self.disk = None
        folder = os.environ.get("TF_GLM_DISK_DIR")
        e1_options = {}
        if p1.EXPERT_WHOLE_PASS:
            e1_options["expert_whole_pass"] = 1
        if self.prefill_rows_idle:
            e1_options.update(prefill_rows_idle=self.prefill_rows_idle, prefill_rows_busy=self.prefill_rows_busy)
        if folder and not session_config.disk:
            from .disk import Store, fingerprint

            options = {"latent": int(LATENT), "prefill_rows": prefill_rows, "long_context": int(long_context),
                       **e1_options}
            if self.cache_last_token:
                options["cache_last_token"] = 1
            stamp = fingerprint(model_dir, rank, world, options)
            self.disk = Store(Path(folder) / f"rank{rank}", float(os.environ.get("TF_GLM_DISK_GIB", "64")) * 2 ** 30,
                              stamp)
            if rank == 0:
                print(f"[tensorfold] kept prompts on disk: {len(self.disk.entries)} in {folder} "
                      f"({self.disk.held() / 2 ** 30:.1f} of {self.disk.budget / 2 ** 30:.0f} GiB)", flush=True)
        from .session_cache import configure

        configure(self, session_config, model_dir, drafter,
                  {"latent": int(LATENT), "prefill_rows": prefill_rows, "long_context": int(long_context),
                   "drafter_bits": drafter_bits, "drafter_window": getattr(self.drafter, "window", 0),
                   "drafter_ring": getattr(self.drafter, "ring", 0),
                   "drafter_block": getattr(self.drafter, "block", 0),
                   "vision": os.environ.get("TF_GLM_VISION", "1"), "max_rows": verify_rows,
                   **e1_options}, slots=parallel)
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler
            from .batched import BatchedDecoder

            self.multi = BatchedDecoder(self, slots=parallel)
            self.scheduler = Scheduler(self.multi, max_streams=parallel,
                                       admit_per_round=self.multi.admit_per_round) if rank == 0 else None
            if rank == 0:
                print(f"[tensorfold] GLM-5.3-Flash: up to {parallel} streams, shared {self.limit}-token "
                      f"reservation budget, {self.cache_bytes / 2**30:.2f} GiB conversation store; "
                      "solitary slot-0 graphs, shared target verification", flush=True)

    def running_requests(self) -> int:
        """Reserved requests, including prefill; queued callers have no reservation."""
        if self.concurrent and getattr(self.multi, "growth", None) is not None:
            return self.multi.live()  # parked responses still own a scheduler slot and an open caller
        return len(self.multi.pool.spans) if self.concurrent else 0

    def _draft_text(self, text: dict) -> dict:
        """The drafter's config with the heads its ranks hold after padding (``dflash2.rank_heads``)."""

        if self.world <= 2:
            return text
        from .dflash2 import rank_heads

        heads, kv = rank_heads(int(text["num_attention_heads"]), int(text["num_key_value_heads"]), self.world)
        return {**text, "num_attention_heads": heads * self.world, "num_key_value_heads": kv * self.world}

    @staticmethod
    def _pad(model_dir: Path, world: int):
        """(padded, real) of each split tensor's dim for the memory estimate (None at two ranks: nothing is padded)."""

        from .split import split_pad

        if world == 2:
            return None
        config = json.loads((Path(model_dir) / "config.json").read_text())
        return lambda name, kind: split_pad(name, kind, config, world)

    def _admit(self, admit, *args, **kwargs) -> dict:
        """``admit``; a refusal (both ranks reach the same one) is printed at once and ends the process: unwinding
        with the NCCL communicator open held a refused start for half an hour before its message appeared."""

        from tensorfold.cuda import comm as nccl

        try:
            return admit(*args, **kwargs)
        except ValueError as exc:
            real = isinstance(nccl.NCCL, type) and isinstance(self.comm, nccl.NCCL)
            if not (self.own_comm and real):         # tests' stand-in communicators unwind as usual
                raise
            print(f"tensorfold: {exc}", flush=True)
            os._exit(1)

    def _calibrate(self) -> dict:
        """Per-piece ms for ``drafter_choice.DrafterChoice``: fastest of interleaved passes, equal on both ranks."""

        import statistics

        import numpy as np

        from .decode import draft, prefill

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        prefill(e, tokens(64), None, mtp=True, drafter=self.drafter)
        hidden = e.pbuf.fnormed[:MAX_ROWS].clone()          # rows for timing the draft steps
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, MAX_ROWS + 1)}
        if self.w.mtp is not None:
            pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
            pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
            pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        if self.drafter is not None:
            d = self.drafter
            taps = e.tap_rows(8, e.pbuf).clone()
            ctx = d.context_end

            def back() -> None:
                if d.context_end != ctx:
                    d.pos_dev.sub_(d.context_end - ctx)
                    d.context_end = ctx

            pieces["block"] = (lambda: d.propose(one[0], 5, None, 0.0), None)
            pieces["taps8"] = (lambda: d.add_taps(taps), back)
        best = {name: float("inf") for name in pieces}
        for turn in range(9):
            for name, (fn, prep) in pieces.items():
                if prep is not None:
                    prep()
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if turn >= 2:
                    best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
            rewind()
            if self.drafter is not None:
                back()
        names = list(best)
        mine = torch.tensor([best[n] for n in names], dtype=torch.float32, device="cuda")
        got = torch.empty((self.world * mine.numel(),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        both = dict(zip(names, got.view(self.world, -1).max(dim=0).values.tolist()))
        e.reset()
        if self.drafter is not None:
            self.drafter.reset()
        rows = list(range(2, MAX_ROWS + 1))
        ys = [both[f"v{r}"] for r in rows]
        slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
        base = statistics.median(y - slope * r for r, y in zip(rows, ys))
        verify = [both["v1"]] + [base + slope * r for r in rows]
        mtp = both.get("m1", 0.0)
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
                "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0), "block": both.get("block", 0.0),
                "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((self.world * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return got.view(self.world, len(values)).tolist()

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((self.world * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _ring(self) -> None:
        if (bell := getattr(self, "follower_doorbell", None)) is not None:
            bell.ring()

    def _await_bell(self) -> None:
        if (bell := getattr(self, "follower_doorbell", None)) is not None:
            bell.wait()

    def _effective(self, code: list[int]) -> list[int]:
        """Resolve auto and MTP policies to the available heads, using EXL3_AUTO for EXL3 with DFlash2 and DFlash2 when MTP is absent."""

        if code[0] == 4 and self.drafter is not None and self.w.cfg.quant == "exl3":
            return encode_policy(EXL3_AUTO)
        if self.w.mtp is None and code[0] in (1, 2, 3, 4, 5):
            return encode_policy(DFLASH_POLICY) if code[0] in (4, 5) else [code[0] + 10] + code[1:]
        return code

    def _drafters(self, code: list[int]) -> tuple[bool, bool, bool]:
        """(auto, MTP drafts, DFlash2 drafts) for a policy code."""

        auto = code[0] in (4, 5)
        dflash = (auto or code[0] // 10 == 1) and self.drafter is not None
        return auto, auto or not dflash, dflash

    def _resume(self, prompt: list[int], code: list[int]):
        """The longest snapshot of a strict prefix of ``prompt`` whose draft caches fit the request's drafters: kept
        on the device, else written to disk (a stand-in ``Store.load`` fills)."""

        _, mtp, dflash = self._drafters(code)
        best = None
        for snap in self.cache:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0)
            if fits and len(snap.ids) < len(prompt) and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        if self.disk is not None and not dflash:
            entry = self.disk.resume(prompt, mtp=mtp)
            if entry is not None and (best is None or len(entry.ids) > len(best.ids)):
                return entry.stub()
        return best

    def _drop(self, snap) -> None:
        """Forget a kept snapshot and free its saved rows now, even while a caller still holds the object."""
        snap.rows, snap.nbytes = None, 0
        self.cache.remove(snap)

    def _remember(self, snap) -> None:
        for c in [c for c in self.cache if c.ids == snap.ids]:
            self._drop(c)
        self.cache.append(snap)
        dropped = False
        while len(self.cache) > 1 and (len(self.cache) > self.cache_entries or self._held_bytes() > self.cache_bytes):
            self._drop(self.cache[0])
            dropped = True
        if dropped:
            import torch

            torch.cuda.empty_cache()

    def _take_over(self, keep: list[int]) -> None:
        """Save the rows of every kept snapshot the next prefill overwrites, dropping the oldest entries past the memory budget; both ranks decide alike."""
        from .decode import row_bytes, save_rows

        live = self.live
        dropped = False

        def resumes(c) -> bool:
            return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

        for snap in list(self.cache):
            n = len(snap.ids)
            if snap not in self.cache or snap.rows is not None or resumes(snap):
                continue
            if self.disk is not None and snap.drafter_end < 0 and self.disk.has(snap.ids):
                # Disk holds only target/MTP rows; DFlash2 snapshots still need their device copy.
                self.cache.remove(snap)
                continue
            if live[:n] != snap.ids:                  # its rows are already gone: nothing to resume from
                self._drop(snap)
                continue
            need = row_bytes(self.e, snap, self.drafter)
            while self._held_bytes() + need > self.cache_bytes:
                old = next((c for c in self.cache if c is not snap and not resumes(c)), None)
                if old is None:
                    break
                self._drop(old)
                dropped = True
            if self._held_bytes() + need > self.cache_bytes:
                self._drop(snap)
                dropped = True
                continue
            save_rows(self.e, snap, self.drafter)
        if dropped:
            import torch

            torch.cuda.empty_cache()             # give the freed rows back rather than keep them in torch's pool

    def _held_bytes(self) -> int:
        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.cache)

    @staticmethod
    def _image_starts(prompt: list[int], images) -> list[int]:
        """Where each image's keyed run begins; refuses a prompt whose runs are not the request's images."""

        import numpy as np

        ids = np.asarray(prompt, dtype=np.int64)
        starts = [int(i) for i in np.flatnonzero(ids < 0) if i == 0 or ids[i - 1] != ids[i]]
        if len(starts) != len(images or ()):
            raise ValueError(f"the prompt holds {len(starts)} image runs for {len(images or ())} images")
        for a, image in zip(starts, images or ()):
            end = a + image.tokens
            if end > len(ids) or (ids[a:end] != image.key).any() or (end < len(ids) and ids[end] == image.key):
                raise ValueError("an image's placeholder tokens do not match the image")
        return starts

    def _image_rows(self, prompt: list[int], cut: int, images) -> tuple | None:
        """The rows of the images a prefill from ``cut`` reads (rank 0 encodes them, rank 1 receives them)."""

        import numpy as np

        torch = self.torch
        ids = np.asarray(prompt, dtype=np.int64)
        pos = np.flatnonzero(ids[cut:] < 0) + cut
        if not pos.size:
            return None
        D = self.w.cfg.hidden
        if self.rank == 0:
            parts = [self.tower.encode(image)[max(a, cut) - a:]
                     for a, image in zip(self._image_starts(prompt, images), images) if a + image.tokens > cut]
            mine = torch.cat(parts).contiguous()
        else:
            mine = torch.zeros((pos.size, D), dtype=torch.bfloat16, device="cuda")
        got = torch.empty((self.world * pos.size * D,), dtype=torch.bfloat16, device="cuda")
        self.comm.all_gather(mine.view(-1), got)
        return pos, got[:pos.size * D].view(pos.size, D)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, images=None, resuming: bool = False) -> dict[str, Any]:
        if code[0] in (16, 17):
            raise ValueError("priced draft policies require batched GLM serving")
        from .decode import DepthPolicy, dflash_decode, mtp_decode, prefill, serial_decode, take_snapshot
        from .drafter_choice import DrafterChoice, auto_decode

        from . import tp3_probe

        if tp3_probe.recorder is not None:            # TF_TP3_CAPTURE: this request's steps in a folder of their own
            tp3_probe.recorder.begin_request(self.rank, draft)
        if resuming:                                  # every rank resumes, or none does (a rank lost its copy)
            both = self._gather_ints([int(hit is not None)])
            if not all(row[0] for row in both):
                hit = None
        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        # a request writes the caches from its resume point: other conversations' rows are saved first, a saved resume point's restored
        from .decode import load_rows

        cut = len(hit.ids) if hit is not None else 0
        self._take_over(list(hit.ids) if hit is not None else [])
        if hit is not None and getattr(hit, "disk", None) is not None:
            hit = self.disk.load(self.e, hit.disk)    # its rows straight into the live caches
        elif hit is not None and hit.rows is not None:
            load_rows(self.e, hit, self.drafter)
            hit.rows, hit.nbytes = None, 0            # live again
        self.live = list(prompt)
        self.e.images = self._image_rows(prompt, cut, images)
        marks = self.disk is not None and draft
        repeat = self.cache_last_token and len(prompt) > 1
        snap = hit if repeat and cut == len(prompt) - 1 else None
        mark = self.checkpoint_after if marks else None
        keep = (lambda checkpoint: self.disk.put(self.e, checkpoint)) if marks else None
        if repeat:
            # End the cold prefill at the same prefix the repeat will restore. Do this for the
            # fresh serial reference too; only cache-enabled requests retain the checkpoint.
            def mark(pos):
                at = self.checkpoint_after(pos) if marks else None
                if pos < len(prompt) - 1:
                    at = min(at, len(prompt) - 1) if at is not None else len(prompt) - 1
                return at

            def keep(checkpoint):
                nonlocal snap
                if draft and len(checkpoint.ids) == len(prompt) - 1:
                    snap = checkpoint
                    self._remember(snap)
                if marks:
                    self.disk.put(self.e, checkpoint)
        try:
            first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=hit,
                            mark=mark, keep=keep)
        finally:
            self.e.images = None
        prefill_s = time.perf_counter() - t0
        if draft:
            if repeat:
                assert snap is not None, "the prompt's penultimate checkpoint was not captured"
                if snap is hit:
                    self._remember(snap)     # promote a repeated hit without copying full-prompt state
            else:
                snap = take_snapshot(self.e, prompt, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                     drafter=drafter)
                self._remember(snap)
        stats: dict[str, Any] = {"prefill_s": prefill_s, "cached": cut}
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            self._write(snap)
            return stats
        policy = decode_policy(code)
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
        elif auto:
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            _, explore, every, margin, sampled_too = policy
            choice = None
            if drafter is not None and (greedy or sampled_too):
                choice = DrafterChoice(self.costs, first="f" if greedy else "m", explore=explore, every=every,
                                       margin=margin)
            from .lookup import PromptLookup

            lookup = PromptLookup(prompt) if os.environ.get("TF_GLM_LOOKUP", "1") != "0" else None
            res = auto_decode(self.e, drafter, first, max_tokens, sampling, choice=choice, m_policy=m_policy,
                              f_policy=DepthPolicy(5, fixed=True, confidence=0.3), stop_eos=stop_eos,
                              on_tokens=on_tokens, lookup=lookup)
        elif use_dflash:
            res = dflash_decode(self.e, self.drafter, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                                on_tokens=on_tokens)
        else:
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens)
        # the caches now hold prompt and reply; only prompts are snapshotted, since a later prompt prefills the reply again
        self.live = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if res.arms:
            stats.update(drafters=res.arms)
        if res.depths:                      # each round's drafts and kept tokens, whichever drafter ran
            stats.update(depths=res.depths, keeps=res.keeps)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        self._write(snap)
        return stats

    def _write(self, snap) -> None:
        """The prompt's rows to disk once its reply is out (decoding writes only past the prompt, so they are intact)."""

        if self.disk is not None and snap is not None:
            self.disk.put(self.e, snap)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True) -> dict[str, Any]:
        """Mirror one rank-0 request on rank 1; draft=False uses serial decoding and fresh prefill as the reference drafted replies must equal."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        images = getattr(self.request, "images", None)
        if images and self.tower is None:
            raise ValueError("this server reads text only: image input needs the vision tower on rank 0 "
                             "(see the startup log)")
        self._image_starts(prompt, images)          # before rank 1 is told of the request
        if self.concurrent:
            spec = getattr(self.request, "policy", None) or self.policy
            serial = not draft or self.serial_only or spec == "0"
            encode_policy("0" if serial else spec)     # reject before queuing or sending rank control
            stats = self.scheduler.submit(list(prompt), max_tokens, sampling, not serial, on_tokens,
                                          stop_eos=bool(getattr(self.request, "stop_eos", True)),
                                          policy="0" if serial else spec, images=images,
                                          priority=getattr(self.request, "priority", None),
                                          prefill_slice_layers=getattr(self.request, "prefill_slice_layers", None),
                                          copy_request=getattr(self.request, "copy_drafts", None),
                                          cofill=getattr(self.request, "cofill", None),
                                          cancelled=getattr(self.request, "cancelled", None),
                                          session_event=getattr(self.request, "session_event", None))
            if getattr(self, "reply_prefill", False):
                self.request.reply_prefill_epoch = stats.pop("_reply_prefill_epoch", None)
            stats.setdefault("policy", "0" if serial else spec)
            stats.setdefault("drafts", not serial)
            return stats
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None) or self.policy
        code = self._effective(encode_policy(spec))
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        hit = self._resume(list(prompt), code) if draft else None
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), len(hit.ids) if hit is not None else 0,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0)] + code
        self._ring()
        self._share(header)
        self._share(list(prompt))
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, images,
                          resuming=hit is not None)
        stats.update(policy=spec, drafts=draft)
        return stats

    def score(self, ids: list[int], score_start: int) -> list[float]:
        """Quality-only teacher-forced NLL; rank zero dispatches identical prefill work to all ranks."""
        if not self.quality_score_enabled:
            raise ValueError("TF_GLM_QUALITY_SCORE is off")
        from .quality_score import score_prefill

        if self.concurrent:
            result: list[float] = []
            self.scheduler.submit(ids, 1, None, False, lambda _: False,
                                  quality_score_start=score_start, quality_result=result)
            return result
        self._ring()
        self._share([-1, score_start])
        self._share(ids)
        self._take_over([])
        self.live = []
        return score_prefill(self.e, ids, score_start)

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        if self.concurrent:
            self.multi.follow()
            return

        while True:
            self._await_bell()
            header = self._share(None)
            if len(header) == 2 and header[0] == -1:
                from .quality_score import score_prefill

                ids = self._share(None)
                self._take_over([])
                self.live = []
                score_prefill(self.e, ids, header[1])
                continue
            max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, *code = header
            prompt = self._share(None)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi)) if temperature > 0 else None
            hit = None
            if cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
                if hit is None and self.disk is not None:
                    entry = self.disk.find(prompt[:cached])
                    hit = entry.stub() if entry is not None else None
                # without it both ranks prefill afresh (``_run`` agrees on it)
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                      resuming=bool(cached))
