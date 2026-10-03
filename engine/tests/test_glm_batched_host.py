"""Pooled decoder behavior on CPU tensors; CUDA arithmetic is tested separately."""

import copy
import importlib
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from test_cuda_geometry import allocations  # noqa: F401  (fake Triton for CPU-only installations)


class _Pool:
    """Host allocator stand-in; the production allocator has its own contract tests."""
    def __init__(self, capacity, guard=8):
        self.capacity, self.guard, self.spans = capacity, guard, {}

    def _find(self, tokens):
        length = (tokens + self.guard + 3) // 4 * 4
        at = 0
        for start, n in sorted(self.spans.values()):
            if start - at >= length:
                return at, length
            at = start + n
        return (at, length) if at + length <= self.capacity else None

    def can_allocate(self, tokens):
        return self._find(tokens) is not None

    def allocate(self, key, tokens):
        found = self._find(tokens)
        if found is not None:
            self.spans[key] = found
        return found

    def release(self, key):
        self.spans.pop(key, None)

    @property
    def available(self):
        return self.capacity - sum(n for _, n in self.spans.values())


class _State:
    def __init__(self, w, capacity, rows):
        self.capacity, self.latent = capacity, True
        self.rows = rows
        self.kc, self.vc = [torch.zeros(capacity, 2)], [None]
        self.index = [(torch.zeros(capacity, 2), torch.zeros(capacity, 2), torch.zeros(capacity // 4 + 2, 2))]
        self.conv, self.rec = torch.zeros(1, 3, 2), torch.zeros(2, 1, 2, 2)
        self.pos_dev, self.mtp_pos_dev = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        self.reset()

    def reset(self):
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0]
        self.mtp_drafted = 0
        self.set_pos(0)
        self.set_mtp_len(0)

    def set_pos(self, n):
        self.pos = n
        self.pos_dev.fill_(n)

    def set_mtp_len(self, n):
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)


class _Buffers:
    def __init__(self, w, rows, capacity):
        self.rows, self.prefill = rows, False
        self.fnormed = torch.zeros(rows, 2)
        self.taps = []

    def set_taps(self, layers, hidden):
        self.taps = [torch.zeros(self.rows, hidden) for _ in layers]


def _logits(tokens, pos):
    out = torch.full((len(tokens), 64), -6.0)
    for i, token in enumerate(tokens):
        best = (token + pos + i + 1) % 64
        out[i, best], out[i, (best + 13) % 64] = 2, 1.8
    return out


class _Drafter:
    def __init__(self, capacity):
        self.block, self.cap, self.tap_layers = 8, capacity + 8, (0, 1)
        self.kc = [torch.zeros(2, self.cap, 2)]
        self.vc = [torch.zeros(2, self.cap, 2)]
        self.pos_dev = torch.zeros(1, dtype=torch.int64)
        self.block_graph, self.tap_graphs = object(), {1: object()}
        self.reset()

    def reset(self):
        self.context_end = 0
        self.pos_dev.zero_()
        self.added, self.proposals = [], []

    def add_taps(self, taps):
        self.added.append(taps.clone())
        n = len(taps)
        for c in self.kc + self.vc:
            c[:, self.context_end:self.context_end + n].copy_(taps[:, :2])
        self.context_end += n
        self.pos_dev.fill_(self.context_end)

    def propose(self, pending, depth, sampling, confidence):
        self.proposals.append((pending, depth, confidence))
        out = []
        for i in range(depth):
            pending = (pending + self.context_end + i + 1) % 64
            out.append(pending)
        return out


@pytest.fixture
def setup_decoder(monkeypatch, allocations):
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.batched")
    pool = ModuleType("tensorfold.families.glm5_next.cuda.pool")
    pool.TokenPool = _Pool
    monkeypatch.setitem(sys.modules, pool.__name__, pool)
    monkeypatch.setattr(mod, "State", _State)
    monkeypatch.setattr(mod, "Buffers", _Buffers)

    def stage(w, b, windows):
        segs, at = [], 0
        for st, tokens in windows:
            assert st.pos + len(tokens) <= st.capacity
            st.tokens = tokens
            segs.append((st, at, at + len(tokens)))
            at += len(tokens)
        assert at <= b.rows
        return segs

    def compute(w, segs, b, eager=True):
        logits = []
        for st, a0, a1 in segs:
            n = a1 - a0
            data = torch.tensor([[token, st.pos + i] for i, token in enumerate(st.tokens)], dtype=torch.float32)
            st.kc[0][st.pos:st.pos + n].copy_(data)
            for tap in b.taps:
                tap[a0:a1].copy_(data)
            b.fnormed[a0:a1].copy_(data)
            logits.append(_logits(st.tokens, st.pos))
        return torch.cat(logits)

    def commit(w, st, b, rows, keep):
        assert 1 <= keep <= rows <= st.rows
        st.set_pos(st.pos + keep)
        st.conv.fill_(st.pos)
        st.cur = [1 - st.cur[0]]
        st.rec[st.cur[0]].fill_(st.pos)

    def prefill(e, prompt, sampling, mtp, drafter, resume):
        begin = 0
        if resume is None:
            e.st.reset()
            if drafter is not None:
                drafter.reset()
        else:
            begin = len(resume.ids)
            mod.owner_restored.append(([c.clone() for c in e.st.kc],
                                       [c[:, :begin].clone() for c in drafter.kc] if drafter else []))
            from tensorfold.families.glm5_next.cuda.decode import restore
            restore(e, resume, drafter)
        data = torch.tensor([[token, i] for i, token in enumerate(prompt)], dtype=torch.float32)
        e.st.kc[0][begin:len(prompt)].copy_(data[begin:])
        for ik, ig, pk in e.st.index:
            ik[begin:len(prompt)].copy_(data[begin:])
            ig[begin:len(prompt)].copy_(data[begin:])
            pk[:len(prompt) // 4 + 1].fill_(len(prompt))
        e.st.set_pos(len(prompt))
        e.st.conv.fill_(len(prompt))
        e.st.rec[e.st.cur[0]].fill_(len(prompt))
        if drafter is not None:
            drafter.add_taps(torch.cat([data[begin:], data[begin:]], dim=1))
        e.last_hidden = data[-1:].clone()
        from tensorfold.families.glm5_next.cuda.decode import sample_rows
        return sample_rows(e.w, _logits(prompt[-1:], len(prompt) - 1), [len(prompt)], sampling)[0]

    monkeypatch.setattr(mod, "stage_streams", stage)
    monkeypatch.setattr(mod, "compute_streams", compute)
    monkeypatch.setattr(mod, "commit", commit)
    monkeypatch.setattr(mod, "prefill", prefill)
    mod.owner_restored = []

    def make(slots=4, limit=512, drafter=True, cache_bytes=100_000, pool_limit=None):
        from tensorfold.families.glm5_next.cuda.copy_drafts import configured_copy_rows

        rows = configured_copy_rows()
        capacity = (pool_limit or limit) + rows + (slots - 1) * 64
        w = SimpleNamespace(mtp=None, cfg=SimpleNamespace(eos=(63,), hidden=2, quant="exl3"),
                            vocab_offset=0, comm=None)
        b = _Buffers(w, rows, capacity)
        b.set_taps((0, 1), 2)
        e = SimpleNamespace(w=w, rows=rows, st=_State(w, capacity, rows), buf=b, last_hidden=None,
                            replays={"main": 0}, graphs=object(), images=None)

        def forward(tokens):
            e.replays["main"] += 1
            return compute(w, stage(w, b, [(e.st, tokens)]), b)

        e.forward = forward
        messages, overlays = [], []
        owner = SimpleNamespace(w=w, e=e, drafter=_Drafter(capacity) if drafter else None, limit=limit,
                                pool_limit=pool_limit or limit,
                                rank=0, policy="fc7:0.3" if drafter else "0", cache_bytes=cache_bytes,
                                cache_entries=8, _share=lambda msg: messages.append(msg),
                                _effective=lambda code: code,
                                _image_rows=lambda prompt, cut, images: overlays.append((list(prompt), cut, images)))
        decoder = mod.BatchedDecoder(owner, slots)
        decoder.messages, decoder.overlays = messages, overlays
        return decoder

    return mod, make


def _stream(prompt, count, *, draft=True, sampling=None, policy=None):
    s = Stream(list(prompt), count, sampling, draft=draft, stop_eos=False)
    if policy is not None:
        s.policy = policy
    return s


def _drain(decoder):
    decoder.finish([s for s in decoder.streams.values() if s.done])
    while decoder.live():
        decoder.finish(decoder.round())


@pytest.mark.parametrize("seed", [0, -1, 2**63 + 123, 2**64 - 1])
def test_sampling_round_trip_preserves_binary64(setup_decoder, seed):
    mod, _ = setup_decoder
    sampling = Sampling(seed, 0.6000000000000001, 17, 0.8999999999999998)
    packed = mod.pack_sampling(sampling)
    restored = mod.unpack_sampling(packed)
    assert restored.seed == seed & (2**64 - 1)
    assert restored.temperature.hex() == sampling.temperature.hex()
    assert restored.top_p.hex() == sampling.top_p.hex()
    assert restored.top_k == sampling.top_k
    assert all(-(2**31) <= n < 2**31 for n in packed)


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_mixed_streams_equal_fresh_serial_and_bound_acceptance(setup_decoder, slots):
    _, make = setup_decoder
    streams = [_stream([3 + i, 11, 5] + [2] * i, 12 + i, draft=i % 3 != 2,
                       sampling=Sampling(7 + i, .73123456789, 16, .92345678901) if i % 2 else None,
                       policy="fc7:0.3" if i % 3 == 0 else "f2") for i in range(slots)]
    expected = []
    for s in streams:
        reference = make(slots)
        plain = _stream(s.prompt, s.count, draft=False, sampling=s.sampling)
        reference.admit(plain)
        _drain(reference)
        expected.append(plain.out)
    decoder = make(slots)
    for s in streams:
        decoder.admit(s)
    _drain(decoder)
    assert [s.out for s in streams] == expected
    for s in streams:
        assert s.drafted_tokens == sum(s.depths)
        assert s.accepted_draft_tokens == sum(k - 1 for k in s.keeps)
        assert all(k <= d + 1 for k, d in zip(s.keeps, s.depths))
        assert s.st.pos == len(s.prompt) + len(s.out) - 1
        if not s.draft:
            assert set(s.depths) == {0}


def test_pool_defer_is_nonmutating_and_global_context_ignores_guard_padding(setup_decoder):
    _, make = setup_decoder
    d = make(2, limit=40)
    a, b = _stream([1, 2], 24), _stream([1, 2], 16)
    d.admit(a)
    before = copy.deepcopy(d.pool.spans)
    assert not d.can_admit(b)
    assert d.pool.spans == before
    d.finish([a])
    assert d.can_admit(b)
    d.admit(b)
    _drain(d)
    assert d.pool.available == d.capacity


def test_larger_shared_pool_admits_two_large_pi_reservations_and_queues_when_full(setup_decoder):
    _, make = setup_decoder
    d = make(2, limit=262, pool_limit=320)
    a, b, c = _stream([1] * 100, 32), _stream([2] * 140, 32), _stream([3], 30)
    with pytest.raises(ValueError, match="do not fit"):
        d.can_admit(_stream([4] * 240, 32))
    d.admit(a)
    assert d.can_admit(b)
    d.admit(b)
    assert not d.can_admit(c)
    d.finish([a])
    assert d.can_admit(c)
    d.admit(c)
    _drain(d)
    assert d.pool.available == d.capacity


def test_invalid_admission_is_recoverable_and_sends_nothing(setup_decoder):
    _, make = setup_decoder
    d = make(2, limit=32)
    for bad in [_stream([], 1), _stream([1] * 32, 1), _stream([1], 2, policy="garbage")]:
        with pytest.raises(ValueError):
            d.can_admit(bad)
        with pytest.raises(ValueError):
            d.admit(bad)
    assert not d.messages and not d.live() and d.broken is None


def test_fragmented_slot_zero_uses_views_without_rebinding_graph_tensors(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    base = d.owner.e
    original = (base.st.kc[0], base.st.pos_dev, base.st.index[0][2], base.graphs)
    a, b = _stream([1], 3), _stream([2] * 8, 8)
    d.admit(a)
    d.admit(b)
    d.finish([a])
    c = _stream([4] * 16, 16)
    d.admit(c)
    assert c.slot == 0 and c.start > 0 and c.engine is not base
    assert c.engine.graphs is None
    assert all(x is y for x, y in zip(original, (base.st.kc[0], base.st.pos_dev, base.st.index[0][2], base.graphs)))
    assert c.st.pos_dev is base.st.pos_dev
    assert c.drafter.kc[0].stride(0) == d.owner.drafter.kc[0].stride(0)
    assert c.drafter.cap == d.owner.drafter.cap
    assert c.drafter.pos_dev is not d.owner.drafter.pos_dev
    assert c.st.index[0][2].shape[0] == c.st.capacity // 4 + 2
    assert b.start + b.span <= c.start
    _drain(d)


def test_single_slot_zero_retains_original_forward(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    s = _stream([1, 2], 20)
    d.admit(s)
    assert s.engine is d.owner.e and s.drafter is d.owner.drafter
    _drain(d)
    assert d.owner.e.replays["main"] == s.rounds


def test_decode_observer_copies_all_rows_and_preserves_output(setup_decoder, monkeypatch, tmp_path):
    _, make = setup_decoder
    baseline = make(2)
    plain = _stream([1, 2], 12)
    baseline.admit(plain)
    _drain(baseline)
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_GLM_DECODE_TAP_EVERY", "1")
    observed = make(2)
    shown = _stream([1, 2], 12)
    observed.admit(shown)
    _drain(observed)
    observed.observer.close()
    assert shown.out == plain.out
    records = [__import__("json").loads(line) for line in (tmp_path / "rank0.jsonl").read_text().splitlines()]
    assert len(records) == shown.rounds
    assert all(row["path"] == "graph-main" and row["graph_paths"] is not None for row in records)
    assert all(row["verify_rows"] == row["streams"][0]["rows"] for row in records)
    assert all(row["streams"][0]["session_key"] and row["streams"][0]["cache_miss_reason"] == "boot"
               for row in records)
    assert all(len(row["streams"][0]["row_segments"]) == row["streams"][0]["rows"] - 1
               and row["streams"][0]["prompt_sha256"] for row in records)
    phases = [__import__("json").loads(line) for line in
              (tmp_path / "rank0-phases.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in phases] == ["prefill"] + ["decode"] * shown.rounds
    assert all(row["host_end_s"] >= row["host_start_s"] for row in phases)
    payload = torch.load(tmp_path / records[0]["tap_file"], weights_only=False)
    assert len(payload["rows"]) == records[0]["verify_rows"]
    assert len(payload["taps"]) == 2
    assert payload["taps"][0].shape[0] == records[0]["verify_rows"]


def test_decode_observer_io_failure_does_not_change_output(setup_decoder, monkeypatch, tmp_path):
    _, make = setup_decoder
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    d = make(2)
    d.observer.file = tmp_path  # opening a directory for append raises OSError
    s = _stream([1, 2], 8)
    d.admit(s)
    _drain(d)
    d.observer.flush()
    assert len(s.out) == 8 and d.observer.enabled is False
    d.observer.close()


def test_decode_observer_at_c8_preserves_mixed_outputs(setup_decoder, monkeypatch, tmp_path):
    _, make = setup_decoder
    def run():
        d = make(8)
        streams = [_stream([i + 1, 3, 5], 9, policy="f2" if i % 2 else "fc7:0.3")
                   for i in range(8)]
        for s in streams:
            d.admit(s)
        _drain(d)
        d.observer.close()
        return [s.out for s in streams]
    baseline = run()
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_GLM_DECODE_TAP_EVERY", "0")
    assert run() == baseline


def test_priced_policy_without_table_rewrites_to_literal_fc7(setup_decoder):
    mod, make = setup_decoder
    d = make(2)
    s = _stream([1, 2], 12, policy="fp7")
    code, _ = d._request(s)
    assert code == mod.encode_policy("fc7:0.3")
    assert s.price_table_missing == "missing"


def test_fair_background_cap_applies_to_priced_policy(setup_decoder):
    mod, make = setup_decoder
    d = make(2)
    d.fair_schedule = True
    d.price_table = object()
    s = _stream([1, 2], 12, policy="fp7")
    s.priority = "background"
    code, _ = d._request(s)
    assert code == [16, 1, 0, 0]


def test_priced_round_names_full_work_and_equals_serial(setup_decoder, monkeypatch, tmp_path):
    mod, make = setup_decoder
    cells = [{"rows": n, "load": 1, "ctx": "short", "path": "eager",
              "median_ms": 10 + n, "spread": .2, "reps": 30} for n in range(1, 9)]
    table = {"schema": "tf-price/v1", "build": "build", "weights": "weights", "drafter": "drafter",
             "kind": "measured", "cells": cells}
    path = tmp_path / "price.json"
    path.write_text(__import__("json").dumps(table))
    monkeypatch.setenv("TF_GLM_PRICE_TABLE", str(path))
    monkeypatch.setenv("TF_GLM_WEIGHT_ID", "weights")
    monkeypatch.setenv("TF_GLM_DRAFTER_ID", "drafter")
    monkeypatch.setenv("TF_GLM_BUILD_ID", "build")
    baseline = make(2)
    plain = _stream([1, 2], 12, draft=False)
    baseline.admit(plain)
    _drain(baseline)
    priced = make(2)
    drafter_type = type(priced.owner.drafter)
    original = drafter_type.propose

    def observed_propose(self, pending, depth, sampling, confidence):
        output = original(self, pending, depth, sampling, confidence)
        if getattr(self, "observe", False):
            self.last_observation = {"claims": [.8] * len(output),
                                     "candidates": [list(range(64)) for _ in output]}
        return output

    monkeypatch.setattr(drafter_type, "propose", observed_propose)
    s = _stream([1, 2], 12, policy="fp7")
    priced.admit(s)
    _drain(priced)
    assert s.out == plain.out
    rounds = [m for m in priced.messages if m[0] == mod.ROUND2]
    assert rounds and all(len(m) == 2 + 3 * m[1] for m in rounds)
    assert all(m[2] >= 0 for m in rounds)


def test_invalid_table_rewrites_priced_request_at_admission(setup_decoder, monkeypatch, tmp_path):
    mod, make = setup_decoder
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"wrong","cells":[]}')
    monkeypatch.setenv("TF_GLM_PRICE_TABLE", str(path))
    d = make(2)
    s = _stream([1, 2], 12, policy="fp7")
    code, _ = d._request(s)
    assert code == mod.encode_policy("fc7:0.3")
    assert s.price_table_missing == "invalid"


def test_priced_instruction_uses_local_prefix_and_rejects_impossible_trim(setup_decoder):
    mod, _ = setup_decoder
    s = SimpleNamespace(drafts=[10, 11, 12], priced=True, generation_cap=3)
    mod.BatchedDecoder._apply_instruction([s], [mod.SRC_D], [2])
    assert s.drafts == [10, 11]
    with pytest.raises(RuntimeError, match="exceeds"):
        mod.BatchedDecoder._apply_instruction([s], [mod.SRC_D], [3])
    with pytest.raises(RuntimeError, match="exceeds"):
        mod.BatchedDecoder._apply_instruction([s], [mod.SRC_C], [1])


def test_path_for_covers_graph_edge_and_fragmented_slot(setup_decoder, monkeypatch):
    mod, _ = setup_decoder
    e = SimpleNamespace(st=SimpleNamespace(pos=10, parity=0, index=None),
                        w=SimpleNamespace(cfg=SimpleNamespace(dense_limit=100)),
                        graphs=SimpleNamespace(main={(6, 0): object()}, sparse={}))
    assert mod.path_for(e, 6) == "graph-main"
    assert mod.path_for(e, 7) == "eager"
    assert mod.path_for(e, 8) == "eager"
    e.graphs = None
    assert mod.path_for(e, 6) == "eager"


def test_snapshots_fit_before_copy_and_restore_complete_draft_prefix(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    d = make(2)
    first = _stream([1, 2, 3, 4, 5], 4)
    d.admit(first)
    _drain(d)
    needed = d.held_bytes()
    d.owner.cache_bytes = needed + needed // 2
    gone = weakref.ref(d.cache[0])
    original = mod.take_snapshot

    def checked(*args, **kwargs):
        assert gone() is None, "eviction must release old copies before growing the store"
        assert d.held_bytes() + needed <= d.owner.cache_bytes
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "take_snapshot", checked)
    second = _stream([7, 8, 9, 10, 11], 4)
    d.admit(second)
    _drain(d)
    monkeypatch.setattr(mod, "take_snapshot", original)
    assert len(d.cache) == 1 and d.held_bytes() <= d.owner.cache_bytes
    snap = d.cache[0]
    assert snap.drafter_end == 5
    assert all(row.shape[1] == 5 for row in snap.rows[-2:])
    target, draft = snap.rows[0].clone(), snap.rows[-2].clone()
    d.owner.e.st.kc[0].fill_(float("nan"))
    d.owner.drafter.kc[0].fill_(float("nan"))
    revisit = _stream(second.prompt + [12], 3)
    d.admit(revisit)
    assert revisit.cached == len(second.prompt)
    restored_target, restored_draft = mod.owner_restored[-1]
    assert torch.equal(restored_target[0][:5], target)
    assert torch.equal(restored_draft[0], draft)


def test_oversized_snapshot_allocates_no_copies_and_no_draft_forces_fresh(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    d = make(2, cache_bytes=1)
    monkeypatch.setattr(mod, "take_snapshot", lambda *a, **kw: pytest.fail("over-budget snapshot copied"))
    s = _stream([1, 2], 4)
    d.admit(s)
    _drain(d)
    assert not d.cache
    plain = _stream(s.prompt + [3], 4, draft=False)
    d.admit(plain)
    assert plain.cached == 0 and not plain.use_dflash


def test_cancel_skips_forward_and_eos_stops_at_first_terminal(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    a, b = _stream([1, 2], 20), _stream([3, 4], 20)
    d.admit(a)
    d.admit(b)
    a.cancelled = lambda: True
    pos = a.st.pos
    done = d.round()
    assert a in done and a.st.pos == pos and b.rounds == 1
    d.finish(done)
    _drain(d)
    eos = _stream([62], 20)
    eos.stop_eos = True
    d.admit(eos)
    assert eos.done and eos.out == [63] and not eos.drafts
    d.finish([eos])


def test_cancel_callback_failure_releases_named_stream_and_preserves_peer(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    a, b = _stream([1, 2], 20), _stream([3, 4], 20)
    d.admit(a)
    d.admit(b)
    a.cancelled = lambda: (_ for _ in ()).throw(ValueError("client failed"))
    done = d.round()
    assert a in done and isinstance(a.error, ValueError) and d.broken is None
    d.finish(done)
    assert d.messages[-1] == [3, 1, a.sid]
    _drain(d)
    reference = make(2)
    plain = _stream(b.prompt, b.count, draft=False)
    reference.admit(plain)
    _drain(reference)
    assert b.out == plain.out


def test_no_draft_bypasses_an_available_saved_prompt(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    first = _stream([1, 2, 3], 4)
    d.admit(first)
    _drain(d)
    assert len(d.cache) == 1
    plain = _stream(first.prompt + [4], 4, draft=False)
    d.admit(plain)
    assert plain.cached == 0 and not plain.use_dflash and not plain.use_mtp
    assert len(d.cache) == 1


def test_fc7_confidence_and_residual_auto_resolution_are_per_request(setup_decoder):
    _, make = setup_decoder
    d = make(4)
    streams = [_stream([1, 2], 12, policy=p) for p in ("fc7:0.3", "f2", "auto:2:8:0.03")]
    for s in streams:
        d.admit(s)
    assert [(s.depth_policy.most, s.depth_policy.confidence) for s in streams] == [(7, .3), (2, 0), (5, .3)]
    assert [s.policy for s in streams] == ["fc7:0.3", "f2", "fc5:0.3"]
    assert [len(s.drafts) for s in streams] == [7, 2, 5]


def test_images_run_after_admit_and_are_cleared_even_on_failure(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    d.owner.tower = object()
    s = _stream([1, 2], 3)
    s.images = [object()]

    def overlay(prompt, cached, images):
        assert d.messages[0][0] == 1 and d.messages[1] == prompt
        assert images is s.images and cached == 0
        return ([], torch.empty(0, 2))

    d.owner._image_rows = overlay
    d.admit(s)
    assert s.engine.images is None


def test_image_prefix_hit_checks_full_digest(setup_decoder):
    mod, make = setup_decoder
    d = make(2)
    snap = mod.Snapshot([1, -7, -7, 2], torch.zeros(1), torch.zeros(1), None, -1, 4)
    snap.image_digests = (b"a" * 32,)
    d.cache.append(snap)
    code = mod.encode_policy("fc7:0.3")
    prompt = snap.ids + [3]
    assert d._saved(prompt, code, images=[SimpleNamespace(digest=b"a" * 32)]) is snap
    assert d._saved(prompt, code, images=[SimpleNamespace(digest=b"b" * 32)]) is None


def test_alternating_sessions_keep_both_latest_ring_snapshots(setup_decoder):
    _, make = setup_decoder
    d = make(2)
    d.owner.drafter.ring, d.owner.drafter.window = 16, 3
    prompts = ([1, 2, 3], [7, 8, 9], [1, 2, 3, 4], [7, 8, 9, 10],
               [1, 2, 3, 4, 5], [7, 8, 9, 10, 11])
    hits = []
    for i, prompt in enumerate(prompts):
        if i == 2:
            # Enough for a new state and tail, but not another full prompt copy.
            d.owner.cache_bytes = d.held_bytes() + d.held_bytes() * 49 // 100
        s = _stream(prompt, 2)
        d.admit(s)
        hits.append(s.cached)
        _drain(d)
        assert d.held_bytes() <= d.owner.cache_bytes
        if i == 2:
            assert [snap.ids for snap in d.cache] == [list(prompts[1]), list(prompts[2])]
            d.owner.cache_bytes = 100_000
    assert hits == [0, 0, 3, 3, 4, 4]
    assert [snap.ids for snap in d.cache] == [list(prompts[-2]), list(prompts[-1])]
    assert all(snap.drafter_rows is not None for snap in d.cache)


@pytest.mark.parametrize("drafter", [False, True], ids=["target-cache", "draft-cache"])
def test_eight_slot_score_between_prompts_restores_saved_rows(setup_decoder, monkeypatch, drafter):
    from tensorfold.families.glm5_next.cuda import quality_score

    mod, make = setup_decoder
    d = make(slots=8, drafter=drafter)
    a = [1, 2, 3, 4]
    first = _stream(a, 2)
    d.admit(first)
    _drain(d)
    snap = d.cache[0]
    assert snap.rows is not None

    def overwrite(e, ids, start):
        e.st.kc[0].fill_(777)
        return [0.5] * (len(ids) - start)

    monkeypatch.setattr(quality_score, "score_prefill", overwrite)
    score = _stream([9, 10], 1, draft=False)
    score.quality_score_start, score.quality_result = 1, []
    assert d.can_admit(score)
    d.begin_admit(score)
    assert score.done and score.quality_result == [0.5]
    assert snap.rows is not None

    resumed = _stream(a + [5], 2)
    d.admit(resumed)
    assert resumed.cached == len(a)
    restored = mod.owner_restored[-1][0][0][:len(a)].clone()
    fresh = make(slots=8, drafter=drafter)
    fresh.admit(_stream(a + [5], 2))
    assert torch.equal(restored, fresh.owner.e.st.kc[0][:len(a)])


def test_post_admit_failure_poisons_decoder_and_drop_frees_reservations(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    d = make(2)
    s = _stream([1, 2], 8)
    d.admit(s)
    monkeypatch.setattr(mod, "sample_streams", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fault")))
    with pytest.raises(RuntimeError, match="fault"):
        d.round()
    assert d.drop() == [s] and d.pool.available == d.capacity
    with pytest.raises(RuntimeError, match="out of step"):
        d.can_admit(_stream([1], 2))


def test_followers_mirror_named_work_with_exact_sampling(setup_decoder):
    _, make = setup_decoder
    leader = make(2)
    a = _stream([1, 3], 14, sampling=Sampling(2**64 - 3, .71234567891, 16, .92345678911))
    b = _stream([2, 5, 8], 12, policy="f2")
    leader.admit(a)
    leader.admit(b)
    observed = []
    follower = make(2)
    follower.rank = follower.owner.rank = 1
    original = follower._finish

    def finish(sids):
        observed.extend((sid, list(follower.streams[sid].out), follower.streams[sid].sampling) for sid in sids)
        original(sids)

    follower._finish = finish
    _drain(leader)
    messages = iter(leader.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert sorted((sid, out) for sid, out, _ in observed) == [(a.sid, a.out), (b.sid, b.out)]
    sampling = next(s for sid, _, s in observed if sid == a.sid)
    assert sampling.temperature.hex() == a.sampling.temperature.hex()
    assert sampling.top_p.hex() == a.sampling.top_p.hex()
    assert follower.pool.available == follower.capacity and follower.broken is not None


@pytest.mark.parametrize("policy", ["fc7:0.3", "fq7:0.3"], ids=["legacy-round", "priced-round2"])
@pytest.mark.parametrize("cause", ["cancel", "callback-error"])
def test_follower_accepts_survivor_round_before_cancel_done(setup_decoder, monkeypatch, policy, cause):
    mod, make = setup_decoder
    if policy.startswith("fq"):
        drafter_type = type(make(2).owner.drafter)
        original_propose = drafter_type.propose

        def observed_propose(self, pending, depth, sampling, confidence):
            output = original_propose(self, pending, depth, sampling, confidence)
            if getattr(self, "observe", False):
                self.last_observation = {"claims": [.8] * len(output),
                                         "candidates": [list(range(64)) for _ in output]}
            return output

        monkeypatch.setattr(drafter_type, "propose", observed_propose)
    leader = make(2)
    cancelled = _stream([1, 2], 20, policy=policy)
    survivor = _stream([3, 4], 20, policy=policy)
    leader.admit(cancelled)
    leader.admit(survivor)
    if cause == "cancel":
        cancelled.cancelled = lambda: True
    else:
        def failed_callback():
            raise ValueError("client failed")
        cancelled.cancelled = failed_callback
    leader.finish(leader.round())
    _drain(leader)
    rounds = [message for message in leader.messages if message[0] in (mod.ROUND, mod.ROUND2)]
    assert rounds[0][0] == (mod.ROUND2 if policy.startswith("fq") else mod.ROUND)
    assert rounds[0][1:3] == [1, survivor.sid]

    follower = make(2)
    follower.rank = follower.owner.rank = 1
    finished = {}
    original_finish = follower._finish

    def finish(sids):
        finished.update((sid, list(follower.streams[sid].out)) for sid in sids)
        original_finish(sids)

    follower._finish = finish
    messages = iter(leader.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message 99"):
        follower.follow()
    assert finished[survivor.sid] == survivor.out
    assert len(finished[cancelled.sid]) == 1


def test_stepped_prefill_serves_short_prompt_before_long_prompt_and_mirrors_follower(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    original = mod.prefill

    def steps(e, prompt, sampling, *, mtp, drafter, resume, rows, slice_end=None):
        begin = len(resume.ids) if resume else 0
        while begin + rows() < len(prompt):
            end = begin + rows()
            e.st.set_pos(end)
            yield end
            begin = end
        return original(e, prompt, sampling, mtp, drafter, resume)

    monkeypatch.setattr(mod, "prefill_steps", steps)
    leader = make(2, limit=256)
    leader.owner.e.prefill_rows = 16
    long = _stream([1] * 80, 4)
    digest = bytes(range(32))
    long.images = [SimpleNamespace(digest=digest)]
    leader.owner.tower = object()
    short = _stream([2, 3], 4)
    leader.begin_admit(long)
    leader.prefill_step()
    leader.begin_admit(short)
    leader.prefill_step()
    assert short.out and not long.out
    assert [m for m in leader.messages if m[0] == mod.FILL] == [[mod.FILL, long.sid, 16],
                                                                [mod.FILL, short.sid, 2]]
    while leader.filling:
        leader.prefill_step()
    _drain(leader)
    assert sorted((len(s.ids), s.drafter_end) for s in leader.cache) == [(2, 2), (80, 80)]
    assert next(s.image_digests for s in leader.cache if len(s.ids) == 80) == (digest,)
    follower = make(2, limit=256)
    follower.owner.e.prefill_rows = 16
    follower.rank = follower.owner.rank = 1
    observed = {}
    original_finish = follower._finish

    def finish(sids):
        observed.update({sid: list(follower.streams[sid].out) for sid in sids})
        original_finish(sids)

    follower._finish = finish
    messages = iter(leader.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert observed == {long.sid: long.out, short.sid: short.out}
    assert next(s.image_digests for s in follower.cache if len(s.ids) == 80) == (digest,)
    assert follower.pool.available == follower.capacity


def test_uncontended_prefill_uses_full_chunks_then_finishes_shorter_big_prompt_first(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    original = mod.prefill
    monkeypatch.setattr(mod, "PREFILL_BUSY_ROWS", 4)
    monkeypatch.setattr(mod, "PREFILL_IDLE_ROWS", 8)

    def steps(e, prompt, sampling, *, mtp, drafter, resume, rows, slice_end=None):
        pos = len(resume.ids) if resume else 0
        while pos + rows() < len(prompt):
            pos += rows()
            e.st.set_pos(pos)
            yield pos
        return original(e, prompt, sampling, mtp, drafter, resume)

    monkeypatch.setattr(mod, "prefill_steps", steps)
    d = make(2, limit=256)
    d.owner.e.prefill_rows = 8
    shorter, longer = _stream([1] * 50, 1), _stream([2] * 70, 3)
    d.begin_admit(shorter)
    d.prefill_step()
    assert d.messages[-1] == [mod.FILL, shorter.sid, 8]
    d.prefill_step(queued=True)
    assert d.messages[-1] == [mod.FILL, shorter.sid, 12]
    d.begin_admit(longer)
    while not shorter.out:
        d.prefill_step()
    assert longer.fill_pos == 0
    assert all(msg[1] == shorter.sid for msg in d.messages if msg[0] == mod.FILL)
    d.finish([shorter])
    d.prefill_step()
    assert d.messages[-1] == [mod.FILL, longer.sid, 8]
    while d.filling:
        d.prefill_step()
    _drain(d)
    follower = make(2, limit=256)
    follower.owner.e.prefill_rows = 8
    follower.rank = follower.owner.rank = 1
    observed = {}
    original_finish = follower._finish

    def finish(sids):
        observed.update({sid: list(follower.streams[sid].out) for sid in sids})
        original_finish(sids)

    follower._finish = finish
    messages = iter(d.messages + [[99]])
    follower.share = lambda _: next(messages)
    with pytest.raises(RuntimeError, match="unknown GLM worker message"):
        follower.follow()
    assert observed == {shorter.sid: shorter.out, longer.sid: longer.out}


def test_real_prefill_stepper_commits_bounded_chunks_and_keeps_taps(setup_decoder, monkeypatch):
    from tensorfold.families.glm5_next.cuda import decode

    seen = []
    st = SimpleNamespace(pos=0)
    b = SimpleNamespace(overlay=None, fnormed=torch.zeros(8, 2))
    drafter = SimpleNamespace(reset=lambda: None, add_taps=lambda taps: seen.append(("taps", len(taps))))
    e = SimpleNamespace(w=SimpleNamespace(mtp=None), st=st, pbuf=b, prefill_rows=8,
                        reset=lambda: setattr(st, "pos", 0),
                        overlay=lambda first, n: (first, n),
                        tap_rows=lambda n, _b: torch.zeros(n, 4),
                        sample=lambda logits, positions, sampling: [int(logits[-1, 0])])

    def stage(w, state, buf, chunk):
        seen.append(("chunk", state.pos, len(chunk), buf.overlay))
        return len(chunk)

    def compute(w, state, buf, n, **kw):
        buf.fnormed[:n].fill_(state.pos + n)
        return torch.full((n, 1), state.pos + n)

    monkeypatch.setattr(decode, "stage", stage)
    monkeypatch.setattr(decode, "compute", compute)
    monkeypatch.setattr(decode, "chunks_for", lambda st, n: 1)
    monkeypatch.setattr(decode, "commit", lambda w, st, b, n, keep: setattr(st, "pos", st.pos + keep))
    steps = decode.prefill_steps(e, list(range(7)), None, mtp=False, drafter=drafter, rows=3)
    assert next(steps) == 3 and st.pos == 3
    assert next(steps) == 6 and st.pos == 6
    with pytest.raises(StopIteration) as done:
        next(steps)
    assert done.value.value == 7 and st.pos == 7
    assert [x[2] for x in seen if x[0] == "chunk"] == [3, 3, 1]
    assert [x[3] for x in seen if x[0] == "chunk"] == [(0, 3), (3, 3), (6, 1)]
    assert [x[1] for x in seen if x[0] == "taps"] == [3, 3, 1]
    assert b.overlay is None
    seen.clear()
    size = [3]
    dynamic = decode.prefill_steps(e, list(range(7)), None, mtp=False, drafter=drafter,
                                   rows=lambda: size[0])
    assert next(dynamic) == 3
    size[0] = 2
    assert next(dynamic) == 5
    with pytest.raises(StopIteration) as resumed:
        next(dynamic)
    assert resumed.value.value == 7 and st.pos == 7
    assert [x[2] for x in seen if x[0] == "chunk"] == [3, 2, 2]


def test_cancelled_stepped_prefill_releases_span_without_forward(setup_decoder, monkeypatch):
    mod, make = setup_decoder
    def steps(*a, **kw):
        pytest.fail("cancelled prefill ran")
        yield

    monkeypatch.setattr(mod, "prefill_steps", steps)
    d = make(2)
    d.owner.e.prefill_rows = 16
    s = _stream([1] * 80, 4)
    s.cancelled = lambda: True
    d.begin_admit(s)
    assert d.prefill_step() == [s]
    d.finish([s])
    assert not s.out and d.pool.available == d.capacity
    assert d.messages[-1] == [mod.DONE, 1, s.sid]


@pytest.mark.parametrize("load", [1, 8])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("tap_every", [0, 1])
def test_observer_preserves_warm_cache_drafts_paths_and_wire_messages(
        setup_decoder, monkeypatch, tmp_path, load, sampled, tap_every):
    """Includes a restored append, cancellation and slot reuse, not just cold tokens."""
    _, make = setup_decoder
    sampling = Sampling(731, .7, 0, 1.) if sampled else None
    def run():
        d = make(8)
        outputs, snapshots = [], []
        for extension in ([], [7, 9]):
            streams = [_stream([i + 1, 3, 5, *extension], 12, sampling=sampling,
                               policy="f7" if i % 2 else "fc7:0.3") for i in range(load)]
            for s in streams:
                d.admit(s)
            if not extension and load == 8:
                streams[-1].cancelled = lambda: True
            _drain(d)
            assert all(s.error is None for s in streams)
            for s in streams:
                outputs.append((s.out[:], s.cached, s.depths[:], s.keeps[:], s.drafter.context_end,
                                s.drafter.proposals[:]))
            snapshots.extend((c.ids[:], c.rec.clone(), c.conv.clone(), [r.clone() for r in c.rows]) for c in d.cache)
        d.observer.close()
        return outputs, snapshots, d.owner.e.replays, d.messages
    plain = run()
    monkeypatch.setenv("TF_GLM_DECODE_OBSERVE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_GLM_DECODE_TAP_EVERY", str(tap_every))
    shown = run()
    assert shown[0] == plain[0]
    assert shown[2:] == plain[2:]
    assert len(shown[1]) == len(plain[1])
    for a, b in zip(plain[1], shown[1]):
        assert a[0] == b[0]
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        assert len(a[3]) == len(b[3]) and all(torch.equal(x, y) for x, y in zip(a[3], b[3]))
    assert any(row[1] > 0 for row in shown[0])
