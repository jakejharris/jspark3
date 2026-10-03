"""E1's real overlap control on CPU rank meshes; arithmetic kernels are row-local stand-ins."""

import copy
import importlib
import runpy
import threading
from collections import deque
from types import FunctionType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from test_cuda_geometry import allocations  # noqa: F401


class Mesh:
    def __init__(self):
        self.barrier = threading.Barrier(3, timeout=5)
        self.box = {}


class Comm:
    def __init__(self, mesh, rank):
        self.mesh, self.rank = mesh, rank

    def all_gather(self, send, recv):
        m = self.mesh
        m.box[self.rank] = send.clone()
        m.barrier.wait()
        for rank in range(3):
            recv[rank * send.numel():(rank + 1) * send.numel()].copy_(m.box[rank])
        m.barrier.wait()

    def exchange(self, sends, recvs):
        m = self.mesh
        grouped = {}
        for peer, tensor in sends:
            grouped.setdefault(peer, []).append(tensor.clone())
        m.box[self.rank] = grouped
        m.barrier.wait()
        offsets = {}
        for peer, tensor in recvs:
            at = offsets.get(peer, 0)
            value = m.box[peer][self.rank][at]
            assert value.shape == tensor.shape
            tensor.copy_(value)
            offsets[peer] = at + 1
        m.barrier.wait()


class DeferredSide:
    """One side queue per rank; only join makes its queued writes visible to the main stream."""

    local = threading.local()

    def __init__(self, device):
        if not hasattr(self.local, "queue"):
            self.local.queue = deque()
        self.target = None

    def tail(self, fn):
        # CUDA queues writes now; it does not re-evaluate the generator's
        # Python loop variables at join. Freeze those cells for this host fake.
        cell = lambda value: (lambda: value).__closure__[0]
        fn = FunctionType(fn.__code__, fn.__globals__, closure=tuple(cell(c.cell_contents) for c in fn.__closure__))
        self.target = [fn, False]
        self.local.queue.append(self.target)

    def join(self):
        while self.target is not None and not self.target[1]:
            entry = self.local.queue.popleft()
            entry[0]()
            entry[1] = True


@pytest.fixture
def pipeline(monkeypatch, allocations):
    f = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    monkeypatch.setattr(f.seqpar, "MIN_ROWS", 3)
    monkeypatch.setattr(f.seqpar, "OVERLAP", True)
    monkeypatch.setattr(f.seqpar, "OVERLAP_MIN_ROWS", 2)
    monkeypatch.setattr(f.seqpar, "Side", DeferredSide)
    monkeypatch.setattr(f.prof, "ENABLED", False)
    monkeypatch.setattr(f.tp3_probe, "recorder", None)
    monkeypatch.setattr(f.tp3_probe, "REDUCE", "gather")
    monkeypatch.setattr(f.p1, "DEAD_WORK", True)

    def pre(w, h, norm, b, lo, hi):
        b.normed[lo:hi].copy_((b.x[lo:hi].float() + h).to(torch.bfloat16))
        b.xs[lo:hi].copy_(b.normed[lo:hi].float().sum(1, keepdim=True))
        b.post[lo:hi].fill_(0.25)
        b.comb[lo:hi].fill_(h)
        f.seqpar.share(w, b)

    def post(out, x, g, weights, comb):
        if isinstance(g, f.seqpar.Reduced):
            parts = [g.own if k == g.rank else g.peers[k] for k in range(3)]
        else:
            parts = list(g)
        total = parts[0].clone()
        for part in parts[1:]:
            total.add_(part)
        out.copy_((x.float() + total * weights + comb).to(torch.bfloat16))

    def partial(w, b, rows, layer, st=None):
        x = b.normed[:rows].float()
        assert torch.isfinite(x).all(), "block read before both halves published normed rows"
        if st is not None:
            cache = st.caches.setdefault(layer.index, torch.zeros(st.capacity, 8))
            cache[st.pos:st.pos + rows].copy_(x)
            prev = torch.cat([torch.zeros(1, 8), cache])[st.pos:st.pos + rows]
            x = x + prev
        part = x * ((w.rank + 1) / 8) + b.xs[:rows] / 64
        part[:, 0] = (2.0 ** 24, 1., -(2.0 ** 24))[w.rank]  # Rank order affects output bits.
        b.part[:rows].copy_(part)
        return f.gather(w, b, rows)

    def attn(layer, w, segs, b, rows, *args):
        st = segs[0][0]
        w.calls.append(("attn", layer.index, rows, st.pos))
        return partial(w, b, rows, layer, st)

    def ffn(layer, w, b, rows):
        w.calls.append(("moe" if layer.mlp is None else "mlp", layer.index, rows))
        return partial(w, b, rows, layer)

    monkeypatch.setattr(f, "hc_pre", pre)
    monkeypatch.setattr(f.glue, "hc_post", post)
    monkeypatch.setattr(f.glue, "embed", lambda ids, table, d, s, out: out.copy_(table[ids.long()]))
    monkeypatch.setattr(f.glue, "stream_mean", lambda x, out: out.copy_(x))
    monkeypatch.setattr(f, "attn_block", attn)
    monkeypatch.setattr(f, "moe_block", ffn)
    monkeypatch.setattr(f, "mlp_block", ffn)
    monkeypatch.setattr(f, "finish_logits", lambda w, segs, b, rows, logits: b.hidden[rows - 1].clone())

    def run(rows, *, whole, sp, sliced=False, tap_start=0, skip_final_ffn=False, grouped=True, gs=64, swiglu=True):
        monkeypatch.setattr(f.p1, "EXPERT_WHOLE_PASS", whole)
        monkeypatch.setattr(f.seqpar, "ENABLED", sp)
        mesh, outputs, errors = Mesh(), [None] * 3, []
        layers = [SimpleNamespace(index=i, mlp=object() if i == 0 else None,
                                  moe=SimpleNamespace(shared=None, experts=SimpleNamespace(gs=gs, swiglu=swiglu)),
                                  attn_hc=(i + 1) / 8, ffn_hc=(i + 1) / 16,
                                  in_norm=None, post_norm=None) for i in range(3)]

        def rank(k):
            try:
                b = object.__new__(f.Buffers)
                b.world, b.rows, b.prefill, b.sp, b.defer = 3, rows, True, None, False
                b.plan = SimpleNamespace(prefill=grouped)
                b.ids = torch.arange(rows, dtype=torch.int32) % 31
                b.x = torch.full((rows, 8), float("nan"), dtype=torch.bfloat16)
                b.normed, b.hidden = torch.full_like(b.x, float("nan")), torch.full_like(b.x, float("nan"))
                b.xs, b.post, b.comb = [torch.full((rows, 1), float("nan")) for _ in range(3)]
                b.part, b.gath = torch.full((rows, 8), float("nan")), torch.full((3 * rows * 8,), float("nan"))
                b.taps, b.tap_at = [torch.full_like(b.x, 77) for _ in layers], {i: [i] for i in range(3)}
                b.overlay = (torch.tensor([0, rows - 1]), torch.full((2, 8), 0.5, dtype=torch.bfloat16))
                w = SimpleNamespace(rank=k, world=3, comm=Comm(mesh, k), layers=layers, calls=[],
                                    embed=(torch.arange(32 * 8).reshape(32, 8) / 256).to(torch.bfloat16),
                                    cfg=SimpleNamespace(hidden=8, streams=1))
                st = SimpleNamespace(pos=0, pos_dev=torch.tensor([0]), capacity=rows + 1, caches={})
                st.shifted = lambda n: shifted(st, n)
                for first, last in ([(0, 1), (1, 3)] if sliced else [(0, 3)]):
                    f.compute_streams(w, [(st, 0, rows)], b, eager=True, layer_start=first, layer_stop=last,
                                      logits=not skip_final_ffn, tap_start=tap_start, skip_final_ffn=skip_final_ffn)
                outputs[k] = (b.hidden.clone(), [t.clone() for t in b.taps], st.caches, w.calls)
                assert not DeferredSide.local.queue
            except BaseException as exc:
                errors.append(exc)
                mesh.barrier.abort()

        def shifted(st, n):
            view = copy.copy(st)
            view.pos, view.pos_dev = st.pos + n, st.pos_dev + n
            return view

        threads = [threading.Thread(target=rank, args=(k,), daemon=True) for k in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(8)
        assert not any(thread.is_alive() for thread in threads), "rank mesh stalled"
        if errors:
            raise errors[0]
        return outputs

    return f, run


def same_bytes(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows,sp", [(4, False), (5, True), (7, True), (128, True), (129, False), (4096, True)])
@pytest.mark.parametrize("sliced", [False, True])
def test_joined_moe_preserves_half_attention_outputs_and_rank_order(pipeline, rows, sp, sliced):
    _, run = pipeline
    before, after = run(rows, whole=False, sp=sp, sliced=sliced), run(rows, whole=True, sp=sp, sliced=sliced)
    for (hidden, taps, caches, calls), (got, got_taps, got_caches, got_calls) in zip(before, after):
        same_bytes(hidden, got)
        assert torch.isfinite(got).all()
        for a, b in zip(taps, got_taps):
            same_bytes(a, b)
        for layer, cache in caches.items():
            same_bytes(cache, got_caches[layer])
        assert [c for c in got_calls if c[0] != "moe"] == [c for c in calls if c[0] != "moe"]
        assert [c for c in got_calls if c[0] == "moe"] == [("moe", 1, rows), ("moe", 2, rows)]
        assert len([c for c in calls if c[0] == "moe"]) == 4


@pytest.mark.parametrize("tap_start,skip", [(3, False), (7, True)])
def test_live_taps_and_dead_final_ffn(pipeline, tap_start, skip):
    _, run = pipeline
    before = run(7, whole=False, sp=True, tap_start=tap_start, skip_final_ffn=skip)
    after = run(7, whole=True, sp=True, tap_start=tap_start, skip_final_ffn=skip)
    for a, b in zip(before, after):
        for left, right in zip(a[1], b[1]):
            same_bytes(left, right)
        if not skip:
            same_bytes(a[0], b[0])
        else:
            assert [c for c in b[3] if c[0] == "moe"] == [("moe", 1, 7)]


def test_missing_side_wait_is_detected(pipeline, monkeypatch):
    _, run = pipeline
    monkeypatch.setattr(DeferredSide, "join", lambda self: None)
    with pytest.raises(AssertionError, match="published"):
        run(7, whole=True, sp=True)


def test_missing_join_at_first_moe_changes_bytes(pipeline, monkeypatch):
    _, run = pipeline
    reference = run(129, whole=True, sp=True)
    real = DeferredSide.join

    def omit_moe_join(self):
        self.joins = getattr(self, "joins", 0) + 1
        # Both initial publication and the dense layer's waits still execute.
        # Only the first MoE's post-attention publication is read too early.
        if self.joins != 4:
            real(self)

    monkeypatch.setattr(DeferredSide, "join", omit_moe_join)
    broken = run(129, whole=True, sp=True)
    assert any(not torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
               for a, b in zip(reference, broken))


@pytest.mark.parametrize("kw", [{"grouped": False}, {"gs": 32}, {"swiglu": False}])
def test_unqualified_expert_paths_keep_original_halves(pipeline, kw):
    _, run = pipeline
    before = run(129, whole=False, sp=True, **kw)
    after = run(129, whole=True, sp=True, **kw)
    for a, b in zip(before, after):
        same_bytes(a[0], b[0])
        assert a[3] == b[3]


def test_mismatched_halves_refuse_whole_pass(pipeline):
    f, _ = pipeline
    for stages in ((object(), None), (object(), object())):
        with pytest.raises(RuntimeError, match="both overlap halves"):
            f.lockstep([iter([stage]) for stage in stages], lambda layer: pytest.fail("early whole pass"))


def test_whole_flag_strict_and_default_off(pipeline, monkeypatch):
    f, _ = pipeline
    monkeypatch.delenv("TF_GLM_EXPERT_WHOLE_PASS", raising=False)
    assert not runpy.run_path(f.p1.__file__)["EXPERT_WHOLE_PASS"]
    monkeypatch.setenv("TF_GLM_EXPERT_WHOLE_PASS", "1")
    assert runpy.run_path(f.p1.__file__)["EXPERT_WHOLE_PASS"]
    monkeypatch.setenv("TF_GLM_EXPERT_WHOLE_PASS", "yes")
    with pytest.raises(ValueError, match="EXPERT_WHOLE_PASS"):
        runpy.run_path(f.p1.__file__)
