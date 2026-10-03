"""TF_GLM_SP (sequence-parallel prompt rows) over three ranks gives a prompt chunk's bytes unchanged: the same normed
rows into every block, the same rank-order sum (p0 + p1) + p2 rounded once to bf16 in ``glue.hc_post`` (the own partial
read in place), the same final rows, taps and logits on every rank. Host side: the real hyper-connection, norm,
embedding and stream-mean kernels in Triton's interpreter (a subprocess, TRITON_INTERPRET=1 before triton is imported)
on three rank threads that trade bytes through an in-process mesh; the tensor-parallel blocks are rank-dependent
stand-ins that read every row."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch  # noqa: E402

from tensorfold.cuda import comm as comm_mod  # noqa: E402
from tensorfold.families.glm5_next.cuda import seqpar  # noqa: E402


def test_shares_are_contiguous_balanced_and_cover_every_row():
    for world in (2, 3, 4):
        for R in range(world, 50):
            starts, counts = seqpar.split(R, world)
            assert sum(counts) == R and max(counts) - min(counts) <= 1 and min(counts) >= 1
            assert list(starts) == [sum(counts[:k]) for k in range(world)]
    assert seqpar.split(2048, 3) == ((0, 683, 1366), (683, 683, 682))
    assert seqpar.split(8192, 3) == ((0, 2731, 5462), (2731, 2731, 2730))


def test_only_prompt_chunks_over_several_ranks_get_a_plan(monkeypatch):
    w = SimpleNamespace(comm=object(), world=3, rank=1)
    chunk, rounds = SimpleNamespace(prefill=True), SimpleNamespace(prefill=False)
    monkeypatch.setattr(seqpar, "MIN_ROWS", 4)
    monkeypatch.setattr(seqpar, "ENABLED", False)
    assert seqpar.plan(w, chunk, 100) is None                               # default off: the all-gather path
    monkeypatch.setattr(seqpar, "ENABLED", True)
    sp = seqpar.plan(w, chunk, 100)
    assert sp.own == (34, 67) and sp.peers == [0, 2] and sp.rows(2) == slice(67, 100)
    assert seqpar.plan(w, rounds, 100) is None                              # decode rounds and the MTP's buffers
    assert seqpar.plan(w, chunk, 3) is None                                 # under TF_GLM_SP_MIN_ROWS
    assert seqpar.plan(SimpleNamespace(comm=None, world=1, rank=0), chunk, 100) is None
    monkeypatch.setattr(seqpar, "MIN_ROWS", 1)
    assert seqpar.plan(w, chunk, 2) is None                                 # a rank would own no row
    monkeypatch.setattr(seqpar.tp3_probe, "REDUCE", "nccl")
    assert seqpar.plan(w, chunk, 100) is None                               # the ring all-reduce probe keeps its path


def test_only_long_eager_prompt_chunks_over_several_ranks_are_pipelined(monkeypatch):
    w, chunk, one = SimpleNamespace(comm=object(), world=3, rank=0), SimpleNamespace(prefill=True), [object()]
    monkeypatch.setattr(seqpar, "OVERLAP_MIN_ROWS", 4)
    monkeypatch.setattr(seqpar, "OVERLAP", False)
    assert seqpar.halves(w, chunk, one, 8192, True) is None                 # default off: the chunk runs whole
    monkeypatch.setattr(seqpar, "OVERLAP", True)
    assert [seqpar.halves(w, chunk, one, R, True) for R in (8192, 5000, 129, 100, 9, 4)] == [4096, 2496, 64, 50, 4, 2]
    assert seqpar.halves(w, chunk, one, 3, True) is None                    # under TF_GLM_SP_OVERLAP_MIN_ROWS
    assert seqpar.halves(w, SimpleNamespace(prefill=False), one, 8192, True) is None     # decode and MTP buffers
    assert seqpar.halves(w, chunk, one, 8192, False) is None                 # captured graphs
    assert seqpar.halves(w, chunk, one * 2, 8192, True) is None              # several streams' rows
    assert seqpar.halves(SimpleNamespace(comm=None, world=1, rank=0), chunk, one, 8192, True) is None
    for name, value in (("REDUCE", "nccl"), ("recorder", object())):         # the TP3 probe and its recorder
        with monkeypatch.context() as m:
            m.setattr(seqpar.tp3_probe, name, value)
            assert seqpar.halves(w, chunk, one, 8192, True) is None
    monkeypatch.setattr(seqpar.prof, "ENABLED", True)                       # the profiler synchronizes every block
    assert seqpar.halves(w, chunk, one, 8192, True) is None


def test_side_work_runs_in_order_without_cuda():
    side, seen = seqpar.Side("cpu"), []
    side.tail(lambda: seen.append(1))
    side.join()
    assert seen == [1] and not side.cuda


def test_final_exchange_sends_hidden_rows_whole_and_tap_rows_from_the_live_suffix():
    calls = []
    w = SimpleNamespace(comm=SimpleNamespace(exchange=lambda s, r: calls.append((s, r))), world=3, rank=1)
    hidden, tap = torch.zeros((10, 4)), torch.zeros((10, 4))
    b = SimpleNamespace(sp=seqpar.Plan(1, *seqpar.split(10, 3)), hidden=hidden, taps=[tap])
    seqpar.finish(w, b, tap_start=5)                                         # shares 0:4, 4:7, 7:10
    rows = lambda ops: [(k, t.data_ptr(), t.shape[0]) for k, t in ops]      # noqa: E731
    (hs, hr), (ts, tr) = calls
    assert rows(hs) == [(0, hidden[4:].data_ptr(), 3), (2, hidden[4:].data_ptr(), 3)]
    assert rows(hr) == [(0, hidden.data_ptr(), 4), (2, hidden[7:].data_ptr(), 3)]
    assert rows(ts) == [(0, tap[5:].data_ptr(), 2), (2, tap[5:].data_ptr(), 2)]
    assert [(k, n) for k, _, n in rows(tr)] == [(0, 0), (2, 3)] and tr[1][1].data_ptr() == tap[7:].data_ptr()


def test_exchange_is_one_group_in_list_order_and_skips_empty_tensors(monkeypatch):
    log = []

    class Lib:
        def ncclGroupStart(self):
            log.append("start")
            return 0

        def ncclGroupEnd(self):
            log.append("end")
            return 0

        def ncclSend(self, ptr, n, dt, peer, comm, stream):
            log.append(("send", ptr, n, dt, peer))
            return 0

        def ncclRecv(self, ptr, n, dt, peer, comm, stream):
            log.append(("recv", ptr, n, dt, peer))
            return 0

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    nccl = object.__new__(comm_mod.NCCL)
    nccl.lib, nccl.comm, nccl.rank, nccl.world = Lib(), None, 1, 3
    a, h = torch.zeros((6, 4), dtype=torch.float32), torch.zeros((6, 8), dtype=torch.bfloat16)
    nccl.exchange([(0, a[2:4]), (0, h[2:4]), (2, a[2:4]), (2, h[:0])], [(0, a[:2]), (2, a[4:])])
    assert log == ["start", ("send", a[2:].data_ptr(), 8, 7, 0), ("send", h[2:].data_ptr(), 16, 9, 0),
                   ("send", a[2:].data_ptr(), 8, 7, 2), ("recv", a.data_ptr(), 8, 7, 0),
                   ("recv", a[4:].data_ptr(), 8, 7, 2), "end"]
    log.clear()
    with pytest.raises(ValueError, match="contiguous"):
        nccl.exchange([(0, a[:, :2])], [])
    assert log == []                                                         # refused before any group opens


CHECK = r'''
import inspect, sys, tempfile, threading
from pathlib import Path

import torch

torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import glm5_tp3_fakes as fakes

D = fakes.D = 512                        # hc_pre's K blocks need 4 x 512 columns (16 blocks of 128)
fakes.TEXT = {**fakes.TEXT, "hidden_size": D, "hc_sinkhorn_iters": 3}     # fewer iterations: interpreter time
from tensorfold.families.glm5_next.cuda import forward, seqpar, weights

WORLD, ROWS, CAP = 3, 24, 64
folder = fakes.write(Path(tempfile.mkdtemp()) / "ckpt")
ws = [weights.load(str(folder), rank=k, world=WORLD, device="cpu") for k in range(WORLD)]


class Mesh:
    """Rendezvous for three rank threads; one runs at a time (Triton's interpreter is not thread-safe)."""

    def __init__(self, n):
        self.n, self.cv, self.count, self.gen, self.box = n, threading.Condition(), 0, 0, {}
        self.turn = threading.Lock()

    def wait(self):
        self.turn.release()
        with self.cv:
            gen = self.gen
            self.count += 1
            if self.count == self.n:
                self.count, self.gen = 0, self.gen + 1
                self.cv.notify_all()
            else:
                while gen == self.gen:
                    self.cv.wait()
        self.turn.acquire()


class Comm:
    """comm.NCCL's two calls the forward makes, with the received bytes counted."""

    def __init__(self, mesh, rank):
        self.mesh, self.rank, self.world, self.got = mesh, rank, mesh.n, 0

    def all_gather(self, send, recv):
        m, n = self.mesh, send.numel()
        m.box[("ag", self.rank)] = send.clone()
        m.wait()
        for k in range(self.world):
            recv[k * n:(k + 1) * n].copy_(m.box[("ag", k)])
        self.got += (self.world - 1) * n * send.element_size()
        m.wait()

    def exchange(self, sends, recvs):
        assert all(t.is_contiguous() for _, t in sends + recvs)
        m = self.mesh
        for peer, t in sends:
            m.box.setdefault((self.rank, peer), []).append(t.clone())
        m.wait()
        for peer, t in recvs:
            got = m.box[(peer, self.rank)].pop(0)
            assert got.shape == t.shape and got.dtype == t.dtype, (got.shape, t.shape, got.dtype, t.dtype)
            t.copy_(got)
            self.got += t.numel() * t.element_size()
        m.wait()


CACHES = {}


def partial(w, b, R, key, at=None):
    """A block's fp32 partial on this rank: reads every row's normed values and sums, and on every 8th column holds
    2^24, 1, -2^24 on ranks 0, 1, 2: only (p0 + p1) + p2 gives 0 there. ``at`` (attention: a layer's cache and the
    chunk's first position): each row is written to the cache at its position and also reads the row before it there,
    so a chunk's bits do not depend on where it was cut, and a later piece needs the earlier one's writes."""

    g = torch.Generator().manual_seed(1000 * key + w.rank)
    m = torch.randn(D, D, generator=g) * 0.05
    x = b.normed[:R].float()
    if at is not None:
        cache, pos = at
        cache[pos:pos + R] = x
        prev = torch.cat([torch.zeros(1, D), cache])[pos:pos + R]
        x = x + prev + 1e-3 * torch.arange(pos, pos + R, dtype=torch.float32)[:, None]
    p = torch.stack([r @ m for r in x]) + 1e-3 * b.xs[:R].sum(1, keepdim=True)   # row by row, as prefill_matmul
    p[:, ::8] = (2.0 ** 24, 1.0, -(2.0 ** 24))[w.rank] * 0.25
    b.part[:R].copy_(p)
    return forward.gather(w, b, R)


def cache(key, store):                   # NaN until written: a read before the write it needs shows in every byte
    return CACHES.setdefault((key, id(store)), torch.full((CAP, D), float("nan")))


def dsa(layer, w, caches, b, R, nch, sparse_np=None):
    kc, _, pos_dev, _, host_pos, _, _ = caches[0]
    assert int(pos_dev[0]) == host_pos, (int(pos_dev[0]), host_pos)
    return partial(w, b, R, 10 * layer.index + 2, (cache(10 * layer.index + 2, kc), host_pos))


HEAD = torch.randn(D, 1024, generator=torch.Generator().manual_seed(5))
forward.kda_block = lambda layer, w, segs, b, R: partial(w, b, R, 10 * layer.index + 1,
                                                        (cache(10 * layer.index + 1, segs[0][0].conv), segs[0][0].pos))
forward.dsa_block = dsa
forward.mlp_block = lambda layer, w, b, R: partial(w, b, R, 10 * layer.index + 3)
forward.moe_block = lambda layer, w, b, R: partial(w, b, R, 10 * layer.index + 4)
forward.mm = lambda b, x, q, xs, out, f32=False: out.copy_(x.float() @ HEAD[:, :out.shape[1]])
bufs = [forward.Buffers(w, ROWS, CAP, prefill=True) for w in ws]
for b in bufs:
    b.set_taps((0, 1, 1), D)
states = [forward.State(w, CAP, ROWS) for w in ws]
emb = torch.randn(2, D, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32).clone()


def run(R, sp, slices=None, overlap=False, **dead):
    """One prompt chunk of R rows on three rank threads (in sliced prefill layer ranges when ``slices``; as two pipelined halves
    with ``overlap``); every scratch buffer and cache starts as rank-dependent NaN/garbage."""

    seqpar.ENABLED, seqpar.MIN_ROWS = sp, 1
    seqpar.OVERLAP, seqpar.OVERLAP_MIN_ROWS = overlap, 2
    CACHES.clear()
    mesh = Mesh(WORLD)
    ids = torch.randint(0, fakes.V, (R,), generator=torch.Generator().manual_seed(R))
    out, errors = [None] * WORLD, []

    def rank(k):
        mesh.turn.acquire()
        try:
            w, b = ws[k], bufs[k]
            w.comm = Comm(mesh, k)
            for t in (b.x, b.normed, b.hidden, b.fnormed, *b.taps):
                t.fill_(float("nan"))
            for t in (b.xs, b.post, b.comb, b.part, b.gath):
                t.fill_(1e30 * (k + 1))
            b.ids[:R].copy_(ids)
            b.overlay = (torch.tensor([0, R - 1]), emb)
            for a, z in slices or [(None, None)]:
                more = {} if a is None else {"layer_start": a, "layer_stop": z}
                logits = forward.compute_streams(w, [(states[k], 0, R)], b, eager=True,
                                                 logits=not dead.get("skip_final_ffn"), **more, **dead)
                assert b.sp is None
            lo, hi = seqpar.split(R, WORLD)[0][k], sum(seqpar.split(R, WORLD)[1][:k + 1])
            out[k] = {"hidden": bits(b.hidden[:R]), "fnormed": bits(b.fnormed[:R]),
                      "logits": bits(logits) if logits is not None else torch.zeros(0),
                      "taps": [bits(t[:R]) for t in b.taps], "mine": bits(b.x[lo:hi]), "got": w.comm.got}
        except BaseException as exc:
            errors.append(exc)
            raise
        finally:
            mesh.turn.release()

    threads = [threading.Thread(target=rank, args=(k,)) for k in range(WORLD)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not errors, errors
    return out


def same(a, b, keys=("hidden", "fnormed", "logits", "mine")):
    return all(torch.equal(a[k], b[k]) for k in keys) and all(torch.equal(x, y) for x, y in zip(a["taps"], b["taps"]))


KEYS = ("hidden", "fnormed", "logits")     # halves own other residual rows: b.x is internal to the pass
MODE = sys.argv[2]


if MODE == "overlap":
    b = bufs[0]                          # every tensor is a row view or scratch only the main stream's blocks use
    names = {k for k, v in vars(b).items() if isinstance(v, torch.Tensor) or hasattr(v, "__dict__")}
    known = set(b.ROWS) | set(b.SHARED) | {"kproj", "gath", "sp", "overlay"}
    assert names <= known, names - known
    v = b.half(5, 9)
    for name in b.ROWS:
        if getattr(b, name, None) is not None:
            assert getattr(v, name).data_ptr() == getattr(b, name)[5].data_ptr() and len(getattr(v, name)) == 4, name
    assert v.gath.data_ptr() == b.gath[WORLD * 5 * D].data_ptr() and v.kproj.data_ptr() == b.kproj[0, 5].data_ptr()
    assert v.defer and not b.defer and v.rows == 4 and v.taps[0].data_ptr() == b.taps[0][5].data_ptr()
    # TF_GLM_SP_OVERLAP: two pipelined halves (each with its own plan, or the all-gather) equal the chunk run whole
    refs = {}
    for R, sp in ((4, False), (5, True), (7, True), (8, True)):     # halves 2+2 (all-gather), 2+3 (mixed), 3+4, 4+4
        refs[R] = ref = run(R, False)
        got = run(R, sp, overlap=True)
        for k in range(WORLD):
            assert same(ref[k], got[k], KEYS), ("rank", k, "rows", R, "sp", sp)
        print(R, "sp" if sp else "all-gather", "halves: equal; bytes per rank", ref[0]["got"], got[0]["got"])
    R, ref = 8, refs[8]
    got = run(R, True, slices=[(0, 1), (1, 2)], overlap=True)
    assert all(same(ref[k], got[k], KEYS) for k in range(WORLD)), "halves in layer slices"
    from tensorfold.families.glm5_next.cuda import prefill_options

    prefill_options.DEAD_WORK = True
    for dead in ({"tap_start": 5}, {"tap_start": 2}, {"tap_start": R, "skip_final_ffn": True}):   # B's, A's, none
        assert all(same(a, b, KEYS) for a, b in zip(run(R, False, **dead), run(R, True, overlap=True, **dead))), dead
    prefill_options.DEAD_WORK = False
    print("halves with layer slices and dead work: equal")
    lockstep, shifted = forward.lockstep, forward.State.shifted
    for name, patch in (("order", lambda: setattr(forward, "lockstep", lambda hs: lockstep(hs[::-1]))),
                        ("shift", lambda: setattr(forward.State, "shifted", lambda st, k: st))):
        patch()
        try:
            bad = run(R, True, overlap=True)
        finally:
            forward.lockstep, forward.State.shifted = lockstep, shifted
        assert not same(ref[0], bad[0], KEYS), name
        print("negative control caught:", name)
    print("OK")
    sys.exit(0)

for R in (3, 4, 5, 16):
    ref, got = run(R, False), run(R, True)
    for k in range(WORLD):
        assert same(ref[k], got[k]), ("rank", k, "rows", R)
        assert same(got[0], got[k], ("hidden", "fnormed")), ("ranks disagree", k, R)     # logits: own vocab share
    print(R, "bytes per rank: all-gather", ref[0]["got"], "sp", [g["got"] for g in got])

# Prefill layer slices (where the branch has them): the same shares at every slice, the final exchange at the last
R = 16
ref = run(R, False)
if "layer_stop" in inspect.signature(forward.compute_streams).parameters:
    got = run(R, True, slices=[(0, 1), (1, 2)])
    assert all(same(ref[k], got[k]) for k in range(WORLD)), "layer slices"
    print("layer slices: equal")

# Prefill dead-work skip (where the branch has it): taps only from tap_start on, or no final FFN at all
if "tap_start" in inspect.signature(forward.compute_streams).parameters:
    from tensorfold.families.glm5_next.cuda import prefill_options

    prefill_options.DEAD_WORK = True
    for dead in ({"tap_start": 9}, {"tap_start": 3}, {"tap_start": R, "skip_final_ffn": True}):
        assert all(same(a, b) for a, b in zip(run(R, False, **dead), run(R, True, **dead))), dead
    prefill_options.DEAD_WORK = False
    print("dead work: equal")

# negative controls: each must change bytes the gate compares
reduce, share, finish = seqpar.reduce, seqpar.share, seqpar.finish


def rotated(w, b, R):                     # NCCL's ring reduce-scatter grouping: (p2 + p0) + p1 on rank 0's rows
    r = reduce(w, b, R)
    full = r.peers.clone()
    full[r.rank] = r.own
    return full.roll(1, 0)


def unowned(w, b, R):                     # the view alone: its own slot was never written (no copy), so it is garbage
    return reduce(w, b, R).peers


for name, fake in (("reduce", rotated), ("reduce", unowned), ("share", lambda w, b: None),
                   ("finish", lambda w, b, *tap: None)):
    setattr(seqpar, name, fake)
    try:
        bad = run(R, True)
    finally:
        seqpar.reduce, seqpar.share, seqpar.finish = reduce, share, finish
    assert not same(ref[0], bad[0]), name
    print("negative control caught:", name, getattr(fake, "__name__", ""))
print("OK")
'''


def _check(mode):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1",
           "PYTHONPATH": f"{root / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    run = subprocess.run([sys.executable, "-c", CHECK, str(root / "tests"), mode], env=env, capture_output=True,
                         text=True, timeout=1200)
    assert run.returncode == 0 and run.stdout.strip().endswith("OK"), run.stdout[-3000:] + run.stderr[-6000:]


def test_three_ranks_split_rows_and_keep_every_byte():
    _check("sp")


def test_pipelined_halves_keep_every_byte():
    _check("overlap")
