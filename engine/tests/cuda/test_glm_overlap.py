"""TF_GLM_SP_OVERLAP on one GPU: a prompt chunk's two pipelined halves on real streams (the big blocks on the main
stream, each half's tail on the side stream) give exactly the bytes of the same halves run in order, with or without
TF_GLM_SP, while every side-stream call is slowed so that a missing wait would read rows not yet written.

One rank of three: the fake comm puts this rank's bytes where NCCL would and fixed patterns where the peers' would go,
so only the streams differ between the two runs."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
WORLD, ROWS, CAP, SPIN = 3, 512, 1024, 2_000_000        # SPIN: GPU cycles each side-stream call and block waits


class Comm:
    """comm.NCCL's calls on the current stream: own bytes copied, peers' bytes from fixed patterns, each slowed."""

    def __init__(self):
        self.patterns = {}

    def pattern(self, t, peer):
        key = (t.shape, t.dtype, peer)
        if key not in self.patterns:
            g = torch.Generator(device="cuda").manual_seed(hash(key) % 2**31)
            self.patterns[key] = (torch.randn(t.shape, generator=g, device="cuda") * 0.1).to(t.dtype)
        return self.patterns[key]

    def all_gather(self, send, recv):
        torch.cuda._sleep(SPIN)
        n = send.numel()
        recv[:n].copy_(send)
        for k in range(1, WORLD):
            recv[k * n:(k + 1) * n].copy_(self.pattern(send, k))

    def exchange(self, sends, recvs):
        torch.cuda._sleep(SPIN)
        for peer, t in recvs:
            t.copy_(self.pattern(t, peer))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import glm5_tp3_fakes as fakes

    fakes.TEXT = {**fakes.TEXT, "hidden_size": 512, "hc_sinkhorn_iters": 3}
    fakes.D = 512
    from tensorfold.families.glm5_next.cuda import weights

    folder = fakes.write(tmp_path_factory.mktemp("ckpt") / "ckpt")
    w = weights.load(str(folder), rank=0, world=WORLD, device="cuda")
    w.comm = Comm()
    return w, fakes


def _blocks(monkeypatch, forward, D):
    """Stand-in blocks on the main stream, slowed: attention writes each row to a layer cache at its position and reads
    the row before it there (so the second half needs the first half's writes), the FFN is row-local."""

    caches, mats = {}, {}

    def partial(w, b, R, key, at=None):
        torch.cuda._sleep(SPIN)
        if key not in mats:
            g = torch.Generator(device="cuda").manual_seed(key)
            mats[key] = torch.randn(D, D, generator=g, device="cuda") * 0.05
        x = b.normed[:R].float()
        if at is not None:
            store, pos = at
            cache = caches.setdefault((key, id(store)), torch.full((CAP, D), float("nan"), device="cuda"))
            cache[pos:pos + R] = x
            x = x + torch.cat([torch.zeros(1, D, device="cuda"), cache])[pos:pos + R]
        b.part[:R].copy_(x @ mats[key] + 1e-3 * b.xs[:R].sum(1, keepdim=True))
        return forward.gather(w, b, R)

    monkeypatch.setattr(forward, "kda_block", lambda layer, w, segs, b, R: partial(
        w, b, R, 10 * layer.index + 1, (segs[0][0].conv, segs[0][0].pos)))
    monkeypatch.setattr(forward, "dsa_block", lambda layer, w, caches_, b, R, nch, sparse_np=None: partial(
        w, b, R, 10 * layer.index + 2, (caches_[0][0], caches_[0][4])))
    monkeypatch.setattr(forward, "mlp_block", lambda layer, w, b, R: partial(w, b, R, 10 * layer.index + 3))
    monkeypatch.setattr(forward, "moe_block", lambda layer, w, b, R: partial(w, b, R, 10 * layer.index + 4))
    head = torch.randn(D, 1024, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda")
    monkeypatch.setattr(forward, "mm", lambda b, x, q, xs, out, f32=False: out.copy_(x.float() @ head[:, :out.size(1)]))
    return caches


def _run(model, monkeypatch, R, sp, streams, *, whole=False, cut=None, **kw):
    from tensorfold.families.glm5_next.cuda import forward, prefill_options, seqpar

    w, fakes = model
    D = w.cfg.hidden
    caches = _blocks(monkeypatch, forward, D)
    monkeypatch.setattr(prefill_options, "EXPERT_WHOLE_PASS", whole)
    monkeypatch.setattr(seqpar, "ENABLED", sp)
    monkeypatch.setattr(seqpar, "MIN_ROWS", 3)
    monkeypatch.setattr(seqpar, "OVERLAP", True)
    monkeypatch.setattr(seqpar, "OVERLAP_MIN_ROWS", 2)
    if not streams:
        real = seqpar.Side.__init__

        def in_order(self, device):
            real(self, device)
            self.cuda = False

        monkeypatch.setattr(seqpar.Side, "__init__", in_order)
    b = forward.Buffers(w, ROWS, CAP, prefill=True)
    b.set_taps((0, 1, 1), D)
    st = forward.State(w, CAP, ROWS)
    for t in (b.x, b.normed, b.hidden, b.fnormed, *b.taps):
        t.fill_(float("nan"))
    for t in (b.xs, b.post, b.comb, b.part, b.gath):
        t.fill_(1e30)
    b.ids[:R].copy_(torch.randint(0, fakes.V, (R,), generator=torch.Generator().manual_seed(R)))
    caches.clear()
    if cut is None:
        logits = forward.compute_streams(w, [(st, 0, R)], b, eager=True, **kw)
    else:
        assert forward.compute_streams(w, [(st, 0, R)], b, eager=True, layer_stop=cut, **kw) is None
        logits = forward.compute_streams(w, [(st, 0, R)], b, eager=True,
                                         layer_start=cut, layer_stop=len(w.layers), **kw)
    torch.cuda.synchronize()
    bits = lambda t: t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32).clone()  # noqa: E731
    return [bits(b.hidden[:R]), bits(b.fnormed[:R]), bits(logits)] + [bits(t[:R]) for t in b.taps]


@cuda
@pytest.mark.parametrize("sp", [False, True], ids=["all-gather", "sp"])
@pytest.mark.parametrize("R", [7, 300, 512])
def test_side_stream_halves_equal_the_same_halves_in_order(model, monkeypatch, R, sp):
    with monkeypatch.context() as m:
        ref = _run(model, m, R, sp, streams=False)
    for _ in range(3):
        with monkeypatch.context() as m:
            got = _run(model, m, R, sp, streams=True)
        assert all(torch.equal(a, b) for a, b in zip(ref, got)), (R, sp)


@cuda
@pytest.mark.parametrize("whole", [False, True])
def test_a_missing_wait_reads_rows_the_side_stream_has_not_written(model, monkeypatch, whole):
    from tensorfold.families.glm5_next.cuda import seqpar

    with monkeypatch.context() as m:
        ref = _run(model, m, 512, True, streams=False, whole=whole)
    with monkeypatch.context() as m:
        m.setattr(seqpar.Side, "join", lambda self: None)
        bad = _run(model, m, 512, True, streams=True, whole=whole)
    assert not all(torch.equal(a, b) for a, b in zip(ref, bad))


@cuda
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("R", [7, 300, 512])
@pytest.mark.parametrize("cut", [None, 1])
def test_e1_whole_experts_preserve_side_stream_order(model, monkeypatch, R, sp, cut):
    with monkeypatch.context() as m:
        ref = _run(model, m, R, sp, streams=False, whole=True)
    for _ in range(3):
        with monkeypatch.context() as m:
            got = _run(model, m, R, sp, streams=True, whole=True, cut=cut)
        assert all(torch.equal(a, b) for a, b in zip(ref, got)), (R, sp, cut)
