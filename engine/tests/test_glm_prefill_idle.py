"""E1 row budgets through real chunk generators and rank-zero FILL commands."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from test_cuda_geometry import allocations  # noqa: F401
from test_glm_batched_host import setup_decoder, _stream
from test_glm_prefill_slices import sliced_decoder, drain_steps, replay


@pytest.fixture
def idle_decoder(setup_decoder, sliced_decoder, monkeypatch):
    mod, make = setup_decoder
    from tensorfold.families.glm5_next.cuda import decode

    monkeypatch.setattr(mod.p1, "ATTENTION_TILES", True)

    def build(idle=True):
        d = make(8, limit=40000, drafter=False)
        d.prefill_rows_idle, d.prefill_rows_busy = (8192 if idle else 0), 4096
        d.w.layers = list(range(6))
        n = 8192 if idle else 4096
        d.owner.e.prefill_rows = n
        d.owner.e.pbuf = SimpleNamespace(rows=n, x=torch.zeros(n, 2), fnormed=torch.zeros(n, 2), overlay=None,
                                         taps=[torch.zeros(n, 2), torch.zeros(n, 2)])
        original = d._engine

        def engine(*args):
            e = original(*args)
            e.reset = e.st.reset
            e.overlay = lambda first, rows: (first, rows)
            e.tap_rows = lambda rows, b: torch.cat([t[:rows] for t in b.taps], dim=1)
            e.sample = lambda logits, pos, sampling: decode.sample_rows(e.w, logits, pos, sampling)
            return e

        d._engine = engine
        return d

    return mod, build


def request(n, count=8, slices=0):
    s = _stream([3] * n, count, draft=False)
    s.prefill_slice_layers, s.cofill = slices, False
    return s


def test_idle_loaded_idle_transition_and_rank_replay(idle_decoder):
    mod, build = idle_decoder
    d = build()
    long = request(25003)
    d.begin_admit(long)
    workspace = d.owner.e.pbuf
    # Queued work is not decoding: a bounded idle chunk is still allowed.
    d.prefill_step(queued=True)
    assert long.fill_pos == 8192
    short = request(11, count=17)
    d.begin_admit(short)
    d.prefill_step()
    assert short.sid not in d.filling and not short.done
    d.prefill_step()
    assert long.fill_pos == 8192 + 4096
    while not short.done:
        d.finish(d.round())
    d.prefill_step()
    assert long.fill_pos == 8192 + 4096 + 8192
    assert d.owner.e.pbuf is workspace and workspace.rows == d.owner.e.prefill_rows == 8192
    drain_steps(d)
    assert all(r == {long.sid: long.out, short.sid: short.out} for r in replay(mod, build, d.messages))
    baseline = build(idle=False)
    plain = request(len(long.prompt))
    baseline.begin_admit(plain)
    drain_steps(baseline)
    assert plain.out == long.out
    for a, b in [(plain.st.kc[0], long.st.kc[0]), (plain.st.conv, long.st.conv)]:
        assert torch.equal(a[:len(long.prompt)].contiguous().view(torch.uint8),
                           b[:len(long.prompt)].contiguous().view(torch.uint8))


def test_load_change_keeps_partial_chunk_endpoint(idle_decoder):
    mod, build = idle_decoder
    d = build()
    short = request(11, count=3)
    d.begin_admit(short)
    d.prefill_step()
    long = request(16001, slices=2)
    d.begin_admit(long)
    d.prefill_step()
    assert (long.fill_stop, long.fill_layer, long.fill_pos) == (4096, 2, 0)
    while not short.done:
        d.finish(d.round())
    d.prefill_step()
    assert (long.fill_stop, long.fill_layer, long.fill_pos) == (4096, 4, 0)
    d.prefill_step()
    assert long.fill_pos == 4096
    d.prefill_step()
    assert (long.fill_stop, long.fill_layer, long.fill_pos) == (12288, 2, 4096)
    drain_steps(d)
    assert all(r == {long.sid: long.out, short.sid: short.out} for r in replay(mod, build, d.messages))


def test_checkpoint_caps_idle_chunk_before_it_is_announced(idle_decoder):
    mod, build = idle_decoder
    d = build()
    s = request(16001)
    d.begin_admit(s)
    # A session checkpoint may reduce an otherwise idle 8192-row assignment.
    d.sessions = SimpleNamespace(store=None, stop=lambda stream, begin, stop: min(stop, 4096))
    d.prefill_step()
    assert s.fill_pos == 4096 and d.messages[-1] == [mod.FILL, s.sid, 4096]
    d.sessions = None
    d.drop()


def test_express_and_reply_caps_survive_idle_capacity(idle_decoder):
    _, build = idle_decoder
    d = build()
    s = request(16001)
    d.begin_admit(s)
    small = SimpleNamespace(engine=SimpleNamespace(prefill_rows=2048))
    assert d._prefill_budget(small) == 2048
    s.reply_prefill = True
    d.reply_prefill_rows = 256
    d.prefill_step()
    assert s.fill_pos == 256
    s.reply_prefill = False
    d.drop()


@pytest.mark.parametrize("loaded", [False, True])
def test_big_cofill_combined_budget_obeys_load(idle_decoder, monkeypatch, loaded):
    mod, build = idle_decoder
    from tensorfold.families.glm5_next.cuda import cofill_big

    d = build()
    if loaded:
        short = request(11)
        d.begin_admit(short)
        d.prefill_step()
    monkeypatch.setattr(mod.p1, "COFILL_BIG", True)
    streams = [request(16001), request(16002)]
    for s in streams:
        s.cofill = True
        d.begin_admit(s)
    pieces = cofill_big.plan(d, streams[0])
    assert len(pieces) == 2
    assert sum(stop - d.streams[sid].fill_pos for sid, stop in pieces) == (4096 if loaded else 8192)
    d.drop()
