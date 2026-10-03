"""L2 prefetch dispatch/lifetime gates on CPU; device bits and timing have separate gates."""

from contextlib import contextmanager, nullcontext
import runpy
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.glm5_next.cuda import forward, l2_prefetch as l2


def test_default_off_and_bad_settings(monkeypatch):
    monkeypatch.delenv("TF_GLM_L2_PREFETCH", raising=False)
    monkeypatch.delenv("TF_GLM_L2_PREFETCH_BYTES", raising=False)
    values = runpy.run_path(l2.__file__)
    assert values["ENABLED"] is False and values["MAX_BYTES"] == 4 * 2**20
    for name, value in (("TF_GLM_L2_PREFETCH", "yes"), ("TF_GLM_L2_PREFETCH_BYTES", "31"),
                        ("TF_GLM_L2_PREFETCH_BYTES", str(16 * 2**20 + 1))):
        with monkeypatch.context() as m:
            m.setenv(name, value)
            with pytest.raises(ValueError):
                runpy.run_path(l2.__file__)


@pytest.mark.parametrize("enabled,prefill,world", [(False, False, 3), (False, True, 3), (True, False, 1)])
def test_excluded_paths_do_not_construct_a_stream(monkeypatch, enabled, prefill, world):
    monkeypatch.setattr(l2, "ENABLED", enabled)
    monkeypatch.setattr(l2, "Prefetch", lambda *_: pytest.fail("unexpected stream allocation"))
    assert l2.make(NS(world=world), prefill=prefill) is None


def test_prefill_has_a_controller_when_enabled(monkeypatch):
    monkeypatch.setattr(l2, "ENABLED", True)
    controller = object()
    monkeypatch.setattr(l2, "Prefetch", lambda w: controller)
    assert l2.make(NS(world=3), prefill=True) is controller


def test_plan_only_names_existing_weights_and_the_next_layer():
    weights = [torch.arange(i + 1) for i in range(5)]
    layers = [NS(index=2, kind="kda", mlp=NS(gu=NS(weight=weights[0])),
                 kda=NS(proj=NS(weight=weights[1]))),
              NS(index=8, kind="dsa", mlp=None, moe=NS(router=weights[2]),
                 dsa=NS(proj=NS(weight=weights[3])))]
    plan = l2.targets(NS(layers=layers, head=NS(weight=weights[4])))
    assert plan[2][0] is weights[0] and plan[2][1] is weights[3]
    assert plan[8][0] is weights[2] and plan[8][1] is weights[4]


def fake_hint(monkeypatch, calls, *, launch_error=False):
    class Event:
        def __init__(self, name):
            self.name = name
        def record(self, stream):
            calls.append(("record", self.name, stream.name))
    class Stream:
        def __init__(self, name):
            self.name = name
        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))
    parent, side = Stream("parent"), Stream("prefetch")
    hint = object.__new__(l2.Prefetch)
    hint.device, hint.stream = "cuda:0", side
    hint.ready, hint.done = Event("ready"), Event("done")
    hint.pending = torch.tensor([19])
    hint.plan = {0: (hint.pending, hint.pending)}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: parent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    def launch(weight):
        calls.append(("hint", int(weight[0])))
        if launch_error:
            raise RuntimeError("launch failed")
    monkeypatch.setattr(l2, "tensor", launch)
    return hint


@pytest.mark.parametrize("transport_error", [False, True])
def test_gather_brackets_prefetch_and_keeps_rank_order(monkeypatch, transport_error):
    calls = []
    hint = fake_hint(monkeypatch, calls)
    def gather(send, receive):
        calls.append(("gather",))
        if transport_error:
            raise RuntimeError("transport failed")
        for rank in range(3):
            receive.view(3, -1)[rank].copy_(send + rank)
    b = NS(part=torch.arange(12, dtype=torch.float32).view(3, 4), gath=torch.empty(36), world=3,
           l2_prefetch=hint, sp=None, defer=False)
    w = NS(comm=NS(all_gather=gather))
    monkeypatch.setattr(forward.tp3_probe, "recorder", None)
    monkeypatch.setattr(forward.tp3_probe, "REDUCE", "ordered")
    if transport_error:
        with pytest.raises(RuntimeError, match="transport failed"):
            forward.gather(w, b, 3)
    else:
        result = forward.gather(w, b, 3)
        assert torch.equal(result, torch.stack([b.part + rank for rank in range(3)]))
    assert calls == [("record", "ready", "parent"), ("wait", "prefetch", "ready"),
                     ("hint", 19), ("record", "done", "prefetch"), ("gather",),
                     ("wait", "parent", "done")]
    assert hint.pending is None
    calls.clear()
    assert hint.start() is None  # MTP/drafter gathers do not inherit the last target hint.
    assert not calls


def test_compile_failure_also_rejoins(monkeypatch):
    calls = []
    hint = fake_hint(monkeypatch, calls, launch_error=True)
    with pytest.raises(RuntimeError, match="launch failed"):
        hint.start()
    assert calls[-2:] == [("record", "done", "prefetch"), ("wait", "parent", "done")]
    assert hint.pending is None


def test_layer_selects_two_hints_in_order(monkeypatch):
    calls = []
    hint = NS(select=lambda layer, phase: calls.append(("select", layer, phase)))
    h = NS(fn=None, base=None, scale=None)
    layer = NS(index=7, kind="kda", attn_hc=h, ffn_hc=h, in_norm=None, post_norm=None, mlp=object())
    b = NS(x=[0], normed=[0], xs=[0], post=[0], comb=[0], hcpart=[0], l2_prefetch=hint, sp=None)
    w = NS(cfg=NS(eps=0, hc_eps=0, hc_iters=0))
    monkeypatch.setattr(forward.glue, "hc_pre", lambda *a: None)
    monkeypatch.setattr(forward.glue, "hc_post", lambda *a: None)
    monkeypatch.setattr(forward, "kda_block", lambda *a: calls.append(("attention",)))
    monkeypatch.setattr(forward, "mlp_block", lambda *a: calls.append(("ffn",)))
    forward.layer_forward(layer, w, [], b, 1)
    assert calls == [("select", 7, 0), ("attention",), ("select", 7, 1), ("ffn",)]


def test_hint_never_accepts_host_or_strided_storage():
    with pytest.raises(ValueError, match="contiguous CUDA weight"):
        l2.tensor(torch.ones(7))


@pytest.mark.parametrize("partitioned", [False, True])
def test_interleaved_halves_consume_hints_on_the_tail_stream(monkeypatch, partitioned):
    calls = []
    hint = fake_hint(monkeypatch, calls)
    hint.pending = None
    hint.plan = {i: (torch.tensor([10 * (i + 1)]), torch.tensor([10 * (i + 1) + 1])) for i in range(2)}
    parent = NS(name="overlap-tail", wait_event=lambda event: calls.append(("wait", "overlap-tail", event.name)))
    current = [NS(name="main")]

    @contextmanager
    def stream(s):
        old, current[0] = current[0], s
        try:
            yield
        finally:
            current[0] = old

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: current[0])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    select = hint.select
    def ordered_select(layer, phase):
        assert current[0].name == "main" and hint.pending is None
        select(layer, phase)
    hint.select = ordered_select

    class Side:
        def __init__(self, device):
            pass
        def tail(self, fn):
            with stream(parent):
                fn()
        def join(self):
            pass

    def gather(send, receive):
        assert current[0] is parent
        calls.append(("gather",))
        receive.zero_()
    def reduce(w, b, rows):
        assert current[0] is parent
        calls.append(("gather",))
        return torch.zeros(3, rows, 4)
    monkeypatch.setattr(forward.seqpar, "Side", Side)
    monkeypatch.setattr(forward.seqpar, "reduce", reduce)
    monkeypatch.setattr(forward, "hc_pre", lambda *a: None)
    monkeypatch.setattr(forward.glue, "hc_post", lambda *a: None)
    monkeypatch.setattr(forward, "kda_block", lambda layer, w, segs, b, rows: forward.gather(w, b, rows))
    monkeypatch.setattr(forward, "mlp_block", lambda layer, w, b, rows: forward.gather(w, b, rows))
    monkeypatch.setattr(forward.tp3_probe, "recorder", None)
    monkeypatch.setattr(forward.tp3_probe, "REDUCE", "gather")
    layers = [NS(index=i, kind="kda", mlp=object(), attn_hc=None, in_norm=None, ffn_hc=None, post_norm=None)
              for i in range(2)]
    w = NS(layers=layers, comm=NS(all_gather=gather))
    halves = []
    for _ in range(2):
        b = NS(x=torch.zeros(1, 4), part=torch.zeros(1, 4), gath=torch.zeros(12), world=3,
               post=torch.zeros(1), comb=torch.zeros(1), tap_at={}, l2_prefetch=hint,
               sp=NS(own=(0, 1)) if partitioned else None, defer=True)
        halves.append(forward._half(w, layers, NS(), b, 1, None, 0, final=False, skip_final_ffn=False))
    forward.lockstep(halves)
    expected = []
    for weight in [10, 10, 11, 11, 20, 20, 21, 21]:
        expected += [("record", "ready", "overlap-tail"), ("wait", "prefetch", "ready"),
                     ("hint", weight), ("record", "done", "prefetch"), ("gather",),
                     ("wait", "overlap-tail", "done")]
    assert calls == expected and hint.pending is None
