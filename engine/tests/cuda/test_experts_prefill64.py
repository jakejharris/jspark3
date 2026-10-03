"""The medium prefill layout preserves bytes, tails, plan reuse and graph replay."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts
from tensorfold.families.glm5_next.cuda import forward, glue, prefill_options
from test_experts import mlx
from test_experts_prefill128 import assert_bytes, ex, run


@pytest.mark.parametrize("rows", [*range(1, 18), 31, 32, 33, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_affine_bytes_and_tails(ex, rows, dtype):
    torch.manual_seed(2200 + rows)
    x = torch.randn((rows, 264), device="cuda", dtype=torch.bfloat16)[:, :256]
    x[:, :64] = 0
    picks = (torch.arange(rows * 3, device="cuda").reshape(rows, 3) % 4).int()
    picks[:, -1] = 4
    assert_bytes(run(ex, x, picks, 64, dtype=dtype),
                 run(ex, x, picks, 64, dtype=dtype, four_warps=True))


def test_plan_reuse_and_graph(ex):
    rows, slots = 257, 3
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    picks = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32).repeat(rows, 1)
    p = experts.Plan(rows, slots, ex.count, "cuda", prefill=True)
    whole = run(ex, x, picks, 64, plan=p, four_warps=True)
    for a, b in [(0, 1), (5, 38), (64, 257), (0, 129), (250, 257)]:
        actual = run(ex, x[a:b], picks[a:b], 64, plan=p, four_warps=True)
        assert_bytes(tuple(t[a * slots:b * slots] for t in whole), actual)
    perm = torch.randperm(rows, device="cuda")
    idx = (perm[:, None] * slots + torch.arange(slots, device="cuda")).flatten()
    assert_bytes(tuple(t[idx] for t in whole), run(ex, x[perm], picks[perm], 64, four_warps=True))
    act, out = whole
    def step():
        experts.route(picks, p)
        experts.gate_up(x, ex, p, act, rows, four_warps=True)
        experts.down(act, ex, p, out, rows)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for offset in (1, 2, 3):
        picks[:, :2].add_(offset).remainder_(4)
        graph.replay()
        assert_bytes(run(ex, x, picks, 64), (act, out))


@pytest.mark.parametrize("flag128", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_forward_flags_and_reused_tail(ex, monkeypatch, flag128, prefill):
    capacity, slots = 8192, 3
    x = torch.randn((capacity, 256), device="cuda", dtype=torch.bfloat16)
    picks = (torch.arange(capacity * slots, device="cuda").reshape(capacity, slots) % 4).int()
    picks[:, -1] = 4
    b = SimpleNamespace(normed=x, mlog=torch.empty((capacity, 4), device="cuda"),
                        pick=torch.empty_like(picks), wts=torch.empty((capacity, slots), device="cuda"),
                        plan=experts.Plan(capacity, slots, 5, "cuda", prefill=prefill),
                        eact=torch.empty((capacity * slots, 192), device="cuda", dtype=torch.bfloat16),
                        ey=torch.empty((capacity, slots, 256), device="cuda",
                                       dtype=torch.bfloat16 if prefill else torch.float32),
                        part=torch.empty((capacity, 256), device="cuda"))
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=256, top_k=2, experts=4, routed_scale=1., norm_topk=True))
    layer = SimpleNamespace(moe=SimpleNamespace(router=None, bias=None, experts=ex, shared=None))
    monkeypatch.setattr(glue, "router", lambda *args: None)
    def select(logits, bias, pick, weights, *args):
        pick.copy_(picks[:pick.shape[0]])
        weights.fill_(1 / slots)
    monkeypatch.setattr(glue, "select", select)
    monkeypatch.setattr(forward, "gather", lambda w, b, r: b.part[:r].clone())
    monkeypatch.setattr(prefill_options, "EXPERT_PREFILL128", flag128)
    calls = []
    original = experts.gate_up
    def gate(*args, **kwargs):
        calls.append(kwargs["four_warps"])
        return original(*args, **kwargs)
    monkeypatch.setattr(experts, "gate_up", gate)
    for rows in (2048, 1, 1023, 1024, 4095, 4096, 8192, 33):
        monkeypatch.setattr(prefill_options, "EXPERT_PREFILL64", False)
        off = forward.moe_block(layer, w, b, rows)
        assert not calls[-1]
        monkeypatch.setattr(prefill_options, "EXPERT_PREFILL64", True)
        on = forward.moe_block(layer, w, b, rows)
        assert calls[-1] == (prefill and 1024 <= rows < 4096)
        assert b.plan.tile == (128 if prefill and flag128 and rows >= 4096 else 64 if prefill else 16)
        assert_bytes((off,), (on,))


def test_odd_k_and_column_tail():
    ex = experts.make([mlx(5, 192, 320, 64, 90), mlx(5, 192, 320, 64, 91)],
                      mlx(5, 320, 192, 64, 92), 64, limit=10.)
    x = torch.randn((129, 328), device="cuda", dtype=torch.bfloat16)[:, :320]
    picks = (torch.arange(129 * 3, device="cuda").reshape(129, 3) % 5).int()
    assert_bytes(run(ex, x, picks, 64), run(ex, x, picks, 64, four_warps=True))
    with pytest.raises(RuntimeError, match="four-warp prefill requires"):
        run(ex, x, picks, 128, four_warps=True)
