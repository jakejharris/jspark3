"""The large expert prefill tile must preserve every activation/output bit and reusable plan semantics."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts
from tensorfold.families.glm5_next.cuda import forward, glue, prefill_options
from test_experts import mlx, want_plan


@pytest.fixture(scope="module")
def ex():
    def signed(n, k, seed):
        w, s, b = mlx(5, n, k, 64, seed)
        s[:, ::2].neg_()
        s[:, :, 0] = 0
        b[:, :3, 0] = 0
        return w, s, b
    return experts.make([signed(192, 256, 10), signed(192, 256, 11)], signed(256, 192, 12), 64, limit=10.)


def run(ex, x, picks, tile, *, dtype=torch.bfloat16, plan=None, four_warps=False):
    rows, slots = picks.shape
    plan = plan or experts.Plan(rows, slots, ex.count, "cuda", prefill=True)
    plan.tile = tile
    experts.route(picks, plan)
    act = torch.full((rows * slots, ex.width), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = torch.full((rows * slots, ex.dims), float("nan"), device="cuda", dtype=dtype)
    experts.gate_up(x, ex, plan, act, rows, four_warps=four_warps)
    experts.down(act, ex, plan, out, rows)
    return act, out


def assert_bytes(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


@pytest.mark.parametrize("rows", [*range(1, 33), 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 321])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_affine_bytes_and_tails(ex, rows, dtype):
    torch.manual_seed(880 + rows)
    x = torch.randn((rows, 264), device="cuda", dtype=torch.bfloat16)[:, :256]
    x[:, :64] = 0
    picks = (torch.arange(rows * 3, device="cuda").reshape(rows, 3) % 4).int()
    picks[:, -1] = 4
    assert_bytes(run(ex, x, picks, 64, dtype=dtype), run(ex, x, picks, 128, dtype=dtype))


def test_plan_shrink_split_and_permute(ex):
    rows, slots = 321, 3
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    picks = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32).repeat(rows, 1)
    plan = experts.Plan(rows, slots, ex.count, "cuda", prefill=True)
    whole = run(ex, x, picks, 128, plan=plan)
    want, members, distinct = want_plan(picks, ex.count, 128)
    assert plan.items[:int(plan.counts[0])].tolist() == want
    assert plan.members.tolist() == members and int(plan.counts[1]) == distinct
    for a, b in [(0, 1), (5, 38), (64, 321), (0, 129), (250, 321)]:
        actual = run(ex, x[a:b], picks[a:b], 128, plan=plan)
        assert_bytes(tuple(t[a * slots:b * slots] for t in whole), actual)
    perm = torch.randperm(rows, device="cuda")
    idx = (perm[:, None] * slots + torch.arange(slots, device="cuda")).flatten()
    assert_bytes(tuple(t[idx] for t in whole), run(ex, x[perm], picks[perm], 128))


def test_graph_replay_changes_experts(ex):
    rows = 257
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    picks = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32).repeat(rows, 1)
    p = experts.Plan(rows, 3, ex.count, "cuda", prefill=True)
    p.tile = 128
    act = torch.empty((rows * 3, 192), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((rows * 3, 256), device="cuda", dtype=torch.bfloat16)
    def step():
        experts.route(picks, p)
        experts.gate_up(x, ex, p, act, rows)
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


def test_forward_flag_large_pass_and_reused_tail(ex, monkeypatch):
    capacity, slots = 8192, 3
    x = torch.randn((capacity, 256), device="cuda", dtype=torch.bfloat16)
    picks = (torch.arange(capacity * slots, device="cuda").reshape(capacity, slots) % 4).int()
    picks[:, -1] = 4
    b = SimpleNamespace(normed=x, mlog=torch.empty((capacity, 4), device="cuda"),
                        pick=torch.empty_like(picks), wts=torch.empty((capacity, slots), device="cuda"),
                        plan=experts.Plan(capacity, slots, 5, "cuda", prefill=True),
                        eact=torch.empty((capacity * slots, 192), device="cuda", dtype=torch.bfloat16),
                        ey=torch.empty((capacity, slots, 256), device="cuda", dtype=torch.bfloat16),
                        part=torch.empty((capacity, 256), device="cuda"))
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=256, top_k=2, experts=4, routed_scale=1., norm_topk=True))
    layer = SimpleNamespace(moe=SimpleNamespace(router=None, bias=None, experts=ex, shared=None))
    monkeypatch.setattr(glue, "router", lambda *args: None)
    def select(logits, bias, pick, weights, *args):
        pick.copy_(picks[:pick.shape[0]])
        weights.fill_(1 / slots)
    monkeypatch.setattr(glue, "select", select)
    monkeypatch.setattr(forward, "gather", lambda w, b, r: b.part[:r].clone())
    for rows in (8192, 1, 4096, 4095, 32, 257, 2048):
        monkeypatch.setattr(prefill_options, "EXPERT_PREFILL128", False)
        off = forward.moe_block(layer, w, b, rows)
        assert b.plan.tile == 64
        monkeypatch.setattr(prefill_options, "EXPERT_PREFILL128", True)
        on = forward.moe_block(layer, w, b, rows)
        assert b.plan.tile == (128 if rows >= 4096 else 64)
        assert_bytes((off,), (on,))
    corrupt = on.clone()
    corrupt.view(torch.uint8).flatten()[0] ^= 1
    with pytest.raises(AssertionError):
        assert_bytes((off,), (corrupt,))


def test_odd_k_groups_and_column_tail():
    ex = experts.make([mlx(5, 192, 320, 64, 90), mlx(5, 192, 320, 64, 91)],
                      mlx(5, 320, 192, 64, 92), 64, limit=10.)
    x = torch.randn((129, 328), device="cuda", dtype=torch.bfloat16)[:, :320]
    picks = (torch.arange(129 * 3, device="cuda").reshape(129, 3) % 5).int()
    for dtype in (torch.float32, torch.bfloat16):
        assert_bytes(run(ex, x, picks, 64, dtype=dtype), run(ex, x, picks, 128, dtype=dtype))


def whole_against_halves(ex, rows, slots, routing, dtype):
    """E1 gate: compare every ACT/down/combine byte for identical inputs, not reply hashes."""
    torch.manual_seed(1741 + rows)
    x = torch.randn((rows, ex.dims + 8), device="cuda", dtype=torch.bfloat16)[:, :ex.dims]
    x[:, :64] = 0
    if routing == "hot":
        picks = torch.arange(slots, device="cuda").repeat(rows, 1).int()
    elif routing == "uniform":
        picks = torch.rand((rows, ex.count - 1), device="cuda").argsort(1)[:, :slots].int().contiguous()
    else:
        picks = (torch.arange(rows * slots, device="cuda").reshape(rows, slots) % (ex.count - 1)).int()
    picks[:, -1] = ex.count - 1
    weights = torch.rand((rows, slots), device="cuda", dtype=torch.float32)
    weights[:, -1] = 1

    def combined(pair):
        act, down = pair
        result = torch.full((rows, ex.dims), float("nan"), device="cuda", dtype=torch.float32)
        glue.combine(down.reshape(rows, slots, ex.dims), weights, result)
        return act, down, result

    stock = combined(run(ex, x, picks, 64, dtype=dtype))
    # Exactly the overlap split, including an odd final row. Reuse the plan as
    # production does, so route/tile changes must not leave stale memberships.
    plan = experts.Plan(rows, slots, ex.count, "cuda", prefill=True)
    split = rows // 2
    halves = [run(ex, x[a:b], picks[a:b], 64, dtype=dtype, plan=plan, four_warps=1024 <= b - a < 4096)
              for a, b in ((0, split), (split, rows))]
    joined = combined(tuple(torch.cat([a[i] for a in halves]) for i in range(2)))
    assert_bytes(stock, joined)
    whole = combined(run(ex, x, picks, 128, dtype=dtype, plan=plan))
    assert_bytes(stock, whole)
    assert_bytes(joined, whole)
    # A one-byte difference must fail even if token selection would be unchanged.
    corrupt = whole[-1].clone()
    corrupt.view(torch.uint8).flatten()[0] ^= 1
    with pytest.raises(AssertionError):
        assert_bytes((joined[-1],), (corrupt,))


@pytest.mark.parametrize("rows", [4096, 4097, 8192])
@pytest.mark.parametrize("routing", ["modulo", "uniform", "hot"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_e1_whole_vs_actual_halves(ex, rows, routing, dtype):
    whole_against_halves(ex, rows, 3, routing, dtype)


def test_e1_whole_vs_halves_full_rank_width():
    # Production rank-local dimensions, 8 routed slots + shared; synthetic g64
    # weights/activations. This is a native arithmetic gate, not a model replay.
    full = experts.make([mlx(17, 704, 4096, 64, 711), mlx(17, 704, 4096, 64, 712)],
                        mlx(17, 4096, 704, 64, 713), 64, limit=10.)
    whole_against_halves(full, 4096, 9, "uniform", torch.bfloat16)
