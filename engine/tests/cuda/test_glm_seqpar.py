"""TF_GLM_SP on one GPU: each of three ranks' row-local launches (hc_pre, hc_post on its rank-ordered partials, the
stream mean) on its share of a full-size chunk's buffers gives exactly the bytes of the same rows in today's full
launch, at the engine's width and chunk sizes, also with its own partial read in place (``seqpar.Reduced``, its slot
never written); a rotated rank order does not."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import glue, seqpar

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
S, D, WORLD = 4, 4096, 3


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


@cuda
@pytest.mark.parametrize("R", [5, 1024, 2047, 2048, 8192])
def test_row_shares_keep_the_full_launch_bits(R):
    gen = torch.Generator().manual_seed(R)
    x = torch.randn((R, S * D), generator=gen).to(torch.bfloat16).cuda()
    fn = (torch.randn((24, S * D), generator=gen) * 0.01).to(torch.bfloat16).cuda()
    base = (torch.randn((24,), generator=gen) * 0.1).cuda()
    scale = torch.tensor([0.5, 0.7, 1.1]).cuda()
    nw = (1 + 0.1 * torch.randn((D,), generator=gen)).to(torch.bfloat16).cuda()
    parts = (torch.randn((WORLD, R, D), generator=gen) * torch.logspace(-4, 4, D)).cuda()
    parts[:, :, ::8] = torch.tensor([2.0 ** 24, 1.0, -(2.0 ** 24)]).view(WORLD, 1, 1).cuda()  # only (p0+p1)+p2 is 0

    def scratch():
        return (torch.full((R, D), float("nan"), dtype=torch.bfloat16, device="cuda"),
                torch.full((R, D // 64), float("nan"), device="cuda"), torch.full((R, S), float("nan"), device="cuda"),
                torch.full((R, S * S), float("nan"), device="cuda"),
                torch.full((R, glue.HC_BLOCKS, 32), float("nan"), device="cuda"))

    # today: every rank runs every row
    normed, xs, post, comb, hcpart = scratch()
    glue.hc_pre(x, fn, base, scale, nw, normed, xs, post, comb, hcpart, 1e-5, 1e-6, 20)
    xo = x.clone()
    glue.hc_post(xo, xo, parts, post, comb)
    hidden = torch.empty((R, D), dtype=torch.bfloat16, device="cuda")
    glue.stream_mean(xo, hidden)

    # TF_GLM_SP: rank k on rows lo:hi of its own full-size buffers, its partials at offsets 0, 1, 2 in a [3, n, D] view
    starts, counts = seqpar.split(R, WORLD)
    gath = torch.empty((WORLD * R * D,), device="cuda")
    for order in ((0, 1, 2), (2, 0, 1)):
        for k in range(WORLD):
            lo, hi = starts[k], starts[k] + counts[k]
            n = hi - lo
            mine = torch.full_like(x, float("nan"))
            mine[lo:hi] = x[lo:hi]
            m_normed, m_xs, m_post, m_comb, m_part = scratch()
            glue.hc_pre(mine[lo:hi], fn, base, scale, nw, m_normed[lo:hi], m_xs[lo:hi], m_post[lo:hi], m_comb[lo:hi],
                        m_part[lo:hi], 1e-5, 1e-6, 20)
            for a, b in ((m_normed, normed), (m_xs, xs), (m_post, post), (m_comb, comb)):
                assert torch.equal(_bits(a[lo:hi]), _bits(b[lo:hi])), (R, k)
            out = gath[:WORLD * n * D].view(WORLD, n, D)
            for i, src in enumerate(order):
                out[i].copy_(parts[src, lo:hi])
            pre = mine[lo:hi].clone()
            glue.hc_post(mine[lo:hi], mine[lo:hi], out, m_post[lo:hi], m_comb[lo:hi])
            m_hidden = torch.full((R, D), float("nan"), dtype=torch.bfloat16, device="cuda")
            glue.stream_mean(mine[lo:hi], m_hidden[lo:hi])
            if order == (0, 1, 2):
                assert torch.equal(_bits(mine[lo:hi]), _bits(xo[lo:hi])), (R, k)
                assert torch.equal(_bits(m_hidden[lo:hi]), _bits(hidden[lo:hi])), (R, k)
                # seqpar.reduce: the own partial where its block wrote it (rows lo:hi of b.part), its slot unwritten
                part = torch.full((R, D), float("nan"), device="cuda")
                part[lo:hi] = parts[k, lo:hi]
                out[k].fill_(float("nan"))
                glue.hc_post(pre, pre, seqpar.Reduced(out, k, part[lo:hi]), m_post[lo:hi], m_comb[lo:hi])
                assert torch.equal(_bits(pre), _bits(xo[lo:hi])), (R, k, "own partial in place")
            else:                         # the ring's grouping on these columns: the control must see it
                assert not torch.equal(_bits(mine[lo:hi]), _bits(xo[lo:hi])), (R, k)
