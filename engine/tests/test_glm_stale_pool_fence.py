"""Plain cache transfers must not turn a saved future pool fence into stale state."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from test_glm_batched_host import setup_decoder, _stream  # noqa: F401
from test_cuda_geometry import allocations  # noqa: F401


@pytest.mark.parametrize("disable_refresh", [False, True])
def test_three_flags_off_requests_refresh_newly_committed_pool(setup_decoder, monkeypatch, disable_refresh):
    """Real admission/save/load paths with CPU pooling and a missing-refresh control."""
    mod, make = setup_decoder
    from tensorfold.families.glm5_next.cuda import session_state

    if disable_refresh:
        monkeypatch.setattr(session_state, "refresh_transferred_pool_fences", lambda *args: None)
    original_prefill = mod.prefill
    observed = []
    boundary = 4097 // 4

    def prefill(e, prompt, sampling, mtp, drafter, resume, **kwargs):
        begin = len(resume.ids) if resume else 0
        pools = [pk.clone() for _, _, pk in e.st.index]
        observed.append((begin, pools[0][boundary].clone()))
        first = original_prefill(e, prompt, sampling, mtp, drafter, resume)
        data = torch.tensor([[token, i] for i, token in enumerate(prompt)], dtype=torch.float32)
        # Match _pool_keys' real write range [begin//4, end//4). Do not poison
        # anything: the first snapshot's future fence starts at allocation zero.
        for (_, _, pk), before in zip(e.st.index, pools):
            pk.copy_(before)
            for at in range(begin // 4, len(prompt) // 4):
                pk[at].copy_(data[4 * at:4 * at + 4].sum(0))
        return first

    monkeypatch.setattr(mod, "prefill", prefill)
    d = make(2, limit=8192, cache_bytes=10_000_000)
    assert d.prefill_slice_layers == 0 and not d.reply_prefill
    assert d.sessions is None and d.growth is None
    parent = d.owner.drafter
    parent.ring, parent.window, parent.cap = 8192, 8128, 2 * 8192
    parent.kc = [torch.zeros(2, parent.cap, 2)]
    parent.vc = [torch.zeros(2, parent.cap, 2)]
    prompt = [1 + ((17 * i + 3) % 61) for i in range(4102)]
    data = torch.tensor([[token, i] for i, token in enumerate(prompt)], dtype=torch.float32)
    fresh_pools = data[:4100].reshape(-1, 4, 2).sum(1)
    saved_pools = []
    for n in (4097, 4101, 4102):
        request = _stream(prompt[:n], 1, policy="fc7:0.3")
        request.prefill_slice_layers = 0
        d.admit(request)
        assert request.done and len(request.out) == 1
        pieces = d.cache[-1].rows[3]
        pooled = torch.cat(pieces, dim=0) if isinstance(pieces, tuple) else pieces
        saved_pools.append(pooled.clone())
        d.finish([request])

    assert [cached for cached, _ in observed] == [0, 4097, 4101]
    assert observed[2][0] // 4 == 1025 > boundary  # third prefill cannot repair pool1024
    assert torch.equal(saved_pools[0][:boundary], fresh_pools[:boundary])
    assert not saved_pools[0][boundary].any()  # naturally unused future fence
    for pooled in saved_pools[1:]:
        # Every earlier committed pool retains exactly its original bytes.
        assert torch.equal(pooled[:boundary], saved_pools[0][:boundary])
        assert torch.equal(pooled[:1025], fresh_pools) != disable_refresh
    if disable_refresh:
        assert not saved_pools[1][boundary].any()
        assert not observed[2][1].any()  # the bad snapshot actually reaches the third request
    else:
        assert torch.equal(saved_pools[1][boundary], fresh_pools[boundary])
        assert torch.equal(observed[2][1], fresh_pools[boundary])
