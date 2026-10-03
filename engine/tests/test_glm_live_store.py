"""The live ABI preserves bytes that reusable prompt snapshots intentionally omit."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.disk_io import CHUNK, HEADER_LIMIT, DirectFile, aligned
from tensorfold.families.glm5_next.cuda.live_store import LiveStore, tensor_views
from tensorfold.families.glm5_next.cuda.session_disk import SessionStore
from test_glm_session_disk import state


def stream(rank=0):
    e, d, _ = state(140, rank=rank, mtp=True)
    d.cap = d.kc[0].stride(0) // d.kc[0].shape[2]
    e.st.mtp_len, e.st.mtp_drafted = 147, 5
    e.images = None
    return SimpleNamespace(sid=11, prompt=list(range(140)), count=200,
        sampling=Sampling(2**63 + 11, .7123456789012345, 23, .9345678901234567),
        policy_code=[1, 7, 0, 0], image_digests=(), eos=(), draft=True,
        engine=e, st=e.st, drafter=d, use_dflash=True, span=208, out=[3, 5, 9], drafts=[17, 19, 20])


def backed(tmp_path, s):
    disk = SessionStore(tmp_path, 256 * 2**20, "same-exact-weights", min_free=0)
    store = LiveStore(disk, "current-worker", 0, 3)
    tensors = tensor_views(s.engine, s.drafter, s.span - 8, s.span // 4)
    size = 2 * (HEADER_LIMIT + sum(aligned(t.numel() * t.element_size()) for _, t in tensors))
    assert disk.reserve_live(s.sid, size)
    return store, tensors


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_both_kda_banks_mtp_proposals_and_ring_leading_tile_survive(tmp_path, rank):
    s = stream(rank)
    store, tensors = backed(tmp_path, s)
    expected = {name: t.view(torch.uint8).clone() for name, t in tensors}
    store.prepare(s, 7, filling={})
    dest = stream(rank + 3)
    dest.engine.last_hidden.zero_()
    for _, t in tensor_views(dest.engine, dest.drafter, 200, 52):
        t.zero_()
    store.load(s, dest.engine, dest.drafter)
    actual = dict(tensor_views(dest.engine, dest.drafter, 200, 52))
    assert all(torch.equal(actual[name].view(torch.uint8), bits) for name, bits in expected.items())
    assert dest.st.cur == s.st.cur and dest.st.mtp_len == 147 and dest.st.mtp_drafted == 5
    assert dest.drafter.context_end == 140 and dest.drafter.pos_dev.item() == 140
    assert store.records and s.sid in store.store.live_credits
    # Ordinary cache eviction cannot select the sole continuation of an open response.
    store.store._evict(set(), needed=store.store.budget)
    assert store.records[s.sid][0].exists()


def test_large_recurrence_is_streamed_in_bounded_tiles_without_row_clone(tmp_path, monkeypatch):
    s = stream()
    s.st.rec = torch.ones(2, 3, 1024, 1024)  # each bank > the session store's 8 MiB row limit
    store, _ = backed(tmp_path, s)
    writes = []
    original = DirectFile.write
    def write(self, at, raw):
        writes.append(len(raw))
        return original(self, at, raw)
    monkeypatch.setattr(DirectFile, "write", write)
    store.prepare(s, 1, filling={})
    assert max(writes) == CHUNK and all(n <= CHUNK for n in writes)


@pytest.mark.parametrize("mutation", ["count", "sampling", "images", "out", "drafts"])
def test_host_continuation_identity_mismatch_rejects_before_destination_write(tmp_path, mutation):
    s = stream()
    store, _ = backed(tmp_path, s)
    store.prepare(s, 1, filling={})
    dest = stream(1)
    before = dest.st.rec.clone()
    if mutation == "count":
        s.count -= 1
    elif mutation == "sampling":
        s.sampling = Sampling(19, .71, 23, .93)
    elif mutation == "images":
        s.image_digests = (b"a" * 32,)
    elif mutation == "out":
        s.out.append(1)
    else:
        s.drafts[0] ^= 1
    with pytest.raises(RuntimeError, match="identity"):
        store.load(s, dest.engine, dest.drafter)
    assert torch.equal(before, dest.st.rec) and store.records


def test_resource_floor_refuses_before_allocating_file(tmp_path, monkeypatch):
    s = stream()
    store, _ = backed(tmp_path, s)
    def fail():
        raise OSError("injected memory floor")
    monkeypatch.setattr(store, "guard", fail)
    with pytest.raises(OSError, match="memory floor"):
        store.prepare(s, 1, filling={})
    assert not list(store.dir.iterdir()) and not store.records


def test_prior_epoch_bytes_and_current_credits_share_the_hard_budget(tmp_path):
    s = stream()
    store, _ = backed(tmp_path, s)
    store.prepare(s, 1, filling={})
    size = store.records[s.sid][0].stat().st_size
    reopened = SessionStore(tmp_path, size + 4096, "same-exact-weights", min_free=0)
    assert reopened.orphan_live_bytes == size
    assert not reopened.reserve_live(2, 8192)
    assert reopened.reserve_live(2, 4096)
    assert not reopened.reserve_live(3, 1)
