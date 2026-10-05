"""Rolling Pi images must retain a reusable prefix before the rewritten tool turn."""

from types import SimpleNamespace

import pytest
import torch

from test_glm_batched_host import setup_decoder, allocations, _stream  # noqa: F401
from test_glm_prefill_slices import sliced_decoder, drain_steps  # noqa: F401
from test_glm_session_batched import attach, finish_writers  # noqa: F401
from tensorfold.families.glm5_next.cuda.session_state import state_hash


def conversation(rotations):
    # Pi puts tool text in an observation, then images in a separate user message.
    # Rotation changes the observation too, so an anchor at the image/user token
    # is already too late. Eight images remain in every outgoing request.
    prompt = [1, 2, 3, 4, 5]
    anchors = []
    for i in range(rotations + 8):
        start = len(prompt)
        prompt += [51, 4, 5, 52, 6]  # assistant call, observation text
        if i < rotations:
            prompt += [7, 8, 9]     # archived-image reference, now part of tool text
        else:
            anchors.append(start)
            prompt += [50, 10, -(i + 1), -(i + 1), 11]
    prompt += [51, 12]
    images = [SimpleNamespace(digest=bytes([i + 1]) * 32) for i in range(rotations, rotations + 8)]
    return prompt, images, anchors[0]


@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("cache_bytes", [0, 100_000], ids=["no-memory-cache", "memory-cache"])
def test_image_rotation_advances_disk_anchor_with_general_checkpoints_off(sliced_decoder, tmp_path,
                                                                         sliced, cache_bytes, monkeypatch):
    from tensorfold.families.glm5_next.cuda import decode

    compute = decode.compute

    def immutable_pools(w, st, b, n, **kwargs):
        result = compute(w, st, b, n, **kwargs)
        # The generic slice fixture fills *all* pool rows with the current
        # length. Production completed pool rows are immutable; model that
        # contract so a parent/child disk chain can share them faithfully.
        for _, _, pk in st.index:
            pk.copy_(torch.arange(len(pk))[:, None].expand_as(pk))
        return result

    monkeypatch.setattr(decode, "compute", immutable_pools)
    _, build = sliced_decoder
    d = build(2)
    d.owner.tower = object()
    d.owner.cache_bytes = cache_bytes
    d.owner.drafter.ring, d.owner.drafter.window = 256, 8
    disk = attach(d, tmp_path / "saved")
    d.sessions.role_ids = frozenset((50, 51, 52))
    d.sessions.assistant_id = 51
    assert not d.sessions.config.checkpoints and not d.sessions.config.cancel
    previous_anchor = 0
    for rotation in range(4):
        prompt, images, anchor = conversation(rotation)
        s = _stream(prompt, 1)
        s.images = images
        s.prefill_slice_layers = 2 if sliced else 0
        d.begin_admit(s)
        assert s.cached == previous_anchor
        drain_steps(d)
        disk.wait_pending()
        saved = next(e for e in disk.entries.values() if len(e.ids) == anchor)
        assert saved.ids.tolist() == prompt[:anchor]
        assert saved.image_digests == ()  # no stale/partial image identity in this anchor
        # Compare restored target + DFlash state with a fresh prefill of the same
        # changed conversation. The fixture substitutes CPU math, not cache logic.
        actual = disk.resume(prompt + [13], mtp=False, dflash=True,
                             image_digests=tuple(im.digest for im in images))
        dest = build(2)
        dest.owner.drafter.ring, dest.owner.drafter.window = 256, 8
        restored = disk.load(dest.owner.e, actual, dest.owner.drafter)
        fresh = build(2)
        fresh.owner.tower = object()
        fresh.owner.drafter.ring, fresh.owner.drafter.window = 256, 8
        fresh_disk = attach(fresh, tmp_path / f"fresh-{rotation}")
        control = _stream(prompt, 1)
        control.images = images
        fresh.begin_admit(control)
        drain_steps(fresh)
        fresh_disk.wait_pending()
        control_entry = fresh_disk.resume(prompt + [13], mtp=False, dflash=True,
                                          image_digests=tuple(im.digest for im in images))
        control_snap = fresh_disk.load(fresh.owner.e, control_entry, fresh.owner.drafter)
        assert state_hash(dest.owner.e, restored) == state_hash(fresh.owner.e, control_snap)
        assert s.out == control.out
        assert anchor > previous_anchor
        previous_anchor = anchor
        d.cache.clear()  # prove disk reuse independently of full-prompt memory snapshots


def test_image_free_and_unmarked_prompts_add_no_checkpoint(tmp_path):
    from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig

    cache = SessionCache(SimpleNamespace(), SessionConfig(disk=True),
                         SimpleNamespace(entries={}, invalidated=[]), role_ids=(50, 51, 52))
    for prompt in ([1, 2, 50, 3, 4], [-1, -1, 2, 3]):
        s = SimpleNamespace(prompt=prompt, cached=0, draft=True, image_digests=())
        cache.begin(s)
        assert cache.prefill_kwargs(s) == {}


def test_user_attachment_without_prior_assistant_uses_its_message_start(tmp_path):
    from tensorfold.families.glm5_next.cuda.session_cache import SessionCache, SessionConfig

    cache = SessionCache(SimpleNamespace(), SessionConfig(disk=True),
                         SimpleNamespace(entries={}, invalidated=[]), role_ids=(50, 51, 52))
    s = SimpleNamespace(prompt=[1, 2, 50, 10, -1, -1, 11], cached=0, draft=True,
                        image_digests=(b"a" * 32,))
    cache.begin(s)
    assert cache.next_mark(s, 0) == 2
    assert cache.next_mark(s, 2) is None


def test_disk_configuration_loads_and_shares_image_anchor_roles_without_dense_checkpoints(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models
    from tensorfold.families.glm5_next.cuda import session_disk
    from tensorfold.families.glm5_next.cuda.session_cache import SessionConfig, configure

    roles = {"<|system|>": 49, "<|user|>": 50, "<|assistant|>": 51, "<|observation|>": 52}
    Tokenizer(models.WordLevel(roles)).save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setenv("TF_GLM_DISK_DIR", str(tmp_path))
    monkeypatch.setattr(session_disk, "exact_fingerprint", lambda *a, **k: "fixture")
    shared = []

    def share(ids):
        shared.append(ids)
        return ids

    owner = SimpleNamespace(rank=0, world=3, _share=share, _gather_ints=lambda ids: [ids] * 3,
                            drafter=None, w=SimpleNamespace(cfg=SimpleNamespace(hidden=2)),
                            e=SimpleNamespace(st=SimpleNamespace(rec=torch.zeros(2, 2), conv=torch.zeros(2))))
    configure(owner, SessionConfig(disk=True), tmp_path, None, {}, slots=8)
    assert owner.session_cache.assistant_id == 51
    assert owner.session_cache.role_ids == frozenset(roles.values())
    assert shared == [[49, 50, 51, 52], [51]]
