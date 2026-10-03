"""GLM-5.3-Flash's kept prompts on disk (TF_GLM_DISK_DIR), on the tiny synthetic checkpoint of test_glm_engine.py:
a conversation that left the device, or one from before a restart, resumes from its chain of written prompts and
replies as a fresh prefill does; files of another engine are removed; a rank without the file prefills afresh."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402

SAMPLING = Sampling(99, 1.0, 20, 0.95)


def _engine(path, disk, rows="32"):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    saved = {k: os.environ.get(k) for k in ("TF_GLM_DISK_DIR",)}
    if disk is None:
        os.environ.pop("TF_GLM_DISK_DIR", None)
    else:
        os.environ["TF_GLM_DISK_DIR"] = str(disk)
    try:
        return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), prefill_rows=int(rows))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_disk")
    _checkpoint(path)
    return path


@pytest.fixture(scope="module")
def cold(model):
    return _engine(model, None)


def _ids(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 990, size=n)]


def _turns(engine, seed=1):
    """Three turns of a conversation; each prompt is the previous prompt, its reply and new text."""
    prompt, prompts = _ids(seed, 70), []
    for turn in range(3):
        reply, _ = _generate(engine, prompt, SAMPLING, tokens=12)
        prompts.append(list(prompt))
        prompt = prompt + reply + _ids(seed + 10 + turn, 9 + turn)
    return prompts, prompt


def test_a_conversation_off_the_device_resumes_from_disk(model, cold, tmp_path):
    e = _engine(model, tmp_path)
    prompts, last = _turns(e)
    assert len(e.disk.entries) == 3
    assert sum(x.parent is None for x in e.disk.entries.values()) == 1          # a chain: each adds its rows
    _forget(e)                                                                  # nothing kept on the device
    warm, stats = _generate(e, last, SAMPLING)
    assert stats["cached"] == len(prompts[-1])
    _forget(cold)
    fresh, stats = _generate(cold, last, SAMPLING)
    assert stats["cached"] == 0 and warm == fresh


def test_a_restart_resumes_from_disk(model, cold, tmp_path):
    e = _engine(model, tmp_path)
    prompts, last = _turns(e, seed=5)
    del e
    torch.cuda.empty_cache()
    again = _engine(model, tmp_path)
    assert len(again.disk.entries) == 3
    warm, stats = _generate(again, last, SAMPLING)
    assert stats["cached"] == len(prompts[-1])
    _forget(cold)
    assert warm == _generate(cold, last, SAMPLING)[0]
    # a middle turn resumes too (its chain up to there), and another chunking would not share the files
    mid = prompts[1] + _ids(77, 5)
    warm, stats = _generate(again, mid, SAMPLING)
    assert stats["cached"] == len(prompts[1])
    _forget(cold)
    assert warm == _generate(cold, mid, SAMPLING)[0]
    other = _engine(model, tmp_path, rows="16")
    assert not other.disk.entries and not list((tmp_path / "rank0").glob("*.prompt"))


def test_a_rank_without_the_prompt_prefills_afresh(model, tmp_path):
    e = _engine(model, tmp_path)
    prompts, last = _turns(e, seed=9)
    _forget(e)
    entry = e.disk.resume(last, mtp=True)
    assert entry is not None
    real = e._gather_ints
    e._gather_ints = lambda values: [values, [0] * len(values)]                # rank 1 has lost it
    try:
        _, stats = _generate(e, last, SAMPLING)
    finally:
        e._gather_ints = real
    assert stats["cached"] == 0


def test_a_prompt_resumes_from_the_last_checkpoint_before_it_leaves_another(model, cold, tmp_path):
    """Checkpoints part way through a prompt: a prompt that shares only its start (an edited middle, another agent
    with the same tool list) resumes from the last one before they part, and replies as a fresh prefill does."""
    e = _engine(model, tmp_path)
    e.checkpoint_after = lambda pos: (pos // 16 + 1) * 16
    shared = _ids(31, 50)
    first = shared + _ids(32, 40)
    _generate(e, first, SAMPLING, tokens=8)
    assert {len(x.ids) for x in e.disk.entries.values()} >= {16, 32, 48, len(first)}
    e.checkpoint_after = lambda pos: (pos // 64 + 1) * 64             # on a chunk's own end (32-row chunks)
    aligned = _ids(35, 150)
    _generate(e, aligned, SAMPLING, tokens=8)
    assert {64, 128} <= {len(x.ids) for x in e.disk.entries.values()}
    e.checkpoint_after = lambda pos: (pos // 16 + 1) * 16
    _forget(e)
    other = shared + _ids(33, 30)                      # parts from ``first`` at token 50
    warm, stats = _generate(e, other, SAMPLING)
    assert stats["cached"] == 48
    _forget(cold)
    fresh, stats = _generate(cold, other, SAMPLING)
    assert stats["cached"] == 0 and warm == fresh
    # drafted still equals serial with checkpoints written along the way
    _forget(e)
    again = shared + _ids(34, 70)
    drafted, _ = _generate(e, again, SAMPLING)
    serial, _ = _generate(e, again, SAMPLING, draft=False)
    assert drafted == serial
