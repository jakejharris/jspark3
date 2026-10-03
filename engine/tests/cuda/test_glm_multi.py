"""GLM-5.3-Flash's concurrent streams on the tiny synthetic checkpoint of test_glm_engine.py: several streams' rows in one forward get each stream's own bits, and every stream decoded together with others equals its serial decoding."""

from __future__ import annotations


import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _generate, _TwoCopies  # noqa: E402

CONTEXT = 2600


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_multi")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=CONTEXT, prefill_rows=256)


def _prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


def _decoder(engine, slots=3):
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    return MultiDecoder(engine.w, slots=slots, capacity=CONTEXT, depth=3, long_context=True)


def test_streams_in_one_forward_get_their_own_bits(engine):
    """Rows of three streams (dense and past the dense limit) in one forward equal each stream's rows alone."""
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import compute_streams, stage_streams
    from tensorfold.families.glm5_next.cuda.multi import _slot

    dec = _decoder(engine)
    w = engine.w
    for slot, (seed, n) in enumerate(((1, 40), (2, 2100), (3, 300))):
        prefill(_slot(w, dec.slots[slot], dec.buf, dec.mbuf, dec.pbuf), _prompt(seed, n), None)
    windows = [(dec.slots[0], [5, 6, 7]), (dec.slots[1], [8]), (dec.slots[2], [9, 10])]
    segs = stage_streams(w, dec.buf, windows)
    together = compute_streams(w, segs, dec.buf, eager=True).clone()
    for (st, tokens), (_, a0, a1) in zip(windows, segs):
        one = stage_streams(w, dec.buf, [(st, tokens)])
        alone = compute_streams(w, one, dec.buf, eager=True)
        assert torch.equal(alone[:a1 - a0], together[a0:a1])


@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
def test_concurrent_streams_equal_their_serial_decoding(engine, sampled):
    """Three requests decoded together (one short, one past the dense limit, one serial) each equal their serial reference."""
    reqs = [(_prompt(11, 60), 24, True), (_prompt(12, 2090), 24, True), (_prompt(13, 90), 16, False)]
    samplings = [Sampling(40 + i, 1.0, 20, 0.95) if sampled else None for i in range(len(reqs))]
    want = [_generate(engine, p, s, draft=False, tokens=n)[0] for (p, n, _), s in zip(reqs, samplings)]
    dec = _decoder(engine)
    streams = []
    for (p, n, drafts), s in zip(reqs, samplings):
        st = Stream(list(p), n, s, draft=drafts, stop_eos=False)
        dec.admit(st)
        streams.append(st)
    rounds = 0
    while dec.live():
        done = dec.round()
        dec.finish(done + [s for s in streams if s.done and s.sid in dec.streams and s not in done])
        rounds += 1
        assert rounds < 200
    for st, ref, (_, n, _) in zip(streams, want, reqs):
        assert len(ref) == n and st.out == ref
    assert streams[0].rounds < 24                       # the drafting streams kept more than one row a round


def test_a_follow_up_resumes_from_its_kept_prompt(engine):
    """A second turn extending a finished stream's prompt resumes from that slot and still equals serial decoding."""
    dec = _decoder(engine, slots=2)
    first = _prompt(21, 120)
    s1 = Stream(list(first), 12, None, draft=True, stop_eos=False)
    dec.admit(s1)
    while dec.live():
        dec.finish(dec.round())
    follow = first + s1.out + _prompt(22, 7)
    s2 = Stream(list(follow), 12, None, draft=True, stop_eos=False)
    dec.admit(s2)
    while dec.live():
        dec.finish(dec.round())
    assert len(s1.out) == 12 and s2.cached == len(first)
    assert s2.out == _generate(engine, follow, None, draft=False, tokens=12)[0]


def test_repeated_prompts_never_lose_a_slot(engine):
    """The same prompt again and again, alone and two at once, keeps every slot either free, kept or busy."""
    dec = _decoder(engine, slots=2)
    prompt = _prompt(31, 50)
    for batch in (1, 1, 2, 1, 2, 2):
        streams = [Stream(list(prompt), 4, None, draft=True, stop_eos=False) for _ in range(batch)]
        for st in streams:
            dec.admit(st)
        while dec.live():
            dec.finish(dec.round())
        held = set(dec.free) | {k[1] for k in dec.kept}
        assert held == {0, 1}, (dec.free, dec.kept)
