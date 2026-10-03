"""Drafts copied from the context (GLM-5.3-Flash on CUDA): the tokens that followed the context's last tokens
earlier, once enough of them agree, up to the round's room and the draft cap."""

from __future__ import annotations

from tensorfold.families.glm5_next.cuda.lookup import PromptLookup


def test_a_run_seen_before_is_proposed():
    p = PromptLookup([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 3, 4, 5, 6, 7], agree=5, drafts=5)
    assert p.propose(5) == [8, 9, 10, 20, 21]
    assert p.propose(2) == [8, 9]
    p.extend([8])
    assert p.propose(5) == [9, 10, 20, 21, 3]


def test_too_short_an_agreement_proposes_nothing():
    assert PromptLookup([1, 2, 3, 9, 9, 1, 2, 3], agree=5).propose(5) == []   # 3 agree, 5 needed
    assert PromptLookup(list(range(10)) + [50] + list(range(3, 10))).propose(5) == []     # 7 agree, 8 by default
    assert PromptLookup([4, 5, 6, 7]).propose(5) == []


def test_the_reply_joins_the_context():
    p = PromptLookup([50, 51], agree=5)
    p.extend([1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5])
    assert p.propose(3) == [6, 9, 1]
