"""Reservations protect active views, fences and the full single-request context."""

import random

import pytest

from tensorfold.families.glm5_next.cuda.pool import TokenPool


def test_full_context_fits_without_truncation_and_refusal_does_not_mutate():
    pool = TokenPool(262152 + 7 * 64)
    assert pool.allocate("full", 262144) == (0, 262152)
    before = dict(pool.spans)
    assert not pool.can_allocate(500)
    assert pool.allocate("extra", 500) is None
    assert pool.spans == before
    pool.release("full")
    assert pool.available == pool.capacity


def test_fragmentation_and_release_are_deterministic():
    pool = TokenPool(64)
    assert pool.allocate("a", 9) == (0, 20)
    assert pool.allocate("b", 9) == (20, 20)
    assert pool.allocate("c", 9) == (40, 20)
    pool.release("b")
    assert pool.available == 24
    assert not pool.can_allocate(13)             # 24 free rows, but no 24-row contiguous gap
    assert pool.allocate("small", 5) == (20, 16)
    pool.release("a")
    pool.release("small")
    assert pool.allocate("joined", 25) == (0, 36)
    with pytest.raises(ValueError, match="already"):
        pool.allocate("joined", 1)


@pytest.mark.parametrize("guard", [8, 16, 64])
def test_random_lifetimes_never_overlap_and_keep_index_fences(guard):
    rng = random.Random(49)
    pool = TokenPool(4096, guard=guard)
    for i in range(1000):
        if pool.spans and rng.random() < 0.45:
            pool.release(rng.choice(list(pool.spans)))
        else:
            requested = rng.randrange(1, 500)
            before = dict(pool.spans)
            fits = pool.can_allocate(requested)
            got = pool.allocate(i, requested)
            assert fits == (got is not None)
            if got is None:
                assert pool.spans == before
            else:
                start, length = got
                assert start % 4 == length % 4 == 0
                assert length >= requested + guard
        spans = sorted(pool.spans.values())
        assert all(a + n <= b for (a, n), (b, _) in zip(spans, spans[1:]))
        assert all(a >= 0 and a + n <= pool.capacity for a, n in spans)
        assert pool.available == pool.capacity - sum(n for _, n in spans)
