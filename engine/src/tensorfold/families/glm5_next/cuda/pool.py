"""Deterministic contiguous reservations over the shared GLM token caches."""

from __future__ import annotations

from typing import Hashable


class TokenPool:
    """Aligned spans include guard rows for speculative writes and index-pool fences.

    The scheduler owns this allocator. All ranks execute the same operations; no
    tensor moves, context truncation or compaction can change a live address.
    """

    def __init__(self, capacity: int, guard: int = 8) -> None:
        if capacity < 1 or guard < 8:
            raise ValueError("a token pool needs positive capacity and at least eight guard rows")
        self.capacity = int(capacity)
        self.guard = (int(guard) + 3) // 4 * 4
        self.spans: dict[Hashable, tuple[int, int]] = {}

    def _size(self, tokens: int) -> int:
        if tokens < 1:
            raise ValueError("a request must reserve a positive number of tokens")
        return (int(tokens) + 3) // 4 * 4 + self.guard

    def _find(self, tokens: int) -> tuple[int, int] | None:
        size, at = self._size(tokens), 0
        for start, length in sorted(self.spans.values()):
            if start - at >= size:
                return at, size
            at = start + length
        return (at, size) if self.capacity - at >= size else None

    def can_allocate(self, tokens: int) -> bool:
        return self._find(tokens) is not None

    def allocate(self, key: Hashable, tokens: int) -> tuple[int, int] | None:
        if key in self.spans:
            raise ValueError("this request already holds a token reservation")
        span = self._find(tokens)
        if span is not None:
            self.spans[key] = span
        return span

    def release(self, key: Hashable) -> None:
        self.spans.pop(key, None)

    def growth(self, key: Hashable, tokens: int) -> tuple[int, int] | None:
        """Propose an in-place extension without publishing it."""
        start, old = self.spans[key]
        size = self._size(tokens)
        if size < old:
            raise ValueError("a grow cannot shrink an owned span")
        end = min((at for other, (at, _) in self.spans.items() if other != key and at > start),
                  default=self.capacity)
        return (start, size) if start + size <= end else None

    def grow(self, key: Hashable, expected: tuple[int, int], tokens: int) -> tuple[int, int]:
        if self.spans.get(key) != expected:
            raise RuntimeError("stale token span in grow transaction")
        span = self.growth(key, tokens)
        if span is None:
            raise RuntimeError("token span growth overlaps another owner")
        self.spans[key] = span
        return span

    @property
    def available(self) -> int:
        return self.capacity - sum(length for _, length in self.spans.values())
