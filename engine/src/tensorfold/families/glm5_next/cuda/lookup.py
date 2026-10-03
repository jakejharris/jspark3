"""Drafts copied from the context: when the last tokens also stand earlier in the prompt or the reply, the tokens that
followed them there are proposed (a model rewriting a file, quoting code or a tool result repeats long runs).

Drafts only propose: every round verifies them against the model's own keyed samples, so replies stay the ones
serial decoding gives.
"""

from __future__ import annotations

import os

import numpy as np

GRAM = 3        # tokens a match is found by
# Measured on file rewrites (GLM-5.3-Flash, two Sparks): 7 drafts once 8 tokens agree gave 1.12-1.38x the MTP-only
# speed; 5 drafts 1.05-1.18x; 5 agreeing tokens also copied into fresh code (3% slower there)
AGREE = int(os.environ.get("TF_GLM_LOOKUP_AGREE", "8"))    # tokens that must agree (the gram and before)
DRAFTS = int(os.environ.get("TF_GLM_LOOKUP_DRAFTS", "7"))  # at most (the widest verify window, 8 rows)
REACH = 16      # tokens compared backwards


class PromptLookup:
    """An index of the context's 3-grams (their latest start); ``propose`` after ``extend``."""

    def __init__(self, prompt, *, agree: int = AGREE, drafts: int = DRAFTS) -> None:
        self.ids: list[int] = [int(t) for t in prompt]
        self.agree, self.most = agree, drafts
        self.index: dict[tuple, int] = {}
        a = np.asarray(self.ids, dtype=np.int64)
        if len(a) > GRAM:
            # every gram whose next token is known, latest start last (so it wins)
            grams = zip(a[:-GRAM].tolist(), a[1:-GRAM + 1].tolist(), a[2:-GRAM + 2].tolist())
            self.index = {g: i for i, g in enumerate(grams)}

    def extend(self, tokens) -> None:
        for t in tokens:
            self.ids.append(int(t))
            j = len(self.ids) - 1                  # the gram ending just before t now has a next token: t
            if j >= GRAM:
                self.index[tuple(self.ids[j - GRAM:j])] = j - GRAM

    def propose(self, room: int) -> list[int]:
        """Up to ``room`` (and DRAFTS) tokens that followed the last tokens earlier, when at least AGREE agree."""

        return self.match(room)[0]

    def match(self, room: int) -> tuple[list[int], int]:
        """The proposal and its backward match length, for callers that price copy drafts."""

        ids = self.ids
        if len(ids) < max(GRAM, self.agree) or room <= 0:
            return [], 0
        start = self.index.get(tuple(ids[-GRAM:]))
        if start is None:
            return [], 0
        end = start + GRAM                          # the earlier run's next token
        n = len(ids)
        k = GRAM
        while k < REACH and start - (k - GRAM) - 1 >= 0 and ids[start - (k - GRAM) - 1] == ids[n - k - 1]:
            k += 1
        if k < self.agree:
            return [], k
        stop = min(end + min(room, self.most), n - 1)
        return (ids[end:stop] if stop > end else []), k
