"""Host checks for the draft pricing decision and its explicit negative controls."""

import pytest

from tensorfold.families.glm5_next.cuda.draft_pricing import (
    Calibration, PriceTable, SegmentTracker, choose_depths, cumulative, threshold_depth,
)


def table(costs, *, load=1, context="short"):
    return PriceTable({"schema": "tf-price/v1", "build": "build", "weights": "weights",
                       "drafter": "drafter", "kind": "measured",
                       "cells": [{"rows": n, "ctx": context, "load": load,
                                  "path": "graph-main" if load == 1 and n <= 6 else "eager",
                                  "median_ms": ms, "spread": 0.1, "reps": 30}
                                 for n, ms in costs.items()]},
                      weights="weights", drafter="drafter", build="build")


def test_staircase_argmax_can_take_two_rows_when_first_row_loses():
    prices = table({1: 10, 2: 20, 3: 21})
    depths, detail = choose_depths([{"sid": 4, "pass_chances": [.6, .58]}], prices,
                                    context="short", load=1, base_rows=1,
                                    path_for_rows=lambda n: "graph-main")
    assert detail["trace"][1]["rate"] < detail["trace"][0]["rate"]
    assert depths == {4: 2}


def test_joint_tie_break_and_prefix_are_deterministic():
    prices = table({2: 10, 3: 10.5, 4: 50, 5: 50, 6: 50}, load=2)
    offers = [{"sid": 9, "pass_chances": [.7, .4]}, {"sid": 3, "pass_chances": [.7, .4]}]
    a, detail = choose_depths(offers, prices, context="short", load=2, base_rows=2)
    b, _ = choose_depths(list(reversed(offers)), prices, context="short", load=2, base_rows=2)
    assert a == b == {9: 0, 3: 1}
    assert detail["cut"] == 1


def test_reached_row_calibration_censors_tail_and_caps_eos():
    c = Calibration()
    c.update("think", [.8, .7, .6], accepted=1, rejected=True)
    assert c.counts["think"][0][1] == 1
    assert c.counts["think"][1][0] == .7
    assert c.counts["think"][2] == [0, 0]
    c.update("text", [.8, .7], accepted=1, rejected=False)
    assert c.counts["text"][1] == [0, 0]


def test_threshold_is_a_prefix_of_full_proposal():
    claims = [.91, .2, .9, .8]
    depth = threshold_depth(claims, .3)
    assert depth == 1
    full = [13, 14, 15, 16]
    assert full[:depth] == [13]
    assert cumulative(claims)[1] < .3


def test_price_table_steps_up_and_rejects_mislabeled_context_or_load():
    prices = table({1: 10, 4: 15})
    assert prices.lookup(2, "short", 1, "graph-main") == 15
    with pytest.raises(ValueError, match="lacks"):
        prices.lookup(2, "8k", 1, "graph-main")
    with pytest.raises(ValueError, match="lacks"):
        prices.lookup(1, "short", 8, "graph-main")


def test_missing_graph_width_does_not_hide_later_eager_width():
    prices = table({1: 10, 3: 2})
    prices.cells[1]["path"] = "eager"
    depths, detail = choose_depths([{"sid": 1, "pass_chances": [.6, .5]}], prices,
                                    context="short", load=1, base_rows=1,
                                    path_for_rows=lambda n: "graph-main" if n <= 2 else "eager")
    assert depths == {1: 2}
    assert [row["rows"] for row in detail["trace"]] == [1, 3]
    with pytest.raises(ValueError, match="lacks"):
        prices.lookup(2, "short", 1, "graph-main")


def test_row_segments_and_direct_eos_outcomes():
    patterns = {"think_open": [1], "think_close": [2], "tool_open": [3], "tool_close": [4]}
    tracker = SegmentTracker(patterns, [1])
    segments = tracker.row_segments([2, 8, 3, 9])
    assert segments == ["think", "text", "text", "tool"]
    cal = Calibration()
    assert cal.update_rows(segments, [.8, .7, .6, .5], [2, 8, 3, 9],
                           [2, 8, 3, 9], 5, (8,)) == ["match", "eos-match", "censored", "censored"]
    assert cal.counts["text"][1][1] == 1
    assert cal.counts["text"][2] == [0, 0]
    assert cal.update_rows(segments, [.8], [2], [5], 2, ()) == ["mismatch"]
    assert cal.update_rows(segments, [.8], [2], [2], 1, ()) == ["censored"]


def test_segment_tracker_uses_token_sequences_and_falls_back_to_text():
    patterns = {"think_open": [1, 2], "think_close": [3, 4],
                "tool_open": [5, 6], "tool_close": [7, 8]}
    s = SegmentTracker(patterns, [99, 1, 2])
    assert s.segment == "think"
    assert s.feed([3]) == "think"
    assert s.feed([4]) == "text"
    assert s.feed([5, 6]) == "tool"
    assert s.feed([7, 8]) == "text"
    assert SegmentTracker(None, [1, 2]).segment == "text"


def test_table_schema_identity_and_nonfinite_values_fail_closed():
    base = {"schema": "tf-price/v1", "build": "build", "weights": "weights",
            "drafter": "drafter", "kind": "measured",
            "cells": [{"rows": 1, "load": 1, "ctx": "short", "path": "graph-main",
                       "median_ms": 10, "spread": 1, "reps": 30}]}
    for changes in ({"schema": "wrong"}, {"weights": "wrong"}, {"build": "wrong"}, {"build": ""}):
        with pytest.raises(ValueError):
            PriceTable(dict(base, **changes), weights="weights", drafter="drafter", build="build")
    for changes in ({"median_ms": float("nan")}, {"reps": 29}):
        with pytest.raises(ValueError):
            PriceTable(dict(base, cells=[dict(base["cells"][0], **changes)]),
                       weights="weights", drafter="drafter", build="build")
