"""Copy drafting continuation learning and measured-price source selection (no GPU)."""

import copy

import pytest

from tensorfold.families.glm5_next.cuda.copy_drafts import CopyDrafts, CopyOffer, CopyPrices, prefer_copy


def price_data(*, loads=(1,), verify=10., draft=1000., lookup=0.):
    return {"version": 1, "build": "test-build", "weights": "test-weights", "drafter": "test-drafter", "cells": [
        {"context": "short", "load": load, "mode": "eager", "source": "synthetic CPU gate, not a boot price",
         "slot": slot, "reps": 3,
         "verify_ms": {str(i): verify for i in range(load, load * 8 + 1)},
         "draft_ms": {"f": [0.] + [draft] * 7, "m": [1.] + [draft] * 7}, "lookup_ms": lookup}
        for load in loads for slot in (("zero", "nonzero") if load == 1 else ("zero",))]}


def test_copy_rates_censor_unreached_rows_and_reset_per_request():
    state = CopyDrafts([])
    offer = CopyOffer([10, 11, 12, 13], 8, 0, [.5, .25, .125, .0625])
    state.selected(offer, 4)
    state.observe(4, 1)
    assert state.rates == {(8, 0): [1, 1], (8, 1): [0, 1]}
    assert state.width == 2 and state.run == 0
    assert CopyDrafts([]).rates == {}
    state.selected(offer, 4)
    state.observe(4, 1, censored=True)
    assert state.rates == {(8, 0): [2, 2], (8, 1): [0, 1]}


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_only_full_accepts_widen_to_the_configured_ceiling(rows):
    s = CopyDrafts([], max_rows=rows)
    sizes = [2, 4, min(8, rows - 1), min(16, rows - 1), rows - 1]
    for n in sizes:
        s.selected(CopyOffer(list(range(n)), 16, s.run, [.9] * n), n)
        s.observe(n, n)
    assert s.width == rows - 1 and s.run == sum(sizes)
    assert any(match == 16 and run > 0 for match, run in s.rates)
    s.selected(CopyOffer(list(range(rows - 1)), 16, s.run, [.9] * (rows - 1)), rows - 1)
    s.observe(rows - 1, 1)
    assert s.width == 2 and s.run == 0
    with pytest.raises(ValueError, match="rows"):
        CopyDrafts([], max_rows=64)


def test_lookup_boundary_overlap_and_only_new_outputs_once():
    s = CopyDrafts([1] * 12, max_rows=8)
    offer = s.offer([1], 7)
    # Upstream's conservative overlap bound offers no unknown continuation.
    assert len(s.lookup.ids) == 13 and offer.tokens == [] and offer.match == 12
    s.selected(CopyOffer([1, 1], 12, 0, [.5, .25]), 2)
    s.observe(2, 0)
    s.offer([9], 7)
    assert s.lookup.ids == [1] * 13 + [9]
    assert s.offer([], 0).tokens == []
    other = CopyDrafts([1] * 8 + [2] * 8)
    other.offer([2], 7)
    assert s.lookup.ids[-1] == 9 and other.lookup.ids[-1] == 2


def test_image_placeholders_are_never_proposed():
    s = CopyDrafts(list(range(8)) + [9, -7, 3] + list(range(8)))
    assert s.offer([], 7).tokens == [9]


def test_price_rejects_expensive_copy_and_missing_cell_and_prices_steps():
    offer = CopyOffer(list(range(7)), 8, 0, [.6, .5, .4, .3, .2, .1, .05])
    kwargs = dict(context=64, load=1, mode="eager", arm="f", model_chances=[.2] * 7)
    prices = CopyPrices(price_data())
    assert prices.choose(offer, **kwargs) == 7
    assert prices.choose(offer, **{**kwargs, "context": 9000}) == 0
    # Negative control for fresh-code pricing: a costly/weak copy must lose,
    # despite a valid eight-token match. An unpriced policy would still copy it.
    expensive = CopyPrices(price_data(draft=0., lookup=1000.))
    assert expensive.choose(offer, **kwargs) == 0
    steps = price_data(draft=5.)
    steps["cells"][0]["verify_ms"] = {"1": 10., "2": 13., "3": 13., "8": 100.}
    assert CopyPrices(steps).choose(offer, **kwargs) == 2


def test_graph_to_eager_prices_and_fixed_peer_costs():
    data = price_data(draft=5.)
    graph = copy.deepcopy(data["cells"][0])
    graph.update(mode="graph-main", verify_ms={str(n): 10. for n in range(1, 7)})
    data["cells"].append(graph)
    data["cells"][0]["verify_ms"] = {"7": 100., "8": 100.}
    offer = CopyOffer(list(range(7)), 16, 5, [.99] * 7)
    assert CopyPrices(data).choose(offer, context=64, load=1, mode=lambda n: "graph-main" if n <= 6 else "eager",
                                   arm="f", model_chances=[.5] * 7) == 5
    assert CopyPrices(data).choose(offer, context=64, load=1, mode="graph-sparse:256",
                                   arm="f", model_chances=[.5] * 7) == 0
    sparse = copy.deepcopy(graph)
    sparse.update(mode="graph-sparse:256", verify_ms={str(n): 10. for n in range(1, 9)})
    data["cells"].append(sparse)
    assert CopyPrices(data).choose(offer, context=64, load=1, mode="graph-sparse:512",
                                   arm="f", model_chances=[.5] * 7) == 0
    mixed = CopyPrices(price_data(loads=(8,)))
    assert mixed.choose(offer, context=64, load=8, mode="eager", arm="f", model_chances=[.5] * 7,
                        peer_rows=56, peer_tokens=10., peer_ms=100.) == 7


def test_shared_source_rule_uses_one_base_and_ties_go_to_copy():
    from test_glm_draft_pricing import table

    prices = table({3: 10, 4: 10.1, 5: 10.2, 6: 30}, load=2)
    kwargs = dict(base_rows=3, base_expected=2.5, context="short", load=2, path_for_rows=lambda n: "eager")
    selected, detail = prefer_copy(prices, [.8, .64, .5], [.9, .81], **kwargs)
    assert selected and detail["copy_rate"] > detail["model_rate"]
    assert prefer_copy(prices, [.8, .64], [.8, .64], **kwargs)[0]
    assert not prefer_copy(prices, [.9, .81], [.1, .01], **kwargs)[0]
    with pytest.raises(ValueError, match="lacks"):
        prefer_copy(prices, [.8], [.9], **{**kwargs, "load": 8})


@pytest.mark.parametrize("field,value", [("lookup_ms", -1), ("lookup_ms", float("nan")), ("source", ""),
                                         ("verify_ms", {"1": float("inf")}), ("verify_ms", {"9": 1}),
                                         ("draft_ms", {"f": [0, -1]})])
def test_invalid_price_evidence_is_refused(field, value):
    data = price_data()
    data["cells"][0][field] = value
    with pytest.raises(ValueError):
        CopyPrices(data)


def replay_prior(rate=.9):
    return {"schema": "glm-copy-continuation-prior-v1", "tokenizer_sha256": "a" * 64,
            "lookup": {"gram": 3, "agree": 8, "reach": 16, "max_drafts": 7, "latest_start_wins": True},
            "match_x_run": {m: {r: {"price_rate": rate, "offered": 100000, "accepted": 1}
                                 for r in [str(i) for i in range(16)] + ["16+"]}
                            for m in [str(i) for i in range(8, 16)] + ["16+"]}}


def test_copy_prior_is_private_per_request_and_not_frozen_by_replay_size():
    prior = replay_prior()
    a, b = CopyDrafts([], prior=prior), CopyDrafts([], prior=prior)
    a.lookup.match = b.lookup.match = lambda room: ([9], 8)
    assert a.offer([], 1).chances == [.9]  # use price_rate, not raw sparse counts
    a.selected(CopyOffer([9], 8, 0, [.9]), 1)
    a.observe(1, 0)
    assert a.offer([], 1).chances == [.8]
    assert b.offer([], 1).chances == [.9] and b.rates == {}
    assert a.prior_rates is not b.prior_rates


def test_prior_exact_run_buckets_and_unavailable_lookup_reset():
    prior = replay_prior()
    prior["match_x_run"]["16+"]["15"]["price_rate"] = .8
    prior["match_x_run"]["16+"]["16+"]["price_rate"] = .95
    s = CopyDrafts([], prior=prior)
    s.run = 15
    s.lookup.match = lambda room: ([11, 12, 13], 16)
    assert s.offer([], 3).chances == pytest.approx([.8, .8 * .95, .8 * .95**2])
    s.selected(s.offer([], 3), 3)
    s.observe(3, 3)
    assert s.run == 18 and s.rates == {(16, 15): [1, 1], (16, 16): [2, 2]}
    s.lookup.match = lambda room: ([], 0)
    assert s.offer([], 7).run == 0 and s.run == 0


@pytest.mark.parametrize("change", ["schema", "agreement", "rate", "missing"])
def test_incompatible_or_malformed_prior_is_refused(change):
    prior = replay_prior()
    if change == "schema":
        prior["schema"] = "diagnostic-one-step-probes"
    elif change == "agreement":
        prior["lookup"]["agree"] = 7
    elif change == "rate":
        prior["match_x_run"]["8"]["0"]["price_rate"] = float("nan")
    else:
        del prior["match_x_run"]["16+"]["16+"]
    with pytest.raises((ValueError, KeyError)):
        CopyDrafts([], prior=prior)
