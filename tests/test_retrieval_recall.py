# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the full-S retrieval-recall diagnostic logic

"""Tests for the pure aggregation logic in :mod:`tools.retrieval_recall`.

The retriever loop, model warm-up, and CLI are an I/O harness (they need the
cross-encoder and the 277 MB haystack) and are excluded from coverage; this
module pins the recall-curve and aggregation maths that the harness reports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "retrieval_recall",
    Path(__file__).resolve().parent.parent / "tools" / "retrieval_recall.py",
)
assert _SPEC and _SPEC.loader
rr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rr)


# --------------------------------------------------------------------------
# recall_curve
# --------------------------------------------------------------------------


def test_recall_curve_monotonic_non_decreasing():
    # gold sessions 11 and 30; ranked order surfaces 11 at pos2, 30 at pos4.
    ranked = [5, 11, 7, 30, 9, 1]
    curve = rr.recall_curve(ranked, [11, 30], ns=(1, 2, 4, 6))
    assert curve == {1: 0.0, 2: 0.5, 4: 1.0, 6: 1.0}
    # recall is non-decreasing in N
    vals = [curve[n] for n in (1, 2, 4, 6)]
    assert vals == sorted(vals)


def test_recall_curve_no_gold_is_full():
    assert rr.recall_curve([1, 2, 3], [], ns=(3,)) == {3: 1.0}


def test_recall_curve_gold_never_retrieved():
    assert rr.recall_curve([1, 2, 3], [99], ns=(3,)) == {3: 0.0}


def test_recall_curve_filters_none_gold():
    assert rr.recall_curve([1, 2], [None, 1], ns=(2,)) == {2: 1.0}


# --------------------------------------------------------------------------
# aggregate_recall
# --------------------------------------------------------------------------


def _rec(qtype, recall, candidates):
    return {"qtype": qtype, "recall": recall, "candidates": candidates}


def test_aggregate_groups_per_type_and_overall():
    ns = (5, 10)
    records = [
        _rec("multi-session", {5: 0.5, 10: 1.0}, 12),
        _rec("multi-session", {5: 1.0, 10: 1.0}, 8),
        _rec("temporal-reasoning", {5: 0.0, 10: 0.5}, 20),
    ]
    agg = rr.aggregate_recall(records, ns=ns)
    assert set(agg) == {"multi-session", "temporal-reasoning", "overall"}
    assert agg["multi-session"]["n"] == 2
    assert agg["overall"]["n"] == 3
    # mean@5 multi-session = (0.5+1.0)/2
    assert agg["multi-session"]["mean@5"] == pytest.approx(0.75)
    # mean@10 overall = (1.0+1.0+0.5)/3
    assert agg["overall"]["mean@10"] == pytest.approx(2.5 / 3)


def test_aggregate_full_rate_and_mean_candidates():
    ns = (5, 10)
    records = [
        _rec("multi-session", {5: 0.5, 10: 1.0}, 12),  # full@10
        _rec("multi-session", {5: 0.0, 10: 0.5}, 8),  # not full@10
    ]
    agg = rr.aggregate_recall(records, ns=ns)
    assert agg["multi-session"]["full@10"] == pytest.approx(0.5)
    assert agg["multi-session"]["mean_candidates"] == pytest.approx(10.0)


def test_aggregate_full_rate_all_recalled():
    agg = rr.aggregate_recall([_rec("x", {3: 1.0}, 5), _rec("x", {3: 1.0}, 5)], ns=(3,))
    assert agg["x"]["full@3"] == 1.0


# --------------------------------------------------------------------------
# _print_table (smoke — formatting must not raise)
# --------------------------------------------------------------------------


def test_print_table_renders(capsys):
    agg = rr.aggregate_recall([_rec("multi-session", {n: 0.5 for n in rr.RECALL_NS}, 15)])
    rr._print_table(agg)
    out = capsys.readouterr().out
    assert "multi-session" in out
    assert "overall" in out
    assert "mean@3" in out
