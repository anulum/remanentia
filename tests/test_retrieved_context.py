# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for retrieved-context assembly

"""Tests for :mod:`retrieved_context` — session selection, budgeting, recall."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrieved_context import (
    RetrievedContext,
    build_selection_diagnostics,
    gold_session_recall,
    rank_ordered_sessions,
    select_sessions,
)


@dataclass
class _Fact:
    session_idx: int


@dataclass
class _Res:
    fact: _Fact


def _results(*session_idxs: int) -> list[_Res]:
    """Build ranked results from a sequence of source-session indices."""
    return [_Res(_Fact(i)) for i in session_idxs]


def _sessions(
    n: int, *, turns: int = 1, content: str = "hello world"
) -> list[list[dict[str, str]]]:
    """n sessions, each with `turns` user turns of fixed content."""
    return [[{"role": "user", "content": content} for _ in range(turns)] for _ in range(n)]


# --------------------------------------------------------------------------
# rank_ordered_sessions
# --------------------------------------------------------------------------


def test_rank_order_dedups_keeping_first_appearance() -> None:
    res = _results(5, 2, 5, 7, 2, 9)
    assert rank_ordered_sessions(res) == [5, 2, 7, 9]


def test_rank_order_empty() -> None:
    assert rank_ordered_sessions([]) == []


def test_rank_order_single_session_many_facts() -> None:
    assert rank_ordered_sessions(_results(3, 3, 3)) == [3]


# --------------------------------------------------------------------------
# select_sessions — selection, capping, budget, ordering
# --------------------------------------------------------------------------


def test_select_caps_at_max_sessions_by_rank() -> None:
    res = _results(0, 1, 2, 3, 4)
    dates = ["2023-01-0%d" % (i + 1) for i in range(5)]
    out = select_sessions(res, _sessions(5), session_dates=dates, max_sessions=3)
    assert set(out.selected_session_idxs) == {0, 1, 2}
    assert out.n_candidate_sessions == 5
    assert out.dropped_to_session_limit == [3, 4]


def test_select_renders_chronologically_oldest_first() -> None:
    # Ranked 2,0,1 but dated so chronological display is 0,1,2.
    res = _results(2, 0, 1)
    dates = ["2023-03-01", "2023-01-01", "2023-02-01"]  # idx0=Mar, idx1=Jan, idx2=Feb
    out = select_sessions(res, _sessions(3), session_dates=dates, max_sessions=3)
    # Chronological: idx1 (Jan) < idx2 (Feb) < idx0 (Mar)
    assert out.selected_session_idxs == [1, 2, 0]
    # Session 1 header carries its date; ordering reflected in the text.
    assert out.session_text.index("2023-01-01") < out.session_text.index("2023-03-01")


def test_select_drops_lowest_rank_to_meet_char_budget() -> None:
    # 4 fat sessions; budget admits only the first two by rank.
    res = _results(0, 1, 2, 3)
    big = _sessions(4, turns=1, content="x" * 500)
    out = select_sessions(res, big, max_sessions=4, char_budget=1200)
    # First two kept (rank 0,1); 2 and 3 dropped to budget.
    assert set(out.selected_session_idxs) == {0, 1}
    assert set(out.dropped_to_budget) == {2, 3}


def test_select_keeps_at_least_the_top_session_over_budget() -> None:
    # A single session larger than the whole budget is still kept (kept is
    # empty when its turn comes, so the guard `if kept and ...` admits it).
    res = _results(0)
    huge = _sessions(1, content="y" * 10_000)
    out = select_sessions(res, huge, max_sessions=5, char_budget=100)
    assert out.selected_session_idxs == [0]
    assert out.dropped_to_budget == []


def test_select_ignores_stale_session_indices() -> None:
    res = _results(0, 99, 1)  # 99 is out of range
    out = select_sessions(res, _sessions(2), max_sessions=5)
    assert set(out.selected_session_idxs) == {0, 1}


def test_select_empty_results_yields_empty_context() -> None:
    out = select_sessions([], _sessions(3), max_sessions=5)
    assert out.selected_session_idxs == []
    assert out.session_text == ""
    assert out.n_candidate_sessions == 0


def test_select_renders_role_and_content() -> None:
    res = _results(0)
    sess = [[{"role": "assistant", "content": "the answer is 42"}]]
    out = select_sessions(res, sess, max_sessions=1)
    assert "[ASSISTANT]: the answer is 42" in out.session_text
    assert "=== Session 1 ===" in out.session_text  # undated → no date in header


def test_select_undated_sessions_sort_last_but_included() -> None:
    res = _results(0, 1)
    dates = ["", "2023-01-01"]  # idx0 undated, idx1 dated
    out = select_sessions(res, _sessions(2), session_dates=dates, max_sessions=2)
    # Dated session sorts before undated (\xff sentinel sorts last).
    assert out.selected_session_idxs == [1, 0]


def test_returns_retrieved_context_type() -> None:
    out = select_sessions(_results(0), _sessions(1))
    assert isinstance(out, RetrievedContext)


def test_build_selection_diagnostics_maps_answer_session_ids() -> None:
    """Diagnostics identify selected, capped, and budget-dropped answer IDs."""
    context = RetrievedContext(
        session_text="",
        selected_session_idxs=[0, 2],
        n_candidate_sessions=5,
        dropped_to_session_limit=[3],
        dropped_to_budget=[4],
    )
    diagnostics = build_selection_diagnostics(
        context,
        haystack_session_ids=["s0", "s1", "s2", "s3", "s4"],
        answer_session_ids=["s2", "s3", "s4", "missing"],
    )

    assert diagnostics.answer_session_recall == pytest.approx(0.25)
    assert diagnostics.selected_answer_session_ids == ["s2"]
    assert diagnostics.missing_answer_session_ids == ["s3", "s4", "missing"]
    assert diagnostics.session_limited_answer_session_ids == ["s3"]
    assert diagnostics.budget_dropped_answer_session_ids == ["s4"]
    assert diagnostics.as_json_dict()["candidate_session_count"] == 5


def test_build_selection_diagnostics_without_answer_ids_has_no_recall() -> None:
    """Rows without answer-session IDs keep selection evidence but no recall."""
    context = select_sessions(_results(1, 0), _sessions(2), max_sessions=1)
    diagnostics = build_selection_diagnostics(context, haystack_session_ids=["s0", "s1"])

    assert diagnostics.answer_session_recall is None
    assert diagnostics.selected_session_idxs == [1]
    assert diagnostics.dropped_to_session_limit == [0]


# --------------------------------------------------------------------------
# gold_session_recall
# --------------------------------------------------------------------------


def test_gold_session_recall() -> None:
    """Gold recall handles full, partial, missing, and empty answer-session sets."""
    cases: list[tuple[list[int], list[int | None], float]] = [
        ([1, 2, 3], [2, 3], 1.0),  # all gold retrieved
        ([1, 2], [2, 3], 0.5),  # half retrieved
        ([4, 5], [2, 3], 0.0),  # pure retrieval miss
        ([], [2, 3], 0.0),  # nothing selected
        ([1, 2, 3], [], 1.0),  # no gold: vacuously complete
        ([1, 2, 3], [None, 2], 1.0),  # None gold ids filtered out
        ([1], [None], 1.0),  # all-None gold: vacuous
    ]
    for selected, gold, expected in cases:
        assert gold_session_recall(selected, gold) == expected


def test_recall_partial_three_gold() -> None:
    assert gold_session_recall([11, 30], [11, 19, 30]) == pytest.approx(2 / 3)
