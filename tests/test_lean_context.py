# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for lean_context

from __future__ import annotations

from dataclasses import dataclass, field

from lean_context import (
    BiTemporalFact,
    Observation,
    _fact_date,
    build_lean_context,
)


@dataclass
class Fact:
    """Minimal AtomicFact-shaped bi-temporal fixture."""

    text: str
    valid_from: str = ""
    valid_until: str = ""
    date_mentions: list[str] = field(default_factory=list)
    session_date: str = ""
    confidence: float = 0.5


def test_protocol_conformance() -> None:
    f: BiTemporalFact = Fact("t")
    assert f.text == "t"


class TestFactDate:
    def test_mention_wins(self) -> None:
        assert (
            _fact_date(Fact("t", valid_from="2024-01-01", date_mentions=["2024-05-01"]))
            == "2024-05-01"
        )

    def test_valid_from_fallback(self) -> None:
        assert (
            _fact_date(Fact("t", valid_from="2024-01-01", session_date="2024-02-02"))
            == "2024-01-01"
        )

    def test_session_date_fallback(self) -> None:
        assert _fact_date(Fact("t", session_date="2024-02-02")) == "2024-02-02"


class TestObservationRender:
    def test_dated_current(self) -> None:
        assert (
            Observation("2024-05-01", "budget is $25k", False).render()
            == "- [2024-05-01] budget is $25k"
        )

    def test_undated_superseded(self) -> None:
        assert Observation("", "old value", True).render() == "- [undated] old value [superseded]"


class TestBuildLeanContext:
    def test_dedupe_keeps_highest_confidence(self) -> None:
        facts = [
            Fact("Budget is $20k", date_mentions=["2024-03-01"], confidence=0.4),
            Fact(
                "budget is $20k", date_mentions=["2024-03-01"], confidence=0.9
            ),  # dup, higher conf
        ]
        lc = build_lean_context(facts)
        assert len(lc.observations) == 1
        assert bool(lc) is True

    def test_skips_empty_text(self) -> None:
        lc = build_lean_context(
            [Fact("   "), Fact("real observation text", date_mentions=["2024-01-01"])]
        )
        assert len(lc.observations) == 1
        assert lc.observations[0].text == "real observation text"

    def test_superseded_marked_and_ordered_after_current(self) -> None:
        facts = [
            Fact("old budget $20k", date_mentions=["2024-05-01"], valid_until="2024-05-10"),
            Fact("new budget $25k", date_mentions=["2024-05-01"]),  # same date, current
        ]
        lc = build_lean_context(facts)
        # same date: current (not superseded) ranks before superseded
        assert lc.observations[0].text == "new budget $25k"
        assert lc.observations[0].superseded is False
        assert lc.observations[1].superseded is True

    def test_drop_superseded(self) -> None:
        facts = [
            Fact("expired", date_mentions=["2024-01-01"], valid_until="2024-02-01"),
            Fact("current", date_mentions=["2024-01-01"]),
        ]
        lc = build_lean_context(facts, drop_superseded=True)
        assert [o.text for o in lc.observations] == ["current"]

    def test_newest_first(self) -> None:
        facts = [
            Fact("older", date_mentions=["2024-01-01"]),
            Fact("newer", date_mentions=["2024-06-01"]),
        ]
        lc = build_lean_context(facts)
        assert [o.text for o in lc.observations] == ["newer", "older"]

    def test_max_observations_cap(self) -> None:
        facts = [Fact(f"obs {i}", date_mentions=[f"2024-0{i + 1}-01"]) for i in range(5)]
        lc = build_lean_context(facts, max_observations=2)
        assert len(lc.observations) == 2

    def test_char_budget_break(self) -> None:
        facts = [
            Fact("a fairly long first observation line here", date_mentions=["2024-02-01"]),
            Fact("a fairly long second observation line here", date_mentions=["2024-01-01"]),
        ]
        lc = build_lean_context(facts, char_budget=50)
        assert len(lc.observations) == 1  # only the first fits past the header

    def test_empty_input(self) -> None:
        lc = build_lean_context([])
        assert lc.observations == ()
        assert lc.rendered == ""
        assert bool(lc) is False

    def test_all_empty_text(self) -> None:
        lc = build_lean_context([Fact("  "), Fact("")])
        assert lc.rendered == ""

    def test_rendered_has_header(self) -> None:
        lc = build_lean_context([Fact("real observation", date_mentions=["2024-01-01"])])
        assert lc.rendered.startswith("OBSERVATIONS")
        assert "- [2024-01-01] real observation" in lc.rendered
