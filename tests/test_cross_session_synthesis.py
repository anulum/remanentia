# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for cross_session_synthesis

from __future__ import annotations

from dataclasses import dataclass, field

from cross_session_synthesis import (
    EntityDigest,
    StatementLine,
    SynthesisResult,
    SynthFact,
    _dedupe_entities,
    _digest_entity,
    _normalise_statement,
    _significant_tokens,
    _statement_date,
    focus_entities,
    synthesise,
)


@dataclass
class Fact:
    """Minimal AtomicFact-shaped fixture for the synthesiser."""

    text: str
    session_idx: int
    entities: list[str]
    date_mentions: list[str] = field(default_factory=list)
    role: str = "user"
    fact_type: str = "state"
    valid_from: str = ""
    session_date: str = ""


class TestSynthFactProtocol:
    def test_fixture_satisfies_protocol(self) -> None:
        """SynthFact is a runtime-checkable structural protocol."""
        assert isinstance(Fact("t", 0, []), SynthFact)


# ─── token / entity selection ─────────────────────────────────────────


class TestSignificantTokens:
    def test_drops_short_and_stopwords(self) -> None:
        assert _significant_tokens("What is the latest budget plan") == {"budget", "plan"}

    def test_keeps_alphanumeric(self) -> None:
        assert _significant_tokens("iPad2024 model") == {"ipad2024", "model"}

    def test_empty_when_all_filtered(self) -> None:
        assert _significant_tokens("what is the most") == set()


class TestDedupeEntities:
    def test_case_insensitive_dedupe_keeps_first(self) -> None:
        assert _dedupe_entities(["Kitchen", "kitchen", "KITCHEN"]) == ["Kitchen"]

    def test_collapses_whitespace(self) -> None:
        assert _dedupe_entities(["  the   project  "]) == ["the project"]

    def test_skips_blank(self) -> None:
        assert _dedupe_entities(["", "   ", "iPad"]) == ["iPad"]


class TestFocusEntities:
    def test_matches_named_entity(self) -> None:
        out = focus_entities(
            "What is my kitchen renovation budget?",
            ["kitchen renovation", "garage", "iPad"],
        )
        assert out == ["kitchen renovation"]

    def test_no_question_tokens_returns_empty(self) -> None:
        # Question made only of short/stop words -> no binding tokens.
        assert focus_entities("what is the most?", ["kitchen renovation"]) == []

    def test_unrelated_entities_excluded(self) -> None:
        assert focus_entities("Tell me about my car", ["bicycle", "boat"]) == []


# ─── dates / normalisation ────────────────────────────────────────────


class TestStatementDate:
    def test_prefers_explicit_mention(self) -> None:
        f = Fact("t", 0, ["e"], date_mentions=["2024-05-01"], valid_from="2024-01-01")
        assert _statement_date(f) == "2024-05-01"

    def test_falls_back_to_valid_from(self) -> None:
        f = Fact("t", 0, ["e"], valid_from="2024-01-01", session_date="2024-02-02")
        assert _statement_date(f) == "2024-01-01"

    def test_falls_back_to_session_date(self) -> None:
        f = Fact("t", 0, ["e"], session_date="2024-02-02")
        assert _statement_date(f) == "2024-02-02"

    def test_empty_when_no_date(self) -> None:
        assert _statement_date(Fact("t", 0, ["e"])) == ""


class TestNormaliseStatement:
    def test_collapses_and_strips(self) -> None:
        assert _normalise_statement("  The  Budget is  $20.  ") == "the budget is $20"


# ─── StatementLine / EntityDigest rendering ───────────────────────────


class TestStatementLine:
    def test_sort_key_dated(self) -> None:
        assert StatementLine("2024-01-01", 2, "x")._sort_key() == (0, "2024-01-01", 2)

    def test_sort_key_undated(self) -> None:
        assert StatementLine("", 3, "x")._sort_key() == (1, "", 3)

    def test_render_dated_most_recent(self) -> None:
        line = StatementLine("2024-05-01", 1, "budget is $25k")
        assert line.render(most_recent=True) == "  - [2024-05-01] budget is $25k  (most recent)"

    def test_render_undated_not_recent(self) -> None:
        line = StatementLine("", 2, "budget is $20k")
        assert line.render(most_recent=False) == "  - [session 3] budget is $20k"


class TestEntityDigest:
    def test_most_recent_index_with_dates(self) -> None:
        d = EntityDigest(
            "e",
            [StatementLine("2024-01-01", 0, "a"), StatementLine("2024-02-01", 1, "b")],
        )
        assert d.most_recent_index() == 1

    def test_most_recent_index_all_undated(self) -> None:
        d = EntityDigest("e", [StatementLine("", 0, "a"), StatementLine("", 1, "b")])
        assert d.most_recent_index() == -1

    def test_render_single_statement_has_no_recent_tag(self) -> None:
        # len == 1 path: the (most recent) tag is suppressed.
        d = EntityDigest("e", [StatementLine("2024-01-01", 0, "a")])
        assert d.render() == "• e:\n  - [2024-01-01] a"

    def test_render_marks_latest(self) -> None:
        d = EntityDigest(
            "kitchen",
            [StatementLine("2024-01-01", 0, "a"), StatementLine("2024-02-01", 1, "b")],
        )
        assert d.render().endswith("(most recent)")


# ─── _digest_entity ───────────────────────────────────────────────────


class TestDigestEntity:
    def test_collects_dedupes_and_orders(self) -> None:
        facts: list[SynthFact] = [
            Fact("budget is $25k", 3, ["kitchen"], date_mentions=["2024-05-01"]),
            Fact("budget is $20k", 1, ["kitchen"], date_mentions=["2024-03-01"]),
            Fact("BUDGET IS $20K", 1, ["kitchen"], date_mentions=["2024-03-01"]),  # dup
            Fact("unrelated", 0, ["garage"]),  # other entity
        ]
        d = _digest_entity("kitchen", facts, max_statements=4)
        assert [s.text for s in d.statements] == ["budget is $20k", "budget is $25k"]

    def test_skips_blank_text(self) -> None:
        facts: list[SynthFact] = [
            Fact("   ", 0, ["kitchen"]),
            Fact("real fact", 1, ["kitchen"]),
        ]
        d = _digest_entity("kitchen", facts, max_statements=4)
        assert [s.text for s in d.statements] == ["real fact"]

    def test_truncates_to_max_statements(self) -> None:
        facts: list[SynthFact] = [
            Fact(f"fact {i}", i, ["kitchen"], date_mentions=[f"2024-0{i + 1}-01"]) for i in range(5)
        ]
        d = _digest_entity("kitchen", facts, max_statements=2)
        assert len(d.statements) == 2


# ─── SynthesisResult ──────────────────────────────────────────────────


class TestSynthesisResult:
    def test_truthy_with_entities(self) -> None:
        assert bool(SynthesisResult([EntityDigest("e", [])], "msg")) is True

    def test_falsy_without_entities(self) -> None:
        assert bool(SynthesisResult([], "")) is False


# ─── synthesise (public entry point) ──────────────────────────────────


class TestSynthesise:
    def _kitchen_facts(self) -> list[SynthFact]:
        return [
            Fact("Kitchen renovation budget is $20,000", 1, ["kitchen renovation"], ["2024-03-01"]),
            Fact(
                "Raised kitchen renovation budget to $25,000",
                3,
                ["kitchen renovation"],
                ["2024-05-10"],
            ),
        ]

    def test_consolidates_knowledge_update(self) -> None:
        r = synthesise(
            "What is my current kitchen renovation budget?",
            self._kitchen_facts(),
            qtype="knowledge-update",
        )
        assert r is not None
        assert "ENTITY SUMMARY" in r.message
        assert r.message.rstrip().endswith("(most recent)")
        assert len(r.entities) == 1

    def test_temporal_gated(self) -> None:
        assert synthesise("q kitchen", self._kitchen_facts(), qtype="temporal-reasoning") is None

    def test_empty_facts(self) -> None:
        assert synthesise("q kitchen", [], qtype="multi-session") is None

    def test_no_focus_entity(self) -> None:
        assert (
            synthesise("unrelated weather forecast?", self._kitchen_facts(), qtype="multi-session")
            is None
        )

    def test_single_statement_entity_dropped(self) -> None:
        facts: list[SynthFact] = [Fact("Bought an iPad", 0, ["iPad"], ["2024-01-01"])]
        assert synthesise("Which iPad did I buy?", facts, qtype="multi-session") is None

    def test_char_budget_break(self) -> None:
        # Two qualifying entities, but the budget admits only the first block.
        facts: list[SynthFact] = [
            Fact("kitchen budget low", 0, ["kitchen"], ["2024-01-01"]),
            Fact("kitchen budget high", 1, ["kitchen"], ["2024-02-01"]),
            Fact("garage budget low", 0, ["garage"], ["2024-01-01"]),
            Fact("garage budget high", 1, ["garage"], ["2024-02-01"]),
        ]
        r = synthesise("kitchen and garage budget?", facts, qtype="multi-session", char_budget=80)
        assert r is not None
        assert len(r.entities) == 1

    def test_max_entities_cap(self) -> None:
        facts: list[SynthFact] = [
            Fact("kitchen budget low", 0, ["kitchen"], ["2024-01-01"]),
            Fact("kitchen budget high", 1, ["kitchen"], ["2024-02-01"]),
            Fact("garage budget low", 0, ["garage"], ["2024-01-01"]),
            Fact("garage budget high", 1, ["garage"], ["2024-02-01"]),
        ]
        r = synthesise("kitchen and garage budget?", facts, qtype="multi-session", max_entities=1)
        assert r is not None
        assert len(r.entities) == 1

    def test_orders_entities_by_evidence(self) -> None:
        facts: list[SynthFact] = [
            Fact("garage door fact one", 0, ["garage"], ["2024-01-01"]),
            Fact("garage door fact two", 1, ["garage"], ["2024-02-01"]),
            Fact("kitchen fact one", 0, ["kitchen"], ["2024-01-01"]),
            Fact("kitchen fact two", 1, ["kitchen"], ["2024-02-01"]),
            Fact("kitchen fact three", 2, ["kitchen"], ["2024-03-01"]),
        ]
        r = synthesise("kitchen and garage?", facts, qtype="multi-session")
        assert r is not None
        # kitchen has 3 statements, garage 2 -> kitchen first.
        assert r.entities[0].entity == "kitchen"
        assert r.entities[1].entity == "garage"
