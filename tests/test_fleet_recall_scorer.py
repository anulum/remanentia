# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for fleet_recall_scorer

"""Multi-angle tests for the fleet-fed recall axis scorer."""

from __future__ import annotations

from pathlib import Path

from fleet_recall_scorer import FleetRecallReport, report_from_ledger, score_fleet_recall
from recall_ledger import RecallLedger, RecallQuery


def _query(
    *,
    by: str = "agent-a",
    found: bool = True,
    abstained: bool | None = None,
    was_used: bool | None = None,
    was_correct: bool | None = None,
    event_id: str = "e1",
) -> RecallQuery:
    return RecallQuery(
        event_id=event_id,
        ts=1.0,
        by=by,
        query="q",
        top_k=5,
        project="",
        returned_ids=("src:a",) if found else (),
        found=found,
        score=1.0 if found else None,
        abstained=abstained,
        was_used=was_used,
        was_correct=was_correct,
    )


class TestScoreFleetRecall:
    def test_empty_stream_is_unmeasured_not_perfect(self) -> None:
        report = score_fleet_recall([])
        assert report.measured is False
        assert report.queries == 0
        assert report.answered_rate == 0.0
        assert report.usage_rate == 0.0
        assert report.fleet_accuracy == 0.0
        assert report.fleet_fed is False

    def test_single_agent_stream_is_not_fleet_fed(self) -> None:
        report = score_fleet_recall([_query(), _query(event_id="e2")])
        assert report.measured is True
        assert report.agents == 1
        assert report.fleet_fed is False

    def test_two_agents_make_it_fleet_fed(self) -> None:
        report = score_fleet_recall([_query(by="agent-a"), _query(by="agent-b", event_id="e2")])
        assert report.agents == 2
        assert report.fleet_fed is True

    def test_answered_rate_counts_found_only(self) -> None:
        records = [_query(), _query(found=False, event_id="e2"), _query(event_id="e3")]
        report = score_fleet_recall(records)
        assert report.queries == 3
        assert report.answered == 2
        assert report.answered_rate == 2 / 3

    def test_usage_rate_scored_only_over_labelled_answered(self) -> None:
        records = [
            _query(was_used=True),
            _query(was_used=False, event_id="e2"),
            _query(event_id="e3"),  # answered, unlabelled — must not dilute the rate
            _query(found=False, was_used=True, event_id="e4"),  # unanswered — ignored
        ]
        report = score_fleet_recall(records)
        assert report.usage_labelled == 2
        assert report.usage_rate == 0.5

    def test_fleet_accuracy_scored_only_over_correctness_labelled(self) -> None:
        records = [
            _query(was_correct=True),
            _query(was_correct=True, event_id="e2"),
            _query(was_correct=False, event_id="e3"),
            _query(event_id="e4"),  # unlabelled
        ]
        report = score_fleet_recall(records)
        assert report.correctness_labelled == 3
        assert report.fleet_accuracy == 2 / 3

    def test_usage_and_correctness_are_independent(self) -> None:
        # Used-but-wrong and correct-but-unused must not bleed into each other.
        records = [
            _query(was_used=True, was_correct=False),
            _query(was_used=False, was_correct=True, event_id="e2"),
        ]
        report = score_fleet_recall(records)
        assert report.usage_rate == 0.5
        assert report.fleet_accuracy == 0.5

    def test_contradiction_counted_and_kept(self) -> None:
        # abstained=True with returned memories is a producer integrity fault.
        records = [_query(abstained=True), _query(abstained=False, event_id="e2")]
        report = score_fleet_recall(records)
        assert report.contradictions == 1
        assert report.answered == 2  # the contradictory record still scores

    def test_abstained_none_is_not_a_contradiction(self) -> None:
        report = score_fleet_recall([_query(abstained=None)])
        assert report.contradictions == 0

    def test_unanswered_labels_do_not_count(self) -> None:
        # An outcome recorded against an unanswered query carries no signal.
        report = score_fleet_recall([_query(found=False, was_used=True, was_correct=True)])
        assert report.usage_labelled == 0
        assert report.correctness_labelled == 0

    def test_as_dict_rounds_and_carries_counts(self) -> None:
        records = [
            _query(by="a", was_used=True, was_correct=True),
            _query(by="b", was_used=True, event_id="e2"),
            _query(by="c", found=False, event_id="e3"),
        ]
        d = score_fleet_recall(records).as_dict()
        assert d == {
            "queries": 3,
            "agents": 3,
            "fleet_fed": True,
            "answered": 2,
            "answered_rate": round(2 / 3, 4),
            "usage_labelled": 2,
            "usage_rate": 1.0,
            "correctness_labelled": 1,
            "fleet_accuracy": 1.0,
            "contradictions": 0,
            "measured": True,
        }

    def test_report_is_frozen(self) -> None:
        report = score_fleet_recall([_query()])
        assert isinstance(report, FleetRecallReport)
        try:
            report.queries = 99  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("FleetRecallReport must be immutable")


class TestReportFromLedger:
    def test_missing_file_is_unmeasured(self, tmp_path: Path) -> None:
        report = report_from_ledger(tmp_path / "absent.jsonl")
        assert report.measured is False

    def test_round_trip_through_production_ledger(self, tmp_path: Path) -> None:
        # The real boundary: RecallLedger writes, the scorer reads the same file.
        path = tmp_path / "ledger.jsonl"
        ledger = RecallLedger(path)
        e1 = ledger.record("who owns portal", ["src:a", "src:b"], top_k=5, by="claude")
        e2 = ledger.record("pin floor", ["src:c"], top_k=5, by="codex")
        ledger.record("unknown thing", [], top_k=5, by="claude")
        ledger.record_outcome(e1, was_used=True, was_correct=True)
        ledger.record_outcome(e2, was_used=False)

        report = report_from_ledger(path)

        assert report.measured is True
        assert report.queries == 3
        assert report.agents == 2
        assert report.fleet_fed is True
        assert report.answered == 2
        assert report.usage_labelled == 2
        assert report.usage_rate == 0.5
        assert report.correctness_labelled == 1
        assert report.fleet_accuracy == 1.0

    def test_later_outcome_supersedes_earlier(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger = RecallLedger(path)
        event = ledger.record("q", ["src:a"], top_k=3, by="claude")
        ledger.record_outcome(event, was_used=False)
        ledger.record_outcome(event, was_used=True)

        report = report_from_ledger(path)

        assert report.usage_labelled == 1
        assert report.usage_rate == 1.0

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        RecallLedger(path).record("q", ["src:a"], top_k=1, by="claude")
        assert report_from_ledger(str(path)).measured is True
