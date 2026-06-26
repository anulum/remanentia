# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for SYNAPSE feed ingestion

"""Tests for :mod:`feed_ingest`.

The suite drives real JSONL feed files and the production Markdown finding sink.
The end-to-end tests use the installed ``synapse_channel`` finding parser and
admission gate, so feed records cross the same honesty boundary as hub-backed
findings.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

import pytest

from finding_ingest import MarkdownFindingSink, SeqCursor

synapse_channel = pytest.importorskip("synapse_channel")

from feed_ingest import (  # noqa: E402
    DEFAULT_FEED_CURSOR_NAME,
    FeedIngestReport,
    default_feed_cursor,
    feed_record_to_finding,
    ingest_feed,
    ingest_from_feed,
    main,
)


def _append_feed(path: Path, *records: dict[str, Any] | str) -> None:
    """Append JSON feed records or raw lines to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, str):
                handle.write(record + "\n")
            else:
                handle.write(json.dumps(record) + "\n")


def _chat(payload: str, *, seq: int = 1, sender: str = "SCPN-FUSION-CORE") -> dict[str, Any]:
    """Return a realistic SYNAPSE chat feed record."""
    return {
        "v": 1,
        "i": seq,
        "ty": "chat",
        "s": sender,
        "to": "all",
        "p": payload,
        "t": 1_782_019_238_735,
        "h": "syn-test",
    }


def _finding_payload(statement: str) -> dict[str, Any]:
    """Return a fully declared finding payload for a ``ty=finding`` feed row."""
    return {
        "statement": statement,
        "subkind": "outcome",
        "evidence_kind": "measured",
        "claim_status": "reference_validated",
        "freshness": "verified_at_source",
        "evidence_ref": "commit:37ae044",
        "provenance": {
            "project": "REMANENTIA",
            "actor": "Arcane Sapience",
            "session": "syn-test",
            "source_event_seq": 9,
            "ts": 1_782_019_238.735,
        },
        "validity": {
            "valid_from": 1_782_019_238.735,
            "valid_to": None,
            "observed_at": 1_782_019_238.735,
        },
        "lifecycle": "active",
        "supersedes": None,
        "verified_at_source": {
            "checked_this_session": True,
            "source_ref": "commit:37ae044",
            "by": "REMANENTIA",
            "at": 1_782_019_238.735,
        },
        "producer_confidence": None,
        "execution_substrate": None,
        "entities": ["LongMemEval"],
        "tags": ["benchmark"],
    }


def _frontmatters(directory: Path) -> list[dict[str, Any]]:
    """Return parsed JSON frontmatter from sink output files."""
    frontmatters: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatters.append(json.loads(text.split("---")[1]))
    return frontmatters


def _mapping(value: object) -> dict[str, Any]:
    """Return ``value`` as a mapping for nested JSON assertions."""
    return cast(dict[str, Any], value)


class _RejectDecision:
    """Admission decision double for rejected feed candidates."""

    verdict = "reject"
    reasons = ("forced rejection",)


class _AcceptDecision:
    """Admission decision double for accepted feed candidates."""

    verdict = "accept"
    reasons: tuple[str, ...] = ()


class TestFeedRecordToFinding:
    def test_explicit_decision_marker_becomes_boundary_finding(self) -> None:
        finding = feed_record_to_finding(_chat("Decision: keep BM25 as the fallback"), line_no=7)

        assert finding is not None
        assert finding["statement"] == "keep BM25 as the fallback"
        assert finding["subkind"] == "decision"
        assert finding["evidence_kind"] == "producer_asserted"
        assert finding["claim_status"] == "bounded_support"
        assert finding["freshness"] == "traceable_unchecked"
        assert finding["evidence_ref"] == "synapse-feed:7"
        assert _mapping(finding["validity"])["valid_from"] == pytest.approx(1_782_019_238.735)

    def test_unmarked_chat_is_not_a_candidate(self) -> None:
        assert feed_record_to_finding(_chat("build is green"), line_no=3) is None

    def test_empty_marker_is_not_a_candidate(self) -> None:
        assert feed_record_to_finding(_chat("Finding:   "), line_no=4) is None

    def test_non_text_unmarked_payload_is_not_a_candidate(self) -> None:
        assert feed_record_to_finding({"ty": "chat", "p": 42}, line_no=4) is None

    def test_unmarked_decision_kind_uses_full_payload_text(self) -> None:
        finding = feed_record_to_finding(
            {"ty": "decision", "s": "REMANENTIA", "p": "keep feed ingest idempotent", "i": "12"},
            line_no=5,
        )

        assert finding is not None
        assert finding["statement"] == "keep feed ingest idempotent"
        assert finding["subkind"] == "decision"
        assert _mapping(finding["provenance"])["source_event_seq"] == 12

    def test_non_text_decision_payload_is_not_a_candidate(self) -> None:
        assert feed_record_to_finding({"ty": "decision", "p": None}, line_no=5) is None

    def test_blank_decision_payload_is_not_a_candidate(self) -> None:
        assert feed_record_to_finding({"ty": "decision", "p": "   "}, line_no=5) is None

    def test_invalid_scalar_defaults_are_boundary_safe(self) -> None:
        finding = feed_record_to_finding(
            {
                "ty": "finding",
                "s": "",
                "to": "all",
                "p": "Finding: source unknown",
                "i": True,
                "t": "bad",
            },
            line_no=6,
        )

        assert finding is not None
        assert _mapping(finding["provenance"])["actor"] == "synapse"
        assert _mapping(finding["provenance"])["source_event_seq"] == 6
        assert _mapping(finding["validity"])["observed_at"] == 0.0

    def test_invalid_string_sequence_defaults_to_feed_line(self) -> None:
        finding = feed_record_to_finding(
            {"ty": "decision", "s": "REMANENTIA", "p": "keep defaulting safe", "i": "not-int"},
            line_no=8,
        )

        assert finding is not None
        assert _mapping(finding["provenance"])["source_event_seq"] == 8

    def test_boolean_timestamp_defaults_to_zero(self) -> None:
        finding = feed_record_to_finding(
            {"ty": "decision", "s": "REMANENTIA", "p": "keep timestamps safe", "t": True},
            line_no=8,
        )

        assert finding is not None
        assert _mapping(finding["validity"])["valid_from"] == 0.0

    def test_mapping_finding_payload_keeps_axes_and_adds_feed_defaults(self) -> None:
        record = {
            "ty": "finding",
            "s": "REMANENTIA",
            "p": {
                "statement": "full-S benchmark page now states the comparable score",
                "evidence_kind": "measured",
                "claim_status": "reference_validated",
                "freshness": "verified_at_source",
                "evidence_ref": "commit:37ae044",
                "verified_at_source": {"checked_this_session": True},
            },
            "t": 1_782_019_238_735,
            "h": "syn-test",
            "i": 9,
        }

        finding = feed_record_to_finding(record, line_no=9)

        assert finding is not None
        assert finding["evidence_kind"] == "measured"
        assert finding["claim_status"] == "reference_validated"
        assert finding["evidence_ref"] == "commit:37ae044"
        assert _mapping(finding["validity"])["observed_at"] == pytest.approx(1_782_019_238.735)
        assert _mapping(finding["provenance"])["source_event_seq"] == 9


class TestIngestFromFeed:
    def test_explicit_feed_records_ingest_through_real_gate(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(
            feed,
            _chat("build is green", seq=1),
            _chat("Decision: keep BM25 as the fallback", seq=2),
            _chat("Finding: full-S benchmark docs now lead with the comparable score", seq=3),
        )
        sink_dir = tmp_path / "findings"
        report = ingest_from_feed(
            feed,
            MarkdownFindingSink(sink_dir),
            tmp_path / "feed_cursor.json",
        )

        assert report == FeedIngestReport(
            scanned=3,
            candidates=2,
            admitted=2,
            skipped=1,
            rejected=0,
            last_seq=3,
            rejections=(),
        )
        fronts = _frontmatters(sink_dir)
        assert [front["subkind"] for front in fronts] == ["decision", "outcome"]
        assert {front["evidence_kind"] for front in fronts} == {"producer_asserted"}
        assert {front["claim_status"] for front in fronts} == {"bounded_support"}
        assert SeqCursor(tmp_path / "feed_cursor.json").load() == 3

    def test_report_advanced_tracks_physical_scan(self) -> None:
        assert FeedIngestReport(1, 0, 0, 1, 0, 1).advanced is True
        assert FeedIngestReport(0, 0, 0, 0, 0, 0).advanced is False

    def test_resume_uses_physical_line_cursor(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        cursor = tmp_path / "feed_cursor.json"
        sink_dir = tmp_path / "findings"
        _append_feed(feed, _chat("Decision: keep the local ledger", seq=1))

        first = ingest_from_feed(feed, MarkdownFindingSink(sink_dir), cursor)
        second = ingest_from_feed(feed, MarkdownFindingSink(sink_dir), cursor)

        assert first.admitted == 1
        assert second.scanned == 0
        assert len(list(sink_dir.glob("*.md"))) == 1

    def test_mapping_finding_payload_ingests_as_reference_validated(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(
            feed,
            {
                "v": 1,
                "i": 9,
                "ty": "finding",
                "s": "REMANENTIA",
                "to": "all",
                "p": _finding_payload("full-S benchmark page now states the comparable score"),
                "t": 1_782_019_238_735,
                "h": "syn-test",
            },
        )

        report = ingest_from_feed(
            feed,
            MarkdownFindingSink(tmp_path / "findings"),
            tmp_path / "feed_cursor.json",
        )

        assert report.admitted == 1
        front = _frontmatters(tmp_path / "findings")[0]
        assert front["verdict"] == "accept"
        assert front["claim_status"] == "reference_validated"
        assert front["freshness"] == "verified_at_source"

    def test_malformed_json_advances_cursor_and_reports_rejection(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(feed, "{not-json", _chat("Decision: keep feed ingest idempotent", seq=2))
        cursor = tmp_path / "feed_cursor.json"

        report = ingest_from_feed(feed, MarkdownFindingSink(tmp_path / "findings"), cursor)

        assert report.scanned == 2
        assert report.candidates == 1
        assert report.admitted == 1
        assert report.rejected == 1
        assert report.rejections[0][0] == 1
        assert "invalid JSON" in report.rejections[0][1]
        assert SeqCursor(cursor).load() == 2

    def test_non_object_json_row_is_rejected(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(feed, "[1, 2, 3]")

        report = ingest_from_feed(
            feed, MarkdownFindingSink(tmp_path / "findings"), tmp_path / "cur.json"
        )

        assert report.scanned == 1
        assert report.rejected == 1
        assert "JSON object" in report.rejections[0][1]

    def test_missing_feed_file_does_not_advance_cursor(self, tmp_path: Path) -> None:
        cursor = tmp_path / "feed_cursor.json"

        report = ingest_from_feed(
            tmp_path / "missing.ndjson",
            MarkdownFindingSink(tmp_path / "findings"),
            cursor,
        )

        assert report.scanned == 0
        assert report.advanced is False
        assert not cursor.exists()

    def test_parse_failure_rejects_candidate_and_advances(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(feed, _chat("Decision: parse failure is contained", seq=1))

        def broken_parse(_raw: Mapping[str, object]) -> object:
            raise ValueError("broken parser")

        report = ingest_feed(
            feed,
            MarkdownFindingSink(tmp_path / "findings"),
            SeqCursor(tmp_path / "cur.json"),
            parse_finding=broken_parse,
            admit=lambda _finding: _AcceptDecision(),
        )

        assert report.candidates == 1
        assert report.rejected == 1
        assert "broken parser" in report.rejections[0][1]
        assert SeqCursor(tmp_path / "cur.json").load() == 1

    def test_gate_rejection_is_reported(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(feed, _chat("Decision: reject path is audited", seq=1))

        report = ingest_feed(
            feed,
            MarkdownFindingSink(tmp_path / "findings"),
            SeqCursor(tmp_path / "cur.json"),
            parse_finding=lambda raw: raw,
            admit=lambda _finding: _RejectDecision(),
        )

        assert report.candidates == 1
        assert report.admitted == 0
        assert report.rejected == 1
        assert report.rejections == ((1, "forced rejection"),)

    def test_limit_caps_physical_feed_rows(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(
            feed,
            _chat("Decision: first", seq=1),
            _chat("Decision: second", seq=2),
        )
        cursor = tmp_path / "feed_cursor.json"
        sink_dir = tmp_path / "findings"

        first = ingest_from_feed(feed, MarkdownFindingSink(sink_dir), cursor, limit=1)
        second = ingest_from_feed(feed, MarkdownFindingSink(sink_dir), cursor, limit=1)

        assert first.last_seq == 1
        assert second.last_seq == 2
        assert len(list(sink_dir.glob("*.md"))) == 2

    def test_default_feed_cursor_uses_findings_parent(self, tmp_path: Path) -> None:
        assert (
            default_feed_cursor(tmp_path)
            == tmp_path / "memory" / "semantic" / DEFAULT_FEED_CURSOR_NAME
        )

    def test_main_uses_environment_paths_and_prints_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        feed = tmp_path / "feed.ndjson"
        _append_feed(feed, _chat("Decision: expose feed ingest as an operator command", seq=1))
        monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
        monkeypatch.setenv("REMANENTIA_SYNAPSE_FEED", str(feed))

        assert main() == 0

        captured = capsys.readouterr()
        assert "feed-ingest: scanned=1 candidates=1 admitted=1" in captured.out
        assert (tmp_path / "memory" / "semantic" / DEFAULT_FEED_CURSOR_NAME).exists()
        assert len(list((tmp_path / "memory" / "semantic" / "findings").glob("*.md"))) == 1
