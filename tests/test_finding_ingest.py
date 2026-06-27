# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the finding ingest bridge

"""Tests for :mod:`finding_ingest`.

The generic core is exercised with injected fakes (no SYNAPSE dependency), which
is the point of the dependency-injected design: the read-side seam is testable
and extractable on its own. A final integration test drives the real
``synapse_channel`` hub store + admission gate end-to-end.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from finding_ingest import (
    DEFAULT_ADMITTING_VERDICTS,
    IngestReport,
    MarkdownFindingSink,
    SeqCursor,
    ingest_findings,
)


class FakeEvent:
    """In-memory hub event used by the injected ingest core tests."""

    def __init__(self, seq: int, payload: dict[str, Any], kind: str = "finding") -> None:
        self.seq = seq
        self.kind = kind
        self.payload = payload


class FakeStore:
    """Returns events strictly above ``after_seq``, filtered by kind."""

    def __init__(self, events: Sequence[FakeEvent]) -> None:
        self._events = events

    def read_since(
        self,
        after_seq: int,
        *,
        kinds: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> Sequence[FakeEvent]:
        rows = [e for e in self._events if e.seq > after_seq]
        if kinds is not None:
            allowed = set(kinds)
            rows = [e for e in rows if e.kind in allowed]
        rows.sort(key=lambda e: e.seq)
        return rows if limit is None else rows[:limit]


class FakeDecision:
    """Admission decision returned by the injected fake gate."""

    def __init__(self, verdict: str, reasons: Sequence[str] = ()) -> None:
        self.verdict = verdict
        self.reasons = reasons


class RecordingSink:
    """Finding sink that records writes for assertions."""

    def __init__(self) -> None:
        self.written: list[tuple[str, int, str]] = []

    def write(self, finding: Any, seq: int, verdict: str) -> None:
        self.written.append((finding["statement"], seq, verdict))


def _payload(statement: str, verdict: str = "accept") -> dict[str, Any]:
    """A payload that carries its desired verdict so the fake gate is trivial."""
    return {"statement": statement, "_verdict": verdict}


def _fake_admit(finding: dict[str, Any]) -> FakeDecision:
    return FakeDecision(finding.get("_verdict", "accept"), reasons=("forced",))


def _identity_parse(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


class TestIngestCore:
    def _run(
        self,
        tmp_path: Path,
        events: Sequence[FakeEvent],
        **kw: Any,
    ) -> tuple[RecordingSink, SeqCursor, IngestReport]:
        sink = RecordingSink()
        cursor = SeqCursor(tmp_path / "cur.json")
        report = ingest_findings(
            FakeStore(events),
            sink,
            cursor,
            parse_finding=_identity_parse,
            admit=_fake_admit,
            **kw,
        )
        return sink, cursor, report

    def test_admits_accept_and_floor_drops_reject(self, tmp_path: Path) -> None:
        events = [
            FakeEvent(1, _payload("a", "accept")),
            FakeEvent(2, _payload("b", "floor")),
            FakeEvent(3, _payload("c", "reject")),
        ]
        sink, cursor, report = self._run(tmp_path, events)
        assert {s for s, _, _ in sink.written} == {"a", "b"}
        assert report.admitted == 2
        assert report.rejected == 1
        assert report.scanned == 3
        assert report.last_seq == 3
        assert report.rejections == ((3, "forced"),)
        assert cursor.load() == 3  # cursor persisted at the highest scanned seq

    def test_floored_finding_records_its_verdict_to_sink(self, tmp_path: Path) -> None:
        sink, _, _ = self._run(tmp_path, [FakeEvent(5, _payload("x", "floor"))])
        assert sink.written == [("x", 5, "floor")]

    def test_resume_from_cursor_skips_processed(self, tmp_path: Path) -> None:
        cursor = SeqCursor(tmp_path / "cur.json")
        cursor.save(10)
        sink = RecordingSink()
        events = [FakeEvent(8, _payload("old")), FakeEvent(12, _payload("new"))]
        report = ingest_findings(
            FakeStore(events), sink, cursor, parse_finding=_identity_parse, admit=_fake_admit
        )
        assert [s for s, _, _ in sink.written] == ["new"]
        assert report.scanned == 1
        assert cursor.load() == 12

    def test_unparsable_payload_is_a_rejection_not_a_crash(self, tmp_path: Path) -> None:
        def boom_parse(payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("statement") == "bad":
                raise ValueError("corrupt")
            return dict(payload)

        sink = RecordingSink()
        cursor = SeqCursor(tmp_path / "cur.json")
        events = [FakeEvent(1, _payload("bad")), FakeEvent(2, _payload("good"))]
        report = ingest_findings(
            FakeStore(events), sink, cursor, parse_finding=boom_parse, admit=_fake_admit
        )
        assert [s for s, _, _ in sink.written] == ["good"]
        assert report.rejected == 1
        assert "corrupt" in report.rejections[0][1]
        assert report.last_seq == 2  # still advanced past the bad event

    def test_no_events_does_not_move_cursor(self, tmp_path: Path) -> None:
        cursor = SeqCursor(tmp_path / "cur.json")
        cursor.save(7)
        sink = RecordingSink()
        report = ingest_findings(
            FakeStore([]), sink, cursor, parse_finding=_identity_parse, admit=_fake_admit
        )
        assert report.scanned == 0
        assert report.advanced is False
        assert cursor.load() == 7  # untouched

    def test_custom_admitting_verdicts(self, tmp_path: Path) -> None:
        events = [FakeEvent(1, _payload("a", "accept")), FakeEvent(2, _payload("b", "floor"))]
        sink, _, report = self._run(tmp_path, events, admitting_verdicts=("accept",))
        assert [s for s, _, _ in sink.written] == ["a"]  # floor now excluded
        assert report.rejected == 1

    def test_kinds_filter_is_passed_through(self, tmp_path: Path) -> None:
        events = [
            FakeEvent(1, _payload("f"), kind="finding"),
            FakeEvent(2, _payload("c"), kind="chat"),
        ]
        sink, _, report = self._run(tmp_path, events, kinds=("finding",))
        assert [s for s, _, _ in sink.written] == ["f"]
        assert report.scanned == 1

    def test_non_string_axes_are_left_to_the_injected_gate(self, tmp_path: Path) -> None:
        payload = _payload("legacy payload with numeric axis")
        payload["evidence_kind"] = 42
        payload["claim_status"] = "reference_validated"

        sink, _, report = self._run(tmp_path, [FakeEvent(1, payload)])

        assert sink.written == [("legacy payload with numeric axis", 1, "accept")]
        assert report.rejected == 0


class TestSeqCursor:
    def test_missing_file_reads_zero(self, tmp_path: Path) -> None:
        assert SeqCursor(tmp_path / "absent.json").load() == 0

    def test_roundtrip(self, tmp_path: Path) -> None:
        c = SeqCursor(tmp_path / "nested" / "cur.json")
        c.save(42)
        assert SeqCursor(tmp_path / "nested" / "cur.json").load() == 42

    def test_corrupt_file_reads_zero(self, tmp_path: Path) -> None:
        p = tmp_path / "cur.json"
        p.write_text("{not json", encoding="utf-8")
        assert SeqCursor(p).load() == 0

    def test_save_is_atomic_no_tmp_left(self, tmp_path: Path) -> None:
        c = SeqCursor(tmp_path / "cur.json")
        c.save(3)
        assert not (tmp_path / "cur.json.tmp").exists()


class _DictFinding(dict[str, Any]):
    """A finding whose as_dict() returns itself, mimicking the real Finding."""

    def as_dict(self) -> dict[str, Any]:
        return dict(self)


class TestMarkdownFindingSink:
    def _finding(
        self,
        statement: str = "reuse cut re-embeds",
        subkind: str = "outcome",
    ) -> _DictFinding:
        return _DictFinding(
            statement=statement,
            subkind=subkind,
            claim_status="bounded_support",
            freshness="verified_at_source",
            evidence_kind="measured",
            provenance={"project": "REMANENTIA"},
            lifecycle="active",
            entities=["vector_index"],
            tags=["perf"],
        )

    def test_writes_markdown_with_honesty_frontmatter(self, tmp_path: Path) -> None:
        sink = MarkdownFindingSink(tmp_path / "findings")
        sink.write(self._finding(), seq=11, verdict="floor")
        files = list((tmp_path / "findings").glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        front = json.loads(text.split("---")[1])
        assert front["verdict"] == "floor"
        assert front["claim_status"] == "bounded_support"
        assert front["freshness"] == "verified_at_source"
        assert front["hub_seq"] == 11
        assert "reuse cut re-embeds" in text

    def test_idempotent_same_statement_one_file(self, tmp_path: Path) -> None:
        sink = MarkdownFindingSink(tmp_path / "findings")
        sink.write(self._finding(), seq=1, verdict="accept")
        sink.write(self._finding(), seq=2, verdict="accept")  # re-ingest
        assert len(list((tmp_path / "findings").glob("*.md"))) == 1

    def test_distinct_statements_distinct_files(self, tmp_path: Path) -> None:
        sink = MarkdownFindingSink(tmp_path / "findings")
        sink.write(self._finding("first thing"), seq=1, verdict="accept")
        sink.write(self._finding("second thing"), seq=2, verdict="accept")
        assert len(list((tmp_path / "findings").glob("*.md"))) == 2

    def test_accepts_plain_dict_finding(self, tmp_path: Path) -> None:
        sink = MarkdownFindingSink(tmp_path / "findings")
        sink.write({"statement": "plain", "subkind": "lesson"}, seq=3, verdict="accept")
        assert len(list((tmp_path / "findings").glob("*.md"))) == 1


class TestDefaultFindingsDir:
    def test_explicit_base(self, tmp_path: Path) -> None:
        from finding_ingest import default_findings_dir

        assert default_findings_dir(tmp_path) == tmp_path / "memory" / "semantic" / "findings"

    def test_env_base(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from finding_ingest import default_findings_dir

        monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
        assert default_findings_dir() == tmp_path / "memory" / "semantic" / "findings"

    def test_falls_back_to_module_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from finding_ingest import default_findings_dir

        monkeypatch.delenv("REMANENTIA_BASE", raising=False)
        result = default_findings_dir()
        assert result.parts[-3:] == ("memory", "semantic", "findings")


def test_ingest_report_advanced_property() -> None:
    assert IngestReport(scanned=2, admitted=1, rejected=1, last_seq=5).advanced is True
    assert IngestReport(scanned=0, admitted=0, rejected=0, last_seq=5).advanced is False


def test_default_admitting_verdicts() -> None:
    assert set(DEFAULT_ADMITTING_VERDICTS) == {"accept", "floor"}


# ── Integration: the real synapse_channel hub store + admission gate ──────────

synapse_channel = pytest.importorskip("synapse_channel")


def _valid_finding_record(statement: str) -> dict[str, Any]:
    """A well-formed finding the real gate accepts (carries a validity window)."""
    return {
        "statement": statement,
        "subkind": "outcome",
        "evidence_kind": "measured",
        "claim_status": "bounded_support",
        "freshness": "verified_at_source",
        "evidence_ref": "commit:0d92a47",
        "provenance": {
            "project": "REMANENTIA",
            "actor": "arcane",
            "session": "s1",
            "source_event_seq": 1,
            "ts": 1.0,
        },
        "validity": {"valid_from": 1.0, "valid_to": None, "observed_at": 1.0},
        "lifecycle": "active",
        "supersedes": None,
        "verified_at_source": {
            "checked_this_session": True,
            "source_ref": "commit:0d92a47",
            "by": "REMANENTIA",
            "at": 1.0,
        },
        "producer_confidence": None,
        "execution_substrate": None,
        "entities": ["vector_index"],
        "tags": ["perf"],
    }


class TestHubIntegration:
    def test_end_to_end_ingest_from_real_hub(self, tmp_path: Path) -> None:
        from finding_ingest import ingest_from_hub
        from synapse_channel.core.journal import record_finding

        hub = tmp_path / "hub.db"
        store = synapse_channel.EventStore(str(hub))
        record_finding(store, _valid_finding_record("vector reuse cut re-embeds 9x"))
        store.close()

        sink = MarkdownFindingSink(tmp_path / "findings")
        cursor = tmp_path / "cur.json"
        report = ingest_from_hub(hub, sink, cursor)

        assert report.admitted == 1
        assert report.rejected == 0
        files = list((tmp_path / "findings").glob("*.md"))
        assert len(files) == 1
        front = json.loads(files[0].read_text(encoding="utf-8").split("---")[1])
        assert front["verdict"] in ("accept", "floor")
        assert front["claim_status"] == "bounded_support"

        # A second pass resumes above the cursor and ingests nothing new.
        again = ingest_from_hub(hub, sink, cursor)
        assert again.scanned == 0

    def test_invalid_finding_is_gated_out(self, tmp_path: Path) -> None:
        from finding_ingest import ingest_from_hub
        from synapse_channel.core.journal import record_finding

        hub = tmp_path / "hub.db"
        store = synapse_channel.EventStore(str(hub))
        bad = _valid_finding_record("no validity window here")
        bad["validity"] = None  # the gate rejects a scientific finding without validity
        record_finding(store, bad)
        store.close()

        report = ingest_from_hub(hub, MarkdownFindingSink(tmp_path / "f"), tmp_path / "cur.json")
        assert report.admitted == 0
        assert report.rejected == 1
        assert not list((tmp_path / "f").glob("*.md"))

    def test_falsified_reference_validated_finding_is_rejected(self, tmp_path: Path) -> None:
        from finding_ingest import ingest_from_hub
        from synapse_channel.core.journal import record_finding

        hub = tmp_path / "hub.db"
        store = synapse_channel.EventStore(str(hub))
        invalid = _valid_finding_record("falsified reference claim cannot enter memory")
        invalid["evidence_kind"] = "falsified"
        invalid["claim_status"] = "reference_validated"
        record_finding(store, invalid)
        store.close()

        report = ingest_from_hub(hub, MarkdownFindingSink(tmp_path / "f"), tmp_path / "cur.json")

        assert report.admitted == 0
        assert report.rejected == 1
        assert "falsified evidence" in report.rejections[0][1]
        assert not list((tmp_path / "f").glob("*.md"))
