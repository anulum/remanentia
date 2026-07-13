# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the finding ingest bridge

"""Real-store and persisted-output tests for :mod:`finding_ingest`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import synapse_channel as sc  # type: ignore[import-untyped]
from synapse_channel.core.journal import record_finding  # type: ignore[import-untyped]

from finding_ingest import (
    DEFAULT_ADMITTING_VERDICTS,
    IngestReport,
    MarkdownFindingSink,
    SeqCursor,
    ingest_from_hub,
    ingest_findings,
)


def _valid_finding_record(statement: str) -> dict[str, Any]:
    """Return a complete finding record accepted by the real admission gate."""
    return {
        "statement": statement,
        "subkind": "outcome",
        "evidence_kind": "measured",
        "claim_status": "bounded-support",
        "freshness": "verified-at-source",
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


def _floor_finding_record(statement: str) -> dict[str, Any]:
    record = _valid_finding_record(statement)
    record["evidence_kind"] = "producer-asserted"
    record["claim_status"] = "reference-validated"
    return record


def _append_event(db_path: Path, payload: Any, *, kind: str = "finding") -> None:
    store = sc.EventStore(str(db_path))
    try:
        store.append(kind, cast(dict[str, Any], payload), durable=True)
    finally:
        store.close()


def _markdown_records(directory: Path) -> list[tuple[dict[str, Any], str]]:
    records = []
    for path in sorted(directory.glob("*.md")):
        _, frontmatter, body = path.read_text(encoding="utf-8").split("---", 2)
        records.append((dict(json.loads(frontmatter)), body.strip()))
    return records


class TestIngestCore:
    def test_admits_accept_and_floor_drops_reject(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _valid_finding_record("accepted"))
        _append_event(hub, _floor_finding_record("floored"))
        rejected = _valid_finding_record("rejected")
        rejected["validity"] = None
        _append_event(hub, rejected)

        sink_dir = tmp_path / "findings"
        cursor = tmp_path / "cur.json"
        report = ingest_from_hub(hub, MarkdownFindingSink(sink_dir), cursor)

        records = _markdown_records(sink_dir)
        assert {body for _, body in records} == {"accepted", "floored"}
        assert {front["verdict"] for front, _ in records} == {"accept", "floor"}
        assert report.admitted == 2
        assert report.rejected == 1
        assert report.scanned == 3
        assert report.last_seq == 3
        assert report.rejections[0][0] == 3
        assert SeqCursor(cursor).load() == 3

    def test_floored_finding_records_its_verdict_to_sink(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _floor_finding_record("floored"))
        sink_dir = tmp_path / "findings"
        report = ingest_from_hub(hub, MarkdownFindingSink(sink_dir), tmp_path / "cur.json")
        assert report.admitted == 1
        assert _markdown_records(sink_dir)[0][0]["verdict"] == "floor"

    def test_resume_from_cursor_skips_processed(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _valid_finding_record("old"))
        _append_event(hub, _valid_finding_record("new"))
        cursor = SeqCursor(tmp_path / "cur.json")
        cursor.save(1)
        sink_dir = tmp_path / "findings"
        report = ingest_from_hub(hub, MarkdownFindingSink(sink_dir), cursor.path)
        assert [body for _, body in _markdown_records(sink_dir)] == ["new"]
        assert report.scanned == 1
        assert cursor.load() == 2

    def test_unparsable_payload_is_a_rejection_not_a_crash(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, ["malformed", "finding"])
        _append_event(hub, _valid_finding_record("good"))
        sink_dir = tmp_path / "findings"
        report = ingest_from_hub(
            hub, MarkdownFindingSink(sink_dir), tmp_path / "cur.json"
        )
        assert [body for _, body in _markdown_records(sink_dir)] == ["good"]
        assert report.rejected == 1
        assert "unparsable payload" in report.rejections[0][1]
        assert report.last_seq == 2

    def test_no_events_does_not_move_cursor(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        store = sc.EventStore(str(hub))
        store.close()
        cursor = SeqCursor(tmp_path / "cur.json")
        cursor.save(7)
        report = ingest_from_hub(hub, MarkdownFindingSink(tmp_path / "findings"), cursor.path)
        assert report.scanned == 0
        assert report.advanced is False
        assert cursor.load() == 7

    def test_custom_admitting_verdicts(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _valid_finding_record("accepted"))
        _append_event(hub, _floor_finding_record("floored"))
        store = sc.EventStore(str(hub))
        sink_dir = tmp_path / "findings"
        try:
            report = ingest_findings(
                store,
                MarkdownFindingSink(sink_dir),
                SeqCursor(tmp_path / "cur.json"),
                parse_finding=sc.Finding.from_dict,
                admit=sc.admit,
                admitting_verdicts=("accept",),
            )
        finally:
            store.close()
        assert [body for _, body in _markdown_records(sink_dir)] == ["accepted"]
        assert report.rejected == 1

    def test_kinds_filter_is_passed_through(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _valid_finding_record("finding"))
        _append_event(hub, _valid_finding_record("chat"), kind="chat")
        sink_dir = tmp_path / "findings"
        report = ingest_from_hub(
            hub,
            MarkdownFindingSink(sink_dir),
            tmp_path / "cur.json",
            kinds=("finding",),
        )
        assert [body for _, body in _markdown_records(sink_dir)] == ["finding"]
        assert report.scanned == 1

    def test_real_store_limit_is_honoured_across_passes(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        _append_event(hub, _valid_finding_record("first"))
        _append_event(hub, _valid_finding_record("second"))
        sink_dir = tmp_path / "findings"
        cursor = tmp_path / "cur.json"
        first = ingest_from_hub(hub, MarkdownFindingSink(sink_dir), cursor, limit=1)
        second = ingest_from_hub(hub, MarkdownFindingSink(sink_dir), cursor, limit=1)
        assert first.scanned == second.scanned == 1
        assert {body for _, body in _markdown_records(sink_dir)} == {"first", "second"}


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
            claim_status="bounded-support",
            freshness="verified-at-source",
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
        assert front["claim_status"] == "bounded-support"
        assert front["freshness"] == "verified-at-source"
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


class TestHubIntegration:
    def test_end_to_end_ingest_from_real_hub(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        store = sc.EventStore(str(hub))
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
        assert front["claim_status"] == "bounded-support"

        # A second pass resumes above the cursor and ingests nothing new.
        again = ingest_from_hub(hub, sink, cursor)
        assert again.scanned == 0

    def test_invalid_finding_is_gated_out(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        store = sc.EventStore(str(hub))
        bad = _valid_finding_record("no validity window here")
        bad["validity"] = None  # the gate rejects a scientific finding without validity
        record_finding(store, bad)
        store.close()

        report = ingest_from_hub(hub, MarkdownFindingSink(tmp_path / "f"), tmp_path / "cur.json")
        assert report.admitted == 0
        assert report.rejected == 1
        assert not list((tmp_path / "f").glob("*.md"))

    def test_falsified_reference_validated_finding_is_rejected(self, tmp_path: Path) -> None:
        hub = tmp_path / "hub.db"
        store = sc.EventStore(str(hub))
        invalid = _valid_finding_record("falsified reference claim cannot enter memory")
        invalid["evidence_kind"] = "falsified"
        invalid["claim_status"] = "reference-validated"
        record_finding(store, invalid)
        store.close()

        report = ingest_from_hub(hub, MarkdownFindingSink(tmp_path / "f"), tmp_path / "cur.json")

        assert report.admitted == 0
        assert report.rejected == 1
        assert "falsified evidence" in report.rejections[0][1]
        assert not list((tmp_path / "f").glob("*.md"))
