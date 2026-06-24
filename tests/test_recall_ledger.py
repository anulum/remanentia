# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the recall query-stream ledger (MS.1)

"""Cover the append-only recall ledger: recording, outcome merge, the
immutable two-record design, malformed-line tolerance, identity
resolution, concurrency, and the default-location factory.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from recall_ledger import RecallLedger, RecallQuery, default_ledger


@pytest.fixture
def ledger(tmp_path: Path) -> RecallLedger:
    return RecallLedger(tmp_path / "runtime" / "recall_ledger.jsonl")


class TestRecord:
    def test_record_returns_id_and_round_trips(self, ledger: RecallLedger):
        eid = ledger.record("dimits shift", ["sem:trace_a", "graph:e1"], top_k=5, project="scpn")
        assert isinstance(eid, str) and len(eid) == 16
        (q,) = list(ledger.queries())
        assert q.event_id == eid
        assert q.query == "dimits shift"
        assert q.returned_ids == ("sem:trace_a", "graph:e1")
        assert q.top_k == 5
        assert q.project == "scpn"
        assert q.found is True
        assert q.abstained is None
        assert q.was_used is None

    def test_creates_parent_directory(self, tmp_path: Path):
        led = RecallLedger(tmp_path / "deep" / "nested" / "ledger.jsonl")
        led.record("q", [], top_k=1)
        assert led.path.exists()

    def test_empty_results_marks_not_found(self, ledger: RecallLedger):
        ledger.record("nothing here", [], top_k=3)
        (q,) = list(ledger.queries())
        assert q.found is False
        assert q.returned_ids == ()

    @pytest.mark.parametrize("abstained", [True, False, None])
    def test_abstained_passthrough(self, ledger: RecallLedger, abstained):
        ledger.record("q", ["s:n"], top_k=1, abstained=abstained)
        (q,) = list(ledger.queries())
        assert q.abstained is abstained

    def test_identity_from_env(self, ledger: RecallLedger, monkeypatch):
        monkeypatch.setenv("REMANENTIA_AGENT", "scpn-control")
        ledger.record("q", [], top_k=1)
        (q,) = list(ledger.queries())
        assert q.by == "scpn-control"

    def test_identity_defaults_to_unknown(self, ledger: RecallLedger, monkeypatch):
        monkeypatch.delenv("REMANENTIA_AGENT", raising=False)
        ledger.record("q", [], top_k=1)
        (q,) = list(ledger.queries())
        assert q.by == "unknown"

    def test_explicit_by_overrides_env(self, ledger: RecallLedger, monkeypatch):
        monkeypatch.setenv("REMANENTIA_AGENT", "env-agent")
        ledger.record("q", [], top_k=1, by="explicit-agent")
        (q,) = list(ledger.queries())
        assert q.by == "explicit-agent"

    def test_blank_env_falls_back_to_unknown(self, ledger: RecallLedger, monkeypatch):
        monkeypatch.setenv("REMANENTIA_AGENT", "   ")
        ledger.record("q", [], top_k=1)
        (q,) = list(ledger.queries())
        assert q.by == "unknown"


class TestOutcome:
    def test_outcome_merges_into_query(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True)
        (q,) = list(ledger.queries())
        assert q.was_used is True

    def test_later_outcome_supersedes(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True)
        ledger.record_outcome(eid, was_used=False)
        (q,) = list(ledger.queries())
        assert q.was_used is False

    def test_outcome_does_not_emit_a_query(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True)
        assert len(list(ledger.queries())) == 1

    def test_outcome_with_non_string_event_id_ignored(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        # Hand-craft a malformed outcome row (numeric event_id) — must not crash.
        with ledger.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "outcome", "event_id": 42, "was_used": True}) + "\n")
        (q,) = list(ledger.queries())
        assert q.event_id == eid
        assert q.was_used is None


class TestReadTolerance:
    def test_missing_file_yields_nothing(self, tmp_path: Path):
        led = RecallLedger(tmp_path / "absent.jsonl")
        assert list(led.queries()) == []

    def test_malformed_and_blank_lines_skipped(self, ledger: RecallLedger):
        ledger.record("good", ["s:n"], top_k=1)
        with ledger.path.open("a", encoding="utf-8") as fh:
            fh.write("not json at all\n")
            fh.write("\n")
            fh.write("42\n")  # valid JSON but not a dict
        (q,) = list(ledger.queries())
        assert q.query == "good"

    def test_query_without_event_id_skipped(self, ledger: RecallLedger):
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        with ledger.path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "query", "query": "no id"}) + "\n")
        assert list(ledger.queries()) == []

    def test_non_query_kinds_skipped(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        with ledger.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "noise", "event_id": "x"}) + "\n")
        (q,) = list(ledger.queries())
        assert q.event_id == eid


class TestEventId:
    def test_event_id_deterministic(self):
        a = RecallLedger._event_id("q", 1.0, "agent")
        b = RecallLedger._event_id("q", 1.0, "agent")
        assert a == b and len(a) == 16

    def test_event_id_varies_by_field(self):
        base = RecallLedger._event_id("q", 1.0, "agent")
        assert RecallLedger._event_id("q2", 1.0, "agent") != base
        assert RecallLedger._event_id("q", 2.0, "agent") != base
        assert RecallLedger._event_id("q", 1.0, "other") != base


class TestConcurrency:
    def test_concurrent_appends_all_land(self, ledger: RecallLedger):
        def worker(i: int) -> None:
            ledger.record(f"q{i}", [f"s:{i}"], top_k=1, by=f"agent{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(list(ledger.queries())) == 40


class TestDefaultLedger:
    def test_env_override(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "custom_ledger.jsonl"
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(target))
        led = default_ledger()
        assert led.path == target

    def test_default_location(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER", raising=False)
        led = default_ledger()
        assert led.path.name == "recall_ledger.jsonl"
        assert led.path.parent.name == "runtime"

    def test_blank_override_uses_default(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", "   ")
        led = default_ledger()
        assert led.path.name == "recall_ledger.jsonl"


class TestMcpRecallHook:
    """The mcp_server recall hook records the query stream, opt-out aware."""

    def test_log_recall_writes(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "hook.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        mcp_server._log_recall("q", ["sem:trace"], 3, "scpn")
        (q,) = list(led.queries())
        assert q.query == "q"
        assert q.returned_ids == ("sem:trace",)
        assert q.top_k == 3
        assert q.project == "scpn"

    def test_log_recall_disabled_is_noop(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "hook.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        mcp_server._log_recall("q", [], 1, "")
        assert list(led.queries()) == []

    def test_get_recall_ledger_is_singleton(self, tmp_path: Path, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(tmp_path / "singleton.jsonl"))
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", None)
        first = mcp_server._get_recall_ledger()
        second = mcp_server._get_recall_ledger()
        assert first is second


def test_recall_query_is_frozen():
    q = RecallQuery(
        event_id="x",
        ts=1.0,
        by="a",
        query="q",
        top_k=1,
        project="",
        returned_ids=(),
        found=False,
    )
    with pytest.raises(AttributeError):
        q.query = "mutated"  # type: ignore[misc]  # frozen dataclass — testing immutability
