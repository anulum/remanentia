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

    def test_score_round_trips(self, ledger: RecallLedger):
        ledger.record("q", ["s:n"], top_k=1, score=0.87)
        (q,) = list(ledger.queries())
        assert q.score == 0.87

    def test_score_defaults_none(self, ledger: RecallLedger):
        ledger.record("q", [], top_k=1)
        (q,) = list(ledger.queries())
        assert q.score is None

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

    def test_correctness_label_merges_independently(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_correct=True)
        (q,) = list(ledger.queries())
        assert q.was_correct is True
        assert q.was_used is None  # usage untouched by a correctness-only outcome

    def test_usage_and_correctness_are_orthogonal(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True)
        ledger.record_outcome(eid, was_correct=False)  # used but wrong
        (q,) = list(ledger.queries())
        assert q.was_used is True
        assert q.was_correct is False

    def test_both_labels_in_one_call(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True, was_correct=True)
        (q,) = list(ledger.queries())
        assert q.was_used is True
        assert q.was_correct is True

    def test_correctness_only_outcome_does_not_clobber_usage(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        ledger.record_outcome(eid, was_used=True)
        ledger.record_outcome(eid, was_correct=True)  # separate call, usage must survive
        (q,) = list(ledger.queries())
        assert q.was_used is True
        assert q.was_correct is True

    def test_record_outcome_needs_at_least_one_label(self, ledger: RecallLedger):
        eid = ledger.record("q", ["s:n"], top_k=1)
        with pytest.raises(ValueError, match="was_used and/or was_correct"):
            ledger.record_outcome(eid)

    def test_default_was_correct_is_none(self, ledger: RecallLedger):
        ledger.record("q", ["s:n"], top_k=1)
        (q,) = list(ledger.queries())
        assert q.was_correct is None


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


class TestLatestFor:
    def test_finds_latest_matching(self, ledger: RecallLedger):
        ledger.record("alpha", ["s:1"], top_k=1)
        e2 = ledger.record("alpha", ["s:2"], top_k=1)
        assert ledger.latest_for("alpha") == e2

    def test_none_when_no_match(self, ledger: RecallLedger):
        ledger.record("alpha", [], top_k=1)
        assert ledger.latest_for("beta") is None

    def test_filters_by_agent(self, ledger: RecallLedger):
        ledger.record("alpha", [], top_k=1, by="a1")
        e = ledger.record("alpha", [], top_k=1, by="a2")
        assert ledger.latest_for("alpha", by="a2") == e
        assert ledger.latest_for("alpha", by="a3") is None


class TestRecallFeedback:
    def test_records_outcome(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "fb.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        led.record("q", ["s:n"], top_k=1)
        msg = mcp_server.handle_recall_feedback("q", True)
        assert "was_used=True" in msg
        (rq,) = list(led.queries())
        assert rq.was_used is True

    def test_no_prior_recall(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "fb.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        msg = mcp_server.handle_recall_feedback("nope", True)
        assert "No prior recall" in msg

    def test_disabled_is_noop(self, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in mcp_server.handle_recall_feedback("q", True)

    def test_dispatch_via_handle_request(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "fb.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        led.record("q", ["s:n"], top_k=1)
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_feedback",
                "arguments": {"query": "q", "was_used": True},
            },
        }
        resp = mcp_server.handle_request(req)
        assert resp is not None and resp["id"] == 7
        (rq,) = list(led.queries())
        assert rq.was_used is True


class TestRecallCorrectness:
    """The correctness seam records the gate label a downstream verifier supplies."""

    def test_records_correctness(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "c.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        led.record("q", ["s:n"], top_k=1)
        msg = mcp_server.handle_recall_correctness("q", False)
        assert "was_correct=False" in msg
        (rq,) = list(led.queries())
        assert rq.was_correct is False
        assert rq.was_used is None  # correctness must not touch usage

    def test_no_prior_recall(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "c.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        assert "No prior recall" in mcp_server.handle_recall_correctness("nope", True)

    def test_disabled_is_noop(self, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in mcp_server.handle_recall_correctness("q", True)

    def test_dispatch_via_handle_request(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "c.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        led.record("q", ["s:n"], top_k=1)
        req = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_correctness",
                "arguments": {"query": "q", "was_correct": True},
            },
        }
        resp = mcp_server.handle_request(req)
        assert resp is not None and resp["id"] == 9
        (rq,) = list(led.queries())
        assert rq.was_correct is True


class TestRecallLoopClosure:
    """Recall→remember loop closure auto-populates was_used through mcp_server."""

    def test_remember_closes_recall_loop(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "lc.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.setattr(mcp_server, "_OUTCOME_TRACKER", None)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)

        event_id = mcp_server._log_recall("how to reuse vectors", ["sem:t"], 3, "")
        assert event_id is not None
        mcp_server._observe_recall(event_id, ["incremental vector index reuse content hash dedup"])
        # An unrelated memory must not close the loop.
        mcp_server._close_recall_loops("notes about coffee brewing temperature")
        assert list(led.queries())[0].was_used is None
        # The echoing memory does.
        mcp_server._close_recall_loops("we did incremental vector index reuse via content hash dedup")
        assert list(led.queries())[0].was_used is True

    def test_observe_noop_when_event_id_none(self, monkeypatch):
        import mcp_server

        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        # Ledger write was skipped/failed → no event id → nothing to track, no raise.
        mcp_server._observe_recall(None, ["a returned memory text long enough to tokenize"])

    def test_loop_closure_disabled_is_noop(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "lc.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.setattr(mcp_server, "_OUTCOME_TRACKER", None)
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        mcp_server._observe_recall("e1", ["incremental vector index reuse content hash dedup"])
        mcp_server._close_recall_loops("incremental vector index reuse content hash dedup")
        assert list(led.queries()) == []


class TestMcpRecallHook:
    """The mcp_server recall hook records the query stream, opt-out aware."""

    def test_log_recall_writes(self, tmp_path: Path, monkeypatch):
        import mcp_server

        led = RecallLedger(tmp_path / "hook.jsonl")
        monkeypatch.setattr(mcp_server, "_RECALL_LEDGER", led)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE", raising=False)
        mcp_server._log_recall("q", ["sem:trace"], 3, "scpn", 0.91)
        (q,) = list(led.queries())
        assert q.query == "q"
        assert q.returned_ids == ("sem:trace",)
        assert q.top_k == 3
        assert q.project == "scpn"
        assert q.score == 0.91

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


class _FakeEmitter:
    """Records bus emits without touching the network."""

    def __init__(self, *, fail: bool = False):
        self.calls: list[tuple] = []
        self._fail = fail

    def emit(self, query_text, *, returned_claim_ids, was_used=False, abstained=False):
        if self._fail:
            raise RuntimeError("emit boom")
        self.calls.append((query_text, list(returned_claim_ids), was_used, abstained))
        return True


class TestMcpRecallBus:
    """The mcp_server recall hook mirrors the query stream onto the fleet bus."""

    def test_log_recall_emits_to_bus(self, monkeypatch):
        import mcp_server

        fake = _FakeEmitter()
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", fake)
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", True)
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")  # isolate the bus sink
        mcp_server._log_recall("q", ["sem:trace"], 3, "scpn", 0.9)
        assert fake.calls == [("q", ["sem:trace"], False, False)]

    def test_bus_abstains_when_nothing_returned(self, monkeypatch):
        import mcp_server

        fake = _FakeEmitter()
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", fake)
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", True)
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        mcp_server._log_recall("unknown", [], 5, "")
        assert fake.calls == [("unknown", [], False, True)]

    def test_emit_recall_bus_none_is_noop(self, monkeypatch):
        import mcp_server

        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", None)
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", True)
        # No emitter → silent return, no exception.
        mcp_server._emit_recall_bus("q", ["a:1"])

    def test_failing_emitter_never_breaks_recall(self, monkeypatch):
        import mcp_server

        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", _FakeEmitter(fail=True))
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", True)
        # A raising emitter must be swallowed, not propagated.
        mcp_server._emit_recall_bus("q", ["a:1"])

    def test_get_bus_emitter_lazy_and_cached(self, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_BUS_DISABLE", "1")  # default_emitter → None
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", None)
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", False)
        first = mcp_server._get_bus_emitter()
        assert first is None
        assert mcp_server._BUS_EMITTER_INIT is True
        # Second call returns the cached resolution without rebuilding.
        assert mcp_server._get_bus_emitter() is None


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
