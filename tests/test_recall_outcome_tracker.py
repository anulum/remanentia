# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the recall outcome tracker

"""Tests for :mod:`recall_outcome_tracker`.

The tracker derives ``was_used`` from recall→remember loop closure. The suite
pins the contract that made the label trustworthy: only a genuine echo of a
returned memory marks a recall used, the signal is per-agent, and recalls that
see no echo are ruled unused (never silently dropped) so the calibration set has
both classes. Time is injected so expiry is exact and deterministic.
"""

from __future__ import annotations

from pathlib import Path

from recall_ledger import RecallLedger
from recall_outcome_tracker import (
    DEFAULT_MIN_CONTAINMENT,
    RecallOutcomeTracker,
    containment,
    tokenize,
)


def _ledger(tmp_path: Path) -> RecallLedger:
    return RecallLedger(tmp_path / "recall_ledger.jsonl")


def _event(ledger: RecallLedger, label: str, by: str) -> str:
    event_id = ledger.record(label, [f"trace:{label}"], top_k=1, by=by)
    assert isinstance(event_id, str)
    return event_id


def _usage(ledger: RecallLedger) -> dict[str, bool | None]:
    return {query.query: query.was_used for query in ledger.queries()}


class TestTokenize:
    def test_lowercases_splits_and_drops_stopwords(self) -> None:
        assert tokenize("The Vector Index is Fresh") == frozenset({"vector", "index", "fresh"})

    def test_punctuation_and_digits(self) -> None:
        assert tokenize("commit 0d92a47, reused 311060!") == frozenset(
            {"commit", "0d92a47", "reused", "311060"}
        )

    def test_empty_text_is_empty_set(self) -> None:
        assert tokenize("   the a an  ") == frozenset()


class TestContainment:
    def test_full_containment(self) -> None:
        assert containment(frozenset({"a", "b"}), frozenset({"a", "b", "c"})) == 1.0

    def test_partial(self) -> None:
        assert containment(frozenset({"a", "b", "c", "d"}), frozenset({"a", "b"})) == 0.5

    def test_empty_memory_is_zero(self) -> None:
        assert containment(frozenset(), frozenset({"a"})) == 0.0


class TestLoopClosure:
    def _tracker(self) -> RecallOutcomeTracker:
        return RecallOutcomeTracker(ttl_s=100.0, min_tokens=3)

    def test_echoed_memory_marks_recall_used(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "agentA")
        t = self._tracker()
        t.observe_recall(
            event_id,
            "agentA",
            ["incremental vector index reuse content hash"],
            now=0.0,
        )
        used = t.note_text(
            "we applied incremental vector index reuse via content hash today",
            "agentA",
            ledger,
            now=1.0,
        )
        assert used == [event_id]
        assert _usage(ledger) == {"e1": True}
        assert t.pending_count == 0

    def test_unrelated_remember_does_not_match(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "agentA")
        t = self._tracker()
        t.observe_recall(
            event_id,
            "agentA",
            ["incremental vector index reuse content hash"],
            now=0.0,
        )
        used = t.note_text(
            "completely different topic about coffee brewing", "agentA", ledger, now=1.0
        )
        assert used == []
        assert _usage(ledger) == {"e1": None}
        assert t.pending_count == 1  # still pending, not ruled used

    def test_match_is_per_agent(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "agentA")
        t = self._tracker()
        t.observe_recall(
            event_id,
            "agentA",
            ["incremental vector index reuse content hash"],
            now=0.0,
        )
        # Same text, different agent — must not close agentA's loop.
        used = t.note_text(
            "incremental vector index reuse content hash", "agentB", ledger, now=1.0
        )
        assert used == []
        assert _usage(ledger) == {"e1": None}
        assert t.pending_count == 1

    def test_short_memories_are_not_buffered(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "agentA")
        t = self._tracker()  # min_tokens=3
        t.observe_recall(event_id, "agentA", ["ok yes"], now=0.0)  # 2 content tokens
        assert t.pending_count == 0
        # Nothing to close, nothing recorded.
        assert t.note_text("ok yes done now", "agentA", ledger, now=1.0) == []
        assert _usage(ledger) == {"e1": None}

    def test_only_strongest_returned_memory_needs_to_echo(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "agentA")
        t = self._tracker()
        t.observe_recall(
            event_id,
            "agentA",
            ["totally unrelated alpha beta gamma", "incremental vector index reuse content hash"],
            now=0.0,
        )
        used = t.note_text(
            "incremental vector index reuse content hash", "agentA", ledger, now=1.0
        )
        assert used == [event_id]
        assert _usage(ledger) == {"e1": True}


class TestExpiryAndBounds:
    def test_stale_recall_ruled_unused(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_1 = _event(ledger, "e1", "a")
        event_2 = _event(ledger, "e2", "a")
        t = RecallOutcomeTracker(ttl_s=10.0, min_tokens=3)
        t.observe_recall(event_1, "a", ["alpha beta gamma delta"], ledger=ledger, now=0.0)
        # A later observe past the TTL expires the first as unused.
        t.observe_recall(event_2, "a", ["epsilon zeta eta theta"], ledger=ledger, now=20.0)
        assert _usage(ledger) == {"e1": False, "e2": None}
        assert t.pending_count == 1

    def test_note_text_expires_before_matching(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_id = _event(ledger, "e1", "a")
        t = RecallOutcomeTracker(ttl_s=10.0, min_tokens=3)
        t.observe_recall(event_id, "a", ["alpha beta gamma delta"], now=0.0)
        used = t.note_text("alpha beta gamma delta", "a", ledger, now=50.0)
        # Expired by TTL before the (matching) text arrives → unused, not used.
        assert used == []
        assert _usage(ledger) == {"e1": False}

    def test_overflow_rules_oldest_unused(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        events = [_event(ledger, label, "a") for label in ("e1", "e2", "e3")]
        t = RecallOutcomeTracker(ttl_s=1000.0, max_pending=2, min_tokens=3)
        t.observe_recall(events[0], "a", ["alpha beta gamma"], ledger=ledger, now=0.0)
        t.observe_recall(events[1], "a", ["delta epsilon zeta"], ledger=ledger, now=1.0)
        t.observe_recall(events[2], "a", ["eta theta iota"], ledger=ledger, now=2.0)
        assert _usage(ledger) == {"e1": False, "e2": None, "e3": None}
        assert t.pending_count == 2

    def test_flush_rules_all_pending_unused(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        event_1 = _event(ledger, "e1", "a")
        event_2 = _event(ledger, "e2", "a")
        t = RecallOutcomeTracker(ttl_s=1000.0, min_tokens=3)
        t.observe_recall(event_1, "a", ["alpha beta gamma"], now=0.0)
        t.observe_recall(event_2, "a", ["delta epsilon zeta"], now=1.0)
        drained = t.flush(ledger)
        assert sorted(drained) == sorted([event_1, event_2])
        assert _usage(ledger) == {"e1": False, "e2": False}
        assert t.pending_count == 0

    def test_expire_without_ledger_still_drops(self) -> None:
        t = RecallOutcomeTracker(ttl_s=10.0, min_tokens=3)
        t.observe_recall("e1", "a", ["alpha beta gamma"], now=0.0)
        # observe_recall is called with no ledger; expiry must still prune.
        t.observe_recall("e2", "a", ["delta epsilon zeta"], now=100.0)
        assert t.pending_count == 1

    def test_overflow_without_ledger_still_enforces_bound(self) -> None:
        t = RecallOutcomeTracker(ttl_s=1000.0, max_pending=1, min_tokens=3)
        t.observe_recall("e1", "a", ["alpha beta gamma"], now=0.0)
        t.observe_recall("e2", "a", ["delta epsilon zeta"], now=1.0)

        assert t.pending_count == 1

    def test_default_threshold_is_high(self) -> None:
        # A guard that the default is strict enough to resist incidental overlap.
        assert DEFAULT_MIN_CONTAINMENT >= 0.5


def test_expire_returns_empty_when_nothing_stale(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    event_id = _event(ledger, "e1", "a")
    t = RecallOutcomeTracker(ttl_s=100.0, min_tokens=3)
    t.observe_recall(event_id, "a", ["alpha beta gamma"], now=0.0)
    assert t.expire(ledger, now=1.0) == []
    assert _usage(ledger) == {"e1": None}


def test_containment_threshold_boundary(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    event_id = _event(ledger, "e1", "a")
    t = RecallOutcomeTracker(ttl_s=100.0, min_tokens=3, min_containment=0.75)
    # memory has 4 tokens; text echoes 3 → containment 0.75, exactly the bar.
    t.observe_recall(event_id, "a", ["alpha beta gamma delta"], now=0.0)
    used = t.note_text("alpha beta gamma omitted", "a", ledger, now=1.0)
    assert used == [event_id]
    assert _usage(ledger) == {"e1": True}


def test_below_threshold_does_not_match(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    event_id = _event(ledger, "e1", "a")
    t = RecallOutcomeTracker(ttl_s=100.0, min_tokens=3, min_containment=0.75)
    t.observe_recall(event_id, "a", ["alpha beta gamma delta"], now=0.0)
    used = t.note_text("alpha beta only two", "a", ledger, now=1.0)  # 2/4 = 0.5 < 0.75
    assert used == []
    assert _usage(ledger) == {"e1": None}
    assert t.pending_count == 1
