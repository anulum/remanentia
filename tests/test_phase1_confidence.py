# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for Phase 1 Dynamic Confidence Scoring

"""Tests for confidence scoring on AtomicFact, KnowledgeNote, and KnowledgeStore.

Covers:
- All 4 event types (confirmed, contradicted, accessed, stale)
- Asymptotic ceiling, floor clamping, repeated events
- Serialisation round-trip with confidence fields
- KnowledgeStore.age_memories integration
- ArcaneRetriever channel weights in RRF fusion
"""

from __future__ import annotations

from fact_decomposer import AtomicFact
from knowledge_store import KnowledgeNote, KnowledgeStore


# ── AtomicFact confidence ───────────────────────────────────────────


class TestAtomicFactConfidence:
    def _make_fact(self, confidence: float = 0.5) -> AtomicFact:
        return AtomicFact(
            text="Test fact",
            session_idx=0,
            turn_idx=0,
            role="user",
            fact_type="state",
            confidence=confidence,
        )

    def test_confirmed_increases(self):
        f = self._make_fact(0.5)
        f.update_confidence("confirmed")
        assert f.confidence > 0.5
        assert f.confirmation_count == 1
        assert f.last_confirmed != ""

    def test_confirmed_asymptotic_ceiling(self):
        """Repeated confirmations approach 1.0 but never exceed it."""
        f = self._make_fact(0.5)
        for _ in range(100):
            f.update_confidence("confirmed")
        assert f.confidence <= 1.0
        assert f.confidence > 0.99

    def test_contradicted_decreases(self):
        f = self._make_fact(0.5)
        f.update_confidence("contradicted")
        assert f.confidence < 0.5
        assert f.contradiction_count == 1

    def test_contradicted_floor_at_zero(self):
        """Repeated contradictions never go below 0.0."""
        f = self._make_fact(0.1)
        for _ in range(50):
            f.update_confidence("contradicted")
        assert f.confidence >= 0.0

    def test_accessed_small_boost(self):
        f = self._make_fact(0.5)
        f.update_confidence("accessed")
        assert f.confidence == 0.52

    def test_accessed_capped_at_one(self):
        f = self._make_fact(0.99)
        f.update_confidence("accessed")
        assert f.confidence <= 1.0

    def test_stale_small_decay(self):
        f = self._make_fact(0.5)
        f.update_confidence("stale")
        assert f.confidence == 0.49

    def test_stale_floor_at_threshold(self):
        """Stale decay stops at 0.1, not 0.0."""
        f = self._make_fact(0.11)
        f.update_confidence("stale")
        assert f.confidence >= 0.1

    def test_confirmation_then_contradiction(self):
        """Contradiction after confirmation produces net decrease."""
        f = self._make_fact(0.5)
        f.update_confidence("confirmed")
        mid = f.confidence
        f.update_confidence("contradicted")
        assert f.confidence < mid

    def test_default_fields(self):
        f = self._make_fact()
        assert f.confidence == 0.5
        assert f.confirmation_count == 0
        assert f.contradiction_count == 0
        assert f.last_confirmed == ""
        assert f.source_quality == "inferred"


# ── KnowledgeNote confidence ────────────────────────────────────────


class TestKnowledgeNoteConfidence:
    def _make_note(self, confidence: float = 0.5) -> KnowledgeNote:
        return KnowledgeNote(
            id="test-id",
            title="Test",
            content="Test content",
            keywords=[],
            source="test",
            created="2026-01-01",
            updated="2026-01-01",
            confidence=confidence,
        )

    def test_confirmed(self):
        n = self._make_note(0.5)
        n.update_confidence("confirmed")
        assert n.confidence == 0.5 + 0.1 * (1.0 - 0.5)
        assert n.confirmation_count == 1

    def test_contradicted(self):
        n = self._make_note(0.5)
        n.update_confidence("contradicted")
        assert n.confidence == 0.35
        assert n.contradiction_count == 1

    def test_accessed(self):
        n = self._make_note(0.5)
        n.update_confidence("accessed")
        assert n.confidence == 0.52

    def test_stale(self):
        n = self._make_note(0.5)
        n.update_confidence("stale")
        assert n.confidence == 0.49

    def test_serialisation_round_trip(self):
        """Confidence fields survive to_dict → from_dict."""
        n = self._make_note(0.8)
        n.update_confidence("confirmed")
        n.update_confidence("contradicted")
        d = n.to_dict()
        assert "confidence" in d
        assert "confirmation_count" in d
        assert "contradiction_count" in d
        assert "source_quality" in d
        restored = KnowledgeNote.from_dict(d)
        assert restored.confidence == n.confidence
        assert restored.confirmation_count == n.confirmation_count
        assert restored.contradiction_count == n.contradiction_count

    def test_from_dict_defaults(self):
        """Loading old notes without confidence fields gets safe defaults."""
        d = {
            "id": "old",
            "title": "Old",
            "content": "Old content",
            "keywords": [],
            "source": "old",
            "created": "",
            "updated": "",
        }
        n = KnowledgeNote.from_dict(d)
        assert n.confidence == 0.5
        assert n.confirmation_count == 0
        assert n.source_quality == "inferred"


# ── KnowledgeStore.age_memories ─────────────────────────────────────


class TestAgeMemories:
    def test_age_memories_decays_active_notes(self):
        store = KnowledgeStore()
        n1 = KnowledgeNote(
            id="n1",
            title="A",
            content="a",
            keywords=[],
            source="s",
            created="",
            updated="",
            confidence=0.8,
        )
        n2 = KnowledgeNote(
            id="n2",
            title="B",
            content="b",
            keywords=[],
            source="s",
            created="",
            updated="",
            confidence=0.3,
        )
        store.notes = {"n1": n1, "n2": n2}
        stats = store.age_memories()
        assert stats["scanned"] == 2
        assert stats["stale"] == 2
        assert n1.confidence < 0.8
        assert n2.confidence < 0.3

    def test_age_memories_skips_superseded(self):
        store = KnowledgeStore()
        n = KnowledgeNote(
            id="n1",
            title="Old",
            content="old",
            keywords=[],
            source="s",
            created="",
            updated="",
            confidence=0.8,
            superseded_by="n2",
        )
        store.notes = {"n1": n}
        stats = store.age_memories()
        assert stats["stale"] == 0
        assert n.confidence == 0.8  # unchanged


# ── ArcaneRetriever channel weights ─────────────────────────────────


class TestChannelWeights:
    def test_session_channel_boosted(self):
        from arcane_retriever import ArcaneRetriever

        assert ArcaneRetriever.CHANNEL_WEIGHTS["session"] == 2.0

    def test_default_channels_at_one(self):
        from arcane_retriever import ArcaneRetriever

        for ch in ("bm25", "entity", "temporal"):
            assert ArcaneRetriever.CHANNEL_WEIGHTS[ch] == 1.0

    def test_cross_encoder_rerank_no_model(self):
        """Without model loaded, rerank returns input unchanged."""
        from arcane_retriever import ArcaneRetriever, FusedResult

        ar = ArcaneRetriever.__new__(ArcaneRetriever)
        ArcaneRetriever._ce_model = None
        ArcaneRetriever._ce_loading = False
        f = AtomicFact(text="test", session_idx=0, turn_idx=0, role="user", fact_type="state")
        results = [FusedResult(fact=f, rrf_score=1.0)]
        out = ar._cross_encoder_rerank("query", results)
        assert out is results

    def test_cross_encoder_rerank_model_false(self):
        """Model failed to load (False sentinel) — returns input unchanged."""
        from arcane_retriever import ArcaneRetriever, FusedResult

        ar = ArcaneRetriever.__new__(ArcaneRetriever)
        ArcaneRetriever._ce_model = False
        ArcaneRetriever._ce_loading = False
        f = AtomicFact(text="test", session_idx=0, turn_idx=0, role="user", fact_type="state")
        results = [FusedResult(fact=f, rrf_score=1.0)]
        out = ar._cross_encoder_rerank("query", results)
        assert out is results
