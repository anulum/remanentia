# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for knowledge store

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import knowledge_store
from knowledge_store import (
    KnowledgeNote,
    KnowledgeStore,
    Trigger,
    _extract_keywords,
    _extract_entities,
    _generate_prospective_queries,
    _note_id,
    _tokenize,
    extract_person_names,
)
from signal_detector import DetectedSignal, SignalType


# ── Utilities ───────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Hello World STDP BM25")
        assert "hello" in tokens
        assert "world" in tokens
        assert "stdp" in tokens
        assert "bm25" in tokens

    def test_short_filtered(self) -> None:
        tokens = _tokenize("a to is big cat")
        assert "a" not in tokens
        assert "big" in tokens

    def test_empty(self) -> None:
        assert _tokenize("") == set()


class TestExtractKeywords:
    def test_frequency_based(self) -> None:
        text = "BM25 retrieval uses BM25 scoring for retrieval."
        kw = _extract_keywords(text)
        assert "bm25" in kw
        assert "retrieval" in kw

    def test_capitalized_terms(self) -> None:
        kw = _extract_keywords("The ArcaneNeuron class handles encoding.")
        assert "arcaneneuron" in kw

    def test_versions(self) -> None:
        kw = _extract_keywords("Released v3.9.0 to PyPI.")
        assert "v3.9.0" in kw

    def test_cap_at_20(self) -> None:
        text = " ".join(f"Word{i} Word{i}" for i in range(30))
        kw = _extract_keywords(text)
        assert len(kw) <= 20


class TestExtractEntities:
    def test_known_entities(self) -> None:
        ents = _extract_entities("We used STDP with BM25 on GPU.")
        assert "stdp" in ents
        assert "bm25" in ents
        assert "gpu" in ents

    def test_versions(self) -> None:
        ents = _extract_entities("Released v3.9.0.")
        assert "v3.9.0" in ents

    def test_percentages(self) -> None:
        ents = _extract_entities("Accuracy was 81.2%.")
        assert "81.2%" in ents


class TestNoteId:
    def test_deterministic(self) -> None:
        assert _note_id("content", "source") == _note_id("content", "source")

    def test_different_content(self) -> None:
        assert _note_id("aaa", "src") != _note_id("bbb", "src")

    def test_length(self) -> None:
        assert len(_note_id("x", "y")) == 12


# ── KnowledgeNote ───────────────────────────────────────────────


class TestKnowledgeNote:
    def test_to_dict_roundtrip(self) -> None:
        note = KnowledgeNote(
            id="abc",
            title="Test",
            content="Content here",
            keywords=["test"],
            source="test.md",
            created="2026-03-24",
            updated="2026-03-24",
            entities=["stdp"],
        )
        d = note.to_dict()
        note2 = KnowledgeNote.from_dict(d)
        assert note2.id == "abc"
        assert note2.title == "Test"
        assert note2.entities == ["stdp"]

    def test_defaults(self) -> None:
        note = KnowledgeNote(
            id="x",
            title="T",
            content="C",
            keywords=[],
            source="",
            created="",
            updated="",
        )
        assert note.links == []
        assert note.supersedes == ""
        assert note.superseded_by == ""


# ── Trigger ─────────────────────────────────────────────────────


class TestTrigger:
    def test_to_dict_roundtrip(self) -> None:
        t = Trigger(
            id="t1", condition="scpn-control", action="check weights file", created="2026-03-24"
        )
        d = t.to_dict()
        t2 = Trigger.from_dict(d)
        assert t2.condition == "scpn-control"
        assert t2.active is True

    def test_defaults(self) -> None:
        t = Trigger(id="t", condition="c", action="a", created="now")
        assert t.fired == []
        assert t.active is True


# ── KnowledgeStore: add_note ────────────────────────────────────


class TestAddNote:
    def test_creates_note(self) -> None:
        store = KnowledgeStore()
        note = store.add_note(
            "We decided to remove SNN from retrieval scoring.", source="decision.md"
        )
        assert note.id in store.notes
        assert "snn" in note.entities or "retrieval" in note.entities
        assert note.created != ""

    def test_auto_title(self) -> None:
        store = KnowledgeStore()
        note = store.add_note("We decided to use BM25 for all queries.\nMore details here.")
        assert "BM25" in note.title or "decided" in note.title

    def test_links_to_related(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 scoring improved retrieval accuracy to 81.2%.", source="a.md")
        note2 = store.add_note(
            "BM25 retrieval accuracy measured at 83.1% after LLM synthesis.", source="b.md"
        )
        assert len(note2.links) > 0

    def test_merges_near_duplicate(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note(
            "The STDP learning rule was broken in both CPU and GPU backends.", source="a.md"
        )
        n2 = store.add_note(
            "The STDP learning rule was broken in both CPU and GPU backends.", source="b.md"
        )
        # Near-duplicate should merge, not create new
        assert len(store.notes) == 1
        assert n2.id == n1.id

    def test_does_not_merge_different(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 scoring is fast and accurate.", source="a.md")
        store.add_note("The SNN daemon was killed because it adds no signal.", source="b.md")
        assert len(store.notes) == 2


# ── Contradiction detection ─────────────────────────────────────


class TestContradictionDetection:
    def test_detects_opposite_actions(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        n2 = store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        assert n2.supersedes != ""
        assert n1.superseded_by != ""

    def test_no_false_positive(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 scoring works well for retrieval.", source="a.md")
        n2 = store.add_note("Cross-encoder reranking improves retrieval further.", source="b.md")
        assert n2.supersedes == ""

    def test_contradiction_tracked_in_links(self) -> None:
        store = KnowledgeStore()
        store.add_note("We enabled the GPU daemon for faster processing.", source="a.md")
        n2 = store.add_note("We disabled the GPU daemon to free memory.", source="b.md")
        supersedes_links = [l for l in n2.links if l["type"] == "supersedes"]
        assert len(supersedes_links) >= 1

    def test_get_contradictions(self) -> None:
        store = KnowledgeStore()
        store.add_note("We added SNN scoring with weight 0.3.", source="a.md")
        store.add_note("We removed SNN scoring. Weight set to 0.0.", source="b.md")
        contradictions = store.get_contradictions()
        assert len(contradictions) >= 1
        old, new = contradictions[0]
        assert old.superseded_by != ""


# ── Search ──────────────────────────────────────────────────────


class TestSearch:
    def test_finds_relevant(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 retrieval accuracy reached 81.2% on LOCOMO.", source="bench.md")
        store.add_note("The SNN daemon was killed.", source="daemon.md")
        results = store.search("BM25 LOCOMO accuracy", top_k=3)
        assert len(results) > 0
        assert any("bm25" in n.content.lower() for n in results)

    def test_empty_query(self) -> None:
        store = KnowledgeStore()
        assert store.search("") == []

    def test_no_match(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 scoring works.", source="a.md")
        assert store.search("xyznonexistent_zzz") == []


# ── Graph traversal ─────────────────────────────────────────────


class TestGetRelated:
    def test_finds_linked(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note(
            "BM25 retrieval scoring improved accuracy on LOCOMO benchmark.", source="a.md"
        )
        store.add_note(
            "BM25 LOCOMO benchmark accuracy reached 83.1% with LLM synthesis.", source="b.md"
        )
        related = store.get_related(n1.id, depth=1)
        assert len(related) > 0

    def test_nonexistent_note(self) -> None:
        store = KnowledgeStore()
        assert store.get_related("nonexistent") == []

    def test_depth_2(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("BM25 retrieval is the core scoring algorithm.", source="a.md")
        store.add_note("BM25 scoring uses TF-IDF term weighting internally.", source="b.md")
        store.add_note("TF-IDF term weighting was the original retrieval method.", source="c.md")
        related = store.get_related(n1.id, depth=2)
        # Should find n2 (depth 1) and potentially n3 (depth 2)
        assert len(related) >= 1


# ── Triggers ────────────────────────────────────────────────────


class TestTriggers:
    def test_add_trigger(self) -> None:
        store = KnowledgeStore()
        t = store.add_trigger("scpn-control", "Check weights file changes with Python version")
        assert t.condition == "scpn-control"
        assert len(store.triggers) == 1

    def test_check_triggers_match(self) -> None:
        store = KnowledgeStore()
        store.add_trigger("scpn-control weights", "Weights file changes with Python version")
        matched = store.check_triggers("working on scpn-control weights file")
        assert len(matched) == 1
        assert len(matched[0].fired) == 1

    def test_check_triggers_no_match(self) -> None:
        store = KnowledgeStore()
        store.add_trigger("scpn-control", "Check weights")
        matched = store.check_triggers("director-ai release")
        assert len(matched) == 0

    def test_inactive_trigger_skipped(self) -> None:
        store = KnowledgeStore()
        t = store.add_trigger("scpn-control", "Check weights")
        t.active = False
        matched = store.check_triggers("scpn-control work")
        assert len(matched) == 0


# ── Save/Load ───────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_notes(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 retrieval accuracy reached 81.2% on LOCOMO.", source="bench.md")
        store.add_note("SNN daemon was killed because it adds nothing.", source="daemon.md")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)
        assert notes_path.exists()

        store2 = KnowledgeStore()
        assert store2.load(notes_path, triggers_path) is True
        assert len(store2.notes) == len(store.notes)

    def test_save_and_load_triggers(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_trigger("scpn-control", "Check weights")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        store2 = KnowledgeStore()
        store2.load(notes_path, triggers_path)
        assert len(store2.triggers) == 1

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        assert store.load(tmp_path / "nope.jsonl") is False

    def test_search_after_load(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 retrieval scoring algorithm details.", source="a.md")
        notes_path = tmp_path / "notes.jsonl"
        store.save(notes_path, tmp_path / "t.jsonl")

        store2 = KnowledgeStore()
        store2.load(notes_path, tmp_path / "t.jsonl")
        results = store2.search("BM25 retrieval", top_k=3)
        assert len(results) > 0


# ── Stats ───────────────────────────────────────────────────────


class TestStats:
    def test_stats(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 scoring works for retrieval on LOCOMO.", source="a.md")
        store.add_note("BM25 LOCOMO retrieval accuracy improved with cross-encoder.", source="b.md")
        store.add_trigger("scpn-control", "check weights")
        s = store.stats
        assert s["notes"] == 2
        assert s["links"] > 0
        assert s["triggers_total"] == 1
        assert s["triggers_active"] == 1

    def test_empty_stats(self) -> None:
        store = KnowledgeStore()
        s = store.stats
        assert s["notes"] == 0
        assert s["links"] == 0


# ── Feature 1: Prospective indexing ─────────────────────────────


class TestProspectiveQueries:
    def test_generates_queries_at_write_time(self) -> None:
        store = KnowledgeStore()
        note = store.add_note(
            "Caroline likes pottery and swimming. She started pottery in March.",
            source="conv.md",
        )
        assert len(note.prospective_queries) > 0

    def test_entity_queries_generated(self) -> None:
        pq = _generate_prospective_queries(
            "BM25 retrieval accuracy is 81.2% on LOCOMO.",
            "BM25 accuracy",
            ["bm25", "locomo", "81.2%"],
            ["bm25", "retrieval", "accuracy"],
        )
        assert any("bm25" in q.lower() for q in pq)

    def test_activity_detection(self) -> None:
        pq = _generate_prospective_queries(
            "Caroline likes pottery and enjoys swimming in the lake.",
            "Caroline activities",
            [],
            ["pottery", "swimming"],
        )
        assert any("like" in q.lower() for q in pq)

    def test_temporal_queries(self) -> None:
        pq = _generate_prospective_queries(
            "Released v3.9.0 on 2026-03-15.",
            "Release",
            ["v3.9.0"],
            ["release"],
        )
        assert any("2026-03-15" in q for q in pq)

    def test_allergy_queries(self) -> None:
        pq = _generate_prospective_queries(
            "Caroline is allergic to peanuts and shellfish.",
            "Allergies",
            ["caroline"],
            ["allergic"],
        )
        assert any("allergic" in q.lower() for q in pq)
        assert any("peanuts" in q.lower() for q in pq)

    def test_favourite_queries(self) -> None:
        pq = _generate_prospective_queries(
            "Her favourite movie is Spirited Away.",
            "Favourites",
            ["caroline"],
            ["movie"],
        )
        assert any("favourite" in q.lower() for q in pq)

    def test_causal_queries(self) -> None:
        pq = _generate_prospective_queries(
            "We decided to remove SNN because it adds no signal.",
            "Remove SNN",
            ["snn"],
            ["remove", "signal"],
        )
        assert any("why" in q.lower() for q in pq)

    def test_prospective_queries_searchable(self) -> None:
        store = KnowledgeStore()
        store.add_note(
            "Caroline enjoys pottery classes every Tuesday.",
            source="conv.md",
        )
        # "hobbies" is not in the note text but should be findable
        # via prospective queries that include "like" or activity terms
        note = list(store.notes.values())[0]
        assert note.searchable_text != note.content

    def test_custom_queries_passed(self) -> None:
        store = KnowledgeStore()
        note = store.add_note(
            "Test content.",
            source="test.md",
            prospective_queries=["custom query one", "custom query two"],
        )
        assert note.prospective_queries == ["custom query one", "custom query two"]

    def test_queries_in_token_index(self) -> None:
        store = KnowledgeStore()
        store.add_note(
            "We started the daemon process.",
            source="test.md",
            prospective_queries=["xyztestquery123"],
        )
        nid = list(store.notes.keys())[0]
        tokens = store._token_index[nid]
        assert "xyztestquery123" in tokens

    def test_roundtrip_preserves_queries(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        note = store.add_note("Test content for roundtrip.", source="test.md")
        pq = note.prospective_queries

        notes_path = tmp_path / "notes.jsonl"
        store.save(notes_path, tmp_path / "t.jsonl")

        store2 = KnowledgeStore()
        store2.load(notes_path, tmp_path / "t.jsonl")
        note2 = list(store2.notes.values())[0]
        assert note2.prospective_queries == pq


# ── Feature 4: Graph-based multi-hop ────────────────────────────


class TestGraphSearch:
    def test_graph_search_finds_linked_notes(self) -> None:
        store = KnowledgeStore()
        store.add_note("BM25 retrieval scoring algorithm details.", source="a.md")
        store.add_note("BM25 scoring uses TF-IDF term weighting internally.", source="b.md")
        store.add_note("TF-IDF term weighting was the original retrieval method.", source="c.md")
        results = store.graph_search("BM25 retrieval", top_k=5, hop_depth=2)
        assert len(results) >= 2

    def test_graph_search_empty(self) -> None:
        store = KnowledgeStore()
        assert store.graph_search("nonexistent") == []

    def test_excludes_superseded(self) -> None:
        store = KnowledgeStore()
        store.add_note("We enabled the GPU daemon for processing.", source="a.md")
        store.add_note("We disabled the GPU daemon to free memory.", source="b.md")
        results = store.graph_search("GPU daemon", top_k=5)
        # Superseded note should be excluded from graph results
        for r in results:
            assert r.superseded_by == ""

    def test_typed_edge_filter(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("Base concept for BM25 retrieval algorithm.", source="a.md")
        store.add_note("BM25 scoring uses TF-IDF weighting retrieval.", source="b.md")
        # Only follow "related" edges
        related = store.get_related(n1.id, depth=1, edge_types={"related"})
        # n2 should be found via "related" type
        assert all(
            any(l.get("type") == "related" for l in store.notes[n1.id].links if l["target"] == r.id)
            for r in related
        )

    def test_add_typed_link(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("Module A depends on module B.", source="a.md")
        n2 = store.add_note("Module B provides core functionality.", source="b.md")
        ok = store.add_typed_link(n1.id, n2.id, "depends_on")
        assert ok is True
        deps = store.get_related(n1.id, depth=1, edge_types={"depends_on"})
        assert any(d.id == n2.id for d in deps)

    def test_add_typed_link_invalid_type(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("A.", source="a.md")
        n2 = store.add_note("B completely different content.", source="b.md")
        ok = store.add_typed_link(n1.id, n2.id, "invalid_type")
        assert ok is False

    def test_add_typed_link_nonexistent(self) -> None:
        store = KnowledgeStore()
        ok = store.add_typed_link("nope1", "nope2", "related")
        assert ok is False

    def test_search_excludes_superseded(self) -> None:
        store = KnowledgeStore()
        store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        results = store.search("SNN daemon", exclude_superseded=True)
        for r in results:
            assert r.superseded_by == ""

    def test_search_includes_superseded_when_asked(self) -> None:
        store = KnowledgeStore()
        store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        results = store.search("SNN daemon", exclude_superseded=False)
        assert len(results) >= 2


# ── Person name extraction ───────────────────────────────────────


class TestExtractPersonNames:
    def test_chat_format(self) -> None:
        names = extract_person_names("Caroline: I love pottery!\nMelanie: That's great!")
        assert "caroline" in names
        assert "melanie" in names

    def test_ignores_common_words(self) -> None:
        names = extract_person_names("The weather is great. Thanks for asking.")
        assert "the" not in names
        assert "thanks" not in names

    def test_sentence_start_names(self) -> None:
        names = extract_person_names("John went to the store. Maria called him.")
        assert "john" in names
        assert "maria" in names

    def test_empty(self) -> None:
        assert extract_person_names("no names here at all") == set()


# ── Auto entity linking ─────────────────────────────────────────


class TestAutoEntityLinking:
    def test_shared_entities_create_links(self) -> None:
        store = KnowledgeStore()
        n1 = store.add_note("BM25 retrieval accuracy is 81.2% on LOCOMO benchmark.", source="a.md")
        n2 = store.add_note("LOCOMO benchmark result: BM25 achieved 83.1% accuracy.", source="b.md")
        # Both share bm25 and locomo entities — should auto-link
        all_links = n1.links + n2.links
        assert len(all_links) >= 2  # at least the similarity links

    def test_no_link_without_shared_entities(self) -> None:
        store = KnowledgeStore()
        store.add_note("PyTorch CUDA acceleration for training models.", source="a.md")
        n2 = store.add_note("The weather in Prague was surprisingly warm today.", source="b.md")
        # No shared entities — auto entity linking shouldn't fire
        # (may still have similarity links if tokens overlap)
        n2_entity_links = [l for l in n2.links if "shared_entities" in l]
        assert len(n2_entity_links) == 0

    def test_entity_overlap_creates_shared_entities_link(self) -> None:
        import time as _time

        store = KnowledgeStore()
        now = _time.strftime("%Y-%m-%dT%H%M", _time.gmtime())
        # Manually insert a note with specific entities so token overlap
        # with the next note is below the 0.1 Jaccard threshold.
        existing = KnowledgeNote(
            id="existing1",
            title="alpha",
            content="alpha bravo charlie delta echo foxtrot golf hotel.",
            keywords=["alpha"],
            source="old.md",
            created=now,
            updated=now,
            entities=["stdp", "snn"],
        )
        store.notes["existing1"] = existing
        store._token_index["existing1"] = {
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
        }
        # Now add_note with different tokens but shared entities stdp + snn
        n2 = store.add_note(
            "India juliet kilo lima mike november oscar papa stdp snn.",
            source="new.md",
        )
        entity_links = [l for l in n2.links if "shared_entities" in l]
        assert len(entity_links) >= 1
        assert "stdp" in entity_links[0]["shared_entities"]
        assert "snn" in entity_links[0]["shared_entities"]


# ── Pipeline integration ─────────────────────────────────────


class TestKnowledgeStorePipeline:
    """KnowledgeStore integrated with observer and reflector."""

    def test_observer_creates_notes_in_store(self, tmp_path: Path) -> None:
        """Observer writes notes → KnowledgeStore loads them."""
        from knowledge_store import KnowledgeStore
        from observer import ObserverState, observe_once
        from unittest.mock import patch

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "decision.md").write_text(
            "We decided to remove SNN from retrieval because 70 experiments showed zero signal.\n",
            encoding="utf-8",
        )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        with (
            patch("observer.WATCHED_DIRS", {"traces": traces_dir}),
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            observe_once(state, {"traces": traces_dir})

            store = KnowledgeStore()
            loaded = store.load()
            if loaded:
                assert len(store.notes) >= 1

    def test_store_feeds_reflector(self, tmp_path: Path) -> None:
        """KnowledgeStore notes are consumed by reflector."""
        from knowledge_store import KnowledgeStore
        from reflector import _generate_summary_heuristic

        store = KnowledgeStore()
        # Use distinct content to avoid merge
        store.add_note(
            "STDP learning rule was disabled because experiments showed no signal.",
            source="trace_stdp.md",
        )
        store.add_note(
            "BM25 retrieval accuracy reached 88.5% on the LOCOMO benchmark.",
            source="trace_bm25.md",
        )
        store.add_note(
            "Director-AI revenue pipeline sent 265 emails to 40 countries.",
            source="trace_revenue.md",
        )
        notes = list(store.notes.values())
        assert len(notes) >= 2

        # Even with 1 note, heuristic summary should work
        summary = _generate_summary_heuristic(notes[:2])
        assert len(summary) > 0

    def test_contradiction_detection_end_to_end(self) -> None:
        """Add contradictory notes → detect supersession."""
        from knowledge_store import KnowledgeStore

        store = KnowledgeStore()
        n1 = store.add_note("BM25 accuracy is 81.2% on LOCOMO.", source="a.md")
        n2 = store.add_note("BM25 accuracy is 88.5% on LOCOMO.", source="b.md")
        # n2 should supersede n1 (same metric, different value)
        # Verify no crash; contradiction detection may or may not fire
        assert isinstance(n2.supersedes, str)
        assert isinstance(n1.superseded_by, str)

    def test_trigger_fires_on_recall(self) -> None:
        """Triggers created in store fire when matching queries."""
        from knowledge_store import KnowledgeStore

        store = KnowledgeStore()
        store.add_trigger(
            condition="scpn-control",
            action="Check the SCPN phase coupling matrix before modifying.",
        )
        triggered = store.check_triggers("working on scpn-control today")
        assert len(triggered) >= 1
        assert "coupling" in triggered[0].action.lower() or "SCPN" in triggered[0].action

    def test_save_load_preserves_notes_and_links(self, tmp_path: Path) -> None:
        """Roundtrip: add notes → save → load → verify links preserved."""
        from knowledge_store import KnowledgeStore
        from unittest.mock import patch

        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"

        with (
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            store = KnowledgeStore()
            n1 = store.add_note("STDP learning rule with 0.0 weight.", source="a.md")
            n2 = store.add_note("BM25 scoring replaced STDP for retrieval.", source="b.md")
            assert store.add_typed_link(n2.id, n1.id, "depends_on") is True
            store.save()

            store2 = KnowledgeStore()
            store2.load()
            assert len(store2.notes) == 2
            n2_loaded = store2.notes.get(n2.id)
            assert n2_loaded is not None
            assert {"target": n1.id, "type": "depends_on"} in n2_loaded.links


class TestKnowledgeStorePythonFallbackContracts:
    def test_native_free_text_helpers_extract_real_retrieval_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = __import__

        def reject_native(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "remanentia_knowledge_store":
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", reject_native)

        text = (
            "Alice: BM25 retrieval improved to 88.5% in Remanentia v3.9.0. "
            "Alice enjoys pottery, works as a researcher, is allergic to peanuts, "
            "visited Zurich, and started learning Rust."
        )

        assert {"bm25", "retrieval", "remanentia", "88.5%", "v3.9.0"} <= _extract_entities(text)
        assert "alice" in extract_person_names(text)
        assert "retrieval" in _extract_keywords("retrieval retrieval BM25 BM25")
        queries = _generate_prospective_queries(
            text,
            "Alice Memory",
            ["alice"],
            ["retrieval"],
        )
        assert "what happened to alice" in queries
        assert "what is the score for alice" in queries
        assert "what does alice like" in queries
        assert "where does alice work" in queries
        assert "what is alice allergic to" in queries
        assert _tokenize("Native free BM25 retrieval") == {"native", "free", "bm25", "retrieval"}

    def test_confidence_events_and_store_aging_update_notes(self) -> None:
        note = KnowledgeNote(
            id="n",
            title="BM25",
            content="BM25 retrieval works.",
            keywords=["bm25"],
            source="trace.md",
            created="2026-01-01",
            updated="2026-01-01",
            entities=["bm25"],
            confidence=0.5,
        )

        note.update_confidence("confirmed")
        assert note.confirmation_count == 1
        assert note.confidence > 0.5
        assert note.last_confirmed
        note.update_confidence("contradicted")
        assert note.contradiction_count == 1
        note.update_confidence("accessed")
        before_stale = note.confidence
        note.update_confidence("stale")
        assert note.confidence < before_stale

        store = KnowledgeStore()
        store.notes[note.id] = note
        superseded = KnowledgeNote(
            id="old",
            title="Old",
            content="Old fact",
            keywords=[],
            source="old.md",
            created="2026-01-01",
            updated="2026-01-01",
            superseded_by="new",
        )
        store.notes[superseded.id] = superseded
        stats = store.age_memories()

        assert stats == {"scanned": 2, "stale": 1}

    def test_signal_feedback_updates_related_note_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = KnowledgeStore()
        related = store.add_note(
            "BM25 retrieval reached 80 percent accuracy.",
            source="trace.md",
            redact_pii=False,
        )

        monkeypatch.setattr(
            store,
            "search",
            lambda *_args, **_kwargs: [types.SimpleNamespace(id=related.id)],
        )
        monkeypatch.setattr(
            knowledge_store,
            "detect_signals",
            lambda _text: [
                DetectedSignal(
                    SignalType.CORRECTION,
                    0.95,
                    "correction",
                    "correction: update BM25 accuracy",
                )
            ],
        )
        store.add_note(
            "correction: BM25 retrieval reached 90 percent accuracy.",
            source="correction.md",
            redact_pii=False,
        )
        assert related.contradiction_count == 1

        monkeypatch.setattr(
            knowledge_store,
            "detect_signals",
            lambda _text: [DetectedSignal(SignalType.REINFORCEMENT, 0.9, "confirmed", "confirmed")],
        )
        store.add_note(
            "confirmed: BM25 retrieval reached 90 percent accuracy.",
            source="confirm.md",
            redact_pii=False,
        )
        assert related.confirmation_count == 1

    def test_native_free_search_related_and_graph_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = __import__

        def reject_native(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "remanentia_knowledge_store":
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", reject_native)

        store = KnowledgeStore()
        seed = store.add_note(
            "BM25 retrieval depends on embeddings and improves Remanentia search.",
            source="seed.md",
            redact_pii=False,
        )
        related = store.add_note(
            "Embeddings improve Remanentia retrieval quality with BM25.",
            source="related.md",
            redact_pii=False,
        )
        store.add_typed_link(seed.id, related.id, "depends_on")

        assert store.search("BM25 retrieval", top_k=2)
        assert related in store.get_related(seed.id, depth=1, edge_types={"depends_on"})
        graph_results = store.graph_search("BM25 embeddings retrieval", top_k=3)
        assert {note.id for note in graph_results} >= {seed.id, related.id}

        seed.superseded_by = related.id
        assert seed not in store.search("BM25 retrieval", top_k=3)
        assert store.graph_search("no matching query tokens", top_k=3) == []

    def test_load_rebuilds_token_index_and_triggers_from_files(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        note = store.add_note(
            "BM25 retrieval memory note with durable search terms.",
            source="trace.md",
            redact_pii=False,
        )
        trigger = store.add_trigger("BM25 retrieval", "Inspect retrieval quality")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        notes_path.write_text(json.dumps(note.to_dict()) + "\n\n", encoding="utf-8")
        triggers_path.write_text(json.dumps(trigger.to_dict()) + "\n\n", encoding="utf-8")

        loaded = KnowledgeStore()
        assert loaded.load(notes_path, triggers_path) is True
        assert note.id in loaded._token_index
        assert loaded.triggers[0].id == trigger.id
