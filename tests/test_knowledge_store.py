# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for knowledge_store.py

from __future__ import annotations


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


# ── Utilities ───────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World STDP BM25")
        assert "hello" in tokens
        assert "world" in tokens
        assert "stdp" in tokens
        assert "bm25" in tokens

    def test_short_filtered(self):
        tokens = _tokenize("a to is big cat")
        assert "a" not in tokens
        assert "big" in tokens

    def test_empty(self):
        assert _tokenize("") == set()


class TestExtractKeywords:
    def test_frequency_based(self):
        text = "BM25 retrieval uses BM25 scoring for retrieval."
        kw = _extract_keywords(text)
        assert "bm25" in kw
        assert "retrieval" in kw

    def test_capitalized_terms(self):
        kw = _extract_keywords("The ArcaneNeuron class handles encoding.")
        assert "arcaneneuron" in kw

    def test_versions(self):
        kw = _extract_keywords("Released v3.9.0 to PyPI.")
        assert "v3.9.0" in kw

    def test_cap_at_20(self):
        text = " ".join(f"Word{i} Word{i}" for i in range(30))
        kw = _extract_keywords(text)
        assert len(kw) <= 20


class TestExtractEntities:
    def test_known_entities(self):
        ents = _extract_entities("We used STDP with BM25 on GPU.")
        assert "stdp" in ents
        assert "bm25" in ents
        assert "gpu" in ents

    def test_versions(self):
        ents = _extract_entities("Released v3.9.0.")
        assert "v3.9.0" in ents

    def test_percentages(self):
        ents = _extract_entities("Accuracy was 81.2%.")
        assert "81.2%" in ents


class TestNoteId:
    def test_deterministic(self):
        assert _note_id("content", "source") == _note_id("content", "source")

    def test_different_content(self):
        assert _note_id("aaa", "src") != _note_id("bbb", "src")

    def test_length(self):
        assert len(_note_id("x", "y")) == 12


# ── KnowledgeNote ───────────────────────────────────────────────


class TestKnowledgeNote:
    def test_to_dict_roundtrip(self):
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

    def test_defaults(self):
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
    def test_to_dict_roundtrip(self):
        t = Trigger(
            id="t1", condition="scpn-control", action="check weights file", created="2026-03-24"
        )
        d = t.to_dict()
        t2 = Trigger.from_dict(d)
        assert t2.condition == "scpn-control"
        assert t2.active is True

    def test_defaults(self):
        t = Trigger(id="t", condition="c", action="a", created="now")
        assert t.fired == []
        assert t.active is True


# ── KnowledgeStore: add_note ────────────────────────────────────


class TestAddNote:
    def test_creates_note(self):
        store = KnowledgeStore()
        note = store.add_note(
            "We decided to remove SNN from retrieval scoring.", source="decision.md"
        )
        assert note.id in store.notes
        assert "snn" in note.entities or "retrieval" in note.entities
        assert note.created != ""

    def test_auto_title(self):
        store = KnowledgeStore()
        note = store.add_note("We decided to use BM25 for all queries.\nMore details here.")
        assert "BM25" in note.title or "decided" in note.title

    def test_links_to_related(self):
        store = KnowledgeStore()
        store.add_note("BM25 scoring improved retrieval accuracy to 81.2%.", source="a.md")
        note2 = store.add_note(
            "BM25 retrieval accuracy measured at 83.1% after LLM synthesis.", source="b.md"
        )
        assert len(note2.links) > 0

    def test_merges_near_duplicate(self):
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

    def test_does_not_merge_different(self):
        store = KnowledgeStore()
        store.add_note("BM25 scoring is fast and accurate.", source="a.md")
        store.add_note("The SNN daemon was killed because it adds no signal.", source="b.md")
        assert len(store.notes) == 2


# ── Contradiction detection ─────────────────────────────────────


class TestContradictionDetection:
    def test_detects_opposite_actions(self):
        store = KnowledgeStore()
        n1 = store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        n2 = store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        assert n2.supersedes != ""
        assert n1.superseded_by != ""

    def test_no_false_positive(self):
        store = KnowledgeStore()
        store.add_note("BM25 scoring works well for retrieval.", source="a.md")
        n2 = store.add_note("Cross-encoder reranking improves retrieval further.", source="b.md")
        assert n2.supersedes == ""

    def test_contradiction_tracked_in_links(self):
        store = KnowledgeStore()
        n1 = store.add_note("We enabled the GPU daemon for faster processing.", source="a.md")
        n2 = store.add_note("We disabled the GPU daemon to free memory.", source="b.md")
        supersedes_links = [l for l in n2.links if l["type"] == "supersedes"]
        assert len(supersedes_links) >= 1

    def test_get_contradictions(self):
        store = KnowledgeStore()
        store.add_note("We added SNN scoring with weight 0.3.", source="a.md")
        store.add_note("We removed SNN scoring. Weight set to 0.0.", source="b.md")
        contradictions = store.get_contradictions()
        assert len(contradictions) >= 1
        old, new = contradictions[0]
        assert old.superseded_by != ""


# ── Search ──────────────────────────────────────────────────────


class TestSearch:
    def test_finds_relevant(self):
        store = KnowledgeStore()
        store.add_note("BM25 retrieval accuracy reached 81.2% on LOCOMO.", source="bench.md")
        store.add_note("The SNN daemon was killed.", source="daemon.md")
        results = store.search("BM25 LOCOMO accuracy", top_k=3)
        assert len(results) > 0
        assert any("bm25" in n.content.lower() for n in results)

    def test_empty_query(self):
        store = KnowledgeStore()
        assert store.search("") == []

    def test_no_match(self):
        store = KnowledgeStore()
        store.add_note("BM25 scoring works.", source="a.md")
        assert store.search("xyznonexistent_zzz") == []


# ── Graph traversal ─────────────────────────────────────────────


class TestGetRelated:
    def test_finds_linked(self):
        store = KnowledgeStore()
        n1 = store.add_note(
            "BM25 retrieval scoring improved accuracy on LOCOMO benchmark.", source="a.md"
        )
        n2 = store.add_note(
            "BM25 LOCOMO benchmark accuracy reached 83.1% with LLM synthesis.", source="b.md"
        )
        related = store.get_related(n1.id, depth=1)
        assert len(related) > 0

    def test_nonexistent_note(self):
        store = KnowledgeStore()
        assert store.get_related("nonexistent") == []

    def test_depth_2(self):
        store = KnowledgeStore()
        n1 = store.add_note("BM25 retrieval is the core scoring algorithm.", source="a.md")
        n2 = store.add_note("BM25 scoring uses TF-IDF term weighting internally.", source="b.md")
        n3 = store.add_note(
            "TF-IDF term weighting was the original retrieval method.", source="c.md"
        )
        related = store.get_related(n1.id, depth=2)
        # Should find n2 (depth 1) and potentially n3 (depth 2)
        assert len(related) >= 1


# ── Triggers ────────────────────────────────────────────────────


class TestTriggers:
    def test_add_trigger(self):
        store = KnowledgeStore()
        t = store.add_trigger("scpn-control", "Check weights file changes with Python version")
        assert t.condition == "scpn-control"
        assert len(store.triggers) == 1

    def test_check_triggers_match(self):
        store = KnowledgeStore()
        store.add_trigger("scpn-control weights", "Weights file changes with Python version")
        matched = store.check_triggers("working on scpn-control weights file")
        assert len(matched) == 1
        assert len(matched[0].fired) == 1

    def test_check_triggers_no_match(self):
        store = KnowledgeStore()
        store.add_trigger("scpn-control", "Check weights")
        matched = store.check_triggers("director-ai release")
        assert len(matched) == 0

    def test_inactive_trigger_skipped(self):
        store = KnowledgeStore()
        t = store.add_trigger("scpn-control", "Check weights")
        t.active = False
        matched = store.check_triggers("scpn-control work")
        assert len(matched) == 0


# ── Save/Load ───────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_notes(self, tmp_path):
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

    def test_save_and_load_triggers(self, tmp_path):
        store = KnowledgeStore()
        store.add_trigger("scpn-control", "Check weights")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        store2 = KnowledgeStore()
        store2.load(notes_path, triggers_path)
        assert len(store2.triggers) == 1

    def test_load_nonexistent(self, tmp_path):
        store = KnowledgeStore()
        assert store.load(tmp_path / "nope.jsonl") is False

    def test_search_after_load(self, tmp_path):
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
    def test_stats(self):
        store = KnowledgeStore()
        store.add_note("BM25 scoring works for retrieval on LOCOMO.", source="a.md")
        store.add_note("BM25 LOCOMO retrieval accuracy improved with cross-encoder.", source="b.md")
        store.add_trigger("scpn-control", "check weights")
        s = store.stats
        assert s["notes"] == 2
        assert s["links"] > 0
        assert s["triggers_total"] == 1
        assert s["triggers_active"] == 1

    def test_empty_stats(self):
        store = KnowledgeStore()
        s = store.stats
        assert s["notes"] == 0
        assert s["links"] == 0


# ── Feature 1: Prospective indexing ─────────────────────────────


class TestProspectiveQueries:
    def test_generates_queries_at_write_time(self):
        store = KnowledgeStore()
        note = store.add_note(
            "Caroline likes pottery and swimming. She started pottery in March.",
            source="conv.md",
        )
        assert len(note.prospective_queries) > 0

    def test_entity_queries_generated(self):
        pq = _generate_prospective_queries(
            "BM25 retrieval accuracy is 81.2% on LOCOMO.",
            "BM25 accuracy",
            ["bm25", "locomo", "81.2%"],
            ["bm25", "retrieval", "accuracy"],
        )
        assert any("bm25" in q.lower() for q in pq)

    def test_activity_detection(self):
        pq = _generate_prospective_queries(
            "Caroline likes pottery and enjoys swimming in the lake.",
            "Caroline activities",
            [],
            ["pottery", "swimming"],
        )
        assert any("like" in q.lower() for q in pq)

    def test_temporal_queries(self):
        pq = _generate_prospective_queries(
            "Released v3.9.0 on 2026-03-15.",
            "Release",
            ["v3.9.0"],
            ["release"],
        )
        assert any("2026-03-15" in q for q in pq)

    def test_causal_queries(self):
        pq = _generate_prospective_queries(
            "We decided to remove SNN because it adds no signal.",
            "Remove SNN",
            ["snn"],
            ["remove", "signal"],
        )
        assert any("why" in q.lower() for q in pq)

    def test_prospective_queries_searchable(self):
        store = KnowledgeStore()
        store.add_note(
            "Caroline enjoys pottery classes every Tuesday.",
            source="conv.md",
        )
        # "hobbies" is not in the note text but should be findable
        # via prospective queries that include "like" or activity terms
        note = list(store.notes.values())[0]
        assert note.searchable_text != note.content

    def test_custom_queries_passed(self):
        store = KnowledgeStore()
        note = store.add_note(
            "Test content.",
            source="test.md",
            prospective_queries=["custom query one", "custom query two"],
        )
        assert note.prospective_queries == ["custom query one", "custom query two"]

    def test_queries_in_token_index(self):
        store = KnowledgeStore()
        store.add_note(
            "We started the daemon process.",
            source="test.md",
            prospective_queries=["xyztestquery123"],
        )
        nid = list(store.notes.keys())[0]
        tokens = store._token_index[nid]
        assert "xyztestquery123" in tokens

    def test_roundtrip_preserves_queries(self, tmp_path):
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
    def test_graph_search_finds_linked_notes(self):
        store = KnowledgeStore()
        store.add_note("BM25 retrieval scoring algorithm details.", source="a.md")
        store.add_note("BM25 scoring uses TF-IDF term weighting internally.", source="b.md")
        store.add_note("TF-IDF term weighting was the original retrieval method.", source="c.md")
        results = store.graph_search("BM25 retrieval", top_k=5, hop_depth=2)
        assert len(results) >= 2

    def test_graph_search_empty(self):
        store = KnowledgeStore()
        assert store.graph_search("nonexistent") == []

    def test_excludes_superseded(self):
        store = KnowledgeStore()
        store.add_note("We enabled the GPU daemon for processing.", source="a.md")
        store.add_note("We disabled the GPU daemon to free memory.", source="b.md")
        results = store.graph_search("GPU daemon", top_k=5)
        # Superseded note should be excluded from graph results
        for r in results:
            assert r.superseded_by == ""

    def test_typed_edge_filter(self):
        store = KnowledgeStore()
        n1 = store.add_note("Base concept for BM25 retrieval algorithm.", source="a.md")
        n2 = store.add_note("BM25 scoring uses TF-IDF weighting retrieval.", source="b.md")
        # Only follow "related" edges
        related = store.get_related(n1.id, depth=1, edge_types={"related"})
        # n2 should be found via "related" type
        assert all(
            any(l.get("type") == "related" for l in store.notes[n1.id].links if l["target"] == r.id)
            for r in related
        )

    def test_add_typed_link(self):
        store = KnowledgeStore()
        n1 = store.add_note("Module A depends on module B.", source="a.md")
        n2 = store.add_note("Module B provides core functionality.", source="b.md")
        ok = store.add_typed_link(n1.id, n2.id, "depends_on")
        assert ok is True
        deps = store.get_related(n1.id, depth=1, edge_types={"depends_on"})
        assert any(d.id == n2.id for d in deps)

    def test_add_typed_link_invalid_type(self):
        store = KnowledgeStore()
        n1 = store.add_note("A.", source="a.md")
        n2 = store.add_note("B completely different content.", source="b.md")
        ok = store.add_typed_link(n1.id, n2.id, "invalid_type")
        assert ok is False

    def test_add_typed_link_nonexistent(self):
        store = KnowledgeStore()
        ok = store.add_typed_link("nope1", "nope2", "related")
        assert ok is False

    def test_search_excludes_superseded(self):
        store = KnowledgeStore()
        store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        results = store.search("SNN daemon", exclude_superseded=True)
        for r in results:
            assert r.superseded_by == ""

    def test_search_includes_superseded_when_asked(self):
        store = KnowledgeStore()
        store.add_note("We started the SNN daemon for retrieval.", source="a.md")
        store.add_note("We killed the SNN daemon because it adds nothing.", source="b.md")
        results = store.search("SNN daemon", exclude_superseded=False)
        assert len(results) >= 2


# ── Person name extraction ───────────────────────────────────────


class TestExtractPersonNames:
    def test_chat_format(self):
        names = extract_person_names("Caroline: I love pottery!\nMelanie: That's great!")
        assert "caroline" in names
        assert "melanie" in names

    def test_ignores_common_words(self):
        names = extract_person_names("The weather is great. Thanks for asking.")
        assert "the" not in names
        assert "thanks" not in names

    def test_sentence_start_names(self):
        names = extract_person_names("John went to the store. Maria called him.")
        assert "john" in names
        assert "maria" in names

    def test_empty(self):
        assert extract_person_names("no names here at all") == set()


# ── Auto entity linking ─────────────────────────────────────────


class TestAutoEntityLinking:
    def test_shared_entities_create_links(self):
        store = KnowledgeStore()
        n1 = store.add_note("BM25 retrieval accuracy is 81.2% on LOCOMO benchmark.", source="a.md")
        n2 = store.add_note("LOCOMO benchmark result: BM25 achieved 83.1% accuracy.", source="b.md")
        # Both share bm25 and locomo entities — should auto-link
        all_links = n1.links + n2.links
        assert len(all_links) >= 2  # at least the similarity links

    def test_no_link_without_shared_entities(self):
        store = KnowledgeStore()
        n1 = store.add_note("PyTorch CUDA acceleration for training models.", source="a.md")
        n2 = store.add_note("The weather in Prague was surprisingly warm today.", source="b.md")
        # No shared entities — auto entity linking shouldn't fire
        # (may still have similarity links if tokens overlap)
        n2_entity_links = [l for l in n2.links if "shared_entities" in l]
        assert len(n2_entity_links) == 0
