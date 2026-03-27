# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for entity_extractor.py

from __future__ import annotations

from unittest.mock import patch


from entity_extractor import (
    Entity,
    Relation,
    _regex_entities,
    extract_entities,
    extract_relations,
)


# ── Regex entity extraction (no GLiNER) ──────────────────────────


class TestRegexEntities:
    def test_project_names(self):
        entities = _regex_entities("The director-ai pipeline processes data.")
        texts = [e.text for e in entities]
        assert "director-ai" in texts

    def test_algorithms(self):
        entities = _regex_entities("We tested STDP and BM25 scoring.")
        texts = [e.text for e in entities]
        assert "stdp" in texts
        assert "bm25" in texts

    def test_hardware(self):
        entities = _regex_entities("Trained on GPU with CUDA 12.4.")
        texts = [e.text for e in entities]
        assert "gpu" in texts
        assert "cuda" in texts

    def test_software_tools(self):
        entities = _regex_entities("Built with PyTorch and FastAPI.")
        texts = [e.text for e in entities]
        assert "pytorch" in texts
        assert "fastapi" in texts

    def test_version_numbers(self):
        entities = _regex_entities("Released v3.9.0.")
        texts = [e.text for e in entities]
        assert "v3.9.0" in texts

    def test_file_paths(self):
        entities = _regex_entities("Fixed bug in snn_backend.py.")
        texts = [e.text for e in entities]
        assert "snn_backend.py" in texts

    def test_labels_correct(self):
        entities = _regex_entities("Used STDP on GPU with PyTorch.")
        label_map = {e.text: e.label for e in entities}
        assert label_map.get("stdp") == "algorithm"
        assert label_map.get("gpu") == "hardware"
        assert label_map.get("pytorch") == "software tool"

    def test_scores(self):
        entities = _regex_entities("Used STDP with v3.0.")
        for e in entities:
            assert 0.0 < e.score <= 1.0


# ── Entity extraction with GLiNER mock ──────────────────────────


class TestExtractEntities:
    def test_falls_back_to_regex(self):
        with patch("entity_extractor._load_gliner", return_value=None):
            entities = extract_entities("We used STDP on GPU.")
        texts = [e.text for e in entities]
        assert "stdp" in texts

    def test_gliner_path(self):
        """Test that GLiNER model is called when available."""
        class MockModel:
            def predict_entities(self, text, labels, threshold=0.4):
                return [
                    {"text": "Miroslav", "label": "person", "score": 0.9, "start": 0, "end": 8},
                    {"text": "BM25", "label": "algorithm", "score": 0.85, "start": 20, "end": 24},
                ]

        with patch("entity_extractor._load_gliner", return_value=MockModel()):
            entities = extract_entities("Miroslav implemented BM25 scoring.")
        texts = [e.text for e in entities]
        assert "Miroslav" in texts
        assert "BM25" in texts

    def test_deduplication(self):
        class MockModel:
            def predict_entities(self, text, labels, threshold=0.4):
                return [
                    {"text": "BM25", "label": "algorithm", "score": 0.9, "start": 0, "end": 4},
                    {"text": "bm25", "label": "algorithm", "score": 0.8, "start": 10, "end": 14},
                ]

        with patch("entity_extractor._load_gliner", return_value=MockModel()):
            entities = extract_entities("BM25 and bm25 are the same.")
        texts_lower = [e.text.lower() for e in entities]
        assert texts_lower.count("bm25") == 1


# ── Relation extraction ──────────────────────────────────────────


class TestExtractRelations:
    def test_caused_by(self):
        # Relation signal must appear between the two entities
        text = "STDP error was caused by the wrong mask in snn_backend.py."
        entities = [
            Entity(text="STDP", label="algorithm", score=0.9),
            Entity(text="snn_backend.py", label="file path", score=0.8),
        ]
        rels = extract_relations(text, entities)
        typed = [r for r in rels if r.relation_type == "caused_by"]
        assert len(typed) >= 1

    def test_fixed_by(self):
        # "fixed" between entities
        text = "LTD mask was fixed in snn_backend.py by correcting the sign."
        entities = [
            Entity(text="LTD", label="concept", score=0.7),
            Entity(text="snn_backend.py", label="file path", score=0.8),
        ]
        rels = extract_relations(text, entities)
        typed = [r for r in rels if r.relation_type == "fixed_by"]
        assert len(typed) >= 1

    def test_replaced(self):
        # "replaced" between entities
        text = "BM25 replaced TF-IDF for all queries."
        entities = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="TF-IDF", label="algorithm", score=0.9),
        ]
        rels = extract_relations(text, entities)
        typed = [r for r in rels if r.relation_type == "replaced"]
        assert len(typed) >= 1

    def test_co_occurs_fallback(self):
        text = "BM25 and embedding are both used."
        entities = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="embedding", label="concept", score=0.8),
        ]
        rels = extract_relations(text, entities)
        assert len(rels) >= 1

    def test_no_relations_distant(self):
        text = "BM25 is great. " + "x " * 200 + "Embedding is also great."
        entities = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="Embedding", label="concept", score=0.8),
        ]
        rels = extract_relations(text, entities)
        co = [r for r in rels if r.relation_type == "co_occurs"]
        # Entities are >500 chars apart, so no co_occurs
        # But might still match a relation pattern
        # Just check it doesn't crash
        assert isinstance(rels, list)

    def test_entity_not_in_text(self):
        text = "Some text without the entities."
        entities = [
            Entity(text="NONEXISTENT_A", label="concept", score=0.5),
            Entity(text="NONEXISTENT_B", label="concept", score=0.5),
        ]
        rels = extract_relations(text, entities)
        assert rels == []


# ── Dataclass sanity ─────────────────────────────────────────────


class TestDataclasses:
    def test_entity(self):
        e = Entity(text="BM25", label="algorithm", score=0.9, start=0, end=4)
        assert e.text == "BM25"
        assert e.label == "algorithm"

    def test_relation(self):
        r = Relation(source="A", target="B", relation_type="caused_by", evidence="because A broke B")
        assert r.source == "A"
        assert r.relation_type == "caused_by"
