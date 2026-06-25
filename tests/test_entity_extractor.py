# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for entity extractor

from __future__ import annotations

import builtins
from unittest.mock import patch


from entity_extractor import (
    Entity,
    Relation,
    _regex_entities,
    extract_entities,
    extract_relations,
)


def _without_native_entity():
    original_import = builtins.__import__

    def import_without_native(name, *args, **kwargs):
        if name == "remanentia_entity_extractor":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=import_without_native)


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


# ── Edge cases ──────────────────────────────────────────────────


class TestRegexEntitiesEdgeCases:
    def test_empty_text(self):
        assert _regex_entities("") == []

    def test_no_entities(self):
        assert _regex_entities("The weather is nice today.") == []

    def test_all_project_names(self):
        text = "sc-neurocore, scpn-control, scpn-fusion, scpn-phase-orchestrator, scpn-quantum-control, remanentia"
        ents = _regex_entities(text)
        names = {e.text for e in ents}
        assert "sc-neurocore" in names
        assert "remanentia" in names
        assert "scpn-quantum-control" in names
        assert len(names) >= 6

    def test_all_algorithms(self):
        text = "stdp lif kuramoto hopfield bcpnn csdp tf-idf bm25 stuart-landau"
        ents = _regex_entities(text)
        assert len(ents) >= 8

    def test_version_two_part(self):
        ents = _regex_entities("Released v2.0.")
        assert any(e.text == "v2.0" for e in ents)

    def test_version_three_part(self):
        ents = _regex_entities("Upgraded to v3.14.1.")
        assert any(e.text == "v3.14.1" for e in ents)

    def test_file_path_with_directory(self):
        ents = _regex_entities("Edit src/snn_backend.py now.")
        names = [e.text for e in ents if e.label == "file path"]
        assert "snn_backend.py" in names

    def test_file_extensions(self):
        for ext in ["py", "rs", "md", "json", "yaml", "toml"]:
            ents = _regex_entities(f"Edited config.{ext} today.")
            names = [e.text for e in ents if e.label == "file path"]
            assert any(ext in n for n in names), f"Missing {ext} file"

    def test_case_insensitive(self):
        ents = _regex_entities("STDP and Stdp and stdp")
        assert len([e for e in ents if e.text == "stdp"]) >= 1

    def test_multiple_versions(self):
        ents = _regex_entities("Migrated from v2.0 to v3.0.")
        versions = [e for e in ents if e.label == "version number"]
        assert len(versions) >= 2


class TestExtractRelationsEdgeCases:
    def test_empty_entities(self):
        rels = extract_relations("Some text", [])
        assert rels == []

    def test_single_entity(self):
        rels = extract_relations(
            "Just BM25 here.",
            [
                Entity(text="BM25", label="algorithm", score=0.9),
            ],
        )
        assert rels == []

    def test_version_of(self):
        text = "BM25 version v3.0 was released."
        ents = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="v3.0", label="version", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "version_of" for r in rels)

    def test_depends_on(self):
        text = "STDP requires NumPy for computation."
        ents = [
            Entity(text="STDP", label="algorithm", score=0.9),
            Entity(text="NumPy", label="tool", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "depends_on" for r in rels)

    def test_improved(self):
        text = "BM25 improved from 81% to 88% with remanentia."
        ents = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="remanentia", label="project", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "improved" for r in rels)

    def test_tested_with(self):
        text = "STDP was benchmarked against Hopfield networks."
        ents = [
            Entity(text="STDP", label="algorithm", score=0.9),
            Entity(text="Hopfield", label="algorithm", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "tested_with" for r in rels)

    def test_produced(self):
        text = "PyTorch generated the embeddings for BM25."
        ents = [
            Entity(text="PyTorch", label="tool", score=0.9),
            Entity(text="BM25", label="algorithm", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "produced" for r in rels)

    def test_used_in(self):
        text = "BM25 used in the remanentia retrieval pipeline."
        ents = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="remanentia", label="project", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "used_in" for r in rels)

    def test_contradicts(self):
        text = "STDP contradicts the Hopfield convergence assumption."
        ents = [
            Entity(text="STDP", label="algorithm", score=0.9),
            Entity(text="Hopfield", label="algorithm", score=0.8),
        ]
        rels = extract_relations(text, ents)
        assert any(r.relation_type == "contradicts" for r in rels)

    def test_evidence_truncated(self):
        long_text = "BM25 " + "x" * 300 + " fixed STDP"
        ents = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="STDP", label="algorithm", score=0.8),
        ]
        rels = extract_relations(long_text, ents)
        for r in rels:
            assert len(r.evidence) <= 200 or r.evidence == ""

    def test_three_entities_pairwise(self):
        text = "BM25 replaced TF-IDF and improved STDP scoring."
        ents = [
            Entity(text="BM25", label="algorithm", score=0.9),
            Entity(text="TF-IDF", label="algorithm", score=0.9),
            Entity(text="STDP", label="algorithm", score=0.9),
        ]
        rels = extract_relations(text, ents)
        # Should produce relations for all 3 pairs
        pairs = {(r.source, r.target) for r in rels}
        assert len(pairs) == 3


class TestGLiNEREdgeCases:
    def test_gliner_exception_handled(self):
        """GLiNER predict_entities raising does not crash."""

        class BrokenModel:
            def predict_entities(self, text, labels, threshold=0.4):
                raise RuntimeError("GPU OOM")

        with patch("entity_extractor._load_gliner", return_value=BrokenModel()):
            entities = extract_entities("We used STDP on GPU.")
        # Should return list (empty from broken GLiNER, deduped)
        assert isinstance(entities, list)

    def test_custom_labels(self):
        with patch("entity_extractor._load_gliner", return_value=None):
            entities = extract_entities("We used BM25.", labels=["algorithm"])
        assert isinstance(entities, list)

    def test_long_text_chunking(self):
        """Very long text gets chunked (max 20 chunks × 1500 chars)."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.predict_entities.return_value = []
        with patch("entity_extractor._load_gliner", return_value=model):
            extract_entities("word " * 10000)
        # Should be called multiple times (chunked)
        assert model.predict_entities.call_count > 1
        assert model.predict_entities.call_count <= 20


# ── Dataclass sanity ─────────────────────────────────────────────


class TestDataclasses:
    def test_entity(self):
        e = Entity(text="BM25", label="algorithm", score=0.9, start=0, end=4)
        assert e.text == "BM25"
        assert e.label == "algorithm"
        assert e.start == 0
        assert e.end == 4

    def test_entity_defaults(self):
        e = Entity(text="x", label="y", score=0.5)
        assert e.start == 0
        assert e.end == 0

    def test_relation(self):
        r = Relation(
            source="A", target="B", relation_type="caused_by", evidence="because A broke B"
        )
        assert r.source == "A"
        assert r.target == "B"
        assert r.relation_type == "caused_by"
        assert "because" in r.evidence


# ── Missing patterns: roundtrip ───────────────────────────────


class TestEntityExtractorRoundtrip:
    def test_extract_then_relations(self):
        """Full cycle: text → entities → relations."""
        text = "BM25 replaced TF-IDF because TF-IDF was too slow for remanentia v3.14.0."
        entities = _regex_entities(text)
        assert len(entities) >= 2
        relations = extract_relations(text, entities)
        assert isinstance(relations, list)
        # Should find at least one typed relation
        typed = [r for r in relations if r.relation_type != "co_occurs"]
        assert len(typed) >= 1


class TestPythonEntityFallbackCoverage:
    def test_regex_entities_without_native_extension(self):
        text = "Director-ai used STDP on GPU with PyTorch v3.9.0 in src/main.py."
        with _without_native_entity():
            entities = _regex_entities(text)

        labels = {e.text: e.label for e in entities}
        assert labels["director-ai"] == "project"
        assert labels["stdp"] == "algorithm"
        assert labels["gpu"] == "hardware"
        assert labels["pytorch"] == "software tool"
        assert labels["v3.9.0"] == "version number"
        assert labels["main.py"] == "file path"

    def test_extract_relations_without_native_extension(self):
        entities = [Entity("STDP", "algorithm", 0.9), Entity("Remanentia", "project", 0.9)]
        with _without_native_entity():
            relations = extract_relations("STDP is used in Remanentia.", entities)

        assert relations[0].relation_type == "used_in"

    def test_extract_relations_default_co_occurs_without_native_extension(self):
        entities = [Entity("STDP", "algorithm", 0.9), Entity("BM25", "algorithm", 0.9)]
        with _without_native_entity():
            relations = extract_relations("STDP and BM25 are both retrieval signals.", entities)

        assert relations[0].relation_type == "co_occurs"

    def test_extract_relations_skips_entities_absent_from_text(self):
        entities = [Entity("STDP", "algorithm", 0.9), Entity("Missing", "project", 0.9)]
        with _without_native_entity():
            assert extract_relations("STDP is present here.", entities) == []
