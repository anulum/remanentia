# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for memory_index.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from memory_index import (
    Document,
    MemoryIndex,
    Paragraph,
    SearchResult,
    _classify_paragraph,
    _classify_query,
    _entity_boost_score,
    _extract_date_context,
    _extract_query_names,
    _has_date_expression,
    _is_person_centric,
    _load_entity_graph,
    _query_entity_ids,
    _recency_boost,
    _generate_prospective_queries,
    _parse_date,
    _split_code,
    _split_paragraphs,
    _tokenize,
    auto_rebuild_if_needed,
    needs_rebuild,
)


SCRATCH_ROOT = Path(__file__).resolve().parent.parent / ".codex_scratch" / "test_memory_index"


def _scratch_case_dir(name: str) -> Path:
    path = SCRATCH_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Tokenizer ────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_filters_short(self):
        tokens = _tokenize("a to is the big cat")
        assert "a" not in tokens
        assert "to" not in tokens
        assert "is" not in tokens
        assert "big" in tokens
        assert "cat" in tokens

    def test_preserves_numbers(self):
        tokens = _tokenize("version 312 active")
        assert "version" in tokens
        assert "312" in tokens

    def test_underscores(self):
        tokens = _tokenize("compute_bm25_score")
        assert "compute_bm25_score" in tokens

    def test_empty(self):
        assert _tokenize("") == []

    def test_mixed_case(self):
        tokens = _tokenize("BM25 Score STDP")
        assert "bm25" in tokens
        assert "score" in tokens
        assert "stdp" in tokens


# ── Paragraph splitting ──────────────────────────────────────────


class TestSplitParagraphs:
    def test_markdown_split(self):
        text = "First paragraph here with enough content.\n\nSecond paragraph also has enough content."
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        assert "First" in paras[0]
        assert "Second" in paras[1]

    def test_filters_short(self):
        text = "Too short\n\nThis paragraph has enough content to pass the length filter."
        paras = _split_paragraphs(text)
        assert len(paras) == 1

    def test_single_block_fallback(self):
        text = "A" * 50  # long enough, no double newlines
        paras = _split_paragraphs(text)
        assert len(paras) == 1

    def test_empty(self):
        assert _split_paragraphs("") == []
        assert _split_paragraphs("short") == []

    def test_long_no_breaks(self):
        text = "A single long block " * 10  # 200 chars, no double newlines
        paras = _split_paragraphs(text)
        assert len(paras) == 1

    def test_many_short_blocks_fallback(self):
        # All blocks < 30 chars, but total > 30 chars
        text = "Short block.\n\nAnother short.\n\nThird.\n\nFourth block here."
        paras = _split_paragraphs(text)
        # Individual blocks are all <30 chars, so fallback to whole text
        assert len(paras) == 1
        assert "Short block" in paras[0]


class TestSplitCode:
    def test_function_extraction(self, sample_code_text):
        chunks = _split_code(sample_code_text)
        assert len(chunks) >= 2  # docstring + at least def + class
        func_chunks = [c for c in chunks if "compute_score" in c]
        assert len(func_chunks) >= 1

    def test_method_extraction(self, sample_code_text):
        chunks = _split_code(sample_code_text)
        method_chunks = [c for c in chunks if "class SearchEngine" in c and "def search" in c]
        assert len(method_chunks) >= 1

    def test_class_extraction(self, sample_code_text):
        chunks = _split_code(sample_code_text)
        class_chunks = [c for c in chunks if "SearchEngine" in c]
        assert len(class_chunks) >= 1

    def test_module_docstring(self, sample_code_text):
        chunks = _split_code(sample_code_text)
        assert any("Module docstring" in c for c in chunks)

    def test_plain_text_fallback(self):
        text = "No functions here, just plain text.\n\nAnother block of plain text content."
        chunks = _split_code(text)
        assert len(chunks) >= 1

    def test_cap_at_50(self):
        text = "\n\n".join(f"def func_{i}():\n    pass" for i in range(60))
        chunks = _split_code(text)
        assert len(chunks) <= 50


# ── Paragraph classification ─────────────────────────────────────


class TestClassifyParagraph:
    def test_code_function(self):
        assert _classify_paragraph("def compute_score():", is_code=True) == "function"

    def test_code_class(self):
        assert _classify_paragraph("class SearchEngine:", is_code=True) == "function"

    def test_code_rust_fn(self):
        assert _classify_paragraph("pub fn search(query: &str)", is_code=True) == "function"

    def test_code_generic(self):
        assert _classify_paragraph("x = 42", is_code=True) == "code"

    def test_decision(self):
        assert _classify_paragraph("We decided to remove SNN scoring") == "decision"

    def test_finding(self):
        assert _classify_paragraph("We found that BM25 outperforms TF-IDF") == "finding"

    def test_metric(self):
        assert _classify_paragraph("P@1 accuracy reached 100 percent on benchmark") == "metric"

    def test_version(self):
        assert _classify_paragraph("Shipped v3.9.0 release to PyPI") == "version"

    def test_discussion_default(self):
        assert _classify_paragraph("Some general discussion about approaches") == "discussion"


# ── Query classification ─────────────────────────────────────────


class TestClassifyQuery:
    def test_location(self):
        intent = _classify_query("where is compute_score")
        assert intent["type"] == "location"
        assert "function" in intent["boost_types"]

    def test_decision(self):
        intent = _classify_query("what did we decide about SNN")
        assert intent["type"] == "decision"

    def test_temporal(self):
        intent = _classify_query("when did we start the STDP experiment")
        assert intent["type"] == "temporal"

    def test_status_recency(self):
        intent = _classify_query("current status of remanentia")
        assert intent["type"] == "status"
        assert intent["recency"] is True

    def test_debugging(self):
        intent = _classify_query("what went wrong with the daemon")
        assert intent["type"] == "debugging"

    def test_metric(self):
        intent = _classify_query("benchmark accuracy score percent")
        assert intent["type"] == "metric"

    def test_explanation(self):
        intent = _classify_query("how does BM25 scoring work")
        assert intent["type"] == "explanation"

    def test_general_fallback(self):
        intent = _classify_query("remanentia project overview")
        assert intent["type"] == "general"


# ── Prospective query generation ─────────────────────────────────


class TestProspectiveQueries:
    def test_function_queries(self):
        text = "def compute_order_parameter(phases):\n    return np.mean(np.exp(1j * phases))"
        queries = _generate_prospective_queries(text, "solver.py", "function")
        assert any("compute_order_parameter" in q for q in queries)
        assert any("where is" in q.lower() for q in queries)

    def test_decision_queries(self):
        text = "We decided to remove SNN from the retrieval pipeline."
        queries = _generate_prospective_queries(text, "trace.md", "decision")
        assert any("why" in q.lower() or "decide" in q.lower() for q in queries)

    def test_metric_queries(self):
        text = "The benchmark scored 66.4% on LOCOMO."
        queries = _generate_prospective_queries(text, "bench.md", "metric")
        assert any("66.4%" in q for q in queries)

    def test_py_file_queries(self):
        text = "Some content about search."
        queries = _generate_prospective_queries(text, "memory_index.py", "code")
        assert any("memory_index" in q for q in queries)

    def test_cap_at_20(self):
        text = "def a():\n def b():\n def c():\n def d():\n " + "Foo Bar Baz Qux " * 20
        queries = _generate_prospective_queries(text, "big.py", "function")
        assert len(queries) <= 20


# ── Date parsing ─────────────────────────────────────────────────


class TestParseDate:
    def test_from_filename(self):
        assert _parse_date("anything", "2026-03-15_decision.md") == "2026-03-15"

    def test_from_text(self):
        assert _parse_date("Created on 2026-03-17 as a test", "trace.md") == "2026-03-17"

    def test_filename_takes_priority(self):
        assert _parse_date("2026-01-01 old date", "2026-03-15_trace.md") == "2026-03-15"

    def test_no_date(self):
        assert _parse_date("no dates here", "trace.md") == ""


# ── Recency boost ────────────────────────────────────────────────


class TestRecencyBoost:
    def test_today_high_boost(self):
        from datetime import date
        today = date.today().isoformat()
        assert _recency_boost(today) == 1.8

    def test_3_day_old_boost(self):
        from datetime import date, timedelta
        d = (date.today() - timedelta(days=3)).isoformat()
        assert _recency_boost(d) == 1.4

    def test_10_day_old_boost(self):
        from datetime import date, timedelta
        d = (date.today() - timedelta(days=10)).isoformat()
        assert _recency_boost(d) == 1.2

    def test_old_date_no_boost(self):
        assert _recency_boost("2020-01-01") == 1.0

    def test_invalid_date(self):
        assert _recency_boost("not-a-date") == 1.0

    def test_empty_string(self):
        assert _recency_boost("") == 1.0


# ── Date context extraction ──────────────────────────────────────


class TestExtractDateContext:
    def test_finds_dates(self):
        text = "This was done on 2026-03-15 as part of the sprint."
        results = _extract_date_context(text)
        assert len(results) == 1
        assert results[0][0] == "2026-03-15"
        assert "sprint" in results[0][1]

    def test_multiple_dates(self):
        text = "Started 2026-03-10, completed 2026-03-15."
        results = _extract_date_context(text)
        assert len(results) == 2

    def test_no_dates(self):
        assert _extract_date_context("no dates here") == []


# ── MemoryIndex build + search ───────────────────────────────────


class TestMemoryIndex:
    def _build_mini_index(self, tmp_path):
        """Build a small index from temp files for testing."""
        # Create some indexable files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "decision.md").write_text(
            "# SNN Removal Decision\n\n"
            "We decided to remove SNN scoring from retrieval.\n"
            "The embedding weight increased from 0.25 to 0.45.\n\n"
            "P@1 improved from 85.7% to 100% on 14 queries.\n",
            encoding="utf-8",
        )
        (docs_dir / "benchmark.md").write_text(
            "# LOCOMO Benchmark\n\n"
            "External benchmark on 1,986 questions.\n"
            "BM25-only: 48.9%. With embedding: 50.0%.\n\n"
            "Multi-hop: 75.4%. Temporal: 15.6%.\n",
            encoding="utf-8",
        )
        (docs_dir / "code.py").write_text(
            '"""Search engine module."""\n\n'
            "def compute_bm25(query, doc):\n"
            "    pass\n\n"
            "def build_index(docs):\n"
            "    pass\n",
            encoding="utf-8",
        )
        return docs_dir

    def test_build_no_gpu(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        # Patch SOURCES to only use our temp dir
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                stats = idx.build(use_gpu_embeddings=False, use_gliner=False)
        assert stats["documents"] >= 3
        assert stats["paragraphs"] > 0
        assert stats["unique_tokens"] > 0
        assert stats["has_embeddings"] is False
        assert idx._built is True

    def test_search_finds_relevant(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("SNN removal decision", top_k=3)
        assert len(results) > 0
        assert results[0].name == "decision.md"

    def test_search_benchmark_query(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("LOCOMO benchmark accuracy", top_k=3)
        assert len(results) > 0
        assert any("benchmark" in r.name.lower() for r in results)

    def test_search_code_query(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("compute_bm25", top_k=5)
        assert len(results) > 0
        # code.py should appear somewhere in results
        names = [r.name for r in results]
        assert "code.py" in names

    def test_location_query_prefers_code_over_notes(self):
        docs_dir = _scratch_case_dir("location_query_prefers_code") / "docs"
        docs_dir.mkdir()
        (docs_dir / "code.py").write_text(
            '"""Memory index module."""\n\n'
            "class MemoryIndex:\n"
            "    def search(self, query):\n"
            "        return []\n",
            encoding="utf-8",
        )
        (docs_dir / "note.md").write_text(
            "# Discussion\n\nWe discussed where MemoryIndex.search is implemented in code.py.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("where is MemoryIndex.search implemented", top_k=3)
        assert len(results) > 0
        assert results[0].name == "code.py"

    def test_search_empty_query(self, tmp_path):
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {}):
            idx.build(use_gpu_embeddings=False, use_gliner=False)
        assert idx.search("") == []

    def test_search_no_match(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("xyznonexistent_zzz_999", top_k=3)
        assert results == []

    def test_deduplicates_by_document(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("benchmark embedding", top_k=10)
        doc_names = [r.name for r in results]
        assert len(doc_names) == len(set(doc_names))

    def test_query_type_boosting(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        # Decision query should boost decision paragraphs
        results = idx.search("what did we decide about retrieval", top_k=3)
        assert len(results) > 0
        # decision.md should rank higher than benchmark.md
        if len(results) >= 2:
            names = [r.name for r in results]
            if "decision.md" in names and "benchmark.md" in names:
                assert names.index("decision.md") < names.index("benchmark.md")

    def test_save_and_load(self, tmp_path):
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        save_path = tmp_path / "test_index.pkl"
        idx.save(save_path)
        assert save_path.exists()

        idx2 = MemoryIndex()
        assert idx2.load(save_path) is True
        assert idx2._built is True
        assert len(idx2.documents) == len(idx.documents)
        assert len(idx2.paragraph_index) == len(idx.paragraph_index)

        # Search should work on loaded index
        results = idx2.search("SNN removal", top_k=3)
        assert len(results) > 0

    def test_load_nonexistent(self, tmp_path):
        idx = MemoryIndex()
        assert idx.load(tmp_path / "nope.pkl") is False

    def test_not_built_flag(self):
        """MemoryIndex starts unbuilt."""
        idx = MemoryIndex()
        assert idx._built is False
        assert idx.documents == []
        assert idx.paragraph_index == []

    def test_cross_encoder_rerank(self, tmp_path):
        """Cross-encoder reranking reorders candidates by relevance score."""
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        # Mock cross-encoder to return known scores
        class MockCrossEncoder:
            def predict(self, pairs, show_progress_bar=False):
                # Score higher for pairs where query terms appear in text
                scores = []
                for q, t in pairs:
                    score = sum(1 for w in q.lower().split() if w in t.lower())
                    scores.append(float(score))
                return scores

        idx._cross_encoder = MockCrossEncoder()
        results = idx.search("SNN removal decision embedding", top_k=3)
        assert len(results) > 0

    def test_cross_encoder_graceful_fallback(self, tmp_path):
        """Search works when cross-encoder import fails."""
        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        # Cross-encoder rerank returns None on failure
        with patch.object(idx, "_cross_encoder_rerank", return_value=None):
            results = idx.search("benchmark accuracy", top_k=3)
        assert len(results) > 0


class TestAddFile:
    def test_add_file_incremental(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "initial.md").write_text(
            "# Initial\n\nSome initial content about BM25 scoring.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        initial_count = len(idx.documents)

        new_file = docs_dir / "added.md"
        new_file.write_text(
            "# New Trace\n\nWe decided to add incremental indexing to Remanentia.",
            encoding="utf-8",
        )
        added = idx.add_file(new_file, source="test")
        assert added > 0
        assert len(idx.documents) == initial_count + 1
        results = idx.search("incremental indexing", top_k=3)
        assert any("added.md" in r.name for r in results)

    def test_add_file_not_built(self, tmp_path):
        idx = MemoryIndex()
        assert idx.add_file(tmp_path / "nope.md") == 0


class TestSearchFilters:
    def test_project_filter(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "alpha.md").write_text(
            "# Alpha\n\nAlpha project content about scoring algorithms.",
            encoding="utf-8",
        )
        (docs_dir / "beta.md").write_text(
            "# Beta\n\nBeta project content about scoring algorithms.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"alpha": docs_dir, "beta": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"alpha": {".md"}, "beta": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        # Without filter: should find both
        results = idx.search("scoring algorithms", top_k=5)
        assert len(results) >= 1

        # With project filter: only matching source
        results_alpha = idx.search("scoring algorithms", top_k=5, project="alpha")
        for r in results_alpha:
            assert "alpha" in r.source.lower() or "alpha" in r.name.lower()

    def test_date_filter(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "2026-03-10_old.md").write_text(
            "# Old Trace\n\nOld content about retrieval experiments.",
            encoding="utf-8",
        )
        (docs_dir / "2026-03-20_new.md").write_text(
            "# New Trace\n\nNew content about retrieval improvements.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        results = idx.search("retrieval", top_k=5, after="2026-03-15")
        names = [r.name for r in results]
        assert "2026-03-20_new.md" in names
        assert "2026-03-10_old.md" not in names


class TestNeedsRebuild:
    def test_no_index_needs_rebuild(self, tmp_path):
        with patch("memory_index.INDEX_PATH", tmp_path / "nope.pkl"):
            assert needs_rebuild() is True

    def test_fresh_index_no_rebuild(self, tmp_path):
        idx_path = tmp_path / "index.pkl"
        idx_path.write_bytes(b"data")

        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "old.md").write_text("old content", encoding="utf-8")
        # Touch the index to be newer
        import os
        os.utime(idx_path, None)

        with patch("memory_index.INDEX_PATH", idx_path):
            with patch("memory_index.SOURCES", {"test": source_dir}):
                assert needs_rebuild() is False

    def test_newer_python_file_needs_rebuild(self):
        case_dir = _scratch_case_dir("needs_rebuild_python_file")
        idx_path = case_dir / "index.pkl"
        idx_path.write_bytes(b"data")

        source_dir = case_dir / "src"
        source_dir.mkdir()
        py_file = source_dir / "memory_index.py"
        py_file.write_text("def search(query):\n    return []\n", encoding="utf-8")

        import os
        old_time = idx_path.stat().st_mtime - 100
        os.utime(idx_path, (old_time, old_time))

        with patch("memory_index.INDEX_PATH", idx_path):
            with patch("memory_index.SOURCES", {"test": source_dir}):
                with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".py"}}):
                    assert needs_rebuild() is True


class TestRustBm25Gate:
    def test_rust_bm25_disabled_below_threshold(self):
        idx = MemoryIndex()
        idx.paragraph_index = [(0, 0)] * 100
        assert idx._should_use_rust_bm25() is False

    def test_rust_bm25_env_override(self, monkeypatch):
        idx = MemoryIndex()
        idx.paragraph_index = [(0, 0)] * 10

        monkeypatch.setenv("REMANENTIA_USE_RUST_BM25", "1")
        assert idx._should_use_rust_bm25() is True

        monkeypatch.setenv("REMANENTIA_USE_RUST_BM25", "0")
        assert idx._should_use_rust_bm25() is False


# ── Dataclass sanity ─────────────────────────────────────────────


class TestDataclasses:
    def test_document_defaults(self):
        d = Document(name="test.md", source="traces", path="/tmp/test.md")
        assert d.paragraphs == []
        assert d.tokens == set()
        assert d.embedding is None
        assert d.date == ""

    def test_search_result(self):
        r = SearchResult(name="test.md", source="traces", score=0.95, snippet="content")
        assert r.score == 0.95
        assert r.paragraph_idx == 0

    def test_paragraph(self):
        p = Paragraph(text="some text", para_type="decision")
        assert p.prospective_queries == []


# ── _has_date_expression ────────────────────────────────────────


class TestHasDateExpression:
    def test_iso_date(self):
        assert _has_date_expression("Fixed on 2026-03-15.") is True

    def test_english_date(self):
        assert _has_date_expression("Released in January 15.") is True

    def test_relative_date(self):
        assert _has_date_expression("Done yesterday.") is True

    def test_no_date(self):
        assert _has_date_expression("No dates here.") is False


# ── Save/load with embeddings ───────────────────────────────────


class TestSaveLoadEmbeddings:
    def _build_idx(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# Test\n\nContent about BM25 scoring.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        return idx

    def test_save_load_no_embeddings(self, tmp_path):
        idx = self._build_idx(tmp_path)
        path = tmp_path / "idx.pkl"
        idx.save(path, quantize=False)
        idx2 = MemoryIndex()
        assert idx2.load(path) is True
        assert idx2._built is True

    def test_save_quantized_with_embeddings(self, tmp_path):
        idx = self._build_idx(tmp_path)
        idx.embeddings = np.random.randn(len(idx.paragraph_index), 384).astype(np.float32)
        path = tmp_path / "idx_q.pkl"
        idx.save(path, quantize=True)
        idx2 = MemoryIndex()
        assert idx2.load(path) is True
        assert idx2.embeddings is not None
        assert idx2.embeddings.shape == idx.embeddings.shape

    def test_save_unquantized_with_embeddings(self, tmp_path):
        idx = self._build_idx(tmp_path)
        idx.embeddings = np.random.randn(len(idx.paragraph_index), 384).astype(np.float32)
        path = tmp_path / "idx_nq.pkl"
        idx.save(path, quantize=False)
        idx2 = MemoryIndex()
        assert idx2.load(path) is True
        assert idx2.embeddings is not None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a pickle")
        idx = MemoryIndex()
        assert idx.load(path) is False

    def test_load_legacy_4tuple_format(self, tmp_path):
        """Load index saved with old 4-tuple document format."""
        import pickle
        data = {
            "documents": [("test.md", "src", "/test.md", ["para1"])],
            "paragraph_index": [(0, 0)],
            "paragraph_tokens": [["test", "para"]],
            "idf": {"test": 1.0, "para": 1.0},
            "embeddings": None,
        }
        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        idx = MemoryIndex()
        assert idx.load(path) is True
        assert idx.documents[0].name == "test.md"
        assert idx.documents[0].date == ""


# ── Temporal query sorting ──────────────────────────────────────


class TestTemporalSorting:
    def test_newest_first(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "2026-03-10_old.md").write_text(
            "# Old\n\nWhen did we start in 2026-03-10 on the project.", encoding="utf-8")
        (docs_dir / "2026-03-20_new.md").write_text(
            "# New\n\nWhen was the latest update on 2026-03-20.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        results = idx.search("when was the latest update", top_k=5)
        assert len(results) > 0

    def test_oldest_first(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "2026-03-10_old.md").write_text(
            "# Old\n\nThe first experiment was on 2026-03-10 with STDP.", encoding="utf-8")
        (docs_dir / "2026-03-20_new.md").write_text(
            "# New\n\nThe first thing we tried with STDP was this.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        results = idx.search("first experiment STDP", top_k=5)
        assert len(results) > 0


# ── Search with use_llm flag ───────────────────────────────────


class TestSearchLLM:
    def test_use_llm_false_works(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Testing Suite\n\nThis is a comprehensive testing document about various testing approaches and testing strategies.",
            encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        results = idx.search("comprehensive testing approaches", top_k=3, use_llm=False)
        assert len(results) > 0

    def test_use_llm_true_no_api_key(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Testing Suite\n\nThis is a comprehensive testing document about various testing approaches and testing strategies.",
            encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        results = idx.search("comprehensive testing approaches", top_k=3, use_llm=True)
        assert len(results) > 0


# ── auto_rebuild_if_needed ──────────────────────────────────────


class TestAutoRebuild:
    def test_rebuild_when_no_index(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nEnough content for indexing here.", encoding="utf-8")
        idx_path = tmp_path / "missing_index.pkl"
        with patch("memory_index.INDEX_PATH", idx_path), \
             patch("memory_index.SOURCES", {"test": docs_dir}), \
             patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
            idx = auto_rebuild_if_needed(use_gpu=False)
        assert idx._built is True

    def test_no_rebuild_when_fresh(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nEnough content here for index.", encoding="utf-8")
        idx_path = tmp_path / "fresh_index.pkl"
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}), \
             patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
            idx.build(use_gpu_embeddings=False, use_gliner=False)
        idx.save(idx_path)
        import os
        os.utime(idx_path, None)
        with patch("memory_index.INDEX_PATH", idx_path), \
             patch("memory_index.SOURCES", {"test": docs_dir}):
            idx2 = auto_rebuild_if_needed(use_gpu=False)
        assert idx2._built is True


# ── Cross-encoder rerank edge cases ─────────────────────────────


class TestCrossEncoderEdge:
    def test_empty_candidates(self, tmp_path):
        idx = MemoryIndex()
        result = idx._cross_encoder_rerank("query", [])
        assert result is None

    def test_cross_encoder_disabled(self, tmp_path):
        idx = MemoryIndex()
        idx._cross_encoder = False
        result = idx._cross_encoder_rerank("query", [(0, 1.0)])
        assert result is None


# ── Prospective queries for findings ────────────────────────────


class TestProspectiveQueriesFindings:
    def test_finding_queries(self):
        text = "We found that the approach failed miserably."
        queries = _generate_prospective_queries(text, "results.md", "finding")
        assert any("find" in q.lower() or "results" in q.lower() for q in queries)


# ── Build with unreadable files ─────────────────────────────────


class TestBuildEdgeCases:
    def test_skips_short_files(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "short.md").write_text("hi", encoding="utf-8")
        (docs_dir / "long.md").write_text(
            "# Test Document\n\nThis is long enough content for the indexing pipeline to accept it properly.",
            encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                stats = idx.build(use_gpu_embeddings=False, use_gliner=False)
        # short.md (2 chars) is below 50-char minimum, only long.md indexed
        assert stats["documents"] == 1

    def test_skips_venv_files(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        venv = docs_dir / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "module.py").write_text("# long enough content for index builder", encoding="utf-8")
        (docs_dir / "real.md").write_text("# Test\n\nReal content for indexing purposes.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                stats = idx.build(use_gpu_embeddings=False, use_gliner=False)
        names = [d.name for d in idx.documents]
        assert "module.py" not in names

    def test_search_triggers_build(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nContent about automated build testing.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}), \
             patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
            results = idx.search("automated build", top_k=3)
        assert idx._built is True

    def test_add_file_short_content(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# Test\n\nSome content long enough for build.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        short = tmp_path / "short.md"
        short.write_text("hi", encoding="utf-8")
        assert idx.add_file(short) == 0

    def test_add_file_nonexistent(self, tmp_path):
        idx = MemoryIndex()
        idx._built = True
        assert idx.add_file(tmp_path / "nope.md") == 0


# ── Search doc_type filter ──────────────────────────────────────


class TestDocTypeFilter:
    def test_doc_type_filter(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "code.py").write_text(
            '"""Module doc."""\n\ndef search_function():\n    pass\n', encoding="utf-8")
        (docs_dir / "note.md").write_text(
            "# Note\n\nThis is a note about search functions and algorithms.", encoding="utf-8")
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"code_test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"code_test": {".py", ".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        results = idx.search("search function", top_k=5, doc_type="code")
        for r in results:
            assert "code" in r.source.lower() or r.name.endswith(".py")


# ── Person-centric helpers ────────────────────────────────────────


class TestExtractQueryNames:
    def test_extracts_capitalized_names(self):
        names = _extract_query_names("What did Alice say to Bob?")
        assert "alice" in names
        assert "bob" in names

    def test_filters_common_words(self):
        names = _extract_query_names("What does Alice have?")
        assert "alice" in names
        assert "what" not in names
        assert "does" not in names

    def test_empty_query(self):
        assert _extract_query_names("no capitalized words here") == set()

    def test_short_names_filtered(self):
        names = _extract_query_names("Is Al okay?")
        assert "al" not in names


class TestIsPersonCentric:
    def test_possessive(self):
        # "her activities" hits _POSSESSIVE_RE but not _PERSON_CENTRIC_RE
        assert _is_person_centric("What are her activities?")

    def test_relationship_keyword(self):
        assert _is_person_centric("What is Alice's relationship with Bob?")

    def test_would_likely(self):
        assert _is_person_centric("what would alice likely do?")

    def test_non_person_query(self):
        assert not _is_person_centric("what is the capital of france?")


# ── Entity graph helpers ──────────────────────────────────────────


class TestEntityGraphHelpers:
    def test_query_entity_ids(self):
        graph = {
            "entities": {
                "e1": {"label": "STDP"},
                "e2": {"label": "memory_index"},
            },
            "relations": [],
        }
        matched = _query_entity_ids("What about STDP?", graph)
        assert "e1" in matched

    def test_entity_boost_score_no_entities(self):
        assert _entity_boost_score("some text", set(), {}) == 0.0

    def test_entity_boost_score_with_match(self):
        graph = {
            "entities": {
                "e1": {"label": "stdp"},
                "e2": {"label": "bm25"},
            },
        }
        score = _entity_boost_score("STDP removal was necessary", {"e1", "e2"}, graph)
        assert score > 0.0

    def test_entity_boost_score_no_match(self):
        graph = {
            "entities": {
                "e1": {"label": "quantum"},
            },
        }
        score = _entity_boost_score("nothing related here", {"e1"}, graph)
        assert score == 0.0

    def test_load_entity_graph_missing_files(self, tmp_path):
        import memory_index
        original = memory_index.GRAPH_DIR
        memory_index.GRAPH_DIR = tmp_path / "nonexistent"
        memory_index._ENTITY_GRAPH = None
        try:
            g = _load_entity_graph()
            assert g["entities"] == {}
            assert g["relations"] == []
        finally:
            memory_index.GRAPH_DIR = original
            memory_index._ENTITY_GRAPH = None

    def test_load_entity_graph_with_data(self, tmp_path):
        import memory_index
        original = memory_index.GRAPH_DIR
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "entities.jsonl").write_text(
            json.dumps({"id": "e1", "type": "concept", "label": "test"}) + "\n",
            encoding="utf-8")
        (graph_dir / "relations.jsonl").write_text(
            json.dumps({"source": "e1", "target": "e2", "weight": 1.0}) + "\n",
            encoding="utf-8")
        memory_index.GRAPH_DIR = graph_dir
        memory_index._ENTITY_GRAPH = None
        try:
            g = _load_entity_graph()
            assert "e1" in g["entities"]
            assert len(g["relations"]) == 1
        finally:
            memory_index.GRAPH_DIR = original
            memory_index._ENTITY_GRAPH = None


# ── Person-name boosting in search ────────────────────────────────


class TestPersonNameBoostInSearch:
    def test_person_centric_query_boosts(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "alice_trace.md").write_text(
            "# Alice's Piano Hobby\n\nAlice has been learning piano since January.\n"
            "She practices every day and loves Chopin.\n"
            "Alice considers piano her favorite hobby.", encoding="utf-8")
        (traces / "weather_trace.md").write_text(
            "# Weather Report\n\nThe temperature today is 22 degrees.\n"
            "Piano lessons are cancelled due to rain.\n"
            "No specific person mentioned here at all.", encoding="utf-8")

        import memory_index
        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # "Alice's hobby" triggers _PERSON_CENTRIC_RE + name extraction
            results = idx.search("What is Alice's hobby?", top_k=5)
            assert len(results) > 0
            assert any("alice" in r.snippet.lower() for r in results)
        finally:
            memory_index.SOURCES = original_sources


# ── IDF zero edge case ────────────────────────────────────────────


class TestIdfZeroEdge:
    def test_zero_idf_token_skipped(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "doc.md").write_text(
            "# Alpha\n\nAlpha beta gamma delta.", encoding="utf-8")

        import memory_index
        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # Force a known token to have IDF=0
            for token in list(idx.idf.keys()):
                if token in idx._inverted_index:
                    idx.idf[token] = 0
                    break
            results = idx.search("alpha beta", top_k=1)
            assert isinstance(results, list)
        finally:
            memory_index.SOURCES = original_sources


class TestTemporalCodeExecution:
    def test_temporal_query_injects_answer(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "meeting.md").write_text(
            "# Meeting Notes\n\n"
            "The project started on 2026-01-15 and the deadline is 2026-06-30.\n"
            "We need to finish the review by 2026-03-01.", encoding="utf-8")

        import memory_index
        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # Temporal query — should attempt code execution
            results = idx.search("when did the project start?", top_k=3)
            assert len(results) > 0
        finally:
            memory_index.SOURCES = original_sources
