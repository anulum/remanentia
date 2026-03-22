# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for memory_index.py

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from memory_index import (
    Document,
    MemoryIndex,
    Paragraph,
    SearchResult,
    _classify_paragraph,
    _classify_query,
    _extract_date_context,
    _recency_boost,
    _generate_prospective_queries,
    _parse_date,
    _split_code,
    _split_paragraphs,
    _tokenize,
    needs_rebuild,
)


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


class TestSplitCode:
    def test_function_extraction(self, sample_code_text):
        chunks = _split_code(sample_code_text)
        assert len(chunks) >= 2  # docstring + at least def + class
        func_chunks = [c for c in chunks if "compute_score" in c]
        assert len(func_chunks) >= 1

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

    def test_cap_at_10(self):
        text = "def a():\n def b():\n def c():\n def d():\n " + "Foo Bar Baz Qux " * 20
        queries = _generate_prospective_queries(text, "big.py", "function")
        assert len(queries) <= 10


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
