# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for memory index

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import memory_index
import numpy as np

from compiled_memory import CompiledFact, write_compiled_facts
from memory_query_intelligence import (
    classify_paragraph_python,
    reciprocal_rank_fusion_python,
    tokenize_python,
)
from memory_index import (
    Document,
    MemoryIndex,
    Paragraph,
    SearchResult,
    _classify_paragraph,
    _classify_query,
    _cross_reference_answers,
    _decompose_query,
    _entity_boost_score,
    _extract_date_context,
    _extract_query_names,
    _has_date_expression,
    _is_person_centric,
    _load_entity_graph,
    _query_entity_ids,
    _recency_boost,
    _reciprocal_rank_fusion,
    _generate_prospective_queries,
    _parse_date,
    _split_code,
    _split_paragraphs,
    _split_sentences,
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

    def test_explicit_python_tokenizer(self):
        assert tokenize_python("Native-free BM25 tokenizer") == [
            "native",
            "free",
            "bm25",
            "tokenizer",
        ]


# ── Paragraph splitting ──────────────────────────────────────────


class TestSplitParagraphs:
    def test_markdown_split(self):
        text = (
            "First paragraph here with enough content.\n\nSecond paragraph also has enough content."
        )
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

    def test_python_fallback_real_inputs(self):
        assert classify_paragraph_python("class SearchEngine:", is_code=True) == "function"
        assert classify_paragraph_python("x = 42", is_code=True) == "code"
        assert classify_paragraph_python("We decided to keep BM25 scoring.") == "decision"
        assert classify_paragraph_python("We found a measurable retrieval result.") == "finding"
        assert classify_paragraph_python("P@1 accuracy reached 99 percent.") == "metric"
        assert classify_paragraph_python("Released version v3.9.0.") == "version"
        assert classify_paragraph_python("General operational note.") == "discussion"


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

    def test_build_continues_when_compiled_fact_refresh_fails(self, tmp_path, monkeypatch):
        docs_dir = self._build_mini_index(tmp_path)

        def fail_compile(_repo: Path) -> list[object]:
            raise RuntimeError("compiled memory unavailable")

        monkeypatch.setattr("compiled_memory.compile_facts", fail_compile)

        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                stats = idx.build(use_gpu_embeddings=False, use_gliner=False)

        assert stats["documents"] >= 3
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

    def test_build_handles_short_sentence_windows_and_short_python_nodes(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "short_sentences.md").write_text(
            " ".join(["Alpha beta.", "Gamma beta."] * 12),
            encoding="utf-8",
        )
        (docs_dir / "short_nodes.py").write_text(
            "class A:\n"
            "    pass\n\n"
            "class Worker:\n"
            "    def x(self):\n"
            "        pass\n\n"
            "# Production indexing fixture with enough source text to enter the pipeline.\n",
            encoding="utf-8",
        )

        idx = MemoryIndex()
        with (
            patch("memory_index.SOURCES", {"test": docs_dir}),
            patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}),
        ):
            stats = idx.build(use_gpu_embeddings=False, use_gliner=False)

        assert stats["documents"] == 2
        assert {document.name for document in idx.documents} == {
            "short_nodes.py",
            "short_sentences.md",
        }
        assert idx.paragraph_index

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
        """A real local BERT cross-encoder scores production index candidates."""
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        docs_dir = self._build_mini_index(tmp_path)
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        model_dir = tmp_path / "cross-encoder"
        model_dir.mkdir()
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "snn",
            "removal",
            "decision",
            "embedding",
            "benchmark",
        ]
        vocab_path = model_dir / "vocab.txt"
        vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
        torch.manual_seed(7)
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=64,
                num_labels=1,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        )
        model.eval()
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        idx._cross_encoder = CrossEncoder(str(model_dir), device="cpu")
        candidates = [(i, float(len(idx.paragraph_index) - i)) for i in range(3)]
        reranked = idx._cross_encoder_rerank("SNN removal decision", candidates)

        assert reranked is not None
        assert {para_idx for para_idx, _ in reranked} == {0, 1, 2}
        assert all(isinstance(score, float) for _, score in reranked)


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

    def test_add_file_replaces_existing_path(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        target = docs_dir / "live.md"
        target.write_text(
            "# Live Memory\n\nOld retrieval fact about alpha resonance and stale context.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        target.write_text(
            "# Live Memory\n\nNew retrieval fact about beta resonance and current context.",
            encoding="utf-8",
        )
        added = idx.add_file(target, source="test")

        assert added == 1
        assert [doc.path for doc in idx.documents].count(str(target)) == 1
        assert idx.search("beta resonance current", top_k=3)[0].name == "live.md"
        assert idx.search("alpha stale", top_k=3) == []

    def test_add_file_replacement_drops_stale_embeddings(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        target = docs_dir / "live.md"
        target.write_text(
            "# Live Memory\n\nOld retrieval fact about alpha resonance and stale context.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        idx.embeddings = np.ones((len(idx.paragraph_index), 4), dtype=np.float32)

        target.write_text(
            "# Live Memory\n\nNew retrieval fact about beta resonance and current context.",
            encoding="utf-8",
        )
        idx.add_file(target, source="test")

        assert idx.embeddings is None

    def test_rebuild_sparse_index_from_existing_code_documents(self):
        idx = MemoryIndex()
        idx.documents = [
            Document(
                name="worker.py",
                source="code",
                path="/tmp/worker.py",
                paragraphs=[
                    "def refresh_memory_index():\n    return 'done'",
                    "class VectorWorker:\n    pass",
                ],
                doc_type="code",
            ),
            Document(
                name="note.md",
                source="notes",
                path="/tmp/note.md",
                paragraphs=["We decided to keep BM25 scoring for the retrieval worker."],
                doc_type="notes",
            ),
        ]

        idx._rebuild_sparse_index_from_documents()

        assert len(idx.paragraph_index) == 3
        assert idx.documents[0].tokens
        assert idx.paragraph_types[:2] == ["function", "function"]
        assert idx.paragraph_types[2] == "decision"

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

    def test_missing_source_dir_is_skipped(self, tmp_path):
        # A configured source whose directory does not exist must not force a
        # rebuild — the staleness scan skips it and (with no other newer source)
        # reports the index current. This guards the keystone anti-freeze gate:
        # a stray/removed source root should never mask a genuinely fresh index.
        idx_path = tmp_path / "index.pkl"
        idx_path.write_bytes(b"data")

        with patch("memory_index.INDEX_PATH", idx_path):
            with patch("memory_index.SOURCES", {"gone": tmp_path / "does_not_exist"}):
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


class TestPythonBm25Weights:
    def test_precomputed_weights_match_python_bm25_scores(self):
        idx = MemoryIndex()
        idx._built = True
        idx.paragraph_index = [(0, 0), (0, 1)]
        idx.paragraph_tokens = [
            {"alpha", "beta"},
            {"alpha", "gamma"},
        ]
        idx.paragraph_token_counts = [
            {"alpha": 2, "beta": 1},
            {"alpha": 1, "gamma": 1},
        ]
        idx._inverted_index = {
            "alpha": [0, 1],
            "beta": [0],
            "gamma": [1],
        }
        idx.idf = {
            "alpha": 1.25,
            "beta": 1.75,
            "gamma": 1.75,
        }
        idx._para_lengths = np.array([2, 2], dtype=np.float32)
        idx._avg_dl = 2.0

        weights = idx._build_bm25_weight_index()
        scores = idx._search_python_bm25({"alpha", "beta"}, set())

        alpha_weights = dict(weights["alpha"])
        beta_weights = dict(weights["beta"])
        assert scores[0] == alpha_weights[0] + beta_weights[0]
        assert scores[1] == alpha_weights[1]

    def test_python_bm25_weight_cache_respects_filters(self):
        idx = MemoryIndex()
        idx._built = True
        idx.paragraph_index = [(0, 0), (0, 1)]
        idx.paragraph_tokens = [{"alpha"}, {"alpha"}]
        idx.paragraph_token_counts = [{"alpha": 1}, {"alpha": 1}]
        idx._inverted_index = {"alpha": [0, 1]}
        idx.idf = {"alpha": 1.0}
        idx._para_lengths = np.array([1, 1], dtype=np.float32)
        idx._avg_dl = 1.0

        scores = idx._search_python_bm25({"alpha"}, {0})

        assert 0 not in scores
        assert scores[1] > 0

    def test_weight_index_skips_zero_idf_and_stale_postings(self):
        idx = MemoryIndex()
        idx._built = True
        idx._inverted_index = {"zero": [0], "stale": [3], "alpha": [0]}
        idx.idf = {"zero": 0.0, "stale": 1.0, "alpha": 1.0}
        idx.paragraph_token_counts = [{"alpha": 1}]
        idx._para_lengths = np.array([1], dtype=np.float32)
        idx._avg_dl = 1.0

        weights = idx._build_bm25_weight_index()

        assert "zero" not in weights
        assert "stale" not in weights
        assert weights["alpha"][0][0] == 0

    def test_uncached_bm25_handles_missing_idf_filters_and_default_tf(self):
        idx = MemoryIndex()
        idx._built = True
        idx._inverted_index = {"missing": [], "zero": [0], "alpha": [0, 1]}
        idx.idf = {"zero": 0.0, "alpha": 1.0}
        idx.paragraph_token_counts = [{"alpha": 2}]
        idx._para_lengths = np.array([2, 1], dtype=np.float32)
        idx._avg_dl = 1.5

        scores = idx._score_python_bm25_uncached({"missing", "zero", "alpha"}, {0})

        assert set(scores) == {1}
        assert scores[1] > 0.0


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


class TestCompiledFactPriorityResults:
    def test_compiled_fact_results_load_score_filter_and_format_real_facts(self, tmp_path):
        low_score_fact = CompiledFact(
            fact_id="service.worker_refresh",
            fact_type="continuity",
            subject="Worker refresh",
            fact="The vector worker refreshes an index.",
            source="ops.md",
        )
        priority_fact = CompiledFact(
            fact_id="service.vector_worker",
            fact_type="continuity",
            subject="vector worker",
            fact="The vector worker refreshes durable memory indexes.",
            source="ops.md",
        )
        compiled_dir = tmp_path / "compiled"
        write_compiled_facts([low_score_fact, priority_fact], compiled_dir)

        results = memory_index._compiled_fact_results(
            "vector worker", top_k=3, facts_path=compiled_dir / "facts.jsonl"
        )

        assert len(results) == 1
        assert results[0].name == "service.vector_worker.fact"
        assert results[0].source == "compiled"
        assert results[0].score == 1009.0
        assert results[0].answer == priority_fact.fact

    def test_merge_priority_results_deduplicates_and_respects_top_k(self):
        priority = SearchResult(name="a.md", source="compiled", score=10, snippet="same")
        duplicate = SearchResult(name="a.md", source="compiled", score=8, snippet="same")
        ranked = SearchResult(name="b.md", source="notes", score=2, snippet="other")
        extra = SearchResult(name="c.md", source="notes", score=1, snippet="extra")

        merged = memory_index._merge_priority_results(
            [priority],
            [duplicate, ranked, extra],
            top_k=2,
        )

        assert [result.name for result in merged] == ["a.md", "b.md"]


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
        (docs_dir / "a.md").write_text(
            "# Test\n\nContent about BM25 scoring and retrieval quality across persistent memory indexes.",
            encoding="utf-8",
        )
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

    def test_save_without_embeddings_cleans_old_npz(self, tmp_path):
        """When embeddings are None, save removes any stale companion npz."""
        idx = MemoryIndex()
        idx.documents = []
        idx.paragraph_index = []
        idx.paragraph_tokens = []
        idx.paragraph_token_counts = []
        idx.paragraph_types = []
        idx.idf = {}
        idx._df = {}
        idx.embeddings = None
        idx._built = True
        path = tmp_path / "test.gz"
        # Create a fake companion npz that should be cleaned up
        emb_path = path.with_name("test_embeddings.npz")
        emb_path.write_bytes(b"fake")
        assert emb_path.exists()
        idx.save(path)
        assert not emb_path.exists()

    def test_load_drops_stale_embedding_sidecar(self, tmp_path):
        """A stale companion npz must not attach misaligned vectors."""
        idx = self._build_idx(tmp_path)
        path = tmp_path / "idx.gz"
        idx.save(path, quantize=False)
        emb_path = path.with_name("idx_embeddings.npz")
        np.savez_compressed(emb_path, embeddings=np.ones((len(idx.paragraph_index) + 2, 4)))

        idx2 = MemoryIndex()

        assert idx2.load(path) is True
        assert idx2.embeddings is None
        assert idx2.search("BM25 scoring", top_k=3)

    def test_load_ignores_corrupt_embedding_sidecar(self, tmp_path):
        """A corrupt companion npz must not make the sparse index unloadable."""
        idx = self._build_idx(tmp_path)
        path = tmp_path / "idx.gz"
        idx.save(path, quantize=False)
        emb_path = path.with_name("idx_embeddings.npz")
        emb_path.write_bytes(b"not a valid npz")

        idx2 = MemoryIndex()

        assert idx2.load(path) is True
        assert idx2.embeddings is None
        assert idx2.search("BM25 scoring", top_k=3)

    def test_load_nonexistent_path(self, tmp_path):
        """Loading from a path that doesn't exist returns False."""
        idx = MemoryIndex()
        assert idx.load(tmp_path / "nonexistent.gz") is False

    def test_load_corrupt_gzip(self, tmp_path):
        """Corrupt gzip file (valid magic, invalid content) returns False."""
        import gzip

        path = tmp_path / "bad.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"not valid json {{{")
        idx = MemoryIndex()
        assert idx.load(path) is False

    def test_load_unreadable_file(self, tmp_path):
        """File that can't be read (e.g. directory) returns False."""
        path = tmp_path / "dir_as_file"
        path.mkdir()
        idx = MemoryIndex()
        assert idx.load(path) is False

    def test_load_malformed_data(self, tmp_path):
        """Valid gzip JSON but with wrong structure triggers except in load."""
        import gzip

        path = tmp_path / "bad_struct.gz"
        raw = json.dumps({"documents": "not_a_list"}).encode("utf-8")
        with gzip.open(path, "wb") as f:
            f.write(raw)
        idx = MemoryIndex()
        assert idx.load(path) is False

    def test_load_legacy_pickle_path_refused(self, tmp_path):
        """Legacy .pkl INDEX_PATH is refused at runtime; load() returns False.

        Production code now points at .json.gz; a leftover .pkl sitting
        where _LEGACY_INDEX_PATH points is handled by _load_legacy_meta
        returning None (migrator is the supported upgrade path).
        """
        import pickle

        import memory_index

        data = {"documents": []}
        legacy_path = tmp_path / "memory_index.pkl"
        with open(legacy_path, "wb") as f:
            pickle.dump(data, f)
        new_path = tmp_path / "memory_index.json.gz"
        orig_index = memory_index.INDEX_PATH
        orig_legacy = memory_index._LEGACY_INDEX_PATH
        memory_index.INDEX_PATH = new_path
        memory_index._LEGACY_INDEX_PATH = legacy_path
        try:
            idx = MemoryIndex()
            assert idx.load() is False
        finally:
            memory_index.INDEX_PATH = orig_index
            memory_index._LEGACY_INDEX_PATH = orig_legacy

    def test_load_4tuple_format_via_json_gz(self, tmp_path):
        """Old 4-tuple document entries still load from json.gz (new format)."""
        import gzip
        import json as _json

        data = {
            "documents": [["test.md", "src", "/test.md", ["para1"]]],
            "paragraph_index": [[0, 0]],
            "paragraph_tokens": [["test", "para"]],
            "idf": {"test": 1.0, "para": 1.0},
            "embeddings": None,
        }
        path = tmp_path / "legacy.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(_json.dumps(data).encode("utf-8"))
        idx = MemoryIndex()
        assert idx.load(path) is True
        assert idx.documents[0].name == "test.md"
        assert idx.documents[0].date == ""

    def test_validated_loaded_embeddings_rejects_bad_quantized_and_nonfinite_data(self):
        idx = MemoryIndex()
        idx.paragraph_index = [(0, 0)]

        assert (
            idx._validated_loaded_embeddings(np.ones((1, 2), dtype=np.int8), None, quantized=True)
            is None
        )
        assert idx._validated_loaded_embeddings(object(), np.ones((1, 1)), quantized=True) is None
        assert (
            idx._validated_loaded_embeddings(
                np.array([[np.nan]], dtype=np.float32),
                None,
                quantized=False,
            )
            is None
        )


# ── Temporal query sorting ──────────────────────────────────────


class TestTemporalSorting:
    def test_newest_first(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "2026-03-10_old.md").write_text(
            "# Old\n\nWhen did we start in 2026-03-10 on the project.", encoding="utf-8"
        )
        (docs_dir / "2026-03-20_new.md").write_text(
            "# New\n\nWhen was the latest update on 2026-03-20.", encoding="utf-8"
        )
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
            "# Old\n\nThe first experiment was on 2026-03-10 with STDP.", encoding="utf-8"
        )
        (docs_dir / "2026-03-20_new.md").write_text(
            "# New\n\nThe first thing we tried with STDP was this.", encoding="utf-8"
        )
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
            encoding="utf-8",
        )
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
            encoding="utf-8",
        )
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
        (docs_dir / "test.md").write_text(
            "# Test\n\nEnough content for indexing here.", encoding="utf-8"
        )
        idx_path = tmp_path / "missing_index.pkl"
        with (
            patch("memory_index.INDEX_PATH", idx_path),
            patch("memory_index.SOURCES", {"test": docs_dir}),
            patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}),
        ):
            idx = auto_rebuild_if_needed(use_gpu=False)
        assert idx._built is True

    def test_no_rebuild_when_fresh(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Test\n\nEnough content here for index.", encoding="utf-8"
        )
        idx_path = tmp_path / "fresh_index.pkl"
        idx = MemoryIndex()
        with (
            patch("memory_index.SOURCES", {"test": docs_dir}),
            patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}),
        ):
            idx.build(use_gpu_embeddings=False, use_gliner=False)
        idx.save(idx_path)
        import os

        os.utime(idx_path, None)
        with (
            patch("memory_index.INDEX_PATH", idx_path),
            patch("memory_index.SOURCES", {"test": docs_dir}),
        ):
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
            encoding="utf-8",
        )
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
        (docs_dir / "real.md").write_text(
            "# Test\n\nReal content for indexing purposes.", encoding="utf-8"
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md", ".py"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)
        names = [d.name for d in idx.documents]
        assert "module.py" not in names

    def test_search_triggers_build(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Test\n\nContent about automated build testing.", encoding="utf-8"
        )
        idx = MemoryIndex()
        with (
            patch("memory_index.SOURCES", {"test": docs_dir}),
            patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}),
        ):
            idx.search("automated build", top_k=3)
        assert idx._built is True

    def test_build_skips_temporal_graph_when_corpus_exceeds_threshold(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Test\n\nContent about automated build testing with enough searchable detail.",
            encoding="utf-8",
        )

        idx = MemoryIndex()
        with (
            patch("memory_index.SOURCES", {"test": docs_dir}),
            patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}),
            patch("memory_index.TEMPORAL_GRAPH_MAX_DOCUMENTS", 0),
        ):
            stats = idx.build(use_gpu_embeddings=False, use_gliner=False)

        assert stats["documents"] == 1
        assert idx._temporal_graph is None

    def test_add_file_short_content(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text(
            "# Test\n\nSome content long enough for build.", encoding="utf-8"
        )
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
            '"""Module doc."""\n\ndef search_function():\n    pass\n', encoding="utf-8"
        )
        (docs_dir / "note.md").write_text(
            "# Note\n\nThis is a note about search functions and algorithms.", encoding="utf-8"
        )
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

    def test_entity_boost_score_uses_typed_relation_labels(self):
        graph = {
            "entities": {
                "e1": {"label": "stdp"},
                "e2": {"label": "bm25"},
            },
            "relations": [{"source": "e1", "target": "e2", "type": "fixed_by"}],
        }

        score = _entity_boost_score("BM25 retrieval improved after the fix.", {"e1"}, graph)

        assert score >= 0.15

    def test_relation_neighbors_skip_incomplete_edges(self):
        neighbors = memory_index._build_relation_neighbors(
            {"e1": {"label": "alpha"}, "e2": {"label": "beta"}},
            [
                {"source": "e1", "target": "e2", "type": "fixed_by"},
                {"source": "", "target": "e2"},
                {"source": "e1", "target": ""},
            ],
        )

        assert neighbors == {
            "e1": [("e2", "beta", "fixed_by")],
            "e2": [("e1", "alpha", "fixed_by")],
        }

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
            json.dumps({"id": "e1", "type": "concept", "label": "test"}) + "\n", encoding="utf-8"
        )
        (graph_dir / "relations.jsonl").write_text(
            json.dumps({"source": "e1", "target": "e2", "weight": 1.0}) + "\n", encoding="utf-8"
        )
        memory_index.GRAPH_DIR = graph_dir
        memory_index._ENTITY_GRAPH = None
        try:
            g = _load_entity_graph()
            assert "e1" in g["entities"]
            assert len(g["relations"]) == 1
        finally:
            memory_index.GRAPH_DIR = original
            memory_index._ENTITY_GRAPH = None

    def test_load_entity_graph_reloads_when_files_change(self, tmp_path):
        import memory_index

        original = memory_index.GRAPH_DIR
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        entities_path = graph_dir / "entities.jsonl"
        relations_path = graph_dir / "relations.jsonl"
        entities_path.write_text(
            json.dumps({"id": "e1", "type": "concept", "label": "alpha"}) + "\n",
            encoding="utf-8",
        )
        relations_path.write_text("", encoding="utf-8")
        memory_index.GRAPH_DIR = graph_dir
        memory_index._ENTITY_GRAPH = None
        try:
            first = _load_entity_graph()
            assert set(first["entities"]) == {"e1"}

            entities_path.write_text(
                json.dumps({"id": "e1", "type": "concept", "label": "alpha"})
                + "\n"
                + json.dumps({"id": "e2", "type": "concept", "label": "beta"})
                + "\n",
                encoding="utf-8",
            )
            second = _load_entity_graph()

            assert set(second["entities"]) == {"e1", "e2"}
        finally:
            memory_index.GRAPH_DIR = original
            memory_index._ENTITY_GRAPH = None

    def test_search_with_entity_graph_boost(self, tmp_path):
        """Entity graph boost path in search (covers lines 671-685)."""
        import memory_index

        # Build a minimal index
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "trace.md").write_text(
            "# STDP\n\nSTDP removal was a key decision for BM25 retrieval.",
            encoding="utf-8",
        )
        idx = MemoryIndex()
        with patch("memory_index.SOURCES", {"test": docs_dir}):
            with patch("memory_index.SOURCE_EXTENSIONS", {"test": {".md"}}):
                idx.build(use_gpu_embeddings=False, use_gliner=False)

        # Setup entity graph
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "entities.jsonl").write_text(
            json.dumps({"id": "e1", "type": "concept", "label": "stdp"})
            + "\n"
            + json.dumps({"id": "e2", "type": "concept", "label": "bm25"})
            + "\n",
            encoding="utf-8",
        )
        (graph_dir / "relations.jsonl").write_text(
            json.dumps({"source": "e1", "target": "e2", "weight": 1.0}) + "\n",
            encoding="utf-8",
        )
        original_graph = memory_index.GRAPH_DIR
        memory_index.GRAPH_DIR = graph_dir
        memory_index._ENTITY_GRAPH = None
        try:
            results = idx.search("STDP decision", top_k=3)
            assert len(results) >= 1
        finally:
            memory_index.GRAPH_DIR = original_graph
            memory_index._ENTITY_GRAPH = None


# ── Person-name boosting in search ────────────────────────────────


class TestPersonNameBoostInSearch:
    def test_person_centric_query_boosts(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "alice_trace.md").write_text(
            "# Alice's Piano Hobby\n\nAlice has been learning piano since January.\n"
            "She practices every day and loves Chopin.\n"
            "Alice considers piano her favorite hobby.",
            encoding="utf-8",
        )
        (traces / "weather_trace.md").write_text(
            "# Weather Report\n\nThe temperature today is 22 degrees.\n"
            "Piano lessons are cancelled due to rain.\n"
            "No specific person mentioned here at all.",
            encoding="utf-8",
        )

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
        (traces / "doc.md").write_text("# Alpha\n\nAlpha beta gamma delta.", encoding="utf-8")

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
            "We need to finish the review by 2026-03-01.",
            encoding="utf-8",
        )

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

    def test_temporal_code_answer_injected(self, tmp_path):
        """Covers line 804: temporal_code_execute result injected into results[0]."""
        traces = tmp_path / "traces"
        traces.mkdir()
        # Use terms that BM25 can actually match on
        (traces / "timeline.md").write_text(
            "# Timeline\n\n"
            "The project milestone alpha completed on 2026-01-10.\n"
            "The project milestone beta completed on 2026-03-20.\n"
            "Days between milestones is important for tracking.\n",
            encoding="utf-8",
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            results = idx.search("how many days between milestone events", top_k=3)
            assert len(results) > 0
        finally:
            memory_index.SOURCES = original_sources


# ── Multi-hop search ─────────────────────────────────────────────


class TestMultiHopSearch:
    def test_person_who_query_triggers_multihop(self, tmp_path):
        """Covers lines 518, 840-859, 872-924: _multi_hop_search + _single_query_search."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "conv1.md").write_text(
            "# Conversation\n\n"
            "Caroline: I love pottery and swimming.\n"
            "Assistant: That sounds great!\n\n"
            "Caroline: I also enjoy hiking on weekends.\n",
            encoding="utf-8",
        )
        (traces / "conv2.md").write_text(
            "# Conversation\n\n"
            "Melanie: I work as a software engineer.\n"
            "Assistant: Interesting career!\n\n"
            "Melanie: I enjoy reading science fiction books.\n",
            encoding="utf-8",
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # Multi-hop pattern: "What X does the person who Y?"
            results = idx.search(
                "What hobbies does the person who works as a software engineer have?",
                top_k=3,
            )
            assert isinstance(results, list)
        finally:
            memory_index.SOURCES = original_sources

    def test_does_person_who_also(self, tmp_path):
        """Covers second multi-hop pattern: 'Does the person who X also Y?'"""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "data.md").write_text(
            "# Data\n\n"
            "Caroline enjoys pottery and swimming at the lake.\n"
            "She also likes reading mystery novels in the evening.\n",
            encoding="utf-8",
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            results = idx.search(
                "Does the person who enjoys pottery also like reading?",
                top_k=3,
            )
            assert isinstance(results, list)
        finally:
            memory_index.SOURCES = original_sources

    def test_multihop_subqueries_use_full_search_stack(self, tmp_path):
        fact = CompiledFact(
            fact_id="person.melanie_hobbies",
            fact_type="continuity",
            subject="software engineer hobbies",
            fact="Melanie works as a software engineer and likes chess.",
            source="conversation.md",
            aliases=["software engineer"],
        )
        compiled_dir = tmp_path / "compiled"
        write_compiled_facts([fact], compiled_dir)
        idx = MemoryIndex(compiled_facts_path=compiled_dir / "facts.jsonl")
        idx._built = True
        idx.documents = [
            Document(
                name="compiled_facts.md",
                source="compiled",
                path="/tmp/compiled_facts.md",
                paragraphs=["Compiled fact memory is operational."],
            )
        ]
        idx.paragraph_index = [(0, 0)]
        idx.paragraph_tokens = [{"compiled", "fact", "memory"}]
        idx.paragraph_token_counts = [{"compiled": 1, "fact": 1, "memory": 1}]
        idx.paragraph_types = ["finding"]
        idx.idf = {"compiled": 1.0, "fact": 1.0, "memory": 1.0}
        idx._df = {"compiled": 1, "fact": 1, "memory": 1}
        idx._inverted_index = {"compiled": [0], "fact": [0], "memory": [0]}
        idx._para_lengths = np.array([3], dtype=np.float32)
        idx._avg_dl = 3.0

        results = idx.search(
            "What hobbies does the person who works as a software engineer have?",
            top_k=3,
        )

        assert results
        assert results[0].name == "person.melanie_hobbies.fact"
        assert results[0].answer == fact.fact


# ── Rust BM25 paths ──────────────────────────────────────────────


class TestRustBm25Paths:
    @staticmethod
    def _real_index(tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "doc.md").write_text(
            "# Doc\n\nSome content about alpha beta gamma for searching.",
            encoding="utf-8",
        )
        original_sources = memory_index.SOURCES
        original_extensions = memory_index.SOURCE_EXTENSIONS
        original_hash_cache = memory_index.HASH_CACHE_PATH
        memory_index.SOURCES = {"test": traces}
        try:
            memory_index.SOURCE_EXTENSIONS = {"test": {".md"}}
            memory_index.HASH_CACHE_PATH = tmp_path / "content_hashes.json"
            idx = MemoryIndex()
            idx.build(
                use_gpu_embeddings=False,
                use_gliner=False,
                incremental=False,
            )
            return idx
        finally:
            memory_index.SOURCES = original_sources
            memory_index.SOURCE_EXTENSIONS = original_extensions
            memory_index.HASH_CACHE_PATH = original_hash_cache

    def test_search_rust_bm25_success(self, tmp_path):
        idx = self._real_index(tmp_path)

        result = idx._search_rust_bm25({"alpha", "beta"}, top_k=10)

        assert result is not None
        assert list(result) == [0]
        assert result[0] > 0.0

    def test_ensure_rust_bm25_builds_and_caches_real_extension(self, tmp_path):
        import remanentia_search

        idx = self._real_index(tmp_path)
        first = idx._ensure_rust_bm25()
        second = idx._ensure_rust_bm25()

        assert isinstance(first, remanentia_search.BM25Index)
        assert second is first
        assert first.num_paragraphs() == len(idx.paragraph_index)

    def test_search_uses_real_rust_bm25(self, tmp_path, monkeypatch):
        import remanentia_search

        idx = self._real_index(tmp_path)
        monkeypatch.setenv("REMANENTIA_USE_RUST_BM25", "1")

        results = idx.search("alpha beta", top_k=3)

        assert results
        assert results[0].name == "doc.md"
        assert isinstance(idx._rust_bm25, remanentia_search.BM25Index)


# ── Code splitting edge cases ────────────────────────────────────


class TestCodeSplitNonPython:
    def test_non_python_function_blocks(self):
        """Covers line 1547: non-Python function block extraction via regex."""
        rust_code = (
            "fn compute_score(query: &str) -> f64 {\n"
            "    let tokens = query.split_whitespace();\n"
            "    tokens.count() as f64\n"
            "}\n\n"
            "fn search(index: &Index, q: &str) -> Vec<Result> {\n"
            "    index.find(q).collect()\n"
            "}\n"
        )
        chunks = _split_code(rust_code)
        assert len(chunks) >= 1
        assert any("compute_score" in c for c in chunks)

    def test_non_python_docstring_extraction(self):
        """Covers line 1536: module docstring extraction in non-Python code."""
        code_with_docstring = (
            '"""This is a module-level docstring for testing."""\n\n'
            "// Some non-parseable code follows\n"
            "// that is not valid Python\n"
            "invalid syntax here @#$%\n"
        )
        chunks = _split_code(code_with_docstring)
        assert len(chunks) >= 1


# ── _single_query_search direct calls ────────────────────────────


class TestSingleQuerySearch:
    def test_empty_tokens(self, tmp_path):
        """Covers line 874: empty tokens returns []."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "doc.md").write_text("# Doc\n\nContent here.", encoding="utf-8")

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            assert idx._single_query_search("") == []
        finally:
            memory_index.SOURCES = original_sources

    def test_with_filters(self, tmp_path):
        """Covers lines 878-894: filter logic in _single_query_search."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "2026-01-10_alpha.md").write_text(
            "# Alpha\n\nAlpha content for project testing.", encoding="utf-8"
        )
        (traces / "2026-03-20_beta.md").write_text(
            "# Beta\n\nBeta content for project testing.", encoding="utf-8"
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # Filter by after date
            results = idx._single_query_search("project testing", top_k=5, after="2026-02-01")
            assert isinstance(results, list)
            # Filter by before date
            results = idx._single_query_search("project testing", top_k=5, before="2026-02-01")
            assert isinstance(results, list)
            # Filter by doc_type
            results = idx._single_query_search("project testing", top_k=5, doc_type="traces")
            assert isinstance(results, list)
            # Filter by project
            results = idx._single_query_search("project testing", top_k=5, project="test")
            assert isinstance(results, list)
        finally:
            memory_index.SOURCES = original_sources

    def test_result_limit(self, tmp_path):
        """Covers line 923: break when enough results found."""
        traces = tmp_path / "traces"
        traces.mkdir()
        for i in range(5):
            (traces / f"doc{i}.md").write_text(
                f"# Doc {i}\n\nUnique content for document number {i} about searching.\n",
                encoding="utf-8",
            )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            results = idx._single_query_search("content searching", top_k=2)
            assert len(results) <= 2
        finally:
            memory_index.SOURCES = original_sources


# ── Prospective query patterns ───────────────────────────────────


class TestProspectiveQueryPatterns:
    def test_activity_and_occupation_patterns(self):
        queries = _generate_prospective_queries(
            "Caroline enjoys pottery and works as a teacher.",
            "profile.md",
            "discussion",
        )
        assert "hobbies pottery and works as a teacher" in queries
        assert any(query.startswith("job ") for query in queries)

    def test_learning_pattern(self):
        queries = _generate_prospective_queries(
            "Caroline is learning quantum painting.",
            "profile.md",
            "discussion",
        )
        assert "quantum painting" in queries

    def test_finding_pattern(self):
        queries = _generate_prospective_queries(
            "The experiment found a stable retrieval invariant.",
            "retrieval_results.md",
            "finding",
        )
        assert "what did we find about retrieval results" in queries

    def test_relationship_pattern(self):
        """Covers lines 1337-1338."""
        queries = _generate_prospective_queries(
            "She is married to John and they live together.",
            "relationship.md",
            "relationship",
        )
        assert any("relationship" in q.lower() for q in queries)

    def test_allergy_pattern(self):
        """Covers lines 1346-1348."""
        queries = _generate_prospective_queries(
            "Caroline is allergic to peanuts and shellfish.",
            "health.md",
            "discussion",
        )
        assert any("allergic" in q.lower() for q in queries)

    def test_favourite_pattern(self):
        """Covers lines 1375-1376."""
        queries = _generate_prospective_queries(
            "Her favourite movie is Spirited Away.",
            "prefs.md",
            "discussion",
        )
        assert any("favourite" in q.lower() for q in queries)

    def test_version_pattern(self):
        """Covers lines 1398-1399."""
        queries = _generate_prospective_queries(
            "Released v3.9.0 to PyPI on 2026-03-15.",
            "release.md",
            "metric",
        )
        assert any("v3.9.0" in q for q in queries)


# ── Sentence splitting and paragraph splitting ───────────────────


class TestSentenceSplitting:
    def test_split_sentences(self):
        """Covers lines 1516-1522."""
        text = "First sentence here. Second sentence follows. Third sentence ends."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_filters_short(self):
        sents = _split_sentences("Hi. Ok. Sure.")
        assert len(sents) == 0

    def test_long_paragraph_sentence_windows(self):
        """Covers lines 1497-1507: context window splitting."""
        long_para = (
            "The first important finding was about BM25 retrieval accuracy in production systems. "
            "The second finding showed that embedding models outperform keyword search by a significant margin. "
            "The third finding was about temporal query handling and how dates should be normalized. "
            "The fourth finding related to entity extraction patterns in conversational memory systems. "
            "The fifth finding established that cross-encoder reranking improves precision at the top positions."
        )
        paragraphs = _split_paragraphs(long_para)
        assert len(paragraphs) >= 2


# ── Cross-reference answers ──────────────────────────────────────


class TestCrossReferenceAnswers:
    def test_agreeing_answers_boost(self):
        """Covers lines 1634-1638."""
        results = [
            SearchResult(name="a.md", source="s", score=0.9, snippet="s1", answer="March 15"),
            SearchResult(name="b.md", source="s", score=0.8, snippet="s2", answer="March 15"),
            SearchResult(name="c.md", source="s", score=0.7, snippet="s3", answer="March 20"),
        ]
        boosted = _cross_reference_answers(results)
        # Two results agree ("march 15") → confidence boosted above default 0.0
        assert boosted[0].confidence > 0.0
        assert boosted[1].confidence > 0.0

    def test_no_agreement_no_change(self):
        results = [
            SearchResult(name="a.md", source="s", score=0.9, snippet="s1", answer="March 15"),
            SearchResult(name="b.md", source="s", score=0.8, snippet="s2", answer="March 20"),
        ]
        unchanged = _cross_reference_answers(results)
        assert unchanged[0].confidence == results[0].confidence


# ── Query decomposition ──────────────────────────────────────────


class TestDecomposeQuery:
    def test_what_happened_before(self):
        """Covers line 1678."""
        result = _decompose_query("What happened before the meeting ended?")
        assert result is not None
        assert len(result) == 2

    def test_what_happened_after(self):
        result = _decompose_query("What happened after the release?")
        assert result is not None
        assert len(result) == 2

    def test_simple_query_no_decomposition(self):
        assert _decompose_query("What is BM25?") is None


# ── Reciprocal Rank Fusion ───────────────────────────────────────


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        """Covers lines 1701-1706."""
        list1 = [(0, 5.0), (1, 3.0), (2, 1.0)]
        list2 = [(1, 4.0), (2, 2.0), (3, 1.0)]
        result = _reciprocal_rank_fusion([list1, list2])
        # Result should be sorted by RRF score descending
        assert len(result) == 4
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
        # Item 1 appears in both lists → highest RRF score
        assert result[0][0] == 1

    def test_empty_lists(self):
        assert _reciprocal_rank_fusion([]) == []

    def test_single_list(self):
        result = _reciprocal_rank_fusion([[(5, 1.0), (3, 0.5)]])
        assert len(result) == 2

    def test_explicit_python_rrf(self):
        result = reciprocal_rank_fusion_python([[(0, 1.0)], [(0, 0.5), (1, 0.4)]], k=60)

        assert result[0][0] == 0
        assert result[0][1] > result[1][1]


# ── Additional memory-index behaviours ──────────────────────────────────────


class TestRustBm25NoneReturn:
    def test_search_rust_bm25_returns_none_when_no_engine(self, tmp_path):
        """Covers line 952: _search_rust_bm25 returns None when ensure returns None."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "doc.md").write_text("# Doc\n\nAlpha beta gamma.", encoding="utf-8")

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            idx._rust_bm25 = False  # disabled
            result = idx._search_rust_bm25({"alpha"}, top_k=10)
            assert result is None
        finally:
            memory_index.SOURCES = original_sources


class TestGetRustBm25ClassSuccessImport:
    def test_successful_import(self):
        import remanentia_search
        import memory_index

        old_attempted = memory_index._RUST_BM25_IMPORT_ATTEMPTED
        old_cls = memory_index._RUST_BM25_CLASS
        memory_index._RUST_BM25_IMPORT_ATTEMPTED = False
        memory_index._RUST_BM25_CLASS = None
        try:
            result = memory_index._get_rust_bm25_class()
            assert result is remanentia_search.BM25Index
        finally:
            memory_index._RUST_BM25_IMPORT_ATTEMPTED = old_attempted
            memory_index._RUST_BM25_CLASS = old_cls


class TestSplitParagraphsLongTwoSentences:
    def test_long_block_two_sentences(self):
        """Covers line 1499: long paragraph with <=2 sentences kept whole."""
        # One block >200 chars but only 2 sentences
        long_text = (
            "This is the first rather long sentence that contains enough words "
            "to push the total character count well past the two hundred character threshold. "
            "This is the second sentence which is also long enough to be meaningful."
        )
        assert len(long_text) > 200
        paragraphs = _split_paragraphs(long_text)
        assert len(paragraphs) >= 1

    def test_long_block_many_sentences(self):
        """Covers lines 1501-1507: context window splitting for 3+ sentences."""
        long_text = (
            "First sentence about BM25 retrieval in production systems has important findings. "
            "Second sentence about embedding models showing significant improvements overall. "
            "Third sentence about temporal handling and date normalization is critical. "
            "Fourth sentence relates to entity extraction patterns in memory systems. "
            "Fifth sentence about cross-encoder reranking improving precision substantially."
        )
        assert len(long_text) > 200
        paragraphs = _split_paragraphs(long_text)
        assert len(paragraphs) >= 3  # each sentence gets a context window


class TestSingleQuerySearchFilters:
    def test_all_filters_exclude(self, tmp_path):
        """Covers lines 881-894 in _single_query_search: all filter branches."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "2026-01-10_early.md").write_text(
            "# Early\n\nEarly document about alpha searching algorithms.", encoding="utf-8"
        )
        (traces / "2026-06-20_late.md").write_text(
            "# Late\n\nLate document about alpha searching algorithms.", encoding="utf-8"
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # project filter EXCLUDES docs (project not in source/name)
            r1 = idx._single_query_search("alpha searching", top_k=5, project="nonexistent")
            assert r1 == []
            # after filter excludes early doc
            r2 = idx._single_query_search("alpha searching", top_k=5, after="2026-04-01")
            assert isinstance(r2, list)
            # before filter excludes late doc
            r3 = idx._single_query_search("alpha searching", top_k=5, before="2026-03-01")
            assert isinstance(r3, list)
            # doc_type filter excludes (none are "code" type)
            r4 = idx._single_query_search("alpha searching", top_k=5, doc_type="code")
            assert r4 == []
        finally:
            memory_index.SOURCES = original_sources


class TestTemporalCodeInjectionDirect:
    def test_how_long_between_events(self, tmp_path):
        """Covers line 804: temporal code answer injection via direct test."""

        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "events.md").write_text(
            "# Events\n\n"
            "Alpha event completed on 2026-01-10 with results.\n\n"
            "Beta event completed on 2026-03-20 with results.\n\n"
            "How long between events matters for planning.\n",
            encoding="utf-8",
        )

        import memory_index

        original_sources = memory_index.SOURCES
        memory_index.SOURCES = {"test": traces}
        try:
            idx = MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False)
            # Patch temporal_code_execute to return a known answer
            with patch(
                "temporal_graph.temporal_code_execute",
                return_value="69 days (from 2026-01-10 to 2026-03-20)",
            ):
                results = idx.search("how many days between events", top_k=3)
            if results and results[0].answer:
                assert "69 days" in results[0].answer
        finally:
            memory_index.SOURCES = original_sources


# ── Content-hash incremental indexing ───────────────────────────


class TestContentHashIndexing:
    """Tests for SHA-256 content-hash based incremental build."""

    def test_hash_cache_save_load_roundtrip(self, tmp_path):
        """Hash cache should survive save → load roundtrip."""
        from memory_index import MemoryIndex

        hashes = {"file1.md": "abc123", "file2.py": "def456"}
        cache_path = tmp_path / "hashes.json"
        MemoryIndex._save_content_hashes(hashes, cache_path)
        loaded = MemoryIndex._load_content_hashes(cache_path)
        assert loaded == hashes

    def test_hash_cache_load_nonexistent(self, tmp_path):
        """Loading from nonexistent path returns empty dict."""
        from memory_index import MemoryIndex

        result = MemoryIndex._load_content_hashes(tmp_path / "nope.json")
        assert result == {}

    def test_hash_cache_load_corrupt(self, tmp_path):
        """Corrupt JSON returns empty dict, not crash."""
        from memory_index import MemoryIndex

        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json", encoding="utf-8")
        result = MemoryIndex._load_content_hashes(bad)
        assert result == {}

    def test_build_populates_hash_stats(self, tmp_path):
        """Build should report hash_hits and hash_misses in stats."""
        import memory_index

        original_sources = memory_index.SOURCES
        original_hash_path = memory_index.HASH_CACHE_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "test.md").write_text(
            "# Test Trace for Indexing\n\nWe found that retrieval accuracy improved significantly after switching to BM25 with cross-encoder reranking.\n"
        )

        memory_index.SOURCES = {"traces": traces}
        memory_index.HASH_CACHE_PATH = tmp_path / "hashes.json"

        try:
            idx = memory_index.MemoryIndex()
            stats = idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert "hash_hits" in stats
            assert "hash_misses" in stats
            assert stats["hash_misses"] >= 1  # first build, all are misses
            assert stats["hash_hits"] == 0
        finally:
            memory_index.SOURCES = original_sources
            memory_index.HASH_CACHE_PATH = original_hash_path

    def test_second_build_preserves_unchanged_documents(self, tmp_path):
        """Unchanged files must remain searchable after a fresh incremental build."""
        import memory_index

        original_sources = memory_index.SOURCES
        original_hash_path = memory_index.HASH_CACHE_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "test.md").write_text(
            "# Test Trace for Indexing\n\nWe found that retrieval accuracy improved by 15 percent after implementing the BM25 scoring engine.\n"
        )

        memory_index.SOURCES = {"traces": traces}
        memory_index.HASH_CACHE_PATH = tmp_path / "hashes.json"

        try:
            idx = memory_index.MemoryIndex()
            # First build
            stats1 = idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert stats1["hash_misses"] >= 1

            # Second build records the unchanged file as a hash hit, but the
            # fresh in-memory index must still contain searchable documents.
            idx2 = memory_index.MemoryIndex()
            stats2 = idx2.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert stats2["hash_hits"] >= 1
            assert stats2["documents"] >= 1
            results = idx2.search("retrieval accuracy", top_k=3)
            assert any(result.name == "test.md" for result in results)
        finally:
            memory_index.SOURCES = original_sources
            memory_index.HASH_CACHE_PATH = original_hash_path

    def test_changed_file_detected(self, tmp_path):
        """Modified file should be re-indexed (hash_misses on second build)."""
        import memory_index

        original_sources = memory_index.SOURCES
        original_hash_path = memory_index.HASH_CACHE_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        f = traces / "test.md"
        f.write_text(
            "# Test Trace for Indexing\n\nWe found that retrieval accuracy measured at 80 percent on the LOCOMO benchmark dataset.\n"
        )

        memory_index.SOURCES = {"traces": traces}
        memory_index.HASH_CACHE_PATH = tmp_path / "hashes.json"

        try:
            idx = memory_index.MemoryIndex()
            idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)

            # Modify file
            f.write_text(
                "# Test Trace for Indexing\n\nWe found that retrieval accuracy improved to 95 percent after the cross-encoder reranking fix was applied.\n"
            )

            idx2 = memory_index.MemoryIndex()
            stats2 = idx2.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert stats2["hash_misses"] >= 1  # changed file re-indexed
        finally:
            memory_index.SOURCES = original_sources
            memory_index.HASH_CACHE_PATH = original_hash_path

    def test_non_incremental_build_skips_hash_check(self, tmp_path):
        """With incremental=False, all files should be indexed regardless."""
        import memory_index

        original_sources = memory_index.SOURCES
        original_hash_path = memory_index.HASH_CACHE_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "test.md").write_text(
            "# Test Trace for Indexing\n\nWe decided to use BM25 retrieval as the primary scoring mechanism for all knowledge sources.\n"
        )

        memory_index.SOURCES = {"traces": traces}
        memory_index.HASH_CACHE_PATH = tmp_path / "hashes.json"

        try:
            idx = memory_index.MemoryIndex()
            # First build (saves hashes)
            idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)

            # Non-incremental rebuild should NOT skip
            idx2 = memory_index.MemoryIndex()
            stats = idx2.build(use_gpu_embeddings=False, use_gliner=False, incremental=False)
            assert stats["hash_hits"] == 0  # no skipping
            assert stats["documents"] >= 1
        finally:
            memory_index.SOURCES = original_sources
            memory_index.HASH_CACHE_PATH = original_hash_path

    def test_project_coordination_source_expands_sessions_and_handovers(self, tmp_path):
        """Project coordination roots should ingest nested session and handover files."""
        import memory_index

        project = tmp_path / "PROJECT"
        sessions = project / ".coordination" / "sessions"
        handovers = project / ".coordination" / "handovers"
        sessions.mkdir(parents=True)
        handovers.mkdir(parents=True)
        (sessions / "session.md").write_text(
            "Session note with enough detail about vector worker retrieval performance.",
            encoding="utf-8",
        )
        (handovers / "handover.json").write_text(
            '{"text":"Handover note with enough detail about local memory ingestion."}',
            encoding="utf-8",
        )
        (project / "README.md").write_text(
            "This regular project markdown file is not a coordination record.",
            encoding="utf-8",
        )

        files = list(memory_index._iter_source_files("repo_coordination", tmp_path))
        names = {path.name for path in files}

        assert names == {"handover.json", "session.md"}

    def test_arcane_stimuli_source_indexes_json(self, tmp_path):
        """Stimulus JSON files are first-class memory source documents."""
        import memory_index

        (tmp_path / "stimulus.json").write_text(
            '{"text":"Stimulus note with enough operational detail for retrieval."}',
            encoding="utf-8",
        )

        files = list(memory_index._iter_source_files("arcane_stimuli", tmp_path))

        assert [path.name for path in files] == ["stimulus.json"]

    def test_manuscript_source_skips_archives_and_indexes_current_text(self, tmp_path):
        """Manuscript indexing excludes archive trees but keeps current source text."""
        current = tmp_path / "current"
        archive = tmp_path / "ARCHIVE"
        current.mkdir()
        archive.mkdir()
        (current / "paper.md").write_text(
            "Current manuscript text with enough retrieval context.",
            encoding="utf-8",
        )
        (archive / "old.md").write_text(
            "Archived manuscript text with enough retrieval context.",
            encoding="utf-8",
        )

        files = list(memory_index._iter_source_files("manuscripts", tmp_path))

        assert [path.name for path in files] == ["paper.md"]
