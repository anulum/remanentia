# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for compiled memory facts

from __future__ import annotations

import gzip
import json
from pathlib import Path

import memory_index
from benchmark_suites import current_operational_queries, historical_regression_queries
from compiled_memory import CompiledFact, compile_facts, load_compiled_facts, search_compiled_facts
from memory_index import MemoryIndex, SearchResult


def _write_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "documents": [("a.md", "source", "a.md", ["paragraph"], "", "source")],
        "paragraph_index": [(0, 0), (0, 0)],
        "idf": {"alpha": 1.0, "beta": 2.0},
        "timestamp": 1_774_654_400.0,
    }
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(meta).encode("utf-8"))


def _write_performance_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-28T17:27:18Z",
                "api": [
                    {
                        "name": "api_health",
                        "last_result_summary": {"daemon_kind": "vector_worker"},
                    },
                    {"name": "api_recall", "p50_ms": 841.245, "p95_ms": 1480.7},
                    {"name": "api_public_vector_search", "p50_ms": 95.347, "p95_ms": 121.404},
                ],
                "vector": {"index": {"count": 1103, "dimension": 768, "total_bytes": 4085522}},
            }
        ),
        encoding="utf-8",
    )


def test_compile_facts_reads_index_snapshot(tmp_path):
    repo = tmp_path
    _write_index(repo / "snn_state" / "memory_index.json.gz")

    facts = compile_facts(repo, out_dir=repo / "memory" / "compiled")
    size_fact = next(f for f in facts if f.fact_id == "index.current_size")

    assert "1 documents" in size_fact.fact
    assert "2 paragraphs" in size_fact.fact
    assert "2 unique tokens" in size_fact.fact
    assert size_fact.truth_mode == "current"
    assert size_fact.scope == "remanentia"


def test_compile_facts_reads_benchmark_locations(tmp_path):
    repo = tmp_path
    bench_dir = repo / ".coordination" / "benchmarks" / "REMANENTIA"
    report_path = bench_dir / "remanentia_performance_2026-04-28_172825.json"
    _write_performance_report(report_path)

    facts = compile_facts(repo, out_dir=repo / "memory" / "compiled")
    fact = next(f for f in facts if f.fact_id == "benchmark.latest_performance_report")
    recall_fact = next(f for f in facts if f.fact_id == "performance.api_recall_p50")
    public_vector_fact = next(
        f for f in facts if f.fact_id == "performance.public_vector_search_p50"
    )
    worker_fact = next(f for f in facts if f.fact_id == "service.official_vector_worker")
    vector_fact = next(f for f in facts if f.fact_id == "performance.vector_index_chunks")

    assert "remanentia_performance_2026-04-28_172825.json" in fact.fact
    assert ".coordination" in fact.fact
    assert "841.245" in recall_fact.fact
    assert "95.347" in public_vector_fact.fact
    assert "vector_worker" in worker_fact.fact
    assert "1,103" in vector_fact.fact
    assert recall_fact.truth_mode == "current"


def test_compile_facts_indexes_module_symbols(tmp_path):
    repo = tmp_path
    module = repo / "entity_extractor.py"
    module.write_text(
        "def extract_entities(text):\n    return []\n\nclass EntityGraph:\n    pass\n"
    )

    facts = compile_facts(repo, out_dir=repo / "memory" / "compiled")
    fact = next(f for f in facts if f.fact_id == "symbols.entity_extractor")

    assert "entity_extractor.py" in fact.fact
    assert "extract_entities" in fact.fact
    assert "EntityGraph" in fact.fact


def test_compile_facts_writes_loadable_jsonl(tmp_path):
    repo = tmp_path
    _write_index(repo / "snn_state" / "memory_index.json.gz")
    out_dir = repo / "memory" / "compiled"

    compile_facts(repo, out_dir=out_dir)
    facts = load_compiled_facts(out_dir / "facts.jsonl")

    assert any(f.fact_id == "index.current_size" for f in facts)


def test_compile_facts_adds_historical_benchmark_expected_facts(tmp_path):
    repo = tmp_path
    _write_index(repo / "snn_state" / "memory_index.json.gz")

    facts = compile_facts(repo, out_dir=repo / "memory" / "compiled")
    fact = next(f for f in facts if f.fact_id == "index.historical_march_size_expected")

    assert fact.truth_mode == "benchmark_expected"
    assert fact.scope == "benchmark"
    assert "1,217" in fact.fact


def test_search_compiled_facts_prefers_alias_and_type():
    facts = [
        CompiledFact(
            fact_id="retrieval.embedding_weight",
            fact_type="decision",
            subject="Embedding retrieval weight",
            fact="The active retrieval weighting records the embedding component as 0.45.",
            source="retrieve.py",
            priority=5.0,
            aliases=["what was decided about the embedding weight"],
        ),
        CompiledFact(
            fact_id="encoding.backends",
            fact_type="factual",
            subject="Encoding backends",
            fact="The available encoding backends are hash, LSH, and embedding.",
            source="encoding.py",
            priority=1.0,
            aliases=["encoding backends"],
        ),
    ]

    result = search_compiled_facts("what was decided about the embedding weight", facts, top_k=1)

    assert result[0][0].fact_id == "retrieval.embedding_weight"
    assert result[0][0].fact.endswith("0.45.")


def test_search_compiled_facts_rejects_type_only_matches():
    facts = [
        CompiledFact(
            fact_id="retrieval.embedding_weight",
            fact_type="decision",
            subject="Embedding retrieval weight",
            fact="The active retrieval weighting records the embedding component as 0.45.",
            source="retrieve.py",
            priority=5.0,
            aliases=["what was decided about the embedding weight"],
        )
    ]

    result = search_compiled_facts("what was the decision about customer knowledge base", facts)

    assert result == []


def test_search_compiled_facts_rejects_generic_benchmark_overlap():
    facts = [
        CompiledFact(
            fact_id="benchmark.latest_retrieval_sweep",
            fact_type="metric",
            subject="Latest retrieval sweep",
            fact="The latest retrieval sweep measured Recall@1 and Recall@5.",
            source="retrieval_sweep.json",
            priority=4.0,
            aliases=["latest retrieval benchmark", "retrieval accuracy"],
        )
    ]

    result = search_compiled_facts("what is the LOCOMO benchmark accuracy", facts)

    assert result == []


def test_search_compiled_facts_prefers_current_truth_by_default():
    facts = [
        CompiledFact(
            fact_id="index.current_size",
            fact_type="metric",
            subject="Unified memory index size",
            fact="The current unified memory index contains 7,377 documents.",
            source="index",
            truth_mode="current",
            priority=4.0,
            aliases=["how many documents are in the unified index"],
        ),
        CompiledFact(
            fact_id="index.historical_size",
            fact_type="metric",
            subject="Historical unified index expected size",
            fact="The historical benchmark expected 1,217 documents.",
            source="benchmark",
            truth_mode="benchmark_expected",
            priority=4.0,
            aliases=["historical unified index expected size"],
        ),
    ]

    result = search_compiled_facts("how many documents are in the unified index", facts, top_k=1)

    assert result[0][0].fact_id == "index.current_size"


def test_search_compiled_facts_can_select_benchmark_expected_truth():
    facts = [
        CompiledFact(
            fact_id="index.current_size",
            fact_type="metric",
            subject="Unified memory index size",
            fact="The current unified memory index contains 7,377 documents.",
            source="index",
            truth_mode="current",
            priority=4.0,
            aliases=["how many documents are in the unified index"],
        ),
        CompiledFact(
            fact_id="index.historical_size",
            fact_type="metric",
            subject="Historical March unified index expected size",
            fact="The historical March benchmark expected 1,217 documents.",
            source="benchmark",
            truth_mode="benchmark_expected",
            priority=4.0,
            aliases=["historical March unified index benchmark documents"],
        ),
    ]

    result = search_compiled_facts(
        "how many documents were expected by the historical March unified index benchmark",
        facts,
        top_k=1,
    )

    assert result[0][0].fact_id == "index.historical_size"


def test_current_operational_queries_use_live_stats(tmp_path):
    repo = tmp_path
    _write_index(repo / "snn_state" / "memory_index.json.gz")
    _write_performance_report(
        repo
        / ".coordination"
        / "benchmarks"
        / "REMANENTIA"
        / "remanentia_performance_2026-04-28_172825.json"
    )

    queries = current_operational_queries(repo)
    combined_gold = " ".join(g for query in queries for g in query["gold"])

    assert "1" in combined_gold
    assert "2" in combined_gold
    assert "841.245" in combined_gold
    assert "95.347" in combined_gold
    assert "vector_worker" in combined_gold
    assert "1,103" in combined_gold


def test_historical_regression_queries_are_explicitly_historical():
    queries = historical_regression_queries()

    assert queries
    assert all(query["suite"] == "historical_regression" for query in queries)
    assert all("historical" in query["q"] for query in queries)


def test_memory_index_search_merges_compiled_facts(monkeypatch):
    idx = MemoryIndex()
    idx._built = True
    idx.paragraph_index = [(0, 0)] * 1001

    def compiled_result(query: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                name="retrieval.embedding_weight.fact",
                source="compiled",
                score=1005.0,
                snippet="The active retrieval weighting records the embedding component as 0.45.",
                answer="The active retrieval weighting records the embedding component as 0.45.",
                confidence=1.0,
            )
        ]

    monkeypatch.setattr(memory_index, "_compiled_fact_results", compiled_result)

    results = idx.search("what was decided about the embedding weight", top_k=1)

    assert results[0].source == "compiled"
    assert "0.45" in results[0].answer
