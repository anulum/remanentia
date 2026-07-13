# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real compiled-fact tests for unified index models

"""Exercise index records and priority facts through real compiled JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from compiled_memory import CompiledFact, write_compiled_facts
from REMANENTIA.memory_index_models import (  # type: ignore[import]
    Document,
    Paragraph,
    SearchResult,
    compiled_fact_results,
    has_operational_compiled_memory,
    merge_priority_results,
)


@dataclass
class IndexState:
    """Concrete state implementing the production operational-index protocol."""

    paragraph_index: list[tuple[int, int]] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


def test_index_records_have_safe_independent_defaults() -> None:
    """Mutable record defaults are independent across concrete instances."""
    first = Document("first.md", "traces", "/first.md")
    second = Document("second.md", "notes", "/second.md")
    first.paragraphs.append("fact")
    paragraph = Paragraph("text", para_type="finding")
    result = SearchResult("first.md", "traces", 1.5, "fact")

    assert second.paragraphs == []
    assert first.tokens == set()
    assert first.embedding is None
    assert first.date == ""
    assert first.doc_type == ""
    assert paragraph.prospective_queries == []
    assert result.paragraph_idx == 0
    assert result.answer == ""
    assert result.confidence == 0.0


def test_compiled_fact_results_traverse_real_jsonl_loader_and_scorer(tmp_path: Path) -> None:
    """Priority formatting uses durable facts and the production search algorithm."""
    low = CompiledFact(
        fact_id="worker.refresh",
        fact_type="continuity",
        subject="Worker refresh",
        fact="The vector worker refreshes an index.",
        source="ops.md",
    )
    priority = CompiledFact(
        fact_id="worker.vector",
        fact_type="continuity",
        subject="vector worker",
        fact="The vector worker refreshes durable memory indexes.",
        source="ops.md",
    )
    compiled_dir = tmp_path / "compiled"
    write_compiled_facts([low, priority], compiled_dir)

    results = compiled_fact_results("vector worker", 5, compiled_dir / "facts.jsonl")

    assert results == [
        SearchResult(
            name="worker.vector.fact",
            source="compiled",
            score=1009.0,
            snippet=priority.fact,
            answer=priority.fact,
            confidence=1.0,
        )
    ]
    assert compiled_fact_results("the", 5, compiled_dir / "facts.jsonl") == []


def test_merge_priority_results_deduplicates_and_honors_limit() -> None:
    """Priority records win while distinct ranked records fill the requested limit."""
    priority = SearchResult("a.md", "compiled", 10.0, "same")
    duplicate = SearchResult("a.md", "compiled", 8.0, "same")
    ranked = SearchResult("b.md", "notes", 2.0, "other", answer="answer")
    extra = SearchResult("c.md", "notes", 1.0, "extra")

    merged = merge_priority_results([priority], [duplicate, ranked, extra], 2)

    assert merged == [priority, ranked]
    assert merge_priority_results([], [extra], 5) == [extra]


def test_operational_compiled_memory_accepts_materialized_or_compiled_state() -> None:
    """Both documented readiness modes are evaluated on concrete index state."""
    assert has_operational_compiled_memory(IndexState()) is False
    assert (
        has_operational_compiled_memory(
            IndexState(documents=[Document("facts.md", "compiled", "/facts.md")])
        )
        is True
    )
    assert has_operational_compiled_memory(IndexState(paragraph_index=[(0, 0)] * 1001)) is True
