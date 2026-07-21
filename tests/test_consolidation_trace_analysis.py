# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Native-independent consolidation trace tests

"""Exercise portable trace analysis directly without import interception."""

from __future__ import annotations

from consolidation_trace_analysis import (
    cluster_traces,
    cluster_traces_python,
    extract_entities,
    extract_entities_python,
    extract_key_lines,
    extract_key_lines_python,
    extract_paragraphs,
)


def test_portable_entity_extraction_handles_short_identifiers_and_paths() -> None:
    """Concrete text covers accepted file paths and rejected short snake names."""
    text = (
        "The fix touched a.py and compute_order_parameter while a_b_c in "
        "ArcaneNeuron used BM25 v1.2.3 at 91%."
    )
    entities = extract_entities_python(text)

    assert {
        "a.py",
        "compute_order_parameter",
        "ArcaneNeuron",
        "bm25",
        "v1.2.3",
        "91%",
    } <= set(entities)
    assert "a_b_c" not in entities
    assert extract_entities(text) == entities


def test_portable_paragraph_and_key_line_filters_use_real_text() -> None:
    """Blank/short blocks and short triggered lines do not become semantic facts."""
    text = "\n\n# Header\n\nshort\n\nA paragraph long enough to become durable semantic memory content."
    assert extract_paragraphs(text) == [
        "A paragraph long enough to become durable semantic memory content."
    ]

    key_text = (
        "fix tiny\n"
        "The critical production finding is long enough to retain.\n"
        "# heading is not context\n"
        "x\n"
    )
    expected = ["The critical production finding is long enough to retain."]
    assert extract_key_lines_python(key_text) == expected
    assert extract_key_lines(key_text) == expected


def test_portable_clustering_handles_an_empty_real_trace_mapping() -> None:
    """Empty trace discovery yields no synthetic cluster."""
    assert cluster_traces_python({}) == []
    assert cluster_traces({}) == []
