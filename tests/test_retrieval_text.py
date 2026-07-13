# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface tests for retrieval text features

"""Exercise production retrieval text functions without behavior substitution."""

from __future__ import annotations

import pytest

from REMANENTIA.retrieval_text import (
    bigrams,
    build_idf,
    expand_query,
    python_bigrams,
    python_build_idf,
    python_expand_query,
    python_stem,
    python_tfidf_score,
    python_tokenize,
    stem,
    tfidf_score,
    tokenize,
)


def test_tokenization_and_bigrams_preserve_retrieval_terms() -> None:
    """The production tokenizer removes noise and retains technical identifiers."""
    text = "The SNN-control loop uses BM25_v2 in 2026."
    expected = ["snn", "control", "loop", "uses", "bm25_v2", "2026"]

    assert python_tokenize(text) == expected
    assert tokenize(text) == expected
    assert python_bigrams(expected) == [
        "snn_control",
        "control_loop",
        "loop_uses",
        "uses_bm25_v2",
        "bm25_v2_2026",
    ]
    assert bigrams(expected) == python_bigrams(expected)
    assert bigrams([]) == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [("saturation", "satur"), ("running", "runn"), ("glass", "glas"), ("is", "is")],
)
def test_stemming_matches_the_production_contract(word: str, expected: str) -> None:
    """Both dispatch and explicit portable implementations honor suffix rules."""
    assert python_stem(word) == expected
    assert stem(word) == expected


def test_query_expansion_adds_only_missing_stems() -> None:
    """Expansion retains the original query and appends deterministic new stems."""
    query = "gyrokinetic transport saturation"
    expected = "gyrokinetic transport saturation satur"

    assert python_expand_query(query) == expected
    assert expand_query(query) == expected
    assert python_expand_query("plasma") == "plasma"


def test_idf_and_tfidf_rank_a_real_trace_corpus() -> None:
    """The production lexical pipeline ranks the trace containing the queried fact first."""
    traces = {
        "plasma-control.md": "Tokamak disruption control uses resonant magnetic perturbation.",
        "cooking-notes.md": "Bread fermentation uses flour water yeast and salt.",
        "network-memory.md": "Persistent memory retrieval combines lexical and neural signals.",
    }
    query = "tokamak disruption control"

    idf = build_idf(traces)
    scores = {name: tfidf_score(query, name, text, idf) for name, text in traces.items()}

    assert max(scores, key=scores.__getitem__) == "plasma-control.md"
    assert scores["plasma-control.md"] > 0.0
    assert scores["cooking-notes.md"] == 0.0
    assert tfidf_score("the and", "plasma.md", traces["plasma-control.md"], idf) == 0.0


def test_filename_terms_receive_the_declared_boost() -> None:
    """A filename match outweighs a single body-only occurrence for equal IDF."""
    idf = {"tokamak": 1.0}

    filename_score = python_tfidf_score("tokamak", "tokamak.md", "unrelated text", idf)
    body_score = python_tfidf_score("tokamak", "notes.md", "tokamak", idf)

    assert filename_score > body_score


def test_dispatch_matches_portable_scoring() -> None:
    """Runtime dispatch and the directly callable portable path remain equivalent."""
    traces = {
        "alpha-memory.md": "alpha alpha beta",
        "gamma-memory.md": "gamma delta",
    }
    portable_idf = python_build_idf(traces)
    dispatched_idf = build_idf(traces)

    assert dispatched_idf == pytest.approx(portable_idf)
    assert tfidf_score(
        "alpha beta", "alpha-memory.md", traces["alpha-memory.md"], dispatched_idf
    ) == pytest.approx(
        python_tfidf_score(
            "alpha beta",
            "alpha-memory.md",
            traces["alpha-memory.md"],
            portable_idf,
        )
    )
