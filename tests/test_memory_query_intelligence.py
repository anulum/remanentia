# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Native-independent query intelligence tests

"""Exercise explicit production implementations without import interception."""

from __future__ import annotations

from memory_query_intelligence import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_python,
    tokenize,
    tokenize_python,
)


def test_explicit_tokenizer_preserves_real_bm25_terms() -> None:
    """The portable production tokenizer handles identifiers and numbers."""
    expected = ["native", "free", "bm25_tokenizer", "2026"]
    assert tokenize_python("A native-free BM25_tokenizer in 2026") == expected
    assert tokenize("A native-free BM25_tokenizer in 2026") == expected


def test_explicit_rrf_fuses_concrete_rankings() -> None:
    """The portable production RRF ranks corroborated paragraphs first."""
    rankings = [[(0, 5.0), (1, 3.0)], [(1, 8.0), (2, 1.0)]]
    result = reciprocal_rank_fusion_python(rankings)

    assert result[0][0] == 1
    assert {paragraph for paragraph, _score in result} == {0, 1, 2}
    assert reciprocal_rank_fusion_python([]) == []
    assert reciprocal_rank_fusion(rankings) == result
