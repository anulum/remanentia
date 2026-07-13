# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Sparse BM25 scoring core

"""Typed sparse BM25 construction and scoring for the unified memory index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

import numpy as np
from numpy.typing import NDArray

K1 = 1.5
B = 0.75
FloatArray = NDArray[np.float32]
WeightIndex = dict[str, list[tuple[int, float]]]


def _document_length_ratio(paragraph_length: float, average_length: float) -> float:
    """Return a safe paragraph-to-corpus length ratio."""
    denominator = average_length if average_length > 0 else 1.0
    return paragraph_length / denominator


def _bm25_weight(idf: float, term_frequency: int, length_ratio: float) -> float:
    """Compute one BM25 term contribution."""
    return idf * term_frequency * (K1 + 1) / (term_frequency + K1 * (1 - B + B * length_ratio))


def build_weight_index(
    inverted_index: Mapping[str, Sequence[int]],
    idf: Mapping[str, float],
    token_counts: Sequence[Mapping[str, int]],
    paragraph_lengths: FloatArray,
    average_length: float,
) -> WeightIndex:
    """Precompute BM25 weights for every valid sparse posting."""
    weights: WeightIndex = {}
    for token, posting in inverted_index.items():
        idf_value = idf.get(token, 0.0)
        if idf_value == 0:
            continue
        weighted_posting: list[tuple[int, float]] = []
        for paragraph_index in posting:
            if paragraph_index < 0 or paragraph_index >= len(paragraph_lengths):
                continue
            term_frequency = (
                token_counts[paragraph_index].get(token, 1)
                if paragraph_index < len(token_counts)
                else 1
            )
            ratio = _document_length_ratio(
                float(paragraph_lengths[paragraph_index]), average_length
            )
            weighted_posting.append(
                (paragraph_index, _bm25_weight(idf_value, term_frequency, ratio))
            )
        if weighted_posting:
            weights[token] = weighted_posting
    return weights


def score_weight_index(
    query_tokens: Set[str],
    filtered_paragraphs: Set[int],
    weight_index: Mapping[str, Sequence[tuple[int, float]]],
) -> dict[int, float]:
    """Accumulate precomputed sparse weights for unfiltered query postings."""
    scores: dict[int, float] = {}
    for token in query_tokens:
        for paragraph_index, weight in weight_index.get(token, ()):
            if paragraph_index in filtered_paragraphs:
                continue
            scores[paragraph_index] = scores.get(paragraph_index, 0.0) + weight
    return scores


def score_uncached(
    query_tokens: Set[str],
    filtered_paragraphs: Set[int],
    inverted_index: Mapping[str, Sequence[int]],
    idf: Mapping[str, float],
    token_counts: Sequence[Mapping[str, int]],
    paragraph_lengths: FloatArray,
    average_length: float,
) -> dict[int, float]:
    """Compute BM25 directly from sparse postings without a weight cache."""
    scores: dict[int, float] = {}
    for token in query_tokens:
        posting = inverted_index.get(token)
        if not posting:
            continue
        idf_value = idf.get(token, 0.0)
        if idf_value == 0:
            continue
        for paragraph_index in posting:
            if (
                paragraph_index in filtered_paragraphs
                or paragraph_index < 0
                or paragraph_index >= len(paragraph_lengths)
            ):
                continue
            term_frequency = (
                token_counts[paragraph_index].get(token, 1)
                if paragraph_index < len(token_counts)
                else 1
            )
            ratio = _document_length_ratio(
                float(paragraph_lengths[paragraph_index]), average_length
            )
            scores[paragraph_index] = scores.get(paragraph_index, 0.0) + _bm25_weight(
                idf_value, term_frequency, ratio
            )
    return scores
