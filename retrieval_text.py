# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Text features for query-probe retrieval

"""Text normalization and TF-IDF scoring for query-probe retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from typing import cast

STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "it",
        "by",
        "as",
        "with",
        "from",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "this",
        "that",
        "these",
        "those",
        "are",
        "not",
        "no",
        "its",
        "into",
        "can",
        "will",
        "would",
        "should",
        "could",
        "may",
        "also",
        "so",
        "if",
        "when",
        "then",
        "than",
        "more",
        "most",
        "all",
        "any",
        "each",
        "every",
        "both",
        "few",
        "many",
        "much",
        "some",
        "such",
        "only",
        "just",
        "about",
        "over",
        "after",
        "before",
        "between",
        "through",
        "during",
        "up",
        "down",
        "out",
        "off",
        "did",
        "do",
        "does",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "why",
        "here",
        "there",
        "their",
        "them",
        "they",
        "we",
        "our",
        "us",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "i",
        "me",
        "my",
        "being",
        "now",
        "very",
    }
)

STEM_SUFFIXES: tuple[str, ...] = (
    "ation",
    "tion",
    "sion",
    "meant",
    "ness",
    "ity",
    "ous",
    "ive",
    "ing",
    "ical",
    "ally",
    "able",
    "ible",
    "full",
    "less",
    "ized",
    "ise",
    "ize",
    "ed",
    "ly",
    "er",
    "est",
    "al",
    "es",
    "s",
)


def python_tokenize(text: str, stopwords: Set[str] = STOPWORDS) -> list[str]:
    """Return lowercase alphanumeric tokens without stopwords or one-letter terms."""
    return [
        word
        for word in re.findall(r"[a-z0-9_]+", text.lower())
        if word not in stopwords and len(word) > 1
    ]


def tokenize(text: str, stopwords: Set[str] = STOPWORDS) -> list[str]:
    """Tokenize with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import tokenize as rust_tokenize  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_tokenize(text, stopwords)
    return cast(list[str], rust_tokenize(text, stopwords))


def python_stem(word: str) -> str:
    """Strip the first recognized suffix while retaining a three-letter stem."""
    for suffix in STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def stem(word: str) -> str:
    """Stem with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import stem as rust_stem
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_stem(word)
    return cast(str, rust_stem(word))


def python_expand_query(query: str, stopwords: Set[str] = STOPWORDS) -> str:
    """Append missing stems to a query for broader lexical matching."""
    tokens = python_tokenize(query, stopwords)
    extra = {python_stem(token) for token in tokens} - set(tokens)
    if not extra:
        return query
    return query + " " + " ".join(sorted(extra))


def expand_query(query: str, stopwords: Set[str] = STOPWORDS) -> str:
    """Expand a query with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import expand_query as rust_expand_query
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_expand_query(query, stopwords)
    return cast(str, rust_expand_query(query, stopwords))


def python_bigrams(tokens: Sequence[str]) -> list[str]:
    """Return adjacent token pairs joined with an underscore."""
    return [f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)]


def bigrams(tokens: list[str]) -> list[str]:
    """Generate bigrams with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import bigrams as rust_bigrams
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_bigrams(tokens)
    return cast(list[str], rust_bigrams(tokens))


def python_build_idf(
    trace_texts: Mapping[str, str], stopwords: Set[str] = STOPWORDS
) -> dict[str, float]:
    """Compute smoothed inverse document frequency for unigrams and bigrams."""
    document_frequency: Counter[str] = Counter()
    for name, text in trace_texts.items():
        tokens = python_tokenize(text + " " + name.replace("-", " ").replace("_", " "), stopwords)
        document_frequency.update(set(tokens) | set(python_bigrams(tokens)))
    document_count = len(trace_texts)
    return {
        term: math.log(1 + document_count / (1 + count))
        for term, count in document_frequency.items()
    }


def build_idf(trace_texts: dict[str, str], stopwords: Set[str] = STOPWORDS) -> dict[str, float]:
    """Build IDF weights with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import build_idf as rust_build_idf
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_build_idf(trace_texts, stopwords)
    return cast(dict[str, float], rust_build_idf(trace_texts, stopwords))


def python_tfidf_score(
    query: str,
    document_name: str,
    document_text: str,
    idf: Mapping[str, float],
    stopwords: Set[str] = STOPWORDS,
) -> float:
    """Score query overlap using sublinear TF, bigrams, and filename boosting."""
    query_tokens = python_tokenize(query, stopwords)
    if not query_tokens:
        return 0.0
    query_terms = set(query_tokens) | set(python_bigrams(query_tokens))

    name_tokens = python_tokenize(document_name.replace("-", " ").replace("_", " "), stopwords)
    document_tokens = python_tokenize(document_text, stopwords)
    document_tf: Counter[str] = Counter(document_tokens + name_tokens * 3)
    document_tf.update(python_bigrams(document_tokens))
    for term in python_bigrams(name_tokens):
        document_tf[term] += 3

    score = sum(
        (1.0 + math.log(document_tf[term])) * idf.get(term, 0.0)
        for term in query_terms
        if term in document_tf
    )
    return score / len(query_terms)


def tfidf_score(
    query: str,
    document_name: str,
    document_text: str,
    idf: dict[str, float],
    stopwords: Set[str] = STOPWORDS,
) -> float:
    """Score TF-IDF with the compiled implementation when it is installed."""
    try:
        from remanentia_retrieve import tfidf_score as rust_tfidf_score
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return python_tfidf_score(query, document_name, document_text, idf, stopwords)
    return cast(
        float,
        rust_tfidf_score(query, document_name, document_text, idf, stopwords),
    )
