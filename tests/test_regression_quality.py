# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval Quality Regression Tests

"""Regression harness for retrieval quality on the shipping path.

Uses synthetic conversations with known answers to catch quality
regressions across code changes. Runs without external data or GPU.
"""
from __future__ import annotations

import math
import re
import time

import pytest

from memory_index import MemoryIndex, _tokenize


CONVERSATIONS = [
    {
        "turns": [
            "Alice said she loves pottery and hiking on weekends.",
            "Bob mentioned he works as a data scientist at Google.",
            "Alice told Bob she started learning piano last month.",
            "They discussed the weather forecast for next week.",
            "Bob said his favorite movie is The Matrix.",
            "Alice mentioned she is allergic to cats.",
            "They talked about the new Italian restaurant downtown.",
            "Bob said he just got back from a trip to Japan.",
            "Alice said she volunteers at the local animal shelter.",
            "Bob mentioned he has been learning Rust programming.",
        ],
        "questions": [
            ("What are Alice's hobbies?", "pottery", "single-hop"),
            ("Where does Bob work?", "Google", "single-hop"),
            ("What instrument is Alice learning?", "piano", "single-hop"),
            ("What is Bob's favorite movie?", "Matrix", "single-hop"),
            ("What is Alice allergic to?", "cats", "single-hop"),
            ("Where did Bob go on his trip?", "Japan", "single-hop"),
            ("What programming language is Bob learning?", "Rust", "single-hop"),
        ],
    },
    {
        "turns": [
            "On 2026-03-01, the team deployed version 3.0 of the API.",
            "Performance testing on 2026-03-05 showed 200ms latency.",
            "A critical bug was found on 2026-03-10 in the auth module.",
            "The fix was deployed on 2026-03-12 after code review.",
            "On 2026-03-15 the team started working on the dashboard.",
            "The dashboard prototype was shown on 2026-03-20.",
        ],
        "questions": [
            ("When was version 3.0 deployed?", "2026-03-01", "temporal"),
            ("When was the critical bug found?", "2026-03-10", "temporal"),
            ("What was the latency in performance testing?", "200ms", "metric"),
        ],
    },
    {
        "turns": [
            "The STDP learning rule was set with weight 0.0 because experiments showed no signal.",
            "Best-paragraph embedding achieves 85.7% precision at rank 1.",
            "TF-IDF combined with embedding at 0.4/0.6 ratio gives 92.9% P@1.",
            "The SNN daemon runs 20,000 LIF neurons for consolidation.",
            "Entity graph has 283 entities and 9,799 relations.",
            "Cross-encoder reranking uses ms-marco-MiniLM-L-6-v2.",
        ],
        "questions": [
            ("What precision does best-paragraph embedding achieve?", "85.7%", "metric"),
            ("How many neurons does the SNN daemon run?", "20,000", "metric"),
            ("What is the TF-IDF to embedding ratio?", "0.4/0.6", "metric"),
        ],
    },
]


@pytest.fixture(scope="module")
def loaded_index(tmp_path_factory):
    """Build an index from the test conversations."""
    tmp = tmp_path_factory.mktemp("regression")
    idx = MemoryIndex()

    for ci, conv in enumerate(CONVERSATIONS):
        for ti, turn in enumerate(conv["turns"]):
            f = tmp / f"conv{ci}_turn{ti}.md"
            f.write_text(turn, encoding="utf-8")

    idx.build(use_gpu_embeddings=False, use_gliner=False)

    # Override sources to point at our tmp files
    idx.documents = []
    idx.paragraph_index = []
    idx.paragraph_tokens = []
    idx.paragraph_types = []
    idx._inverted_index = {}

    from memory_index import _split_paragraphs, _tokenize, _classify_paragraph, _generate_prospective_queries
    from collections import Counter
    import numpy as np

    for ci, conv in enumerate(CONVERSATIONS):
        for ti, turn in enumerate(conv["turns"]):
            doc_idx = len(idx.documents)
            from memory_index import Document
            doc = Document(name=f"conv{ci}_turn{ti}", source="test",
                           path=str(tmp / f"conv{ci}_turn{ti}.md"),
                           paragraphs=[turn])
            idx.documents.append(doc)
            p_idx = len(idx.paragraph_tokens)
            idx.paragraph_index.append((doc_idx, 0))
            tokens = set(_tokenize(turn))
            idx.paragraph_tokens.append(tokens)
            idx.paragraph_types.append("discussion")
            for t in tokens:
                if t not in idx._inverted_index:
                    idx._inverted_index[t] = []
                idx._inverted_index[t].append(p_idx)

    n_docs = len(idx.paragraph_tokens)
    df = Counter()
    for tokens in idx.paragraph_tokens:
        for t in tokens:
            df[t] += 1
    idx.idf = {t: math.log(1 + n_docs / (1 + c)) for t, c in df.items()}
    idx._para_lengths = np.array([len(t) for t in idx.paragraph_tokens], dtype=np.float32)
    idx._avg_dl = float(np.mean(idx._para_lengths))
    idx._built = True

    return idx


class TestRetrievalQuality:
    """Regression tests: each query must find its answer in top-3 results."""

    def _check_query(self, idx, question, expected_answer, category):
        results = idx.search(question, top_k=3)
        assert results, f"No results for: {question}"
        found = False
        for r in results:
            if expected_answer.lower() in r.snippet.lower():
                found = True
                break
            if r.answer and expected_answer.lower() in r.answer.lower():
                found = True
                break
        return found

    @pytest.mark.parametrize("ci,qi", [
        (ci, qi)
        for ci, conv in enumerate(CONVERSATIONS)
        for qi in range(len(conv["questions"]))
    ])
    def test_retrieval_quality(self, loaded_index, ci, qi):
        conv = CONVERSATIONS[ci]
        question, expected, category = conv["questions"][qi]
        found = self._check_query(loaded_index, question, expected, category)
        assert found, f"[{category}] '{question}' — expected '{expected}' in top-3"


class TestSearchPerformance:
    """Warm search must stay under 50ms."""

    def test_warm_search_latency(self, loaded_index):
        # Warmup
        loaded_index.search("warmup query", top_k=3)

        t0 = time.monotonic()
        for _ in range(50):
            loaded_index.search("What are Alice's hobbies?", top_k=3)
        avg_ms = (time.monotonic() - t0) / 50 * 1000
        assert avg_ms < 50, f"Average search latency {avg_ms:.1f}ms exceeds 50ms budget"

    def test_inverted_index_populated(self, loaded_index):
        assert len(loaded_index._inverted_index) > 0
        assert len(loaded_index._para_lengths) == len(loaded_index.paragraph_tokens)
