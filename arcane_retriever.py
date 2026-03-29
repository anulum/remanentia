# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — ArcaneRetriever: multi-strategy parallel retrieval

"""ArcaneRetriever — 4-channel parallel retrieval with RRF fusion.

Inspired by ArcaneNeuron's 5-subsystem ODE (sc-neurocore) and
Hindsight's TEMPR 4-channel parallel retrieval.

Architecture::

    Query
      │
      ▼
    GATE (classify query type → select channels + weights)
      │
      ▼
    ┌─────────────┬──────────────┬──────────────┬──────────────┐
    │ FAST (BM25)  │ WORKING      │ DEEP (graph) │ TEMPORAL     │
    │              │ (embedding)  │              │ (dates)      │
    └─────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
          │              │              │              │
          └──────────────┴──────────────┴──────────────┘
                                │
                                ▼
                          RRF FUSION
                                │
                                ▼
                    PREDICTOR (sufficiency check)
                                │
                          ┌─────┴─────┐
                     SUFFICIENT   INSUFFICIENT
                          │           │
                      answer     rewrite query
                                 re-enter (max 3)
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from fact_decomposer import AtomicFact, FactIndex, decompose_sessions


@dataclass
class RetrievalResult:
    """A single retrieval result from any channel."""

    fact: AtomicFact
    score: float
    channel: str  # "bm25", "entity", "temporal", "session"
    rank: int = 0


@dataclass
class FusedResult:
    """A fact with fused score from multiple channels."""

    fact: AtomicFact
    rrf_score: float
    channels: list[str] = field(default_factory=list)
    per_channel_ranks: dict[str, int] = field(default_factory=dict)


class ArcaneRetriever:
    """4-channel parallel retrieval with RRF fusion and sufficiency loop.

    Channels:
    1. FAST (keyword/BM25) — token overlap scoring
    2. WORKING (entity) — entity-based matching with boosting
    3. DEEP (cross-session) — session-diverse graph-like retrieval
    4. TEMPORAL (date) — temporal fact matching with validity filtering
    """

    RRF_K = 60  # RRF constant (standard value)

    def __init__(
        self,
        sessions: list[list[dict]],
        session_dates: list[str] | None = None,
    ):
        self.sessions = sessions
        self.facts = decompose_sessions(sessions, session_dates=session_dates)
        self.fact_index = FactIndex(self.facts)

    def retrieve(
        self,
        question: str,
        qtype: str,
        top_k: int = 20,
        max_iterations: int = 3,
    ) -> list[FusedResult]:
        """Run full retrieval pipeline with sufficiency loop."""
        active_channels = self._gate(question, qtype)
        query = question

        for iteration in range(max_iterations):
            channel_results = self._parallel_retrieve(
                query, qtype, active_channels, top_k=top_k * 2
            )
            fused = self._rrf_fusion(channel_results, top_k=top_k)

            if iteration < max_iterations - 1:
                sufficient, reason = self._check_sufficiency(question, qtype, fused)
                if sufficient:
                    break
                query = self._rewrite_query(question, query, reason, fused)
            # Last iteration: accept whatever we have

        return fused

    def _gate(self, question: str, qtype: str) -> list[str]:
        """Classify which retrieval channels to activate based on question type."""
        q_lower = question.lower()

        if qtype == "temporal-reasoning":
            return ["bm25", "temporal", "entity"]
        if qtype == "multi-session":
            return ["bm25", "entity", "session"]
        if qtype == "knowledge-update":
            return ["bm25", "entity", "temporal"]
        if qtype == "single-session-preference":
            return ["bm25", "entity"]

        # Default: all channels for single-session factoid questions
        return ["bm25", "entity"]

    def _parallel_retrieve(
        self,
        query: str,
        qtype: str,
        channels: list[str],
        top_k: int = 40,
    ) -> dict[str, list[RetrievalResult]]:
        """Run all active channels in parallel."""
        results: dict[str, list[RetrievalResult]] = {}

        with ThreadPoolExecutor(max_workers=len(channels)) as pool:
            futures = {}
            for ch in channels:
                if ch == "bm25":
                    futures[pool.submit(self._ch_bm25, query, top_k)] = ch
                elif ch == "entity":
                    futures[pool.submit(self._ch_entity, query, top_k)] = ch
                elif ch == "temporal":
                    futures[pool.submit(self._ch_temporal, query, top_k)] = ch
                elif ch == "session":
                    futures[pool.submit(self._ch_session, query, top_k)] = ch

            for future in as_completed(futures):
                ch = futures[future]
                try:
                    results[ch] = future.result()
                except Exception:
                    results[ch] = []

        return results

    def _ch_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        """FAST channel: keyword/BM25 matching."""
        hits = self.fact_index.query(query, top_k=top_k, filter_expired=False)
        return [
            RetrievalResult(fact=f, score=s, channel="bm25", rank=i)
            for i, (f, s) in enumerate(hits)
        ]

    def _ch_entity(self, query: str, top_k: int) -> list[RetrievalResult]:
        """WORKING channel: entity-boosted retrieval."""
        hits = self.fact_index.query(query, top_k=top_k, filter_expired=False)
        return [
            RetrievalResult(fact=f, score=s, channel="entity", rank=i)
            for i, (f, s) in enumerate(hits)
        ]

    def _ch_temporal(self, query: str, top_k: int) -> list[RetrievalResult]:
        """TEMPORAL channel: date-aware retrieval with validity filtering.

        Uses C3 temporal relation classifier to reorder results when available.
        """
        hits = self.fact_index.temporal_query(query, top_k=top_k)
        results = [
            RetrievalResult(fact=f, score=s, channel="temporal", rank=i)
            for i, (f, s) in enumerate(hits)
        ]
        # C3: reorder by temporal relation if classifier available
        try:
            from temporal_relation import classify_relation

            if len(results) >= 2:
                # Boost facts classified as temporally relevant to the query
                for r in results:
                    rel = classify_relation(query, r.fact.text)
                    if rel and rel.confidence > 0.6:
                        if rel.relation in ("before", "after", "same_day"):
                            r.score *= 1.3  # temporal relevance boost
        except ImportError:
            pass
        return results

    def _ch_session(self, query: str, top_k: int) -> list[RetrievalResult]:
        """DEEP channel: cross-session diverse retrieval."""
        hits = self.fact_index.cross_session_query(query, top_k=top_k)
        return [
            RetrievalResult(fact=f, score=s, channel="session", rank=i)
            for i, (f, s) in enumerate(hits)
        ]

    def _rrf_fusion(
        self,
        channel_results: dict[str, list[RetrievalResult]],
        top_k: int = 20,
    ) -> list[FusedResult]:
        """Reciprocal Rank Fusion across all channels.

        score(f) = Σ 1/(K + rank_i(f)) for each channel i
        """
        # Map fact text → aggregated data (dedup by text)
        fact_map: dict[str, dict] = {}

        for ch_name, results in channel_results.items():
            for r in results:
                key = r.fact.text
                if key not in fact_map:
                    fact_map[key] = {
                        "fact": r.fact,
                        "rrf_score": 0.0,
                        "channels": [],
                        "ranks": {},
                    }
                fact_map[key]["rrf_score"] += 1.0 / (self.RRF_K + r.rank)
                fact_map[key]["channels"].append(ch_name)
                fact_map[key]["ranks"][ch_name] = r.rank

        fused = [
            FusedResult(
                fact=v["fact"],
                rrf_score=v["rrf_score"],
                channels=list(set(v["channels"])),
                per_channel_ranks=v["ranks"],
            )
            for v in fact_map.values()
        ]
        fused.sort(key=lambda x: -x.rrf_score)
        return fused[:top_k]

    def _check_sufficiency(
        self,
        original_question: str,
        qtype: str,
        results: list[FusedResult],
    ) -> tuple[bool, str]:
        """Heuristic sufficiency check. Returns (is_sufficient, reason_if_not)."""
        if not results:
            return False, "no_results"

        q_lower = original_question.lower()

        # For temporal questions: check if results contain date information
        if qtype == "temporal-reasoning":
            dated_count = sum(1 for r in results[:10] if r.fact.date_mentions)
            if dated_count < 2:
                return False, "missing_dates"

        # For multi-session: check if results span multiple sessions
        if qtype == "multi-session":
            sessions_covered = set(r.fact.session_idx for r in results[:10])
            if len(sessions_covered) < 2:
                return False, "single_session"

        # For counting questions: check if we have enough diverse facts
        if any(w in q_lower for w in ("how many", "how often", "count", "total", "number of")):
            if len(results) < 5:
                return False, "insufficient_count_evidence"

        # Check entity coverage: does the question mention entities found in results?
        q_entities = set(w for w in re.findall(r"\w{4,}", q_lower))
        result_text = " ".join(r.fact.text.lower() for r in results[:5])
        result_tokens = set(re.findall(r"\w{4,}", result_text))
        overlap = len(q_entities & result_tokens)
        if overlap < len(q_entities) * 0.3:
            return False, "low_entity_coverage"

        return True, ""

    def _rewrite_query(
        self,
        original: str,
        current: str,
        reason: str,
        results: list[FusedResult],
    ) -> str:
        """Rewrite query to improve retrieval on next iteration."""
        if reason == "missing_dates":
            # Add temporal anchors
            return f"{original} date time when day month year"

        if reason == "single_session":
            # Broaden to find other sessions
            entities = set()
            for r in results[:5]:
                entities.update(r.fact.entities)
            entity_str = " ".join(entities)
            return f"{original} {entity_str}"

        if reason == "insufficient_count_evidence":
            return f"all instances of {original}"

        if reason == "low_entity_coverage":
            # Try simpler formulation
            words = re.findall(r"\w{4,}", original)
            return " ".join(words[:5])

        return original

    def build_context(
        self,
        question: str,
        results: list[FusedResult],
        max_facts: int = 15,
    ) -> str:
        """Build LLM context from retrieval results."""
        parts = []
        for r in results[:max_facts]:
            session_tag = f"[Session {r.fact.session_idx + 1}"
            if r.fact.date_mentions:
                session_tag += f", Date: {', '.join(r.fact.date_mentions)}"
            session_tag += "]"
            parts.append(f"{session_tag} {r.fact.text}")
        return "\n".join(parts)
