# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — ArcaneRetriever multi-strategy parallel retrieval

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

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any
from datetime import date

from fact_decomposer import AtomicFact, FactIndex, decompose_sessions
from context_builder import build_hierarchical_context


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
    CHANNEL_WEIGHTS = {"bm25": 1.0, "entity": 1.0, "temporal": 1.0, "session": 2.0}
    DEFAULT_DECAY_HALF_LIFE_DAYS = 30  # recency half-life in days

    def __init__(
        self,
        sessions: list[list[dict[str, str]]],
        session_dates: list[str] | None = None,
        recency_half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS,
        reference_date: str | None = None,
    ):
        self.sessions = sessions
        self.session_dates = session_dates or []
        self.recency_half_life_days = recency_half_life_days
        self._reference_date = reference_date
        self._reference_date_parsed = self._parse_iso_date(reference_date)
        self._recency_weight_cache: dict[str, float] = {}
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

        reranked = self._cross_encoder_rerank(question, fused, top_n=top_k)
        # Mark top results as accessed to boost confidence
        for r in reranked[:5]:
            r.fact.update_confidence("accessed")
        return reranked[:top_k]

    def _gate(self, question: str, qtype: str) -> list[str]:
        """Classify which retrieval channels to activate based on question type."""
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
        if set(channels).issubset({"bm25", "entity"}):
            hits = self.fact_index.query(query, top_k=top_k, filter_expired=False)
            return {
                ch: [
                    RetrievalResult(fact=f, score=s, channel=ch, rank=i)
                    for i, (f, s) in enumerate(hits)
                ]
                for ch in channels
            }

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

        Ordering comes entirely from ``fact_index.temporal_query`` — the rank
        assigned here is what RRF fusion consumes (fusion is rank-based, so a
        per-result ``score`` is carried for reference but never affects the
        fused order). A future enhancement may re-rank these hits with the C3
        ``temporal_relation`` classifier, but that must re-sort and re-number
        the ranks (not merely scale ``score``) and be benchmarked on the
        temporal-reasoning split before it is enabled — see the debug-campaign
        tracker.
        """
        hits = self.fact_index.temporal_query(query, top_k=top_k)
        return [
            RetrievalResult(fact=f, score=s, channel="temporal", rank=i)
            for i, (f, s) in enumerate(hits)
        ]

    def _ch_session(self, query: str, top_k: int) -> list[RetrievalResult]:
        """DEEP channel: cross-session diverse retrieval."""
        hits = self.fact_index.cross_session_query(query, top_k=top_k)
        return [
            RetrievalResult(fact=f, score=s, channel="session", rank=i)
            for i, (f, s) in enumerate(hits)
        ]

    @staticmethod
    def _parse_iso_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value[:10])
        except (ValueError, TypeError):
            return None

    def _recency_weight(self, fact: AtomicFact) -> float:
        """Compute recency decay weight for a fact.

        Uses exponential decay: weight = 2^(-age_days / half_life).
        Facts without dates get weight 1.0 (no penalty).
        """
        if self.recency_half_life_days <= 0:
            return 1.0

        ref = self._reference_date_parsed
        if ref is None:
            return 1.0

        # Use fact's valid_from, or try session date, or fallback to no decay
        fact_date_str = fact.valid_from
        if not fact_date_str and self.session_dates and fact.session_idx < len(self.session_dates):
            fact_date_str = self.session_dates[fact.session_idx]
        if not fact_date_str:
            return 1.0

        fact_date_key = fact_date_str[:10]
        cached = self._recency_weight_cache.get(fact_date_key)
        if cached is not None:
            return cached

        fact_date = self._parse_iso_date(fact_date_key)
        if fact_date is None:
            return 1.0

        age_days = (ref - fact_date).days
        if age_days <= 0:
            return 1.0  # future or same-day facts: full weight

        weight = math.pow(2.0, -age_days / self.recency_half_life_days)
        self._recency_weight_cache[fact_date_key] = weight
        return weight

    def _rrf_fusion(
        self,
        channel_results: dict[str, list[RetrievalResult]],
        top_k: int = 20,
    ) -> list[FusedResult]:
        """Reciprocal Rank Fusion across all channels with recency decay.

        score(f) = recency_weight(f) * Σ 1/(K + rank_i(f)) for each channel i

        Recency weight uses exponential decay with configurable half-life.
        """
        # Map fact text → aggregated data (dedup by text)
        fact_map: dict[str, dict[str, Any]] = {}

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
                fact_map[key]["rrf_score"] += self.CHANNEL_WEIGHTS.get(ch_name, 1.0) / (
                    self.RRF_K + r.rank
                )
                fact_map[key]["channels"].append(ch_name)
                fact_map[key]["ranks"][ch_name] = r.rank

        fused = []
        for v in fact_map.values():
            recency = self._recency_weight(v["fact"])
            fused.append(
                FusedResult(
                    fact=v["fact"],
                    rrf_score=v["rrf_score"] * recency,
                    channels=list(set(v["channels"])),
                    per_channel_ranks=v["ranks"],
                )
            )
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
        try:
            from remanentia_fact_decomposer import (  # type: ignore[import-untyped]  # Rust extension; no stubs
                tokenize_words as _rust_tw,
            )
        except ImportError:
            _rust_tw = None

        if _rust_tw is not None:  # pragma: no cover
            q_entities = set(_rust_tw(q_lower))
        else:
            q_entities = set(re.findall(r"\w{4,}", q_lower))
        result_text = " ".join(r.fact.text.lower() for r in results[:5])
        if _rust_tw is not None:  # pragma: no cover
            result_tokens = set(_rust_tw(result_text))
        else:
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
            try:
                from remanentia_fact_decomposer import tokenize_words as _rust_tw

                words = _rust_tw(original)  # pragma: no cover
            except ImportError:
                words = re.findall(r"\w{4,}", original)
            return " ".join(words[:5])

        return original

    _ce_model = None
    _ce_loading = False

    def _load_ce(self) -> None:
        """Lazy-load cross-encoder model in background thread."""
        if ArcaneRetriever._ce_model is not None or ArcaneRetriever._ce_loading:
            return
        ArcaneRetriever._ce_loading = True

        def _load() -> None:  # pragma: no cover — runs in background thread
            try:
                from sentence_transformers import CrossEncoder

                from device_utils import safe_device

                device = safe_device()
                ArcaneRetriever._ce_model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device
                )
            except Exception:
                ArcaneRetriever._ce_model = False
            ArcaneRetriever._ce_loading = False

        import threading

        threading.Thread(target=_load, daemon=True).start()

    def _cross_encoder_rerank(
        self, query: str, results: list[FusedResult], top_n: int = 10
    ) -> list[FusedResult]:
        """Rerank results with the cross-encoder; on by default.

        Set ``REMANENTIA_ARCANE_CE_DISABLE=1`` to skip reranking — for
        latency-sensitive live/MCP use that wants to avoid loading the
        cross-encoder model. The model loads lazily in a background thread on
        first use, so the first call(s) return un-reranked results until it is
        ready; subsequent calls (model loaded) are reranked.
        """
        if os.getenv("REMANENTIA_ARCANE_CE_DISABLE") == "1":
            return results
        if ArcaneRetriever._ce_model is None:
            self._load_ce()
        if not ArcaneRetriever._ce_model or ArcaneRetriever._ce_loading:
            return results

        pairs = [(query, r.fact.text) for r in results[: top_n * 2]]
        if not pairs:
            return results

        scores = ArcaneRetriever._ce_model.predict(pairs, show_progress_bar=False)
        for i, score in enumerate(scores):
            results[i].rrf_score = float(score)

        results.sort(key=lambda x: -x.rrf_score)
        return results

    def build_context(
        self,
        question: str,
        results: list[FusedResult],
        max_facts: int = 15,
        sort_chronologically: bool = False,
    ) -> str:
        """Build hierarchical LLM context from retrieval results (DeerFlow adaptation).

        By default, results are kept in RRF/rerank order (relevance-first),
        which is what knowledge-update, multi-session, and single-session
        questions need. When *sort_chronologically* is ``True`` (for
        temporal-reasoning questions), results are sorted by date with
        intraday HH:MM tiebreaking — critical for ordering and duration
        questions.
        """
        if sort_chronologically:
            results = _sort_results_chronologically(results)
        facts = [r.fact for r in results]
        h_ctx = build_hierarchical_context(
            facts, reference_date=self._reference_date, session_dates=self.session_dates
        )
        return str(h_ctx.to_prompt_string())


def _sort_results_chronologically(results: list[FusedResult]) -> list[FusedResult]:
    """Sort fused results by earliest date for chronological context.

    Resolution order for the primary key:

    1. ``valid_from`` (extracted from sentence, ISO ``YYYY-MM-DD``)
    2. ``date_mentions[0]`` (any explicit date mention)
    3. ``session_date`` (LongMemEval-format full timestamp with HH:MM)

    Same-day ties are broken by ``session_date`` (which carries HH:MM
    when available), then by ``(session_idx, turn_idx)`` to preserve the
    chronological session order established by Fix #3 in
    bench_longmemeval. Undated results are appended at the end.
    """
    from date_normalizer import _parse_session_datetime

    def _normalised_session_dt(s: str) -> str:
        """Convert ``2023/05/22 (Mon) 09:38`` to ``2023-05-22T09:38`` for sort."""
        if not s:
            return ""
        dt = _parse_session_datetime(s)
        if dt is None:
            return s  # pragma: no cover — defensive fallback for unknown date format
        return str(dt.isoformat())

    def _date_key(r: FusedResult) -> tuple[str, str, int, int]:
        primary = (
            r.fact.valid_from
            or (r.fact.date_mentions[0] if r.fact.date_mentions else "")
            or _normalised_session_dt(r.fact.session_date)
            or "\xff"  # undated → sort last
        )
        secondary = _normalised_session_dt(r.fact.session_date)
        return (primary, secondary, r.fact.session_idx, r.fact.turn_idx)

    return sorted(results, key=_date_key)
