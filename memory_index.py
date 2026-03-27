# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Unified Memory Index

"""Unified index over all GOTM knowledge sources.

BM25 first pass + optional GPU embedding rerank.

Usage::
    from memory_index import MemoryIndex
    idx = MemoryIndex()
    idx.build()
    results = idx.search("STDP learning rule fix", top_k=5)
"""
from __future__ import annotations

import ast
import json
import math
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
GOTM_ROOT = BASE.parent
INDEX_PATH = BASE / "snn_state" / "memory_index.pkl"
GRAPH_DIR = BASE / "memory" / "graph"

# All knowledge sources to index
SOURCES = {
    # Arcane Sapience core
    "traces": BASE / "reasoning_traces",
    "paper": BASE / "paper",
    "semantic": BASE / "memory" / "semantic",
    "disposition": BASE / "disposition",
    # Coordination
    "sessions_as": GOTM_ROOT / ".coordination" / "sessions" / "arcane-sapience",
    "sessions_codex": GOTM_ROOT / ".coordination" / "sessions" / "CODEX",
    "handovers_as": GOTM_ROOT / ".coordination" / "handovers" / "arcane-sapience",
    "handovers_codex": GOTM_ROOT / ".coordination" / "handovers" / "codex",
    # Cross-repo research
    "qc_research": GOTM_ROOT / ".coordination" / "handovers" / "scpn-quantum-control",
    "po_research": GOTM_ROOT / ".coordination" / "handovers" / "scpn-phase-orchestrator",
    "nc_research": GOTM_ROOT / "03_CODE" / "sc-neurocore" / "docs" / "internal",
    # Claude memory
    "claude_memory": Path.home() / ".claude" / "projects" / "C--aaa-God-of-the-Math-Collection" / "memory",
    # INDEXER catalog
    "indexer": GOTM_ROOT / "INDEXER",
    # Code: Remanentia
    "code_remanentia": BASE,
    # Code: key repos (top-level Python files only, not venvs/node_modules)
    "code_orchestrator": GOTM_ROOT / "03_CODE" / "scpn-phase-orchestrator" / "src",
    "code_quantum": GOTM_ROOT / "03_CODE" / "scpn-quantum-control" / "src",
    "code_neurocore": GOTM_ROOT / "03_CODE" / "sc-neurocore" / "src",
    "code_director": GOTM_ROOT / "03_CODE" / "DIRECTOR_AI" / "src",
}

# File extensions to index per source type
SOURCE_EXTENSIONS = {
    "code_remanentia": {".py"},
    "code_orchestrator": {".py", ".rs"},
    "code_quantum": {".py", ".rs"},
    "code_neurocore": {".py", ".rs"},
    "code_director": {".py"},
    "indexer": {".md", ".yaml"},
}

SKIP_PATH_PARTS = (
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".git",
    "target",
    "dist",
    ".egg",
)
MIN_FILE_CHARS = 50
MAX_FILE_CHARS = 1_000_000
MAX_TEXT_PARAGRAPH_CHARS = 10_000
MAX_FALLBACK_TEXT_CHARS = 2_000
MAX_CODE_CHUNK_CHARS = 1000
MAX_CODE_CHUNKS = 200
GRAPH_BOOST_QUERY_TYPES = {"general", "decision", "debugging", "explanation"}
RUST_BM25_MIN_PARAGRAPHS = 50_000
LOCATION_STOPWORDS = {
    "where", "what", "which", "file", "find", "locate", "defined", "definition",
    "implemented", "implementation", "method", "function", "class", "module",
    "work", "works", "does", "code", "source", "path", "show",
}

_RUST_BM25_CLASS = None
_RUST_BM25_IMPORT_ATTEMPTED = False


@dataclass
class Document:
    name: str
    source: str
    path: str
    paragraphs: list[str] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    embedding: np.ndarray | None = None
    date: str = ""  # parsed date for temporal search
    doc_type: str = ""  # document type for filtering


@dataclass
class Paragraph:
    text: str
    para_type: str = ""  # function, decision, finding, metric, discussion
    prospective_queries: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    name: str
    source: str
    score: float
    snippet: str
    paragraph_idx: int = 0
    answer: str = ""
    confidence: float = 0.0  # 0.0-1.0, computed from score distribution


class MemoryIndex:
    def __init__(self):
        self.documents: list[Document] = []
        self.paragraph_index: list[tuple[int, int]] = []  # (doc_idx, para_idx)
        self.paragraph_tokens: list[set[str]] = []
        self.paragraph_token_counts: list[dict[str, int]] = []  # token → count per paragraph
        self.paragraph_types: list[str] = []  # function, decision, finding, etc.
        self.idf: dict[str, float] = {}
        self._df: dict[str, int] = {}  # document frequency counts for incremental IDF
        self._inverted_index: dict[str, list[int]] = {}  # token → paragraph indices
        self.embeddings: np.ndarray | None = None
        self._built = False
        self._embed_model = None
        self._cross_encoder = None
        self._ce_loading = False
        self._embed_loading = False
        self._temporal_graph = None
        self._para_lengths: np.ndarray = np.array([], dtype=np.float32)
        self._avg_dl: float = 1.0
        self._answer_extractor = None
        self._llm_answer_extractor = None
        self._rust_bm25 = None
        self._rust_bm25_dirty = False

    def build(self, use_gpu_embeddings: bool = True, use_gliner: bool = True,
              use_llm_indexing: bool = False) -> dict:
        """Scan all sources, build BM25 index + GPU embeddings + GLiNER entities."""
        t0 = time.monotonic()
        self.documents = []
        self.paragraph_index = []
        self.paragraph_tokens = []
        self.all_entities: list[dict] = []
        self.all_relations: list[dict] = []

        # GLiNER model (load once, reuse)
        gliner_model = None
        if use_gliner:  # pragma: no cover
            try:
                from entity_extractor import _load_gliner
                gliner_model = _load_gliner()
            except Exception:
                pass

        # Scan all sources
        for source_name, source_dir in SOURCES.items():
            if not source_dir.exists():  # pragma: no cover
                continue
            for f in _iter_source_files(source_name, source_dir):
                try:
                    text = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):  # pragma: no cover
                    continue
                if not _should_index_text(text):
                    continue

                is_code = f.suffix in (".py", ".rs", ".v", ".ts", ".js")
                paragraphs = _split_paragraphs(text, is_code=is_code)
                all_tokens = set()

                # Tag paragraphs + generate prospective queries
                enriched_paragraphs = []
                _llm_pq = None
                if use_llm_indexing:  # pragma: no cover
                    try:
                        from answer_extractor import llm_generate_prospective_queries
                        _llm_pq = llm_generate_prospective_queries
                    except ImportError:
                        pass
                for p in paragraphs:
                    p_type = _classify_paragraph(p, is_code=is_code)
                    pq = _generate_prospective_queries(p, f.name, p_type)
                    if _llm_pq:  # pragma: no cover
                        pq_llm = _llm_pq(p, f.name)
                        pq = pq + pq_llm
                    enriched_paragraphs.append(p)
                    # Add prospective queries as extra searchable text
                    p_with_pq = p + " " + " ".join(pq) if pq else p
                    all_tokens.update(_tokenize(p_with_pq))

                doc_date = _parse_date(text, f.name)

                doc = Document(
                    name=f.name,
                    source=source_name,
                    path=str(f),
                    paragraphs=enriched_paragraphs,
                    tokens=all_tokens,
                    date=doc_date,
                    doc_type="code" if is_code else source_name,
                )
                doc_idx = len(self.documents)
                self.documents.append(doc)

                for para_idx, para in enumerate(enriched_paragraphs):
                    self.paragraph_index.append((doc_idx, para_idx))
                    p_type = _classify_paragraph(para, is_code=is_code)
                    pq = _generate_prospective_queries(para, f.name, p_type)
                    combined_text = para + " " + " ".join(pq)
                    token_list = _tokenize(combined_text)
                    tokens_with_pq = set(token_list)
                    token_counts = _token_counts(token_list)
                    self.paragraph_tokens.append(tokens_with_pq)
                    self.paragraph_token_counts.append(token_counts)
                    self.paragraph_types.append(p_type)

        # Build inverted index + IDF
        n_docs = len(self.paragraph_tokens)
        df: Counter = Counter()
        inv: dict[str, list[int]] = {}
        for i, tokens in enumerate(self.paragraph_tokens):
            for t in tokens:
                df[t] += 1
                if t not in inv:
                    inv[t] = []
                inv[t].append(i)
        self._inverted_index = inv
        self._df = dict(df)
        self.idf = {t: math.log(1 + n_docs / (1 + count))
                     for t, count in df.items()}
        self._para_lengths = np.array([len(t) for t in self.paragraph_tokens], dtype=np.float32)
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
        self._rust_bm25_dirty = True

        # GPU embeddings if available
        if use_gpu_embeddings:  # pragma: no cover
            try:
                self._compute_embeddings()
            except Exception:
                pass

        # Build temporal graph from all indexed documents
        try:
            from temporal_graph import TemporalGraph
            self._temporal_graph = TemporalGraph()
            doc_texts = [(d.name, "\n\n".join(d.paragraphs)) for d in self.documents]
            self._temporal_graph.build_from_documents(doc_texts)
        except Exception:  # pragma: no cover
            self._temporal_graph = None

        self._built = True
        elapsed = time.monotonic() - t0

        stats = {
            "documents": len(self.documents),
            "paragraphs": len(self.paragraph_index),
            "unique_tokens": len(self.idf),
            "has_embeddings": self.embeddings is not None,
            "temporal_events": self._temporal_graph.stats["events"] if self._temporal_graph else 0,
            "build_time_s": round(elapsed, 1),
            "sources": {s: sum(1 for d in self.documents if d.source == s)
                        for s in SOURCES if any(d.source == s for d in self.documents)},
        }
        return stats

    def _compute_embeddings(self):  # pragma: no cover
        """Compute paragraph embeddings on GPU via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            return

        if self._embed_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        texts = []
        for doc_idx, para_idx in self.paragraph_index:
            texts.append(self.documents[doc_idx].paragraphs[para_idx][:512])

        # Batch encode on GPU
        self.embeddings = self._embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def add_file(self, path: Path, source: str = "traces") -> int:
        """Incrementally add a single file to the index. Returns number of paragraphs added."""
        if not self._built:
            return 0
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0
        if len(text) < 50:
            return 0

        is_code = path.suffix in (".py", ".rs", ".v", ".ts", ".js")
        paragraphs = _split_paragraphs(text, is_code=is_code)
        if not paragraphs:  # pragma: no cover
            return 0

        doc_date = _parse_date(text, path.name)
        doc = Document(
            name=path.name, source=source, path=str(path),
            paragraphs=paragraphs, date=doc_date,
            doc_type="code" if is_code else source,
        )
        doc_idx = len(self.documents)
        self.documents.append(doc)

        n_existing = len(self.paragraph_tokens)
        for para_idx, para in enumerate(paragraphs):
            p_idx = len(self.paragraph_tokens)
            self.paragraph_index.append((doc_idx, para_idx))
            p_type = _classify_paragraph(para, is_code=is_code)
            pq = _generate_prospective_queries(para, path.name, p_type)
            combined = para + " " + " ".join(pq)
            token_list = _tokenize(combined)
            tokens = set(token_list)
            token_counts = _token_counts(token_list)
            self.paragraph_tokens.append(tokens)
            self.paragraph_token_counts.append(token_counts)
            self.paragraph_types.append(p_type)
            n_total = len(self.paragraph_tokens)
            for t in tokens:
                self._df[t] = self._df.get(t, 0) + 1
                self.idf[t] = math.log(1 + n_total / (1 + self._df[t]))
                if t not in self._inverted_index:
                    self._inverted_index[t] = []
                self._inverted_index[t].append(p_idx)

        # Update para_lengths array
        new_lengths = np.array([len(self.paragraph_tokens[n_existing + i])
                                for i in range(len(paragraphs))], dtype=np.float32)
        self._para_lengths = np.concatenate([self._para_lengths, new_lengths]) if len(self._para_lengths) > 0 else new_lengths
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0

        # Compute embeddings for new paragraphs if model is loaded
        if self._embed_model is not None:  # pragma: no cover
            new_texts = [paragraphs[i][:512] for i in range(len(paragraphs))]
            new_embs = self._embed_model.encode(
                new_texts, batch_size=64, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True)
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embs])
            else:
                self.embeddings = new_embs
        self._rust_bm25_dirty = True

        return len(paragraphs)

    def warm_models(self):  # pragma: no cover — requires GPU models
        """Opt-in model warmup for embedding and cross-encoder rerankers."""
        self._start_model_warmup()

    def _start_model_warmup(self):  # pragma: no cover — requires GPU models
        """Start loading embedding + cross-encoder models in background."""
        import threading
        if self._embed_model is None and not self._embed_loading and self.embeddings is not None:
            self._embed_loading = True
            def _load_embed():  # pragma: no cover — downloads real model
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                except Exception:
                    pass
                self._embed_loading = False
            threading.Thread(target=_load_embed, daemon=True).start()

        if self._cross_encoder is None and not self._ce_loading:
            self._ce_loading = True
            def _load_ce():  # pragma: no cover — downloads real model
                try:
                    from sentence_transformers import CrossEncoder
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._cross_encoder = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
                except Exception:
                    self._cross_encoder = False
                self._ce_loading = False
            threading.Thread(target=_load_ce, daemon=True).start()

    def _cross_encoder_rerank(self, query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]] | None:
        """Rerank candidates using cross-encoder if loaded.

        Non-blocking: if model is still loading, returns None (BM25 results
        used). Model loads in background — next query benefits.
        """
        if not candidates or self._cross_encoder is False:
            return None
        if self._cross_encoder is None or self._ce_loading:
            return None

        pairs = []
        for para_idx, _ in candidates:
            doc_idx, p_idx = self.paragraph_index[para_idx]
            text = self.documents[doc_idx].paragraphs[p_idx][:512]
            pairs.append((query, text))

        try:
            ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            reranked = [(candidates[i][0], float(ce_scores[i]))
                        for i in range(len(candidates))]
            reranked.sort(key=lambda x: -x[1])
            return reranked
        except Exception:  # pragma: no cover
            return None

    def search(self, query: str, top_k: int = 5,
               project: str = "", after: str = "", before: str = "",
               doc_type: str = "", use_llm: bool = False) -> list[SearchResult]:
        """Search with query intelligence, optional structured filters.

        Filters (applied before scoring):
            project: filter by source name containing this string
            after: only docs with date >= this (YYYY-MM-DD)
            before: only docs with date <= this (YYYY-MM-DD)
            doc_type: filter by document type (code, traces, sessions, etc.)
        """
        if not self._built:
            self.build()

        # Query decomposition: break multi-hop queries into sub-queries
        sub_queries = _decompose_query(query)
        if sub_queries:
            return self._multi_hop_search(query, sub_queries, top_k=top_k,
                                          project=project, after=after, before=before,
                                          doc_type=doc_type, use_llm=use_llm)

        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        # Pre-compute filter sets for paragraph indices
        _filtered_out = set()
        if project or after or before or doc_type:
            for i in range(len(self.paragraph_index)):
                doc_idx = self.paragraph_index[i][0]
                if doc_idx >= len(self.documents):  # pragma: no cover
                    continue
                doc = self.documents[doc_idx]
                if project and project.lower() not in doc.source.lower() and project.lower() not in doc.name.lower():
                    _filtered_out.add(i)
                if after and doc.date and doc.date < after:
                    _filtered_out.add(i)
                if before and doc.date and doc.date > before:  # pragma: no cover
                    _filtered_out.add(i)
                if doc_type and doc_type.lower() not in doc.doc_type.lower():  # pragma: no cover
                    _filtered_out.add(i)

        # Query classification
        intent = _classify_query(query)
        boost_types = set(intent.get("boost_types", []))
        lookup_terms = _extract_lookup_terms(query) if intent["type"] == "location" else set()
        recency_cache: dict[str, float] = {}

        # BM25 scoring via inverted index (only visit paragraphs containing query tokens)
        candidate_scores = None
        if not (project or after or before or doc_type) and self._should_use_rust_bm25():
            candidate_scores = self._search_rust_bm25(
                q_tokens,
                top_k=max(top_k * 80, 256),
            )
        if candidate_scores is None:
            candidate_scores = self._search_python_bm25(q_tokens, _filtered_out)

        # Entity-centric person-name boost (bench_locomo: +3-5pp)
        query_names = _extract_query_names(query) if _is_person_centric(query) else set()

        # Apply boosts to candidates with score > 0
        scores = []
        for i, score in candidate_scores.items():
            doc_idx = self.paragraph_index[i][0] if i < len(self.paragraph_index) else 0
            para_idx = self.paragraph_index[i][1] if i < len(self.paragraph_index) else 0
            doc = self.documents[doc_idx] if doc_idx < len(self.documents) else None
            para_text = ""
            if doc is not None and para_idx < len(doc.paragraphs):
                para_text = doc.paragraphs[para_idx]

            if boost_types and i < len(self.paragraph_types):
                if self.paragraph_types[i] in boost_types:
                    score *= 1.5

            if intent["type"] == "location" and doc is not None:
                if doc.doc_type == "code":
                    score *= 2.0
                    if lookup_terms:
                        para_lower = para_text.lower()
                        doc_name_lower = doc.name.lower()
                        doc_path_lower = doc.path.lower()
                        match_count = sum(
                            1 for term in lookup_terms
                            if term in para_lower or term in doc_name_lower or term in doc_path_lower
                        )
                        if match_count:
                            score *= 1.0 + 0.4 * min(match_count, 3)
                else:
                    score *= 0.6

            if intent.get("recency") and doc is not None and doc.date:
                boost = recency_cache.get(doc.date)
                if boost is None:
                    boost = _recency_boost(doc.date)
                    recency_cache[doc.date] = boost
                score *= boost

            if intent["type"] == "temporal":
                if para_text and _has_date_expression(para_text):
                    score *= 1.4

            if query_names:
                if para_text:
                    para_lower = para_text.lower()
                    if any(name in para_lower for name in query_names):
                        score *= 1.3

            scores.append((i, score))

        # Entity graph boost: paragraphs mentioning query-related entities
        scores.sort(key=lambda x: -x[1])
        if intent["type"] in GRAPH_BOOST_QUERY_TYPES and scores:
            try:
                graph = _load_entity_graph()
                q_entities = _query_entity_ids(query, graph)
                if q_entities:
                    graph_window = min(len(scores), max(top_k * 20, 64))
                    for idx_s in range(graph_window):
                        i, score = scores[idx_s]
                        doc_idx = self.paragraph_index[i][0] if i < len(self.paragraph_index) else 0
                        p_idx = self.paragraph_index[i][1] if i < len(self.paragraph_index) else 0
                        if doc_idx < len(self.documents):
                            para_text = self.documents[doc_idx].paragraphs[p_idx] if p_idx < len(self.documents[doc_idx].paragraphs) else ""
                            eb = _entity_boost_score(para_text, q_entities, graph)
                            if eb > 0:
                                scores[idx_s] = (i, score * (1.0 + eb))
                    scores.sort(key=lambda x: -x[1])
            except Exception:  # pragma: no cover
                pass

        # Stage 1: Take top candidates for embedding rerank
        candidates = scores[:top_k * 6] if self.embeddings is not None else scores[:top_k * 3]

        # Stage 2: Reciprocal Rank Fusion with bi-encoder
        if self.embeddings is not None and self._embed_model is not None and candidates:  # pragma: no cover
            try:
                q_emb = self._embed_model.encode(
                    query, normalize_embeddings=True, convert_to_numpy=True)
                emb_scored = []
                for para_idx, _ in candidates:
                    emb_sim = float(np.dot(q_emb, self.embeddings[para_idx]))
                    emb_scored.append((para_idx, emb_sim))
                emb_scored.sort(key=lambda x: -x[1])
                candidates = _reciprocal_rank_fusion(
                    [candidates, emb_scored], k=60)
            except Exception:
                pass

        # Stage 3: Cross-encoder rerank on top candidates
        ce_candidates = candidates[:top_k * 3]
        if ce_candidates:
            ce_candidates = self._cross_encoder_rerank(query, ce_candidates)
            if ce_candidates:
                candidates = ce_candidates

        # Temporal sorting: for temporal queries, sort by document date
        if intent["type"] == "temporal" and candidates:
            q = query.lower()
            newest_first = any(w in q for w in ["recent", "latest", "last", "newest"])
            oldest_first = any(w in q for w in ["first", "earliest", "oldest", "original"])
            if newest_first or oldest_first:
                dated = []
                for para_idx, score in candidates:
                    doc_idx = self.paragraph_index[para_idx][0] if para_idx < len(self.paragraph_index) else 0
                    date = self.documents[doc_idx].date if doc_idx < len(self.documents) else ""
                    dated.append((para_idx, score, date))
                dated.sort(key=lambda x: (x[2] if oldest_first else "", -x[1]))
                if newest_first:  # pragma: no cover
                    dated.sort(key=lambda x: x[2], reverse=True)
                candidates = [(p, s) for p, s, _ in dated]

        # Temporal graph augmentation: inject events from temporal graph into results
        if intent["type"] == "temporal" and self._temporal_graph:
            try:
                t_events = self._temporal_graph.query_temporal(query, top_k=3)
                for ev in t_events:
                    # Find the document index for this event's source
                    for di, doc in enumerate(self.documents):
                        if doc.name == ev.source:
                            # Check if this doc is already in candidates
                            already_in = any(
                                self.paragraph_index[pi][0] == di
                                for pi, _ in candidates[:top_k]
                            )
                            if not already_in:  # pragma: no cover
                                # Find a paragraph index for this document
                                for pi, (d_idx, _) in enumerate(self.paragraph_index):
                                    if d_idx == di:
                                        candidates.insert(0, (pi, 2.0))
                                        break
                            break
            except Exception:  # pragma: no cover
                pass

        # Answer extraction — always on (bench_locomo: +16pp)
        _answer_extractor = self._get_answer_extractor()
        _llm_extractor = self._get_llm_answer_extractor() if use_llm else None

        # Build results, deduplicate by document
        seen_docs = set()
        results = []
        for para_idx, score in candidates:
            doc_idx, p_idx = self.paragraph_index[para_idx]
            if doc_idx in seen_docs:
                continue
            seen_docs.add(doc_idx)
            doc = self.documents[doc_idx]
            snippet = doc.paragraphs[p_idx][:300]
            answer = ""
            if _answer_extractor:
                answer = _answer_extractor(query, doc.paragraphs[p_idx]) or ""
            if not answer and _llm_extractor:
                answer = _llm_extractor(query, doc.paragraphs[p_idx]) or ""
            results.append(SearchResult(
                name=doc.name,
                source=doc.source,
                score=round(score, 4),
                snippet=snippet,
                paragraph_idx=p_idx,
                answer=answer,
            ))
            if len(results) >= top_k:
                break

        # Confidence scoring: normalise scores to [0, 1] with gap analysis
        if results:
            max_score = results[0].score if results[0].score > 0 else 1.0
            for i, r in enumerate(results):
                base_conf = min(1.0, r.score / max_score) if max_score > 0 else 0.0
                # Boost confidence if answer was extracted
                if r.answer:
                    base_conf = min(1.0, base_conf + 0.15)
                # Reduce confidence for low absolute scores
                if r.score < 1.0:
                    base_conf *= 0.7
                results[i] = SearchResult(
                    name=r.name, source=r.source, score=r.score,
                    snippet=r.snippet, paragraph_idx=r.paragraph_idx,
                    answer=r.answer, confidence=round(base_conf, 3),
                )

        # Cross-reference answer verification: boost confidence when answers agree
        if len(results) >= 2:
            results = _cross_reference_answers(results)

        # Temporal code execution for date arithmetic queries
        if intent["type"] == "temporal" and results:
            try:
                from temporal_graph import TemporalEvent, temporal_code_execute, parse_dates
                t_events = []
                for r in results[:5]:
                    doc_idx = next((di for di, d in enumerate(self.documents) if d.name == r.name), None)
                    if doc_idx is not None:
                        p_text = self.documents[doc_idx].paragraphs[r.paragraph_idx] if r.paragraph_idx < len(self.documents[doc_idx].paragraphs) else ""
                        for d in parse_dates(p_text):
                            t_events.append(TemporalEvent(date=d, text=p_text[:200], source=r.name, paragraph_idx=r.paragraph_idx))
                if t_events:
                    code_answer = temporal_code_execute(query, t_events)
                    if code_answer and not results[0].answer:
                        results[0] = SearchResult(
                            name=results[0].name, source=results[0].source,
                            score=results[0].score, snippet=results[0].snippet,
                            paragraph_idx=results[0].paragraph_idx,
                            answer=code_answer)
            except Exception:  # pragma: no cover
                pass

        # Multi-paragraph synthesis: combine top results into a grounded answer
        if use_llm and results and not results[0].answer:  # pragma: no cover
            try:
                from answer_extractor import llm_synthesize_answer
                paras = [r.snippet for r in results[:3]]
                synthesized = llm_synthesize_answer(query, paras)
                if synthesized:
                    results[0] = SearchResult(
                        name=results[0].name, source=results[0].source,
                        score=results[0].score, snippet=results[0].snippet,
                        paragraph_idx=results[0].paragraph_idx,
                        answer=synthesized,
                    )
            except ImportError:  # pragma: no cover
                pass

        return results

    def _multi_hop_search(self, original_query: str, sub_queries: list[str],
                          top_k: int = 5, **kwargs) -> list[SearchResult]:
        """Run sub-queries, combine results, re-rank by original query."""
        all_results: dict[str, SearchResult] = {}
        for sq in sub_queries:
            results = self.search.__wrapped__(self, sq, top_k=top_k, **kwargs) \
                if hasattr(self.search, '__wrapped__') else \
                self._single_query_search(sq, top_k=top_k, **kwargs)
            for r in results:
                if r.name not in all_results or r.score > all_results[r.name].score:
                    all_results[r.name] = r

        # Re-rank combined results by relevance to original query
        q_tokens = set(_tokenize(original_query))
        scored = []
        for r in all_results.values():
            combined_text = (r.snippet + " " + r.answer).lower()
            overlap = sum(1 for t in q_tokens if t in combined_text)
            scored.append((r, r.score + overlap * 0.2))
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[:top_k]]

    def _single_query_search(self, query: str, top_k: int = 5,
                             project: str = "", after: str = "", before: str = "",
                             doc_type: str = "", use_llm: bool = False) -> list[SearchResult]:
        """Single query search (no decomposition). Used by _multi_hop_search."""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        _filtered_out = set()
        if project or after or before or doc_type:
            for i in range(len(self.paragraph_index)):
                doc_idx = self.paragraph_index[i][0]
                if doc_idx >= len(self.documents):
                    continue
                doc = self.documents[doc_idx]
                if project and project.lower() not in doc.source.lower() and project.lower() not in doc.name.lower():
                    _filtered_out.add(i)
                if after and doc.date and doc.date < after:
                    _filtered_out.add(i)
                if before and doc.date and doc.date > before:
                    _filtered_out.add(i)
                if doc_type and doc_type.lower() not in doc.doc_type.lower():
                    _filtered_out.add(i)

        candidate_scores = self._search_python_bm25(q_tokens, _filtered_out)
        scores = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_k * 3]

        _answer_extractor = self._get_answer_extractor()
        seen_docs = set()
        results = []
        for para_idx, score in scores:
            doc_idx, p_idx = self.paragraph_index[para_idx]
            if doc_idx in seen_docs:
                continue
            seen_docs.add(doc_idx)
            doc = self.documents[doc_idx]
            snippet = doc.paragraphs[p_idx][:300]
            answer = ""
            if _answer_extractor:
                answer = _answer_extractor(query, doc.paragraphs[p_idx]) or ""
            results.append(SearchResult(
                name=doc.name, source=doc.source, score=round(score, 4),
                snippet=snippet, paragraph_idx=p_idx, answer=answer,
            ))
            if len(results) >= top_k:
                break
        return results

    def _search_python_bm25(self, q_tokens: set[str], filtered_out: set[int]) -> dict[int, float]:
        k1, b = 1.5, 0.75
        avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0
        candidate_scores: dict[int, float] = {}
        para_lengths = self._para_lengths
        token_counts = self.paragraph_token_counts
        for qt in q_tokens:
            posting = self._inverted_index.get(qt)
            if not posting:
                continue
            idf_val = self.idf.get(qt, 0)
            if idf_val == 0:
                continue
            for i in posting:
                if i in filtered_out:
                    continue
                tf = token_counts[i].get(qt, 1) if i < len(token_counts) else 1
                dl_ratio = para_lengths[i] / avg_dl
                candidate_scores[i] = candidate_scores.get(i, 0.0) + (
                    idf_val * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl_ratio))
                )
        return candidate_scores

    def _search_rust_bm25(self, q_tokens: set[str], top_k: int) -> dict[int, float] | None:
        bm25 = self._ensure_rust_bm25()
        if bm25 is None:
            return None
        try:
            results = bm25.search(list(q_tokens), top_k)
            return {int(para_idx): float(score) for para_idx, score in results}
        except Exception:
            self._rust_bm25 = False
            self._rust_bm25_dirty = False
            return None

    def _should_use_rust_bm25(self) -> bool:
        force = os.environ.get("REMANENTIA_USE_RUST_BM25", "").strip().lower()
        if force:
            return force not in {"0", "false", "no", "off"}
        return len(self.paragraph_index) >= RUST_BM25_MIN_PARAGRAPHS

    def _ensure_rust_bm25(self):
        if self._rust_bm25 is False:
            return None
        if self._rust_bm25 is not None and not self._rust_bm25_dirty:
            return self._rust_bm25

        rust_cls = _get_rust_bm25_class()
        if rust_cls is None:
            self._rust_bm25 = False
            self._rust_bm25_dirty = False
            return None

        try:
            rust_bm25 = rust_cls()
            rust_bm25.build(
                [list(tokens) for tokens in self.paragraph_tokens],
                [(int(doc_idx), int(para_idx)) for doc_idx, para_idx in self.paragraph_index],
            )
            self._rust_bm25 = rust_bm25
            self._rust_bm25_dirty = False
            return rust_bm25
        except Exception:
            self._rust_bm25 = False
            self._rust_bm25_dirty = False
            return None

    def _get_answer_extractor(self):
        if self._answer_extractor is None:
            try:
                from answer_extractor import extract_answer
                self._answer_extractor = extract_answer
            except ImportError:  # pragma: no cover
                self._answer_extractor = False
        return self._answer_extractor if self._answer_extractor else None

    def _get_llm_answer_extractor(self):
        if self._llm_answer_extractor is None:
            try:
                from answer_extractor import llm_extract_answer
                self._llm_answer_extractor = llm_extract_answer
            except ImportError:  # pragma: no cover
                self._llm_answer_extractor = False
        return self._llm_answer_extractor if self._llm_answer_extractor else None

    def save(self, path: Path | None = None, quantize: bool = True):
        """Save index to disk. Quantizes embeddings to int8 by default (~4x smaller)."""
        path = path or INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        emb_data = None
        emb_scale = None
        if self.embeddings is not None:
            if quantize:
                scale = np.max(np.abs(self.embeddings), axis=1, keepdims=True)
                scale = np.where(scale == 0, 1.0, scale)
                emb_data = (self.embeddings / scale * 127).astype(np.int8)
                emb_scale = scale.astype(np.float32)
            else:
                emb_data = self.embeddings

        data = {
            "documents": [(d.name, d.source, d.path, d.paragraphs, d.date, d.doc_type)
                          for d in self.documents],
            "paragraph_index": self.paragraph_index,
            "paragraph_tokens": [list(t) for t in self.paragraph_tokens],
            "paragraph_token_counts": self.paragraph_token_counts,
            "paragraph_types": self.paragraph_types,
            "idf": self.idf,
            "_df": self._df,
            "embeddings": emb_data,
            "emb_scale": emb_scale,
            "quantized": quantize and self.embeddings is not None,
            "timestamp": time.time(),
        }
        # Atomic write: temp file + rename
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        tmp.replace(path)

    def load(self, path: Path | None = None) -> bool:
        """Load index from disk."""
        path = path or INDEX_PATH
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.documents = []
            for entry in data["documents"]:
                if len(entry) == 6:
                    n, s, p, paras, date, dtype = entry
                    self.documents.append(Document(name=n, source=s, path=p,
                                                    paragraphs=paras, date=date, doc_type=dtype))
                else:
                    n, s, p, paras = entry
                    self.documents.append(Document(name=n, source=s, path=p, paragraphs=paras))
            self.paragraph_index = data["paragraph_index"]
            self.paragraph_tokens = [set(t) for t in data["paragraph_tokens"]]
            self.paragraph_token_counts = data.get("paragraph_token_counts", [])
            if not self.paragraph_token_counts:
                self.paragraph_token_counts = [{t: 1 for t in tokens} for tokens in self.paragraph_tokens]
            self.idf = data["idf"]
            self._df = data.get("_df", {})
            self.paragraph_types = data.get("paragraph_types", [])
            # Rebuild inverted index from paragraph tokens
            inv: dict[str, list[int]] = {}
            for i, tokens in enumerate(self.paragraph_tokens):
                for t in tokens:
                    if t not in inv:
                        inv[t] = []
                    inv[t].append(i)
            self._inverted_index = inv
            self._para_lengths = np.array([len(t) for t in self.paragraph_tokens], dtype=np.float32)
            self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
            self._rust_bm25_dirty = True
            self._rust_bm25 = None
            # Dequantize embeddings if needed
            if data.get("quantized") and data.get("emb_scale") is not None:
                emb_int8 = data["embeddings"]
                scale = data["emb_scale"]
                self.embeddings = (emb_int8.astype(np.float32) / 127.0) * scale
            else:
                self.embeddings = data.get("embeddings")
            self._built = True
            return True
        except Exception:
            return False


# ── Entity graph for retrieval boosting ──────────────────────────

_ENTITY_GRAPH: dict | None = None


def _load_entity_graph() -> dict:
    """Load entities + relations from JSONL. Cached after first call."""
    global _ENTITY_GRAPH
    if _ENTITY_GRAPH is not None:
        return _ENTITY_GRAPH
    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    if entities_path.exists():
        for line in entities_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                e = json.loads(line)
                entities[e["id"]] = e
    if relations_path.exists():
        for line in relations_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                relations.append(json.loads(line))
    _ENTITY_GRAPH = {"entities": entities, "relations": relations}
    return _ENTITY_GRAPH


def _query_entity_ids(query: str, graph: dict) -> set[str]:
    """Find entity IDs mentioned in query text."""
    q_lower = query.lower()
    q_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", q_lower))
    matched = set()
    for eid, e in graph["entities"].items():
        label = e.get("label", eid).lower()
        if label in q_lower or label in q_tokens:
            matched.add(eid)
    return matched


def _entity_boost_score(para_text: str, query_entities: set[str], graph: dict) -> float:
    """Score how many query-related entities appear in paragraph text.

    Typed relations (caused_by, fixed_by, etc.) get 2x the boost of co_occurs,
    since they carry stronger semantic signal.
    """
    if not query_entities:
        return 0.0
    p_lower = para_text.lower()
    boost = 0.0
    for eid in query_entities:
        label = graph["entities"].get(eid, {}).get("label", eid).lower()
        if label in p_lower:
            boost += 0.1
    # Extra boost for typed relations between query entities and paragraph entities
    for rel in graph.get("relations", []):
        src, tgt = rel.get("source", ""), rel.get("target", "")
        if rel.get("type", "co_occurs") != "co_occurs":
            if (src in query_entities and tgt.lower() in p_lower) or \
               (tgt in query_entities and src.lower() in p_lower):
                boost += 0.15
    return boost


# ── Entity-centric boosting (ported from bench_locomo.py) ────────

_PERSON_CENTRIC_RE = re.compile(
    r"\b(relationship|hobby|hobbies|interest|interests|career|job|status|"
    r"personality|feel|feeling|prefer|favorite|partake|destress|self-care|"
    r"political|leaning|member|community)\b", re.IGNORECASE)

_POSSESSIVE_RE = re.compile(
    r"\b(his|her|their|'s)\s+(hobby|hobbies|interest|interests|career|"
    r"relationship|status|personality|feeling|preference|activity|activities)\b",
    re.IGNORECASE)


def _extract_query_names(query: str) -> set[str]:
    names = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
        word = m.group(1).lower()
        if word not in {"what", "when", "where", "who", "how", "why", "would",
                        "could", "does", "did", "has", "have", "the", "which",
                        "likely", "yes", "not"}:
            names.add(word)
    return names


def _is_person_centric(query: str) -> bool:
    if _PERSON_CENTRIC_RE.search(query):
        return True
    if _POSSESSIVE_RE.search(query):
        return True
    q_lower = query.lower()
    return any(w in q_lower for w in ["would ", "could ", "likely "])


# ── Query intelligence ────────────────────────────────────────────

def _classify_query(query: str) -> dict:
    """Classify query intent for search routing."""
    q = query.lower()
    intent = {
        "type": "general",
        "boost_types": [],  # paragraph types to boost
        "date_filter": None,
        "recency": False,
    }

    # Order matters: specific patterns before broad ones.
    if any(w in q for w in ["where is", "find the", "locate", "which file", "what file"]):
        intent["type"] = "location"
        intent["boost_types"] = ["function", "code"]
    elif any(w in q for w in ["what did we decide", "decision", "chose", "rejected", "why did we"]):
        intent["type"] = "decision"
        intent["boost_types"] = ["decision"]
    elif any(w in q for w in ["what went wrong", "failure", "bug", "error", "fix"]):
        intent["type"] = "debugging"
        intent["boost_types"] = ["finding", "decision"]
    elif any(w in q for w in ["status", "progress", "current", "latest"]):
        intent["type"] = "status"
        intent["recency"] = True
    elif any(w in q for w in ["performance", "benchmark", "accuracy", "score", "percent"]):
        intent["type"] = "metric"
        intent["boost_types"] = ["metric"]
    elif any(w in q for w in ["when", "date", "timeline", "before", "after", "first", "last"]):
        intent["type"] = "temporal"
        intent["recency"] = "latest" in q or "recent" in q or "last" in q
    elif any(w in q for w in ["how does", "how to", "explain", "what is"]):
        intent["type"] = "explanation"
        intent["boost_types"] = ["function", "finding"]

    return intent


def _classify_paragraph(text: str, is_code: bool = False) -> str:
    """Tag paragraph with its semantic type."""
    t = text.lower()

    if is_code:
        if re.match(r"\s*(def |fn |pub fn |class |impl )", text):
            return "function"
        return "code"

    if any(w in t for w in ["decided", "decision", "chose", "rejected", "we will", "the plan"]):
        return "decision"
    if any(w in t for w in ["found", "finding", "result", "measured", "shows that", "proved"]):
        return "finding"
    if any(w in t for w in ["P@1", "percent", "accuracy", "precision", "score", "benchmark"]):
        return "metric"
    if any(w in t for w in ["version", "v0.", "v1.", "v2.", "v3.", "release", "shipped"]):
        return "version"

    return "discussion"


def _generate_prospective_queries(text: str, doc_name: str, para_type: str) -> list[str]:
    """Generate hypothetical future queries for this paragraph.

    Kumiho technique: pre-answer questions at write time so retrieval
    becomes lookup. Expanded to 12 pattern categories.
    """
    queries = []
    text_lower = text.lower()

    # 1. Named entities (capitalised phrases)
    caps = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    for c in caps[:5]:
        if len(c) > 3:
            queries.append(f"what is {c}")
            queries.append(c.lower())

    # 2. Function/class names
    funcs = re.findall(r"(?:def |fn |class )\s*(\w+)", text)
    for f in funcs[:3]:
        queries.append(f"where is {f}")
        queries.append(f"how does {f} work")
        queries.append(f)

    # 3. Activities and preferences ("likes pottery", "enjoys hiking")
    for m in re.finditer(
        r"(?:likes?|loves?|enjoys?|prefers?|hates?|dislikes?|"
        r"interested in|passionate about|into)\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        activity = m.group(1).strip().lower()
        queries.append(f"hobbies {activity}")
        queries.append(f"interests {activity}")
        queries.append(f"what does {caps[0].lower() if caps else 'the person'} like")
        queries.append(activity)

    # 4. Occupation/role ("works as", "is a", "employed at")
    for m in re.finditer(
        r"(?:works? (?:as|at|for)|employed (?:at|by)|is a |job (?:is|as))\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        role = m.group(1).strip().lower()
        queries.append(f"where does {caps[0].lower() if caps else 'the person'} work")
        queries.append(f"job {role}")
        queries.append(f"career {role}")
        queries.append(role)

    # 5. Relationships ("married to", "friends with", "dating")
    for m in re.finditer(
        r"(?:married to|dating|friends? with|partner|spouse|sibling|brother|sister)\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        queries.append(f"relationship status")
        queries.append(f"who is {caps[0].lower() if caps else 'the person'} dating")

    # 6. Allergies, health, restrictions
    for m in re.finditer(
        r"(?:allergic to|allergy|intolerant|sensitive to|cannot eat|vegetarian|vegan)\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        subject = m.group(1).strip().lower()
        queries.append(f"allergic {subject}")
        queries.append(f"what is {caps[0].lower() if caps else 'the person'} allergic to")

    # 7. Travel/location ("went to", "visited", "lives in", "from")
    for m in re.finditer(
        r"(?:went to|visited|trip to|lives? in|moved to|from|travel(?:led|ed)? to)\s+(.{3,30}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        place = m.group(1).strip().lower()
        queries.append(f"where did {caps[0].lower() if caps else 'the person'} go")
        queries.append(f"trip {place}")
        queries.append(place)

    # 8. Learning/skills ("learning", "studying", "started")
    for m in re.finditer(
        r"(?:learning|studying|started|taking up|practicing)\s+(.{3,30}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        skill = m.group(1).strip().lower()
        queries.append(f"what is {caps[0].lower() if caps else 'the person'} learning")
        queries.append(skill)

    # 9. Favourites ("favourite", "favorite")
    for m in re.finditer(
        r"(?:favou?rite)\s+(\w+)\s+(?:is|was)\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        queries.append(f"favourite {m.group(1).lower()}")
        queries.append(m.group(2).strip().lower())
    for m in re.finditer(
        r"(?:favou?rite)\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text, re.I
    ):
        queries.append(f"favourite {m.group(1).strip().lower()}")

    # 10. Decision/finding/metric type-specific (original patterns)
    if para_type == "decision":
        subjects = re.findall(
            r"(?:decided|chose|rejected|will)\s+(?:to\s+)?(.{10,40}?)(?:\.|,|$)", text, re.I)
        for s in subjects[:2]:
            queries.append(f"why did we {s.strip().lower()}")
            queries.append(f"what did we decide about {s.strip().lower()}")
    if para_type == "finding":
        queries.append(f"what did we find about {doc_name.replace('.md','').replace('_',' ')}")
    if para_type == "metric":
        numbers = re.findall(r"\d+\.?\d*%", text)
        for n in numbers[:2]:
            queries.append(f"what score {n}")

    # 11. Version/date-specific queries
    versions = re.findall(r"v\d+\.\d+(?:\.\d+)?", text)
    for v in versions[:2]:
        queries.append(f"what version {v}")
        queries.append(f"when was {v} released")
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    for d in dates[:2]:
        queries.append(f"what happened on {d}")

    # 12. File/code-based queries
    if ".py" in doc_name or ".rs" in doc_name:
        base = doc_name.split(".")[0]
        queries.append(f"what does {base} do")
        queries.append(f"where is {base}")

    # Deduplicate, preserve order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:20]


_DATE_EXPR = re.compile(
    r"\d{4}-\d{2}-\d{2}"                               # ISO date
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"  # English date
    r"|\b(?:yesterday|today|last\s+(?:week|month|year))\b"  # Relative dates
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b",  # Abbreviated
    re.IGNORECASE,
)


def _has_date_expression(text: str) -> bool:
    """Check if text contains any date-like expression."""
    return bool(_DATE_EXPR.search(text))


def _recency_boost(date_str: str) -> float:
    """Compute recency boost relative to today. More recent = higher boost."""
    try:
        from datetime import datetime, date
        doc_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        today = date.today()
        days_ago = (today - doc_date).days
        if days_ago <= 2:
            return 1.8
        elif days_ago <= 5:
            return 1.4
        elif days_ago <= 14:
            return 1.2
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def _extract_date_context(text: str) -> list[tuple[str, str]]:
    """Extract dates with surrounding context from text."""
    results = []
    for m in re.finditer(r"(\d{4}-\d{2}-\d{2})", text):
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 100)
        context = text[start:end].strip()
        results.append((m.group(1), context))
    return results


def _parse_date(text: str, filename: str) -> str:
    """Extract date from filename or text content."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1)
    # From text content
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text[:500])
    if m:
        return m.group(1)
    return ""


def _split_paragraphs(text: str, is_code: bool = False) -> list[str]:
    """Split text into searchable units.

    For markdown: sentences with context windows (finer granularity than
    paragraphs for better retrieval precision). Short paragraphs (<200 chars)
    are kept whole. Longer ones are split into sentences with 1-sentence
    overlap for context continuity.
    For code: functions/classes (def/fn/class blocks) + module docstring.
    """
    if is_code:
        return _split_code(text)

    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if len(stripped) < 30:
            continue
        if len(stripped) <= 200:
            paragraphs.append(stripped[:MAX_TEXT_PARAGRAPH_CHARS])
        else:
            # Sentence-level splitting with context windows
            sents = _split_sentences(stripped)
            if len(sents) <= 2:
                paragraphs.append(stripped[:MAX_TEXT_PARAGRAPH_CHARS])
            else:
                for i in range(len(sents)):
                    # Context window: current sentence + 1 before + 1 after
                    start = max(0, i - 1)
                    end = min(len(sents), i + 2)
                    window = " ".join(sents[start:end])
                    if len(window) > 30:
                        paragraphs.append(window[:MAX_TEXT_PARAGRAPH_CHARS])
    if not paragraphs and len(text.strip()) > 30:
        paragraphs.append(text.strip()[:MAX_FALLBACK_TEXT_CHARS])
    return paragraphs


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Split on sentence boundaries but not on common abbreviations
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sents = []
    for p in parts:
        p = p.strip()
        if len(p) > 10:
            sents.append(p)
    return sents


def _split_code(text: str) -> list[str]:
    """Split code into function/class blocks for indexing."""
    py_chunks = _split_python_code(text)
    if py_chunks:
        return py_chunks[:MAX_CODE_CHUNKS]

    chunks = []

    # Module docstring (first triple-quoted block)
    doc_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if doc_match and doc_match.start() < 500:
        chunks.append(doc_match.group(1).strip()[:MAX_CODE_CHUNK_CHARS])

    # Python: def and class blocks
    for match in re.finditer(r'^((?:def|class|fn|pub fn|impl)\s+\w+.*?)(?=\n(?:def |class |fn |pub fn |impl |\Z))',
                              text, re.MULTILINE | re.DOTALL):
        block = match.group(1).strip()
        if len(block) > 30:
            # Keep first 500 chars (signature + docstring + start of body)
            chunks.append(block[:MAX_CODE_CHUNK_CHARS])

    # If no functions found, treat as plain text
    if not chunks:
        for block in text.split("\n\n"):
            stripped = block.strip()
            if len(stripped) > 30:
                chunks.append(stripped[:MAX_CODE_CHUNK_CHARS])

    return chunks[:MAX_CODE_CHUNKS]  # cap at 50 chunks per file


def _split_python_code(text: str) -> list[str]:
    """Split Python into module docstring, classes, functions, and class methods."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    chunks = []

    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append(module_doc.strip()[:MAX_CODE_CHUNK_CHARS])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block = _extract_python_block(lines, node)
            if len(block) > 30:
                chunks.append(block[:MAX_CODE_CHUNK_CHARS])
        elif isinstance(node, ast.ClassDef):
            class_header = f"class {node.name}:"
            class_doc = ast.get_docstring(node)
            if class_doc:
                chunks.append(f'{class_header}\n"""{class_doc.strip()}"""'[:MAX_CODE_CHUNK_CHARS])
            else:
                class_block = _extract_python_block(lines, node)
                if len(class_block) > 30:
                    chunks.append(class_block[:MAX_CODE_CHUNK_CHARS])

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_block = _extract_python_block(lines, child)
                    if len(method_block) > 30:
                        chunks.append(f"{class_header}\n{method_block}"[:MAX_CODE_CHUNK_CHARS])

    return chunks[:MAX_CODE_CHUNKS]


def _extract_python_block(lines: list[str], node: ast.AST) -> str:
    start = max(getattr(node, "lineno", 1) - 1, 0)
    end = max(getattr(node, "end_lineno", start + 1), start + 1)
    return "\n".join(lines[start:end]).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25. Lowercase, 3+ char words."""
    return re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower())


def _token_counts(token_list: list[str]) -> dict[str, int]:
    """Count occurrences of each token for real TF in BM25."""
    counts: dict[str, int] = {}
    for t in token_list:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _cross_reference_answers(results: list[SearchResult]) -> list[SearchResult]:
    """Boost confidence when multiple results corroborate the same answer.

    If results[0] and results[1] both extract the same answer, confidence
    goes up. If they extract different answers, no change.
    """
    answers_with_idx = [(i, r.answer.lower().strip()) for i, r in enumerate(results) if r.answer]
    if len(answers_with_idx) < 2:
        return results

    # Count answer agreement
    from collections import Counter
    answer_counts = Counter(a for _, a in answers_with_idx)
    most_common_answer, count = answer_counts.most_common(1)[0]

    if count >= 2:
        # Multiple results agree — boost confidence for agreeing results
        for i, answer in answers_with_idx:
            if answer == most_common_answer:
                old = results[i]
                boosted = min(1.0, old.confidence + 0.1 * (count - 1))
                results[i] = SearchResult(
                    name=old.name, source=old.source, score=old.score,
                    snippet=old.snippet, paragraph_idx=old.paragraph_idx,
                    answer=old.answer, confidence=round(boosted, 3),
                )

    return results


def _decompose_query(query: str) -> list[str] | None:
    """Decompose a multi-hop query into sub-queries.

    Returns None for simple queries (no decomposition needed).
    Multi-hop patterns: relative clauses, "the person who", "the one that".
    """
    q = query.lower()

    # "What X does the person who Y have/do?"
    m = re.match(
        r"what\s+(.+?)\s+(?:does|did|do)\s+the\s+(?:person|one|guy|woman|man)\s+who\s+(.+?)(?:\s+have|\s+do|\s+like|\?|$)",
        q, re.I
    )
    if m:
        return [f"who {m.group(2).strip()}", f"what {m.group(1).strip()}"]

    # "Does the person who X also Y?"
    m = re.match(
        r"(?:does|did|do)\s+the\s+(?:person|one)\s+who\s+(.+?)\s+(?:also\s+)?(.+?)(?:\?|$)",
        q, re.I
    )
    if m:
        return [f"who {m.group(1).strip()}", m.group(2).strip()]

    # "What happened before/after X did Y?"
    m = re.match(
        r"what\s+happened\s+(before|after)\s+(.+?)(?:\?|$)",
        q, re.I
    )
    if m:
        return [m.group(2).strip(), f"what happened {m.group(1)} {m.group(2).strip()}"]

    return None


def _extract_lookup_terms(query: str) -> set[str]:
    return {
        token for token in _tokenize(query)
        if token not in LOCATION_STOPWORDS and not token.isdigit()
    }


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    RRF score = sum(1 / (k + rank_i)) for each list where the item appears.
    Scale-invariant — no need to normalise heterogeneous score distributions.
    k=60 is the standard constant from Cormack et al. (2009).
    """
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (para_idx, _score) in enumerate(ranked):
            rrf_scores[para_idx] = rrf_scores.get(para_idx, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return result


def _get_rust_bm25_class():
    global _RUST_BM25_CLASS, _RUST_BM25_IMPORT_ATTEMPTED
    if not _RUST_BM25_IMPORT_ATTEMPTED:
        _RUST_BM25_IMPORT_ATTEMPTED = True
        try:
            from remanentia_search import BM25Index
            _RUST_BM25_CLASS = BM25Index
        except Exception:
            _RUST_BM25_CLASS = False
    return _RUST_BM25_CLASS if _RUST_BM25_CLASS else None


def _iter_source_files(source_name: str, source_dir: Path):
    exts = SOURCE_EXTENSIONS.get(source_name, {".md"})
    files = []
    for ext in exts:
        files.extend(source_dir.rglob(f"*{ext}"))
    for f in sorted(set(files)):
        if any(skip in str(f) for skip in SKIP_PATH_PARTS):
            continue
        yield f


def _should_index_text(text: str) -> bool:
    return MIN_FILE_CHARS <= len(text) <= MAX_FILE_CHARS


def needs_rebuild() -> bool:
    """Check if any source has newer files than the index."""
    if not INDEX_PATH.exists():
        return True
    idx_mtime = INDEX_PATH.stat().st_mtime
    for source_name, source_dir in SOURCES.items():
        if not source_dir.exists():  # pragma: no cover
            continue
        for f in _iter_source_files(source_name, source_dir):
            if f.stat().st_mtime > idx_mtime:  # pragma: no cover
                return True
    return False


def auto_rebuild_if_needed(use_gpu: bool = True) -> MemoryIndex:
    """Load index, rebuild if sources changed."""
    idx = MemoryIndex()
    if idx.load() and not needs_rebuild():
        return idx
    stats = idx.build(use_gpu_embeddings=use_gpu, use_gliner=False)
    idx.save()
    return idx


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    idx = MemoryIndex()

    if "--build" in sys.argv:
        use_gpu = "--no-gpu" not in sys.argv
        use_gliner = "--gliner" in sys.argv
        use_llm_idx = "--llm-index" in sys.argv
        print("Building index...")
        stats = idx.build(use_gpu_embeddings=use_gpu, use_gliner=use_gliner,
                          use_llm_indexing=use_llm_idx)
        print(json.dumps(stats, indent=2))
        idx.save()
        print(f"Saved to {INDEX_PATH}")

    elif "--watch" in sys.argv:
        print("Watching for changes... (Ctrl+C to stop)")
        interval = 60
        while True:
            if needs_rebuild():
                # Consolidate first so new semantic memories are indexed in the same pass
                try:
                    from consolidation_engine import consolidate
                    result = consolidate()
                    if result.get("memories_written", 0) > 0:
                        print(f"Consolidated: {result['memories_written']} memories, "
                              f"{result['entities_found']} entities")
                except Exception as e:
                    print(f"Consolidation error: {e}")
                print("Changes detected, rebuilding...")
                stats = idx.build(use_gpu_embeddings=True, use_gliner=False)
                idx.save()
                print(f"Rebuilt: {stats['documents']} docs, {stats['paragraphs']} paragraphs")
            time.sleep(interval)

    elif "--check" in sys.argv:
        if needs_rebuild():
            print("Index is STALE — rebuild needed")
        else:
            print("Index is CURRENT")

    elif len(sys.argv) > 1:
        query = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
        if not idx.load():
            print("No index found. Run: python memory_index.py --build")
            sys.exit(1)
        results = idx.search(query, top_k=5)
        for r in results:
            print(f"  [{r.source}] {r.name} (score={r.score:.3f})")
            print(f"    {r.snippet[:150]}")
            print()
    else:
        print("Usage:")
        print("  python memory_index.py --build              # build index (GPU embeddings)")
        print("  python memory_index.py --build --gliner     # build with GLiNER entities")
        print("  python memory_index.py --build --no-gpu     # build without GPU")
        print("  python memory_index.py --watch              # auto-rebuild on changes")
        print("  python memory_index.py --check              # check if rebuild needed")
        print("  python memory_index.py 'query text'         # search")
