# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Unified memory index

"""Unified index over configured Remanentia knowledge sources.

BM25 first pass + optional GPU embedding rerank.

Usage::
    from memory_index import MemoryIndex
    idx = MemoryIndex()
    idx.build()
    results = idx.search("STDP learning rule fix", top_k=5)
"""

from __future__ import annotations

import gzip
import json
import math
import os
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import numpy as np

import hashlib as _hashlib

import memory_dates as _memory_dates
from memory_entity_scoring import (
    Entity,
    EntityGraph,
    build_relation_neighbors as _build_relation_neighbors,
    entity_boost_score as _entity_boost_score,
    extract_query_names as _extract_query_names,
    is_person_centric as _is_person_centric,
    query_entity_ids as _query_entity_ids,
)
from memory_sources import DEFAULT_TEXT_EXTENSIONS, load_source_config
import text_chunking as _text_chunking

_extract_date_context = _memory_dates.extract_date_context
_has_date_expression = _memory_dates.has_date_expression
_parse_date = _memory_dates.parse_document_date
_recency_boost = _memory_dates.recency_boost
MAX_CODE_CHUNK_CHARS = _text_chunking.MAX_CODE_CHUNK_CHARS
MAX_CODE_CHUNKS = _text_chunking.MAX_CODE_CHUNKS
MAX_FALLBACK_TEXT_CHARS = _text_chunking.MAX_FALLBACK_TEXT_CHARS
MAX_TEXT_PARAGRAPH_CHARS = _text_chunking.MAX_TEXT_PARAGRAPH_CHARS
_split_code = _text_chunking.split_code
_split_paragraphs = _text_chunking.split_paragraphs
_split_python_code = _text_chunking.split_python_code
_split_sentences = _text_chunking.split_sentences

BASE = Path(__file__).parent
INDEX_PATH = BASE / "snn_state" / "memory_index.json.gz"
_LEGACY_INDEX_PATH = BASE / "snn_state" / "memory_index.pkl"
INDEX_EMB_PATH = BASE / "snn_state" / "memory_index_embeddings.npz"
HASH_CACHE_PATH = BASE / "snn_state" / "content_hashes.json"
GRAPH_DIR = BASE / "memory" / "graph"
log = logging.getLogger(__name__)
_SOURCE_CONFIG = load_source_config(BASE)
SOURCES = dict(_SOURCE_CONFIG.sources)
SOURCE_EXTENSIONS = {label: set(suffixes) for label, suffixes in _SOURCE_CONFIG.extensions.items()}
AnswerExtractor = Callable[[str, str], str | None]

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
MANUSCRIPT_SKIP_PATH_PARTS = (
    "/ARCHIVE/",
    "/DOCX_ARCHIVE/",
    "/ACADEMIA_INPUT/",
    "/SWARM_INPUT/",
    "/PDF EXPORTS/",
)
MIN_FILE_CHARS = 50
MAX_FILE_CHARS = 1_000_000
GRAPH_BOOST_QUERY_TYPES = {"general", "decision", "debugging", "explanation"}
RUST_BM25_MIN_PARAGRAPHS = 50_000
TEMPORAL_GRAPH_MAX_DOCUMENTS = 20_000
COMPILED_FACT_MIN_SCORE = 8.0
COMPILED_FACT_EARLY_SCORE = 1008.0
LOCATION_STOPWORDS = {
    "where",
    "what",
    "which",
    "file",
    "find",
    "locate",
    "defined",
    "definition",
    "implemented",
    "implementation",
    "method",
    "function",
    "class",
    "module",
    "work",
    "works",
    "does",
    "code",
    "source",
    "path",
    "show",
}

_RUST_BM25_CLASS: Any | bool | None = None
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


def _compiled_fact_results(query: str, top_k: int) -> list[SearchResult]:
    try:
        from compiled_memory import load_compiled_facts, search_compiled_facts
    except Exception:
        return []
    try:
        matches = search_compiled_facts(query, load_compiled_facts(), top_k=top_k)
    except Exception:
        log.debug("Compiled memory search failed", exc_info=True)
        return []
    results: list[SearchResult] = []
    for fact, score in matches:
        if score < COMPILED_FACT_MIN_SCORE:
            continue
        snippet = fact.fact[:300]
        results.append(
            SearchResult(
                name=f"{fact.fact_id}.fact",
                source="compiled",
                score=round(1000.0 + score, 4),
                snippet=snippet,
                paragraph_idx=0,
                answer=fact.fact,
                confidence=1.0,
            )
        )
    return results


def _merge_priority_results(
    priority_results: list[SearchResult],
    ranked_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen = set()
    for result in priority_results + ranked_results:
        key = (result.source, result.name, result.answer or result.snippet)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged


def _has_operational_compiled_memory(index: MemoryIndex) -> bool:
    return len(index.paragraph_index) > 1000 or any(d.source == "compiled" for d in index.documents)


class MemoryIndex:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.paragraph_index: list[tuple[int, int]] = []  # (doc_idx, para_idx)
        self.paragraph_tokens: list[set[str]] = []
        self.paragraph_token_counts: list[dict[str, int]] = []  # token → count per paragraph
        self.paragraph_types: list[str] = []  # function, decision, finding, etc.
        self.idf: dict[str, float] = {}
        self._df: dict[str, int] = {}  # document frequency counts for incremental IDF
        self._inverted_index: dict[str, list[int]] = {}  # token → paragraph indices
        self._bm25_weight_index: dict[str, list[tuple[int, float]]] = {}
        self._bm25_weight_dirty = True
        self.embeddings: np.ndarray | None = None
        self._built = False
        self._embed_model: Any | None = None
        self._cross_encoder: Any | bool | None = None
        self._ce_loading = False
        self._embed_loading = False
        self._temporal_graph: Any | None = None
        self._para_lengths: np.ndarray = np.array([], dtype=np.float32)
        self._avg_dl: float = 1.0
        self._answer_extractor: AnswerExtractor | bool | None = None
        self._llm_answer_extractor: AnswerExtractor | bool | None = None
        self._rust_bm25: Any | bool | None = None
        self._rust_bm25_dirty = False
        self._content_hashes: dict[str, str] = {}  # file path → SHA-256 of content
        self._hash_hits = 0  # files skipped because hash unchanged
        self._hash_misses = 0  # files (re-)indexed because new or changed

    def build(
        self,
        use_gpu_embeddings: bool = True,
        use_gliner: bool = True,
        use_llm_indexing: bool = False,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Scan all sources, build BM25 index + GPU embeddings + GLiNER entities.

        When *incremental* is True and a hash cache exists, files whose
        SHA-256 content hash has not changed since the last build are
        reported as hash hits. The build still materializes a complete
        in-memory index; callers that need a true partial update should use
        :meth:`add_file` against an already-loaded index.
        """
        t0 = time.monotonic()
        self.documents = []
        self.paragraph_index = []
        self.paragraph_tokens = []
        self.paragraph_token_counts = []
        self.paragraph_types = []
        self.idf = {}
        self._df = {}
        self._inverted_index = {}
        self._bm25_weight_index = {}
        self._bm25_weight_dirty = True
        self.all_entities: list[dict[str, Any]] = []
        self.all_relations: list[dict[str, Any]] = []

        try:
            from compiled_memory import compile_facts

            compile_facts(BASE)
        except Exception:
            log.debug("Compiled memory refresh failed", exc_info=True)

        # Load previous content hashes for incremental builds
        old_hashes: dict[str, str] = {}
        if incremental:
            old_hashes = self._load_content_hashes()
        new_hashes: dict[str, str] = {}
        self._hash_hits = 0
        self._hash_misses = 0

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

                # Content-hash check: skip files unchanged since last build
                file_key = str(f)
                content_hash = _hashlib.sha256(text.encode("utf-8")).hexdigest()
                new_hashes[file_key] = content_hash
                if incremental and file_key in old_hashes and old_hashes[file_key] == content_hash:
                    self._hash_hits += 1
                else:
                    self._hash_misses += 1

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
        df: Counter[str] = Counter()
        inv: dict[str, list[int]] = {}
        for i, tokens in enumerate(self.paragraph_tokens):
            for t in tokens:
                df[t] += 1
                if t not in inv:
                    inv[t] = []
                inv[t].append(i)
        self._inverted_index = inv
        self._df = dict(df)
        self.idf = {t: math.log(1 + n_docs / (1 + count)) for t, count in df.items()}
        self._para_lengths = np.array([len(t) for t in self.paragraph_tokens], dtype=np.float32)
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
        self._bm25_weight_index = {}
        self._bm25_weight_dirty = True
        self._rust_bm25_dirty = True

        # GPU embeddings if available
        if use_gpu_embeddings:  # pragma: no cover
            try:
                self._compute_embeddings()
            except Exception:
                log.debug("GPU embedding computation failed", exc_info=True)

        # Build temporal graph from indexed documents when the corpus is small enough
        # for the in-memory event graph to remain useful.
        if len(self.documents) <= TEMPORAL_GRAPH_MAX_DOCUMENTS:
            try:
                from temporal_graph import TemporalGraph

                self._temporal_graph = TemporalGraph()
                doc_texts = [(d.name, "\n\n".join(d.paragraphs)) for d in self.documents]
                self._temporal_graph.build_from_documents(doc_texts)
            except Exception:  # pragma: no cover
                self._temporal_graph = None
        else:
            self._temporal_graph = None

        self._built = True

        # Persist content hashes for next incremental build
        self._content_hashes = new_hashes
        self._save_content_hashes(new_hashes)

        elapsed = time.monotonic() - t0

        stats: dict[str, Any] = {
            "documents": len(self.documents),
            "paragraphs": len(self.paragraph_index),
            "unique_tokens": len(self.idf),
            "has_embeddings": self.embeddings is not None,
            "temporal_events": self._temporal_graph.stats["events"] if self._temporal_graph else 0,
            "build_time_s": round(elapsed, 1),
            "hash_hits": self._hash_hits,
            "hash_misses": self._hash_misses,
            "sources": {
                s: sum(1 for d in self.documents if d.source == s)
                for s in SOURCES
                if any(d.source == s for d in self.documents)
            },
        }
        return stats

    def _compute_embeddings(self) -> None:  # pragma: no cover
        """Compute paragraph embeddings on GPU via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return

        if self._embed_model is None:
            from device_utils import safe_device

            device = safe_device()
            _local = BASE / "models" / "temporal-embed-v1"
            _model_id = str(_local) if _local.exists() else "all-MiniLM-L6-v2"
            self._embed_model = SentenceTransformer(_model_id, device=device)

        texts: list[str] = []
        for doc_idx, para_idx in self.paragraph_index:
            texts.append(self.documents[doc_idx].paragraphs[para_idx][:512])

        # Batch encode on GPU
        self.embeddings = cast(
            np.ndarray,
            self._embed_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
        )

    def _rebuild_sparse_index_from_documents(self) -> None:
        """Rebuild BM25 structures after document-level replacement."""
        self.paragraph_index = []
        self.paragraph_tokens = []
        self.paragraph_token_counts = []
        self.paragraph_types = []

        df: Counter[str] = Counter()
        inv: dict[str, list[int]] = {}
        for doc_idx, doc in enumerate(self.documents):
            is_code = doc.doc_type == "code" or Path(doc.path).suffix in (
                ".py",
                ".rs",
                ".v",
                ".ts",
                ".js",
            )
            doc_tokens: set[str] = set()
            for para_idx, para in enumerate(doc.paragraphs):
                p_idx = len(self.paragraph_tokens)
                self.paragraph_index.append((doc_idx, para_idx))
                p_type = _classify_paragraph(para, is_code=is_code)
                pq = _generate_prospective_queries(para, doc.name, p_type)
                token_list = _tokenize(para + " " + " ".join(pq))
                tokens = set(token_list)
                doc_tokens.update(tokens)
                self.paragraph_tokens.append(tokens)
                self.paragraph_token_counts.append(_token_counts(token_list))
                self.paragraph_types.append(p_type)
                for token in tokens:
                    df[token] += 1
                    inv.setdefault(token, []).append(p_idx)
            doc.tokens = doc_tokens

        n_docs = len(self.paragraph_tokens)
        self._df = dict(df)
        self._inverted_index = inv
        self.idf = {token: math.log(1 + n_docs / (1 + count)) for token, count in df.items()}
        self._para_lengths = np.array([len(t) for t in self.paragraph_tokens], dtype=np.float32)
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
        self._bm25_weight_index = {}
        self._bm25_weight_dirty = True
        self._rust_bm25_dirty = True

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

        file_key = str(path)
        replacing_existing = any(doc.path == file_key for doc in self.documents)
        if replacing_existing:
            self.documents = [doc for doc in self.documents if doc.path != file_key]
            if self.embeddings is not None:
                self.embeddings = None
            self._rebuild_sparse_index_from_documents()

        doc_date = _parse_date(text, path.name)
        doc = Document(
            name=path.name,
            source=source,
            path=str(path),
            paragraphs=paragraphs,
            tokens=set(),
            date=doc_date,
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
            doc.tokens.update(tokens)
            self.paragraph_tokens.append(tokens)
            self.paragraph_token_counts.append(token_counts)
            self.paragraph_types.append(p_type)
            n_total = len(self.paragraph_tokens)
            for t in tokens:
                self._df[t] = self._df.get(t, 0) + 1
                if t not in self._inverted_index:
                    self._inverted_index[t] = []
                self._inverted_index[t].append(p_idx)

        n_total = len(self.paragraph_tokens)
        self.idf = {t: math.log(1 + n_total / (1 + count)) for t, count in self._df.items()}

        # Update para_lengths array
        new_lengths = np.array(
            [len(self.paragraph_tokens[n_existing + i]) for i in range(len(paragraphs))],
            dtype=np.float32,
        )
        self._para_lengths = (
            np.concatenate([self._para_lengths, new_lengths])
            if len(self._para_lengths) > 0
            else new_lengths
        )
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
        self._bm25_weight_index = {}
        self._bm25_weight_dirty = True

        self._content_hashes[str(path)] = _hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Compute embeddings if the model is loaded. Replacement invalidates
        # paragraph positions, so rebuild all embeddings instead of appending.
        if replacing_existing and self._embed_model is not None:  # pragma: no cover
            self._compute_embeddings()
        elif self._embed_model is not None:  # pragma: no cover
            new_texts = [paragraphs[i][:512] for i in range(len(paragraphs))]
            new_embs = self._embed_model.encode(
                new_texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embs])
            else:
                self.embeddings = new_embs
        self._rust_bm25_dirty = True

        return len(paragraphs)

    def warm_models(self) -> None:  # pragma: no cover — requires GPU models
        """Opt-in model warmup for embedding and cross-encoder rerankers."""
        self._start_model_warmup()

    def _start_model_warmup(self) -> None:  # pragma: no cover — requires GPU models
        """Start loading embedding + cross-encoder models in background."""
        import threading

        if self._embed_model is None and not self._embed_loading and self.embeddings is not None:
            self._embed_loading = True

            def _load_embed() -> None:  # pragma: no cover — downloads real model
                try:
                    from sentence_transformers import SentenceTransformer

                    from device_utils import safe_device

                    device = safe_device()
                    _local = BASE / "models" / "temporal-embed-v1"
                    _model_id = str(_local) if _local.exists() else "all-MiniLM-L6-v2"
                    self._embed_model = SentenceTransformer(_model_id, device=device)
                except Exception:
                    log.debug("Embedding model load failed", exc_info=True)
                self._embed_loading = False

            threading.Thread(target=_load_embed, daemon=True).start()

        if self._cross_encoder is None and not self._ce_loading:
            self._ce_loading = True

            def _load_ce() -> None:  # pragma: no cover — downloads real model
                try:
                    from sentence_transformers import CrossEncoder

                    from device_utils import safe_device

                    device = safe_device()
                    _local_ce = BASE / "models" / "temporal-ce-v1"
                    _ce_id = (
                        str(_local_ce)
                        if _local_ce.exists()
                        else "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                    self._cross_encoder = CrossEncoder(_ce_id, device=device)
                except Exception:
                    self._cross_encoder = False
                self._ce_loading = False

            threading.Thread(target=_load_ce, daemon=True).start()

    def _cross_encoder_rerank(
        self, query: str, candidates: list[tuple[int, float]]
    ) -> list[tuple[int, float]] | None:
        """Rerank candidates using cross-encoder if loaded.

        Non-blocking: if model is still loading, returns None (BM25 results
        used). Model loads in background — next query benefits.
        """
        if not candidates or self._cross_encoder is False:
            return None
        if self._cross_encoder is None or self._ce_loading:
            return None
        cross_encoder = cast(Any, self._cross_encoder)

        pairs = []
        for para_idx, _ in candidates:
            doc_idx, p_idx = self.paragraph_index[para_idx]
            text = self.documents[doc_idx].paragraphs[p_idx][:512]
            pairs.append((query, text))

        try:
            ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            reranked = [(candidates[i][0], float(ce_scores[i])) for i in range(len(candidates))]
            reranked.sort(key=lambda x: -x[1])
            return reranked
        except Exception:  # pragma: no cover
            return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: str = "",
        after: str = "",
        before: str = "",
        doc_type: str = "",
        use_llm: bool = False,
        _decompose: bool = True,
    ) -> list[SearchResult]:
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
        sub_queries = _decompose_query(query) if _decompose else None
        if sub_queries:
            return self._multi_hop_search(
                query,
                sub_queries,
                top_k=top_k,
                project=project,
                after=after,
                before=before,
                doc_type=doc_type,
                use_llm=use_llm,
            )

        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        compiled_results = (
            _compiled_fact_results(query, top_k)
            if _has_operational_compiled_memory(self)
            and not (project or after or before or doc_type)
            else []
        )
        if compiled_results and compiled_results[0].score >= COMPILED_FACT_EARLY_SCORE:
            return compiled_results[:top_k]

        # Pre-compute filter sets for paragraph indices
        _filtered_out = set()
        if project or after or before or doc_type:
            for i in range(len(self.paragraph_index)):
                doc_idx = self.paragraph_index[i][0]
                if doc_idx >= len(self.documents):  # pragma: no cover
                    continue
                filter_doc = self.documents[doc_idx]
                if (
                    project
                    and project.lower() not in filter_doc.source.lower()
                    and project.lower() not in filter_doc.name.lower()
                ):
                    _filtered_out.add(i)
                if after and filter_doc.date and filter_doc.date < after:
                    _filtered_out.add(i)
                if before and filter_doc.date and filter_doc.date > before:  # pragma: no cover
                    _filtered_out.add(i)
                if (
                    doc_type and doc_type.lower() not in filter_doc.doc_type.lower()
                ):  # pragma: no cover
                    _filtered_out.add(i)

        # Query classification
        intent = _classify_query(query)
        boost_types = set(intent.get("boost_types", []))
        lookup_terms = _extract_lookup_terms(query) if intent["type"] == "location" else set()
        recency_cache: dict[str, float] = {}

        # BM25 scoring via inverted index (only visit paragraphs containing query tokens)
        candidate_scores: dict[int, float] | None = None
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
            candidate_doc: Document | None = (
                self.documents[doc_idx] if doc_idx < len(self.documents) else None
            )
            para_text = ""
            if candidate_doc is not None and para_idx < len(candidate_doc.paragraphs):
                para_text = candidate_doc.paragraphs[para_idx]

            if boost_types and i < len(self.paragraph_types):
                if self.paragraph_types[i] in boost_types:
                    score *= 1.5

            if intent["type"] == "location" and candidate_doc is not None:
                if candidate_doc.doc_type == "code":
                    score *= 2.0
                    if lookup_terms:
                        para_lower = para_text.lower()
                        doc_name_lower = candidate_doc.name.lower()
                        doc_path_lower = candidate_doc.path.lower()
                        match_count = sum(
                            1
                            for term in lookup_terms
                            if term in para_lower
                            or term in doc_name_lower
                            or term in doc_path_lower
                        )
                        if match_count:
                            score *= 1.0 + 0.4 * min(match_count, 3)
                else:
                    score *= 0.6

            if intent.get("recency") and candidate_doc is not None and candidate_doc.date:
                boost = recency_cache.get(candidate_doc.date)
                if boost is None:
                    boost = _recency_boost(candidate_doc.date)
                    recency_cache[candidate_doc.date] = boost
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
                            para_text = (
                                self.documents[doc_idx].paragraphs[p_idx]
                                if p_idx < len(self.documents[doc_idx].paragraphs)
                                else ""
                            )
                            eb = _entity_boost_score(para_text, q_entities, graph)
                            if eb > 0:
                                scores[idx_s] = (i, score * (1.0 + eb))
                    scores.sort(key=lambda x: -x[1])
            except Exception:  # pragma: no cover
                log.debug("Entity graph boost failed", exc_info=True)

        # Stage 1: Take top candidates for embedding rerank
        candidates = scores[: top_k * 6] if self.embeddings is not None else scores[: top_k * 3]

        # Stage 2: Reciprocal Rank Fusion with bi-encoder
        if (
            self.embeddings is not None and self._embed_model is not None and candidates
        ):  # pragma: no cover
            try:
                q_emb = self._embed_model.encode(
                    query, normalize_embeddings=True, convert_to_numpy=True
                )
                emb_scored = []
                for para_idx, _ in candidates:
                    emb_sim = float(np.dot(q_emb, self.embeddings[para_idx]))
                    emb_scored.append((para_idx, emb_sim))
                emb_scored.sort(key=lambda x: -x[1])
                candidates = _reciprocal_rank_fusion([candidates, emb_scored], k=60)
            except Exception:
                log.debug("Embedding rerank failed", exc_info=True)

        # Stage 3: Cross-encoder rerank on top candidates
        ce_candidates = candidates[: top_k * 3]
        if ce_candidates:
            reranked_candidates = self._cross_encoder_rerank(query, ce_candidates)
            if reranked_candidates:
                candidates = reranked_candidates

        # Temporal sorting: for temporal queries, sort by document date
        if intent["type"] == "temporal" and candidates:
            q = query.lower()
            newest_first = any(w in q for w in ["recent", "latest", "last", "newest"])
            oldest_first = any(w in q for w in ["first", "earliest", "oldest", "original"])
            if newest_first or oldest_first:
                dated = []
                for para_idx, score in candidates:
                    doc_idx = (
                        self.paragraph_index[para_idx][0]
                        if para_idx < len(self.paragraph_index)
                        else 0
                    )
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
                                self.paragraph_index[pi][0] == di for pi, _ in candidates[:top_k]
                            )
                            if not already_in:  # pragma: no cover
                                # Find a paragraph index for this document
                                for pi, (d_idx, _) in enumerate(self.paragraph_index):
                                    if d_idx == di:
                                        candidates.insert(0, (pi, 2.0))
                                        break
                            break
            except Exception:  # pragma: no cover
                log.debug("Temporal fallback candidate injection failed", exc_info=True)

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
            results.append(
                SearchResult(
                    name=doc.name,
                    source=doc.source,
                    score=round(score, 4),
                    snippet=snippet,
                    paragraph_idx=p_idx,
                    answer=answer,
                )
            )
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
                    name=r.name,
                    source=r.source,
                    score=r.score,
                    snippet=r.snippet,
                    paragraph_idx=r.paragraph_idx,
                    answer=r.answer,
                    confidence=round(base_conf, 3),
                )

        if compiled_results:
            results = _merge_priority_results(compiled_results, results, top_k)

        # Cross-reference answer verification: boost confidence when answers agree
        if len(results) >= 2:
            results = _cross_reference_answers(results)

        # Temporal code execution for date arithmetic queries
        if intent["type"] == "temporal" and results:
            try:
                from temporal_graph import TemporalEvent, temporal_code_execute, parse_dates

                t_events = []
                for r in results[:5]:
                    result_doc_idx: int | None = next(
                        (di for di, d in enumerate(self.documents) if d.name == r.name), None
                    )
                    if result_doc_idx is not None:
                        p_text = (
                            self.documents[result_doc_idx].paragraphs[r.paragraph_idx]
                            if r.paragraph_idx < len(self.documents[result_doc_idx].paragraphs)
                            else ""
                        )
                        for d in parse_dates(p_text):
                            t_events.append(
                                TemporalEvent(
                                    date=d,
                                    text=p_text[:200],
                                    source=r.name,
                                    paragraph_idx=r.paragraph_idx,
                                )
                            )
                if t_events:
                    code_answer = temporal_code_execute(query, t_events)
                    if (
                        code_answer and not results[0].answer
                    ):  # pragma: no cover — answer_extractor usually fills first
                        results[0] = SearchResult(
                            name=results[0].name,
                            source=results[0].source,
                            score=results[0].score,
                            snippet=results[0].snippet,
                            paragraph_idx=results[0].paragraph_idx,
                            answer=code_answer,
                        )
            except Exception:  # pragma: no cover
                log.debug("Temporal code execution failed", exc_info=True)

        # Multi-paragraph synthesis: combine top results into a grounded answer
        if use_llm and results and not results[0].answer:  # pragma: no cover
            try:
                from answer_extractor import llm_synthesize_answer

                paras = [r.snippet for r in results[:3]]
                synthesized = llm_synthesize_answer(query, paras)
                if synthesized:
                    results[0] = SearchResult(
                        name=results[0].name,
                        source=results[0].source,
                        score=results[0].score,
                        snippet=results[0].snippet,
                        paragraph_idx=results[0].paragraph_idx,
                        answer=synthesized,
                    )
            except ImportError:  # pragma: no cover
                pass

        return results

    def _multi_hop_search(
        self, original_query: str, sub_queries: list[str], top_k: int = 5, **kwargs: Any
    ) -> list[SearchResult]:
        """Run sub-queries, combine results, re-rank by original query."""
        all_results: dict[str, SearchResult] = {}
        for sq in sub_queries:
            results = self.search(sq, top_k=top_k, _decompose=False, **kwargs)
            for r in results:
                if r.name not in all_results or r.score > all_results[r.name].score:
                    all_results[r.name] = r

        # Re-rank combined results by relevance to original query
        q_tokens = set(_tokenize(original_query))
        scored: list[tuple[SearchResult, float]] = []
        for r in all_results.values():
            combined_text = (r.snippet + " " + r.answer).lower()
            overlap = sum(1 for t in q_tokens if t in combined_text)
            scored.append((r, r.score + overlap * 0.2))
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[:top_k]]

    def _single_query_search(
        self,
        query: str,
        top_k: int = 5,
        project: str = "",
        after: str = "",
        before: str = "",
        doc_type: str = "",
        use_llm: bool = False,
    ) -> list[SearchResult]:
        """Single query search (no decomposition). Used by _multi_hop_search."""
        return self.search(
            query,
            top_k=top_k,
            project=project,
            after=after,
            before=before,
            doc_type=doc_type,
            use_llm=use_llm,
            _decompose=False,
        )

    def _search_python_bm25(self, q_tokens: set[str], filtered_out: set[int]) -> dict[int, float]:
        if self._bm25_weight_dirty:
            self._build_bm25_weight_index()

        weight_index = self._bm25_weight_index
        if weight_index:
            candidate_scores: dict[int, float] = {}
            for qt in q_tokens:
                for para_idx, weight in weight_index.get(qt, ()):
                    if para_idx in filtered_out:
                        continue
                    candidate_scores[para_idx] = candidate_scores.get(para_idx, 0.0) + weight
            return candidate_scores

        return self._score_python_bm25_uncached(q_tokens, filtered_out)

    def _build_bm25_weight_index(self) -> dict[str, list[tuple[int, float]]]:
        k1, b = 1.5, 0.75
        avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0
        para_lengths = self._para_lengths
        token_counts = self.paragraph_token_counts
        weight_index: dict[str, list[tuple[int, float]]] = {}

        for token, posting in self._inverted_index.items():
            idf_val = self.idf.get(token, 0.0)
            if idf_val == 0:
                continue
            weighted_posting: list[tuple[int, float]] = []
            for para_idx in posting:
                if para_idx >= len(para_lengths):
                    continue
                tf = token_counts[para_idx].get(token, 1) if para_idx < len(token_counts) else 1
                dl_ratio = float(para_lengths[para_idx]) / avg_dl
                weight = idf_val * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl_ratio))
                weighted_posting.append((para_idx, weight))
            if weighted_posting:
                weight_index[token] = weighted_posting

        self._bm25_weight_index = weight_index
        self._bm25_weight_dirty = False
        return weight_index

    def _score_python_bm25_uncached(
        self, q_tokens: set[str], filtered_out: set[int]
    ) -> dict[int, float]:
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
            if idf_val == 0:  # pragma: no cover — defensive: idf should never be exactly zero
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

    def _ensure_rust_bm25(self) -> Any | None:
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

    def _get_answer_extractor(self) -> AnswerExtractor | None:
        if self._answer_extractor is None:
            try:
                from answer_extractor import extract_answer

                self._answer_extractor = extract_answer
            except ImportError:  # pragma: no cover
                self._answer_extractor = False
        return self._answer_extractor if callable(self._answer_extractor) else None

    def _get_llm_answer_extractor(self) -> AnswerExtractor | None:
        if self._llm_answer_extractor is None:
            try:
                from answer_extractor import llm_extract_answer

                self._llm_answer_extractor = llm_extract_answer
            except ImportError:  # pragma: no cover
                self._llm_answer_extractor = False
        return self._llm_answer_extractor if callable(self._llm_answer_extractor) else None

    @staticmethod
    def _load_content_hashes(path: Path | None = None) -> dict[str, str]:
        """Load SHA-256 content hashes from the cache file."""
        path = path or HASH_CACHE_PATH
        if not path.exists():
            return {}
        try:
            return cast(dict[str, str], json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _save_content_hashes(hashes: dict[str, str], path: Path | None = None) -> None:
        """Persist SHA-256 content hashes for the next incremental build."""
        path = path or HASH_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hashes), encoding="utf-8")

    def save(self, path: Path | None = None, quantize: bool = True) -> None:
        """Save index to disk as JSON+gzip (metadata) + npz (embeddings).

        Quantizes embeddings to int8 by default (~4x smaller).
        """
        path = path or INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        emb_data: np.ndarray | None = None
        emb_scale: np.ndarray | None = None
        embeddings = self.embeddings
        has_emb = embeddings is not None
        if embeddings is not None:
            if quantize:
                scale = np.max(np.abs(embeddings), axis=1, keepdims=True)
                scale = np.where(scale == 0, 1.0, scale)
                emb_data = (embeddings / scale * 127).astype(np.int8)
                emb_scale = scale.astype(np.float32)
            else:
                emb_data = embeddings

        meta = {
            "documents": [
                (d.name, d.source, d.path, d.paragraphs, d.date, d.doc_type) for d in self.documents
            ],
            "paragraph_index": self.paragraph_index,
            "paragraph_tokens": [list(t) for t in self.paragraph_tokens],
            "paragraph_token_counts": self.paragraph_token_counts,
            "paragraph_types": self.paragraph_types,
            "idf": self.idf,
            "_df": self._df,
            "quantized": quantize and has_emb,
            "timestamp": time.time(),
        }

        # Atomic write: temp file + rename
        tmp = path.with_suffix(".tmp")
        raw = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with gzip.open(tmp, "wb") as f:
            f.write(raw)
        tmp.replace(path)

        # Save embeddings separately as npz
        stem = path.stem.replace(".json", "")
        emb_path = path.with_name(stem + "_embeddings.npz")
        if emb_data is not None:
            arrays: dict[str, np.ndarray] = {"embeddings": emb_data}
            if emb_scale is not None:
                arrays["emb_scale"] = emb_scale
            cast(Any, np.savez_compressed)(emb_path, **arrays)
        elif emb_path.exists():
            emb_path.unlink()

    def load(self, path: Path | None = None) -> bool:
        """Load index from disk (JSON+gzip or legacy pickle)."""
        path = path or INDEX_PATH
        data = self._load_index_data(path)
        if data is None:
            return False
        try:
            self.documents = []
            for entry in data["documents"]:
                if len(entry) == 6:
                    n, s, p, paras, date, dtype = entry
                    self.documents.append(
                        Document(
                            name=n, source=s, path=p, paragraphs=paras, date=date, doc_type=dtype
                        )
                    )
                else:
                    n, s, p, paras = entry
                    self.documents.append(Document(name=n, source=s, path=p, paragraphs=paras))
            self.paragraph_index = data["paragraph_index"]
            self.paragraph_tokens = [set(t) for t in data["paragraph_tokens"]]
            self.paragraph_token_counts = data.get("paragraph_token_counts", [])
            if not self.paragraph_token_counts:
                self.paragraph_token_counts = [
                    {t: 1 for t in tokens} for tokens in self.paragraph_tokens
                ]
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
            self._avg_dl = (
                float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0
            )
            self._bm25_weight_index = {}
            self._bm25_weight_dirty = True
            self._rust_bm25_dirty = True
            self._rust_bm25 = None
            # Load embeddings
            emb_data = data.get("embeddings")
            emb_scale = data.get("emb_scale")
            self.embeddings = self._validated_loaded_embeddings(
                emb_data,
                emb_scale,
                quantized=bool(data.get("quantized")),
            )
            self._built = True
            return True
        except Exception:
            return False

    def _validated_loaded_embeddings(
        self,
        emb_data: Any,
        emb_scale: Any,
        *,
        quantized: bool,
    ) -> np.ndarray | None:
        if emb_data is None:
            return None
        try:
            embeddings: np.ndarray = np.asarray(emb_data)
            if quantized:
                if emb_scale is None:
                    return None
                scale = np.asarray(emb_scale, dtype=np.float32)
                embeddings = (embeddings.astype(np.float32) / 127.0) * scale
            else:
                embeddings = embeddings.astype(np.float32, copy=False)
        except Exception:
            log.debug("Loaded embedding sidecar has invalid array data", exc_info=True)
            return None

        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.paragraph_index):
            log.debug(
                "Ignoring embedding sidecar with shape %s for %d paragraphs",
                embeddings.shape,
                len(self.paragraph_index),
            )
            return None
        if not np.isfinite(embeddings).all():
            log.debug("Ignoring embedding sidecar containing non-finite values")
            return None
        return embeddings

    def _load_index_data(self, path: Path) -> dict[str, Any] | None:
        """Load index data from JSON+gzip (new) or pickle (legacy).

        Format is detected by file magic bytes (gzip = 0x1f8b), not extension.
        """
        if not path.exists():
            # Check legacy path as fallback
            if path == INDEX_PATH and _LEGACY_INDEX_PATH.exists():
                path = _LEGACY_INDEX_PATH
            else:
                return None
        # Detect format by magic bytes
        try:
            with open(path, "rb") as f:
                magic = f.read(2)
        except Exception:
            return None
        if magic == b"\x1f\x8b":
            # gzip JSON format
            try:
                with gzip.open(path, "rb") as f:
                    meta = cast(dict[str, Any], json.loads(f.read()))
                # Load embeddings from companion npz
                stem = path.stem.replace(".json", "")
                emb_path = path.with_name(stem + "_embeddings.npz")
                if emb_path.exists():
                    try:
                        with np.load(emb_path, allow_pickle=False) as emb:
                            meta["embeddings"] = emb.get("embeddings")
                            meta["emb_scale"] = emb.get("emb_scale")
                    except Exception:
                        log.debug("Embedding sidecar load failed: %s", emb_path, exc_info=True)
                return meta
            except Exception:
                return None
        # Legacy pickle no longer accepted. Caller gets None and the
        # migrator message surfaces in the outer load() diagnostics.
        return None


# ── Entity graph for retrieval boosting ──────────────────────────

_ENTITY_GRAPH: EntityGraph | None = None
_ENTITY_GRAPH_SIGNATURE: tuple[tuple[str, int | None, int | None], ...] | None = None


def _entity_graph_signature() -> tuple[tuple[str, int | None, int | None], ...]:
    signature: list[tuple[str, int | None, int | None]] = []
    for path in (GRAPH_DIR / "entities.jsonl", GRAPH_DIR / "relations.jsonl"):
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(path), None, None))
    return tuple(signature)


def _load_entity_graph() -> EntityGraph:
    """Load entities + relations from JSONL. Cached after first call."""
    global _ENTITY_GRAPH, _ENTITY_GRAPH_SIGNATURE
    signature = _entity_graph_signature()
    if _ENTITY_GRAPH is not None and signature == _ENTITY_GRAPH_SIGNATURE:
        return _ENTITY_GRAPH
    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"
    entities: dict[str, Entity] = {}
    relations: list[Entity] = []
    if entities_path.exists():
        for line in entities_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                e = cast(Entity, json.loads(line))
                entities[str(e["id"])] = e
    if relations_path.exists():
        for line in relations_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                relations.append(cast(Entity, json.loads(line)))
    _ENTITY_GRAPH = {
        "entities": entities,
        "relations": relations,
        "relation_neighbors": _build_relation_neighbors(entities, relations),
    }
    _ENTITY_GRAPH_SIGNATURE = signature
    return _ENTITY_GRAPH


# ── Query intelligence ────────────────────────────────────────────


def _classify_query(query: str) -> dict[str, Any]:
    """Classify query intent for search routing."""
    q = query.lower()
    intent: dict[str, Any] = {
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
    try:
        _rust_cls = import_module("remanentia_search").classify_paragraph

        return cast(str, _rust_cls(text, is_code))  # pragma: no cover
    except ImportError:
        pass
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
        text,
        re.I,
    ):
        activity = m.group(1).strip().lower()
        queries.append(f"hobbies {activity}")
        queries.append(f"interests {activity}")
        queries.append(f"what does {caps[0].lower() if caps else 'the person'} like")
        queries.append(activity)

    # 4. Occupation/role ("works as", "is a", "employed at")
    for m in re.finditer(
        r"(?:works? (?:as|at|for)|employed (?:at|by)|is a |job (?:is|as))\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        role = m.group(1).strip().lower()
        queries.append(f"where does {caps[0].lower() if caps else 'the person'} work")
        queries.append(f"job {role}")
        queries.append(f"career {role}")
        queries.append(role)

    # 5. Relationships ("married to", "friends with", "dating")
    for m in re.finditer(
        r"(?:married to|dating|friends? with|partner|spouse|sibling|brother|sister)\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        queries.append(f"relationship status")
        queries.append(f"who is {caps[0].lower() if caps else 'the person'} dating")

    # 6. Allergies, health, restrictions
    for m in re.finditer(
        r"(?:allergic to|allergy|intolerant|sensitive to|cannot eat|vegetarian|vegan)\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        subject = m.group(1).strip().lower()
        queries.append(f"allergic {subject}")
        queries.append(f"what is {caps[0].lower() if caps else 'the person'} allergic to")

    # 7. Travel/location ("went to", "visited", "lives in", "from")
    for m in re.finditer(
        r"(?:went to|visited|trip to|lives? in|moved to|from|travel(?:led|ed)? to)\s+(.{3,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        place = m.group(1).strip().lower()
        queries.append(f"where did {caps[0].lower() if caps else 'the person'} go")
        queries.append(f"trip {place}")
        queries.append(place)

    # 8. Learning/skills ("learning", "studying", "started")
    for m in re.finditer(
        r"(?:learning|studying|started|taking up|practicing)\s+(.{3,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        skill = m.group(1).strip().lower()
        queries.append(f"what is {caps[0].lower() if caps else 'the person'} learning")
        queries.append(skill)

    # 9. Favourites ("favourite", "favorite")
    for m in re.finditer(
        r"(?:favou?rite)\s+(\w+)\s+(?:is|was)\s+(.{3,40}?)(?:[.,;!?\n]|$)", text, re.I
    ):
        queries.append(f"favourite {m.group(1).lower()}")
        queries.append(m.group(2).strip().lower())
    for m in re.finditer(r"(?:favou?rite)\s+(.{3,40}?)(?:[.,;!?\n]|$)", text, re.I):
        queries.append(f"favourite {m.group(1).strip().lower()}")

    # 10. Decision/finding/metric type-specific (original patterns)
    if para_type == "decision":
        subjects = re.findall(
            r"(?:decided|chose|rejected|will)\s+(?:to\s+)?(.{10,40}?)(?:\.|,|$)", text, re.I
        )
        for s in subjects[:2]:
            queries.append(f"why did we {s.strip().lower()}")
            queries.append(f"what did we decide about {s.strip().lower()}")
    if para_type == "finding":
        queries.append(f"what did we find about {doc_name.replace('.md', '').replace('_', ' ')}")
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


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25. Lowercase, 3+ char words."""
    try:
        _rust_tok = import_module("remanentia_search").tokenize

        return cast(list[str], _rust_tok(text))  # pragma: no cover
    except ImportError:
        pass
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
                    name=old.name,
                    source=old.source,
                    score=old.score,
                    snippet=old.snippet,
                    paragraph_idx=old.paragraph_idx,
                    answer=old.answer,
                    confidence=round(boosted, 3),
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
        r"what\s+([^?]{1,80}?)\s+(?:does|did|do)\s+the\s+(?:person|one|guy|woman|man)\s+who\s+([^?]{1,80}?)(?:\s+have|\s+do|\s+like|\?|$)",
        q,
        re.I,
    )
    if m:
        return [f"who {m.group(2).strip()}", f"what {m.group(1).strip()}"]

    # "Does the person who X also Y?"
    m = re.match(
        r"(?:does|did|do)\s+the\s+(?:person|one)\s+who\s+([^?]{1,80}?)\s+(?:also\s+)?([^?]{1,80}?)(?:\?|$)",
        q,
        re.I,
    )
    if m:
        return [f"who {m.group(1).strip()}", m.group(2).strip()]

    # "What happened before/after X did Y?"
    m = re.match(r"what\s+happened\s+(before|after)\s+([^?]{1,80}?)(?:\?|$)", q, re.I)
    if m:
        return [m.group(2).strip(), f"what happened {m.group(1)} {m.group(2).strip()}"]

    return None


def _extract_lookup_terms(query: str) -> set[str]:
    return {
        token
        for token in _tokenize(query)
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
    try:
        _rust_rrf = import_module("remanentia_retrieve").reciprocal_rank_fusion

        return cast(list[tuple[int, float]], _rust_rrf(ranked_lists, k))  # pragma: no cover
    except ImportError:
        pass
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (para_idx, _score) in enumerate(ranked):
            rrf_scores[para_idx] = rrf_scores.get(para_idx, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return result


def _get_rust_bm25_class() -> Any | None:
    global _RUST_BM25_CLASS, _RUST_BM25_IMPORT_ATTEMPTED
    if not _RUST_BM25_IMPORT_ATTEMPTED:
        _RUST_BM25_IMPORT_ATTEMPTED = True
        try:
            _RUST_BM25_CLASS = import_module("remanentia_search").BM25Index
        except Exception:
            _RUST_BM25_CLASS = False
    return _RUST_BM25_CLASS if _RUST_BM25_CLASS else None


def _source_roots(source_name: str, source_dir: Path) -> list[Path]:
    if source_name in {"repo_coordination", "webmaster_coordination"}:
        roots: list[Path] = []
        for root in source_dir.glob("*/.coordination"):
            roots.extend([root / "sessions", root / "handovers"])
        return [root for root in roots if root.exists()]
    return [source_dir]


def _iter_source_files(source_name: str, source_dir: Path) -> Iterator[Path]:
    exts = SOURCE_EXTENSIONS.get(source_name, set(DEFAULT_TEXT_EXTENSIONS))
    files: list[Path] = []
    for root in _source_roots(source_name, source_dir):
        for ext in exts:
            files.extend(root.rglob(f"*{ext}"))
    for f in sorted(set(files)):
        if any(skip in str(f) for skip in SKIP_PATH_PARTS):
            continue
        if source_name == "manuscripts":
            normalised_path = f.as_posix()
            if any(skip in normalised_path for skip in MANUSCRIPT_SKIP_PATH_PARTS):
                continue
        yield f


def _should_index_text(text: str) -> bool:
    return MIN_FILE_CHARS <= len(text) <= MAX_FILE_CHARS


def needs_rebuild() -> bool:
    """Check if any source has newer files than the index."""
    if not INDEX_PATH.exists() and not _LEGACY_INDEX_PATH.exists():
        return True
    idx_file = INDEX_PATH if INDEX_PATH.exists() else _LEGACY_INDEX_PATH
    idx_mtime = idx_file.stat().st_mtime
    for source_name, source_dir in SOURCES.items():
        if not source_dir.exists():
            continue
        for f in _iter_source_files(source_name, source_dir):
            if f.stat().st_mtime > idx_mtime:
                return True
    return False


def auto_rebuild_if_needed(use_gpu: bool = True) -> MemoryIndex:
    """Load index, rebuild if sources changed."""
    idx = MemoryIndex()
    if idx.load() and not needs_rebuild():
        return idx
    idx.build(use_gpu_embeddings=use_gpu, use_gliner=False)
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
        stats = idx.build(
            use_gpu_embeddings=use_gpu, use_gliner=use_gliner, use_llm_indexing=use_llm_idx
        )
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
                        print(
                            f"Consolidated: {result['memories_written']} memories, "
                            f"{result['entities_found']} entities"
                        )
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
