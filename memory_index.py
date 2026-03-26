# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
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

import hashlib
import json
import math
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


class MemoryIndex:
    def __init__(self):
        self.documents: list[Document] = []
        self.paragraph_index: list[tuple[int, int]] = []  # (doc_idx, para_idx)
        self.paragraph_tokens: list[set[str]] = []
        self.paragraph_types: list[str] = []  # function, decision, finding, etc.
        self.idf: dict[str, float] = {}
        self._inverted_index: dict[str, list[int]] = {}  # token → paragraph indices
        self.embeddings: np.ndarray | None = None
        self._built = False
        self._embed_model = None
        self._cross_encoder = None
        self._ce_loading = False
        self._embed_loading = False
        self._temporal_graph = None
        self._avg_dl: float = 1.0

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
                from entity_extractor import extract_entities, extract_relations
                from entity_extractor import _load_gliner
                gliner_model = _load_gliner()
            except Exception:
                pass

        # Scan all sources
        for source_name, source_dir in SOURCES.items():
            if not source_dir.exists():  # pragma: no cover
                continue
            # Determine which extensions to index
            exts = SOURCE_EXTENSIONS.get(source_name, {".md"})
            files = []
            for ext in exts:
                files.extend(source_dir.rglob(f"*{ext}"))
            # Skip venvs, node_modules, __pycache__, .git
            files = [f for f in sorted(set(files))
                     if not any(skip in str(f) for skip in
                                [".venv", "venv", "node_modules", "__pycache__",
                                 ".git", "target", "dist", ".egg"])]
            for f in files:
                try:
                    text = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):  # pragma: no cover
                    continue
                if len(text) < 50 or len(text) > 1_000_000:
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
                    # Include prospective query tokens in the searchable set
                    p_type = _classify_paragraph(para, is_code=is_code)
                    pq = _generate_prospective_queries(para, f.name, p_type)
                    combined_text = para + " " + " ".join(pq)
                    tokens_with_pq = set(_tokenize(combined_text))
                    self.paragraph_tokens.append(tokens_with_pq)
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
        self.idf = {t: math.log(1 + n_docs / (1 + count))
                     for t, count in df.items()}
        self._para_lengths = np.array([len(t) for t in self.paragraph_tokens], dtype=np.float32)
        self._avg_dl = float(np.mean(self._para_lengths)) if len(self._para_lengths) > 0 else 1.0

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
            tokens = set(_tokenize(combined))
            self.paragraph_tokens.append(tokens)
            self.paragraph_types.append(p_type)
            for t in tokens:
                if t not in self.idf:
                    self.idf[t] = math.log(1 + len(self.paragraph_tokens) / 2)
                if t not in self._inverted_index:
                    self._inverted_index[t] = []
                self._inverted_index[t].append(p_idx)

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

        return len(paragraphs)

    def _start_model_warmup(self):
        """Start loading embedding + cross-encoder models in background."""
        import threading
        if self._embed_model is None and not self._embed_loading and self.embeddings is not None:
            self._embed_loading = True
            def _load_embed():
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
            def _load_ce():
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

        self._start_model_warmup()

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

        # BM25 scoring via inverted index (only visit paragraphs containing query tokens)
        k1, b = 1.5, 0.75
        avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0

        candidate_scores: dict[int, float] = {}
        para_lengths = self._para_lengths
        for qt in q_tokens:
            posting = self._inverted_index.get(qt)
            if not posting:
                continue
            idf_val = self.idf.get(qt, 0)
            if idf_val == 0:
                continue
            for i in posting:
                if i in _filtered_out:
                    continue
                dl_ratio = para_lengths[i] / avg_dl
                candidate_scores[i] = candidate_scores.get(i, 0.0) + (
                    idf_val * (k1 + 1) / (1 + k1 * (1 - b + b * dl_ratio)))

        # Entity-centric person-name boost (bench_locomo: +3-5pp)
        query_names = _extract_query_names(query) if _is_person_centric(query) else set()

        # Apply boosts to candidates with score > 0
        scores = []
        for i, score in candidate_scores.items():
            if boost_types and i < len(self.paragraph_types):
                if self.paragraph_types[i] in boost_types:
                    score *= 1.5

            if intent.get("recency"):
                doc_idx = self.paragraph_index[i][0] if i < len(self.paragraph_index) else 0
                if doc_idx < len(self.documents) and self.documents[doc_idx].date:
                    score *= _recency_boost(self.documents[doc_idx].date)

            if intent["type"] == "temporal":
                doc_idx = self.paragraph_index[i][0] if i < len(self.paragraph_index) else 0
                para_idx = self.paragraph_index[i][1] if i < len(self.paragraph_index) else 0
                if doc_idx < len(self.documents):
                    para_text = self.documents[doc_idx].paragraphs[para_idx] if para_idx < len(self.documents[doc_idx].paragraphs) else ""
                    if _has_date_expression(para_text):
                        score *= 1.4

            if query_names:
                doc_idx = self.paragraph_index[i][0] if i < len(self.paragraph_index) else 0
                para_idx = self.paragraph_index[i][1] if i < len(self.paragraph_index) else 0
                if doc_idx < len(self.documents):
                    para_text = self.documents[doc_idx].paragraphs[para_idx].lower() if para_idx < len(self.documents[doc_idx].paragraphs) else ""
                    if any(name in para_text for name in query_names):
                        score *= 1.3

            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])

        # Stage 1: Take top candidates for embedding rerank
        candidates = scores[:top_k * 6] if self.embeddings is not None else scores[:top_k * 3]

        # Stage 2: Bi-encoder rerank if model already loaded
        if self.embeddings is not None and self._embed_model is not None and candidates:  # pragma: no cover
            try:
                q_emb = self._embed_model.encode(
                    query, normalize_embeddings=True, convert_to_numpy=True)
                for j, (para_idx, bm25_score) in enumerate(candidates):
                    emb_sim = float(np.dot(q_emb, self.embeddings[para_idx]))
                    candidates[j] = (para_idx, 0.4 * bm25_score + 0.6 * emb_sim)
                candidates.sort(key=lambda x: -x[1])
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
        _answer_extractor = None
        try:
            from answer_extractor import extract_answer
            _answer_extractor = extract_answer
        except ImportError:  # pragma: no cover
            pass

        _llm_extractor = None
        if use_llm:
            try:
                from answer_extractor import llm_extract_answer
                _llm_extractor = llm_extract_answer
            except ImportError:  # pragma: no cover
                pass

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
            "paragraph_types": self.paragraph_types,
            "idf": self.idf,
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
            self.idf = data["idf"]
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

    This is the Kumiho technique (March 2026, 98.5% recall).
    Pre-answer questions at write time so retrieval becomes lookup.
    """
    queries = []

    # Extract key nouns/phrases
    # Capitalized phrases (likely names, projects, concepts)
    caps = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    for c in caps[:5]:
        if len(c) > 3:
            queries.append(f"what is {c}")
            queries.append(c.lower())

    # Function names
    funcs = re.findall(r"(?:def |fn |class )\s*(\w+)", text)
    for f in funcs[:3]:
        queries.append(f"where is {f}")
        queries.append(f"how does {f} work")
        queries.append(f)

    # If it's a decision, generate "why" and "what did we decide" variants
    if para_type == "decision":
        # Extract the subject
        subjects = re.findall(r"(?:decided|chose|rejected|will)\s+(?:to\s+)?(.{10,40}?)(?:\.|,|$)", text, re.I)
        for s in subjects[:2]:
            queries.append(f"why did we {s.strip().lower()}")
            queries.append(f"what did we decide about {s.strip().lower()}")

    # If it's a finding, generate "what did we find" variants
    if para_type == "finding":
        queries.append(f"what did we find about {doc_name.replace('.md','').replace('_',' ')}")

    # If it's a metric, generate "what is the score" variants
    if para_type == "metric":
        numbers = re.findall(r"\d+\.?\d*%", text)
        for n in numbers[:2]:
            queries.append(f"what score {n}")

    # File-based queries
    if ".py" in doc_name or ".rs" in doc_name:
        base = doc_name.split(".")[0]
        queries.append(f"what does {base} do")
        queries.append(f"where is {base}")

    return queries[:10]  # cap


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

    For markdown: paragraphs (text between blank lines).
    For code: functions/classes (def/fn/class blocks) + module docstring.
    """
    if is_code:
        return _split_code(text)

    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if len(stripped) > 30:
            paragraphs.append(stripped[:10_000])
    if not paragraphs and len(text.strip()) > 30:
        paragraphs.append(text.strip()[:2000])
    return paragraphs


def _split_code(text: str) -> list[str]:
    """Split code into function/class blocks for indexing."""
    chunks = []

    # Module docstring (first triple-quoted block)
    doc_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if doc_match and doc_match.start() < 500:
        chunks.append(doc_match.group(1).strip()[:500])

    # Python: def and class blocks
    for match in re.finditer(r'^((?:def|class|fn|pub fn|impl)\s+\w+.*?)(?=\n(?:def |class |fn |pub fn |impl |\Z))',
                              text, re.MULTILINE | re.DOTALL):
        block = match.group(1).strip()
        if len(block) > 30:
            # Keep first 500 chars (signature + docstring + start of body)
            chunks.append(block[:500])

    # If no functions found, treat as plain text
    if not chunks:
        for block in text.split("\n\n"):
            stripped = block.strip()
            if len(stripped) > 30:
                chunks.append(stripped[:500])

    return chunks[:50]  # cap at 50 chunks per file


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25. Lowercase, 3+ char words."""
    return re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower())


def needs_rebuild() -> bool:
    """Check if any source has newer files than the index."""
    if not INDEX_PATH.exists():
        return True
    idx_mtime = INDEX_PATH.stat().st_mtime
    for source_dir in SOURCES.values():
        if not source_dir.exists():  # pragma: no cover
            continue
        for f in source_dir.rglob("*.md"):
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
