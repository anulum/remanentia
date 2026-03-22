# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Unified Memory Index

"""Unified index over all GOTM knowledge sources.

Indexes 400+ documents from reasoning traces, session logs, handovers,
research documents, semantic memories, Claude memory files, disposition
files, and the INDEXER catalog.

Search: BM25 first pass + GPU embedding rerank.
Build: ~30s (scan + paragraph split + BM25 index + GPU embeddings).
Query: <100ms warm.

Usage::
    from memory_index import MemoryIndex
    idx = MemoryIndex()
    idx.build()  # first time, ~30s
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
    "traces": BASE / "reasoning_traces",
    "paper": BASE / "paper",
    "semantic": BASE / "memory" / "semantic",
    "sessions_as": GOTM_ROOT / ".coordination" / "sessions" / "arcane-sapience",
    "sessions_codex": GOTM_ROOT / ".coordination" / "sessions" / "CODEX",
    "handovers_as": GOTM_ROOT / ".coordination" / "handovers" / "arcane-sapience",
    "handovers_codex": GOTM_ROOT / ".coordination" / "handovers" / "codex",
    "qc_research": GOTM_ROOT / ".coordination" / "handovers" / "scpn-quantum-control",
    "po_research": GOTM_ROOT / ".coordination" / "handovers" / "scpn-phase-orchestrator",
    "nc_research": GOTM_ROOT / "03_CODE" / "sc-neurocore" / "docs" / "internal",
    "claude_memory": Path.home() / ".claude" / "projects" / "C--aaa-God-of-the-Math-Collection" / "memory",
    "disposition": BASE / "disposition",
}


@dataclass
class Document:
    name: str
    source: str
    path: str
    paragraphs: list[str] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    name: str
    source: str
    score: float
    snippet: str
    paragraph_idx: int = 0


class MemoryIndex:
    def __init__(self):
        self.documents: list[Document] = []
        self.paragraph_index: list[tuple[int, int]] = []  # (doc_idx, para_idx)
        self.paragraph_tokens: list[set[str]] = []
        self.idf: dict[str, float] = {}
        self.embeddings: np.ndarray | None = None
        self._built = False
        self._embed_model = None

    def build(self, use_gpu_embeddings: bool = True, use_gliner: bool = True) -> dict:
        """Scan all sources, build BM25 index + GPU embeddings + GLiNER entities."""
        t0 = time.monotonic()
        self.documents = []
        self.paragraph_index = []
        self.paragraph_tokens = []
        self.all_entities: list[dict] = []
        self.all_relations: list[dict] = []

        # GLiNER model (load once, reuse)
        gliner_model = None
        if use_gliner:
            try:
                from entity_extractor import extract_entities, extract_relations
                from entity_extractor import _load_gliner
                gliner_model = _load_gliner()
            except Exception:
                pass

        # Scan all sources
        for source_name, source_dir in SOURCES.items():
            if not source_dir.exists():
                continue
            for f in sorted(source_dir.rglob("*.md")):
                try:
                    text = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if len(text) < 50:
                    continue

                paragraphs = _split_paragraphs(text)
                all_tokens = set()
                for p in paragraphs:
                    all_tokens.update(_tokenize(p))

                # Entity extraction
                doc_entities = []
                doc_relations = []
                if gliner_model is not None:
                    try:
                        from entity_extractor import extract_entities, extract_relations
                        doc_entities = extract_entities(text[:3000])
                        doc_relations = extract_relations(text[:3000], doc_entities)
                    except Exception:
                        pass

                doc = Document(
                    name=f.name,
                    source=source_name,
                    path=str(f),
                    paragraphs=paragraphs,
                    tokens=all_tokens,
                )
                doc_idx = len(self.documents)
                self.documents.append(doc)

                for para_idx, para in enumerate(paragraphs):
                    self.paragraph_index.append((doc_idx, para_idx))
                    self.paragraph_tokens.append(set(_tokenize(para)))

        # Build IDF
        n_docs = len(self.paragraph_tokens)
        df: Counter = Counter()
        for tokens in self.paragraph_tokens:
            for t in tokens:
                df[t] += 1
        self.idf = {t: math.log(1 + n_docs / (1 + count))
                     for t, count in df.items()}

        # GPU embeddings if available
        if use_gpu_embeddings:
            try:
                self._compute_embeddings()
            except Exception:
                pass

        self._built = True
        elapsed = time.monotonic() - t0

        stats = {
            "documents": len(self.documents),
            "paragraphs": len(self.paragraph_index),
            "unique_tokens": len(self.idf),
            "has_embeddings": self.embeddings is not None,
            "build_time_s": round(elapsed, 1),
            "sources": {s: sum(1 for d in self.documents if d.source == s)
                        for s in SOURCES if any(d.source == s for d in self.documents)},
        }
        return stats

    def _compute_embeddings(self):
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

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search: BM25 first pass + embedding rerank."""
        if not self._built:
            self.build()

        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        # BM25 scoring over all paragraphs
        k1, b = 1.5, 0.75
        avg_dl = np.mean([len(t) for t in self.paragraph_tokens]) if self.paragraph_tokens else 1

        scores = []
        for i, p_tokens in enumerate(self.paragraph_tokens):
            score = 0.0
            dl = len(p_tokens)
            for qt in q_tokens:
                if qt not in p_tokens:
                    continue
                tf = 1  # binary for simplicity (token is present)
                idf_val = self.idf.get(qt, 0)
                score += idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: -x[1])

        # Take top candidates for embedding rerank
        candidates = scores[:top_k * 3] if self.embeddings is not None else scores[:top_k]

        # Embedding rerank if available
        if self.embeddings is not None and candidates:
            try:
                q_emb = self._embed_model.encode(
                    query, normalize_embeddings=True, convert_to_numpy=True)
                for j, (para_idx, bm25_score) in enumerate(candidates):
                    emb_sim = float(np.dot(q_emb, self.embeddings[para_idx]))
                    # Combined: 0.4 BM25 + 0.6 embedding
                    candidates[j] = (para_idx, 0.4 * bm25_score + 0.6 * emb_sim)
                candidates.sort(key=lambda x: -x[1])
            except Exception:
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
            results.append(SearchResult(
                name=doc.name,
                source=doc.source,
                score=round(score, 4),
                snippet=snippet,
                paragraph_idx=p_idx,
            ))
            if len(results) >= top_k:
                break

        return results

    def save(self, path: Path | None = None):
        """Save index to disk (without embeddings — recompute on load)."""
        path = path or INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [(d.name, d.source, d.path, d.paragraphs) for d in self.documents],
            "paragraph_index": self.paragraph_index,
            "paragraph_tokens": [list(t) for t in self.paragraph_tokens],
            "idf": self.idf,
            "timestamp": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path | None = None) -> bool:
        """Load index from disk."""
        path = path or INDEX_PATH
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.documents = [
                Document(name=n, source=s, path=p, paragraphs=paras)
                for n, s, p, paras in data["documents"]
            ]
            self.paragraph_index = data["paragraph_index"]
            self.paragraph_tokens = [set(t) for t in data["paragraph_tokens"]]
            self.idf = data["idf"]
            self._built = True
            return True
        except Exception:
            return False


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs. Each becomes a searchable unit."""
    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if len(stripped) > 30:
            paragraphs.append(stripped)
    if not paragraphs and len(text.strip()) > 30:
        paragraphs.append(text.strip()[:2000])
    return paragraphs


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25. Lowercase, 3+ char words."""
    return re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower())


def needs_rebuild() -> bool:
    """Check if any source has newer files than the index."""
    if not INDEX_PATH.exists():
        return True
    idx_mtime = INDEX_PATH.stat().st_mtime
    for source_dir in SOURCES.values():
        if not source_dir.exists():
            continue
        for f in source_dir.rglob("*.md"):
            if f.stat().st_mtime > idx_mtime:
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
        print("Building index...")
        stats = idx.build(use_gpu_embeddings=use_gpu, use_gliner=use_gliner)
        print(json.dumps(stats, indent=2))
        idx.save()
        print(f"Saved to {INDEX_PATH}")

    elif "--watch" in sys.argv:
        print("Watching for changes... (Ctrl+C to stop)")
        interval = 60
        while True:
            if needs_rebuild():
                print(f"Changes detected, rebuilding...")
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
