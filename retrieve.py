# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Query-probe retrieval for persistent SNN

# Active retrieval stack: memory_index.py (BM25 + embedding + CE)
# This file kept for reference (70+ experiments in experimental/)

"""Query-probe retrieval system for the persistent SNN.

The SNN daemon accumulates STDP-modified synaptic weights from
reasoning traces and session stimuli. This module closes the loop
by using those weights as an associative memory.

Architecture::

    query text
        │
        ▼
    encode (backend selected from retrieval checkpoint metadata)
        │
        ▼
    deterministic 50-step LIF burst under W
        │
        ▼
    cosine similarity vs cached trace spike features ──► SNN affinity score
        │
        ▼
    TF-IDF + filename overlap + entity graph + paragraph embedding
        │
        ▼
    combined: 0.30 × keyword + 0.25 × name + 0.10 × graph + 0.45 × embedding
        │
        ▼
    ranked trace list with scores

Usage — CLI::

    python retrieve.py "disruption prediction"
    python retrieve.py "gyrokinetic" --top 5 --content
    python retrieve.py "sawtooth NTM" --context
    python retrieve.py --summary
    python retrieve.py --rebuild
    python retrieve.py --eval   # precision benchmark

Usage — Python API::

    from retrieve import retrieve, retrieve_context
    results = retrieve("scpn-control Dimits shift")
    context = retrieve_context("disruption mitigation", top_k=3)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .retrieval_catalog import (
        append_history as _append_history,
        chunk_trace_catalog as _chunk_trace_catalog,
        classify_trace_tier as _classify_trace_tier,
        find_related_traces as _find_related_traces,
        read_history as _read_history,
        suggest_queries as _suggest_queries,
        summarize_traces as _summarize_traces,
        tier_boost as _tier_boost,
    )
except ImportError:
    from retrieval_catalog import (
        append_history as _append_history,
        chunk_trace_catalog as _chunk_trace_catalog,
        classify_trace_tier as _classify_trace_tier,
        find_related_traces as _find_related_traces,
        read_history as _read_history,
        suggest_queries as _suggest_queries,
        summarize_traces as _summarize_traces,
        tier_boost as _tier_boost,
    )

try:
    from .retrieval_text import (
        STOPWORDS as _STOPWORDS,
        build_idf as _build_idf,
        expand_query as _expand_query,
        stem as _stem,  # noqa: F401 - retained as a compatibility surface
        tfidf_score as _tfidf_score,
        tokenize as _tokenize,
    )
except ImportError:
    from retrieval_text import (
        STOPWORDS as _STOPWORDS,
        build_idf as _build_idf,
        expand_query as _expand_query,
        stem as _stem,  # noqa: F401 - retained as a compatibility surface
        tfidf_score as _tfidf_score,
        tokenize as _tokenize,
    )

try:
    from .retrieval_network_io import (
        NetworkPaths as _NetworkPaths,
        load_checkpoint as _load_checkpoint_from_disk,
        load_network as _load_network_from_disk,
        normalize_backend as _normalize_network_backend,
        read_json as _read_json_file,
        resolve_network_config as _resolve_network_configuration,
        resolve_path as _resolve_network_path,
        write_json_atomic as _write_json_file_atomic,
    )
except ImportError:
    from retrieval_network_io import (
        NetworkPaths as _NetworkPaths,
        load_checkpoint as _load_checkpoint_from_disk,
        load_network as _load_network_from_disk,
        normalize_backend as _normalize_network_backend,
        read_json as _read_json_file,
        resolve_network_config as _resolve_network_configuration,
        resolve_path as _resolve_network_path,
        write_json_atomic as _write_json_file_atomic,
    )

try:
    from .retrieval_cache_io import (
        load_json_gz as _load_json_gz_file,
        load_npz_dict as _load_npz_cache,
        load_pickle_disabled as _load_disabled_pickle,
        load_query_feature_cache as _load_query_features_from_disk,
        load_trace_index_cache as _load_trace_index_from_disk,
        persist_query_feature_cache as _persist_query_features_to_disk,
        persist_trace_index_cache as _persist_trace_index_to_disk,
        save_json_gz as _save_json_gz_file,
        trace_fingerprint as _fingerprint_trace_files,
    )
except ImportError:
    from retrieval_cache_io import (
        load_json_gz as _load_json_gz_file,
        load_npz_dict as _load_npz_cache,
        load_pickle_disabled as _load_disabled_pickle,
        load_query_feature_cache as _load_query_features_from_disk,
        load_trace_index_cache as _load_trace_index_from_disk,
        persist_query_feature_cache as _persist_query_features_to_disk,
        persist_trace_index_cache as _persist_trace_index_to_disk,
        save_json_gz as _save_json_gz_file,
        trace_fingerprint as _fingerprint_trace_files,
    )

try:
    from .retrieval_entity_graph import (
        entity_graph_score as _score_entity_graph,
        load_entity_graph as _load_entity_graph,
    )
except ImportError:
    from retrieval_entity_graph import (
        entity_graph_score as _score_entity_graph,
        load_entity_graph as _load_entity_graph,
    )

try:
    from .retrieval_live_service import (
        load_live_service_config as _load_live_retrieval_config,
        retrieve_via_live_service as _run_live_retrieval,
    )
except ImportError:
    from retrieval_live_service import (
        load_live_service_config as _load_live_retrieval_config,
        retrieve_via_live_service as _run_live_retrieval,
    )

try:
    from .retrieval_spiking import (
        cosine_similarity as _cosine_sim,
        encode_text as _encode,
        snn_affinity as _snn_affinity,  # noqa: F401 - private compatibility surface
        spike_feature as _spike_feature,
    )
except ImportError:
    from retrieval_spiking import (
        cosine_similarity as _cosine_sim,
        encode_text as _encode,
        snn_affinity as _snn_affinity,  # noqa: F401 - private compatibility surface
        spike_feature as _spike_feature,
    )

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ArcSap.Retrieve")

BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "snn_state"
TRACES_DIR = BASE_DIR / "reasoning_traces"
SEMANTIC_DIR = BASE_DIR / "memory" / "semantic"
RETRIEVAL_STATE_PATH = STATE_DIR / "retrieval_state.json"
DEFAULT_NETWORK_PATH = STATE_DIR / "identity_net.pkl"
EMBED_NETWORK_PATH = STATE_DIR / "identity_net_embedding_trained.pkl"
EMBED_CACHE_PATH = STATE_DIR / "embedding_cache.npz"
FINGERPRINT_PATH = STATE_DIR / "trace_fingerprints.json"
TRACE_INDEX_CACHE_PATH = STATE_DIR / "trace_index_cache.json.gz"
QUERY_FEATURE_CACHE_PATH = STATE_DIR / "query_feature_cache.json.gz"
# Legacy paths for backward-compatible loading
_LEGACY_EMBED_CACHE = STATE_DIR / "embedding_cache.pkl"
_LEGACY_TRACE_CACHE = STATE_DIR / "trace_index_cache.pkl"
_LEGACY_QUERY_CACHE = STATE_DIR / "query_feature_cache.pkl"
HISTORY_PATH = STATE_DIR / "retrieval_history.jsonl"
LIVE_REQUESTS_DIR = STATE_DIR / "live_retrieval_requests"
LIVE_RESPONSES_DIR = STATE_DIR / "live_retrieval_responses"


def _save_json_gz(path: Path, data: dict) -> None:
    """Save dict as gzipped JSON."""
    _save_json_gz_file(path, data)


def _load_json_gz(path: Path) -> dict | None:
    """Load dict from gzipped JSON, returning None on failure."""
    return _load_json_gz_file(path)


def _load_pickle_safe(path: Path) -> dict | None:
    """Legacy pickle loader, disabled 2026-04-17 for security.

    Kept as a named function so callers still compile; always returns
    None. Operators with legacy .pkl files must convert them first via
    ``python tools/migrate_pickle_to_npz.py``.
    """
    return _load_disabled_pickle(path)


def _load_npz_dict(path: Path) -> dict | None:
    """Load npz as dict[str, ndarray], returning None on failure."""
    return _load_npz_cache(path)


_PROBE_DURATION = 0.1
_PROBE_DT = 0.001
_EMBED_RERANK_MIN = 8
_EMBED_RERANK_FACTOR = 3
_WEIGHT_KW = 0.30
_WEIGHT_SNN = 0.00  # proven to add zero across 70+ experiments
_WEIGHT_NAME = 0.25
_WEIGHT_EMB = 0.45  # best-paragraph embedding is the strongest signal
_LIVE_RETRIEVAL_TIMEOUT_S = 30.0
_LIVE_SERVICE_STALE_S = 300.0

_NETWORK_CACHE: dict[tuple[str, int, int, str], dict] = {}
_TRACE_INDEX_CACHE: dict[str, dict] = {}
_QUERY_FEATURE_CACHE: dict[tuple[str, str], dict] = {}
_EMBED_CACHE_LOADED = False
_QUERY_FEATURE_CACHE_LOADED = False


# ── Network I/O ───────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    return _read_json_file(path)


def _normalize_backend(backend: str | None) -> str:
    return _normalize_network_backend(backend)


def _resolve_path(path_text: str) -> Path:
    return _resolve_network_path(path_text, base_dir=BASE_DIR)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_json_file_atomic(path, payload)


def _infer_backend_from_path(path: Path) -> str:
    return "embedding" if "embedding" in path.stem else "lsh"


def _activate_backend(backend: str) -> None:
    try:
        from encoding import get_backend, set_backend
    except ImportError:
        return
    normalized = _normalize_backend(backend)
    if get_backend() != normalized:
        set_backend(normalized)


def _resolve_network_config(state_path: Path | None = None) -> dict:
    return _resolve_network_configuration(
        _NetworkPaths(
            base_dir=BASE_DIR,
            state_dir=STATE_DIR,
            retrieval_state_path=RETRIEVAL_STATE_PATH,
            default_network_path=DEFAULT_NETWORK_PATH,
            embedding_network_path=EMBED_NETWORK_PATH,
        ),
        state_path,
    )


def _load_checkpoint(path: Path) -> dict:
    """Load network checkpoint from npz. Legacy pickle no longer accepted."""
    return _load_checkpoint_from_disk(path)


def _load_network(state_path: Path | None = None) -> dict:
    data = _load_network_from_disk(
        _NetworkPaths(
            base_dir=BASE_DIR,
            state_dir=STATE_DIR,
            retrieval_state_path=RETRIEVAL_STATE_PATH,
            default_network_path=DEFAULT_NETWORK_PATH,
            embedding_network_path=EMBED_NETWORK_PATH,
        ),
        _NETWORK_CACHE,
        state_path,
    )
    _activate_backend(str(data["_encoding_backend"]))
    return data


def _trace_fingerprint(trace_files: list[Path]) -> str:
    return _fingerprint_trace_files(trace_files)


def _load_trace_index_cache(cache_key: str) -> dict | None:
    return _load_trace_index_from_disk(TRACE_INDEX_CACHE_PATH, cache_key)


def _persist_trace_index_cache(cache_key: str, payload: dict) -> None:
    _persist_trace_index_to_disk(TRACE_INDEX_CACHE_PATH, cache_key, payload)


def _build_trace_index(tdir: Path, data: dict) -> dict:
    trace_files = sorted(tdir.glob("*.md"))
    # Include semantic memories in search corpus
    semantic_files = []
    if SEMANTIC_DIR.exists():
        semantic_files = sorted(SEMANTIC_DIR.rglob("*.md"))
    all_files = trace_files + semantic_files
    cache_key = (
        f"{data['_state_signature']}:{data['_encoding_backend']}:{_trace_fingerprint(all_files)}"
    )
    cached = _TRACE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    trace_texts: dict[str, str] = {}
    for tf in trace_files:
        trace_texts[tf.name] = tf.read_text(encoding="utf-8")
    for sf in semantic_files:
        rel = str(sf.relative_to(SEMANTIC_DIR))
        trace_texts[f"[semantic] {rel}"] = sf.read_text(encoding="utf-8")

    disk_cached = _load_trace_index_cache(cache_key)
    if disk_cached is not None:
        cached = {
            "trace_files": trace_files,
            "trace_texts": trace_texts,
            "trace_spikes": disk_cached["trace_spikes"],
            "trace_names_lower": disk_cached["trace_names_lower"],
            "idf": disk_cached["idf"],
        }
        _TRACE_INDEX_CACHE.clear()
        _TRACE_INDEX_CACHE[cache_key] = cached
        return cached

    n = len(data["v"])
    w = data["w"]
    trace_spikes: dict[str, np.ndarray] = {}
    trace_names_lower: dict[str, str] = {}

    for trace_name, text in trace_texts.items():
        stimulus = _encode(text, n)
        trace_spikes[trace_name] = _spike_feature(w, stimulus)
        trace_names_lower[trace_name] = trace_name.lower().replace("-", " ").replace("_", " ")

    cached = {
        "trace_files": trace_files,
        "trace_texts": trace_texts,
        "trace_spikes": trace_spikes,
        "trace_names_lower": trace_names_lower,
        "idf": _build_idf(trace_texts),
    }
    _persist_trace_index_cache(cache_key, cached)
    _TRACE_INDEX_CACHE.clear()
    _TRACE_INDEX_CACHE[cache_key] = cached
    return cached


def _load_query_feature_cache() -> None:
    global _QUERY_FEATURE_CACHE_LOADED
    if _QUERY_FEATURE_CACHE_LOADED:
        return
    _QUERY_FEATURE_CACHE.update(_load_query_features_from_disk(QUERY_FEATURE_CACHE_PATH))
    _QUERY_FEATURE_CACHE_LOADED = True


def _persist_query_feature_cache() -> None:
    _persist_query_features_to_disk(QUERY_FEATURE_CACHE_PATH, _QUERY_FEATURE_CACHE)


def _get_query_features(query: str, data: dict) -> dict:
    _load_query_feature_cache()
    cache_key = (data["_state_signature"], query)
    cached = _QUERY_FEATURE_CACHE.get(cache_key)
    if cached is None:
        stimulus = _encode(query, len(data["v"]))
        cached = {
            "stimulus": stimulus,
            "spikes": _spike_feature(data["w"], stimulus),
        }
        if len(_QUERY_FEATURE_CACHE) > 256:
            items = list(_QUERY_FEATURE_CACHE.items())[-255:]
            _QUERY_FEATURE_CACHE.clear()
            _QUERY_FEATURE_CACHE.update(items)
        _QUERY_FEATURE_CACHE[cache_key] = cached
        _persist_query_feature_cache()
    return cached


# ── History & summaries ───────────────────────────────────────────


def _log_retrieval(query: str, results: list[dict]) -> None:
    """Append query and results to retrieval history."""
    _append_history(HISTORY_PATH, query, results)


def retrieval_history(limit: int = 50) -> list[dict]:
    """Read recent retrieval history."""
    return _read_history(HISTORY_PATH, limit)


def trace_summaries(traces_dir: Path | None = None) -> list[dict]:
    """Extract 1-line summaries from each trace file.

    Takes the first non-heading, non-empty line after the title.
    """
    return _summarize_traces(traces_dir or TRACES_DIR)


# ── Tiered memory (hot/warm/cold) ─────────────────────────────────

def _trace_tier(path: Path) -> str:
    """Classify trace into hot/warm/cold based on modification time."""
    return _classify_trace_tier(path)


# ── Entity graph signal ──────────────────────────────────────────

_GRAPH_ENTITIES: dict | None = None
_GRAPH_RELATIONS: list | None = None


def _load_graph_once():
    """Lazy-load entity graph for retrieval boosting."""
    global _GRAPH_ENTITIES, _GRAPH_RELATIONS
    if _GRAPH_ENTITIES is not None:
        return
    _GRAPH_ENTITIES, _GRAPH_RELATIONS = _load_entity_graph(BASE_DIR / "memory" / "graph")


def _entity_graph_score(query: str, trace_name: str) -> float:
    """Score based on shared entity connections between query and trace.

    Finds entities mentioned in the query, then checks how many of those
    entities have graph connections to entities in the trace filename.
    """
    _load_graph_once()
    if not _GRAPH_ENTITIES or not _GRAPH_RELATIONS:
        return 0.0

    return _score_entity_graph(query, trace_name, _GRAPH_ENTITIES, _GRAPH_RELATIONS)


def _filename_bonus(query: str, name_lower: str, idf: dict[str, float]) -> float:
    """IDF-weighted filename overlap bonus.

    The filename is a dense topic label for a trace. Weighting overlap by IDF
    rewards rare discriminative matches such as `daemon` or `dimits` more than
    generic terms, and avoids overvaluing long queries with many unmatched words.
    """
    try:
        from remanentia_retrieve import filename_bonus as _rust_fb

        return _rust_fb(query, name_lower, idf, _STOPWORDS)  # pragma: no cover
    except ImportError:
        pass
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    total = sum(idf.get(token, 1.0) for token in q_tokens)
    if total <= 1e-12:
        return 0.0
    matched = sum(idf.get(token, 1.0) for token in q_tokens if token in name_lower)
    return matched / total


# ── Cross-trace linking ───────────────────────────────────────────


def related_traces(trace_name: str, top_k: int = 3, traces_dir: Path | None = None) -> list[dict]:
    """Find traces most similar to the given trace by keyword overlap.

    The `trace_name` is expected to be the name of a markdown trace file
    located directly in `TRACES_DIR` (for example, a value previously
    returned from `_list_traces`). To avoid directory traversal or
    unintended file access, the resolved path is constrained to the
    traces directory before being used.
    """
    return _find_related_traces(traces_dir or TRACES_DIR, trace_name, top_k)


def query_suggestions(prefix: str, limit: int = 8) -> list[str]:
    """Autocomplete suggestions from retrieval history."""
    return _suggest_queries(HISTORY_PATH, prefix, limit)


# ── Trace chunking ───────────────────────────────────────────────


def chunk_traces(traces_dir: Path | None = None) -> list[dict]:
    """Group related traces into chunks by project + date + keyword overlap.

    Returns list of chunks:
        {"name": "scpn-control 2026-03-17", "traces": [...], "summary": "..."}
    """
    return _chunk_trace_catalog(traces_dir or TRACES_DIR)


# ── Embedding content similarity ─────────────────────────────────

_EMBED_CACHE: dict[str, np.ndarray] = {}


def _load_embed_cache() -> None:
    global _EMBED_CACHE_LOADED
    if _EMBED_CACHE_LOADED:
        return
    # Try new npz first, then legacy pickle
    for path, loader in (
        (EMBED_CACHE_PATH, _load_npz_dict),
        (_LEGACY_EMBED_CACHE, _load_pickle_safe),
    ):
        if not path.exists():
            continue
        cached = loader(path)
        if isinstance(cached, dict):
            _EMBED_CACHE.update(cached)
            break
    _EMBED_CACHE_LOADED = True


def _persist_embed_cache() -> None:
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    persistable = {
        key: value
        for key, value in _EMBED_CACHE.items()
        if isinstance(value, np.ndarray) and (key.startswith("t:") or key.startswith("q:"))
    }
    np.savez_compressed(EMBED_CACHE_PATH, **persistable)


def _live_service_config() -> dict | None:
    return _load_live_retrieval_config(
        STATE_DIR / "current_state.json",
        base_dir=BASE_DIR,
        default_request_dir=LIVE_REQUESTS_DIR,
        default_response_dir=LIVE_RESPONSES_DIR,
        minimum_timeout_s=_LIVE_RETRIEVAL_TIMEOUT_S,
        stale_after_s=_LIVE_SERVICE_STALE_S,
    )


def _retrieve_via_live_service(
    query: str,
    top_k: int,
    include_content: bool,
) -> list[dict] | None:
    config = _live_service_config()
    if config is None:
        return None
    return _run_live_retrieval(
        config,
        query=query,
        top_k=top_k,
        include_content=include_content,
    )


def rebuild_caches(
    traces_dir: Path | None = None,
    state_path: Path | None = None,
    clear_embedding_cache: bool = False,
) -> dict:
    """Rebuild persistent retrieval caches for the active checkpoint."""
    global _EMBED_CACHE_LOADED, _QUERY_FEATURE_CACHE_LOADED
    tdir = traces_dir or TRACES_DIR
    data = _load_network(state_path)
    # Remove both new and legacy cache files
    for p in (TRACE_INDEX_CACHE_PATH, _LEGACY_TRACE_CACHE):
        if p.exists():
            p.unlink()
    _TRACE_INDEX_CACHE.clear()
    _QUERY_FEATURE_CACHE.clear()
    for p in (QUERY_FEATURE_CACHE_PATH, _LEGACY_QUERY_CACHE):
        if p.exists():
            p.unlink()
    _QUERY_FEATURE_CACHE_LOADED = False
    if clear_embedding_cache:
        for p in (EMBED_CACHE_PATH, _LEGACY_EMBED_CACHE):
            if p.exists():
                p.unlink()
        _EMBED_CACHE.clear()
        _EMBED_CACHE_LOADED = False
    started = time.perf_counter()
    trace_index = _build_trace_index(tdir, data)
    return {
        "checkpoint_path": data["_checkpoint_path"],
        "encoding_backend": data["_encoding_backend"],
        "trace_count": len(trace_index["trace_texts"]),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "trace_cache_path": str(TRACE_INDEX_CACHE_PATH),
        "query_cache_path": str(QUERY_FEATURE_CACHE_PATH),
        "embedding_cache_path": str(EMBED_CACHE_PATH),
        "cleared_embedding_cache": clear_embedding_cache,
    }


def _embedding_similarity(query: str, trace_text: str) -> float:
    """Semantic similarity between query and full trace content via embeddings.

    Uses sentence-transformers to compare the MEANING of the query
    against the full trace text. This discriminates traces that share
    keywords but differ in content — the real trace has a rich story
    that confounders can't replicate.

    Falls back to 0.0 if sentence-transformers is not available.
    """
    _load_embed_cache()

    # Cache embeddings by content hash
    q_hash = hashlib.md5(query.encode()).hexdigest()
    t_hash = hashlib.md5(trace_text[:2000].encode()).hexdigest()
    q_key = f"q:{q_hash}"
    t_key = f"t:{t_hash}"

    model = None
    if q_key not in _EMBED_CACHE or t_key not in _EMBED_CACHE:
        try:
            from encoding import _get_embed_model

            model = _get_embed_model()
        except ImportError:
            return 0.0

    if q_key not in _EMBED_CACHE:
        _EMBED_CACHE[q_key] = model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        _persist_embed_cache()

    if t_key not in _EMBED_CACHE:
        # Split trace into paragraphs, encode each, take best match.
        # Long traces contain many topics — the best paragraph match
        # discriminates better than encoding the whole document, which
        # dilutes the signal for any specific query.
        paragraphs = [p.strip() for p in trace_text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [trace_text[:2000]]
        p_embs = model.encode(
            paragraphs[:10],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        _EMBED_CACHE[t_key] = p_embs
        _persist_embed_cache()

    q_emb = _EMBED_CACHE[q_key]
    p_embs = _EMBED_CACHE[t_key]
    if p_embs.ndim == 1:
        return float(_cosine_sim(q_emb, p_embs))
    # Best-paragraph matching: max similarity across paragraphs
    sims = [float(_cosine_sim(q_emb, p)) for p in p_embs]
    return max(sims) if sims else 0.0


# ── Core retrieval ────────────────────────────────────────────────


def retrieve(
    query: str,
    top_k: int = 3,
    include_content: bool = False,
    traces_dir: Path | None = None,
    state_path: Path | None = None,
) -> list[dict]:
    """Query the SNN associative memory.

    Scoring: 0.20 × tfidf_keyword + 0.30 × snn_affinity
             + 0.25 × filename_bonus + 0.25 × embedding_similarity

    Returns list of dicts: trace, score, kw_score, snn_score.
    """
    if traces_dir is None and state_path is None:
        live_results = _retrieve_via_live_service(
            query=query,
            top_k=top_k,
            include_content=include_content,
        )
        if live_results is not None:
            _log_retrieval(query, live_results)
            return live_results

    tdir = traces_dir or TRACES_DIR
    data = _load_network(state_path)

    if not tdir.exists():
        return []

    trace_index = _build_trace_index(tdir, data)
    if not trace_index["trace_files"]:
        return []

    trace_texts = trace_index["trace_texts"]
    trace_spikes = trace_index["trace_spikes"]
    trace_names_lower = trace_index["trace_names_lower"]
    idf = trace_index["idf"]
    _get_query_features(query, data)

    scored = []
    for trace_name, text in trace_texts.items():
        kw = _tfidf_score(query, trace_name, text, idf)

        name_lower = trace_names_lower[trace_name]
        name_bonus = _filename_bonus(query, name_lower, idf)

        # Entity graph signal (Phase 1: 4-way retrieval)
        graph_score = _entity_graph_score(query, trace_name)

        # Tiered memory: recency boost (all tiers persist, hot scores higher)
        trace_path = tdir / trace_name
        if not trace_path.exists() and SEMANTIC_DIR.exists():
            # Semantic memory — check in semantic dir
            trace_path = SEMANTIC_DIR / trace_name.replace("[semantic] ", "")
        tier = _trace_tier(trace_path) if trace_path.exists() else "cold"
        tier_boost = _tier_boost(tier)
        base_score = (
            _WEIGHT_KW * kw + _WEIGHT_NAME * name_bonus + 0.10 * graph_score  # entity graph boost
        ) * tier_boost

        scored.append(
            {
                "trace": trace_name,
                "score": round(base_score, 4),
                "kw_score": round(kw, 4),
                "graph_score": round(graph_score, 4),
                "emb_score": 0.0,
                "tier": tier,
                "_base_score": base_score,
                "_tier_boost": tier_boost,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Expensive embedding similarity only reranks the strongest lexical/SNN candidates.
    rerank_k = min(len(scored), max(top_k * _EMBED_RERANK_FACTOR, _EMBED_RERANK_MIN))
    for entry in scored[:rerank_k]:
        emb_sim = _embedding_similarity(query, trace_texts[entry["trace"]])
        entry["emb_score"] = round(emb_sim, 4)
        entry["score"] = round(
            entry["_base_score"] + _WEIGHT_EMB * emb_sim * entry["_tier_boost"],
            4,
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Query expansion: if best score is weak, retry with stemmed terms
    if scored and scored[0]["score"] < 0.15:
        expanded = _expand_query(query)
        if expanded != query:
            expanded_features = _get_query_features(expanded, data)
            expanded_spikes = expanded_features["spikes"]
            for entry in scored:
                kw_exp = _tfidf_score(expanded, entry["trace"], trace_texts[entry["trace"]], idf)
                snn_exp = _cosine_sim(expanded_spikes, trace_spikes[entry["trace"]])
                name_lower = trace_names_lower[entry["trace"]]
                name_bonus = _filename_bonus(expanded, name_lower, idf)
                expanded_score = (
                    _WEIGHT_KW * kw_exp + _WEIGHT_SNN * snn_exp + _WEIGHT_NAME * name_bonus
                ) * entry["_tier_boost"]
                entry["score"] = max(entry["score"], round(expanded_score, 4))
                entry["expanded"] = True
            scored.sort(key=lambda x: x["score"], reverse=True)

    results = scored[:top_k]
    for entry in results:
        entry.pop("_base_score", None)
        entry.pop("_tier_boost", None)

    _log_retrieval(query, results)

    if include_content:
        for r in results:
            trace_path = tdir / r["trace"]
            if trace_path.exists():
                r["content"] = trace_path.read_text(encoding="utf-8")

    return results


def retrieve_context(
    query: str,
    top_k: int = 3,
    max_chars: int = 4000,
    traces_dir: Path | None = None,
    state_path: Path | None = None,
) -> str:
    """Retrieve and format traces as an LLM-injectable context block."""
    results = retrieve(
        query, top_k=top_k, include_content=True, traces_dir=traces_dir, state_path=state_path
    )
    if not results:
        return ""

    lines = ["# SNN-Retrieved Memory (query: %s)\n" % query]
    budget = max_chars - len(lines[0])

    for r in results:
        header = "## %s (relevance: %.3f)\n" % (r["trace"], r["score"])
        content = r.get("content", "(no content)")
        available = budget - len(header) - 10
        if available <= 0:
            break
        if len(content) > available:
            content = content[:available] + "\n...(truncated)"
        lines.append(header)
        lines.append(content)
        lines.append("")
        budget -= len(header) + len(content) + 1

    return "\n".join(lines)


def network_summary() -> dict:
    """Human-readable summary of SNN memory state."""
    state_path = STATE_DIR / "current_state.json"
    state = _read_json(state_path) or {}
    data = _load_network()
    w = data["w"]
    n = len(data["v"])

    w_norm = w / (w.max() + 1e-12)
    out_strength = w_norm.sum(axis=1)
    hubs = np.argsort(out_strength)[-10:][::-1]

    return {
        "active_retrieval_mode": "gpu_live_service"
        if state.get("live_retrieval_available")
        else "checkpoint",
        "retrieval_checkpoint": data.get("_checkpoint_path"),
        "retrieval_backend": data.get("_encoding_backend"),
        "retrieval_source": data.get("_retrieval_source"),
        "live_retrieval_available": bool(state.get("live_retrieval_available", False)),
        "live_retrieval_transport": state.get("live_retrieval_transport"),
        "neurons": n,
        "live_neurons": state.get("n_neurons", n),
        "simulated_time_s": state.get("t", 0),
        "cycles": state.get("cycle", 0),
        "traces_ingested": state.get("traces_processed", 0),
        "stimuli_ingested": state.get("stimuli_processed", 0),
        "arcane_neurons": state.get("arcane_neurons", 0),
        "mean_v_deep": state.get("mean_v_deep", 0),
        "weight_mean": float(w.mean()),
        "weight_std": float(w.std()),
        "weight_sparsity": float((w < 0.01).sum() / w.size),
        "hub_neurons": hubs.tolist(),
        "hub_strengths": out_strength[hubs].tolist(),
    }


# ── Evaluation ────────────────────────────────────────────────────

EVAL_GROUND_TRUTH = [
    ("Dimits shift gyrokinetic nonlinear saturation", "scpn-control_dimits-convergence"),
    ("revenue monetization personal stakes conversation", "revenue-strategy-discussion"),
    ("revenue CI anatomy ecosystem workflows", "revenue-strategy-ci-anatomy"),
    ("personal partnership hours 17000 work", "personal-moment-partnership"),
    ("neuromorphic LIF STDP sc-neurocore", "sc-neurocore_continuity-contribution"),
    ("sc-neurocore competitive sprint audit", "sc-neurocore_competitive-sprint"),
    ("quantum VQE qubit expansion single commit", "quantum_control_expansion"),
    ("director-ai audit NLI hallucination", "director-ai_audit-decisions"),
    ("scpn-fusion port architecture modules", "scpn-fusion_port-architecture"),
    ("phase-orchestrator audit elite items", "scpn-phase-orchestrator_audit"),
    ("phase-orchestrator continuity bridge", "scpn-phase-orchestrator_continuity"),
    ("quantum-control continuity identity", "scpn-quantum-control_continuity"),
    ("daemon singleton lock fixes three bugs", "director-ai_daemon-fixes"),
    ("sc-neurocore migration git directory", "sc-neurocore_migration"),
]


def run_eval(traces_dir: Path | None = None, state_path: Path | None = None) -> dict:
    """Precision@1 and MRR over the ground-truth query set."""
    correct = 0
    reciprocal_ranks = []
    details = []

    for query, expected_substr in EVAL_GROUND_TRUTH:
        results = retrieve(query, top_k=14, traces_dir=traces_dir, state_path=state_path)
        rank = None
        for i, r in enumerate(results):
            if expected_substr in r["trace"]:
                rank = i + 1
                break
        hit = rank == 1
        if hit:
            correct += 1
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)
        details.append(
            {
                "query": query,
                "expected": expected_substr,
                "got": results[0]["trace"] if results else "(none)",
                "rank": rank,
                "score": results[0]["score"] if results else 0,
                "hit": hit,
            }
        )

    total = len(EVAL_GROUND_TRUTH)
    return {
        "precision_at_1": correct / total,
        "mrr": sum(reciprocal_ranks) / total,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ── Extended evaluation ──────────────────────────────────────────


def run_extended_eval(traces_dir: Path | None = None, state_path: Path | None = None) -> dict:
    """Extended evaluation: interference, negative retrieval, encoding quality."""
    results = {"precision": run_eval(traces_dir, state_path)}

    # Negative retrieval: queries that should NOT match any trace
    negative_queries = [
        "quantum computing with trapped ions on Mars",
        "blockchain cryptocurrency trading algorithm",
        "cooking recipe for chocolate cake",
        "medieval castle architecture design",
    ]
    neg_scores = []
    for q in negative_queries:
        r = retrieve(q, top_k=1, traces_dir=traces_dir, state_path=state_path)
        neg_scores.append(r[0]["score"] if r else 0.0)
    results["negative_retrieval"] = {
        "max_score": max(neg_scores),
        "mean_score": sum(neg_scores) / len(neg_scores),
        "queries": negative_queries,
        "scores": neg_scores,
        "pass": max(neg_scores) < 0.5,
    }

    # Encoding quality: measure similarity between semantically related pairs
    try:
        from encoding import similarity, get_backend

        backend = get_backend()
        pairs = [
            ("transport coefficient", "transportation model", True),
            ("disruption prediction", "disruption mitigation", True),
            ("STDP learning", "spike timing plasticity", True),
            ("plasma physics", "chocolate recipe", False),
            ("gyrokinetic solver", "cake baking", False),
        ]
        pair_results = []
        for a, b, should_match in pairs:
            sim = similarity(a, b)
            correct = (sim > 0.1) == should_match
            pair_results.append(
                {
                    "a": a,
                    "b": b,
                    "similarity": sim,
                    "expected_match": should_match,
                    "correct": correct,
                }
            )
        results["encoding_quality"] = {
            "backend": backend,
            "accuracy": sum(p["correct"] for p in pair_results) / len(pair_results),
            "pairs": pair_results,
        }
    except ImportError:
        results["encoding_quality"] = {"error": "encoding.py not available"}

    return results


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Query the Arcane Sapience SNN associative memory",
        epilog="Examples:\n"
        "  %(prog)s 'disruption prediction'\n"
        "  %(prog)s 'gyrokinetic transport' --top 5 --content\n"
        "  %(prog)s --summary\n"
        "  %(prog)s --eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="search query text")
    parser.add_argument("--top", type=int, default=3, help="number of results")
    parser.add_argument("--content", action="store_true", help="include trace content")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--summary", action="store_true", help="show network summary")
    parser.add_argument("--eval", action="store_true", help="run precision benchmark")
    parser.add_argument("--context", action="store_true", help="LLM context block")
    parser.add_argument("--rebuild", action="store_true", help="rebuild persistent trace cache")
    parser.add_argument(
        "--clear-embed-cache",
        action="store_true",
        help="also clear cached paragraph embeddings when rebuilding",
    )
    args = parser.parse_args()

    if args.summary:
        s = network_summary()
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            print("=== SNN Memory Summary ===")
            for k, v in s.items():
                print(f"  {k}: {v}")
        return

    if args.rebuild:
        rebuilt = rebuild_caches(clear_embedding_cache=args.clear_embed_cache)
        if args.json:
            print(json.dumps(rebuilt, indent=2))
        else:
            print("=== Retrieval Cache Rebuild ===")
            for k, v in rebuilt.items():
                print(f"  {k}: {v}")
        return

    if args.eval:
        ev = run_eval()
        print(f"Precision@1: {ev['correct']}/{ev['total']} = {ev['precision_at_1'] * 100:.0f}%")
        print(f"MRR: {ev['mrr']:.3f}\n")
        for d in ev["details"]:
            marker = "OK" if d["hit"] else "MISS"
            print(
                f"  {marker:4s}  rank={str(d['rank']):>4s}  {d['query'][:45]:<45s}  -> {d['got'][:40]}"
            )
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    if args.context:
        ctx = retrieve_context(args.query, top_k=args.top)
        sys.stdout.buffer.write(ctx.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        return

    results = retrieve(args.query, top_k=args.top, include_content=args.content)

    if args.json:
        out = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "content"}
            if "content" in r:
                entry["content_length"] = len(r["content"])
                entry["content_preview"] = r["content"][:200]
            out.append(entry)
        print(json.dumps(out, indent=2))
    else:
        if not results:
            print("No traces found.")
            return
        print(f"Query: '{args.query}'\n")
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['score']:.3f}] {r['trace']}  (kw={r['kw_score']:.3f} snn={r['snn_score']:.3f})"
            )
            if "content" in r:
                preview = r["content"][:200].replace("\n", " ")
                print(f"     {preview}...")
        print()


if __name__ == "__main__":
    main()
