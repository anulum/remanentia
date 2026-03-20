# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — SNN Associative Memory Retrieval

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

    python 04_ARCANE_SAPIENCE/retrieve.py "disruption prediction"
    python 04_ARCANE_SAPIENCE/retrieve.py "gyrokinetic" --top 5 --content
    python 04_ARCANE_SAPIENCE/retrieve.py "sawtooth NTM" --context
    python 04_ARCANE_SAPIENCE/retrieve.py --summary
    python 04_ARCANE_SAPIENCE/retrieve.py --rebuild
    python 04_ARCANE_SAPIENCE/retrieve.py --eval   # precision benchmark

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
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ArcSap.Retrieve")

BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "snn_state"
TRACES_DIR = BASE_DIR / "reasoning_traces"
SEMANTIC_DIR = BASE_DIR / "memory" / "semantic"
RETRIEVAL_STATE_PATH = STATE_DIR / "retrieval_state.json"
DEFAULT_NETWORK_PATH = STATE_DIR / "identity_net.pkl"
EMBED_NETWORK_PATH = STATE_DIR / "identity_net_embedding_trained.pkl"
EMBED_CACHE_PATH = STATE_DIR / "embedding_cache.pkl"
FINGERPRINT_PATH = STATE_DIR / "trace_fingerprints.json"
TRACE_INDEX_CACHE_PATH = STATE_DIR / "trace_index_cache.pkl"
QUERY_FEATURE_CACHE_PATH = STATE_DIR / "query_feature_cache.pkl"
HISTORY_PATH = STATE_DIR / "retrieval_history.jsonl"
LIVE_REQUESTS_DIR = STATE_DIR / "live_retrieval_requests"
LIVE_RESPONSES_DIR = STATE_DIR / "live_retrieval_responses"

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

_HASH_PRIMES = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]
_NETWORK_CACHE: dict[tuple[str, int, int, str], dict] = {}
_TRACE_INDEX_CACHE: dict[str, dict] = {}
_QUERY_FEATURE_CACHE: dict[tuple[str, str], dict] = {}
_EMBED_CACHE_LOADED = False
_QUERY_FEATURE_CACHE_LOADED = False

# Common words that carry no discriminating signal
_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it by as with from was were "
    "be been have has had this that these those are not no its the into can "
    "will would should could may also so if when then than more most all any "
    "each every both few many much some such only just about over after before "
    "between through during up down out off did do does how what which who whom "
    "where why here there their them they we our us you your he she his her "
    "i me my we us being now very".split()
)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping stopwords."""
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


_STEM_SUFFIXES = [
    "ation", "tion", "sion", "ment", "ness", "ity", "ous",
    "ive", "ing", "ical", "ally", "able", "ible", "ful",
    "less", "ized", "ise", "ize", "ed", "ly", "er", "est",
    "al", "es", "s",
]


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer."""
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _expand_query(query: str) -> str:
    """Expand query with stems for broader matching.

    "gyrokinetic transport saturation" →
    "gyrokinetic transport saturation gyrokinet transport saturat"
    """
    tokens = _tokenize(query)
    stems = {_stem(t) for t in tokens}
    extra = stems - set(tokens)
    if extra:
        return query + " " + " ".join(sorted(extra))
    return query


def _bigrams(tokens: list[str]) -> list[str]:
    """Generate bigrams from token list."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def _encode(text: str, n_neurons: int) -> np.ndarray:
    """Encode text to neuron activation pattern.

    Uses the active backend from encoding.py. Retrieval activates the backend
    declared by the selected checkpoint before any text is encoded.
    """
    try:
        from encoding import encode_text
        return encode_text(text, n_neurons)
    except ImportError:
        pass
    # Fallback to inline hash encoding if encoding.py not available
    import hashlib

    pattern = np.zeros(n_neurons)
    tokens = _tokenize(text)

    for word in tokens:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.15, 1.0)

    for bg in _bigrams(tokens):
        h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES[:5]:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.25, 1.0)

    return pattern


# ── TF-IDF machinery ──────────────────────────────────────────────


def _build_idf(trace_texts: dict[str, str]) -> dict[str, float]:
    """Compute inverse document frequency over unigrams + bigrams.

    IDF(term) = log(N / (1 + df(term)))
    Terms appearing in every document get IDF ≈ 0 (no discrimination).
    Terms unique to one document get IDF ≈ log(N) (high discrimination).
    """
    n_docs = len(trace_texts)
    df: Counter[str] = Counter()
    for name, text in trace_texts.items():
        tokens = _tokenize(text + " " + name.replace("-", " ").replace("_", " "))
        terms = set(tokens) | set(_bigrams(tokens))
        for t in terms:
            df[t] += 1
    return {t: math.log(1 + n_docs / (1 + count)) for t, count in df.items()}


def _tfidf_score(query: str, doc_name: str, doc_text: str, idf: dict[str, float]) -> float:
    """TF-IDF with sublinear TF, bigrams, and filename boosting.

    score = Σ_{t ∈ query_terms ∩ doc_terms} (1 + log(tf)) × idf(t)

    Filename terms get 3x boost (the filename is the most condensed
    description of a trace's topic).
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    q_terms = set(q_tokens) | set(_bigrams(q_tokens))

    # Document terms: body + filename (filename boosted)
    name_tokens = _tokenize(doc_name.replace("-", " ").replace("_", " "))
    doc_tokens = _tokenize(doc_text)
    all_tokens = doc_tokens + name_tokens * 3  # filename 3x weight
    doc_tf: Counter[str] = Counter(all_tokens)
    # Add bigrams
    for bg in _bigrams(doc_tokens):
        doc_tf[bg] += 1
    for bg in _bigrams(name_tokens):
        doc_tf[bg] += 3  # filename bigrams boosted too

    score = 0.0
    for t in q_terms:
        if t in doc_tf:
            tf = 1.0 + math.log(doc_tf[t])  # sublinear TF
            score += tf * idf.get(t, 0.0)

    # Normalize by query length to make scores comparable
    return score / len(q_terms)


# ── Network I/O ───────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_backend(backend: str | None) -> str:
    if backend in {"hash", "lsh", "embedding"}:
        return backend
    return "lsh"


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


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
    if state_path is not None:
        path = _resolve_path(str(state_path))
        if not path.exists():
            raise FileNotFoundError(f"No SNN state at {path}.")
        return {
            "checkpoint_path": path,
            "encoding_backend": _infer_backend_from_path(path),
            "source": "explicit",
        }

    manifest = _read_json(RETRIEVAL_STATE_PATH)
    if manifest:
        raw_path = manifest.get("checkpoint_path")
        if raw_path:
            path = _resolve_path(str(raw_path))
            if path.exists():
                return {
                    "checkpoint_path": path,
                    "encoding_backend": _normalize_backend(manifest.get("encoding_backend")),
                    "source": manifest.get("source", "retrieval_state"),
                }

    current_state = _read_json(STATE_DIR / "current_state.json")
    if current_state:
        raw_path = current_state.get("retrieval_checkpoint_path")
        if raw_path:
            path = _resolve_path(str(raw_path))
            if path.exists():
                return {
                    "checkpoint_path": path,
                    "encoding_backend": _normalize_backend(current_state.get("retrieval_backend")),
                    "source": "current_state",
                }

    if EMBED_NETWORK_PATH.exists():
        return {
            "checkpoint_path": EMBED_NETWORK_PATH.resolve(),
            "encoding_backend": "embedding",
            "source": "embedding_checkpoint_fallback",
        }

    if DEFAULT_NETWORK_PATH.exists():
        current_state = current_state or {}
        return {
            "checkpoint_path": DEFAULT_NETWORK_PATH.resolve(),
            "encoding_backend": _normalize_backend(
                current_state.get("retrieval_backend")
                or current_state.get("encoding_backend")
                or _infer_backend_from_path(DEFAULT_NETWORK_PATH)
            ),
            "source": "legacy_checkpoint_fallback",
        }

    raise FileNotFoundError(
        "No compatible SNN checkpoint found. "
        "Expected retrieval_state.json, identity_net_embedding_trained.pkl, "
        "or identity_net.pkl."
    )


def _load_network(state_path: Path | None = None) -> dict:
    config = _resolve_network_config(state_path)
    path = config["checkpoint_path"]
    stat = path.stat()
    cache_key = (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        config["encoding_backend"],
    )
    cached = _NETWORK_CACHE.get(cache_key)
    if cached is None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected network payload in {path}")
        signature = hashlib.md5(
            f"{path}:{stat.st_mtime_ns}:{stat.st_size}:{config['encoding_backend']}".encode(
                "utf-8"
            )
        ).hexdigest()
        cached = dict(data)
        cached["_checkpoint_path"] = str(path)
        cached["_encoding_backend"] = config["encoding_backend"]
        cached["_state_signature"] = signature
        cached["_retrieval_source"] = config["source"]
        _NETWORK_CACHE.clear()
        _NETWORK_CACHE[cache_key] = cached

    _activate_backend(config["encoding_backend"])
    return cached


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _spike_feature(w: np.ndarray, stim: np.ndarray, steps: int = 50) -> np.ndarray:
    """Deterministic spike-count feature for a stimulus under fixed weights."""
    n = len(stim)
    dt_ms = 1.0
    v_rest, v_thresh, v_reset, tau_m = -65.0, -55.0, -70.0, 10.0

    v = np.random.default_rng(0).uniform(-70.0, -55.0, n)
    i_ext = 0.3 + stim * 2.0
    spike_count = np.zeros(n, dtype=np.float32)

    for _ in range(steps):
        fired = (v >= v_thresh).astype(np.float32)
        i_syn = w.dot(fired) if hasattr(w, "dot") else (w @ fired)
        dv = (-(v - v_rest) / tau_m + i_ext + i_syn * 0.5) * dt_ms
        v += dv
        spiked = v >= v_thresh
        spike_count += spiked.astype(np.float32)
        v[spiked] = v_reset

    return spike_count


def _snn_affinity(w: np.ndarray, query_stim: np.ndarray, trace_stim: np.ndarray) -> float:
    """Compare deterministic query/trace spike-count features."""
    return _cosine_sim(_spike_feature(w, query_stim), _spike_feature(w, trace_stim))


def _trace_fingerprint(trace_files: list[Path]) -> str:
    digest = hashlib.md5()
    for tf in trace_files:
        stat = tf.stat()
        digest.update(tf.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _load_trace_index_cache(cache_key: str) -> dict | None:
    if not TRACE_INDEX_CACHE_PATH.exists():
        return None
    try:
        with open(TRACE_INDEX_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
    except Exception:
        return None
    if not isinstance(cached, dict) or cached.get("cache_key") != cache_key:
        return None
    trace_spikes = cached.get("trace_spikes")
    trace_names_lower = cached.get("trace_names_lower")
    idf = cached.get("idf")
    if not isinstance(trace_spikes, dict) or not isinstance(trace_names_lower, dict) or not isinstance(idf, dict):
        return None
    return {
        "trace_spikes": trace_spikes,
        "trace_names_lower": trace_names_lower,
        "idf": idf,
    }


def _persist_trace_index_cache(cache_key: str, payload: dict) -> None:
    TRACE_INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    to_store = {
        "cache_key": cache_key,
        "updated_at": time.time(),
        "trace_count": len(payload["trace_spikes"]),
        "trace_spikes": payload["trace_spikes"],
        "trace_names_lower": payload["trace_names_lower"],
        "idf": payload["idf"],
    }
    with open(TRACE_INDEX_CACHE_PATH, "wb") as f:
        pickle.dump(to_store, f)


def _build_trace_index(tdir: Path, data: dict) -> dict:
    trace_files = sorted(tdir.glob("*.md"))
    # Include semantic memories in search corpus
    semantic_files = []
    if SEMANTIC_DIR.exists():
        semantic_files = sorted(SEMANTIC_DIR.rglob("*.md"))
    all_files = trace_files + semantic_files
    cache_key = (
        f"{data['_state_signature']}:"
        f"{data['_encoding_backend']}:"
        f"{_trace_fingerprint(all_files)}"
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
    if QUERY_FEATURE_CACHE_PATH.exists():
        try:
            with open(QUERY_FEATURE_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                _QUERY_FEATURE_CACHE.update(cached)
        except Exception:
            pass
    _QUERY_FEATURE_CACHE_LOADED = True


def _persist_query_feature_cache() -> None:
    QUERY_FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(list(_QUERY_FEATURE_CACHE.items())[-256:])
    with open(QUERY_FEATURE_CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)


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
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "query": query,
        "timestamp": time.time(),
        "results": [{"trace": r["trace"], "score": r["score"]} for r in results[:5]],
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def retrieval_history(limit: int = 50) -> list[dict]:
    """Read recent retrieval history."""
    if not HISTORY_PATH.exists():
        return []
    rows = []
    for line in HISTORY_PATH.read_text().strip().split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-limit:]


def trace_summaries(traces_dir: Path | None = None) -> list[dict]:
    """Extract 1-line summaries from each trace file.

    Takes the first non-heading, non-empty line after the title.
    """
    tdir = traces_dir or TRACES_DIR
    if not tdir.exists():
        return []

    summaries = []
    for f in sorted(tdir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        text = f.read_text(encoding="utf-8")
        summary = ""
        past_heading = False
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                past_heading = True
                continue
            if not past_heading or not stripped or stripped.startswith("---"):
                continue
            # Skip bold key-value frontmatter: **Key:** value
            if stripped.startswith("**") and ":**" in stripped:
                continue
            # Skip bare metadata lines
            if stripped.startswith("- **") and ":**" in stripped:
                continue
            summary = stripped.lstrip("- ").strip()
            break
        summaries.append({"name": f.name, "summary": summary or "(no summary)"})
    return summaries


# ── Tiered memory (hot/warm/cold) ─────────────────────────────────

_TIER_HOT_HOURS = 24
_TIER_WARM_DAYS = 7
_TIER_WEIGHTS = {"hot": 1.02, "warm": 1.0, "cold": 0.98}


def _trace_tier(path: Path) -> str:
    """Classify trace into hot/warm/cold based on modification time."""
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours <= _TIER_HOT_HOURS:
        return "hot"
    if age_hours <= _TIER_WARM_DAYS * 24:
        return "warm"
    return "cold"


def _tier_boost(tier: str) -> float:
    """Recency boost factor for tiered retrieval. All tiers persist forever."""
    return _TIER_WEIGHTS.get(tier, 1.0)


# ── Entity graph signal ──────────────────────────────────────────

_GRAPH_ENTITIES: dict | None = None
_GRAPH_RELATIONS: list | None = None


def _load_graph_once():
    """Lazy-load entity graph for retrieval boosting."""
    global _GRAPH_ENTITIES, _GRAPH_RELATIONS
    if _GRAPH_ENTITIES is not None:
        return
    graph_dir = BASE_DIR / "memory" / "graph"
    _GRAPH_ENTITIES = {}
    _GRAPH_RELATIONS = []
    ent_path = graph_dir / "entities.jsonl"
    rel_path = graph_dir / "relations.jsonl"
    if ent_path.exists():
        for line in ent_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                e = json.loads(line)
                _GRAPH_ENTITIES[e["id"]] = e
    if rel_path.exists():
        for line in rel_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                _GRAPH_RELATIONS.append(json.loads(line))


def _entity_graph_score(query: str, trace_name: str) -> float:
    """Score based on shared entity connections between query and trace.

    Finds entities mentioned in the query, then checks how many of those
    entities have graph connections to entities in the trace filename.
    """
    _load_graph_once()
    if not _GRAPH_ENTITIES or not _GRAPH_RELATIONS:
        return 0.0

    q_lower = query.lower()
    q_entities = [eid for eid in _GRAPH_ENTITIES if eid in q_lower]
    if not q_entities:
        return 0.0

    t_lower = trace_name.lower().replace("-", " ").replace("_", " ")
    t_entities = [eid for eid in _GRAPH_ENTITIES if eid in t_lower]
    if not t_entities:
        return 0.0

    # Count weighted connections between query entities and trace entities
    score = 0.0
    for r in _GRAPH_RELATIONS:
        src, tgt = r.get("source", ""), r.get("target", "")
        w = r.get("weight", 1)
        if (src in q_entities and tgt in t_entities) or \
           (tgt in q_entities and src in t_entities):
            score += w
        elif src in q_entities and src in t_entities:
            score += w * 0.5
        elif tgt in q_entities and tgt in t_entities:
            score += w * 0.5

    # Normalize by max possible
    max_w = max((r.get("weight", 1) for r in _GRAPH_RELATIONS), default=1)
    return min(score / max(max_w * len(q_entities), 1), 1.0)


def _filename_bonus(query: str, name_lower: str, idf: dict[str, float]) -> float:
    """IDF-weighted filename overlap bonus.

    The filename is a dense topic label for a trace. Weighting overlap by IDF
    rewards rare discriminative matches such as `daemon` or `dimits` more than
    generic terms, and avoids overvaluing long queries with many unmatched words.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    total = sum(idf.get(token, 1.0) for token in q_tokens)
    if total <= 1e-12:
        return 0.0
    matched = sum(idf.get(token, 1.0) for token in q_tokens if token in name_lower)
    return matched / total


# ── Cross-trace linking ───────────────────────────────────────────


def related_traces(
    trace_name: str, top_k: int = 3, traces_dir: Path | None = None
) -> list[dict]:
    """Find traces most similar to the given trace by keyword overlap."""
    tdir = traces_dir or TRACES_DIR
    target = tdir / trace_name
    if not target.exists():
        return []

    target_text = target.read_text(encoding="utf-8")
    target_tokens = set(_tokenize(target_text + " " + trace_name.replace("-", " ").replace("_", " ")))
    if not target_tokens:
        return []

    scored = []
    for f in tdir.glob("*.md"):
        if f.name == trace_name:
            continue
        text = f.read_text(encoding="utf-8")
        tokens = set(_tokenize(text + " " + f.name.replace("-", " ").replace("_", " ")))
        overlap = target_tokens & tokens
        if not overlap:
            continue
        jaccard = len(overlap) / len(target_tokens | tokens)
        scored.append({"trace": f.name, "similarity": round(jaccard, 4), "shared_terms": len(overlap)})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def query_suggestions(prefix: str, limit: int = 8) -> list[str]:
    """Autocomplete suggestions from retrieval history."""
    if not HISTORY_PATH.exists() or len(prefix) < 2:
        return []

    prefix_lower = prefix.lower()
    seen = set()
    suggestions = []
    for line in reversed(HISTORY_PATH.read_text().strip().split("\n")):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = entry.get("query", "")
        if q.lower().startswith(prefix_lower) and q not in seen:
            seen.add(q)
            suggestions.append(q)
            if len(suggestions) >= limit:
                break
    return suggestions


# ── Trace chunking ───────────────────────────────────────────────


def chunk_traces(traces_dir: Path | None = None) -> list[dict]:
    """Group related traces into chunks by project + date + keyword overlap.

    Returns list of chunks:
        {"name": "scpn-control 2026-03-17", "traces": [...], "summary": "..."}
    """
    tdir = traces_dir or TRACES_DIR
    if not tdir.exists():
        return []

    # Group by project prefix and date
    groups: dict[str, list[Path]] = {}
    for f in sorted(tdir.glob("*.md")):
        name = f.stem
        # Extract date and project from filename patterns
        # e.g., "2026-03-17T0519_scpn-control_dimits-convergence"
        parts = name.split("_", 2)
        date = parts[0].split("T")[0] if parts else "unknown"
        project = parts[1] if len(parts) > 1 else "general"
        key = f"{project} {date}"
        groups.setdefault(key, []).append(f)

    chunks = []
    for key, files in groups.items():
        traces = [f.name for f in files]
        # Auto-summary: project + date + count
        parts = key.split(" ", 1)
        project = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        summary = f"{project}: {len(files)} trace{'s' if len(files) > 1 else ''} ({date})"
        chunks.append({
            "name": key,
            "project": project,
            "date": date,
            "traces": traces,
            "count": len(files),
            "summary": summary,
        })

    chunks.sort(key=lambda c: c["date"], reverse=True)
    return chunks


# ── Embedding content similarity ─────────────────────────────────

_EMBED_CACHE: dict[str, np.ndarray] = {}


def _load_embed_cache() -> None:
    global _EMBED_CACHE_LOADED
    if _EMBED_CACHE_LOADED:
        return
    if EMBED_CACHE_PATH.exists():
        try:
            with open(EMBED_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                _EMBED_CACHE.update(cached)
        except Exception:
            pass
    _EMBED_CACHE_LOADED = True


def _persist_embed_cache() -> None:
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    persistable = {
        key: value
        for key, value in _EMBED_CACHE.items()
        if isinstance(value, np.ndarray) and (key.startswith("t:") or key.startswith("q:"))
    }
    with open(EMBED_CACHE_PATH, "wb") as f:
        pickle.dump(persistable, f)


def _live_service_config() -> dict | None:
    state = _read_json(STATE_DIR / "current_state.json")
    if not state or not state.get("live_retrieval_available"):
        return None
    timestamp = float(state.get("timestamp", 0.0) or 0.0)
    if timestamp and time.time() - timestamp > _LIVE_SERVICE_STALE_S:
        return None
    request_dir = _resolve_path(
        str(state.get("live_retrieval_request_dir") or LIVE_REQUESTS_DIR)
    )
    response_dir = _resolve_path(
        str(state.get("live_retrieval_response_dir") or LIVE_RESPONSES_DIR)
    )
    if not request_dir.exists() or not response_dir.exists():
        return None
    advertised_timeout = float(
        state.get("live_retrieval_timeout_s", _LIVE_RETRIEVAL_TIMEOUT_S)
    )
    return {
        "request_dir": request_dir,
        "response_dir": response_dir,
        "timeout_s": max(advertised_timeout, _LIVE_RETRIEVAL_TIMEOUT_S),
        "transport": state.get("live_retrieval_transport", "filesystem"),
        "cycle": state.get("cycle"),
        "n_neurons": state.get("n_neurons"),
    }


def _retrieve_via_live_service(
    query: str,
    top_k: int,
    include_content: bool,
) -> list[dict] | None:
    config = _live_service_config()
    if config is None:
        return None

    request_id = hashlib.md5(
        f"{query}:{top_k}:{include_content}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    request_path = config["request_dir"] / f"{request_id}.json"
    response_path = config["response_dir"] / f"{request_id}.json"
    payload = {
        "id": request_id,
        "query": query,
        "top_k": int(top_k),
        "include_content": bool(include_content),
        "created_at": time.time(),
    }
    try:
        _write_json_atomic(request_path, payload)
    except OSError:
        return None

    deadline = time.time() + max(config["timeout_s"], 1.0)
    while time.time() < deadline:
        if response_path.exists():
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            for path in (response_path, request_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if response.get("status") == "ok" and isinstance(response.get("results"), list):
                return response["results"]
            return None
        time.sleep(0.05)

    for path in (request_path, response_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def rebuild_caches(
    traces_dir: Path | None = None,
    state_path: Path | None = None,
    clear_embedding_cache: bool = False,
) -> dict:
    """Rebuild persistent retrieval caches for the active checkpoint."""
    global _EMBED_CACHE_LOADED, _QUERY_FEATURE_CACHE_LOADED
    tdir = traces_dir or TRACES_DIR
    data = _load_network(state_path)
    if TRACE_INDEX_CACHE_PATH.exists():
        TRACE_INDEX_CACHE_PATH.unlink()
    _TRACE_INDEX_CACHE.clear()
    _QUERY_FEATURE_CACHE.clear()
    if QUERY_FEATURE_CACHE_PATH.exists():
        QUERY_FEATURE_CACHE_PATH.unlink()
    _QUERY_FEATURE_CACHE_LOADED = False
    if clear_embedding_cache and EMBED_CACHE_PATH.exists():
        EMBED_CACHE_PATH.unlink()
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
    query_features = _get_query_features(query, data)
    query_spikes = query_features["spikes"]

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
            _WEIGHT_KW * kw
            + _WEIGHT_NAME * name_bonus
            + 0.10 * graph_score  # entity graph boost
        ) * tier_boost

        scored.append({
            "trace": trace_name,
            "score": round(base_score, 4),
            "kw_score": round(kw, 4),
            "graph_score": round(graph_score, 4),
            "emb_score": 0.0,
            "tier": tier,
            "_base_score": base_score,
            "_tier_boost": tier_boost,
        })

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
                    _WEIGHT_KW * kw_exp
                    + _WEIGHT_SNN * snn_exp
                    + _WEIGHT_NAME * name_bonus
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
    results = retrieve(query, top_k=top_k, include_content=True,
                       traces_dir=traces_dir, state_path=state_path)
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
        "active_retrieval_mode": "gpu_live_service" if state.get("live_retrieval_available") else "checkpoint",
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
        details.append({
            "query": query,
            "expected": expected_substr,
            "got": results[0]["trace"] if results else "(none)",
            "rank": rank,
            "score": results[0]["score"] if results else 0,
            "hit": hit,
        })

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
            pair_results.append({"a": a, "b": b, "similarity": sim, "expected_match": should_match, "correct": correct})
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
        print(f"Precision@1: {ev['correct']}/{ev['total']} = {ev['precision_at_1']*100:.0f}%")
        print(f"MRR: {ev['mrr']:.3f}\n")
        for d in ev["details"]:
            marker = "OK" if d["hit"] else "MISS"
            print(f"  {marker:4s}  rank={str(d['rank']):>4s}  {d['query'][:45]:<45s}  -> {d['got'][:40]}")
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
            print(f"  {i}. [{r['score']:.3f}] {r['trace']}  (kw={r['kw_score']:.3f} snn={r['snn_score']:.3f})")
            if "content" in r:
                preview = r["content"][:200].replace("\n", " ")
                print(f"     {preview}...")
        print()


if __name__ == "__main__":
    main()
