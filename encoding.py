# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Text encoding for SNN stimulus patterns.

Three encoding backends, from worst to best:

1. **hash** (current default) — MD5 hash per word. No semantic similarity.
   "transport" and "convection" activate unrelated neurons.

2. **lsh** — Locality-sensitive hashing via character n-grams + random
   projections. Similar words activate overlapping neuron sets.
   "transport" and "transportation" share ~60% of activated neurons.
   Pure numpy, no external dependencies.

3. **embedding** (future) — Sentence-transformer dense vectors projected
   into neuron space. Full semantic similarity. Requires PyTorch.

Usage::

    from encoding import encode_text, set_backend

    # Use LSH (recommended)
    set_backend("lsh")
    pattern = encode_text("gyrokinetic transport saturation", n_neurons=2000)

    # Characters that share n-grams will activate overlapping neurons
    p1 = encode_text("transport coefficient", 2000)
    p2 = encode_text("transportation model", 2000)
    # cosine(p1, p2) ≈ 0.4-0.6 (vs ≈ 0.0 with hash encoding)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

_HASH_PRIMES = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]

_BACKEND = "lsh"  # default to LSH — strictly better than hash

# LSH projection matrix (lazily initialized per n_neurons)
_LSH_CACHE: dict[int, np.ndarray] = {}
_LSH_SEED = 42
_NGRAM_SIZE = 3
_LSH_DIM = 256  # character n-gram vocabulary mapped to this many dimensions


def set_backend(backend: str) -> None:
    """Set encoding backend: "hash", "lsh", or "embedding"."""
    global _BACKEND
    if backend not in ("hash", "lsh", "embedding"):
        raise ValueError(f"Unknown backend: {backend}")
    _BACKEND = backend


def get_backend() -> str:
    return _BACKEND


def _char_ngrams(word: str, n: int = _NGRAM_SIZE) -> list[str]:
    """Extract character n-grams from a word.

    "transport" with n=3 -> ["tra", "ran", "ans", "nsp", "spo", "por", "ort"]

    Words sharing a root produce overlapping n-gram sets:
    "transport"      -> {tra, ran, ans, nsp, spo, por, ort}
    "transportation" -> {tra, ran, ans, nsp, spo, por, ort, rta, tat, ati, tio, ion}
    Overlap: 7/12 = 58%
    """
    padded = f"<{word}>"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _ngram_vector(text: str) -> np.ndarray:
    """Convert text to a sparse vector of character n-gram counts.

    Each unique n-gram hashes to a position in a fixed-size vector.
    Similar texts produce similar vectors because they share n-grams.
    """
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    vec = np.zeros(_LSH_DIM, dtype=np.float32)
    for word in tokens:
        if len(word) <= 1:
            continue
        for ng in _char_ngrams(word):
            # Stable hash to position in vector
            idx = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16) % _LSH_DIM
            vec[idx] += 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _get_lsh_projection(n_neurons: int) -> np.ndarray:
    """Get or create the random projection matrix for LSH.

    Projects from _LSH_DIM-dimensional n-gram space to n_neurons-dimensional
    neuron space. The projection preserves relative distances (Johnson-Lindenstrauss).
    The matrix is deterministic (seeded) so the same text always activates
    the same neurons.
    """
    if n_neurons not in _LSH_CACHE:
        rng = np.random.default_rng(_LSH_SEED)
        # Gaussian random projection (JL lemma)
        proj = rng.standard_normal((n_neurons, _LSH_DIM)).astype(np.float32)
        proj /= np.sqrt(_LSH_DIM)
        _LSH_CACHE[n_neurons] = proj
    return _LSH_CACHE[n_neurons]


def encode_lsh(text: str, n_neurons: int) -> np.ndarray:
    """LSH encoding: character n-grams → random projection → neuron activations.

    Similar texts produce overlapping activation patterns because:
    1. Similar words share character n-grams
    2. Shared n-grams contribute to the same dimensions
    3. Random projection preserves these relationships (JL lemma)
    """
    ngram_vec = _ngram_vector(text)
    proj = _get_lsh_projection(n_neurons)
    # Project and apply ReLU (only positive activations)
    raw = proj @ ngram_vec
    pattern = np.clip(raw, 0, None)
    # Normalize to [0, 1] range
    maxval = pattern.max()
    if maxval > 0:
        pattern /= maxval
    # Sparsify: keep only top 15% of activations
    threshold = np.percentile(pattern[pattern > 0], 85) if (pattern > 0).sum() > 10 else 0
    pattern[pattern < threshold] = 0
    return pattern


def encode_hash(text: str, n_neurons: int) -> np.ndarray:
    """Original MD5 hash encoding (kept for backward compatibility)."""
    pattern = np.zeros(n_neurons)
    tokens = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 1]

    for word in tokens:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.15, 1.0)

    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]}_{tokens[i + 1]}"
        h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES[:5]:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.25, 1.0)

    return pattern


_EMBED_MODEL = None
_EMBED_PROJ: dict[int, np.ndarray] = {}


def _get_embed_model():
    """Lazy-load sentence-transformers model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                _EMBED_MODEL = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu",
                    local_files_only=True,
                )
            except TypeError:
                try:
                    _EMBED_MODEL = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        local_files_only=True,
                    )
                except TypeError:
                    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                try:
                    _EMBED_MODEL = _EMBED_MODEL.to("cpu")
                except Exception:
                    pass
            except Exception as exc:
                raise ImportError(
                    "Embedding backend requires a locally cached "
                    "all-MiniLM-L6-v2 model."
                ) from exc
        except ImportError:
            raise ImportError(
                "Embedding backend requires sentence-transformers. "
                "Install: pip install sentence-transformers"
            )
    return _EMBED_MODEL


def _get_embed_projection(embed_dim: int, n_neurons: int) -> np.ndarray:
    """Get or create projection from embedding space to neuron space."""
    key = (embed_dim, n_neurons)
    if key not in _EMBED_PROJ:
        rng = np.random.default_rng(_LSH_SEED)
        proj = rng.standard_normal((n_neurons, embed_dim)).astype(np.float32)
        proj /= np.sqrt(embed_dim)
        _EMBED_PROJ[key] = proj
    return _EMBED_PROJ[key]


def encode_embedding(text: str, n_neurons: int) -> np.ndarray:
    """Embedding encoding: sentence-transformers → random projection → neuron space.

    Uses all-MiniLM-L6-v2 (384-dim, fast, good quality). The dense
    embedding captures full semantic meaning — "transport" and
    "convection" are close in embedding space even though they share
    no characters.

    The projection to neuron space preserves relative distances
    (Johnson-Lindenstrauss). ReLU + sparsification keeps activations
    sparse for efficient SNN processing.
    """
    model = _get_embed_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    proj = _get_embed_projection(len(embedding), n_neurons)
    raw = proj @ embedding
    pattern = np.clip(raw, 0, None)
    maxval = pattern.max()
    if maxval > 0:
        pattern /= maxval
    threshold = np.percentile(pattern[pattern > 0], 85) if (pattern > 0).sum() > 10 else 0
    pattern[pattern < threshold] = 0
    return pattern


def encode_text(text: str, n_neurons: int) -> np.ndarray:
    """Encode text to neuron activation pattern using the active backend."""
    if _BACKEND == "lsh":
        return encode_lsh(text, n_neurons)
    elif _BACKEND == "hash":
        return encode_hash(text, n_neurons)
    elif _BACKEND == "embedding":
        return encode_embedding(text, n_neurons)
    raise ValueError(f"Unknown backend: {_BACKEND}")


def similarity(text_a: str, text_b: str, n_neurons: int = 2000) -> float:
    """Cosine similarity between two texts in neuron activation space."""
    a = encode_text(text_a, n_neurons)
    b = encode_text(text_b, n_neurons)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == "__main__":
    # Demonstrate LSH vs hash encoding
    pairs = [
        ("transport coefficient", "transportation model"),
        ("gyrokinetic saturation", "gyrokinetic transport"),
        ("disruption prediction", "disruption mitigation"),
        ("weight saturation", "synaptic scaling"),
        ("STDP learning rule", "spike timing plasticity"),
        ("python pytest", "rust cargo test"),
    ]

    print("Encoding comparison — semantic similarity:\n")
    has_embed = False
    try:
        from sentence_transformers import SentenceTransformer
        has_embed = True
    except ImportError:
        pass

    for a, b in pairs:
        set_backend("hash")
        sim_hash = similarity(a, b)
        set_backend("lsh")
        sim_lsh = similarity(a, b)
        line = f"  '{a}' vs '{b}'\n    hash={sim_hash:.3f}  lsh={sim_lsh:.3f}"
        if has_embed:
            set_backend("embedding")
            sim_embed = similarity(a, b)
            line += f"  embed={sim_embed:.3f}"
        print(line)
        print()

    set_backend("lsh")
