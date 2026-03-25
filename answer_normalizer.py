# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
"""Normalize LLM and ground-truth answers for comparison.

Handles hedging ("Likely yes, because..."), explanation stripping,
and answer-core extraction to improve fuzzy match accuracy.
"""
from __future__ import annotations

import re


_YES_PATTERNS = re.compile(
    r"^(yes|likely\s+yes|probably\s+yes|most\s+likely\s+yes|yeah|yep|correct|true)",
    re.IGNORECASE,
)
_NO_PATTERNS = re.compile(
    r"^(no|likely\s+no|probably\s+no|most\s+likely\s+no|nah|nope|incorrect|false|unlikely)",
    re.IGNORECASE,
)
_EXPLANATION_SPLIT = re.compile(r"[,;.!]\s+(?:because|since|as|though|but|however|due|given|considering)")
_HEDGING_PREFIX = re.compile(r"^(?:I think|I believe|I would say|It seems|Based on)\s+", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Strip hedging, explanations, extract answer core.

    >>> normalize_answer("Likely yes, because she enjoys reading")
    'likely yes'
    >>> normalize_answer("No, since she never mentioned it")
    'no'
    >>> normalize_answer("  Running, pottery  ")
    'running, pottery'
    """
    text = text.strip()
    if not text:
        return ""

    text = _HEDGING_PREFIX.sub("", text).strip()

    yes_m = _YES_PATTERNS.match(text)
    no_m = _NO_PATTERNS.match(text)

    if yes_m or no_m:
        m = yes_m or no_m
        core = m.group(0).strip().lower()
        # "likely no" stays as "likely no", not just "no"
        return core

    # Non yes/no: strip trailing explanation if present
    parts = _EXPLANATION_SPLIT.split(text, maxsplit=1)
    result = parts[0].strip()

    # Strip trailing period
    if result.endswith("."):
        result = result[:-1].strip()

    return result.lower()


def extract_answer_items(text: str) -> set[str]:
    """Extract individual items from comma/and-separated answer lists.

    >>> sorted(extract_answer_items("pottery, camping, painting, swimming"))
    ['camping', 'painting', 'pottery', 'swimming']
    >>> sorted(extract_answer_items("running and pottery"))
    ['pottery', 'running']
    """
    text = normalize_answer(text)
    # Split on comma, "and", "or"
    # Split comma first, then "and"/"or" within each part
    items = re.split(r"\s*,\s+(?:and\s+)?|\s*,\s*|\s+and\s+|\s+or\s+", text)
    return {item.strip().lower() for item in items if item.strip()}


def answers_match(predicted: str, ground_truth: str, threshold: float = 0.25) -> bool:
    """Compare predicted vs ground truth with normalization.

    Handles:
    - Yes/no core matching ("Likely yes, because X" matches "Yes")
    - List-item overlap ("pottery, running" partially matches "pottery, camping, painting, swimming")
    - Normalized string containment

    >>> answers_match("Likely yes, because she loves books", "Yes")
    True
    >>> answers_match("No", "Likely no")
    True
    >>> answers_match("pottery, running", "pottery, camping, painting, swimming")
    True
    >>> answers_match("completely unrelated answer", "The actual answer")
    False
    """
    p_norm = normalize_answer(predicted)
    g_norm = normalize_answer(ground_truth)

    if not p_norm or not g_norm:
        return False

    # Direct match
    if p_norm == g_norm:
        return True

    # Containment
    if p_norm in g_norm or g_norm in p_norm:
        return True

    # Yes/no polarity matching
    p_yes = bool(_YES_PATTERNS.match(p_norm))
    p_no = bool(_NO_PATTERNS.match(p_norm))
    g_yes = bool(_YES_PATTERNS.match(g_norm))
    g_no = bool(_NO_PATTERNS.match(g_norm))

    if (p_yes and g_yes) or (p_no and g_no):
        return True
    if (p_yes and g_no) or (p_no and g_yes):
        return False

    # List-item overlap
    p_items = extract_answer_items(predicted)
    g_items = extract_answer_items(ground_truth)
    if p_items and g_items:
        overlap = len(p_items & g_items)
        if overlap > 0 and overlap / max(len(g_items), 1) >= threshold:
            return True

    return False


_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embed_model = False
    return _embed_model if _embed_model else None


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using MiniLM embeddings.

    Returns 0.0 if embeddings unavailable.

    >>> 0.0 <= semantic_similarity("I am single", "finding acceptance") <= 1.0
    True
    """
    model = _get_embed_model()
    if not model:
        return 0.0
    import numpy as np
    embs = model.encode([text_a, text_b], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.dot(embs[0], embs[1]))
