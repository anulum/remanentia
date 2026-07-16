# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Ordered text-to-spike encoding primitives

"""Model-free ordered encoding: splitting, projection and cue corruption.

The pinned sentence-embedding adapter lives in :mod:`snn_memory.sentence_encoder`
so that these deterministic numeric primitives stay testable without the large
local embedding checkpoint.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.state import FloatArray

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_events(text: str) -> list[str]:
    """Split text into ordered non-empty sentence or paragraph events."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def require_sentences(sentences: list[str]) -> None:
    """Reject an empty ordered-sentence list before encoding."""
    if not sentences:
        raise ValueError("at least one non-empty sentence is required")


def as_embedding_matrix(values: object, expected_rows: int) -> FloatArray:
    """Coerce and validate a model's output as a finite 2-D matrix, one row per input."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != expected_rows or not np.all(np.isfinite(array)):
        raise ValueError("encoder returned invalid embeddings")
    return array


def directory_digest(path: Path) -> str:
    """Hash a checkpoint tree with domain separation and length framing."""
    digest = hashlib.sha256()
    digest.update(b"remanentia.encoder-directory.v1\x00")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("encoder checkpoint directory is empty")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        content = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def embeddings_to_currents(
    embeddings: FloatArray,
    model: ModelConfig,
    encoder: EncoderConfig,
    *,
    input_current: float,
) -> FloatArray:
    """Project embeddings and map amplitudes to time-to-first-spike currents."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("embeddings must contain ordered rows")
    rng = np.random.default_rng(encoder.projection_seed)
    projection = rng.standard_normal((embeddings.shape[1], encoder.feature_dim))
    projection /= np.sqrt(embeddings.shape[1])
    features = embeddings @ projection
    expansion = rng.standard_normal((encoder.feature_dim, model.n_neurons))
    expansion /= np.sqrt(encoder.feature_dim)
    amplitudes = features @ expansion
    keep = max(1, int(model.n_neurons * encoder.active_fraction))
    total = embeddings.shape[0] * (encoder.packet_ms + encoder.silent_ms)
    currents = np.zeros((total, model.n_neurons), dtype=np.float64)
    for event_index, row in enumerate(amplitudes):
        active = np.argpartition(row, -keep)[-keep:]
        selected = row[active]
        span = float(np.ptp(selected))
        scaled = np.ones_like(selected) if span == 0.0 else (selected - selected.min()) / span
        latencies = np.floor((1.0 - scaled) * (encoder.packet_ms - 1)).astype(np.int64)
        start = event_index * (encoder.packet_ms + encoder.silent_ms)
        currents[start + latencies, active] = input_current
    return currents


def corrupt_currents(
    currents: FloatArray,
    fraction: float,
    seed: int,
) -> FloatArray:
    """Remove a seeded fraction of non-zero cue events without reordering."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("corruption fraction must be in [0, 1)")
    result = currents.copy()
    positions = np.argwhere(result > 0.0)
    count = int(len(positions) * fraction)
    rng = np.random.default_rng(seed)
    for row, column in positions[rng.choice(len(positions), count, replace=False)]:
        result[row, column] = 0.0
    return result
