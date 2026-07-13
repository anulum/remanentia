# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pinned local sentence-embedding adapter

"""Pinned local sentence-transformer adapter for the temporal encoder.

This module is the only encoder surface that depends on the large local
embedding checkpoint (``.snn_models/`` is git-ignored). Its behaviour is
exercised end to end by the local ``tests/test_snn_memory_sentence_encoder.py``,
``tests/test_snn_memory_cli_e2e.py`` and ``tests/test_snn_memory_benchmark_e2e.py``
integration tests; the deterministic encoding maths it composes live in the
model-free :mod:`snn_memory.encoder` module and are unit-tested there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.encoder import directory_digest, embeddings_to_currents, split_events
from snn_memory.state import FloatArray


@dataclass(frozen=True)
class EncodedTrace:
    """Ordered current packets and their source event boundaries."""

    currents: FloatArray
    event_count: int
    checkpoint_digest: str


class LocalSentenceEncoder:
    """Sentence-transformer adapter pinned to a local checkpoint directory."""

    def __init__(self, checkpoint: Path | str) -> None:
        path = Path(checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"local encoder checkpoint not found: {path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError("sentence-transformers is required for temporal encoding") from exc
        self._path = path.resolve()
        self._model = SentenceTransformer(str(self._path), local_files_only=True, device="cpu")
        self.digest = directory_digest(self._path)

    def encode(self, sentences: list[str]) -> FloatArray:
        """Return one finite float64 embedding row per ordered sentence."""
        if not sentences:
            raise ValueError("at least one non-empty sentence is required")
        values = np.asarray(
            self._model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float64,
        )
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("encoder returned invalid embeddings")
        return values


def encode_trace(
    text: str,
    sentence_encoder: LocalSentenceEncoder,
    model: ModelConfig,
    encoder: EncoderConfig,
    *,
    input_current: float,
) -> EncodedTrace:
    """Encode real ordered text through a pinned local model into spike currents."""
    events = split_events(text)
    embeddings = sentence_encoder.encode(events)
    currents = embeddings_to_currents(embeddings, model, encoder, input_current=input_current)
    return EncodedTrace(currents, len(events), sentence_encoder.digest)
