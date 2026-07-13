# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Model-free temporal encoder tests

"""Hermetic tests for the deterministic model-free encoding primitives."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.encoder import (
    corrupt_currents,
    directory_digest,
    embeddings_to_currents,
    split_events,
)


def test_split_events_keeps_ordered_non_empty_sentences() -> None:
    events = split_events("First idea. Second idea!\n\nThird line")
    assert events == ["First idea.", "Second idea!", "Third line"]


def test_directory_digest_is_stable_and_rejects_empty_directory(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    first = directory_digest(tmp_path)
    second = directory_digest(tmp_path)
    assert first == second and len(first) == 64
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        directory_digest(empty)


def test_embeddings_to_currents_are_deterministic_ordered_packets() -> None:
    model = ModelConfig(n_neurons=24, connectivity=0.3)
    encoder = EncoderConfig(feature_dim=16, packet_ms=5, silent_ms=1)
    embeddings = np.random.default_rng(0).standard_normal((3, 12))
    first = embeddings_to_currents(embeddings, model, encoder, input_current=18.0)
    second = embeddings_to_currents(embeddings, model, encoder, input_current=18.0)
    assert first.shape == (3 * (5 + 1), 24)
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) > 0


def test_embeddings_to_currents_rejects_non_matrix_input() -> None:
    model = ModelConfig(n_neurons=24)
    encoder = EncoderConfig(feature_dim=16)
    with pytest.raises(ValueError, match="ordered rows"):
        embeddings_to_currents(np.zeros(12), model, encoder, input_current=18.0)


def test_corrupt_currents_removes_seeded_events_and_bounds_fraction() -> None:
    currents = np.zeros((6, 8), dtype=np.float64)
    for row, column in ((0, 1), (2, 3), (4, 5), (5, 7)):
        currents[row, column] = 18.0
    corrupted = corrupt_currents(currents, 0.5, 11)
    assert np.count_nonzero(corrupted) < np.count_nonzero(currents)
    assert np.all((corrupted == 0.0) | (corrupted == currents))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        corrupt_currents(currents, 1.0, 11)
