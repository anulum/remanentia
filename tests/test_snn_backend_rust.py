# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Rust parity tests for arcane_stdp.encode_text
"""Property and parity tests for Rust encode_text in arcane_stdp.

Skipped entirely when arcane_stdp crate is not installed (CI).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

stdp = pytest.importorskip("arcane_stdp")


class TestEncodeText:
    """Test arcane_stdp.encode_text statistical properties."""

    def test_returns_correct_shape(self):
        result = np.asarray(stdp.encode_text("hello world", 1000))
        assert result.shape == (1000,)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        result = np.asarray(stdp.encode_text("BM25 retrieval scoring", 5000))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_nonempty_text_produces_nonzero(self):
        result = np.asarray(stdp.encode_text("spiking neural network", 1000))
        assert result.sum() > 0.0

    def test_empty_text_is_zero(self):
        result = np.asarray(stdp.encode_text("", 1000))
        assert result.sum() == 0.0

    def test_single_char_text_is_zero(self):
        result = np.asarray(stdp.encode_text("a", 1000))
        assert result.sum() == 0.0

    def test_deterministic(self):
        r1 = np.asarray(stdp.encode_text("test input", 500))
        r2 = np.asarray(stdp.encode_text("test input", 500))
        np.testing.assert_array_equal(r1, r2)

    def test_different_texts_differ(self):
        r1 = np.asarray(stdp.encode_text("BM25 retrieval", 1000))
        r2 = np.asarray(stdp.encode_text("STDP learning rule", 1000))
        assert not np.array_equal(r1, r2)

    def test_sparsity_reasonable(self):
        result = np.asarray(stdp.encode_text("BM25 retrieval scoring accuracy", 5000))
        active = (result > 0).sum()
        sparsity = 1.0 - active / len(result)
        assert 0.5 < sparsity < 1.0, f"Sparsity {sparsity:.2f} outside expected range"

    def test_bigrams_add_signal(self):
        unigram_text = "alpha"
        bigram_text = "alpha beta"
        r1 = np.asarray(stdp.encode_text(unigram_text, 1000))
        r2 = np.asarray(stdp.encode_text(bigram_text, 1000))
        assert r2.sum() > r1.sum()

    def test_large_neuron_count(self):
        result = np.asarray(stdp.encode_text("test input data", 20000))
        assert result.shape == (20000,)
        assert result.sum() > 0.0

    def test_unicode_input(self):
        result = np.asarray(stdp.encode_text("Šotek café naïve", 1000))
        assert result.shape == (1000,)


class TestEncodeTextParity:
    """Statistical parity between Rust (FNV) and Python (MD5) encode_text."""

    def test_same_sparsity_range(self):
        from snn_backend import encode_text as py_encode

        text = "BM25 retrieval scoring improved accuracy on LOCOMO benchmark"
        n = 5000
        py = py_encode(text, n)
        rust = np.asarray(stdp.encode_text(text, n))

        py_active = (py > 0).sum()
        rust_active = (rust > 0).sum()
        # Both should activate similar number of neurons (within 30%)
        ratio = rust_active / max(py_active, 1)
        assert 0.7 < ratio < 1.3, f"Activation ratio {ratio:.2f} outside parity range"

    def test_same_value_range(self):
        from snn_backend import encode_text as py_encode

        text = "spiking neural network STDP learning"
        n = 5000
        py = py_encode(text, n)
        rust = np.asarray(stdp.encode_text(text, n))

        assert abs(py.max() - rust.max()) < 0.3
        assert abs(py.mean() - rust.mean()) < 0.05


class TestEncodeTextPerformance:
    def test_speed(self):
        text = "BM25 retrieval scoring improved accuracy " * 20
        t0 = time.perf_counter()
        for _ in range(1000):
            stdp.encode_text(text, 5000)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 5000, f"1000 encode_text calls took {ms:.1f}ms"
