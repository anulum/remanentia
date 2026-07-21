# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface persistent query-feature tests

"""Exercise query encoding, LIF computation, and gzip persistence end to end."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from retrieval_cache_io import load_query_feature_cache
from retrieval_query_features import (
    QueryFeatureStore,
    RetrievalNetworkState,
)


@pytest.fixture
def hash_backend() -> Iterator[None]:
    """Use the real deterministic hash encoder and restore backend state."""
    import encoding

    original = encoding.get_backend()
    encoding.set_backend("hash")
    try:
        yield
    finally:
        encoding.set_backend(original)


def _network(signature: str = "checkpoint-a") -> RetrievalNetworkState:
    return {
        "v": np.zeros(32, dtype=np.float64),
        "w": np.eye(32, dtype=np.float64) * 0.03,
        "_state_signature": signature,
    }


def test_query_features_run_real_encoding_lif_and_memory_cache(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """A cache miss executes production numerics and a repeat reuses the result."""
    cache_path = tmp_path / "state" / "query-features.json.gz"
    store = QueryFeatureStore(cache_path)

    features = store.get("tokamak disruption control", _network())
    memory_hit = store.get("tokamak disruption control", _network())

    assert memory_hit is features
    assert features["stimulus"].shape == (32,)
    assert features["stimulus"].any()
    assert features["spikes"].shape == (32,)
    assert features["spikes"].dtype == np.float32
    assert cache_path.is_file()
    assert store.entry_count == 1


def test_fresh_store_restores_real_gzip_without_rewriting_it(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """A new process-shaped store restores arrays from disk on its first request."""
    cache_path = tmp_path / "query-features.json.gz"
    built = QueryFeatureStore(cache_path).get("persistent memory", _network())
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(cache_path, ns=(fixed_mtime_ns, fixed_mtime_ns))

    restored = QueryFeatureStore(cache_path).get("persistent memory", _network())

    assert restored is not built
    np.testing.assert_array_equal(restored["stimulus"], built["stimulus"])
    np.testing.assert_array_equal(restored["spikes"], built["spikes"])
    assert cache_path.stat().st_mtime_ns == fixed_mtime_ns


def test_signature_invalidation_capacity_and_clear_use_real_cache_file(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """Checkpoint identity separates entries while bounded persistence evicts oldest."""
    cache_path = tmp_path / "query-features.json.gz"
    store = QueryFeatureStore(cache_path, capacity=3)
    store.get("alpha", _network("checkpoint-a"))
    store.get("alpha", _network("checkpoint-b"))
    store.get("beta", _network("checkpoint-b"))
    store.get("gamma", _network("checkpoint-b"))

    persisted = load_query_feature_cache(cache_path)
    assert list(persisted) == [
        ("checkpoint-b", "alpha"),
        ("checkpoint-b", "beta"),
        ("checkpoint-b", "gamma"),
    ]
    assert store.entry_count == 3

    store.clear()
    assert store.entry_count == 0
    restored = store.get("alpha", _network("checkpoint-b"))
    np.testing.assert_array_equal(
        restored["stimulus"], persisted[("checkpoint-b", "alpha")]["stimulus"]
    )
    assert store.entry_count == 3


def test_store_rejects_non_positive_capacity(tmp_path: Path) -> None:
    """Invalid bounds fail before any cache mutation."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        QueryFeatureStore(tmp_path / "query-features.json.gz", capacity=0)
