# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface retrieval cache persistence tests

"""Exercise gzip, NPZ, fingerprint, trace-index and query caches on real files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from retrieval_cache_io import (
    QueryFeatureCache,
    load_json_gz,
    load_npz_dict,
    load_pickle_disabled,
    load_query_feature_cache,
    load_trace_index_cache,
    persist_query_feature_cache,
    persist_trace_index_cache,
    save_json_gz,
    trace_fingerprint,
)


def test_gzip_json_roundtrip_preserves_numpy_and_unicode(tmp_path: Path) -> None:
    """Production JSON conversion persists arrays, NumPy scalars, and Unicode."""
    path = tmp_path / "features.json.gz"
    save_json_gz(
        path,
        {
            "project": "Remanentia Šotek",
            "stimulus": np.array([0.25, 0.5], dtype=np.float32),
            "count": np.int64(2),
        },
    )

    assert load_json_gz(path) == {
        "project": "Remanentia Šotek",
        "stimulus": [0.25, 0.5],
        "count": 2,
    }


def test_gzip_json_rejects_invalid_surfaces_and_unsupported_values(tmp_path: Path) -> None:
    """Missing, corrupt, and non-object caches fail closed without broad exceptions."""
    corrupt = tmp_path / "corrupt.json.gz"
    corrupt.write_bytes(b"not gzip")
    array_payload = tmp_path / "array.json.gz"
    import gzip

    with gzip.open(array_payload, "wb") as cache_file:
        cache_file.write(b"[1,2]")

    assert load_json_gz(tmp_path / "missing.json.gz") is None
    assert load_json_gz(corrupt) is None
    assert load_json_gz(array_payload) is None
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_json_gz(tmp_path / "unsupported.json.gz", {"value": object()})
    assert load_pickle_disabled(tmp_path / "legacy.pkl") is None


def test_npz_cache_loads_numeric_arrays_and_rejects_unsafe_or_invalid_data(
    tmp_path: Path,
) -> None:
    """NPZ loading owns numeric arrays and refuses object-pickle or corrupt archives."""
    valid = tmp_path / "valid.npz"
    unsafe = tmp_path / "unsafe.npz"
    corrupt = tmp_path / "corrupt.npz"
    np.savez(valid, embedding=np.arange(4, dtype=np.float32))
    np.savez(unsafe, payload=np.array([object()], dtype=object))
    corrupt.write_bytes(b"not an npz")

    loaded = load_npz_dict(valid)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["embedding"], np.arange(4, dtype=np.float32))
    assert load_npz_dict(unsafe) is None
    assert load_npz_dict(corrupt) is None
    assert load_npz_dict(tmp_path / "missing.npz") is None


def test_trace_fingerprint_tracks_real_file_identity(tmp_path: Path) -> None:
    """Content size and mtime changes invalidate the deterministic trace fingerprint."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    initial = trace_fingerprint([first, second])
    assert initial == trace_fingerprint([first, second])
    first.write_text("alpha expanded", encoding="utf-8")
    os.utime(first, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns + 1))

    assert trace_fingerprint([first, second]) != initial
    assert trace_fingerprint([second, first]) != trace_fingerprint([first, second])


def test_trace_index_cache_roundtrip_restores_numeric_spike_arrays(tmp_path: Path) -> None:
    """Trace spike arrays survive gzip JSON persistence as production NumPy arrays."""
    path = tmp_path / "trace-index.json.gz"
    persist_trace_index_cache(
        path,
        "state-42",
        {
            "trace_spikes": {
                "alpha.md": np.array([1, 0, 2], dtype=np.float32),
                "beta.md": np.array([0, 3, 0], dtype=np.float32),
            },
            "trace_names_lower": {"alpha.md": "alpha md", "beta.md": "beta md"},
            "idf": {"alpha": 1.5, "beta": 1.5},
        },
    )

    loaded = load_trace_index_cache(path, "state-42")
    assert loaded is not None
    assert load_trace_index_cache(path, "other-state") is None
    spikes = loaded["trace_spikes"]
    assert isinstance(spikes, dict)
    np.testing.assert_array_equal(spikes["alpha.md"], np.array([1, 0, 2], dtype=np.float32))


@pytest.mark.parametrize(
    "payload",
    [
        {"cache_key": "key"},
        {
            "cache_key": "key",
            "trace_spikes": [],
            "trace_names_lower": {},
            "idf": {},
        },
        {
            "cache_key": "key",
            "trace_spikes": {},
            "trace_names_lower": [],
            "idf": {},
        },
        {
            "cache_key": "key",
            "trace_spikes": {},
            "trace_names_lower": {},
            "idf": [],
        },
    ],
)
def test_trace_index_cache_rejects_incomplete_schema(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """Every required trace-index mapping is validated before cache reuse."""
    path = tmp_path / "invalid-trace-index.json.gz"
    save_json_gz(path, payload)
    assert load_trace_index_cache(path, "key") is None


def test_query_feature_cache_roundtrip_filters_and_bounds_real_entries(tmp_path: Path) -> None:
    """Tuple keys and arrays roundtrip while persistence retains the newest entries."""
    path = tmp_path / "query-features.json.gz"
    cache: QueryFeatureCache = {
        ("sig-1", "alpha"): {
            "stimulus": np.array([1.0, 0.0]),
            "spikes": np.array([2.0, 0.0], dtype=np.float32),
        },
        ("sig-2", "beta"): {
            "stimulus": np.array([0.0, 1.0]),
            "spikes": np.array([0.0, 3.0], dtype=np.float32),
        },
        ("sig-3", "gamma"): {
            "stimulus": np.array([0.5, 0.5]),
            "spikes": np.array([1.0, 1.0], dtype=np.float32),
        },
    }
    persist_query_feature_cache(path, cache, limit=2)

    loaded = load_query_feature_cache(path)
    assert set(loaded) == {("sig-2", "beta"), ("sig-3", "gamma")}
    np.testing.assert_array_equal(loaded[("sig-3", "gamma")]["spikes"], [1.0, 1.0])

    save_json_gz(
        path,
        {
            "plain-key": {"stimulus": [1], "spikes": [1]},
            "sig\0not-a-mapping": "invalid",
            "sig\0missing-spikes": {"stimulus": [1]},
            "sig\0missing-stimulus": {"spikes": [1]},
            "sig\0valid": {"stimulus": [0.2], "spikes": [4]},
        },
    )
    assert set(load_query_feature_cache(path)) == {("sig", "valid")}
    assert load_query_feature_cache(tmp_path / "absent.json.gz") == {}
