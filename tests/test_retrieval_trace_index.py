# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface trace-index construction tests

"""Exercise Markdown-to-LIF indexing through real memory and gzip caches."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from REMANENTIA.retrieval_trace_index import (
    RetrievalNetworkState,
    TraceIndexMemoryCache,
    build_trace_index,
)


@pytest.fixture
def hash_backend() -> Iterator[None]:
    """Use the real deterministic hash encoder and restore global backend state."""
    import encoding

    original = encoding.get_backend()
    encoding.set_backend("hash")
    try:
        yield
    finally:
        encoding.set_backend(original)


def _network(signature: str = "state-a") -> RetrievalNetworkState:
    return {
        "v": np.zeros(32, dtype=np.float64),
        "w": np.eye(32, dtype=np.float64) * 0.02,
        "_state_signature": signature,
        "_encoding_backend": "hash",
    }


def test_empty_trace_workspace_builds_and_persists_empty_index(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """An absent semantic directory and empty trace directory remain valid real inputs."""
    traces = tmp_path / "traces"
    traces.mkdir()
    cache_path = tmp_path / "state" / "trace-index.json.gz"
    memory_cache: TraceIndexMemoryCache = {}

    index = build_trace_index(
        traces,
        tmp_path / "absent-semantic",
        cache_path,
        _network(),
        memory_cache,
    )

    assert index == {
        "trace_files": [],
        "trace_texts": {},
        "trace_spikes": {},
        "trace_names_lower": {},
        "idf": {},
    }
    assert cache_path.is_file()
    assert len(memory_cache) == 1


def test_real_trace_and_semantic_corpus_builds_lif_and_lexical_features(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """Root traces and nested semantic memories traverse encoding, LIF, and IDF."""
    traces = tmp_path / "traces"
    semantic = tmp_path / "semantic"
    traces.mkdir()
    (traces / "tokamak-control.md").write_text(
        "Tokamak disruption control uses resonant magnetic perturbation.",
        encoding="utf-8",
    )
    (traces / "persistent_memory.md").write_text(
        "Persistent memory retrieval combines lexical and neural signals.",
        encoding="utf-8",
    )
    nested_trace = traces / "nested" / "ignored.md"
    nested_trace.parent.mkdir()
    nested_trace.write_text("Nested reasoning trace is outside the root corpus.", encoding="utf-8")
    semantic_record = semantic / "findings" / "measured.md"
    semantic_record.parent.mkdir(parents=True)
    semantic_record.write_text(
        "Measured retrieval precision reached ninety two percent.",
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache" / "trace-index.json.gz"
    memory_cache: TraceIndexMemoryCache = {}

    index = build_trace_index(traces, semantic, cache_path, _network(), memory_cache)

    assert [path.name for path in index["trace_files"]] == [
        "persistent_memory.md",
        "tokamak-control.md",
    ]
    assert set(index["trace_texts"]) == {
        "persistent_memory.md",
        "tokamak-control.md",
        "[semantic] findings/measured.md",
    }
    assert "ignored.md" not in index["trace_texts"]
    assert index["trace_names_lower"]["tokamak-control.md"] == "tokamak control.md"
    assert "tokamak" in index["idf"]
    assert "persistent" in index["idf"]
    for spikes in index["trace_spikes"].values():
        assert spikes.shape == (32,)
        assert spikes.dtype == np.float32
    assert cache_path.is_file()


def test_memory_and_disk_cache_paths_are_real_and_trace_changes_invalidate_them(
    tmp_path: Path,
    hash_backend: None,
) -> None:
    """Stable inputs reuse memory/disk artifacts; a real trace edit creates a new key."""
    traces = tmp_path / "traces"
    traces.mkdir()
    trace = traces / "control.md"
    trace.write_text("Tokamak control baseline.", encoding="utf-8")
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    cache_path = tmp_path / "cache" / "trace-index.json.gz"
    network = _network()
    memory_cache: TraceIndexMemoryCache = {}

    built = build_trace_index(traces, semantic, cache_path, network, memory_cache)
    memory_hit = build_trace_index(traces, semantic, cache_path, network, memory_cache)
    assert memory_hit is built

    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(cache_path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    disk_memory: TraceIndexMemoryCache = {}
    restored = build_trace_index(traces, semantic, cache_path, network, disk_memory)
    assert restored is not built
    np.testing.assert_array_equal(
        restored["trace_spikes"]["control.md"],
        built["trace_spikes"]["control.md"],
    )
    assert cache_path.stat().st_mtime_ns == fixed_mtime_ns

    trace.write_text("Tokamak control baseline with a measured disruption result.", encoding="utf-8")
    os.utime(trace, ns=(trace.stat().st_atime_ns, trace.stat().st_mtime_ns + 1))
    invalidated = build_trace_index(traces, semantic, cache_path, network, disk_memory)

    assert invalidated is not restored
    assert "measured disruption" in invalidated["trace_texts"]["control.md"]
    assert len(disk_memory) == 1
    assert cache_path.stat().st_mtime_ns != fixed_mtime_ns
