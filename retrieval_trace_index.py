# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval trace-index construction

"""Build the in-memory and durable retrieval index from real Markdown traces."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt

try:
    from .retrieval_cache_io import (  # type: ignore[import-not-found]
        load_trace_index_cache,
        persist_trace_index_cache,
        trace_fingerprint,
    )
    from .retrieval_spiking import (  # type: ignore[import-not-found]
        FloatArray,
        encode_text,
        spike_feature,
    )
    from .retrieval_text import build_idf  # type: ignore[import-not-found]
except ImportError:
    from retrieval_cache_io import (
        load_trace_index_cache,
        persist_trace_index_cache,
        trace_fingerprint,
    )
    from retrieval_spiking import FloatArray, encode_text, spike_feature
    from retrieval_text import build_idf


class RetrievalNetworkState(TypedDict):
    """Network fields required to build a trace index."""

    v: FloatArray
    w: FloatArray
    _state_signature: str
    _encoding_backend: str


class TraceIndex(TypedDict):
    """Loaded trace corpus and its derived retrieval features."""

    trace_files: list[Path]
    trace_texts: dict[str, str]
    trace_spikes: dict[str, npt.NDArray[np.float32]]
    trace_names_lower: dict[str, str]
    idf: dict[str, float]


TraceIndexMemoryCache = dict[str, TraceIndex]


def _read_trace_corpus(
    traces_dir: Path,
    semantic_dir: Path,
) -> tuple[list[Path], list[Path], dict[str, str]]:
    trace_files = sorted(traces_dir.glob("*.md"))
    semantic_files = sorted(semantic_dir.rglob("*.md")) if semantic_dir.exists() else []
    trace_texts = {
        trace_file.name: trace_file.read_text(encoding="utf-8")
        for trace_file in trace_files
    }
    trace_texts.update(
        {
            f"[semantic] {semantic_file.relative_to(semantic_dir)}": semantic_file.read_text(
                encoding="utf-8"
            )
            for semantic_file in semantic_files
        }
    )
    return trace_files, semantic_files, trace_texts


def _restore_disk_index(
    trace_files: list[Path],
    trace_texts: dict[str, str],
    disk_index: dict[str, object],
) -> TraceIndex:
    return {
        "trace_files": trace_files,
        "trace_texts": trace_texts,
        "trace_spikes": cast(
            dict[str, npt.NDArray[np.float32]], disk_index["trace_spikes"]
        ),
        "trace_names_lower": cast(dict[str, str], disk_index["trace_names_lower"]),
        "idf": cast(dict[str, float], disk_index["idf"]),
    }


def build_trace_index(
    traces_dir: Path,
    semantic_dir: Path,
    cache_path: Path,
    network: RetrievalNetworkState,
    memory_cache: TraceIndexMemoryCache,
) -> TraceIndex:
    """Build or restore a retrieval trace index for one network state."""
    trace_files, semantic_files, trace_texts = _read_trace_corpus(traces_dir, semantic_dir)
    cache_key = (
        f"{network['_state_signature']}:{network['_encoding_backend']}:"
        f"{trace_fingerprint(trace_files + semantic_files)}"
    )
    cached = memory_cache.get(cache_key)
    if cached is not None:
        return cached

    disk_index = load_trace_index_cache(cache_path, cache_key)
    if disk_index is not None:
        restored = _restore_disk_index(trace_files, trace_texts, disk_index)
        memory_cache.clear()
        memory_cache[cache_key] = restored
        return restored

    neuron_count = len(network["v"])
    trace_spikes: dict[str, npt.NDArray[np.float32]] = {}
    trace_names_lower: dict[str, str] = {}
    for trace_name, text in trace_texts.items():
        stimulus = encode_text(text, neuron_count)
        trace_spikes[trace_name] = spike_feature(network["w"], stimulus)
        trace_names_lower[trace_name] = trace_name.lower().replace("-", " ").replace("_", " ")

    built: TraceIndex = {
        "trace_files": trace_files,
        "trace_texts": trace_texts,
        "trace_spikes": trace_spikes,
        "trace_names_lower": trace_names_lower,
        "idf": build_idf(trace_texts),
    }
    persist_trace_index_cache(cache_path, cache_key, built)
    memory_cache.clear()
    memory_cache[cache_key] = built
    return built
