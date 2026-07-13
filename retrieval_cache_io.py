# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Safe retrieval feature-cache persistence

"""Persist retrieval feature caches without unsafe pickle or fake array data."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FeatureArray: TypeAlias = npt.NDArray[np.generic]
QueryCacheKey: TypeAlias = tuple[str, str]
QueryFeatures: TypeAlias = dict[str, FeatureArray]
QueryFeatureCache: TypeAlias = dict[QueryCacheKey, QueryFeatures]


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json_gz(path: Path, data: Mapping[str, object]) -> None:
    """Write compact gzip JSON, including NumPy arrays and scalar values."""
    raw = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    with gzip.open(path, "wb") as cache_file:
        cache_file.write(raw)


def load_json_gz(path: Path) -> dict[str, object] | None:
    """Read a gzip JSON object, returning ``None`` for invalid cache data."""
    try:
        with gzip.open(path, "rb") as cache_file:
            payload = json.loads(cache_file.read())
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def load_pickle_disabled(path: Path) -> None:
    """Reject the retired pickle cache format without deserializing its bytes."""
    del path
    return None


def load_npz_dict(path: Path) -> dict[str, FeatureArray] | None:
    """Read a pickle-free NPZ cache into owned arrays."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError):
        return None


def trace_fingerprint(trace_files: list[Path]) -> str:
    """Hash trace filenames, sizes, and nanosecond mtimes in caller order."""
    digest = hashlib.md5()
    for trace_file in trace_files:
        stat = trace_file.stat()
        digest.update(trace_file.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def load_trace_index_cache(cache_path: Path, cache_key: str) -> dict[str, object] | None:
    """Load and validate a trace-index cache, restoring arrays from JSON lists."""
    cached = load_json_gz(cache_path)
    if cached is None or cached.get("cache_key") != cache_key:
        return None
    trace_spikes = cached.get("trace_spikes")
    trace_names_lower = cached.get("trace_names_lower")
    idf = cached.get("idf")
    if not all(isinstance(value, dict) for value in (trace_spikes, trace_names_lower, idf)):
        return None
    spike_mapping = cast(dict[str, object], trace_spikes)
    return {
        "trace_spikes": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in spike_mapping.items()
        },
        "trace_names_lower": cast(dict[str, object], trace_names_lower),
        "idf": cast(dict[str, object], idf),
    }


def persist_trace_index_cache(
    cache_path: Path,
    cache_key: str,
    payload: Mapping[str, object],
) -> None:
    """Persist a trace-index cache with array-safe JSON conversion."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    trace_spikes = cast(Mapping[str, object], payload["trace_spikes"])
    save_json_gz(
        cache_path,
        {
            "cache_key": cache_key,
            "updated_at": time.time(),
            "trace_count": len(trace_spikes),
            "trace_spikes": trace_spikes,
            "trace_names_lower": payload["trace_names_lower"],
            "idf": payload["idf"],
        },
    )


def load_query_feature_cache(cache_path: Path) -> QueryFeatureCache:
    """Load tuple-keyed query features and restore their NumPy arrays."""
    cached = load_json_gz(cache_path)
    if cached is None:
        return {}
    features: QueryFeatureCache = {}
    for encoded_key, raw_features in cached.items():
        if "\0" not in encoded_key or not isinstance(raw_features, dict):
            continue
        signature, query = encoded_key.split("\0", 1)
        stimulus = raw_features.get("stimulus")
        spikes = raw_features.get("spikes")
        if stimulus is None or spikes is None:
            continue
        features[(signature, query)] = {
            "stimulus": np.asarray(stimulus),
            "spikes": np.asarray(spikes, dtype=np.float32),
        }
    return features


def persist_query_feature_cache(
    cache_path: Path,
    cache: Mapping[QueryCacheKey, QueryFeatures],
    *,
    limit: int = 256,
) -> None:
    """Persist only the newest bounded set of tuple-keyed query features."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(cache.items())[-limit:]
    payload: dict[str, object] = {
        f"{signature}\0{query}": features
        for (signature, query), features in items
    }
    save_json_gz(cache_path, payload)
