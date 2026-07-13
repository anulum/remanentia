# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Persistent retrieval query features

"""Build and persist deterministic query encoding and spiking features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

try:
    from .retrieval_cache_io import (  # type: ignore[import-not-found]
        QueryFeatureCache,
        QueryFeatures,
        load_query_feature_cache,
        persist_query_feature_cache,
    )
    from .retrieval_spiking import (  # type: ignore[import-not-found]
        FloatArray,
        encode_text,
        spike_feature,
    )
except ImportError:
    from retrieval_cache_io import (
        QueryFeatureCache,
        QueryFeatures,
        load_query_feature_cache,
        persist_query_feature_cache,
    )
    from retrieval_spiking import FloatArray, encode_text, spike_feature


class RetrievalNetworkState(TypedDict):
    """Network fields required to derive query features."""

    v: FloatArray
    w: FloatArray
    _state_signature: str


@dataclass
class QueryFeatureStore:
    """Lazily load, compute, bound, and persist query features."""

    cache_path: Path
    capacity: int = 256
    _cache: QueryFeatureCache = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")

    def _load(self) -> None:
        if self._loaded:
            return
        self._cache.update(load_query_feature_cache(self.cache_path))
        self._loaded = True

    def get(self, query: str, network: RetrievalNetworkState) -> QueryFeatures:
        """Return cached features or compute them through production numerics."""
        self._load()
        cache_key = (network["_state_signature"], query)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        stimulus = encode_text(query, len(network["v"]))
        features: QueryFeatures = {
            "stimulus": stimulus,
            "spikes": spike_feature(network["w"], stimulus),
        }
        self._cache[cache_key] = features
        while len(self._cache) > self.capacity:
            del self._cache[next(iter(self._cache))]
        persist_query_feature_cache(
            self.cache_path,
            self._cache,
            limit=self.capacity,
        )
        return features

    def clear(self) -> None:
        """Forget loaded in-memory state so the next request rereads disk."""
        self._cache.clear()
        self._loaded = False

    @property
    def entry_count(self) -> int:
        """Return the number of currently loaded entries."""
        return len(self._cache)


__all__ = ["QueryFeatureStore", "RetrievalNetworkState"]
