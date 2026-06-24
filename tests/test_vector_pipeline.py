# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the vector pipeline's index factory

"""Tests for ``load_or_build_memory_index`` staleness gating.

The refresh worker fed the vector index a frozen cache because this factory
returned the persisted ``MemoryIndex`` whenever it merely existed. These tests
pin the corrected contract: reuse the cache only when it is current, rebuild
whenever a source is newer (or no cache exists). The ``MemoryIndex`` is faked
so the gate's three branches are exercised without touching the real 910 MB
index or any embedding model.
"""

from __future__ import annotations

import vector_pipeline


class FakeMemoryIndex:
    """Records whether build/save ran, with a configurable load outcome."""

    def __init__(self, *, load_ok: bool):
        self._load_ok = load_ok
        self.built = False
        self.saved = False
        self.build_kwargs: dict = {}

    def load(self) -> bool:
        return self._load_ok

    def build(self, **kwargs) -> None:
        self.built = True
        self.build_kwargs = kwargs

    def save(self, *args, **kwargs) -> None:
        self.saved = True


def _patch(monkeypatch, fake: FakeMemoryIndex, *, needs_rebuild: bool):
    monkeypatch.setattr("memory_index.MemoryIndex", lambda: fake)
    monkeypatch.setattr("memory_index.needs_rebuild", lambda: needs_rebuild)


class TestLoadOrBuildMemoryIndex:
    def test_reuses_current_cache_without_rebuilding(self, monkeypatch):
        fake = FakeMemoryIndex(load_ok=True)
        _patch(monkeypatch, fake, needs_rebuild=False)

        result = vector_pipeline.load_or_build_memory_index()

        assert result is fake
        assert fake.built is False
        assert fake.saved is False

    def test_rebuilds_when_a_source_is_newer(self, monkeypatch):
        fake = FakeMemoryIndex(load_ok=True)
        _patch(monkeypatch, fake, needs_rebuild=True)

        result = vector_pipeline.load_or_build_memory_index()

        assert result is fake
        assert fake.built is True
        assert fake.saved is True

    def test_builds_when_no_cache_exists(self, monkeypatch):
        fake = FakeMemoryIndex(load_ok=False)
        # needs_rebuild is short-circuited by the failed load; still safe to set.
        _patch(monkeypatch, fake, needs_rebuild=False)

        result = vector_pipeline.load_or_build_memory_index()

        assert result is fake
        assert fake.built is True
        assert fake.saved is True

    def test_passes_gpu_embeddings_flag_through_to_build(self, monkeypatch):
        fake = FakeMemoryIndex(load_ok=False)
        _patch(monkeypatch, fake, needs_rebuild=False)

        vector_pipeline.load_or_build_memory_index(use_gpu_embeddings=True)

        assert fake.build_kwargs.get("use_gpu_embeddings") is True
