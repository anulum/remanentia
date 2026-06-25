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

import json
from io import StringIO
from pathlib import Path

import numpy as np

import vector_pipeline
from memory_index import Document, MemoryIndex
from vector_index import VectorSearchResult
from vector_pipeline import (
    PublicVectorResultPolicy,
    VectorRefreshWorkerConfig,
    chunks_from_memory_index,
    main,
    public_vector_results,
    run_vector_cli,
    run_vector_refresh_worker,
    vector_index_status,
)


class KeywordEmbeddingProvider:
    def embed_texts(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    0.25,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


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


def _built_memory_index() -> MemoryIndex:
    memory = MemoryIndex()
    memory.documents = [
        Document(
            name="decision.md",
            source="trace",
            path="reasoning_traces/decision.md",
            paragraphs=[
                "alpha memory retrieval paragraph",
                "beta factual verifier paragraph",
            ],
            tokens={"alpha", "beta"},
            date="2026-04-27",
            doc_type="trace",
        )
    ]
    memory.paragraph_index = [(0, 0), (0, 1)]
    memory._built = True
    return memory


def _patch(monkeypatch, fake: FakeMemoryIndex, *, needs_rebuild: bool):
    monkeypatch.setattr("memory_index.MemoryIndex", lambda: fake)
    monkeypatch.setattr("memory_index.needs_rebuild", lambda: needs_rebuild)


def _decode_json_stream(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    cursor = 0
    payloads = []
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        payload, cursor = decoder.raw_decode(text, cursor)
        payloads.append(payload)
    return payloads


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


class LazyMemoryIndex:
    def __init__(self) -> None:
        self._built = False
        self.build_calls = 0
        self.documents = []
        self.paragraph_index = []

    def build(self, **kwargs) -> None:
        self.build_calls += 1
        self._built = True
        self.documents = _built_memory_index().documents
        self.paragraph_index = [(0, 0)]


def test_chunks_from_memory_index_builds_lazy_index():
    memory = LazyMemoryIndex()

    chunks = chunks_from_memory_index(memory)

    assert memory.build_calls == 1
    assert chunks[0].text == "alpha memory retrieval paragraph"


def test_worker_records_errors_and_sleeps_between_cycles(tmp_path: Path):
    sleeps: list[float] = []

    def failing_factory(**kwargs):
        raise RuntimeError("index unavailable")

    output = StringIO()
    result = run_vector_refresh_worker(
        VectorRefreshWorkerConfig(
            index_dir=tmp_path / "vec",
            heartbeat_path=tmp_path / "heartbeat.json",
            interval_s=2.5,
            max_cycles=2,
        ),
        KeywordEmbeddingProvider(),
        memory_index_factory=failing_factory,
        sleeper=sleeps.append,
        output=output,
    )

    heartbeats = _decode_json_stream(output.getvalue())
    stored = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in heartbeats] == ["error", "error"]
    assert result["result"]["error_type"] == "RuntimeError"
    assert stored["cycle"] == 2
    assert sleeps == [2.5]


def test_status_and_cli_status_read_existing_manifest(tmp_path: Path):
    index_dir = tmp_path / "vec"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text('{"count": 7}\n', encoding="utf-8")
    output = StringIO()

    status = vector_index_status(index_dir)
    code = run_vector_cli(["status", "--index-dir", str(index_dir)], output=output)

    assert status["manifest"] == {"count": 7}
    assert code == 0
    assert json.loads(output.getvalue())["manifest"] == {"count": 7}


def test_cli_watch_runs_refresh_worker_with_injected_memory(tmp_path: Path):
    output = StringIO()

    code = run_vector_cli(
        [
            "watch",
            "--index-dir",
            str(tmp_path / "vec"),
            "--heartbeat-path",
            str(tmp_path / "heartbeat.json"),
            "--cycles",
            "1",
            "--force-first",
        ],
        provider=KeywordEmbeddingProvider(),
        memory_index=_built_memory_index(),
        output=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["result"]["action"] == "rebuilt"


def test_main_delegates_to_cli(capsys, tmp_path: Path):
    code = main(["status", "--index-dir", str(tmp_path / "missing")])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["exists"] is False


def test_public_results_source_allowlist_and_truncation():
    allowed = VectorSearchResult(
        chunk_id="one",
        text="alpha secret public text that should truncate",
        source="trace",
        score=0.9,
        metadata={"path": "reasoning_traces/decision.md"},
    )
    denied = VectorSearchResult(
        chunk_id="two",
        text="beta",
        source="private",
        score=0.8,
        metadata={"path": "reasoning_traces/private.md"},
    )
    policy = PublicVectorResultPolicy(
        allowed_sources=("trace",),
        redacted_terms=(" ", "secret"),
        max_text_chars=20,
    )

    public = public_vector_results([allowed, denied], policy)

    assert len(public) == 1
    assert public[0]["text"] == "alpha [redacted]..."
    assert public[0]["redactions"] == 1


def test_safe_source_path_falls_back_to_filename_for_external_absolute_path():
    assert vector_pipeline._safe_source_path("/tmp/outside-memory.md") == "outside-memory.md"


def test_load_redaction_terms_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.txt"

    try:
        vector_pipeline._load_redaction_terms(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing redaction file should fail")


def test_load_redaction_terms_skips_blank_and_comment_lines(tmp_path: Path):
    terms = tmp_path / "terms.txt"
    terms.write_text("\n# comment\nalpha\n  beta  \n", encoding="utf-8")

    assert vector_pipeline._load_redaction_terms(terms) == ["alpha", "beta"]
