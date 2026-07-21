# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the vector pipeline's index factory

"""Real storage and command-path tests for the vector pipeline."""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import memory_index
import numpy as np
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import vector_pipeline
from memory_index import Document, MemoryIndex
from vector_index import VectorSearchResult
from vector_pipeline import (
    PublicVectorResultPolicy,
    VectorRefreshWorkerConfig,
    chunks_from_memory_index,
    main,
    public_vector_results,
    refresh_memory_vector_index,
    run_vector_cli,
    run_vector_refresh_worker,
    vector_index_status,
)


class KeywordEmbeddingProvider:
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    0.25,
                ]
            )
        embeddings: np.ndarray = np.asarray(rows, dtype=np.float32)
        return embeddings


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


def _configure_real_memory_storage(monkeypatch: MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "memory"
    source_dir.mkdir()
    index_path = tmp_path / "state" / "memory_index.json.gz"
    monkeypatch.setattr(memory_index, "SOURCES", {"test": source_dir})
    monkeypatch.setattr(memory_index, "SOURCE_EXTENSIONS", {"test": {".md"}})
    monkeypatch.setattr(memory_index, "INDEX_PATH", index_path)
    monkeypatch.setattr(memory_index, "_LEGACY_INDEX_PATH", tmp_path / "state" / "legacy.pkl")
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", tmp_path / "state" / "hashes.json")
    return source_dir, index_path


def _write_memory_source(path: Path, fact: str) -> None:
    path.write_text(f"# Retrieval memory\n\n{fact}\n", encoding="utf-8")


def _build_saved_memory_index(source_dir: Path) -> None:
    _write_memory_source(
        source_dir / "memory.md",
        "Alpha retrieval remains the current persisted production fact.",
    )
    index = MemoryIndex()
    index.build(use_gpu_embeddings=False, use_gliner=False)
    index.save()


def _decode_json_stream(text: str) -> list[dict[str, Any]]:
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
    def test_reuses_current_cache_without_rebuilding(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        source_dir, index_path = _configure_real_memory_storage(monkeypatch, tmp_path)
        _build_saved_memory_index(source_dir)
        saved_mtime = index_path.stat().st_mtime_ns

        result = vector_pipeline.load_or_build_memory_index()

        assert result.search("alpha persisted production", top_k=1)[0].name == "memory.md"
        assert index_path.stat().st_mtime_ns == saved_mtime

    def test_rebuilds_when_a_source_is_newer(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        source_dir, index_path = _configure_real_memory_storage(monkeypatch, tmp_path)
        _build_saved_memory_index(source_dir)
        saved_mtime = index_path.stat().st_mtime_ns
        source_path = source_dir / "memory.md"
        _write_memory_source(
            source_path,
            "Gamma retrieval is the updated production fact after refresh.",
        )
        os.utime(source_path, ns=(saved_mtime + 1_000_000, saved_mtime + 1_000_000))

        result = vector_pipeline.load_or_build_memory_index()

        assert result.search("gamma updated production", top_k=1)[0].name == "memory.md"
        assert result.search("alpha persisted", top_k=1) == []
        assert index_path.stat().st_mtime_ns > saved_mtime

    def test_builds_when_no_cache_exists(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        source_dir, index_path = _configure_real_memory_storage(monkeypatch, tmp_path)
        _write_memory_source(
            source_dir / "memory.md",
            "Beta retrieval is built from the live source when no cache exists.",
        )

        result = vector_pipeline.load_or_build_memory_index()

        assert index_path.exists()
        assert result.search("beta live source", top_k=1)[0].name == "memory.md"


def test_chunks_from_memory_index_builds_unbuilt_real_index(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_dir, _ = _configure_real_memory_storage(monkeypatch, tmp_path)
    _write_memory_source(
        source_dir / "memory.md",
        "Alpha memory retrieval is indexed through the production build path.",
    )
    memory = MemoryIndex()

    chunks = chunks_from_memory_index(memory)

    assert memory._built is True
    assert chunks[0].text.startswith("Alpha memory retrieval")


def test_worker_records_real_filesystem_errors_and_sleeps_between_cycles(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    sleeps: list[float] = []
    source_dir, _ = _configure_real_memory_storage(monkeypatch, tmp_path)
    _write_memory_source(
        source_dir / "memory.md",
        "Alpha memory reaches the worker before its vector storage failure.",
    )
    blocked_index_path = tmp_path / "vec"
    blocked_index_path.write_text("not a directory", encoding="utf-8")
    output = StringIO()
    result = run_vector_refresh_worker(
        VectorRefreshWorkerConfig(
            index_dir=blocked_index_path,
            heartbeat_path=tmp_path / "heartbeat.json",
            interval_s=2.5,
            max_cycles=2,
        ),
        KeywordEmbeddingProvider(),
        sleeper=sleeps.append,
        output=output,
    )

    heartbeats = _decode_json_stream(output.getvalue())
    stored = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in heartbeats] == ["error", "error"]
    assert result["result"]["error_type"] == "FileExistsError"
    assert stored["cycle"] == 2
    assert sleeps == [2.5]


def test_status_and_cli_status_read_existing_manifest(tmp_path: Path) -> None:
    index_dir = tmp_path / "vec"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text('{"count": 7}\n', encoding="utf-8")
    output = StringIO()

    status = vector_index_status(index_dir)
    code = run_vector_cli(["status", "--index-dir", str(index_dir)], output=output)

    assert status["manifest"] == {"count": 7}
    assert code == 0
    assert json.loads(output.getvalue())["manifest"] == {"count": 7}


class TestManifestResilience:
    """A torn/unreadable manifest must self-heal, never stall the worker.

    The manifest is a rebuildable derived artefact and the refresh gate reads
    it *before* the rebuild that would repair it, so an unguarded ``json.loads``
    turned a crash mid-write into a permanent stall — the per-cycle ``except``
    in the worker can log but never get past the poisoned read.
    """

    def test_read_manifest_treats_corrupt_json_as_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text('{"corpus_fingerprint": "abc", trunc', encoding="utf-8")

        assert vector_pipeline._read_manifest(path) == {}

    def test_read_manifest_treats_non_object_as_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")

        assert vector_pipeline._read_manifest(path) == {}

    def test_merge_manifest_writes_atomically_over_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{ half-written", encoding="utf-8")

        vector_pipeline._merge_manifest(path, {"corpus_fingerprint": "deadbeef"})

        assert json.loads(path.read_text(encoding="utf-8")) == {"corpus_fingerprint": "deadbeef"}
        # atomic_write_text cleans up its own tmpfile — no debris left behind.
        assert list(tmp_path.glob("*.tmp")) == []

    def test_refresh_rebuilds_instead_of_raising_on_corrupt_manifest(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "vec"
        provider = KeywordEmbeddingProvider()
        # First refresh builds the index and writes a valid manifest.
        first = refresh_memory_vector_index(_built_memory_index(), index_dir, provider)
        assert first["action"] == "rebuilt"

        # Corrupt the manifest as a crash mid-write would.
        (index_dir / "manifest.json").write_text("{ torn", encoding="utf-8")

        # The corpus is unchanged, so the pre-bug behaviour would have raised
        # JSONDecodeError; the fix must instead rebuild and repair the manifest.
        second = refresh_memory_vector_index(_built_memory_index(), index_dir, provider)
        assert second["action"] == "rebuilt"
        assert second["changed"] is True
        # Manifest is valid JSON again and carries the fingerprint.
        healed = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        assert healed["corpus_fingerprint"] == second["corpus_fingerprint"]

    def test_status_does_not_raise_on_corrupt_manifest(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "vec"
        index_dir.mkdir()
        (index_dir / "manifest.json").write_text("{ not json", encoding="utf-8")

        status = vector_index_status(index_dir)

        assert status["manifest"] == {}


def test_cli_watch_runs_refresh_worker_with_real_memory(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_dir, _ = _configure_real_memory_storage(monkeypatch, tmp_path)
    _write_memory_source(
        source_dir / "memory.md",
        "Alpha memory is refreshed through the production watch command.",
    )
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
        output=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["result"]["action"] == "rebuilt"


def test_cli_build_refresh_search_and_estimate_roundtrip(tmp_path: Path) -> None:
    index_dir = tmp_path / "vec"
    provider = KeywordEmbeddingProvider()
    memory = _built_memory_index()

    build_output = StringIO()
    assert (
        run_vector_cli(
            ["build", "--index-dir", str(index_dir), "--batch-size", "1"],
            provider=provider,
            memory_index=memory,
            output=build_output,
        )
        == 0
    )
    assert json.loads(build_output.getvalue())["count"] == 2

    refresh_output = StringIO()
    assert (
        run_vector_cli(
            ["refresh", "--index-dir", str(index_dir)],
            provider=provider,
            memory_index=memory,
            output=refresh_output,
        )
        == 0
    )
    assert json.loads(refresh_output.getvalue())["action"] == "rebuilt"

    unchanged_output = StringIO()
    assert (
        run_vector_cli(
            ["refresh", "--index-dir", str(index_dir)],
            provider=provider,
            memory_index=memory,
            output=unchanged_output,
        )
        == 0
    )
    assert json.loads(unchanged_output.getvalue())["action"] == "skipped"

    forced_output = StringIO()
    assert (
        run_vector_cli(
            ["refresh", "--index-dir", str(index_dir), "--force"],
            provider=provider,
            memory_index=memory,
            output=forced_output,
        )
        == 0
    )
    assert json.loads(forced_output.getvalue())["action"] == "rebuilt"

    search_output = StringIO()
    assert (
        run_vector_cli(
            ["search", "alpha", "--index-dir", str(index_dir), "--top", "1"],
            provider=provider,
            output=search_output,
        )
        == 0
    )
    assert json.loads(search_output.getvalue())[0]["text"].startswith("alpha")

    redactions = tmp_path / "redactions.txt"
    redactions.write_text("retrieval\n", encoding="utf-8")
    public_output = StringIO()
    assert (
        run_vector_cli(
            [
                "search",
                "alpha",
                "--index-dir",
                str(index_dir),
                "--public",
                "--public-source",
                "trace",
                "--public-path-prefix",
                "reasoning_traces",
                "--redaction-file",
                str(redactions),
                "--max-text-chars",
                "0",
            ],
            provider=provider,
            output=public_output,
        )
        == 0
    )
    public_results = json.loads(public_output.getvalue())
    assert public_results[0]["text"] == "alpha memory [redacted] paragraph"
    assert public_results[0]["metadata"]["paragraph_idx"] == 0

    denied_output = StringIO()
    assert (
        run_vector_cli(
            ["search", "alpha", "--index-dir", str(index_dir), "--public"],
            provider=provider,
            output=denied_output,
        )
        == 0
    )
    assert json.loads(denied_output.getvalue()) == []

    estimate_output = StringIO()
    assert (
        run_vector_cli(
            ["estimate", "25", "--dimension", "3"],
            output=estimate_output,
        )
        == 0
    )
    assert json.loads(estimate_output.getvalue())["count"] == 25


def test_real_memory_build_filters_private_paths_and_bounds_chunks(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_dir, _ = _configure_real_memory_storage(monkeypatch, tmp_path)
    private_dir = source_dir / ".coordination"
    private_dir.mkdir()
    _write_memory_source(
        source_dir / "public.md",
        " ".join(["Alpha production retrieval sentence"] * 80),
    )
    _write_memory_source(
        private_dir / "private.md",
        "Private coordination memory remains outside public vector output.",
    )
    memory = MemoryIndex()
    memory.build(use_gpu_embeddings=False, use_gliner=False)
    memory.paragraph_index.append(memory.paragraph_index[0])
    memory.save()
    loaded_memory = MemoryIndex()
    assert loaded_memory.load() is True

    public_chunks = chunks_from_memory_index(loaded_memory)
    all_chunks = chunks_from_memory_index(loaded_memory, include_private=True)

    assert public_chunks
    assert len({chunk.chunk_id for chunk in all_chunks}) == len(all_chunks)
    assert all(len(chunk.text) <= vector_pipeline.VECTOR_CHUNK_MAX_CHARS for chunk in all_chunks)
    assert not any("private.md" in chunk.metadata["path"] for chunk in public_chunks)
    assert any("private.md" in chunk.metadata["path"] for chunk in all_chunks)


def test_unbounded_worker_writes_first_real_cycle_before_external_stop(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_dir, _ = _configure_real_memory_storage(monkeypatch, tmp_path)
    _write_memory_source(
        source_dir / "memory.md",
        "Alpha memory is refreshed before the external worker stop signal.",
    )
    output = StringIO()

    def stop_after_first_interval(_interval: float) -> None:
        raise StopIteration

    try:
        run_vector_refresh_worker(
            VectorRefreshWorkerConfig(
                index_dir=tmp_path / "vec",
                heartbeat_path=tmp_path / "heartbeat.json",
                interval_s=0.01,
            ),
            KeywordEmbeddingProvider(),
            sleeper=stop_after_first_interval,
            output=output,
        )
    except StopIteration:
        pass
    else:
        raise AssertionError("external worker stop was not propagated")

    assert json.loads(output.getvalue())["status"] == "ok"
    assert (tmp_path / "heartbeat.json").exists()


def test_zero_cycle_worker_returns_without_writing(tmp_path: Path) -> None:
    result = run_vector_refresh_worker(
        VectorRefreshWorkerConfig(
            index_dir=tmp_path / "vec",
            heartbeat_path=tmp_path / "heartbeat.json",
            max_cycles=0,
        ),
        KeywordEmbeddingProvider(),
    )

    assert result == {}
    assert not (tmp_path / "heartbeat.json").exists()


def test_main_delegates_to_cli(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    code = main(["status", "--index-dir", str(tmp_path / "missing")])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["exists"] is False


def test_public_results_source_allowlist_and_truncation() -> None:
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


def test_safe_source_path_falls_back_to_filename_for_external_absolute_path() -> None:
    assert vector_pipeline._safe_source_path("/tmp/outside-memory.md") == "outside-memory.md"


def test_load_redaction_terms_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    try:
        vector_pipeline._load_redaction_terms(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing redaction file should fail")


def test_load_redaction_terms_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    terms = tmp_path / "terms.txt"
    terms.write_text("\n# comment\nalpha\n  beta  \n", encoding="utf-8")

    assert vector_pipeline._load_redaction_terms(terms) == ["alpha", "beta"]
