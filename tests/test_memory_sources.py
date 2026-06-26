# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for memory source configuration

"""Tests for configurable MemoryIndex source roots."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_default_source_config_uses_neutral_repo_roots(tmp_path: Path) -> None:
    """Default source roots should be package-local and public-deployment neutral."""
    from memory_sources import build_source_config

    config = build_source_config(base=tmp_path)

    assert set(config.sources) == {
        "traces",
        "paper",
        "semantic",
        "compiled",
        "code_remanentia",
    }
    assert config.sources["traces"] == tmp_path / "reasoning_traces"
    assert config.sources["code_remanentia"] == tmp_path
    assert config.extensions["compiled"] == frozenset({".md", ".jsonl"})


def test_file_config_merges_extra_sources_relative_to_config_file(tmp_path: Path) -> None:
    """A JSON source file should merge external roots without hardcoded paths."""
    from memory_sources import build_source_config

    config_dir = tmp_path / "settings"
    archive_dir = tmp_path / "archives" / "decisions"
    config_dir.mkdir()
    archive_dir.mkdir(parents=True)
    config_path = config_dir / "memory_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": {
                    "decision_archive": {
                        "path": "../archives/decisions",
                        "extensions": [".md", "jsonl"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = build_source_config(base=tmp_path / "repo", config_path=config_path)

    assert config.sources["decision_archive"] == archive_dir
    assert config.extensions["decision_archive"] == frozenset({".md", ".jsonl"})
    assert "code_remanentia" in config.sources


def test_inline_config_can_replace_default_sources(tmp_path: Path) -> None:
    """Inline JSON should support hermetic deployments with only configured roots."""
    from memory_sources import build_source_config

    docs_dir = tmp_path / "docs"
    inline = json.dumps(
        {
            "extends_defaults": False,
            "sources": {
                "manuals": {
                    "path": str(docs_dir),
                    "extensions": [".md"],
                }
            },
        }
    )

    config = build_source_config(
        base=tmp_path / "repo",
        environ={"REMANENTIA_MEMORY_SOURCES_JSON": inline},
    )

    assert config.sources == {"manuals": docs_dir}
    assert config.extensions == {"manuals": frozenset({".md"})}


def test_invalid_source_config_rejects_private_or_ambiguous_shapes(tmp_path: Path) -> None:
    """Invalid source definitions should fail before indexing starts."""
    from memory_sources import build_source_config

    with pytest.raises(ValueError, match="sources must be an object"):
        build_source_config(
            base=tmp_path,
            environ={"REMANENTIA_MEMORY_SOURCES_JSON": json.dumps({"sources": []})},
        )

    with pytest.raises(ValueError, match="must define a non-empty path"):
        build_source_config(
            base=tmp_path,
            environ={
                "REMANENTIA_MEMORY_SOURCES_JSON": json.dumps(
                    {"sources": {"broken": {"extensions": [".md"]}}}
                )
            },
        )

    with pytest.raises(ValueError, match="source label"):
        build_source_config(
            base=tmp_path,
            environ={
                "REMANENTIA_MEMORY_SOURCES_JSON": json.dumps(
                    {"sources": {"not/path": {"path": "notes", "extensions": [".md"]}}}
                )
            },
        )


def test_env_config_file_and_inline_config_merge_sources(tmp_path: Path) -> None:
    """Env-file and inline JSON configs should merge through the public loader."""
    from memory_sources import build_source_config

    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    file_config = config_dir / "sources.json"
    file_config.write_text(
        json.dumps(
            {
                "sources": {
                    "from_file": {
                        "path": "file-root",
                        "extensions": [".md"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    inline = json.dumps(
        {
            "sources": {
                "from_inline": {
                    "path": "inline-root",
                    "extensions": [".jsonl"],
                }
            }
        }
    )

    config = build_source_config(
        base=tmp_path,
        environ={
            "REMANENTIA_MEMORY_SOURCES_CONFIG": str(file_config),
            "REMANENTIA_MEMORY_SOURCES_JSON": inline,
        },
    )

    assert config.sources["from_file"] == config_dir / "file-root"
    assert config.extensions["from_file"] == frozenset({".md"})
    assert config.sources["from_inline"] == tmp_path / "inline-root"
    assert config.extensions["from_inline"] == frozenset({".jsonl"})


def test_scalar_source_uses_text_extensions(tmp_path: Path) -> None:
    """String source entries should resolve paths and use text suffix defaults."""
    from memory_sources import DEFAULT_TEXT_EXTENSIONS, build_source_config

    config = build_source_config(
        base=tmp_path,
        environ={
            "REMANENTIA_MEMORY_SOURCES_JSON": json.dumps(
                {"extends_defaults": False, "sources": {"notes": "notes"}}
            )
        },
    )

    assert config.sources == {"notes": tmp_path / "notes"}
    assert config.extensions == {"notes": DEFAULT_TEXT_EXTENSIONS}


def test_invalid_top_level_and_extension_shapes_fail(tmp_path: Path) -> None:
    """Malformed configs should fail with actionable ValueErrors."""
    from memory_sources import build_source_config

    bad_cases = [
        ("memory source config must be an object", []),
        ("extends_defaults must be a boolean", {"extends_defaults": "yes"}),
        ("must be a string path or object", {"sources": {"bad": 42}}),
        ("extensions must be a list", {"sources": {"bad": {"path": "notes", "extensions": ".md"}}}),
        (
            "extensions must contain non-empty strings",
            {"sources": {"bad": {"path": "notes", "extensions": [""]}}},
        ),
        ("extensions must not be empty", {"sources": {"bad": {"path": "notes", "extensions": []}}}),
    ]

    for message, config in bad_cases:
        with pytest.raises(ValueError, match=message):
            build_source_config(
                base=tmp_path,
                environ={"REMANENTIA_MEMORY_SOURCES_JSON": json.dumps(config)},
            )


def test_memory_index_builds_from_inline_configured_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryIndex should consume configured sources through its normal build path."""
    docs_dir = tmp_path / "configured_sources"
    docs_dir.mkdir()
    (docs_dir / "decision.md").write_text(
        "Decision: configurable memory source roots are active.\n\n"
        "Finding: MemoryIndex indexed this document through environment-provided "
        "source configuration, not through a hardcoded workspace layout.\n",
        encoding="utf-8",
    )

    inline = json.dumps(
        {
            "extends_defaults": False,
            "sources": {
                "configured": {
                    "path": str(docs_dir),
                    "extensions": [".md"],
                }
            },
        }
    )
    monkeypatch.setenv("REMANENTIA_MEMORY_SOURCES_JSON", inline)

    import memory_index as loaded_memory_index

    memory_index = importlib.reload(loaded_memory_index)
    monkeypatch.setattr("compiled_memory.compile_facts", lambda _repo: [])
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", tmp_path / "hashes.json")

    try:
        idx = memory_index.MemoryIndex()
        stats = idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=False)
    finally:
        monkeypatch.delenv("REMANENTIA_MEMORY_SOURCES_JSON", raising=False)
        importlib.reload(memory_index)

    assert stats["sources"] == {"configured": 1}
    assert [doc.source for doc in idx.documents] == ["configured"]
    assert "configurable memory source roots" in idx.documents[0].paragraphs[0]
