# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — API documentation wiring tests

"""Tests for the packaged-module API documentation wiring gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_api_docs_wiring.py"


def _load_module() -> Any:
    """Load the API documentation wiring checker as a module."""

    spec = importlib.util.spec_from_file_location("check_api_docs_wiring", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_packaged_modules_have_matching_api_pages() -> None:
    """The live packaged Python modules should match the API reference pages."""
    checker = _load_module()

    result = checker.audit_api_docs(ROOT)

    assert result.ok, result.format()


def test_checker_reports_missing_pages_and_orphan_pages(tmp_path: Path) -> None:
    """The checker should reject both undocumented packages and stale API pages."""
    checker = _load_module()
    docs_api = tmp_path / "docs" / "api"
    docs_api.mkdir(parents=True)
    (tmp_path / "mkdocs.yml").write_text(
        "nav:\n  - API Reference:\n    - documented: api/documented.md\n"
        "    - orphan: api/orphan.md\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["documented", "undocumented"]\n',
        encoding="utf-8",
    )
    (docs_api / "documented.md").write_text("# documented\n\n::: documented\n", encoding="utf-8")
    (docs_api / "orphan.md").write_text("# orphan\n\n::: orphan\n", encoding="utf-8")

    result = checker.audit_api_docs(tmp_path)

    assert not result.ok
    formatted = result.format()
    assert "missing API pages: undocumented" in formatted
    assert "API pages without packaged modules: orphan" in formatted


def test_checker_reports_api_pages_missing_from_mkdocs_nav(tmp_path: Path) -> None:
    """The checker should reject API pages that are not published in MkDocs nav."""
    checker = _load_module()
    docs_api = tmp_path / "docs" / "api"
    docs_api.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["published"]\n',
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text("nav:\n  - Home: index.md\n", encoding="utf-8")
    (docs_api / "published.md").write_text("# published\n\n::: published\n", encoding="utf-8")

    result = checker.audit_api_docs(tmp_path)

    assert not result.ok
    assert "API pages missing from MkDocs nav: published" in result.format()


def test_checker_reports_nav_entries_without_api_pages(tmp_path: Path) -> None:
    """The checker should reject MkDocs API nav entries with no backing page."""
    checker = _load_module()
    docs_api = tmp_path / "docs" / "api"
    docs_api.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["published"]\n',
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "nav:\n  - API Reference:\n    - ghost: api/ghost.md\n",
        encoding="utf-8",
    )
    (docs_api / "published.md").write_text("# published\n\n::: published\n", encoding="utf-8")

    result = checker.audit_api_docs(tmp_path)

    assert not result.ok
    assert "MkDocs API nav entries without pages: ghost" in result.format()


def test_checker_handles_missing_docs_and_mkdocs_files(tmp_path: Path) -> None:
    """The checker should fail closed when docs/api or MkDocs nav are absent."""
    checker = _load_module()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["runtime_module"]\n',
        encoding="utf-8",
    )

    result = checker.audit_api_docs(tmp_path)

    assert not result.ok
    assert "missing API pages: runtime_module" in result.format()
    assert result.api_pages == ()
    assert result.mkdocs_pages == ()


def test_main_returns_zero_for_complete_wiring(tmp_path: Path, capsys: Any) -> None:
    """The CLI should return zero and print a pass line for complete wiring."""
    checker = _load_module()
    docs_api = tmp_path / "docs" / "api"
    docs_api.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["published"]\n',
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "nav:\n  - API Reference:\n    - published: api/published.md\n",
        encoding="utf-8",
    )
    (docs_api / "published.md").write_text("# published\n\n::: published\n", encoding="utf-8")

    code = checker.main(["--root", str(tmp_path)])

    assert code == 0
    assert "PASS: 1 packaged modules match 1 API pages" in capsys.readouterr().out


def test_main_returns_nonzero_for_broken_wiring(tmp_path: Path, capsys: Any) -> None:
    """The CLI should fail closed when the package/docs surface diverges."""
    checker = _load_module()
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "mkdocs.yml").write_text("nav:\n  - Home: index.md\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["runtime_module"]\n',
        encoding="utf-8",
    )

    code = checker.main(["--root", str(tmp_path)])

    assert code == 1
    assert "missing API pages: runtime_module" in capsys.readouterr().out
