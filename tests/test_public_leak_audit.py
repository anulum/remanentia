# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Public leak audit tests

"""Tests for the public release leak audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "public_leak_audit.py"


def _load_module() -> Any:
    """Load the release-audit script as a module for direct API tests."""

    spec = importlib.util.spec_from_file_location("public_leak_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_tracked_public_surface_has_no_internal_leaks() -> None:
    """The live tracked public surface should pass the release leak audit."""
    audit = _load_module()

    result = audit.audit_public_surface(ROOT)

    assert result.ok, result.format()


def test_audit_tool_and_tests_are_release_clean() -> None:
    """New audit files should be clean before they become tracked files."""
    audit = _load_module()

    result = audit.audit_paths([SCRIPT, Path(__file__)], root=ROOT)

    assert result.ok, result.format()


def test_reports_private_workspace_and_agent_identity_leaks(tmp_path: Path) -> None:
    """Synthetic public files with private paths or agent labels should fail."""
    audit = _load_module()
    public_file = tmp_path / "README.md"
    public_file.write_text(
        "Load from /media/anulum/GOTM/private and ask Claude to continue.\n",  # public-leak-audit: allow
        encoding="utf-8",
    )

    result = audit.audit_paths([public_file], root=tmp_path)

    assert not result.ok
    formatted = result.format()
    assert "private workspace path" in formatted
    assert "agent identity label" in formatted
    assert "README.md:1" in formatted


def test_skips_internal_docs_and_binary_files(tmp_path: Path) -> None:
    """Internal docs and binary-like files should not produce public findings."""
    audit = _load_module()
    internal = tmp_path / "docs" / "internal" / "handover.md"
    internal.parent.mkdir(parents=True)
    internal.write_text(  # public-leak-audit: allow
        "Claude used /media/anulum/GOTM in this private note.\n",  # public-leak-audit: allow
        encoding="utf-8",
    )
    binary = tmp_path / "docs" / "assets" / "binary.md"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00GOTM")

    result = audit.audit_paths([internal, binary], root=tmp_path)

    assert result.ok, result.format()


def test_skips_unreadable_oversized_invalid_and_cache_files(tmp_path: Path) -> None:
    """Non-public and non-text candidates should not enter the scan count."""
    audit = _load_module()
    missing = tmp_path / "missing.md"
    oversized = tmp_path / "big.md"
    oversized.write_bytes(b"x" * (audit.MAX_TEXT_BYTES + 1))
    invalid = tmp_path / "invalid.md"
    invalid.write_bytes(b"\xff\xfe")
    cache = tmp_path / ".venv" / "note.md"
    cache.parent.mkdir()
    cache.write_text("GOTM\n", encoding="utf-8")  # public-leak-audit: allow

    result = audit.audit_paths([missing, oversized, invalid, cache], root=tmp_path)

    assert result.ok, result.format()
    assert result.scanned == 0


def test_external_absolute_path_is_reported_as_absolute(tmp_path: Path) -> None:
    """Findings outside the requested root should keep an absolute display path."""
    audit = _load_module()
    outside = tmp_path / "outside.md"
    outside.write_text("agentic-shared\n", encoding="utf-8")  # public-leak-audit: allow

    result = audit.audit_paths([outside], root=tmp_path / "repo")

    assert not result.ok
    assert str(outside) in result.format(tmp_path / "repo")


def test_main_returns_nonzero_for_explicit_leak_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should return 1 and print findings for explicit leak paths."""
    audit = _load_module()
    leaked = tmp_path / "module.py"
    leaked.write_text(  # public-leak-audit: allow
        'REFERENCE = "04_ARCANE_SAPIENCE"\n',  # public-leak-audit: allow
        encoding="utf-8",
    )

    code = audit.main(["--root", str(tmp_path), str(leaked)])

    assert code == 1
    assert "private workspace label" in capsys.readouterr().out


def test_main_returns_zero_for_clean_explicit_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should return 0 and print a pass line for clean explicit paths."""
    audit = _load_module()
    clean = tmp_path / "module.py"
    clean.write_text('REFERENCE = "public corpus"\n', encoding="utf-8")

    code = audit.main(["--root", str(tmp_path), str(clean)])

    assert code == 0
    assert "PASS:" in capsys.readouterr().out
