# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the exported claim-axis schema

"""Tests for :mod:`claim_schema`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from claim_axes import FALSIFIED, REFERENCE_VALIDATED
from claim_schema import (
    SCHEMA_PATH,
    build_claim_schema,
    main,
    render_claim_schema,
    schema_is_current,
    write_claim_schema,
)

JsonMap = dict[str, Any]


def test_build_claim_schema_exports_axes_and_refuted_invariant() -> None:
    """The generated schema exposes the exact shared claim-axis contract."""
    schema = cast(JsonMap, build_claim_schema())
    properties = cast(JsonMap, schema["properties"])
    axis_sets = cast(JsonMap, schema["x-remanentia-axis-sets"])
    evidence_kind = cast(JsonMap, properties["evidence_kind"])
    claim_status = cast(JsonMap, properties["claim_status"])
    invariants = cast(list[dict[str, str]], schema["x-remanentia-invariants"])
    render_modes = cast(list[str], schema["x-remanentia-render-modes"])

    assert schema["$id"] == "https://remanentia.com/schemas/claim_axes.schema.json"
    assert evidence_kind["enum"] == sorted(axis_sets["evidence_kind"])
    assert FALSIFIED in evidence_kind["enum"]
    assert REFERENCE_VALIDATED in claim_status["enum"]
    assert {
        "evidence_kind": FALSIFIED,
        "claim_status": REFERENCE_VALIDATED,
        "action": "reject-before-persistence",
    } in invariants
    assert {"validated", "boundary", "refuted"} == set(render_modes)


def test_render_claim_schema_is_deterministic_and_newline_terminated() -> None:
    """The persisted schema text is stable for committed generated artefacts."""
    first = render_claim_schema()
    second = render_claim_schema()

    assert first == second
    assert first.endswith("\n")
    assert json.loads(first) == build_claim_schema()


def test_write_claim_schema_creates_parent_directory(tmp_path: Path) -> None:
    """Schema export writes through the real filesystem path used by operators."""
    output = tmp_path / "nested" / "claim_axes.schema.json"

    written = write_claim_schema(output)

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == build_claim_schema()


def test_module_cli_writes_schema_and_checks_current_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone CLI writes and validates the generated schema artefact."""
    output = tmp_path / "claim_axes.schema.json"

    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_claim_schema()
    assert "Wrote claim-axis schema" in capsys.readouterr().out

    assert main(["--output", str(output), "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_module_cli_detects_stale_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check mode fails closed when the generated schema drifts."""
    output = tmp_path / "claim_axes.schema.json"
    output.write_text("{}\n", encoding="utf-8")

    assert main(["--output", str(output), "--check"]) == 1

    captured = capsys.readouterr()
    assert "not up to date" in captured.err


def test_schema_check_reports_missing_file_as_not_current(tmp_path: Path) -> None:
    """Check mode treats a missing schema file as stale rather than current."""
    assert schema_is_current(tmp_path / "missing.schema.json") is False


def test_top_level_cli_routes_claim_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The main ``remanentia`` CLI exposes the same schema export path."""
    output = tmp_path / "claim_axes.schema.json"
    from cli import main as remanentia_main

    monkeypatch.setattr(sys, "argv", ["remanentia", "claim-schema", "--output", str(output)])

    remanentia_main()

    assert json.loads(output.read_text(encoding="utf-8")) == build_claim_schema()
    assert "Wrote claim-axis schema" in capsys.readouterr().out


def test_committed_claim_schema_is_current() -> None:
    """The committed schema artefact stays synchronised with the code contract."""
    assert SCHEMA_PATH.is_file()
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == build_claim_schema()
