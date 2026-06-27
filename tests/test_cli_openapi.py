# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CLI OpenAPI export tests

"""Tests for the `remanentia openapi` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cli import main


def test_main_parses_openapi_export_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real argparse path writes an OpenAPI schema file."""
    output = tmp_path / "openapi.json"

    with patch("sys.argv", ["remanentia", "openapi", "--output", str(output)]):
        main()

    schema = json.loads(output.read_text(encoding="utf-8"))
    assert schema["info"]["title"] == "Remanentia"
    assert "BearerAuth" in schema["components"]["securitySchemes"]
    assert str(output) in capsys.readouterr().out
