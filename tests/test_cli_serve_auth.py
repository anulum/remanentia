# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CLI REST authentication tests

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cli import cmd_serve, main


def _serve_args(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token_file: Path | None = None,
    require_auth: bool = False,
) -> argparse.Namespace:
    """Build a `remanentia serve` namespace for direct command tests."""
    return argparse.Namespace(
        host=host,
        port=port,
        token_file=None if token_file is None else str(token_file),
        require_auth=require_auth,
    )


def test_cmd_serve_loads_token_file_before_importing_fastapi_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load a local bearer token file before `uvicorn.run("api:app", ...)`."""
    monkeypatch.delenv("REMANENTIA_API_TOKEN", raising=False)
    token_file = tmp_path / "api.token"
    token_file.write_text("file-token\n", encoding="utf-8")

    with patch("uvicorn.run") as run:
        cmd_serve(_serve_args(token_file=token_file, require_auth=True))

    assert os.environ["REMANENTIA_API_TOKEN"] == "file-token"
    run.assert_called_once_with("api:app", host="127.0.0.1", port=8765)


def test_cmd_serve_require_auth_accepts_existing_environment_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow fail-closed production startup when the token is already in env."""
    monkeypatch.setenv("REMANENTIA_API_TOKEN", "env-token")

    with patch("uvicorn.run") as run:
        cmd_serve(_serve_args(require_auth=True))

    run.assert_called_once_with("api:app", host="127.0.0.1", port=8765)


def test_cmd_serve_require_auth_refuses_open_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse production startup when no bearer token is configured."""
    monkeypatch.delenv("REMANENTIA_API_TOKEN", raising=False)

    with patch("uvicorn.run") as run:
        with pytest.raises(SystemExit, match="--require-auth"):
            cmd_serve(_serve_args(require_auth=True))

    run.assert_not_called()


def test_cmd_serve_rejects_empty_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail closed when the configured token file contains no token."""
    monkeypatch.delenv("REMANENTIA_API_TOKEN", raising=False)
    token_file = tmp_path / "empty.token"
    token_file.write_text("\n", encoding="utf-8")

    with patch("uvicorn.run") as run:
        with pytest.raises(SystemExit, match="API token file is empty"):
            cmd_serve(_serve_args(token_file=token_file, require_auth=True))

    run.assert_not_called()
    assert "REMANENTIA_API_TOKEN" not in os.environ


def test_main_parses_serve_auth_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the real argparse path for `remanentia serve` auth flags."""
    monkeypatch.delenv("REMANENTIA_API_TOKEN", raising=False)
    token_file = tmp_path / "api.token"
    token_file.write_text("parsed-token\n", encoding="utf-8")

    with (
        patch(
            "sys.argv",
            [
                "remanentia",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8766",
                "--token-file",
                str(token_file),
                "--require-auth",
            ],
        ),
        patch("uvicorn.run") as run,
    ):
        main()

    assert os.environ["REMANENTIA_API_TOKEN"] == "parsed-token"
    run.assert_called_once_with("api:app", host="127.0.0.1", port=8766)
