# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — OpenAPI export tests

"""Real-surface tests for the REST API OpenAPI export contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openapi_export import build_openapi_schema, main, write_openapi_schema


def _operation(schema: dict[str, object], path: str, method: str) -> dict[str, object]:
    """Return one operation object from an OpenAPI schema."""
    paths = schema["paths"]
    assert isinstance(paths, dict)
    path_item = paths[path]
    assert isinstance(path_item, dict)
    operation = path_item[method]
    assert isinstance(operation, dict)
    return operation


def test_build_openapi_schema_marks_private_endpoints_with_bearer_auth() -> None:
    """Private endpoints carry the same bearer-auth boundary as runtime middleware."""
    schema = build_openapi_schema()

    openapi_version = schema["openapi"]
    assert isinstance(openapi_version, str)
    assert openapi_version.startswith("3.")
    info = schema["info"]
    assert isinstance(info, dict)
    assert info["title"] == "Remanentia"

    components = schema["components"]
    assert isinstance(components, dict)
    security_schemes = components["securitySchemes"]
    assert isinstance(security_schemes, dict)
    assert security_schemes["BearerAuth"] == {"type": "http", "scheme": "bearer"}

    assert _operation(schema, "/health", "get").get("security") == []
    assert _operation(schema, "/vector/search/public", "post").get("security") == []
    assert _operation(schema, "/recall", "post")["security"] == [{"BearerAuth": []}]
    assert _operation(schema, "/status", "get")["security"] == [{"BearerAuth": []}]


def test_write_openapi_schema_creates_stable_json(tmp_path: Path) -> None:
    """The writer creates parent directories and persists deterministic JSON."""
    destination = tmp_path / "schema" / "remanentia_openapi.json"

    schema = write_openapi_schema(destination)

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted == schema
    text = destination.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"BearerAuth"' in text


def test_build_openapi_schema_tolerates_non_operation_path_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path-level OpenAPI extensions and malformed members do not block export."""
    from api import app

    def fake_openapi() -> dict[str, object]:
        return {
            "openapi": "3.1.0",
            "info": {"title": "Remanentia"},
            "paths": {
                "/health": {"parameters": [], "get": {}},
                "/extension": "not-a-path-item",
                "/mixed": {"post": "not-an-operation"},
            },
        }

    monkeypatch.setattr(app, "openapi", fake_openapi)

    schema = build_openapi_schema()

    assert _operation(schema, "/health", "get")["security"] == []
    assert schema["x-remanentia-export"] == {
        "private_path_policy": "all non-exempt operations require BearerAuth",
        "security_scheme": "BearerAuth",
    }


def test_build_openapi_schema_rejects_malformed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-object `paths` field is rejected as a broken OpenAPI schema."""
    from api import app

    def fake_openapi() -> dict[str, object]:
        return {"openapi": "3.1.0", "info": {"title": "Remanentia"}, "paths": []}

    monkeypatch.setattr(app, "openapi", fake_openapi)

    with pytest.raises(TypeError, match="paths"):
        build_openapi_schema()


def test_main_writes_openapi_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone module entrypoint writes the requested schema path."""
    destination = tmp_path / "standalone_openapi.json"

    assert main(["--output", str(destination)]) == 0

    assert json.loads(destination.read_text(encoding="utf-8"))["info"]["title"] == "Remanentia"
    captured = capsys.readouterr()
    assert str(destination) in captured.out
