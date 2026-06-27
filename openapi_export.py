# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — OpenAPI schema export

"""Deterministic OpenAPI export for the Remanentia REST API."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import cast

HTTP_METHODS = frozenset({"get", "put", "post", "delete", "patch", "options", "head", "trace"})
SECURITY_SCHEME_NAME = "BearerAuth"
DEFAULT_SCHEMA_PATH = Path("docs/openapi/remanentia_openapi.json")

JsonObject = dict[str, object]


def build_openapi_schema() -> JsonObject:
    """Build the public OpenAPI schema from the live FastAPI application.

    The runtime API uses bearer-token middleware with two intentionally public
    endpoints. FastAPI cannot infer that custom middleware boundary by itself,
    so the exporter annotates every generated operation with the same private
    versus public path policy that the application enforces at request time.
    """
    from api import _AUTH_EXEMPT_PATHS, app

    schema = cast(JsonObject, copy.deepcopy(app.openapi()))
    components = _ensure_object(schema, "components")
    security_schemes = _ensure_object(components, "securitySchemes")
    security_schemes[SECURITY_SCHEME_NAME] = {"type": "http", "scheme": "bearer"}

    public_security: list[object] = []
    private_security: list[JsonObject] = [{SECURITY_SCHEME_NAME: []}]
    paths = _as_object(schema.get("paths", {}), "paths")
    for path, path_item_value in paths.items():
        path_item = _as_optional_object(path_item_value)
        if path_item is None:
            continue
        security = public_security if path in _AUTH_EXEMPT_PATHS else private_security
        for method, operation_value in path_item.items():
            if method not in HTTP_METHODS:
                continue
            operation = _as_optional_object(operation_value)
            if operation is not None:
                operation["security"] = security

    schema["x-remanentia-auth-exempt-paths"] = sorted(_AUTH_EXEMPT_PATHS)
    schema["x-remanentia-export"] = {
        "private_path_policy": "all non-exempt operations require BearerAuth",
        "security_scheme": SECURITY_SCHEME_NAME,
    }
    return schema


def write_openapi_schema(path: Path = DEFAULT_SCHEMA_PATH) -> JsonObject:
    """Write the deterministic OpenAPI schema JSON to ``path`` and return it."""
    schema = build_openapi_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema


def main(argv: list[str] | None = None) -> int:
    """Run the standalone OpenAPI export command."""
    parser = argparse.ArgumentParser(description="Export the Remanentia OpenAPI schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Destination JSON path",
    )
    args = parser.parse_args(argv)
    output = cast(Path, args.output)
    write_openapi_schema(output)
    print(f"Wrote OpenAPI schema: {output}")
    return 0


def _ensure_object(parent: JsonObject, key: str) -> JsonObject:
    """Return a mutable object child, creating it when missing."""
    value = parent.get(key)
    if value is None:
        child: JsonObject = {}
        parent[key] = child
        return child
    return _as_object(value, key)


def _as_object(value: object, label: str) -> JsonObject:
    """Return ``value`` as a JSON object or raise for malformed schemas."""
    if not isinstance(value, dict):
        raise TypeError(f"OpenAPI field must be an object: {label}")
    return cast(JsonObject, value)


def _as_optional_object(value: object) -> JsonObject | None:
    """Return ``value`` as a JSON object when it has object shape."""
    if not isinstance(value, dict):
        return None
    return cast(JsonObject, value)


if __name__ == "__main__":
    raise SystemExit(main())
