# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP tool-call rate limiting tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import mcp_server
from api_security import ToolAuditLogger

JsonObject = dict[str, Any]


def _reset_mcp_security_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset MCP security singletons and env knobs between JSON-RPC tests."""
    monkeypatch.setattr(mcp_server, "MCP_AUDIT_LOGGER", ToolAuditLogger(None))
    monkeypatch.setattr(mcp_server, "_MCP_RATE_LIMITER", None)
    monkeypatch.setattr(mcp_server, "_MCP_RATE_LIMIT_CONFIG", None)
    for name in (
        "REMANENTIA_MCP_RATE_LIMIT",
        "REMANENTIA_MCP_RATE",
        "REMANENTIA_MCP_BURST",
        "REMANENTIA_MCP_CLIENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def _status_call(request_id: str, *, arguments: JsonObject | None = None) -> JsonObject:
    """Build a real MCP status tool-call request."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": "remanentia_status",
            "arguments": {} if arguments is None else arguments,
        },
    }


def _expect_object(response: JsonObject | None) -> JsonObject:
    """Return a JSON-RPC response object, failing fast if the server returned None."""
    assert response is not None
    return response


def _error(response: JsonObject) -> JsonObject:
    """Return the JSON-RPC error payload from a response."""
    error = response.get("error")
    assert isinstance(error, dict)
    return error


def _content_text(response: JsonObject) -> str:
    """Return the first MCP text content item."""
    result = response.get("result")
    assert isinstance(result, dict)
    content = result.get("content")
    assert isinstance(content, list)
    item = content[0]
    assert isinstance(item, dict)
    text = item.get("text")
    assert isinstance(text, str)
    return text


def test_mcp_tools_call_is_rate_limited_after_configured_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Throttle `tools/call` through the real JSON-RPC dispatch path."""
    _reset_mcp_security_state(monkeypatch)
    monkeypatch.setenv("REMANENTIA_MCP_RATE", "60")
    monkeypatch.setenv("REMANENTIA_MCP_BURST", "1")
    monkeypatch.setenv("REMANENTIA_MCP_CLIENT_ID", "agent-a")

    first = _expect_object(mcp_server.handle_request(_status_call("first")))
    second = _expect_object(mcp_server.handle_request(_status_call("second")))

    assert "result" in first
    assert _error(second) == {
        "code": mcp_server.MCP_RATE_LIMIT_ERROR_CODE,
        "message": "MCP tool rate limit exceeded",
        "data": {"retry_after_seconds": "1"},
    }


def test_mcp_rate_limit_opt_out_preserves_stdio_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow operators to disable MCP stdio throttling for trusted local sessions."""
    _reset_mcp_security_state(monkeypatch)
    monkeypatch.setenv("REMANENTIA_MCP_RATE_LIMIT", "off")
    monkeypatch.setenv("REMANENTIA_MCP_RATE", "60")
    monkeypatch.setenv("REMANENTIA_MCP_BURST", "1")

    first = _expect_object(mcp_server.handle_request(_status_call("first")))
    second = _expect_object(mcp_server.handle_request(_status_call("second")))

    assert "result" in first
    assert "result" in second
    assert "error" not in second


def test_mcp_rate_limited_calls_are_audited_without_argument_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record throttled calls as metadata-only audit events."""
    _reset_mcp_security_state(monkeypatch)
    audit_path = tmp_path / "mcp_tool_audit.jsonl"
    monkeypatch.setattr(mcp_server, "MCP_AUDIT_LOGGER", ToolAuditLogger(audit_path))
    monkeypatch.setenv("REMANENTIA_MCP_RATE", "60")
    monkeypatch.setenv("REMANENTIA_MCP_BURST", "1")

    request = _status_call("second", arguments={"query": "private recall text", "top_k": 3})
    _expect_object(mcp_server.handle_request(_status_call("first")))
    _expect_object(mcp_server.handle_request(request))

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    throttled = records[-1]
    assert throttled["server"] == "mcp"
    assert throttled["method"] == "tools/call"
    assert throttled["tool"] == "remanentia_status"
    assert throttled["request_id"] == "second"
    assert throttled["argument_keys"] == ["query", "top_k"]
    assert throttled["outcome"] == "rate_limited"
    assert "private recall text" not in json.dumps(throttled)


def test_mcp_rate_limit_rejects_non_positive_operator_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when MCP rate-limit environment values are non-positive."""
    _reset_mcp_security_state(monkeypatch)
    monkeypatch.setenv("REMANENTIA_MCP_RATE", "0")
    rate_response = _expect_object(mcp_server.handle_request(_status_call("bad-rate")))

    _reset_mcp_security_state(monkeypatch)
    monkeypatch.setenv("REMANENTIA_MCP_RATE", "60")
    monkeypatch.setenv("REMANENTIA_MCP_BURST", "0")
    burst_response = _expect_object(mcp_server.handle_request(_status_call("bad-burst")))

    assert _error(rate_response) == {"code": -32000, "message": "Tool call failed"}
    assert _error(burst_response) == {"code": -32000, "message": "Tool call failed"}


def test_mcp_tools_call_normalises_non_object_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle malformed `tools/call` params as an unknown tool, not a crash."""
    _reset_mcp_security_state(monkeypatch)
    response = _expect_object(
        mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "bad-params",
                "method": "tools/call",
                "params": "not an object",
            }
        )
    )

    assert _content_text(response) == "Unknown tool: "
