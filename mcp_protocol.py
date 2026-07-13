# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP JSON-RPC protocol and tool dispatch

"""Transport-independent MCP schemas and one-request JSON-RPC dispatch."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

log = logging.getLogger(__name__)

MCP_RATE_LIMIT_ERROR_CODE = -32029


class RateLimiter(Protocol):
    """Token-bucket surface required by the MCP dispatcher."""

    def allow(self, key: str) -> bool:
        """Return whether one request may proceed."""

    def retry_after_seconds(self) -> str:
        """Return the retry delay for a rejected request."""


class AuditLogger(Protocol):
    """Metadata-only audit surface required by the MCP dispatcher."""

    def record(self, **fields: object) -> None:
        """Persist one tool-call metadata record."""


ToolHandler = Callable[[dict[str, Any]], str]


TOOLS: list[dict[str, Any]] = [
    {
        "name": "remanentia_recall",
        "description": "Deep memory recall across traces, knowledge, graph, and time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall"},
                "top_k": {"type": "integer", "default": 3},
                "project": {"type": "string", "default": ""},
                "after": {"type": "string", "default": ""},
                "before": {"type": "string", "default": ""},
                "llm": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remanentia_remember",
        "description": "Persist a memory and optional prospective trigger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "type": {"type": "string", "default": "context"},
                "project": {"type": "string", "default": ""},
                "trigger": {"type": "string", "default": ""},
            },
            "required": ["content"],
        },
    },
    {
        "name": "remanentia_status",
        "description": "Check Remanentia system status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "remanentia_graph",
        "description": "Query the entity relationship graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "default": ""},
                "top": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "remanentia_recall_feedback",
        "description": "Report whether a prior recall was used.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "was_used": {"type": "boolean"},
            },
            "required": ["query", "was_used"],
        },
    },
    {
        "name": "remanentia_recall_correctness",
        "description": "Report a verifier verdict for a prior recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "was_correct": {"type": "boolean"},
            },
            "required": ["query", "was_correct"],
        },
    },
]


def _rate_limit_response(rid: object, retry_after: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {
            "code": MCP_RATE_LIMIT_ERROR_CODE,
            "message": "MCP tool rate limit exceeded",
            "data": {"retry_after_seconds": retry_after},
        },
    }


def dispatch_request(
    request: dict[str, Any],
    *,
    handlers: Mapping[str, ToolHandler],
    audit_logger: AuditLogger,
    limiter_factory: Callable[[], RateLimiter | None],
    rate_key: Callable[[], str],
) -> dict[str, Any] | None:
    """Dispatch one MCP JSON-RPC request through supplied production handlers."""
    method = request.get("method", "")
    rid = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remanentia", "version": "0.5.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "notifications/initialized":
        return None
    if method != "tools/call":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    started = time.monotonic()
    params = request.get("params", {})
    if not isinstance(params, dict):
        params = {}
    tool_name = params.get("name", "")
    args = params.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    outcome = "ok"
    error_type = ""

    try:
        limiter = limiter_factory()
        if limiter is not None and not limiter.allow(rate_key()):
            outcome = "rate_limited"
            return _rate_limit_response(rid, limiter.retry_after_seconds())

        handler = handlers.get(str(tool_name))
        if handler is None:
            outcome = "unknown_tool"
            text = f"Unknown tool: {tool_name}"
        else:
            text = handler(args)
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"content": [{"type": "text", "text": text}]},
        }
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        log.exception("MCP tool call failed: %s", tool_name)
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32000, "message": "Tool call failed"},
        }
    finally:
        audit_logger.record(
            server="mcp",
            method="tools/call",
            tool=tool_name,
            request_id=str(rid),
            argument_keys=list(args),
            outcome=outcome,
            duration_ms=(time.monotonic() - started) * 1000.0,
            error_type=error_type,
        )
