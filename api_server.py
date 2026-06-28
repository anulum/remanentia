# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — HTTP API server

"""Lightweight HTTP API for Remanentia memory system.

Exposes recall, status, consolidation, and health endpoints.
Designed for cross-service integration (SPO bridge, SYNAPSE, etc).

Usage::

    python api_server.py                    # port 8001
    python api_server.py --port 8002        # custom port

Endpoints:
    GET  /health       → {"status": "ok"}
    POST /recall       → {"results": [...]}
    GET  /status       → {"entities": N, "memories": N, ...}
    POST /consolidate  → {"status": "ok", "new_memories": N}
    POST /remember     → {"status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, TypeAlias, cast

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from api_security import (  # noqa: E402 — path inserted above
    DEFAULT_BODY_LIMIT,
    DEFAULT_BURST,
    DEFAULT_RATE_PER_MINUTE,
    BearerAuth,
    RequestAuditLogger,
    TokenBucketLimiter,
    enforce_body_size,
)

PORT = 8001
_PUBLIC_PATHS = frozenset({"/health"})  # never require auth or rate-limit
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class RemanentiaHTTPServer(HTTPServer):
    """HTTP server carrying Remanentia security dependencies for handlers."""

    auth: BearerAuth
    limiter: TokenBucketLimiter
    body_limit: int
    audit_logger: RequestAuditLogger


def _json_default(obj: object) -> JsonValue:
    """Handle NumPy types in JSON serialisation."""
    import numpy as np

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return cast(JsonValue, obj.tolist())
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class RemanentiaHandler(BaseHTTPRequestHandler):
    """HTTP handler for Remanentia API.

    Security layers evaluated in order on every request:

    1. Body size (POST only) — 413 if declared Content-Length exceeds
       ``self.server.body_limit`` bytes.
    2. Rate limit — 429 if :class:`TokenBucketLimiter` refuses the call.
    3. Bearer auth — 401 if ``Authorization: Bearer <token>`` mismatches
       ``self.server.auth``.

    ``/health`` is exempt from all three so external monitors can ping
    a locked-down server.
    """

    # ---- security gates -------------------------------------------------

    def _client_ip(self) -> str:
        """Return the remote client IP used for rate-limit buckets."""

        # http.server uses (host, port); first element is safe for buckets.
        return self.client_address[0] if self.client_address else "unknown"

    @property
    def _remanentia_server(self) -> RemanentiaHTTPServer:
        """Expose the typed server attributes installed by :func:`build_server`."""

        return cast(RemanentiaHTTPServer, self.server)

    def _security_gates(self, method: str) -> bool:
        """Run body/rate/auth checks. Return False if response already sent."""
        if self.path in _PUBLIC_PATHS:
            return True

        server = self._remanentia_server
        limiter = server.limiter
        auth = server.auth

        if method == "POST":
            try:
                declared = int(self.headers.get("Content-Length", 0))
                enforce_body_size(declared, server.body_limit)
            except ValueError as e:
                self._json_response({"error": str(e)}, 413)
                return False

        if not limiter.allow(self._client_ip()):
            self._json_response(
                {"error": "rate limit exceeded"},
                429,
                headers={"Retry-After": limiter.retry_after_seconds()},
            )
            return False

        if not auth.check_header(self.headers.get("Authorization")):
            self._json_response({"error": "authentication required"}, 401)
            return False

        return True

    # ---- request dispatch ----------------------------------------------

    def do_GET(self) -> None:
        """Dispatch supported GET endpoints to their handlers."""

        if not self._security_gates("GET"):
            return
        if self.path == "/health":
            self._json_response({"status": "ok", "timestamp": time.time()})
        elif self.path == "/status":
            self._handle_status()
        else:
            self._json_response({"error": f"Unknown path: {self.path}"}, 404)

    def do_POST(self) -> None:
        """Read and dispatch supported JSON POST endpoints."""

        if not self._security_gates("POST"):
            return
        body = self._read_body()

        if self.path == "/recall":
            self._handle_recall(body)
        elif self.path == "/consolidate":
            self._handle_consolidate(body)
        elif self.path == "/remember":
            self._handle_remember(body)
        else:
            self._json_response({"error": f"Unknown path: {self.path}"}, 404)

    def _handle_status(self) -> None:
        """Return counts for graph entities, relations, memories, and traces."""

        entities = 0
        memories = 0
        traces = 0

        graph_dir = BASE / "memory" / "graph"
        entities_path = graph_dir / "entities.jsonl"
        if entities_path.exists():
            with entities_path.open() as f:
                entities = sum(1 for _ in f)

        semantic_dir = BASE / "memory" / "semantic"
        if semantic_dir.exists():
            memories = sum(1 for _ in semantic_dir.rglob("*.md"))

        traces_dir = BASE / "reasoning_traces"
        if traces_dir.exists():
            traces = sum(1 for _ in traces_dir.glob("*.md"))

        relations = 0
        relations_path = graph_dir / "relations.jsonl"
        if relations_path.exists():
            with relations_path.open() as f:
                relations = sum(1 for _ in f)

        self._json_response(
            {
                "status": "ok",
                "entities": entities,
                "relations": relations,
                "memories": memories,
                "traces": traces,
            }
        )

    def _handle_recall(self, body: Mapping[str, JsonValue]) -> None:
        """Run memory recall for a parsed JSON request body."""

        query = body.get("query", "")
        top_k = body.get("top_k", 5)

        if not isinstance(query, str) or not query:
            self._json_response({"error": "query required"}, 400)
            return

        if not isinstance(top_k, int):
            self._json_response({"error": "top_k must be an integer"}, 400)
            return

        if top_k == 0:
            self._json_response({"results": [], "query": query})
            return

        try:
            from memory_recall import recall

            ctx = recall(query, top_k=top_k)

            results: list[JsonValue] = []
            if ctx.trace:
                trace_result: JsonObject = {
                    "name": str(ctx.trace),
                    "score": float(ctx.trace_score),
                    "snippet": str(ctx.trace_snippet)[:300],
                    "type": "trace",
                }
                results.append(trace_result)
            for sm in ctx.semantic_memories[:top_k]:
                semantic_result: JsonObject = {
                    "name": str(sm.get("file", "")),
                    "score": float(sm.get("score", 0.0)),
                    "snippet": str(sm.get("content", ""))[:300],
                    "type": "semantic",
                }
                results.append(semantic_result)

            self._json_response(
                {
                    "results": results,
                    "query": query,
                    "entities": ctx.entities[:10],
                    "novelty": ctx.novelty_score if hasattr(ctx, "novelty_score") else 0.5,
                }
            )
        except Exception as e:
            self._json_response({"error": str(e), "results": []}, 500)

    def _handle_consolidate(self, body: Mapping[str, JsonValue]) -> None:
        """Run consolidation for a parsed JSON request body."""

        force = body.get("force", False)
        if not isinstance(force, bool):
            self._json_response({"error": "force must be a boolean"}, 400)
            return

        try:
            from consolidation_engine import consolidate

            result = consolidate(force=force)
            self._json_response(
                {
                    "status": "ok",
                    "new_memories": result.get("new_memories", 0)
                    if isinstance(result, dict)
                    else 0,
                }
            )
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_remember(self, body: Mapping[str, JsonValue]) -> None:
        """Store a knowledge note from a parsed JSON request body."""

        content = body.get("content", "")
        trigger = body.get("trigger", "")

        if not isinstance(content, str) or not content:
            self._json_response({"error": "content required"}, 400)
            return
        if trigger is not None and not isinstance(trigger, str):
            self._json_response({"error": "trigger must be a string"}, 400)
            return

        try:
            from knowledge_store import KnowledgeStore

            ks = KnowledgeStore()
            ks.add_note(content=content, keywords=[trigger] if trigger else None)
            self._json_response({"status": "ok", "stored": content[:80]})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _read_body(self) -> JsonObject:
        """Read a JSON request body and return an object mapping."""

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return cast(JsonObject, payload)

    def _json_response(
        self,
        data: Mapping[str, JsonValue],
        status: int = 200,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Write a JSON response with audit metadata and CORS headers."""

        body = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
        self._audit_response(status)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _audit_response(self, status: int) -> None:
        """Record metadata-only audit rows for private endpoints."""

        if self.path in _PUBLIC_PATHS:
            return
        server = self._remanentia_server
        logger = server.audit_logger
        auth = server.auth
        logger.record(
            server="stdlib",
            method=self.command,
            path=self.path,
            client=self._client_ip(),
            status=status,
            outcome="ok" if status < 400 else "error",
            auth_enabled=auth.enabled,
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress non-404 access logs from the stdlib request handler."""

        # Quiet logging — only errors
        if args and "404" in str(args[0]):
            super().log_message(fmt, *args)


def build_server(
    host: str,
    port: int,
    *,
    auth: BearerAuth | None = None,
    limiter: TokenBucketLimiter | None = None,
    body_limit: int = DEFAULT_BODY_LIMIT,
) -> RemanentiaHTTPServer:
    """Construct an HTTPServer with security attributes attached.

    Factored out of :func:`main` so tests can drive the handler without
    starting a blocking process.
    """
    server = RemanentiaHTTPServer((host, port), RemanentiaHandler)
    server.auth = auth if auth is not None else BearerAuth.from_env()
    rate_per_minute = float(os.environ.get("REMANENTIA_API_RATE", DEFAULT_RATE_PER_MINUTE))
    server.limiter = (
        limiter
        if limiter is not None
        else TokenBucketLimiter(
            rate_per_minute=rate_per_minute,
            burst=int(os.environ.get("REMANENTIA_API_BURST", DEFAULT_BURST)),
        )
    )
    server.body_limit = body_limit
    server.audit_logger = RequestAuditLogger.from_env(
        BASE / ".coordination" / "runtime" / "api_server_audit.jsonl"
    )
    return server


def main() -> None:  # pragma: no cover — blocking server entry point
    """Parse CLI arguments and start the blocking stdlib HTTP server."""

    p = argparse.ArgumentParser(description="Remanentia HTTP API server")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--token-file",
        help="path to a file containing the bearer token (overrides REMANENTIA_API_TOKEN)",
    )
    p.add_argument(
        "--require-auth",
        action="store_true",
        help="refuse to start if no token is configured",
    )
    p.add_argument(
        "--body-limit",
        type=int,
        default=int(os.environ.get("REMANENTIA_API_MAX_BODY", DEFAULT_BODY_LIMIT)),
        help="max POST body bytes (default 1 MiB)",
    )
    args = p.parse_args()

    auth = BearerAuth.from_file(args.token_file) if args.token_file else BearerAuth.from_env()
    if args.require_auth and not auth.enabled:
        print(
            "[SECURITY] --require-auth specified but no token configured; refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    server = build_server(args.host, args.port, auth=auth, body_limit=args.body_limit)
    print(f"Remanentia API running on http://{args.host}:{args.port}")
    print(f"  auth: {'ENABLED (Bearer)' if auth.enabled else 'DISABLED (dev only)'}")
    print(f"  body limit: {args.body_limit} B")
    print("  GET  /health")
    print('  POST /recall         {"query": "...", "top_k": 5}')
    print("  GET  /status")
    print('  POST /consolidate    {"force": false}')
    print('  POST /remember       {"content": "...", "trigger": "..."}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
