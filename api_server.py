# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — HTTP API Server

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
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

PORT = 8001


def _json_default(obj):
    """Handle NumPy types in JSON serialisation."""
    import numpy as np
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class RemanentiaHandler(BaseHTTPRequestHandler):
    """HTTP handler for Remanentia API."""

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok", "timestamp": time.time()})
        elif self.path == "/status":
            self._handle_status()
        else:
            self._json_response({"error": f"Unknown path: {self.path}"}, 404)

    def do_POST(self):
        body = self._read_body()

        if self.path == "/recall":
            self._handle_recall(body)
        elif self.path == "/consolidate":
            self._handle_consolidate(body)
        elif self.path == "/remember":
            self._handle_remember(body)
        else:
            self._json_response({"error": f"Unknown path: {self.path}"}, 404)

    def _handle_status(self):
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

        self._json_response({
            "status": "ok",
            "entities": entities,
            "relations": relations,
            "memories": memories,
            "traces": traces,
        })

    def _handle_recall(self, body: dict):
        query = body.get("query", "")
        top_k = body.get("top_k", 5)

        if not query:
            self._json_response({"error": "query required"}, 400)
            return

        if top_k == 0:
            self._json_response({"results": [], "query": query})
            return

        try:
            from memory_recall import recall
            ctx = recall(query, top_k=top_k)

            results = []
            if ctx.trace:
                results.append({
                    "name": ctx.trace,
                    "score": float(ctx.trace_score),
                    "snippet": ctx.trace_snippet[:300],
                    "type": "trace",
                })
            for sm in ctx.semantic_memories[:top_k]:
                results.append({
                    "name": sm.get("file", ""),
                    "score": float(sm.get("score", 0.0)),
                    "snippet": str(sm.get("content", ""))[:300],
                    "type": "semantic",
                })

            self._json_response({
                "results": results,
                "query": query,
                "entities": ctx.entities[:10],
                "novelty": ctx.novelty_score if hasattr(ctx, "novelty_score") else 0.5,
            })
        except Exception as e:
            self._json_response({"error": str(e), "results": []}, 500)

    def _handle_consolidate(self, body: dict):
        force = body.get("force", False)
        try:
            from consolidation_engine import consolidate
            result = consolidate(force=force)
            self._json_response({
                "status": "ok",
                "new_memories": result.get("new_memories", 0) if isinstance(result, dict) else 0,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_remember(self, body: dict):
        content = body.get("content", "")
        trigger = body.get("trigger", "")

        if not content:
            self._json_response({"error": "content required"}, 400)
            return

        try:
            from knowledge_store import KnowledgeStore
            ks = KnowledgeStore()
            ks.add_note(content=content, tags=[trigger] if trigger else [])
            self._json_response({"status": "ok", "stored": content[:80]})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quiet logging — only errors
        if args and "404" in str(args[0]):
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser(description="Remanentia HTTP API server")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    server = HTTPServer((args.host, args.port), RemanentiaHandler)
    print(f"Remanentia API running on http://{args.host}:{args.port}")
    print(f"  GET  /health")
    print(f"  POST /recall         {{\"query\": \"...\", \"top_k\": 5}}")
    print(f"  GET  /status")
    print(f"  POST /consolidate    {{\"force\": false}}")
    print(f"  POST /remember       {{\"content\": \"...\", \"trigger\": \"...\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
