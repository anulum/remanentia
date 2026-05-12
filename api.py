# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — API module

"""FastAPI server for Remanentia memory system.

Usage::
    cd workspace-internal
    uvicorn api:app --port 8001
    # or
    python api.py

Endpoints:
    GET  /health
    POST /recall          {"query": "...", "top_k": 3, "format": "summary"}
    POST /consolidate     {"force": false}
    GET  /status
    GET  /entities
    GET  /graph?top=15
    GET  /graph/entity/{id}
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api_security import (
    DEFAULT_BODY_LIMIT,
    DEFAULT_BURST,
    DEFAULT_RATE_PER_MINUTE,
    BearerAuth,
    TokenBucketLimiter,
    enforce_body_size,
)

app = FastAPI(
    title="Remanentia",
    description="Persistent AI memory with SNN-orchestrated consolidation",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
STATE_DIR = BASE / "snn_state"
GRAPH_DIR = BASE / "memory" / "graph"
VECTOR_WORKER_MAX_AGE_S = 1800
AUTH = BearerAuth.from_env()
BODY_LIMIT = int(os.environ.get("REMANENTIA_API_BODY_LIMIT_BYTES", str(DEFAULT_BODY_LIMIT)))
LIMITER = TokenBucketLimiter(
    rate_per_minute=float(
        os.environ.get("REMANENTIA_API_RATE_PER_MINUTE", str(DEFAULT_RATE_PER_MINUTE))
    ),
    burst=int(os.environ.get("REMANENTIA_API_RATE_BURST", str(DEFAULT_BURST))),
)
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/vector/search/public"})
_RATE_EXEMPT_PATHS = frozenset({"/health"})


@app.middleware("http")
async def require_bearer_token(request: Request, call_next):
    """Apply FastAPI request security gates before endpoint handlers run."""
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            enforce_body_size(int(request.headers.get("content-length", "0")), BODY_LIMIT)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=413)

    if request.url.path not in _RATE_EXEMPT_PATHS:
        client = request.client.host if request.client else "unknown"
        if not LIMITER.allow(client):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)

    if request.url.path not in _AUTH_EXEMPT_PATHS and not AUTH.check_header(
        request.headers.get("Authorization")
    ):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    return await call_next(request)


class RecallRequest(BaseModel):
    query: str
    top_k: int = 3
    format: str = "summary"
    include_content: bool = False


class PublicVectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source: str = ""


class ConsolidateRequest(BaseModel):
    force: bool = False


@app.get("/health")
def health():
    legacy_daemon = _legacy_daemon_state()
    vector_worker = _vector_worker_state()
    daemon_state = (
        "alive" if vector_worker["state"] == "alive" else str(legacy_daemon.get("state", "stale"))
    )
    return {
        "status": "ok",
        "daemon": daemon_state,
        "daemon_kind": "vector_worker" if vector_worker["state"] == "alive" else "legacy",
        "legacy_daemon": legacy_daemon["state"],
        "vector_worker": vector_worker["state"],
        "version": "0.2.0",
    }


@app.post("/recall")
def recall_endpoint(req: RecallRequest):
    from memory_recall import recall

    ctx = recall(req.query, top_k=req.top_k, include_content=req.include_content)

    if req.format == "context":
        return _json_safe({"context": ctx.to_llm_context(), "elapsed_ms": ctx.elapsed_ms})

    return _json_safe(
        {
            "query": ctx.query,
            "trace": ctx.trace,
            "score": ctx.trace_score,
            "snippet": ctx.trace_snippet[:300] if ctx.trace_snippet else "",
            "semantic_memories": [
                {"path": s["path"], "key_point": s.get("key_point", "")[:200]}
                for s in ctx.semantic_memories[:5]
            ],
            "entities": ctx.entities,
            "related": ctx.related_entities[:8],
            "before": ctx.before,
            "after": ctx.after,
            "cross_project": ctx.cross_project[:3],
            "novelty": ctx.novelty_score,
            "elapsed_ms": ctx.elapsed_ms,
        }
    )


@app.post("/vector/search/public")
def public_vector_search_endpoint(req: PublicVectorSearchRequest):
    """Public-safe vector search.

    The result policy is server-controlled. Callers can narrow by source,
    but cannot widen the public corpus allowlist.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    if req.top_k <= 0:
        return {"query": req.query, "results": []}

    from vector_index import HttpEmbeddingClient, VectorIndexError
    from vector_pipeline import (
        DEFAULT_VECTOR_INDEX_DIR,
        PublicVectorResultPolicy,
        public_vector_results,
        search_memory_vector_index,
    )

    policy = PublicVectorResultPolicy(
        allowed_sources=tuple(_split_env("REMANENTIA_PUBLIC_VECTOR_SOURCES")),
        allowed_path_prefixes=tuple(_split_env("REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES")),
        redacted_terms=tuple(_read_terms_from_env("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE")),
        max_text_chars=int(os.environ.get("REMANENTIA_PUBLIC_VECTOR_MAX_TEXT_CHARS", "800")),
    )
    try:
        provider = HttpEmbeddingClient.from_env("REMANENTIA_EMBEDDING")
        raw_results = search_memory_vector_index(
            Path(os.environ.get("REMANENTIA_VECTOR_INDEX_DIR", str(DEFAULT_VECTOR_INDEX_DIR))),
            req.query,
            provider,
            top_k=req.top_k,
            source=req.source,
        )
    except (OSError, ValueError, VectorIndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"query": req.query, "results": public_vector_results(raw_results, policy)}


@app.post("/consolidate")
def consolidate_endpoint(req: ConsolidateRequest):
    from consolidation_engine import consolidate

    return consolidate(force=req.force)


@app.get("/status")
def status():
    traces_dir = BASE / "reasoning_traces"
    semantic_dir = BASE / "memory" / "semantic"

    n_traces = len(list(traces_dir.glob("*.md"))) if traces_dir.exists() else 0
    n_semantic = len(list(semantic_dir.rglob("*.md"))) if semantic_dir.exists() else 0

    n_entities = 0
    n_relations = 0
    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"
    if entities_path.exists():
        n_entities = sum(1 for ln in entities_path.read_text().strip().split("\n") if ln.strip())
    if relations_path.exists():  # pragma: no cover
        n_relations = sum(1 for ln in relations_path.read_text().strip().split("\n") if ln.strip())

    daemon_state = {}
    state_path = STATE_DIR / "current_state.json"
    if state_path.exists():
        s = json.loads(state_path.read_text(encoding="utf-8"))
        daemon_state = {
            "cycle": s.get("cycle"),
            "neurons": s.get("n_neurons"),
            "vram_mb": s.get("vram_mb"),
            "live_retrieval": s.get("live_retrieval_available"),
            "age_s": round(time.time() - s.get("timestamp", 0)),
        }

    return {
        "episodic_traces": n_traces,
        "semantic_memories": n_semantic,
        "entities": n_entities,
        "relations": n_relations,
        "daemon": daemon_state,
        "vector_worker": _vector_worker_state(),
    }


@app.get("/entities")
def list_entities():
    entities_path = GRAPH_DIR / "entities.jsonl"
    if not entities_path.exists():
        return []
    return [json.loads(ln) for ln in entities_path.read_text().strip().split("\n") if ln.strip()]


@app.get("/graph")
def graph(top: int = Query(default=15, le=100)):
    relations_path = GRAPH_DIR / "relations.jsonl"
    if not relations_path.exists():
        return []
    rels = [json.loads(ln) for ln in relations_path.read_text().strip().split("\n") if ln.strip()]
    return sorted(rels, key=lambda r: -r.get("weight", 0))[:top]


@app.get("/graph/entity/{entity_id}")
def entity_detail(entity_id: str):
    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"

    entity = None
    if entities_path.exists():
        for ln in entities_path.read_text().strip().split("\n"):
            if ln.strip():
                e = json.loads(ln)
                if e["id"] == entity_id:
                    entity = e
                    break
    if not entity:
        return {"error": f"Entity '{entity_id}' not found"}

    connections = []
    if relations_path.exists():
        for ln in relations_path.read_text().strip().split("\n"):
            if ln.strip():
                r = json.loads(ln)
                if r["source"] == entity_id or r["target"] == entity_id:
                    other = r["target"] if r["source"] == entity_id else r["source"]
                    connections.append(
                        {
                            "entity": other,
                            "weight": r["weight"],
                            "relation": r["type"],
                            "evidence": r.get("evidence", []),
                        }
                    )

    connections.sort(key=lambda x: -x["weight"])
    return {"entity": entity, "connections": connections}


def _split_env(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _json_safe(value):
    """Convert scalar values returned by numeric backends into JSON-safe types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            return value
    return value


def _legacy_daemon_state() -> dict[str, object]:
    state_path = STATE_DIR / "current_state.json"
    if not state_path.exists():
        return {"state": "stale"}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unreadable"}
    age_s = round(time.time() - float(payload.get("timestamp", 0)))
    return {
        "age_s": age_s,
        "cycle": payload.get("cycle"),
        "state": "alive" if age_s < 120 else "stale",
    }


def _vector_worker_state() -> dict[str, object]:
    state_path = STATE_DIR / "vector_refresh_worker.json"
    if not state_path.exists():
        return {"state": "missing"}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unreadable"}
    age_s = round(time.time() - float(payload.get("timestamp_unix", 0)))
    state = "alive" if age_s < VECTOR_WORKER_MAX_AGE_S else "stale"
    return {
        "age_s": age_s,
        "cycle": payload.get("cycle"),
        "last_action": (payload.get("result") or {}).get("action")
        if isinstance(payload.get("result"), dict)
        else None,
        "pid": payload.get("pid"),
        "state": state,
        "status": payload.get("status"),
    }


def _read_terms_from_env(name: str) -> list[str]:
    path_value = os.environ.get(name, "").strip()
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"{name} points to a missing file")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
