# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — API Server

"""FastAPI server for Remanentia memory system.

Usage::
    cd 04_ARCANE_SAPIENCE
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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class RecallRequest(BaseModel):
    query: str
    top_k: int = 3
    format: str = "summary"
    include_content: bool = False


class ConsolidateRequest(BaseModel):
    force: bool = False


@app.get("/health")
def health():
    state_path = STATE_DIR / "current_state.json"
    daemon_alive = False
    if state_path.exists():
        s = json.loads(state_path.read_text(encoding="utf-8"))
        daemon_alive = (time.time() - s.get("timestamp", 0)) < 120
    return {
        "status": "ok",
        "daemon": "alive" if daemon_alive else "stale",
        "version": "0.2.0",
    }


@app.post("/recall")
def recall_endpoint(req: RecallRequest):
    from memory_recall import recall
    ctx = recall(req.query, top_k=req.top_k, include_content=req.include_content)

    if req.format == "context":
        return {"context": ctx.to_llm_context(), "elapsed_ms": ctx.elapsed_ms}

    return {
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
        n_entities = sum(1 for l in entities_path.read_text().strip().split("\n") if l.strip())
    if relations_path.exists():  # pragma: no cover
        n_relations = sum(1 for l in relations_path.read_text().strip().split("\n") if l.strip())

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
    }


@app.get("/entities")
def list_entities():
    entities_path = GRAPH_DIR / "entities.jsonl"
    if not entities_path.exists():
        return []
    return [json.loads(l) for l in entities_path.read_text().strip().split("\n") if l.strip()]


@app.get("/graph")
def graph(top: int = Query(default=15, le=100)):
    relations_path = GRAPH_DIR / "relations.jsonl"
    if not relations_path.exists():
        return []
    rels = [json.loads(l) for l in relations_path.read_text().strip().split("\n") if l.strip()]
    return sorted(rels, key=lambda r: -r.get("weight", 0))[:top]


@app.get("/graph/entity/{entity_id}")
def entity_detail(entity_id: str):
    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"

    entity = None
    if entities_path.exists():
        for l in entities_path.read_text().strip().split("\n"):
            if l.strip():
                e = json.loads(l)
                if e["id"] == entity_id:
                    entity = e
                    break
    if not entity:
        return {"error": f"Entity '{entity_id}' not found"}

    connections = []
    if relations_path.exists():
        for l in relations_path.read_text().strip().split("\n"):
            if l.strip():
                r = json.loads(l)
                if r["source"] == entity_id or r["target"] == entity_id:
                    other = r["target"] if r["source"] == entity_id else r["source"]
                    connections.append({
                        "entity": other, "weight": r["weight"],
                        "relation": r["type"], "evidence": r.get("evidence", []),
                    })

    connections.sort(key=lambda x: -x["weight"])
    return {"entity": entity, "connections": connections}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
