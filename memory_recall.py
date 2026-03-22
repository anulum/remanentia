# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Deep Memory Recall

"""Structured recall: retrieval + consolidation + graph + temporal context.

Usage::
    from memory_recall import recall
    context = recall("Dimits shift convergence")
    print(context.summary)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
TRACES_DIR = BASE / "reasoning_traces"
SEMANTIC_DIR = BASE / "memory" / "semantic"
GRAPH_DIR = BASE / "memory" / "graph"
HISTORY_PATH = BASE / "snn_state" / "retrieval_history.jsonl"


@dataclass
class MemoryContext:
    """Structured recall result — everything the system knows about a query."""
    query: str
    # Primary match
    trace: str = ""
    trace_score: float = 0.0
    trace_snippet: str = ""
    # Consolidated knowledge
    semantic_memories: list[dict] = field(default_factory=list)
    # Entity graph
    entities: list[str] = field(default_factory=list)
    related_entities: list[dict] = field(default_factory=list)
    # Temporal context
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    # Cross-project
    cross_project: list[dict] = field(default_factory=list)
    # Novelty
    novelty_score: float = 0.0
    # Retrieval metadata
    elapsed_ms: float = 0.0
    sources_consulted: int = 0

    @property
    def summary(self) -> str:
        lines = [f"Query: {self.query}"]
        if self.trace:
            lines.append(f"Match: {self.trace} (score={self.trace_score:.3f})")
        if self.trace_snippet:
            lines.append(f"Snippet: {self.trace_snippet[:200]}")
        if self.semantic_memories:
            lines.append(f"Consolidated knowledge: {len(self.semantic_memories)} memories")
            for sm in self.semantic_memories[:3]:
                lines.append(f"  - {sm.get('path', '?')}: {sm.get('key_point', '')[:100]}")
        if self.entities:
            lines.append(f"Entities: {', '.join(self.entities[:10])}")
        if self.related_entities:
            lines.append(f"Related: {len(self.related_entities)} connections")
            for re_ in self.related_entities[:5]:
                lines.append(f"  - {re_['entity']} (weight={re_['weight']}, via {re_['relation']})")
        if self.before or self.after:
            lines.append(f"Timeline: {len(self.before)} before, {len(self.after)} after")
        if self.cross_project:
            lines.append(f"Cross-project: {len(self.cross_project)} insights")
        lines.append(f"Novelty: {self.novelty_score:.2f}")
        lines.append(f"Recall took {self.elapsed_ms:.0f}ms, consulted {self.sources_consulted} sources")
        return "\n".join(lines)

    def to_llm_context(self) -> str:
        """Format for injection into LLM context window."""
        parts = []
        if self.trace_snippet:
            parts.append(f"[Matched trace: {self.trace}]\n{self.trace_snippet}")
        for sm in self.semantic_memories[:3]:
            parts.append(f"[Consolidated: {sm.get('path', '')}]\n{sm.get('content', '')[:500]}")
        if self.related_entities:
            ents = ", ".join(f"{r['entity']}({r['weight']})" for r in self.related_entities[:8])
            parts.append(f"[Related concepts: {ents}]")
        if self.before:
            parts.append(f"[Before: {', '.join(self.before[:3])}]")
        if self.after:
            parts.append(f"[After: {', '.join(self.after[:3])}]")
        if self.cross_project:
            cp = "; ".join(f"{c['project']}: {c['insight'][:80]}" for c in self.cross_project[:3])
            parts.append(f"[Cross-project: {cp}]")
        return "\n\n".join(parts)


# ── Entity graph queries ─────────────────────────────────────────

def _load_entities() -> dict[str, dict]:
    path = GRAPH_DIR / "entities.jsonl"
    if not path.exists():
        return {}
    entities = {}
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            e = json.loads(line)
            entities[e["id"]] = e
    return entities


def _load_relations() -> list[dict]:
    path = GRAPH_DIR / "relations.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]


def _find_related(entity_id: str, relations: list[dict], top_k: int = 10) -> list[dict]:
    """Find entities most strongly connected to the given entity."""
    connected = []
    for r in relations:
        if r["source"] == entity_id:
            connected.append({"entity": r["target"], "weight": r["weight"],
                              "relation": r["type"], "evidence": r.get("evidence", [])})
        elif r["target"] == entity_id:
            connected.append({"entity": r["source"], "weight": r["weight"],
                              "relation": r["type"], "evidence": r.get("evidence", [])})
    connected.sort(key=lambda x: -x["weight"])
    return connected[:top_k]


def _entities_for_query(query: str, entities: dict) -> list[str]:
    """Find entities mentioned in the query."""
    q_lower = query.lower()
    found = []
    for eid, edata in entities.items():
        if eid in q_lower or edata.get("label", "").lower() in q_lower:
            found.append(eid)
    return found


# ── Semantic memory search ───────────────────────────────────────

def _search_semantic(query: str, top_k: int = 5) -> list[dict]:
    """Search consolidated semantic memories."""
    if not SEMANTIC_DIR.exists():
        return []

    q_tokens = set(re.findall(r"\w+", query.lower()))
    results = []

    for md_file in SEMANTIC_DIR.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        # Parse frontmatter
        meta = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
                content = parts[2].strip()

        text_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(q_tokens & text_tokens)
        if overlap == 0:
            continue

        score = overlap / max(len(q_tokens), 1)

        # Extract first key point
        key_point = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- ") and len(stripped) > 20:
                key_point = stripped.lstrip("- ").strip()
                break

        results.append({
            "path": str(md_file.relative_to(BASE)),
            "score": score,
            "project": meta.get("project", ""),
            "type": meta.get("type", ""),
            "date": meta.get("date", ""),
            "source_traces": meta.get("source_traces", ""),
            "content": content[:1000],
            "key_point": key_point,
        })

    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── Temporal context ─────────────────────────────────────────────

def _temporal_context(trace_name: str) -> tuple[list[str], list[str]]:
    """Find traces that came before and after the given trace."""
    all_traces = sorted(TRACES_DIR.glob("*.md"), key=lambda f: f.name)
    names = [f.name for f in all_traces]

    if trace_name not in names:
        return [], []

    idx = names.index(trace_name)
    before = names[max(0, idx - 3):idx]
    after = names[idx + 1:idx + 4]
    return before, after


# ── Cross-project insights ───────────────────────────────────────

def _cross_project_insights(entities: list[str], primary_project: str,
                             all_entities: dict, relations: list[dict]) -> list[dict]:
    """Find insights from other projects that share entities."""
    insights = []
    seen_projects = {primary_project}

    for eid in entities:
        related = _find_related(eid, relations, top_k=20)
        for r in related:
            r_entity = all_entities.get(r["entity"], {})
            r_project = r_entity.get("type", "")
            if r_project == "project" and r["entity"] not in seen_projects:
                seen_projects.add(r["entity"])
                # Find what connects them
                shared = []
                for rel in relations:
                    if (rel["source"] == r["entity"] or rel["target"] == r["entity"]):
                        other = rel["target"] if rel["source"] == r["entity"] else rel["source"]
                        if other in entities:
                            shared.append(other)
                if shared:
                    insights.append({
                        "project": r["entity"],
                        "shared_concepts": shared,
                        "insight": f"{r['entity']} shares concepts: {', '.join(shared[:5])}",
                        "weight": r["weight"],
                    })

    insights.sort(key=lambda x: -x["weight"])
    return insights[:5]


# ── Novelty assessment ───────────────────────────────────────────

def _assess_novelty(query: str, entities: dict) -> float:
    """How novel is this query relative to known entities?
    High novelty = query mentions things we haven't seen before.
    """
    q_tokens = set(re.findall(r"\w{4,}", query.lower()))
    known_tokens = set()
    for eid, edata in entities.items():
        known_tokens.add(eid)
        known_tokens.update(re.findall(r"\w{4,}", edata.get("label", "").lower()))

    if not q_tokens:
        return 0.0

    unknown = q_tokens - known_tokens
    return len(unknown) / len(q_tokens)


# ── Main recall function ─────────────────────────────────────────

def recall(
    query: str,
    top_k: int = 3,
    include_content: bool = True,
) -> MemoryContext:
    """Deep memory recall: retrieval + consolidation + graph + temporal.

    This is the main entry point for rich context retrieval.
    """
    t0 = time.monotonic()
    ctx = MemoryContext(query=query)
    sources = 0

    # 1. Primary retrieval via existing retrieve.py
    try:
        from retrieve import retrieve
        results = retrieve(query, top_k=1, include_content=include_content)
        if results:
            ctx.trace = results[0]["trace"]
            ctx.trace_score = results[0]["score"]
            if include_content and "content" in results[0]:
                ctx.trace_snippet = results[0]["content"][:500]
            elif ctx.trace:
                trace_path = TRACES_DIR / ctx.trace
                if trace_path.exists():
                    ctx.trace_snippet = trace_path.read_text(encoding="utf-8")[:500]
        sources += 1
    except Exception:
        pass

    # 2. Search consolidated semantic memories
    semantic = _search_semantic(query, top_k=5)
    ctx.semantic_memories = semantic
    sources += len(semantic)

    # 3. Entity graph
    all_entities = _load_entities()
    relations = _load_relations()
    query_entities = _entities_for_query(query, all_entities)

    # Also extract entities from matched trace
    if ctx.trace:
        trace_text = ""
        trace_path = TRACES_DIR / ctx.trace
        if trace_path.exists():
            trace_text = trace_path.read_text(encoding="utf-8")
        for eid in all_entities:
            if eid in trace_text.lower() and eid not in query_entities:
                query_entities.append(eid)

    ctx.entities = query_entities

    # Find related entities across all query entities
    all_related = {}
    for eid in query_entities:
        for r in _find_related(eid, relations, top_k=10):
            key = r["entity"]
            if key not in all_related or r["weight"] > all_related[key]["weight"]:
                all_related[key] = r
    ctx.related_entities = sorted(all_related.values(), key=lambda x: -x["weight"])[:10]
    sources += len(relations)

    # 4. Temporal context
    if ctx.trace:
        ctx.before, ctx.after = _temporal_context(ctx.trace)

    # 5. Cross-project insights
    if query_entities:
        primary = query_entities[0] if query_entities else ""
        ctx.cross_project = _cross_project_insights(
            query_entities, primary, all_entities, relations)

    # 6. Novelty
    ctx.novelty_score = _assess_novelty(query, all_entities)

    ctx.elapsed_ms = (time.monotonic() - t0) * 1000
    ctx.sources_consulted = sources

    return ctx


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    query = " ".join(sys.argv[1:]) or "Dimits shift convergence"
    print(f"Recalling: {query}\n")
    ctx = recall(query, include_content=False)
    print(ctx.summary)
    print("\n--- LLM Context ---")
    print(ctx.to_llm_context())
