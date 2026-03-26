# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP Server (Model Context Protocol)

"""MCP server for Remanentia — lets any MCP-compatible agent
(Claude Code, Cursor, etc.) use Remanentia as a memory tool.

Provides four tools:
  - remanentia_recall: Search memory
  - remanentia_remember: Persist a memory
  - remanentia_status: System status
  - remanentia_graph: Entity relationship query

Usage (stdio transport)::
    python 04_ARCANE_SAPIENCE/mcp_server.py

Configure in .mcp.json::
    {
      "mcpServers": {
        "remanentia": {
          "command": "python",
          "args": ["04_ARCANE_SAPIENCE/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(os.environ.get("REMANENTIA_BASE", Path(__file__).parent))
GRAPH_DIR = BASE / "memory" / "graph"


_UNIFIED_INDEX = None


_KNOWLEDGE_STORE = None


def _get_knowledge_store():
    global _KNOWLEDGE_STORE
    if _KNOWLEDGE_STORE is not None:
        return _KNOWLEDGE_STORE
    try:
        from knowledge_store import KnowledgeStore
        _KNOWLEDGE_STORE = KnowledgeStore()
        _KNOWLEDGE_STORE.load()
    except Exception:  # pragma: no cover
        _KNOWLEDGE_STORE = KnowledgeStore() if _KNOWLEDGE_STORE is None else _KNOWLEDGE_STORE
    return _KNOWLEDGE_STORE


def handle_recall(query: str, top_k: int = 5,
                  project: str = "", after: str = "", before: str = "",
                  llm: bool = False) -> str:
    """Memory recall via unified BM25 + embedding index + knowledge notes."""
    global _UNIFIED_INDEX

    # Check prospective triggers
    trigger_lines = []
    try:
        ks = _get_knowledge_store()
        matched_triggers = ks.check_triggers(query)
        for t in matched_triggers:  # pragma: no cover
            trigger_lines.append(f"[TRIGGER] {t.action}")
    except Exception:  # pragma: no cover
        pass

    try:
        from memory_index import MemoryIndex
        if _UNIFIED_INDEX is None:
            _UNIFIED_INDEX = MemoryIndex()
            if not _UNIFIED_INDEX.load():  # pragma: no cover
                _UNIFIED_INDEX = None
                return _lightweight_recall(query, top_k)

        use_llm = llm or bool(os.environ.get("REMANENTIA_LLM_ANSWERS"))
        results = _UNIFIED_INDEX.search(
            query, top_k=top_k, project=project, after=after, before=before,
            use_llm=use_llm)
        if not results and not trigger_lines:
            return f"No memories found for: {query}"

        parts = list(trigger_lines)
        for r in results:
            header = f"[{r.source}] {r.name} (score={r.score:.1f})"
            if r.answer:
                header += f"\nAnswer: {r.answer}"
            parts.append(f"{header}\n{r.snippet}")

        # Knowledge store graph search — multi-hop traversal via prospective queries
        try:
            ks = _get_knowledge_store()
            seen_snippets = {r.snippet[:100] for r in results}
            ks_notes = ks.graph_search(query, top_k=top_k, hop_depth=2)
            for note in ks_notes:
                snippet = note.content[:300]
                if snippet[:100] not in seen_snippets:
                    seen_snippets.add(snippet[:100])
                    parts.append(f"[knowledge] {note.title} (type={note.note_type})\n{snippet}")
        except Exception:  # pragma: no cover
            pass

        return "\n\n".join(parts)

    except Exception:  # pragma: no cover
        return _lightweight_recall(query, top_k)


def handle_remember(content: str, memory_type: str = "context", project: str = "",
                    trigger: str = "") -> str:
    """Persist a memory as a reasoning trace and update the index."""
    from datetime import datetime

    traces_dir = BASE / "reasoning_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H%M")
    safe_type = memory_type.replace(" ", "_")[:20]
    safe_project = project.replace(" ", "_")[:20] if project else "general"
    filename = f"{ts}_{safe_project}_{safe_type}.md"
    path = traces_dir / filename

    lines = [f"# {memory_type.title()}: {project or 'general'}",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"Project: {project or 'general'}",
             f"Type: {memory_type}", "",
             content]
    path.write_text("\n".join(lines), encoding="utf-8")

    # Invalidate lightweight recall cache
    global _RECALL_INDEX
    _RECALL_INDEX = None

    # Incremental index update if unified index is loaded
    global _UNIFIED_INDEX
    if _UNIFIED_INDEX is not None and _UNIFIED_INDEX._built:
        try:
            _UNIFIED_INDEX.add_file(path)
        except Exception:  # pragma: no cover
            pass

    # Create knowledge note
    try:
        ks = _get_knowledge_store()
        ks.add_note(content, source=filename)
        if trigger:  # pragma: no cover
            ks.add_trigger(trigger, content)
        ks.save()
    except Exception:  # pragma: no cover
        pass

    # Consolidate pending traces (fast short-circuit when nothing new)
    try:
        from consolidation_engine import consolidate
        consolidate()
        _RECALL_INDEX = None
    except Exception:  # pragma: no cover
        pass

    return f"Remembered: {filename} ({len(content)} chars)"


_RECALL_INDEX: dict[str, tuple[set, str]] | None = None


def _build_recall_index() -> dict[str, tuple[set, str]]:
    """Build in-memory token index of all traces and semantic memories.

    Called once, cached for the process lifetime.
    """
    import re
    global _RECALL_INDEX
    if _RECALL_INDEX is not None:
        return _RECALL_INDEX

    index = {}
    traces_dir = BASE / "reasoning_traces"
    semantic_dir = BASE / "memory" / "semantic"

    if traces_dir.exists():
        for f in traces_dir.glob("*.md"):
            text = f.read_text(encoding="utf-8")
            tokens = set(re.findall(r"\w{3,}", text.lower()))
            index[f.name] = (tokens, text[:500])

    if semantic_dir.exists():
        for f in semantic_dir.rglob("*.md"):
            text = f.read_text(encoding="utf-8")
            tokens = set(re.findall(r"\w{3,}", text.lower()))
            rel = str(f.relative_to(semantic_dir))
            index[f"[semantic] {rel}"] = (tokens, text[:500])

    _RECALL_INDEX = index
    return index


def _lightweight_recall(query: str, top_k: int = 3) -> str:
    """Fast recall from cached in-memory index.

    First call: ~2s (reads files). Subsequent calls: <50ms.
    """
    import re
    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    if not q_tokens:
        return "Empty query."

    index = _build_recall_index()
    scored = []
    for name, (t_tokens, snippet) in index.items():
        overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        if overlap > 0:
            scored.append((name, overlap, snippet))

    scored.sort(key=lambda x: -x[1])
    top = scored[:top_k]

    if not top:
        return f"No memories found for: {query}"

    parts = []
    for name, score, snippet in top:
        parts.append(f"[{name} (score={score:.2f})]\n{snippet}")

    return "\n\n".join(parts)


def handle_status() -> str:
    """System status summary."""
    import io
    old_stdout = sys.stdout
    try:
        sys.stdout = buf = io.StringIO()
        from cli import cmd_status
        cmd_status(type("Args", (), {})())
        return buf.getvalue()
    except Exception as e:  # pragma: no cover
        return f"Status error: {e}"
    finally:
        sys.stdout = old_stdout


def handle_graph(entity: str = "", top: int = 10) -> str:
    """Entity graph query."""
    relations_path = GRAPH_DIR / "relations.jsonl"
    if not relations_path.exists():
        return "No relations. Run consolidation first."

    rels = [json.loads(l) for l in relations_path.read_text().strip().split("\n") if l.strip()]

    if entity:
        matches = [r for r in rels if r["source"] == entity or r["target"] == entity]
        matches.sort(key=lambda r: -r.get("weight", 0))
        lines = [f"Connections for '{entity}':"]
        for r in matches[:top]:
            other = r["target"] if r["source"] == entity else r["source"]
            lines.append(f"  {other} (weight={r['weight']}, {len(r.get('evidence',[]))} traces)")
        return "\n".join(lines)

    top_rels = sorted(rels, key=lambda r: -r.get("weight", 0))[:top]
    lines = [f"Top {len(top_rels)} entity relationships:"]
    for r in top_rels:
        lines.append(f"  {r['source']} <-> {r['target']} weight={r['weight']}")
    return "\n".join(lines)


# ── MCP Protocol (stdio JSON-RPC) ────────────────────────────────

TOOLS = [
    {
        "name": "remanentia_recall",
        "description": "Deep memory recall. Returns matched trace, consolidated knowledge, entity graph connections, temporal context (before/after), and cross-project insights. Use this when you need context about past work, decisions, or findings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 3},
                "project": {"type": "string", "description": "Filter by project/source name", "default": ""},
                "after": {"type": "string", "description": "Only docs after date (YYYY-MM-DD)", "default": ""},
                "before": {"type": "string", "description": "Only docs before date (YYYY-MM-DD)", "default": ""},
                "llm": {"type": "boolean", "description": "Use LLM for answer extraction (costs API credits)", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remanentia_remember",
        "description": "Persist a memory for future recall. Optionally set a trigger condition for prospective memory — the memory will be surfaced automatically when a future query matches the trigger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "type": {"type": "string", "description": "Memory type: decision, finding, metric, context", "default": "context"},
                "project": {"type": "string", "description": "Project name (optional)", "default": ""},
                "trigger": {"type": "string", "description": "Prospective trigger: when future queries match this condition, surface this memory automatically", "default": ""},
            },
            "required": ["content"],
        },
    },
    {
        "name": "remanentia_status",
        "description": "Check Remanentia system status: daemon, memory counts, disk usage.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "remanentia_graph",
        "description": "Query the entity relationship graph. Optionally filter by entity name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to query (empty = top relationships)", "default": ""},
                "top": {"type": "integer", "description": "Number of results", "default": 10},
            },
        },
    },
]


def handle_request(request: dict) -> dict:
    method = request.get("method", "")
    rid = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remanentia", "version": "0.2.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "remanentia_recall":
            text = handle_recall(args.get("query", ""), args.get("top_k", 3),
                                 project=args.get("project", ""), after=args.get("after", ""),
                                 before=args.get("before", ""),
                                 llm=args.get("llm", False))
        elif tool_name == "remanentia_remember":
            text = handle_remember(args.get("content", ""), args.get("type", "context"),
                                   args.get("project", ""), args.get("trigger", ""))
        elif tool_name == "remanentia_status":
            text = handle_status()
        elif tool_name == "remanentia_graph":
            text = handle_graph(args.get("entity", ""), args.get("top", 10))
        else:
            text = f"Unknown tool: {tool_name}"

        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {"content": [{"type": "text", "text": text}]},
        }

    if method == "notifications/initialized":
        return None

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():  # pragma: no cover
    """Run MCP server on stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
