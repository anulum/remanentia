# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP server (Model Context Protocol)

"""MCP server for Remanentia — lets any MCP-compatible agent
(Cursor and others) use Remanentia as a memory tool.

Provides four tools:
  - remanentia_recall: Search memory
  - remanentia_remember: Persist a memory
  - remanentia_status: System status
  - remanentia_graph: Entity relationship query

Usage (stdio transport)::
    python workspace-internal/mcp_server.py

Configure in .mcp.json::
    {
      "mcpServers": {
        "remanentia": {
          "command": "python",
          "args": ["workspace-internal/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time as _time
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(os.environ.get("REMANENTIA_BASE", Path(__file__).parent))
GRAPH_DIR = BASE / "memory" / "graph"

_lock = threading.Lock()
log = logging.getLogger(__name__)

_UNIFIED_INDEX = None


_KNOWLEDGE_STORE = None

# Async consolidation: debounce to at most once per 10 seconds
_consolidation_pending = False
_consolidation_last = 0.0
_CONSOLIDATION_DEBOUNCE_S = 10


def _runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load runtime integrations without making static checks transitive."""
    return getattr(import_module(module_name), attr_name)


def _get_knowledge_store():
    global _KNOWLEDGE_STORE
    if _KNOWLEDGE_STORE is not None:
        return _KNOWLEDGE_STORE
    with _lock:
        if _KNOWLEDGE_STORE is not None:  # pragma: no cover — thread safety guard
            return _KNOWLEDGE_STORE
        try:
            KnowledgeStore = _runtime_attr("knowledge_store", "KnowledgeStore")
            _KNOWLEDGE_STORE = KnowledgeStore()
            _KNOWLEDGE_STORE.load()
        except Exception:  # pragma: no cover
            KnowledgeStore = _runtime_attr("knowledge_store", "KnowledgeStore")
            _KNOWLEDGE_STORE = KnowledgeStore() if _KNOWLEDGE_STORE is None else _KNOWLEDGE_STORE
    return _KNOWLEDGE_STORE


def handle_recall(
    query: str,
    top_k: int = 5,
    project: str = "",
    after: str = "",
    before: str = "",
    llm: bool = False,
) -> str:
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
        log.debug("Prospective trigger check failed", exc_info=True)

    try:
        MemoryIndex = _runtime_attr("memory_index", "MemoryIndex")

        if _UNIFIED_INDEX is None:
            with _lock:
                if _UNIFIED_INDEX is None:
                    idx = MemoryIndex()
                    if not idx.load():  # pragma: no cover
                        return _lightweight_recall(query, top_k)
                    _UNIFIED_INDEX = idx

        use_llm = llm or bool(os.environ.get("REMANENTIA_LLM_ANSWERS"))
        if use_llm:
            get_llm_backend = _runtime_attr("answer_extractor", "get_llm_backend")

            if get_llm_backend() is None:
                resolve_backend = _runtime_attr("llm_backend", "resolve_backend")
                set_llm_backend = _runtime_attr("answer_extractor", "set_llm_backend")

                backend_name = os.environ.get("REMANENTIA_LLM_BACKEND", "auto")
                set_llm_backend(resolve_backend(backend_name))
        results = _UNIFIED_INDEX.search(
            query, top_k=top_k, project=project, after=after, before=before, use_llm=use_llm
        )
        if not results and not trigger_lines:
            return f"No memories found for: {query}"

        # Guarded tier: when enabled and director-ai is available, check
        # every extracted answer span against the retrieved evidence before
        # returning it. An answer that falls below the block_below
        # threshold is silently dropped (snippet-only view), and the
        # reason is written to the audit log for each filtered hit.
        guarded_on = bool(os.environ.get("REMANENTIA_GUARDED"))
        guard_log: list[str] = []
        if guarded_on and results:
            try:
                facts_from_results = _runtime_attr("memory_guarded", "facts_from_results")
                is_available = _runtime_attr("memory_guarded", "is_available")
                score_memory_answer = _runtime_attr("memory_guarded", "score_memory_answer")

                if is_available():  # pragma: no cover — needs optional dep
                    facts = facts_from_results(results)
                    for r in results:
                        if not r.answer:
                            continue
                        gr = score_memory_answer(query, r.answer, facts)
                        if gr is None:
                            continue
                        if gr.blocked:
                            guard_log.append(f"[guard] blocked {r.name}: {gr.reason}")
                            r.answer = ""
            except Exception:  # pragma: no cover
                log.debug("Guarded recall scoring failed", exc_info=True)

        parts = list(trigger_lines)
        for r in results:
            header = f"[{r.source}] {r.name} (score={r.score:.1f})"
            if r.answer:
                header += f"\nAnswer: {r.answer}"
            parts.append(f"{header}\n{r.snippet}")
        parts.extend(guard_log)

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
            log.debug("Knowledge graph recall failed", exc_info=True)

        return "\n\n".join(parts)

    except Exception:  # pragma: no cover
        return _lightweight_recall(query, top_k)


def handle_remember(
    content: str, memory_type: str = "context", project: str = "", trigger: str = ""
) -> str:
    """Persist a memory as a reasoning trace and update the index.

    PII in the supplied content (emails, phone numbers, IBAN, credit
    cards, API-key-shaped tokens) is redacted before the trace hits
    disk. The redaction leaves ``[REDACTED:TAG]`` placeholders so the
    shape of the original text is preserved for retrieval while the
    raw values never persist.
    """
    from datetime import datetime

    atomic_write_text = _runtime_attr("file_utils", "atomic_write_text")
    redact = _runtime_attr("pii_redactor", "redact")

    content = redact(content).text

    traces_dir = BASE / "reasoning_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H%M")
    safe_type = memory_type.replace(" ", "_")[:20]
    safe_project = project.replace(" ", "_")[:20] if project else "general"
    filename = f"{ts}_{safe_project}_{safe_type}.md"
    path = traces_dir / filename

    lines = [
        f"# {memory_type.title()}: {project or 'general'}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Project: {project or 'general'}",
        f"Type: {memory_type}",
        "",
        content,
    ]

    atomic_write_text(path, "\n".join(lines))

    # Invalidate lightweight recall cache
    global _RECALL_INDEX
    _RECALL_INDEX = None

    # Incremental index update if unified index is loaded
    global _UNIFIED_INDEX
    if _UNIFIED_INDEX is not None and _UNIFIED_INDEX._built:
        with _lock:
            try:
                _UNIFIED_INDEX.add_file(path)
            except Exception:  # pragma: no cover
                log.debug("Incremental unified-index update failed", exc_info=True)

    # Create knowledge note
    try:
        ks = _get_knowledge_store()
        ks.add_note(content, source=filename)
        if trigger:  # pragma: no cover
            ks.add_trigger(trigger, content)
        ks.save()
    except Exception:  # pragma: no cover
        log.debug("Knowledge note persistence failed", exc_info=True)

    # Async consolidation with debounce
    _schedule_consolidation()

    return f"Remembered: {filename} ({len(content)} chars)"


def _schedule_consolidation():
    """Run consolidation in a background thread, debounced."""
    global _consolidation_pending, _consolidation_last, _RECALL_INDEX
    now = _time.monotonic()
    if now - _consolidation_last < _CONSOLIDATION_DEBOUNCE_S:
        _consolidation_pending = True
        return
    _consolidation_last = now
    _consolidation_pending = False

    def _run():
        global _RECALL_INDEX, _consolidation_last
        try:
            consolidate = _runtime_attr("consolidation_engine", "consolidate")

            consolidate()
            _RECALL_INDEX = None
        except Exception:  # pragma: no cover
            log.debug("Background consolidation failed", exc_info=True)
        _consolidation_last = _time.monotonic()

    threading.Thread(target=_run, daemon=True).start()


_RECALL_INDEX: dict[str, tuple[set, str]] | None = None

_rust_mcp_tok = None
try:
    _rust_mcp_tok = _runtime_attr("remanentia_search", "tokenize")  # pragma: no cover
except ImportError:
    pass


def _mcp_tok(text: str) -> set[str]:
    """Tokenise text into 3+ char words. Uses Rust when available."""
    if _rust_mcp_tok is not None:
        return set(_rust_mcp_tok(text))  # pragma: no cover
    import re

    return set(re.findall(r"\w{3,}", text.lower()))


def _build_recall_index() -> dict[str, tuple[set, str]]:
    """Build in-memory token index of all traces and semantic memories.

    Called once, cached for the process lifetime.
    Uses Rust (remanentia_search) when available.
    """
    global _RECALL_INDEX
    if _RECALL_INDEX is not None:
        return _RECALL_INDEX

    index = {}
    traces_dir = BASE / "reasoning_traces"
    semantic_dir = BASE / "memory" / "semantic"

    if traces_dir.exists():
        for f in traces_dir.glob("*.md"):
            text = f.read_text(encoding="utf-8")
            tokens = _mcp_tok(text)
            index[f.name] = (tokens, text[:500])

    if semantic_dir.exists():
        for f in semantic_dir.rglob("*.md"):
            text = f.read_text(encoding="utf-8")
            tokens = _mcp_tok(text)
            rel = str(f.relative_to(semantic_dir))
            index[f"[semantic] {rel}"] = (tokens, text[:500])

    _RECALL_INDEX = index
    return index


def _lightweight_recall(query: str, top_k: int = 3) -> str:
    """Fast recall from cached in-memory index.

    First call: ~2s (reads files). Subsequent calls: <50ms.
    """
    q_tokens = _mcp_tok(query)
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
        cmd_status = _runtime_attr("cli", "cmd_status")

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
            lines.append(f"  {other} (weight={r['weight']}, {len(r.get('evidence', []))} traces)")
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
                "project": {
                    "type": "string",
                    "description": "Filter by project/source name",
                    "default": "",
                },
                "after": {
                    "type": "string",
                    "description": "Only docs after date (YYYY-MM-DD)",
                    "default": "",
                },
                "before": {
                    "type": "string",
                    "description": "Only docs before date (YYYY-MM-DD)",
                    "default": "",
                },
                "llm": {
                    "type": "boolean",
                    "description": "Use LLM for answer extraction (costs API credits)",
                    "default": False,
                },
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
                "type": {
                    "type": "string",
                    "description": "Memory type: decision, finding, metric, context",
                    "default": "context",
                },
                "project": {
                    "type": "string",
                    "description": "Project name (optional)",
                    "default": "",
                },
                "trigger": {
                    "type": "string",
                    "description": "Prospective trigger: when future queries match this condition, surface this memory automatically",
                    "default": "",
                },
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
                "entity": {
                    "type": "string",
                    "description": "Entity to query (empty = top relationships)",
                    "default": "",
                },
                "top": {"type": "integer", "description": "Number of results", "default": 10},
            },
        },
    },
]


def handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    rid = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remanentia", "version": "0.3.1"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "remanentia_recall":
            text = handle_recall(
                args.get("query", ""),
                args.get("top_k", 3),
                project=args.get("project", ""),
                after=args.get("after", ""),
                before=args.get("before", ""),
                llm=args.get("llm", False),
            )
        elif tool_name == "remanentia_remember":
            text = handle_remember(
                args.get("content", ""),
                args.get("type", "context"),
                args.get("project", ""),
                args.get("trigger", ""),
            )
        elif tool_name == "remanentia_status":
            text = handle_status()
        elif tool_name == "remanentia_graph":
            text = handle_graph(args.get("entity", ""), args.get("top", 10))
        else:
            text = f"Unknown tool: {tool_name}"

        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"content": [{"type": "text", "text": text}]},
        }

    if method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def _parse_cli(argv: list[str] | None = None) -> None:
    """Apply CLI flags to the environment before the server starts.

    Two flags that are pure env-var convenience for a Remanentia
    client that prefers ``python mcp_server.py --llm --local-llm``
    over a four-line ``REMANENTIA_...=...`` shell prefix:

    - ``--llm`` / ``--no-llm`` toggles ``REMANENTIA_LLM_ANSWERS``
      so ``handle_recall`` synthesises an answer rather than
      returning raw retrieval context.
    - ``--local-llm`` pins ``REMANENTIA_LLM_BACKEND=local``, which
      ``resolve_backend`` converts into a ``LocalLLMBackend``
      pointing at the configured Ollama endpoint.

    The real auth, streaming, and session-id plumbing stays on the
    MCP protocol layer; these two switches are the minimum needed
    for a developer to say "use my local model" without editing
    their shell profile.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="mcp_server.py",
        description="Run the Remanentia MCP server on stdio.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="synthesise LLM answers for recall (sets REMANENTIA_LLM_ANSWERS=1)",
    )
    p.add_argument(
        "--local-llm",
        action="store_true",
        help="route LLM answers through the local Ollama backend "
        "(sets REMANENTIA_LLM_BACKEND=local)",
    )
    p.add_argument(
        "--guarded",
        action="store_true",
        help="enable Director-AI grounding check on every returned answer "
        "(sets REMANENTIA_GUARDED=1; requires `pip install remanentia[guarded]`)",
    )
    args, _unknown = p.parse_known_args(argv)
    if args.llm:
        os.environ.setdefault("REMANENTIA_LLM_ANSWERS", "1")
    if args.local_llm:
        os.environ["REMANENTIA_LLM_BACKEND"] = "local"
    if args.guarded:
        os.environ.setdefault("REMANENTIA_GUARDED", "1")


def main():  # pragma: no cover
    """Run MCP server on stdio."""
    _parse_cli()
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
