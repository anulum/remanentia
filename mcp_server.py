# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP server (Model Context Protocol)

"""MCP server for Remanentia — lets any MCP-compatible agent
(Cursor and others) use Remanentia as a memory tool.

Provides six tools:
  - remanentia_recall: Search memory
  - remanentia_remember: Persist a memory
  - remanentia_status: System status
  - remanentia_graph: Entity relationship query
  - remanentia_recall_feedback: Rate whether a recall was used (usage label)
  - remanentia_recall_correctness: Rate whether a recall was correct (gate label)

Usage (stdio transport)::
    python mcp_server.py

Configure in .mcp.json::
    {
      "mcpServers": {
        "remanentia": {
          "command": "python",
          "args": ["mcp_server.py"]
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

from api_security import TokenBucketLimiter
from mcp_protocol import (
    MCP_RATE_LIMIT_ERROR_CODE as _PROTOCOL_RATE_LIMIT_ERROR_CODE,
    TOOLS as _PROTOCOL_TOOLS,
    dispatch_request,
)
from mcp_storage import build_recall_index, query_graph, recall_from_index

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
DEFAULT_MCP_RATE_PER_MINUTE = 600.0
DEFAULT_MCP_BURST = 120
MCP_RATE_LIMIT_ERROR_CODE = _PROTOCOL_RATE_LIMIT_ERROR_CODE
TOOLS = _PROTOCOL_TOOLS


def _runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load runtime integrations without making static checks transitive."""
    return getattr(import_module(module_name), attr_name)


def _default_mcp_audit_logger() -> Any:
    """Build the default metadata-only MCP tool audit logger."""
    ToolAuditLogger = _runtime_attr("api_security", "ToolAuditLogger")
    return ToolAuditLogger.from_env(BASE / ".coordination" / "runtime" / "mcp_tool_audit.jsonl")


MCP_AUDIT_LOGGER = _default_mcp_audit_logger()
_MCP_RATE_LIMITER: TokenBucketLimiter | None = None
_MCP_RATE_LIMIT_CONFIG: tuple[float, int, str] | None = None


_RECALL_LEDGER = None

_BUS_EMITTER = None
_BUS_EMITTER_INIT = False

_OUTCOME_TRACKER = None


def _env_flag_disabled(value: str | None) -> bool:
    """Return True when an environment switch explicitly disables a feature."""
    return value is not None and value.strip().lower() in {"", "0", "false", "off", "no"}


def _read_positive_float_env(var: str, default: float) -> float:
    """Read a positive float environment value or return *default* when unset."""
    value = os.environ.get(var)
    if value is None or value.strip() == "":
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{var} must be positive")
    return parsed


def _read_positive_int_env(var: str, default: int) -> int:
    """Read a positive integer environment value or return *default* when unset."""
    value = os.environ.get(var)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{var} must be positive")
    return parsed


def _mcp_rate_limiter() -> TokenBucketLimiter | None:
    """Return the process-wide MCP tool-call limiter for current env settings."""
    if _env_flag_disabled(os.environ.get("REMANENTIA_MCP_RATE_LIMIT")):
        return None

    rate = _read_positive_float_env("REMANENTIA_MCP_RATE", DEFAULT_MCP_RATE_PER_MINUTE)
    burst = _read_positive_int_env("REMANENTIA_MCP_BURST", DEFAULT_MCP_BURST)
    client_id = os.environ.get("REMANENTIA_MCP_CLIENT_ID", "stdio")
    config = (rate, burst, client_id)

    global _MCP_RATE_LIMITER, _MCP_RATE_LIMIT_CONFIG
    if _MCP_RATE_LIMITER is None or config != _MCP_RATE_LIMIT_CONFIG:
        _MCP_RATE_LIMITER = TokenBucketLimiter(rate_per_minute=rate, burst=burst)
        _MCP_RATE_LIMIT_CONFIG = config
    return _MCP_RATE_LIMITER


def _mcp_rate_limit_key() -> str:
    """Return the token-bucket key for the current MCP client/session."""
    return os.environ.get("REMANENTIA_MCP_CLIENT_ID", "stdio")


def _get_recall_ledger() -> Any:
    """Return the process-wide recall query-stream ledger (lazy singleton)."""
    global _RECALL_LEDGER
    if _RECALL_LEDGER is None:
        default_ledger = _runtime_attr("recall_ledger", "default_ledger")
        _RECALL_LEDGER = default_ledger()
    return _RECALL_LEDGER


def _get_bus_emitter() -> Any | None:
    """Return the process-wide recall bus emitter, or ``None`` if disabled.

    Lazy singleton like the ledger, but ``None`` is a valid resolved value
    (bus opt-out or ``synapse_channel`` absent), so a separate init flag
    distinguishes "not yet built" from "built, no bus".
    """
    global _BUS_EMITTER, _BUS_EMITTER_INIT
    if not _BUS_EMITTER_INIT:
        default_emitter = _runtime_attr("bus_recall", "default_emitter")
        _BUS_EMITTER = default_emitter()
        _BUS_EMITTER_INIT = True
    return _BUS_EMITTER


def _emit_recall_bus(query: str, returned_ids: list[str]) -> None:
    """Mirror a recall onto the fleet bus (MS.1 second sink); never raise.

    Carries the recall-time facts the wire supports — the query, what came
    back, and whether the system objectively abstained (nothing returned).
    The realised ``was_used`` outcome stays in the local ledger until an
    outcome-linking wire seam exists. Opt out via
    ``REMANENTIA_RECALL_BUS_DISABLE`` (handled in ``bus_recall``).
    """
    try:
        emitter = _get_bus_emitter()
        if emitter is None:
            return
        emitter.emit(
            query, returned_claim_ids=returned_ids, was_used=False, abstained=not returned_ids
        )
    except Exception:  # pragma: no cover — telemetry must never break recall
        log.debug("Recall bus emit failed", exc_info=True)


def _get_outcome_tracker() -> Any:
    """Return the process-wide recall outcome tracker (lazy singleton).

    Watches recall→remember loop closure to auto-derive the ``was_used``
    label, so the calibration stream populates without anyone rating recalls.
    Shares the ledger's disable switch — it is the same labelling stream.
    """
    global _OUTCOME_TRACKER
    if _OUTCOME_TRACKER is None:
        RecallOutcomeTracker = _runtime_attr("recall_outcome_tracker", "RecallOutcomeTracker")
        _OUTCOME_TRACKER = RecallOutcomeTracker()
    return _OUTCOME_TRACKER


def _recall_identity() -> str:
    """The querying-agent identity, shared by the ledger and the tracker."""
    return str(_runtime_attr("recall_ledger", "_identity")())


def _log_recall(
    query: str,
    returned_ids: list[str],
    top_k: int,
    project: str,
    score: float | None = None,
) -> str | None:
    """Record a recall in the query-stream ledger (MS.1); never raise.

    The query stream is the query-weighted calibration + salience source
    the fleet-memory design needs; ``score`` is the top retrieval score,
    the conformal gate's per-recall nonconformity signal. One recall, two
    sinks: the durable local ledger and the fleet bus. The ledger is
    disabled via ``REMANENTIA_RECALL_LEDGER_DISABLE`` and the bus via
    ``REMANENTIA_RECALL_BUS_DISABLE`` — independently. Telemetry must never
    break a recall, so every failure is swallowed. Returns the ledger
    ``event_id`` (for loop-closure tracking) or ``None`` when the ledger is
    disabled or the write failed.
    """
    event_id: str | None = None
    if not os.environ.get("REMANENTIA_RECALL_LEDGER_DISABLE"):
        try:
            event_id = _get_recall_ledger().record(
                query, returned_ids, top_k=top_k, project=project, score=score
            )
        except Exception:  # pragma: no cover — telemetry must never break recall
            log.debug("Recall ledger write failed", exc_info=True)
    _emit_recall_bus(query, returned_ids)
    return event_id


def _observe_recall(event_id: str | None, returned_texts: list[str]) -> None:
    """Buffer a served recall so a later remember can close the used-loop.

    No-op when the ledger is disabled or the recall was not logged; never
    raises (telemetry must not break a recall).
    """
    if event_id is None or os.environ.get("REMANENTIA_RECALL_LEDGER_DISABLE"):
        return
    try:
        _get_outcome_tracker().observe_recall(
            event_id, _recall_identity(), returned_texts, ledger=_get_recall_ledger()
        )
    except Exception:  # pragma: no cover — telemetry must never break recall
        log.debug("Recall outcome observe failed", exc_info=True)


def _close_recall_loops(content: str) -> None:
    """Mark recent recalls this agent's new memory echoes as ``was_used``.

    Called when a memory is remembered: if its text substantially echoes a
    recently recalled memory, that recall demonstrably informed the work.
    No-op when the ledger is disabled; never raises.
    """
    if os.environ.get("REMANENTIA_RECALL_LEDGER_DISABLE"):
        return
    try:
        _get_outcome_tracker().note_text(content, _recall_identity(), _get_recall_ledger())
    except Exception:  # pragma: no cover — telemetry must never break remember
        log.debug("Recall loop-closure failed", exc_info=True)


def _get_knowledge_store() -> Any:
    """Return the process-wide knowledge-store singleton."""
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
    *,
    base: Path | None = None,
) -> str:
    """Memory recall via unified BM25 + embedding index + knowledge notes."""
    global _UNIFIED_INDEX

    workspace = (base or BASE).resolve()
    if workspace != BASE.resolve():
        return _lightweight_recall(query, top_k, base=workspace)

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
        returned_ids = [f"{r.source}:{r.name}" for r in results]
        top_score = max((r.score for r in results), default=None)
        if not results and not trigger_lines:
            _log_recall(query, returned_ids, top_k, project, top_score)
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

        event_id = _log_recall(query, returned_ids, top_k, project, top_score)
        _observe_recall(event_id, [r.snippet for r in results])
        return "\n\n".join(parts)

    except Exception:  # pragma: no cover
        return _lightweight_recall(query, top_k)


def handle_remember(
    content: str,
    memory_type: str = "context",
    project: str = "",
    trigger: str = "",
    *,
    base: Path | None = None,
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

    workspace = (base or BASE).resolve()
    default_workspace = workspace == BASE.resolve()
    traces_dir = workspace / "reasoning_traces"
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
    if default_workspace and _UNIFIED_INDEX is not None and _UNIFIED_INDEX._built:
        with _lock:
            try:
                _UNIFIED_INDEX.add_file(path)
            except Exception:  # pragma: no cover
                log.debug("Incremental unified-index update failed", exc_info=True)

    # Create knowledge note
    try:
        if default_workspace:
            ks = _get_knowledge_store()
            notes_path = None
            triggers_path = None
        else:
            KnowledgeStore = _runtime_attr("knowledge_store", "KnowledgeStore")
            ks = KnowledgeStore()
            notes_path = workspace / "memory" / "knowledge_notes.jsonl"
            triggers_path = workspace / "memory" / "triggers.jsonl"
            ks.load(notes_path=notes_path, triggers_path=triggers_path)
        ks.add_note(content, source=filename)
        if trigger:  # pragma: no cover
            ks.add_trigger(trigger, content)
        ks.save(notes_path=notes_path, triggers_path=triggers_path)
    except Exception:  # pragma: no cover
        log.debug("Knowledge note persistence failed", exc_info=True)

    # Close any recall→use loops this new memory echoes (auto-derive was_used)
    if default_workspace:
        _close_recall_loops(content)

    # Async consolidation with debounce
    if default_workspace:
        _schedule_consolidation()

    return f"Remembered: {filename} ({len(content)} chars)"


def _schedule_consolidation() -> None:
    """Run consolidation in a background thread, debounced."""
    global _consolidation_pending, _consolidation_last, _RECALL_INDEX
    now = _time.monotonic()
    if now - _consolidation_last < _CONSOLIDATION_DEBOUNCE_S:
        _consolidation_pending = True
        return
    _consolidation_last = now
    _consolidation_pending = False

    def _run() -> None:
        global _RECALL_INDEX, _consolidation_last
        try:
            consolidate = _runtime_attr("consolidation_engine", "consolidate")

            consolidate()
            _RECALL_INDEX = None
        except Exception:  # pragma: no cover
            log.debug("Background consolidation failed", exc_info=True)
        _consolidation_last = _time.monotonic()

    threading.Thread(target=_run, daemon=True).start()


_RECALL_INDEX: dict[str, tuple[set[str], str]] | None = None
_RECALL_INDEX_BASE: Path | None = None

_rust_mcp_tok: Any | None = None
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


def _build_recall_index(base: Path | None = None) -> dict[str, tuple[set[str], str]]:
    """Build in-memory token index of all traces and semantic memories.

    Called once, cached for the process lifetime.
    Uses Rust (remanentia_search) when available.
    """
    global _RECALL_INDEX, _RECALL_INDEX_BASE
    workspace = (base or BASE).resolve()
    if _RECALL_INDEX is not None and workspace == _RECALL_INDEX_BASE:
        return _RECALL_INDEX
    _RECALL_INDEX = build_recall_index(workspace, _mcp_tok)
    _RECALL_INDEX_BASE = workspace
    return _RECALL_INDEX


def _lightweight_recall(query: str, top_k: int = 3, *, base: Path | None = None) -> str:
    """Fast recall from cached in-memory index.

    First call: ~2s (reads files). Subsequent calls: <50ms.
    """
    return recall_from_index(
        query,
        top_k=top_k,
        index=_build_recall_index(base),
        tokenizer=_mcp_tok,
    )


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


def handle_graph(entity: str = "", top: int = 10, *, graph_dir: Path | None = None) -> str:
    """Entity graph query."""
    return query_graph(graph_dir or GRAPH_DIR, entity=entity, top=top)


def handle_recall_feedback(
    query: str,
    was_used: bool,
    by: str = "",
    *,
    ledger: Any | None = None,
) -> str:
    """Record whether a prior recall for *query* was used downstream.

    Attaches a ``was_used`` outcome to the most recent matching recall in the
    query-stream ledger (MS.1). A *usage* signal for cold-start calibration and
    a retrieval-precision monitor — not the safety label (that is ``was_correct``,
    set via :func:`handle_recall_correctness`). Usually populated automatically by
    recall→remember loop closure; this tool is the explicit override. Disabled
    via ``REMANENTIA_RECALL_LEDGER_DISABLE``.
    """
    if os.environ.get("REMANENTIA_RECALL_LEDGER_DISABLE"):
        return "Recall ledger disabled; feedback ignored."
    try:
        outcome_ledger = ledger if ledger is not None else _get_recall_ledger()
        event_id = outcome_ledger.latest_for(query, by=by or None)
        if event_id is None:
            return f"No prior recall found for: {query}"
        outcome_ledger.record_outcome(event_id, was_used=was_used)
        return f"Recorded was_used={was_used} for recall of: {query}"
    except Exception as e:  # pragma: no cover — telemetry must never break recall
        log.debug("Recall feedback failed", exc_info=True)
        return f"Feedback error: {e}"


def handle_recall_correctness(
    query: str,
    was_correct: bool,
    by: str = "",
    *,
    ledger: Any | None = None,
) -> str:
    """Record whether a prior recall's memories were *correct* — the gate label.

    The seam a downstream verifier wires its verdict into: an answer that used a
    recalled memory and passed verification ⇒ ``was_correct=True``; an answer
    flagged, halted, or corrected by the verifier ⇒ ``was_correct=False``. This
    is the label the conformal abstention gate calibrates on (a *correctness*
    guarantee, distinct from the usage-only ``was_used``). Attaches to the most
    recent matching recall. Disabled via ``REMANENTIA_RECALL_LEDGER_DISABLE``.
    """
    if os.environ.get("REMANENTIA_RECALL_LEDGER_DISABLE"):
        return "Recall ledger disabled; correctness ignored."
    try:
        outcome_ledger = ledger if ledger is not None else _get_recall_ledger()
        event_id = outcome_ledger.latest_for(query, by=by or None)
        if event_id is None:
            return f"No prior recall found for: {query}"
        outcome_ledger.record_outcome(event_id, was_correct=was_correct)
        return f"Recorded was_correct={was_correct} for recall of: {query}"
    except Exception as e:  # pragma: no cover — telemetry must never break recall
        log.debug("Recall correctness failed", exc_info=True)
        return f"Correctness error: {e}"


def handle_request(
    request: dict[str, Any],
    *,
    base: Path | None = None,
    ledger: Any | None = None,
) -> dict[str, Any] | None:
    """Dispatch one stdio JSON-RPC request through production tool handlers."""
    handlers = {
        "remanentia_recall": lambda args: handle_recall(
            args.get("query", ""),
            args.get("top_k", 3),
            project=args.get("project", ""),
            after=args.get("after", ""),
            before=args.get("before", ""),
            llm=args.get("llm", False),
            base=base,
        ),
        "remanentia_remember": lambda args: handle_remember(
            args.get("content", ""),
            args.get("type", "context"),
            args.get("project", ""),
            args.get("trigger", ""),
            base=base,
        ),
        "remanentia_status": lambda _args: handle_status(),
        "remanentia_graph": lambda args: handle_graph(
            args.get("entity", ""),
            args.get("top", 10),
            graph_dir=(base / "memory" / "graph") if base is not None else None,
        ),
        "remanentia_recall_feedback": lambda args: handle_recall_feedback(
            args.get("query", ""), bool(args.get("was_used", False)), ledger=ledger
        ),
        "remanentia_recall_correctness": lambda args: handle_recall_correctness(
            args.get("query", ""), bool(args.get("was_correct", False)), ledger=ledger
        ),
    }
    return dispatch_request(
        request,
        handlers=handlers,
        audit_logger=MCP_AUDIT_LOGGER,
        limiter_factory=_mcp_rate_limiter,
        rate_key=_mcp_rate_limit_key,
    )


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


def main() -> None:  # pragma: no cover
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
