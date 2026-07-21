# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Observer (filesystem watcher)

"""Watch project directories for changes, auto-create knowledge notes.

Inspired by Mastra's Observational Memory (94.87% LongMemEval) and
OpenClaw's Heartbeat component: background agents watch conversations,
maintain a dense observation log, and perform scheduled maintenance.

Remanentia's Observer does the same for filesystem artifacts, with
added heartbeat capabilities:
- Periodic consolidation of episodic traces
- Memory aging (lifecycle state transitions)
- Capacity monitoring with consolidation pressure
- Proactive gap detection (stale topic coverage)
"""

from __future__ import annotations

import json
import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from store_paths import default_base

BASE = default_base()
COORDINATION_ROOT = Path(os.environ.get("REMANENTIA_COORDINATION_ROOT", BASE / ".coordination"))
STATE_PATH = BASE / "memory" / "observer_state.json"
_LOGGER = logging.getLogger(__name__)

WATCHED_DIRS = {
    "traces": BASE / "reasoning_traces",
    "semantic": BASE / "memory" / "semantic",
    "sessions": COORDINATION_ROOT / "sessions" / "remanentia",
    "handovers": COORDINATION_ROOT / "handovers" / "remanentia",
}

# Paragraph must contain one of these to become a knowledge note
_SIGNAL_WORDS = {
    "decided",
    "decision",
    "found",
    "finding",
    "result",
    "measured",
    "fixed",
    "broke",
    "shipped",
    "released",
    "removed",
    "added",
    "accuracy",
    "percent",
    "score",
    "benchmark",
    "p@1",
    "because",
    "root cause",
    "confirmed",
    "rejected",
    "chose",
    "version",
    "v0.",
    "v1.",
    "v2.",
    "v3.",
    "critical",
    "important",
    "key insight",
    "conclusion",
}


class ObserverState:
    """Track which files have been processed and their last modification time."""

    def __init__(self) -> None:
        self.processed: dict[str, float] = {}  # path → mtime

    def is_new_or_changed(self, path: Path) -> bool:
        key = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:  # pragma: no cover
            return False
        if key not in self.processed or self.processed[key] < mtime:
            return True
        return False

    def mark_processed(self, path: Path) -> None:
        try:
            self.processed[str(path)] = path.stat().st_mtime
        except OSError:  # pragma: no cover
            pass

    def save(self, path: Path | None = None) -> None:
        path = path or STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.processed, indent=2), encoding="utf-8")

    def load(self, path: Path | None = None) -> bool:
        path = path or STATE_PATH
        if not path.exists():
            return False
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return False
            self.processed = {str(key): float(value) for key, value in loaded.items()}
            return True
        except (TypeError, ValueError, json.JSONDecodeError, OSError):  # pragma: no cover
            return False


def _has_signal(text: str) -> bool:
    """Check if text contains substantive content worth noting."""
    t = text.lower()
    return any(w in t for w in _SIGNAL_WORDS)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, filter for substantial content."""
    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if len(stripped) > 50 and _has_signal(stripped):
            paragraphs.append(stripped)
    return paragraphs


def extract_notes_from_file(path: Path) -> list[dict[str, str]]:
    """Extract knowledge note candidates from a file.

    Returns list of dicts with 'content' and 'source' keys.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover
        return []

    if len(text) < 50:
        return []

    paragraphs = _split_into_paragraphs(text)
    notes: list[dict[str, str]] = []
    for para in paragraphs:
        notes.append({"content": para, "source": path.name})
    return notes


def observe_once(
    state: ObserverState,
    watched_dirs: dict[str, Path] | None = None,
    *,
    notes_path: Path | None = None,
    triggers_path: Path | None = None,
    unified_index: Any | None = None,
) -> dict[str, Any]:
    """Scan all watched directories, create notes for new/changed files.

    Optional paths isolate the production note store, while ``unified_index``
    selects an already-built index for incremental updates. Returns scan and
    persistence counts.
    """
    dirs = watched_dirs or WATCHED_DIRS
    files_scanned = 0
    files_new = 0
    notes_created = 0

    new_files: list[tuple[Path, str]] = []
    pending_notes: list[dict[str, str]] = []
    for source_name, source_dir in dirs.items():
        if not source_dir.exists():
            continue
        for f in sorted(source_dir.rglob("*.md")):
            files_scanned += 1
            if not state.is_new_or_changed(f):
                continue
            files_new += 1
            note_dicts = extract_notes_from_file(f)
            for nd in note_dicts:
                pending_notes.append(nd)
            new_files.append((f, source_name))
            # The cursor (mark_processed) is deliberately NOT advanced here.
            # A file must not be recorded as observed until its extracted notes
            # are durably in the store; otherwise a store failure below drops
            # the notes while the caller persists a cursor that has skipped
            # them — silent, permanent knowledge loss. Marking is deferred to
            # after the commit.

    if pending_notes:
        try:
            knowledge_module = import_module("knowledge_store")
            knowledge_store_cls = cast(Any, knowledge_module).KnowledgeStore
            store = knowledge_store_cls()
            store.load(notes_path, triggers_path)
        except Exception:
            # Store unavailable: leave every scanned-new file un-marked so the
            # next cycle re-observes and retries (at-least-once), rather than
            # losing the notes.
            return {
                "files_scanned": files_scanned,
                "files_new": files_new,
                "notes_created": 0,
                "error": "store load failed",
            }

        for nd in pending_notes:
            store.add_note(nd["content"], source=nd["source"])
        notes_created = len(pending_notes)
        store.save(notes_path, triggers_path)

    # Notes are now durably stored (or there were none) — safe to advance the
    # cursor over every scanned-new file.
    for f, _source_name in new_files:
        state.mark_processed(f)

    # Incrementally update the unified index for new files
    if new_files:
        try:
            mcp_module = import_module("mcp_server")
            active_index = unified_index or getattr(mcp_module, "_UNIFIED_INDEX", None)
            if active_index is not None and getattr(active_index, "_built", False):
                for f, source_name in new_files:
                    active_index.add_file(f, source=source_name)
        except Exception:  # pragma: no cover
            _LOGGER.debug("Unified index incremental update failed", exc_info=True)

    return {
        "files_scanned": files_scanned,
        "files_new": files_new,
        "notes_created": notes_created,
    }


def heartbeat(
    state: ObserverState,
    watched_dirs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Run one heartbeat cycle: observe + consolidate + age + capacity check.

    This is the autonomous maintenance tick inspired by OpenClaw's Heartbeat.
    Returns combined stats from all sub-operations.
    """
    result: dict[str, Any] = {"observe": {}, "consolidate": {}, "aging": {}, "capacity": {}}

    # 1. Observe filesystem changes
    result["observe"] = observe_once(state, watched_dirs)
    if watched_dirs is not None and result["observe"].get("files_scanned", 0) == 0:
        result["consolidate"] = {"status": "nothing_to_consolidate", "pending": 0}
        result["aging"] = {"status": "skipped", "reason": "no_files_scanned"}
        result["capacity"] = {"status": "skipped", "reason": "no_files_scanned"}
        return result

    # 2. Consolidate any pending traces
    try:
        from consolidation_engine import consolidate, get_pending_traces

        pending = get_pending_traces()
        if pending:
            result["consolidate"] = consolidate()
        else:
            result["consolidate"] = {"status": "nothing_to_consolidate", "pending": 0}
    except Exception as exc:  # pragma: no cover
        result["consolidate"] = {"error": str(exc)}  # pragma: no cover

    # 3. Age memories (lifecycle transitions)
    try:
        from consolidation_engine import age_memories

        result["aging"] = age_memories()
    except Exception as exc:  # pragma: no cover
        result["aging"] = {"error": str(exc)}  # pragma: no cover

    # 4. Capacity monitoring
    try:
        from consolidation_engine import capacity_report

        report = capacity_report()
        over_capacity = {k: v for k, v in report.items() if v.get("needs_consolidation")}
        result["capacity"] = {
            "categories_checked": len(report),
            "categories_over_threshold": len(over_capacity),
            "over_capacity": list(over_capacity.keys()),
        }
    except Exception as exc:  # pragma: no cover
        result["capacity"] = {"error": str(exc)}  # pragma: no cover

    return result


def observe_loop(
    interval: int = 30, watched_dirs: dict[str, Path] | None = None
) -> None:  # pragma: no cover
    """Poll watched directories forever, creating notes on changes."""
    state = ObserverState()
    state.load()
    print(
        f"Observer started. Watching {len(watched_dirs or WATCHED_DIRS)} directories every {interval}s."
    )
    print("Press Ctrl+C to stop.\n")

    heartbeat_every = 10  # run full heartbeat every N cycles (~5 min at 30s)
    cycle = 0
    try:
        while True:
            cycle += 1
            if cycle % heartbeat_every == 0:
                # Full heartbeat: observe + consolidate + age + capacity
                result = heartbeat(state, watched_dirs)
                obs = result["observe"]
                cons = result["consolidate"]
                cap = result["capacity"]
                print(
                    f"[{time.strftime('%H:%M:%S')}] HEARTBEAT: "
                    f"files_new={obs.get('files_new', 0)}, "
                    f"consolidated={cons.get('memories_written', 0)}, "
                    f"over_capacity={cap.get('categories_over_threshold', 0)}"
                )
                state.save()
            else:
                # Quick observe only
                result = observe_once(state, watched_dirs)
                if result["files_new"] > 0:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Scanned {result['files_scanned']} files, "
                        f"{result['files_new']} new, "
                        f"{result['notes_created']} notes created"
                    )
                    state.save()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nObserver stopped.")
        state.save()
