# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Observer (filesystem watcher → knowledge notes)

"""Watch project directories for changes, auto-create knowledge notes.

Inspired by Mastra's Observational Memory (94.87% LongMemEval):
background agents watch conversations and maintain a dense observation log.
Remanentia's Observer does the same for filesystem artifacts.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

BASE = Path(__file__).parent
GOTM_ROOT = BASE.parent
STATE_PATH = BASE / "memory" / "observer_state.json"

WATCHED_DIRS = {
    "traces": BASE / "reasoning_traces",
    "semantic": BASE / "memory" / "semantic",
    "sessions": GOTM_ROOT / ".coordination" / "sessions" / "arcane-sapience",
    "handovers": GOTM_ROOT / ".coordination" / "handovers" / "arcane-sapience",
}

# Paragraph must contain one of these to become a knowledge note
_SIGNAL_WORDS = {
    "decided", "decision", "found", "finding", "result", "measured",
    "fixed", "broke", "shipped", "released", "removed", "added",
    "accuracy", "percent", "score", "benchmark", "p@1",
    "because", "root cause", "confirmed", "rejected", "chose",
    "version", "v0.", "v1.", "v2.", "v3.",
    "critical", "important", "key insight", "conclusion",
}


class ObserverState:
    """Track which files have been processed and their last modification time."""

    def __init__(self):
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

    def mark_processed(self, path: Path):
        try:
            self.processed[str(path)] = path.stat().st_mtime
        except OSError:  # pragma: no cover
            pass

    def save(self, path: Path | None = None):
        path = path or STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.processed, indent=2), encoding="utf-8")

    def load(self, path: Path | None = None) -> bool:
        path = path or STATE_PATH
        if not path.exists():
            return False
        try:
            self.processed = json.loads(path.read_text(encoding="utf-8"))
            return True
        except (json.JSONDecodeError, OSError):  # pragma: no cover
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


def extract_notes_from_file(path: Path) -> list[dict]:
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
    notes = []
    for para in paragraphs:
        notes.append({"content": para, "source": path.name})
    return notes


def observe_once(state: ObserverState,
                 watched_dirs: dict[str, Path] | None = None) -> dict:
    """Scan all watched directories, create notes for new/changed files.

    Returns stats: files_scanned, files_new, notes_created.
    """
    dirs = watched_dirs or WATCHED_DIRS
    files_scanned = 0
    files_new = 0
    notes_created = 0

    try:
        from knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.load()
    except Exception:  # pragma: no cover
        return {"files_scanned": 0, "files_new": 0, "notes_created": 0, "error": "store load failed"}

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
                store.add_note(nd["content"], source=nd["source"])
                notes_created += 1
            state.mark_processed(f)

    if notes_created > 0:
        store.save()

    return {
        "files_scanned": files_scanned,
        "files_new": files_new,
        "notes_created": notes_created,
    }


def observe_loop(interval: int = 30,
                 watched_dirs: dict[str, Path] | None = None):  # pragma: no cover
    """Poll watched directories forever, creating notes on changes."""
    state = ObserverState()
    state.load()
    print(f"Observer started. Watching {len(watched_dirs or WATCHED_DIRS)} directories every {interval}s.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            result = observe_once(state, watched_dirs)
            if result["files_new"] > 0:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Scanned {result['files_scanned']} files, "
                      f"{result['files_new']} new, "
                      f"{result['notes_created']} notes created")
                state.save()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nObserver stopped.")
        state.save()
