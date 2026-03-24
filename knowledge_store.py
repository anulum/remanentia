# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Knowledge Store (A-MEM Zettelkasten)

"""Interconnected atomic knowledge notes with self-organizing links.

Inspired by A-MEM (NeurIPS 2025) and the Zettelkasten method.
Each note is an atomic fact/decision/finding with bi-directional links
to related notes. New notes trigger updates to existing notes.
Contradictions are detected and tracked, never silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).parent
STORE_PATH = BASE / "memory" / "knowledge_notes.jsonl"
TRIGGERS_PATH = BASE / "memory" / "triggers.jsonl"

# Contradiction signal words
_POSITIVE_ACTIONS = {"added", "enabled", "started", "created", "fixed", "improved", "shipped", "works"}
_NEGATIVE_ACTIONS = {"removed", "disabled", "killed", "deleted", "broke", "failed", "dead", "dropped"}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))


def _note_id(content: str, source: str) -> str:
    return hashlib.md5(f"{content[:200]}:{source}".encode()).hexdigest()[:12]


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]{4,}", text.lower())
    # Frequency-based: keep tokens appearing 2+ times, or capitalized terms
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    keywords = [t for t, c in freq.items() if c >= 2]
    # Add capitalized terms (likely names/concepts)
    caps = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)*", text)
    keywords.extend(c.lower() for c in caps if len(c) > 3)
    # Add version numbers
    keywords.extend(re.findall(r"v\d+\.\d+(?:\.\d+)?", text))
    return sorted(set(keywords))[:20]


def _extract_entities(text: str) -> set[str]:
    entities = set()
    text_lower = text.lower()
    known = [
        "stdp", "bm25", "lif", "snn", "embedding", "pytorch", "cuda", "gpu",
        "locomo", "remanentia", "director-ai", "sc-neurocore", "scpn",
        "consolidation", "retrieval", "daemon", "mcp", "fastapi",
    ]
    for k in known:
        if k in text_lower:
            entities.add(k)
    for m in re.finditer(r"v\d+\.\d+(?:\.\d+)?", text):
        entities.add(m.group())
    for m in re.finditer(r"\d+\.?\d*%", text):
        entities.add(m.group())
    return entities


@dataclass
class KnowledgeNote:
    id: str
    title: str
    content: str
    keywords: list[str]
    source: str
    created: str
    updated: str
    links: list[dict] = field(default_factory=list)
    supersedes: str = ""
    superseded_by: str = ""
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "content": self.content,
            "keywords": self.keywords, "source": self.source,
            "created": self.created, "updated": self.updated,
            "links": self.links, "supersedes": self.supersedes,
            "superseded_by": self.superseded_by, "entities": self.entities,
        }

    @staticmethod
    def from_dict(d: dict) -> KnowledgeNote:
        return KnowledgeNote(
            id=d["id"], title=d["title"], content=d["content"],
            keywords=d.get("keywords", []), source=d.get("source", ""),
            created=d.get("created", ""), updated=d.get("updated", ""),
            links=d.get("links", []), supersedes=d.get("supersedes", ""),
            superseded_by=d.get("superseded_by", ""),
            entities=d.get("entities", []),
        )


@dataclass
class Trigger:
    id: str
    condition: str
    action: str
    created: str
    fired: list[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id, "condition": self.condition, "action": self.action,
            "created": self.created, "fired": self.fired, "active": self.active,
        }

    @staticmethod
    def from_dict(d: dict) -> Trigger:
        return Trigger(
            id=d["id"], condition=d["condition"], action=d["action"],
            created=d.get("created", ""), fired=d.get("fired", []),
            active=d.get("active", True),
        )


class KnowledgeStore:
    def __init__(self):
        self.notes: dict[str, KnowledgeNote] = {}
        self.triggers: list[Trigger] = []
        self._token_index: dict[str, set[str]] = {}  # note_id -> tokens

    def add_note(self, content: str, source: str = "",
                 title: str = "", keywords: list[str] | None = None) -> KnowledgeNote:
        """Create a knowledge note, link to related notes, detect contradictions.

        If content is near-duplicate of existing note (similarity > 0.9), merges instead.
        If content contradicts existing note (same entity + opposite action), tracks supersession.
        """
        now = time.strftime("%Y-%m-%dT%H%M", time.gmtime())
        if not title:
            title = content.split("\n")[0].strip().lstrip("#- ").strip()[:80]
        if not keywords:
            keywords = _extract_keywords(content)
        entities = sorted(_extract_entities(content))

        nid = _note_id(content, source)
        note_tokens = _tokenize(content)

        # Check for near-duplicate (merge if similarity > 0.9)
        best_sim = 0.0
        best_match_id = ""
        for eid, etokens in self._token_index.items():
            if not etokens or not note_tokens:  # pragma: no cover
                continue
            overlap = len(note_tokens & etokens)
            union = len(note_tokens | etokens)
            sim = overlap / union if union > 0 else 0
            if sim > best_sim:
                best_sim = sim
                best_match_id = eid

        if best_sim > 0.9 and best_match_id in self.notes:
            existing = self.notes[best_match_id]
            existing.updated = now
            # Append new content if it adds information
            if len(content) > len(existing.content):  # pragma: no cover
                existing.content = content
                existing.title = title
            existing.keywords = sorted(set(existing.keywords + keywords))[:20]
            existing.entities = sorted(set(existing.entities + entities))
            self._token_index[best_match_id] = note_tokens | self._token_index.get(best_match_id, set())
            return existing

        # Check for contradictions
        contradiction_id = self._detect_contradiction(content, entities)

        note = KnowledgeNote(
            id=nid, title=title, content=content, keywords=keywords,
            source=source, created=now, updated=now, entities=entities,
        )

        if contradiction_id and contradiction_id in self.notes:
            old = self.notes[contradiction_id]
            old.superseded_by = nid
            note.supersedes = contradiction_id
            note.links.append({"target": contradiction_id, "type": "supersedes"})
            old.links.append({"target": nid, "type": "superseded_by"})

        # Link to top-3 most similar existing notes
        similarities = []
        for eid, etokens in self._token_index.items():
            if eid == nid:  # pragma: no cover
                continue
            if not etokens or not note_tokens:  # pragma: no cover
                continue
            overlap = len(note_tokens & etokens)
            union = len(note_tokens | etokens)
            sim = overlap / union if union > 0 else 0
            if sim > 0.1:
                similarities.append((eid, sim))
        similarities.sort(key=lambda x: -x[1])

        for linked_id, sim in similarities[:3]:
            if linked_id == contradiction_id:
                continue
            note.links.append({"target": linked_id, "type": "related", "similarity": round(sim, 3)})
            if linked_id in self.notes:
                self.notes[linked_id].links.append(
                    {"target": nid, "type": "related", "similarity": round(sim, 3)})
                self.notes[linked_id].updated = now

        self.notes[nid] = note
        self._token_index[nid] = note_tokens
        return note

    def _detect_contradiction(self, content: str, entities: list[str]) -> str | None:
        """Check if new content contradicts any existing note.

        Contradiction signals:
        - Same entity + opposite action verbs
        - Same metric entity + different numeric value
        """
        content_lower = content.lower()
        content_actions = _POSITIVE_ACTIONS & _tokenize(content)
        content_neg_actions = _NEGATIVE_ACTIONS & _tokenize(content)
        content_entities = set(entities)

        for nid, note in self.notes.items():
            if note.superseded_by:  # pragma: no cover
                continue
            note_entities = set(note.entities)
            shared_entities = content_entities & note_entities
            if not shared_entities:
                continue

            note_lower = note.content.lower()
            note_actions = _POSITIVE_ACTIONS & _tokenize(note.content)
            note_neg_actions = _NEGATIVE_ACTIONS & _tokenize(note.content)

            # Opposite actions on shared entities
            if (content_actions and note_neg_actions) or (content_neg_actions and note_actions):
                return nid

            # Different percentage for same context
            content_pcts = set(re.findall(r"\d+\.?\d*%", content))
            note_pcts = set(re.findall(r"\d+\.?\d*%", note.content))
            if content_pcts and note_pcts and content_pcts != note_pcts:
                # Only if they share meaningful entities (not just "%" in both)
                if len(shared_entities) >= 2:
                    return nid

        return None

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeNote]:
        """BM25-lite search over knowledge notes."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for nid, tokens in self._token_index.items():
            overlap = len(q_tokens & tokens)
            if overlap > 0:
                score = overlap / max(len(q_tokens), 1)
                scored.append((nid, score))

        scored.sort(key=lambda x: -x[1])
        results = []
        for nid, _ in scored[:top_k]:
            if nid in self.notes:
                results.append(self.notes[nid])
        return results

    def get_related(self, note_id: str, depth: int = 1) -> list[KnowledgeNote]:
        """Get notes connected to the given note, up to depth hops."""
        if note_id not in self.notes:
            return []

        visited = {note_id}
        frontier = [note_id]
        result = []

        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                if nid not in self.notes:  # pragma: no cover
                    continue
                for link in self.notes[nid].links:
                    target = link["target"]
                    if target not in visited and target in self.notes:
                        visited.add(target)
                        next_frontier.append(target)
                        result.append(self.notes[target])
            frontier = next_frontier

        return result

    def get_contradictions(self) -> list[tuple[KnowledgeNote, KnowledgeNote]]:
        """Return all active contradiction pairs (old, new)."""
        pairs = []
        for note in self.notes.values():
            if note.supersedes and note.supersedes in self.notes:
                old = self.notes[note.supersedes]
                pairs.append((old, note))
        return pairs

    def check_triggers(self, query: str) -> list[Trigger]:
        """Check which triggers match the given query context."""
        q_tokens = _tokenize(query)
        matched = []
        for trigger in self.triggers:
            if not trigger.active:
                continue
            cond_tokens = _tokenize(trigger.condition)
            if not cond_tokens:  # pragma: no cover
                continue
            overlap = len(q_tokens & cond_tokens)
            if overlap / max(len(cond_tokens), 1) > 0.5:
                trigger.fired.append(time.strftime("%Y-%m-%dT%H%M", time.gmtime()))
                matched.append(trigger)
        return matched

    def add_trigger(self, condition: str, action: str) -> Trigger:
        """Add a prospective memory trigger."""
        tid = hashlib.md5(f"{condition}:{action}".encode()).hexdigest()[:12]
        trigger = Trigger(
            id=tid, condition=condition, action=action,
            created=time.strftime("%Y-%m-%dT%H%M", time.gmtime()),
        )
        self.triggers.append(trigger)
        return trigger

    def save(self, notes_path: Path | None = None, triggers_path: Path | None = None):
        notes_path = notes_path or STORE_PATH
        triggers_path = triggers_path or TRIGGERS_PATH
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(n.to_dict()) for n in self.notes.values()]
        notes_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        tlines = [json.dumps(t.to_dict()) for t in self.triggers]
        triggers_path.write_text("\n".join(tlines) + "\n" if tlines else "", encoding="utf-8")

    def load(self, notes_path: Path | None = None, triggers_path: Path | None = None) -> bool:
        notes_path = notes_path or STORE_PATH
        triggers_path = triggers_path or TRIGGERS_PATH
        self.notes = {}
        self._token_index = {}
        self.triggers = []

        if notes_path.exists():
            for line in notes_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                note = KnowledgeNote.from_dict(json.loads(line))
                self.notes[note.id] = note
                self._token_index[note.id] = _tokenize(note.content)

        if triggers_path.exists():
            for line in triggers_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                self.triggers.append(Trigger.from_dict(json.loads(line)))

        return len(self.notes) > 0 or len(self.triggers) > 0

    @property
    def stats(self) -> dict:
        n_links = sum(len(n.links) for n in self.notes.values())
        n_contradictions = sum(1 for n in self.notes.values() if n.supersedes)
        n_active_triggers = sum(1 for t in self.triggers if t.active)
        return {
            "notes": len(self.notes),
            "links": n_links,
            "contradictions": n_contradictions,
            "triggers_total": len(self.triggers),
            "triggers_active": n_active_triggers,
            "keywords": len(set(k for n in self.notes.values() for k in n.keywords)),
        }
