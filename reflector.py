# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Reflector (LLM-powered deep consolidation)

"""Periodic deep consolidation: summarize, link, detect gaps.

Inspired by Mastra's Reflector agent and A-MEM's self-organizing notes.
Processes recent knowledge notes and:
1. Clusters related notes
2. Generates dense summaries (LLM)
3. Generates prospective queries per note (Kumiho technique)
4. Identifies unresolved contradictions and knowledge gaps
5. Produces a human-readable digest
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent


def _cluster_notes(notes: list) -> list[list]:
    """Group notes by shared keywords/entities. Greedy overlap clustering."""
    if not notes:
        return []

    clusters: list[list[int]] = []
    assigned = set()

    for i, note in enumerate(notes):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        i_keywords = set(note.keywords) | set(note.entities)

        for j in range(i + 1, len(notes)):
            if j in assigned:  # pragma: no cover
                continue
            j_keywords = set(notes[j].keywords) | set(notes[j].entities)
            overlap = len(i_keywords & j_keywords)
            if overlap >= 2:
                cluster.append(j)
                assigned.add(j)
                i_keywords |= j_keywords

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def _generate_summary_heuristic(notes: list) -> str:
    """Generate a summary from a cluster of notes without LLM."""
    if not notes:
        return ""
    titles = [n.title for n in notes if n.title]
    entities = sorted(set(e for n in notes for e in n.entities))[:10]
    first_content = notes[0].content[:200]

    lines = [f"Summary of {len(notes)} related notes:"]
    if entities:
        lines.append(f"Entities: {', '.join(entities)}")
    for n in notes[:5]:
        lines.append(f"- {n.title}")
    lines.append(f"\nKey content: {first_content}")
    return "\n".join(lines)


def _generate_summary_llm(notes: list, model: str = "claude-haiku-4-5-20251001") -> str | None:
    """Generate a dense summary from a cluster of notes via LLM."""
    try:
        from answer_extractor import _get_client
        client = _get_client()
        if not client:
            return None
    except ImportError:  # pragma: no cover
        return None

    content = "\n\n".join(f"[{n.source}] {n.content[:300]}" for n in notes[:5])  # pragma: no cover
    try:  # pragma: no cover
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize these related knowledge notes in 2-3 sentences. "
                    "Focus on decisions, findings, and metrics. Be precise.\n\n"
                    f"{content}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:  # pragma: no cover
        return None


def _generate_prospective_queries_llm(note, model: str = "claude-haiku-4-5-20251001") -> list[str]:  # pragma: no cover
    """Generate hypothetical future queries for a note via LLM (Kumiho technique)."""
    try:
        from answer_extractor import _get_client
        client = _get_client()
        if not client:
            return []
    except ImportError:
        return []

    try:
        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Generate 3 short search queries someone might use to find this information. "
                    "One per line, no numbering.\n\n"
                    f"{note.content[:500]}"
                ),
            }],
        )
        text = response.content[0].text.strip()
        return [q.strip() for q in text.split("\n") if q.strip() and len(q.strip()) > 5][:5]
    except Exception:
        return []


def _identify_gaps(notes: list) -> list[str]:
    """Identify knowledge gaps — decisions without measured outcomes,
    findings without follow-up, open questions."""
    gaps = []
    decisions = [n for n in notes if any(w in n.content.lower()
                 for w in ("decided", "chose", "will"))]
    findings = [n for n in notes if any(w in n.content.lower()
                for w in ("found", "measured", "scored"))]

    decision_entities = set(e for n in decisions for e in n.entities)
    finding_entities = set(e for n in findings for e in n.entities)

    # Decisions without corresponding findings
    unmeasured = decision_entities - finding_entities
    for entity in sorted(unmeasured)[:5]:
        gaps.append(f"Decision about '{entity}' has no measured outcome")

    return gaps


def _identify_contradictions(notes: list) -> list[str]:
    """List unresolved contradictions."""
    contradictions = []
    for n in notes:
        if n.supersedes and not n.superseded_by:
            contradictions.append(
                f"'{n.title}' supersedes an older note (may need verification)")
    return contradictions


def reflect_once(days: int = 7, use_llm: bool = False) -> dict:
    """Run one reflection cycle over recent knowledge notes.

    Returns: summary notes created, gaps found, contradictions, digest text.
    """
    try:
        from knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        if not store.load():
            return {"status": "no_notes", "notes": 0}
    except Exception:  # pragma: no cover
        return {"status": "error", "notes": 0}

    # Filter to recent notes
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    recent = [n for n in store.notes.values()
              if n.created >= cutoff or n.updated >= cutoff]

    if not recent:
        return {"status": "nothing_recent", "notes": 0, "days": days}

    # Cluster related notes
    clusters = _cluster_notes(recent)

    # Generate summaries
    summaries = []
    for cluster_indices in clusters:
        cluster_notes = [recent[i] for i in cluster_indices]
        if use_llm:  # pragma: no cover
            summary = _generate_summary_llm(cluster_notes)
            if not summary:
                summary = _generate_summary_heuristic(cluster_notes)
        else:
            summary = _generate_summary_heuristic(cluster_notes)
        summaries.append({
            "notes": len(cluster_notes),
            "entities": sorted(set(e for n in cluster_notes for e in n.entities))[:10],
            "summary": summary,
        })

    # Generate prospective queries for un-queried notes
    pq_count = 0
    if use_llm:  # pragma: no cover
        for note in recent[:20]:
            queries = _generate_prospective_queries_llm(note)
            if queries:
                note.keywords = sorted(set(note.keywords + queries))[:20]
                pq_count += len(queries)

    # Identify gaps and contradictions
    gaps = _identify_gaps(recent)
    contradictions = _identify_contradictions(recent)

    # Build digest
    digest_lines = [
        f"# Reflection Digest ({time.strftime('%Y-%m-%d %H:%M')})",
        f"\nProcessed {len(recent)} notes from last {days} days.",
        f"Formed {len(clusters)} clusters.",
    ]

    if summaries:
        digest_lines.append("\n## Cluster Summaries")
        for i, s in enumerate(summaries):
            digest_lines.append(f"\n### Cluster {i+1} ({s['notes']} notes)")
            if s["entities"]:
                digest_lines.append(f"Entities: {', '.join(s['entities'])}")
            digest_lines.append(s["summary"])

    if gaps:
        digest_lines.append("\n## Knowledge Gaps")
        for g in gaps:
            digest_lines.append(f"- {g}")

    if contradictions:
        digest_lines.append("\n## Unresolved Contradictions")
        for c in contradictions:
            digest_lines.append(f"- {c}")

    digest_lines.append(f"\n## Stats")
    digest_lines.append(f"- Notes processed: {len(recent)}")
    digest_lines.append(f"- Clusters: {len(clusters)}")
    digest_lines.append(f"- Summaries: {len(summaries)}")
    digest_lines.append(f"- Prospective queries generated: {pq_count}")
    digest_lines.append(f"- Knowledge gaps: {len(gaps)}")
    digest_lines.append(f"- Contradictions: {len(contradictions)}")

    digest = "\n".join(digest_lines)

    # Save digest
    digest_dir = BASE / "memory" / "digests"
    digest_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digest_dir / f"digest_{time.strftime('%Y-%m-%d')}.md"
    digest_path.write_text(digest, encoding="utf-8")

    # Save updated store (prospective queries may have been added)
    if pq_count > 0:  # pragma: no cover
        store.save()

    return {
        "status": "ok",
        "notes": len(recent),
        "clusters": len(clusters),
        "summaries": len(summaries),
        "prospective_queries": pq_count,
        "gaps": len(gaps),
        "contradictions": len(contradictions),
        "digest_path": str(digest_path),
        "digest": digest,
    }
