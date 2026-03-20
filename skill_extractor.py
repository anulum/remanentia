# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Skill extraction from reasoning traces.

Analyzes traces to identify recurring patterns and distills them into
reusable skills. A skill is a structured pattern:

    {
        "name": "ci-version-mismatch-fix",
        "trigger": "CI fails due to tool version mismatch",
        "action": "Update pyproject.toml + pre-commit-config.yaml + CI matrix",
        "evidence": ["trace_a.md", "trace_b.md"],
        "frequency": 3,
        "last_seen": "2026-03-18"
    }

Usage::

    python 04_ARCANE_SAPIENCE/skill_extractor.py
    python 04_ARCANE_SAPIENCE/skill_extractor.py --list
    python 04_ARCANE_SAPIENCE/skill_extractor.py --query "CI failure"
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRACES_DIR = BASE_DIR / "reasoning_traces"
SKILLS_PATH = BASE_DIR / "snn_state" / "skills.json"

# Patterns that indicate a skill (trigger → action pairs)
_SKILL_MARKERS = [
    (r"fix(?:ed|ing)?", r"(?:update|change|modify|add|remove|replace)"),
    (r"bug|error|fail(?:ure|ed)?|broke", r"(?:fix|resolve|patch|workaround)"),
    (r"chose|decision|trade.?off", r"(?:because|reason|rationale)"),
    (r"pattern|approach|strategy", r"(?:works?|better|cleaner|faster)"),
    (r"refactor", r"(?:extract|split|merge|rename|move)"),
]


def _tokenize_lower(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def extract_skills(traces_dir: Path | None = None) -> list[dict]:
    """Extract skills from reasoning traces.

    Scans all traces for recurring trigger→action patterns. Groups
    similar patterns by keyword overlap. Returns structured skills.
    """
    tdir = traces_dir or TRACES_DIR
    if not tdir.exists():
        return []

    # Parse traces into structured entries
    entries = []
    for f in sorted(tdir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip().lstrip("- ")
            if not stripped or stripped.startswith("#"):
                continue
            # Check if this line describes a skill-like pattern
            lower = stripped.lower()
            for trigger_re, action_re in _SKILL_MARKERS:
                if re.search(trigger_re, lower) and re.search(action_re, lower):
                    entries.append({
                        "text": stripped[:200],
                        "source": f.name,
                        "tokens": set(_tokenize_lower(stripped)),
                    })
                    break

    if not entries:
        return []

    # Cluster entries by token overlap (greedy single-linkage)
    clusters: list[list[dict]] = []
    used = set()
    for i, entry in enumerate(entries):
        if i in used:
            continue
        cluster = [entry]
        used.add(i)
        for j in range(i + 1, len(entries)):
            if j in used:
                continue
            overlap = entry["tokens"] & entries[j]["tokens"]
            union = entry["tokens"] | entries[j]["tokens"]
            if len(overlap) / max(len(union), 1) > 0.3:
                cluster.append(entries[j])
                used.add(j)
        if len(cluster) >= 1:
            clusters.append(cluster)

    # Convert clusters to skills
    skills = []
    for cluster in clusters:
        # Most common tokens across the cluster = skill name
        all_tokens = Counter()
        for entry in cluster:
            all_tokens.update(entry["tokens"])
        # Filter common words
        stop = {"the", "and", "was", "for", "this", "that", "with", "from", "not", "but"}
        key_tokens = [t for t, c in all_tokens.most_common(6) if t not in stop and len(t) > 2]
        name = "-".join(key_tokens[:4]) if key_tokens else "unnamed"

        sources = sorted(set(e["source"] for e in cluster))
        representative = max(cluster, key=lambda e: len(e["text"]))

        skills.append({
            "name": name,
            "description": representative["text"],
            "evidence": sources,
            "frequency": len(cluster),
            "key_terms": key_tokens,
            "last_seen": sources[-1] if sources else "",
        })

    # Sort by frequency (most recurring = most valuable)
    skills.sort(key=lambda s: s["frequency"], reverse=True)
    return skills


def save_skills(skills: list[dict]) -> Path:
    """Persist extracted skills to JSON."""
    SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKILLS_PATH.write_text(json.dumps(skills, indent=2) + "\n")
    return SKILLS_PATH


def load_skills() -> list[dict]:
    """Load previously extracted skills."""
    if not SKILLS_PATH.exists():
        return []
    try:
        return json.loads(SKILLS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def query_skills(query: str, top_k: int = 5) -> list[dict]:
    """Find skills matching a query."""
    skills = load_skills()
    if not skills:
        return []

    q_tokens = set(_tokenize_lower(query))
    scored = []
    for skill in skills:
        s_tokens = set(skill.get("key_terms", []))
        overlap = q_tokens & s_tokens
        if overlap:
            score = len(overlap) / max(len(q_tokens), 1)
            scored.append({**skill, "match_score": round(score, 3)})

    scored.sort(key=lambda s: s["match_score"], reverse=True)
    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Extract skills from reasoning traces")
    parser.add_argument("--list", action="store_true", help="List extracted skills")
    parser.add_argument("--query", type=str, help="Query skills")
    parser.add_argument("--extract", action="store_true", help="Run extraction and save")
    args = parser.parse_args()

    if args.query:
        results = query_skills(args.query)
        if not results:
            print("No matching skills.")
            return
        for s in results:
            print(f"  [{s['match_score']:.2f}] {s['name']} (freq={s['frequency']})")
            print(f"         {s['description'][:80]}")
        return

    if args.list:
        skills = load_skills()
        if not skills:
            print("No skills extracted yet. Run with --extract first.")
            return
        for s in skills:
            print(f"  {s['name']} (freq={s['frequency']}, evidence={len(s['evidence'])} traces)")
            print(f"    {s['description'][:80]}")
        return

    # Default: extract and save
    skills = extract_skills()
    if skills:
        path = save_skills(skills)
        print(f"Extracted {len(skills)} skills -> {path}")
        for s in skills[:10]:
            print(f"  {s['name']} (freq={s['frequency']})")
    else:
        print("No skills found in traces.")


if __name__ == "__main__":
    main()
