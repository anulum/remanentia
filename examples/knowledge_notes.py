# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: Knowledge Store

"""Demonstrate the Zettelkasten-style knowledge store.

Shows: adding notes, searching, graph traversal, contradiction detection,
prospective queries, and persistence.

Usage::

    cd remanentia/
    python examples/knowledge_notes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_store import KnowledgeStore

ks = KnowledgeStore()

# Add notes — entities, keywords, and prospective queries are auto-extracted
n1 = ks.add_note(
    "BM25 retrieval accuracy is 81.2% on the LOCOMO benchmark. "
    "The main bottleneck is temporal questions at 42.7%.",
    source="session_log.md",
)
print(f"Note 1: id={n1.id}, entities={n1.entities}, type={n1.title}")
print(f"  Prospective queries: {n1.prospective_queries[:3]}")

n2 = ks.add_note(
    "Switched from binary TF to real term frequency in BM25 scoring. "
    "This improved LOCOMO multi-hop from 75% to 82.6%.",
    source="decision_log.md",
)
print(f"\nNote 2: id={n2.id}, entities={n2.entities}")

# Notes sharing entities are automatically linked
print(f"\nNote 1 links: {len(n1.links)}")
for link in n1.links:
    print(f"  -> {link['target']} ({link['type']})")

# Keyword search
print("\n=== Keyword search ===")
results = ks.search("LOCOMO accuracy", top_k=3)
for note, score in results:
    print(f"  {note.id} (score={score:.1f}): {note.content[:80]}")

# Graph search — follows links to find related notes
print("\n=== Graph search (2-hop) ===")
related = ks.graph_search("BM25 scoring", top_k=5, hop_depth=2)
for note in related:
    print(f"  {note.id}: {note.content[:80]}")

# Contradiction detection — add a note that contradicts an existing one
n3 = ks.add_note(
    "BM25 retrieval accuracy dropped to 65.0% after the index rebuild.",
    source="regression_log.md",
)
if n3.supersedes:
    print(f"\nContradiction detected: {n3.id} supersedes {n3.supersedes}")

# Prospective triggers — fire automatically on matching queries
trigger = ks.add_trigger(
    condition="authentication",
    action="We switched from JWT to session tokens on 2026-03-15",
)
print(f"\nTrigger added: '{trigger.condition}' -> '{trigger.action[:50]}'")

matched = ks.check_triggers("what authentication method do we use")
for t in matched:
    print(f"  Triggered: {t.action}")

# Stats
stats = ks.stats
print(f"\nStore stats: {stats}")
