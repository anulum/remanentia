# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: Temporal Queries

"""Demonstrate the temporal graph: event extraction, date queries,
ordering, and code-based date arithmetic.

Usage::

    cd remanentia/
    python examples/temporal_queries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporal_graph import TemporalGraph, TemporalEvent, temporal_code_execute

tg = TemporalGraph()

# Build a graph from document text
documents = [
    ("decision.md", "We decided on 2026-03-15 to remove SNN from retrieval scoring."),
    ("benchmark.md", "LOCOMO scored 81.2% on 2026-03-23. Temporal accuracy was 42.7%."),
    ("release.md", "Released v0.3.0 on 2026-03-20 with the new BM25 engine."),
    ("meeting.md", "Team review scheduled for March 28, 2026 at the office."),
]
tg.build_from_documents(documents)

print(f"Events: {tg.stats['events']}")
print(f"Unique dates: {tg.stats['unique_dates']}")
print(f"Date range: {tg.stats['date_range']}")

# Query: temporal keyword matching
print("\n=== 'When was the release?' ===")
results = tg.query_temporal("when was the release?", top_k=3)
for ev in results:
    print(f"  {ev.date}: {ev.text[:80]} (source={ev.source})")

# Query: most recent event
print("\n=== 'Latest event' ===")
results = tg.query_temporal("latest event", top_k=1)
for ev in results:
    print(f"  {ev.date}: {ev.text[:80]}")

# Query: events before a date
print("\n=== 'What happened before 2026-03-20?' ===")
results = tg.query_temporal("what happened before 2026-03-20", top_k=5)
for ev in results:
    print(f"  {ev.date}: {ev.text[:80]}")

# Date arithmetic via temporal_code_execute
events = [
    TemporalEvent(date="2026-03-15", text="SNN removal decision", source="a.md"),
    TemporalEvent(date="2026-03-23", text="LOCOMO benchmark run", source="b.md"),
]

print("\n=== 'How many days between events?' ===")
answer = temporal_code_execute("how many days between events", events)
if answer:
    print(f"  {answer}")

print("\n=== 'Did removal happen before or after benchmark?' ===")
answer = temporal_code_execute("did removal happen before or after benchmark", events)
if answer:
    print(f"  {answer}")

# Persistence
save_path = Path("temporal_demo.jsonl")
tg.save(save_path)
print(f"\nSaved temporal graph to {save_path}")

tg2 = TemporalGraph()
tg2.load(save_path)
print(f"Loaded: {tg2.stats['events']} events")
save_path.unlink()
