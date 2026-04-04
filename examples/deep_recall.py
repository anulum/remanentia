# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: deep recall

"""Demonstrate the full recall pipeline: BM25 retrieval + entity graph +
temporal context + cross-project search, all combined into a MemoryContext.

Usage::

    cd remanentia/
    python examples/deep_recall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_recall import recall

# recall() is the main entry point — combines all retrieval channels
ctx = recall("what did we decide about BM25 scoring", top_k=3)

print(f"Query: {ctx.query}")
print(f"Elapsed: {ctx.elapsed_ms:.0f}ms")
print(f"Sources consulted: {ctx.sources_consulted}")
print()

# Primary match (best trace or document)
if ctx.trace:
    print(f"Primary match: {ctx.trace} (score={ctx.trace_score:.3f})")
    print(f"  Snippet: {ctx.trace_snippet[:200]}")
else:
    print("No primary match found.")

# Semantic memories (consolidated facts)
if ctx.semantic_memories:
    print(f"\nSemantic memories: {len(ctx.semantic_memories)}")
    for sm in ctx.semantic_memories[:3]:
        print(f"  {sm.get('path', 'unknown')}: {sm.get('key_point', '')[:100]}")

# Entity graph context
if ctx.entities:
    print(f"\nEntities: {ctx.entities}")
if ctx.related_entities:
    print(f"Related: {[r.get('id', '') for r in ctx.related_entities[:5]]}")

# Temporal context
if ctx.before:
    print(f"\nBefore: {ctx.before[:3]}")
if ctx.after:
    print(f"After: {ctx.after[:3]}")

# Cross-project links
if ctx.cross_project:
    print(f"\nCross-project: {[c.get('project', '') for c in ctx.cross_project]}")

# Novelty (how surprising this query is relative to stored patterns)
print(f"\nNovelty score: {ctx.novelty_score:.3f}")

# The summary property gives a text overview
print(f"\n{'=' * 60}")
print(ctx.summary)

# For LLM integration, use to_llm_context()
llm_ctx = ctx.to_llm_context()
print(f"\nLLM context length: {len(llm_ctx)} chars")
