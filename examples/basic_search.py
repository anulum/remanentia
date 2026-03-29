# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: Basic Search

"""Basic search against the BM25 index.

Builds (or loads) the index, runs a query, prints results with answers.

Usage::

    cd remanentia/
    python examples/basic_search.py
"""

import sys
from pathlib import Path

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_index import MemoryIndex

# Build or load the index (no GPU embeddings, no GLiNER — fast mode)
idx = MemoryIndex()
if not idx.load():
    print("No cached index found. Building from sources...")
    stats = idx.build(use_gpu_embeddings=False, use_gliner=False)
    print(f"Built: {stats.get('documents', 0)} docs, {stats.get('paragraphs', 0)} paragraphs")
    idx.save()
else:
    print(f"Loaded cached index: {len(idx.documents)} docs, {len(idx.paragraph_index)} paragraphs")

# Search
query = "what did we decide about retrieval"
results = idx.search(query, top_k=5)

print(f"\nQuery: {query}")
print(f"Results: {len(results)}")
print("-" * 60)

for i, r in enumerate(results):
    print(f"\n[{i + 1}] {r.name} (score={r.score:.4f}, source={r.source})")
    if r.answer:
        print(f"    Answer: {r.answer}")
    print(f"    {r.snippet[:200]}")
