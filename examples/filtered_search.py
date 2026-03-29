# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: Filtered Search

"""Demonstrate search filters: project, date range, document type.

Usage::

    cd remanentia/
    python examples/filtered_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_index import MemoryIndex

idx = MemoryIndex()
if not idx.load():
    idx.build(use_gpu_embeddings=False, use_gliner=False)

# Filter by project name (matches against source path and document name)
print("=== Filter by project ===")
results = idx.search("authentication decision", top_k=5, project="director-ai")
for r in results:
    print(f"  {r.name} (score={r.score:.3f})")

# Filter by date range (YYYY-MM-DD format, matches document dates)
print("\n=== Filter by date range (after 2026-03-15) ===")
results = idx.search("retrieval scoring", top_k=5, after="2026-03-15")
for r in results:
    print(f"  {r.name} (score={r.score:.3f})")

# Filter by document type (matches against doc_type field: code, traces, sessions)
print("\n=== Filter by doc_type=code ===")
results = idx.search("BM25 scoring", top_k=5, doc_type="code")
for r in results:
    print(f"  {r.name} (score={r.score:.3f})")

# Combine filters
print("\n=== Combined: project + date ===")
results = idx.search(
    "STDP learning",
    top_k=5,
    project="remanentia",
    after="2026-03-01",
    before="2026-03-31",
)
for r in results:
    print(f"  {r.name} (score={r.score:.3f}): {r.snippet[:80]}")
