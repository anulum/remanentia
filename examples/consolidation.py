# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: Consolidation Pipeline

"""Demonstrate the episodic → semantic consolidation pipeline.

Consolidation reads new reasoning traces, extracts entities and relations,
clusters by project and date, and writes semantic memory files with YAML
frontmatter. Also updates the entity graph.

Usage::

    cd remanentia/
    python examples/consolidation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consolidation_engine import consolidate, get_pending_traces

# Check for pending (unconsolidated) traces
pending = get_pending_traces()
print(f"Pending traces: {len(pending)}")
for p in pending[:5]:
    print(f"  {p}")

# Run consolidation (use force=True to reconsolidate all, not just new)
result = consolidate(force=False)

print(f"\nConsolidation result:")
if isinstance(result, dict):
    for key, value in result.items():
        print(f"  {key}: {value}")
else:
    print(f"  {result}")
