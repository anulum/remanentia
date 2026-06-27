# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — claim-axis schema export

"""Export the Remanentia claim-axis contract as deterministic JSON Schema.

The in-process claim vocabulary lives in :mod:`claim_axes`; this module turns it
into a stable artefact that external consumers can use without importing
Remanentia Python code. The schema is deliberately additive: finding records may
carry extra fields, but the evidence/status/admission/freshness axes and the
falsified/reference-validated rejection invariant stay machine-readable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claim_axes import (
    ADMISSIONS,
    CLAIM_STATUSES,
    EVIDENCE_KINDS,
    FALSIFIED,
    FRESHNESSES,
    REFERENCE_VALIDATED,
)
from recall_gate import BOUNDARY, REFUTED, VALIDATED

SCHEMA_VERSION = "2026-06-27"
"""Version of the exported claim-axis schema contract."""

SCHEMA_PATH = Path("docs/schema/remanentia_claim_axes.schema.json")
"""Default committed schema artefact path."""


def build_claim_schema() -> dict[str, object]:
    """Return the JSON Schema describing Remanentia claim axes.

    Returns
    -------
    dict[str, object]
        Draft 2020-12 JSON Schema with sorted axis enumerations and
        Remanentia-specific extension fields for render modes and invariants.
    """
    evidence_kinds = sorted(EVIDENCE_KINDS)
    claim_statuses = sorted(CLAIM_STATUSES)
    admissions = sorted(ADMISSIONS)
    freshnesses = sorted(FRESHNESSES)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://remanentia.com/schemas/claim_axes.schema.json",
        "title": "Remanentia Claim Axes",
        "description": (
            "Shared evidence, claim-status, admission, and freshness axes for "
            "Remanentia findings and recall rendering."
        ),
        "type": "object",
        "additionalProperties": True,
        "required": ["evidence_kind", "claim_status"],
        "properties": {
            "evidence_kind": {
                "type": "string",
                "enum": evidence_kinds,
                "description": "Evidence basis for the finding.",
            },
            "claim_status": {
                "type": "string",
                "enum": claim_statuses,
                "description": "Claim-boundary status after validation or review.",
            },
            "admission": {
                "type": "string",
                "enum": admissions,
                "description": "Admission-boundary value used by recall rendering.",
            },
            "freshness": {
                "type": "string",
                "enum": freshnesses,
                "description": "Source-freshness value for validation rendering.",
            },
        },
        "allOf": [
            {
                "not": {
                    "properties": {
                        "evidence_kind": {"const": FALSIFIED},
                        "claim_status": {"const": REFERENCE_VALIDATED},
                    },
                    "required": ["evidence_kind", "claim_status"],
                }
            }
        ],
        "x-remanentia-schema-version": SCHEMA_VERSION,
        "x-remanentia-axis-sets": {
            "evidence_kind": evidence_kinds,
            "claim_status": claim_statuses,
            "admission": admissions,
            "freshness": freshnesses,
        },
        "x-remanentia-render-modes": sorted((VALIDATED, BOUNDARY, REFUTED)),
        "x-remanentia-invariants": [
            {
                "evidence_kind": FALSIFIED,
                "claim_status": REFERENCE_VALIDATED,
                "action": "reject-before-persistence",
            }
        ],
    }


def render_claim_schema() -> str:
    """Return the deterministic JSON representation of the claim schema.

    Returns
    -------
    str
        Pretty-printed JSON with sorted keys and a trailing newline, suitable
        for a committed generated artefact.
    """
    return json.dumps(build_claim_schema(), indent=2, sort_keys=True) + "\n"


def write_claim_schema(output: str | Path = SCHEMA_PATH) -> Path:
    """Write the claim schema artefact.

    Parameters
    ----------
    output
        Destination path for the schema JSON.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_claim_schema(), encoding="utf-8")
    return path


def schema_is_current(path: str | Path = SCHEMA_PATH) -> bool:
    """Return whether *path* already contains the generated schema text.

    Parameters
    ----------
    path
        Schema file to compare against the current code contract.

    Returns
    -------
    bool
        ``True`` when the file exists and matches the deterministic renderer.
    """
    candidate = Path(path)
    try:
        return candidate.read_text(encoding="utf-8") == render_claim_schema()
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    """Run the claim-schema export command.

    Parameters
    ----------
    argv
        Optional argument vector without the executable name.

    Returns
    -------
    int
        Process status code. ``0`` means the schema was written or verified;
        ``1`` means ``--check`` found a missing or stale schema artefact.
    """
    parser = argparse.ArgumentParser(description="Export the Remanentia claim-axis JSON Schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCHEMA_PATH,
        help="Destination schema JSON path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the schema file is missing or stale",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if bool(args.check):
        if schema_is_current(output):
            print(f"Claim-axis schema is up to date: {output}")
            return 0
        print(f"Claim-axis schema is not up to date: {output}", file=sys.stderr)
        return 1
    write_claim_schema(output)
    print(f"Wrote claim-axis schema: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
