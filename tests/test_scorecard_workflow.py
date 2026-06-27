# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for Scorecard workflow visibility handling

"""Tests for the OpenSSF Scorecard workflow contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORECARD_WORKFLOW = ROOT / ".github" / "workflows" / "scorecard.yml"


def test_scorecard_analysis_skips_private_repositories() -> None:
    """Scorecard should not fail private pushes on GitHub token scope limits."""
    text = SCORECARD_WORKFLOW.read_text(encoding="utf-8")

    assert "github.event.repository.private == false" in text
