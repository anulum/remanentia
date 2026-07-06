# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the reader context-window budget

from __future__ import annotations

from reader_budget import _TRUNCATION_MARKER, fit_context


class TestUnbounded:
    def test_nonpositive_budget_joins_all_verbatim(self):
        sections = ["alpha", "beta", "gamma"]
        assert fit_context(sections, 0) == "alpha\n\nbeta\n\ngamma"
        assert fit_context(sections, -1) == "alpha\n\nbeta\n\ngamma"

    def test_unbounded_skips_blank_sections(self):
        assert fit_context(["alpha", "   ", "", "beta"], 0) == "alpha\n\nbeta"


class TestWithinBudget:
    def test_all_sections_fit_returns_verbatim(self):
        sections = ["alpha", "beta"]  # 5 + 2 (sep) + 4 = 11 chars
        assert fit_context(sections, 100) == "alpha\n\nbeta"

    def test_exact_fit_boundary_keeps_everything(self):
        sections = ["alpha", "beta"]
        exact = len("alpha") + len("\n\n") + len("beta")  # 11
        assert fit_context(sections, exact) == "alpha\n\nbeta"


class TestOverflow:
    def test_high_priority_survives_low_priority_dropped(self):
        # Budget fits the first section whole and leaves 10 chars of the second
        # after the separator and the truncation marker.
        keep = "A" * 20
        drop = "B" * 100
        budget = len(keep) + len("\n\n") + len(_TRUNCATION_MARKER) + 10
        out = fit_context([keep, drop], budget)
        assert out.startswith(keep)
        assert out.endswith(_TRUNCATION_MARKER)
        assert "B" * 10 in out  # exactly the 10-char head that fit
        assert len(out) <= budget

    def test_overflowing_section_truncated_with_marker(self):
        big = "x" * 500
        out = fit_context([big], 100)
        assert out.endswith(_TRUNCATION_MARKER)
        assert len(out) <= 100
        assert out[:10] == "x" * 10

    def test_later_sections_dropped_after_truncation(self):
        out = fit_context(["A" * 40, "B" * 40, "C" * 40], 50)
        assert "C" * 5 not in out  # third section never reached
        assert out.startswith("A" * 40)

    def test_budget_too_small_even_for_marker_drops_section(self):
        # room = budget - used - sep - len(marker) <= 0 → nothing added.
        out = fit_context(["Z" * 100], len(_TRUNCATION_MARKER) - 1)
        assert out == ""

    def test_first_fits_second_cannot_even_truncate(self):
        first = "A" * 10
        second = "B" * (len(_TRUNCATION_MARKER) + 10)  # overflows and cannot truncate
        # Budget leaves exactly zero room after first + separator + marker.
        budget = len(first) + len("\n\n") + len(_TRUNCATION_MARKER)
        out = fit_context([first, second], budget)
        assert out == first


class TestBlankHandling:
    def test_blank_sections_do_not_consume_separator_or_budget(self):
        assert fit_context(["", "alpha", "  ", "beta"], 100) == "alpha\n\nbeta"

    def test_all_blank_returns_empty(self):
        assert fit_context(["", "   ", "\n"], 100) == ""
        assert fit_context([], 0) == ""
