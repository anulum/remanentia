# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for lineage_completeness

from __future__ import annotations

from lineage_completeness import (
    AnswerLineage,
    ProvenanceNode,
    answer_provenance_visible,
    is_lineage_complete,
    lineage_completeness,
)

# A sound store: c2 -> c1 -> w0 (origin). orphan has no parent and is not origin.
_STORE = {
    "w0": ProvenanceNode(id="w0", origin=True),
    "c1": ProvenanceNode(id="c1", parent="w0"),
    "c2": ProvenanceNode(id="c2", parent="c1"),
    "orphan": ProvenanceNode(id="orphan"),  # no origin, no parent
    "dangling": ProvenanceNode(id="dangling", parent="ghost"),  # parent missing
    "loopA": ProvenanceNode(id="loopA", parent="loopB"),
    "loopB": ProvenanceNode(id="loopB", parent="loopA"),
}


class TestIsLineageComplete:
    def test_reaches_origin(self) -> None:
        assert is_lineage_complete("c2", _STORE) is True
        assert is_lineage_complete("w0", _STORE) is True

    def test_missing_node_dangling(self) -> None:
        assert is_lineage_complete("ghost", _STORE) is False
        assert is_lineage_complete("dangling", _STORE) is False

    def test_chain_ends_without_origin(self) -> None:
        assert is_lineage_complete("orphan", _STORE) is False

    def test_cycle(self) -> None:
        assert is_lineage_complete("loopA", _STORE) is False


class TestAnswerProvenanceVisible:
    def test_no_citations(self) -> None:
        assert answer_provenance_visible(AnswerLineage("a", ()), _STORE) is False

    def test_all_complete(self) -> None:
        assert answer_provenance_visible(AnswerLineage("a", ("c2", "c1")), _STORE) is True

    def test_one_incomplete_fails(self) -> None:
        assert answer_provenance_visible(AnswerLineage("a", ("c1", "orphan")), _STORE) is False


class TestLineageCompleteness:
    def test_mixed(self) -> None:
        answers = [
            AnswerLineage("good", ("c2",)),
            AnswerLineage("bad-orphan", ("orphan",)),
            AnswerLineage("bad-empty", ()),
        ]
        report = lineage_completeness(answers, _STORE)
        assert report.total == 3
        assert report.visible == 1
        assert abs(report.completeness - 1 / 3) < 1e-9
        assert set(report.incomplete_answers) == {"bad-orphan", "bad-empty"}

    def test_empty_is_vacuously_complete(self) -> None:
        report = lineage_completeness([], _STORE)
        assert report.total == 0
        assert report.completeness == 1.0
        assert report.incomplete_answers == ()
