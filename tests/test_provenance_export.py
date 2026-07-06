# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for provenance store export

from __future__ import annotations

import json

import pytest

from lineage_completeness import ProvenanceNode, is_lineage_complete, lineage_completeness
from provenance_export import (
    build_provenance_store,
    provenance_node_from_note,
    render_provenance_jsonl,
)


def _note(node_id, *, source_quality="inferred", links=None, **extra):
    note = {"id": node_id, "source_quality": source_quality}
    if links is not None:
        note["links"] = links
    note.update(extra)
    return note


class TestNodeProjection:
    def test_stated_note_is_origin(self):
        node = provenance_node_from_note(_note("a", source_quality="stated"))
        assert node == ProvenanceNode(id="a", origin=True, parent=None)

    def test_inferred_note_is_not_origin(self):
        node = provenance_node_from_note(_note("b", source_quality="inferred"))
        assert node.origin is False

    def test_corrected_note_is_not_origin(self):
        assert provenance_node_from_note(_note("c", source_quality="corrected")).origin is False

    def test_missing_source_quality_defaults_non_origin(self):
        assert provenance_node_from_note({"id": "d"}).origin is False

    def test_derived_from_becomes_parent(self):
        note = _note("child", links=[{"target": "parent", "type": "derived_from"}])
        assert provenance_node_from_note(note).parent == "parent"

    def test_first_derived_from_wins_over_later(self):
        note = _note(
            "x",
            links=[
                {"target": "p1", "type": "derived_from"},
                {"target": "p2", "type": "derived_from"},
            ],
        )
        assert provenance_node_from_note(note).parent == "p1"

    def test_non_derivation_links_ignored(self):
        note = _note(
            "y",
            links=[
                {"target": "z", "type": "related"},
                {"target": "root", "type": "derived_from"},
            ],
        )
        assert provenance_node_from_note(note).parent == "root"

    def test_no_links_has_no_parent(self):
        assert provenance_node_from_note(_note("solo")).parent is None


class TestMalformedLinks:
    def test_links_not_a_sequence(self):
        assert provenance_node_from_note(_note("a", links=42)).parent is None

    def test_link_entry_not_a_mapping_skipped(self):
        note = _note("a", links=["nope", {"target": "root", "type": "derived_from"}])
        assert provenance_node_from_note(note).parent == "root"

    def test_blank_target_skipped(self):
        note = _note(
            "a",
            links=[
                {"target": "", "type": "derived_from"},
                {"target": "real", "type": "derived_from"},
            ],
        )
        assert provenance_node_from_note(note).parent == "real"

    def test_non_string_target_skipped(self):
        note = _note("a", links=[{"target": 5, "type": "derived_from"}])
        assert provenance_node_from_note(note).parent is None


class TestBadId:
    def test_missing_id_raises(self):
        with pytest.raises(ValueError, match="non-empty string id"):
            provenance_node_from_note({"source_quality": "stated"})

    def test_blank_id_raises(self):
        with pytest.raises(ValueError, match="non-empty string id"):
            provenance_node_from_note({"id": ""})

    def test_non_string_id_raises(self):
        with pytest.raises(ValueError, match="non-empty string id"):
            provenance_node_from_note({"id": 7})


class TestBuildStore:
    def test_builds_by_id(self):
        store = build_provenance_store([_note("a", source_quality="stated"), _note("b")])
        assert set(store) == {"a", "b"}
        assert store["a"].origin is True

    def test_last_write_wins_on_repeated_id(self):
        store = build_provenance_store(
            [_note("a", source_quality="inferred"), _note("a", source_quality="stated")]
        )
        assert store["a"].origin is True

    def test_empty_notes_empty_store(self):
        assert build_provenance_store([]) == {}


class TestRenderJsonl:
    def test_sorted_and_shaped(self):
        store = {
            "b": ProvenanceNode(id="b", origin=False, parent="a"),
            "a": ProvenanceNode(id="a", origin=True, parent=None),
        }
        lines = render_provenance_jsonl(store).splitlines()
        assert [json.loads(x)["id"] for x in lines] == ["a", "b"]  # id-sorted
        assert json.loads(lines[0]) == {"id": "a", "origin": True, "parent": None}
        assert json.loads(lines[1]) == {"id": "b", "origin": False, "parent": "a"}

    def test_empty_store_empty_string(self):
        assert render_provenance_jsonl({}) == ""


class TestRoundTripToScorer:
    def test_export_loads_and_scores(self, tmp_path):
        # A two-hop chain: answer cites the child, which derives from a stated root.
        notes = [
            _note("root", source_quality="stated"),
            _note("mid", links=[{"target": "root", "type": "derived_from"}]),
            _note("leaf", links=[{"target": "mid", "type": "derived_from"}]),
            _note("orphan", source_quality="inferred"),  # no derivation → incomplete
        ]
        store = build_provenance_store(notes)
        path = tmp_path / "provenance.jsonl"
        path.write_text(render_provenance_jsonl(store), encoding="utf-8")

        # Load through the scorecard's reader (the real consumer) and score.
        from scorecard_report import load_provenance_store

        loaded = load_provenance_store(path)
        assert is_lineage_complete("leaf", loaded) is True  # leaf → mid → root(origin)
        assert is_lineage_complete("orphan", loaded) is False  # inferred, no source

        from lineage_completeness import AnswerLineage

        report = lineage_completeness(
            [
                AnswerLineage(answer_id="q1", cited_ids=("leaf",)),
                AnswerLineage(answer_id="q2", cited_ids=("orphan",)),
            ],
            loaded,
        )
        assert report.completeness == 0.5
        assert report.incomplete_answers == ("q2",)
