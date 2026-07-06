# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the resumable hypothesis checkpoint

"""Pin the durability + resume contract the long-run benchmark depends on.

The checkpoint only earns its place if it (a) survives a kill mid-run — every
appended record is on disk before the call returns — and (b) reloads cleanly past
the debris an interrupted write leaves: a blank line, a half-written JSON fragment,
or a line that is valid JSON but not a record. Both halves are pinned here against
a real temp file so the fsync path actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis_checkpoint import append_record, completed_ids, load_completed


class TestLoadCompleted:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_completed(tmp_path / "absent.jsonl") == []

    def test_reads_valid_records_in_order(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        p.write_text(
            '{"question_id": "a", "hypothesis": "one"}\n'
            '{"question_id": "b", "hypothesis": "two"}\n',
            encoding="utf-8",
        )
        recs = load_completed(p)
        assert [r["question_id"] for r in recs] == ["a", "b"]
        assert recs[0]["hypothesis"] == "one"

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        p.write_text(
            '{"question_id": "a", "hypothesis": "one"}\n'
            "\n"
            "   \n"
            '{"question_id": "b", "hypothesis": "two"}\n',
            encoding="utf-8",
        )
        assert completed_ids(load_completed(p)) == {"a", "b"}

    def test_partial_trailing_fragment_is_skipped(self, tmp_path: Path) -> None:
        # A write interrupted mid-record leaves an unparsable last line.
        p = tmp_path / "h.jsonl"
        p.write_text(
            '{"question_id": "a", "hypothesis": "one"}\n'
            '{"question_id": "b", "hyp',  # truncated — no newline, no close
            encoding="utf-8",
        )
        recs = load_completed(p)
        assert [r["question_id"] for r in recs] == ["a"]

    def test_valid_json_that_is_not_a_record_is_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        p.write_text(
            "[1, 2, 3]\n"  # a list — valid JSON, not a record
            "42\n"  # a number
            '{"no_id": true}\n'  # object without question_id
            '{"question_id": "keep"}\n',
            encoding="utf-8",
        )
        assert completed_ids(load_completed(p)) == {"keep"}


class TestCompletedIds:
    def test_collects_ids_as_strings(self) -> None:
        recs = [{"question_id": 7}, {"question_id": "x"}]
        assert completed_ids(recs) == {"7", "x"}

    def test_records_without_id_are_excluded(self) -> None:
        recs = [{"question_id": "a"}, {"hypothesis": "orphan"}]
        assert completed_ids(recs) == {"a"}


class TestAppendRecord:
    def test_append_creates_parent_and_round_trips(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "dir" / "h.jsonl"
        append_record(p, {"question_id": "a", "hypothesis": "α diet"})
        append_record(p, {"question_id": "b", "hypothesis": "β"})
        recs = load_completed(p)
        assert [r["question_id"] for r in recs] == ["a", "b"]
        # Unicode is preserved (ensure_ascii=False).
        assert recs[0]["hypothesis"] == "α diet"

    def test_each_append_is_one_flushed_line(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        append_record(p, {"question_id": "a"})
        # The record is on disk immediately — no close/flush of a later handle needed.
        raw = p.read_text(encoding="utf-8")
        assert raw.count("\n") == 1
        assert json.loads(raw.strip())["question_id"] == "a"
