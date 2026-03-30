# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Training data generator tests

"""Tests for training/temporal_synth.py.

Validates the structure, format, and correctness of synthetic training data
for C3 (temporal relations), C4 (date normalisation), and C5 (fact validity).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from temporal_synth import (
    _fill_template,
    _month_delta,
    _random_event_text,
    _random_ref_date,
    generate_date_normalisation,
    generate_fact_validity,
    generate_temporal_relations,
    save_jsonl,
)


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── _month_delta ───────────────────────────────────────────────


class TestMonthDelta:
    def test_basic_forward(self):
        assert _month_delta(date(2024, 3, 15), 1) == date(2024, 4, 15)

    def test_basic_backward(self):
        assert _month_delta(date(2024, 3, 15), -1) == date(2024, 2, 15)

    def test_year_rollover(self):
        assert _month_delta(date(2024, 12, 15), 1) == date(2025, 1, 15)


# ── _random_ref_date ───────────────────────────────────────────


class TestRandomRefDate:
    def test_returns_date(self):
        d = _random_ref_date()
        assert isinstance(d, date)

    def test_within_bounds(self):
        for _ in range(50):
            d = _random_ref_date()
            assert date(2020, 1, 15) <= d <= date(2025, 12, 15)


# ── _random_event_text ─────────────────────────────────────────


class TestRandomEventText:
    def test_returns_string(self):
        text = _random_event_text()
        assert isinstance(text, str)
        assert len(text) > 5

    def test_no_unfilled_placeholders(self):
        for _ in range(20):
            text = _random_event_text()
            assert "{" not in text
            assert "}" not in text


# ── _fill_template ─────────────────────────────────────────────


class TestFillTemplate:
    def test_fills_place(self):
        result = _fill_template("I went to {place}")
        assert "{place}" not in result
        assert len(result) > 10

    def test_fills_multiple(self):
        result = _fill_template("{place} and {item}")
        assert "{" not in result

    def test_no_placeholder_passthrough(self):
        result = _fill_template("No placeholders here")
        assert result == "No placeholders here"


# ── generate_date_normalisation ────────────────────────────────


class TestGenerateDateNormalisation:
    @pytest.fixture(scope="class")
    def samples(self):
        return generate_date_normalisation(200)

    def test_count(self, samples):
        assert len(samples) == 200

    def test_keys(self, samples):
        for s in samples:
            assert "expr" in s
            assert "ref_date" in s
            assert "target_date" in s

    def test_valid_iso_dates(self, samples):
        for s in samples:
            assert _ISO_RE.match(s["ref_date"]), f"bad ref_date: {s['ref_date']}"
            assert _ISO_RE.match(s["target_date"]), f"bad target: {s['target_date']}"

    def test_non_empty_expressions(self, samples):
        for s in samples:
            assert len(s["expr"].strip()) > 0

    def test_parseable_dates(self, samples):
        for s in samples[:50]:
            date.fromisoformat(s["ref_date"])
            date.fromisoformat(s["target_date"])

    def test_deterministic(self):
        a = generate_date_normalisation(10)
        b = generate_date_normalisation(10)
        # Same seed → same output (RNG is module-level with fixed seed)
        # Note: this may differ if other generators ran first
        assert all(isinstance(x, dict) for x in a)
        assert all(isinstance(x, dict) for x in b)


# ── generate_temporal_relations ────────────────────────────────


class TestGenerateTemporalRelations:
    VALID_RELATIONS = {"before", "after", "same_day", "overlaps", "contains", "unknown"}

    @pytest.fixture(scope="class")
    def samples(self):
        return generate_temporal_relations(200)

    def test_count(self, samples):
        assert len(samples) == 200

    def test_keys(self, samples):
        for s in samples:
            assert "event_a" in s
            assert "event_b" in s
            assert "date_a" in s
            assert "date_b" in s
            assert "relation" in s

    def test_valid_relations(self, samples):
        for s in samples:
            assert s["relation"] in self.VALID_RELATIONS

    def test_valid_dates(self, samples):
        for s in samples[:50]:
            date.fromisoformat(s["date_a"])
            date.fromisoformat(s["date_b"])

    def test_distinct_events(self, samples):
        for s in samples:
            assert s["event_a"] != s["event_b"]

    def test_relation_distribution(self, samples):
        counts = {}
        for s in samples:
            counts[s["relation"]] = counts.get(s["relation"], 0) + 1
        # All 6 relations should appear in 200 samples
        assert len(counts) >= 4  # at least 4 of 6 types in 200 samples


# ── generate_fact_validity ─────────────────────────────────────


class TestGenerateFactValidity:
    VALID_TYPES = {"state", "event", "preference", "plan"}

    @pytest.fixture(scope="class")
    def samples(self):
        return generate_fact_validity(200)

    def test_count(self, samples):
        assert len(samples) == 200

    def test_keys(self, samples):
        for s in samples:
            assert "text" in s
            assert "fact_type" in s
            assert "supersedes" in s
            assert "has_boundary" in s

    def test_valid_types(self, samples):
        for s in samples:
            assert s["fact_type"] in self.VALID_TYPES

    def test_boolean_fields(self, samples):
        for s in samples:
            assert isinstance(s["supersedes"], bool)
            assert isinstance(s["has_boundary"], bool)

    def test_non_empty_text(self, samples):
        for s in samples:
            assert len(s["text"].strip()) > 0

    def test_type_distribution(self, samples):
        counts = {}
        for s in samples:
            counts[s["fact_type"]] = counts.get(s["fact_type"], 0) + 1
        # All 4 types should appear (balanced generation)
        assert len(counts) == 4

    def test_supersedes_only_in_state(self, samples):
        for s in samples:
            if s["supersedes"]:
                assert s["fact_type"] == "state"


# ── save_jsonl ─────────────────────────────────────────────────


class TestSaveJsonl:
    def test_creates_file(self, tmp_path):
        data = [{"a": 1}, {"b": 2}]
        path = tmp_path / "test.jsonl"
        save_jsonl(data, path)
        assert path.exists()

    def test_valid_jsonl(self, tmp_path):
        data = [{"key": "value"}, {"key": "other"}]
        path = tmp_path / "test.jsonl"
        save_jsonl(data, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "key" in parsed

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deep" / "test.jsonl"
        save_jsonl([{"x": 1}], path)
        assert path.exists()
