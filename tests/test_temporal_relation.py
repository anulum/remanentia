# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal relation classifier tests

"""Tests for temporal_relation.py (C3 runtime wrapper).

Covers model loading, relation classification, event ordering,
confidence thresholds, and graceful degradation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from temporal_relation import (
    LABELS,
    RelationResult,
    classify_relation,
    order_events,
)


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    def test_label_count(self):
        assert len(LABELS) == 6

    def test_expected_labels(self):
        assert "before" in LABELS
        assert "after" in LABELS
        assert "same_day" in LABELS
        assert "overlaps" in LABELS
        assert "contains" in LABELS
        assert "unknown" in LABELS


# ── RelationResult dataclass ───────────────────────────────────


class TestRelationResult:
    def test_fields(self):
        r = RelationResult(
            relation="before",
            confidence=0.9,
            probabilities={"before": 0.9, "after": 0.05, "same_day": 0.05},
        )
        assert r.relation == "before"
        assert r.confidence == 0.9
        assert "before" in r.probabilities

    def test_equality(self):
        a = RelationResult(relation="before", confidence=0.9, probabilities={})
        b = RelationResult(relation="before", confidence=0.9, probabilities={})
        assert a == b


# ── Model loading ──────────────────────────────────────────────


class TestModelLoading:
    def test_model_missing_returns_none(self):
        import temporal_relation

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None
        with patch.object(temporal_relation, "_MODEL_DIR") as md:
            md.__truediv__ = lambda s, x: MagicMock(exists=lambda: False)
            r = classify_relation("event A", "event B")
            assert r is None

    def test_already_loaded(self):
        import temporal_relation

        temporal_relation._model = MagicMock()
        temporal_relation._tokenizer = MagicMock()
        temporal_relation._config = {"labels": LABELS, "max_seq_len": 128}
        assert temporal_relation._load_model() is True
        # cleanup
        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None

    def test_load_model_exception_returns_false(self):
        import temporal_relation

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None
        model_pt = MagicMock(exists=lambda: True)
        mock_dir = MagicMock()
        mock_dir.__truediv__ = lambda self, x: model_pt
        with patch.object(temporal_relation, "_MODEL_DIR", mock_dir):
            with patch("temporal_relation.json.load", side_effect=OSError("corrupt")):
                result = temporal_relation._load_model()
                assert result is False


# ── Classification with mocked model ──────────────────────────


class TestClassifyRelationMocked:
    def _setup_mock(self):
        import temporal_relation

        mock_model = MagicMock()
        # Return logits where "before" (idx 0) has highest score
        logits = torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.0]])
        mock_model.return_value = logits

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        temporal_relation._model = mock_model
        temporal_relation._tokenizer = mock_tok
        temporal_relation._config = {
            "labels": LABELS,
            "max_seq_len": 128,
            "model_name": "test",
            "num_classes": 6,
        }

    def _cleanup(self):
        import temporal_relation

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None

    def test_returns_highest_prob_label(self):
        self._setup_mock()
        r = classify_relation("I started yoga", "I visited the dentist")
        assert r is not None
        assert r.relation == "before"
        assert r.confidence > 0.5
        self._cleanup()

    def test_probabilities_sum_to_one(self):
        self._setup_mock()
        r = classify_relation("event A", "event B")
        assert r is not None
        total = sum(r.probabilities.values())
        assert abs(total - 1.0) < 0.01
        self._cleanup()

    def test_all_labels_in_probabilities(self):
        self._setup_mock()
        r = classify_relation("event A", "event B")
        assert r is not None
        for label in LABELS:
            assert label in r.probabilities
        self._cleanup()

    def test_empty_event_strings(self):
        self._setup_mock()
        r = classify_relation("", "")
        assert r is not None  # model handles empty input via tokenizer padding
        self._cleanup()


# ── Event ordering ─────────────────────────────────────────────


class TestOrderEvents:
    def test_single_event_unchanged(self):
        events = [("I went to the gym", "2023-04-10")]
        result = order_events(events)
        assert len(result) == 1
        assert result[0][0] == "I went to the gym"
        assert result[0][2] == ""  # no relation for first

    def test_empty_list(self):
        assert order_events([]) == []

    def test_model_unavailable_preserves_order(self):
        import temporal_relation

        temporal_relation._model = None
        events = [("A", ""), ("B", ""), ("C", "")]
        result = order_events(events)
        assert len(result) == 3
        assert result[0][0] == "A"
        assert result[1][0] == "B"
        assert result[2][0] == "C"

    def test_two_events_before_relation(self):
        import temporal_relation

        # Mock model: "before" (idx 0) highest
        mock_model = MagicMock()
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        mock_model.return_value = logits

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        temporal_relation._model = mock_model
        temporal_relation._tokenizer = mock_tok
        temporal_relation._config = {
            "labels": LABELS,
            "max_seq_len": 128,
            "model_name": "test",
            "num_classes": 6,
        }

        events = [("First event", ""), ("Second event", "")]
        result = order_events(events)
        assert len(result) == 2

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None

    def test_two_events_after_relation(self):
        """Exercises the 'after' branch (lines 145-146)."""
        import temporal_relation

        # Mock model: "after" (idx 1) highest
        mock_model = MagicMock()
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]])
        mock_model.return_value = logits

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        temporal_relation._model = mock_model
        temporal_relation._tokenizer = mock_tok
        temporal_relation._config = {
            "labels": LABELS,
            "max_seq_len": 128,
            "model_name": "test",
            "num_classes": 6,
        }

        events = [("Later event", ""), ("Earlier event", "")]
        result = order_events(events)
        assert len(result) == 2
        # "after" means event[0] is after event[1], so event[1] gets before_count
        # Earlier event should come first in ordering
        assert result[0][0] == "Earlier event"

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None

    def test_three_events_pairwise(self):
        """Tests N > 2 pairwise comparison."""
        import temporal_relation

        mock_model = MagicMock()
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        mock_model.return_value = logits

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        temporal_relation._model = mock_model
        temporal_relation._tokenizer = mock_tok
        temporal_relation._config = {
            "labels": LABELS,
            "max_seq_len": 128,
            "model_name": "test",
            "num_classes": 6,
        }

        events = [("A", ""), ("B", ""), ("C", "")]
        result = order_events(events)
        assert len(result) == 3

        temporal_relation._model = None
        temporal_relation._tokenizer = None
        temporal_relation._config = None
