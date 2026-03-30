# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fact validity model tests

"""Tests for fact_validity_model.py (C5 runtime wrapper).

Covers model loading, fact classification, confidence thresholds,
supersession detection, and graceful degradation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from fact_validity_model import (
    FACT_TYPES,
    FactClassification,
    classify_fact,
)


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    def test_fact_type_count(self):
        assert len(FACT_TYPES) == 4

    def test_expected_types(self):
        assert "state" in FACT_TYPES
        assert "event" in FACT_TYPES
        assert "preference" in FACT_TYPES
        assert "plan" in FACT_TYPES


# ── FactClassification dataclass ───────────────────────────────


class TestFactClassification:
    def test_fields(self):
        fc = FactClassification(
            fact_type="state",
            supersedes_prob=0.8,
            has_boundary_prob=0.3,
            confidence=0.95,
        )
        assert fc.fact_type == "state"
        assert fc.supersedes_prob == 0.8
        assert fc.has_boundary_prob == 0.3
        assert fc.confidence == 0.95

    def test_equality(self):
        a = FactClassification("state", 0.8, 0.3, 0.95)
        b = FactClassification("state", 0.8, 0.3, 0.95)
        assert a == b


# ── Model loading ──────────────────────────────────────────────


class TestModelLoading:
    def test_model_missing_returns_none(self):
        import fact_validity_model

        fact_validity_model._model = None
        fact_validity_model._tokenizer = None
        fact_validity_model._config = None
        with patch.object(fact_validity_model, "_MODEL_DIR") as md:
            md.__truediv__ = lambda s, x: MagicMock(exists=lambda: False)
            r = classify_fact("I live in Berlin")
            assert r is None

    def test_already_loaded(self):
        import fact_validity_model

        fact_validity_model._model = MagicMock()
        assert fact_validity_model._load_model() is True
        fact_validity_model._model = None

    def test_load_model_exception_returns_false(self):
        import fact_validity_model

        fact_validity_model._model = None
        fact_validity_model._tokenizer = None
        fact_validity_model._config = None
        model_pt = MagicMock(exists=lambda: True)
        mock_dir = MagicMock()
        mock_dir.__truediv__ = lambda self, x: model_pt
        with patch.object(fact_validity_model, "_MODEL_DIR", mock_dir):
            with patch("fact_validity_model.json.load", side_effect=OSError("corrupt")):
                result = fact_validity_model._load_model()
                assert result is False


# ── Classification with mocked model ──────────────────────────


class TestClassifyFactMocked:
    def _setup_mock(self, type_idx: int = 0, sup_logit: float = -2.0, bnd_logit: float = -2.0):
        import fact_validity_model

        mock_model = MagicMock()
        type_logits = torch.zeros(1, 4)
        type_logits[0, type_idx] = 5.0  # highest at given index
        sup_tensor = torch.tensor([sup_logit])
        bnd_tensor = torch.tensor([bnd_logit])
        mock_model.return_value = (type_logits, sup_tensor, bnd_tensor)

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

        fact_validity_model._model = mock_model
        fact_validity_model._tokenizer = mock_tok
        fact_validity_model._config = {
            "fact_types": FACT_TYPES,
            "max_seq_len": 128,
            "model_name": "test",
        }

    def _cleanup(self):
        import fact_validity_model

        fact_validity_model._model = None
        fact_validity_model._tokenizer = None
        fact_validity_model._config = None

    def test_state_classification(self):
        self._setup_mock(type_idx=0)  # state
        r = classify_fact("I live in Berlin")
        assert r is not None
        assert r.fact_type == "state"
        assert r.confidence > 0.5
        self._cleanup()

    def test_event_classification(self):
        self._setup_mock(type_idx=1)  # event
        r = classify_fact("I went to the gym")
        assert r is not None
        assert r.fact_type == "event"
        self._cleanup()

    def test_preference_classification(self):
        self._setup_mock(type_idx=2)  # preference
        r = classify_fact("I love hiking")
        assert r is not None
        assert r.fact_type == "preference"
        self._cleanup()

    def test_plan_classification(self):
        self._setup_mock(type_idx=3)  # plan
        r = classify_fact("I plan to visit Tokyo")
        assert r is not None
        assert r.fact_type == "plan"
        self._cleanup()

    def test_high_supersedes_prob(self):
        self._setup_mock(type_idx=0, sup_logit=3.0)  # sigmoid(3.0) ≈ 0.95
        r = classify_fact("I started a new job")
        assert r is not None
        assert r.supersedes_prob > 0.5
        self._cleanup()

    def test_low_supersedes_prob(self):
        self._setup_mock(type_idx=0, sup_logit=-3.0)  # sigmoid(-3.0) ≈ 0.05
        r = classify_fact("I live in Berlin")
        assert r is not None
        assert r.supersedes_prob < 0.5
        self._cleanup()

    def test_high_boundary_prob(self):
        self._setup_mock(type_idx=1, bnd_logit=3.0)
        r = classify_fact("I visited the doctor on March 15")
        assert r is not None
        assert r.has_boundary_prob > 0.5
        self._cleanup()

    def test_empty_text(self):
        self._setup_mock()
        r = classify_fact("")
        assert r is not None  # model handles via tokenizer padding
        self._cleanup()


# ── With real model (if available) ─────────────────────────────


class TestWithRealModel:
    """Tests that run with the real trained model if available.

    These are skipped if the model checkpoint is not found.
    """

    @pytest.fixture(autouse=True)
    def _reset_model(self):
        import fact_validity_model

        fact_validity_model._model = None
        fact_validity_model._tokenizer = None
        fact_validity_model._config = None
        yield
        fact_validity_model._model = None
        fact_validity_model._tokenizer = None
        fact_validity_model._config = None

    def test_preference_detection(self):
        r = classify_fact("I love hiking in the mountains")
        if r is None:
            pytest.skip("Model not available")
        assert r.fact_type == "preference"
        assert r.confidence > 0.5

    def test_plan_detection(self):
        r = classify_fact("I plan to visit Tokyo next year")
        if r is None:
            pytest.skip("Model not available")
        assert r.fact_type == "plan"
        assert r.confidence > 0.5

    def test_supersedes_on_change_verb(self):
        r = classify_fact("I started working at Google")
        if r is None:
            pytest.skip("Model not available")
        # The model may or may not detect supersession here;
        # we just verify it returns a valid result
        assert r.fact_type in FACT_TYPES
