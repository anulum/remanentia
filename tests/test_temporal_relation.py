# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for temporal relation classifier

"""Tests for temporal_relation.py (C3 runtime wrapper).

Covers model loading, relation classification, event ordering,
confidence thresholds, and graceful degradation.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import temporal_relation
from temporal_relation import (
    LABELS,
    RelationResult,
    classify_relation,
    order_events,
)


def _reset_model_state() -> None:
    """Clear the module-level lazy model cache."""
    temporal_relation._model = None
    temporal_relation._tokenizer = None
    temporal_relation._config = None


def _install_mock_classifier(logits: torch.Tensor) -> None:
    """Install a callable classifier and tokenizer into the runtime cache."""
    mock_model = MagicMock()
    mock_model.return_value = logits

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

    temporal_relation._model = mock_model
    temporal_relation._tokenizer = mock_tokenizer
    temporal_relation._config = {
        "labels": LABELS,
        "max_seq_len": 128,
        "model_name": "test",
        "num_classes": 6,
    }


class _FakeScalar:
    """Minimal scalar object exposing the tensor ``item`` protocol."""

    def __init__(self, value: float) -> None:
        """Store the scalar value returned by ``item``."""
        self._value = value

    def item(self) -> float:
        """Return the scalar value."""
        return self._value


class _FakeProbabilities:
    """Softmax-like probability vector for fake model inference."""

    def __init__(self) -> None:
        """Build a deterministic distribution with ``before`` as top label."""
        self._values = [0.70, 0.10, 0.08, 0.05, 0.04, 0.03]

    def squeeze(self, dim: int) -> "_FakeProbabilities":
        """Return self for the single-batch squeeze used by inference."""
        assert dim == 0
        return self

    def __getitem__(self, index: int) -> _FakeScalar:
        """Return a fake scalar probability by index."""
        return _FakeScalar(self._values[index])

    def argmax(self) -> _FakeScalar:
        """Return index zero as the highest-probability class."""
        return _FakeScalar(0)


class _FakeNoGrad:
    """Context manager matching ``torch.no_grad``."""

    def __enter__(self) -> None:
        """Enter the fake no-grad context."""

    def __exit__(self, *_exc: object) -> None:
        """Exit the fake no-grad context."""


class _FakeHiddenState:
    """Hidden-state container supporting CLS slicing."""

    def __getitem__(self, _key: object) -> object:
        """Return a fake CLS embedding."""
        return object()


class _FakeBackbone:
    """Transformer backbone replacement with the config used by the wrapper."""

    config = SimpleNamespace(hidden_size=4)

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        """Return a fake last hidden state for classifier forwarding."""
        assert input_ids is not None
        assert attention_mask is not None
        return SimpleNamespace(last_hidden_state=_FakeHiddenState())


class _FakeModule:
    """Small stand-in for ``torch.nn.Module``."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Delegate calls to the subclass ``forward`` method."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Require subclasses to provide a forward method."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: object) -> object:
        """Accept fake checkpoint state."""
        return state_dict

    def eval(self) -> object:
        """Return self after switching to inference mode."""
        return self


class _FakeSequential:
    """Classification head replacement returning deterministic logits."""

    def __init__(self, *_layers: object) -> None:
        """Accept the layers constructed by the wrapper."""

    def __call__(self, _cls_embedding: object) -> object:
        """Return fake logits."""
        return object()


class _FakeAutoModel:
    """Replacement for ``transformers.AutoModel``."""

    @staticmethod
    def from_pretrained(model_name: str) -> _FakeBackbone:
        """Return a fake backbone for the configured model name."""
        assert model_name == "fake-bert"
        return _FakeBackbone()


class _FakeTokenizer:
    """Tokenizer replacement returning tensor-like input mappings."""

    def __call__(
        self,
        event_a: str,
        event_b: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, object]:
        """Return the input mapping consumed by ``classify_relation``."""
        assert event_a
        assert event_b
        assert max_length == 64
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        return {"input_ids": object(), "attention_mask": object()}


class _FakeAutoTokenizer:
    """Replacement for ``transformers.AutoTokenizer``."""

    @staticmethod
    def from_pretrained(model_dir: str) -> _FakeTokenizer:
        """Return a fake tokenizer for the model directory."""
        assert model_dir
        return _FakeTokenizer()


def _install_fake_training_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake torch and transformers modules for model-load tests."""
    fake_torch = ModuleType("torch")
    fake_nn = ModuleType("torch.nn")
    fake_transformers = ModuleType("transformers")
    fake_torch_dynamic = cast(Any, fake_torch)
    fake_nn_dynamic = cast(Any, fake_nn)
    fake_transformers_dynamic = cast(Any, fake_transformers)

    def fake_layer(*_args: object, **_kwargs: object) -> object:
        """Return a fake neural-network layer."""
        return object()

    def fake_relu() -> object:
        """Return a fake activation layer."""
        return object()

    def fake_dropout(_rate: object) -> object:
        """Return a fake dropout layer."""
        return object()

    def fake_load(*_args: object, **_kwargs: object) -> dict[str, str]:
        """Return fake checkpoint weights."""
        return {"weights": "ok"}

    def fake_softmax(_logits: object, *, dim: int) -> _FakeProbabilities:
        """Return deterministic fake softmax probabilities."""
        assert dim == -1
        return _FakeProbabilities()

    fake_nn_dynamic.Module = _FakeModule
    fake_nn_dynamic.Sequential = _FakeSequential
    fake_nn_dynamic.Linear = fake_layer
    fake_nn_dynamic.ReLU = fake_relu
    fake_nn_dynamic.Dropout = fake_dropout

    fake_torch_dynamic.nn = fake_nn
    fake_torch_dynamic.load = fake_load
    fake_torch_dynamic.no_grad = _FakeNoGrad
    fake_torch_dynamic.softmax = fake_softmax

    fake_transformers_dynamic.AutoModel = _FakeAutoModel
    fake_transformers_dynamic.AutoTokenizer = _FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    """Validate exported temporal relation constants."""

    def test_label_count(self) -> None:
        """Expose six temporal relation labels."""
        assert len(LABELS) == 6

    def test_expected_labels(self) -> None:
        """Expose the expected C3 temporal relation vocabulary."""
        assert "before" in LABELS
        assert "after" in LABELS
        assert "same_day" in LABELS
        assert "overlaps" in LABELS
        assert "contains" in LABELS
        assert "unknown" in LABELS


# ── RelationResult dataclass ───────────────────────────────────


class TestRelationResult:
    """Validate the relation result data contract."""

    def test_fields(self) -> None:
        """Store relation, confidence, and per-label probabilities."""
        r = RelationResult(
            relation="before",
            confidence=0.9,
            probabilities={"before": 0.9, "after": 0.05, "same_day": 0.05},
        )
        assert r.relation == "before"
        assert r.confidence == 0.9
        assert "before" in r.probabilities

    def test_equality(self) -> None:
        """Compare dataclass instances by value."""
        a = RelationResult(relation="before", confidence=0.9, probabilities={})
        b = RelationResult(relation="before", confidence=0.9, probabilities={})
        assert a == b


class TestConfigHelpers:
    """Validate model configuration normalization helpers."""

    def test_config_model_name_rejects_blank_value(self) -> None:
        """Reject missing model names before external model loading."""
        with pytest.raises(ValueError, match="model_name"):
            temporal_relation._config_model_name({"model_name": ""})

    def test_config_labels_accepts_tuple_labels(self) -> None:
        """Accept tuple-backed labels from typed callers."""
        assert temporal_relation._config_labels({"labels": tuple(LABELS)}) == tuple(LABELS)

    def test_config_labels_falls_back_for_invalid_labels(self) -> None:
        """Use default labels when config labels are not all strings."""
        assert temporal_relation._config_labels({"labels": ["before", 7]}) == LABELS


# ── Model loading ──────────────────────────────────────────────


class TestModelLoading:
    """Validate lazy model-loading behavior."""

    def test_model_missing_returns_none(self, tmp_path: Path) -> None:
        """Return ``None`` when the trained checkpoint is absent."""
        _reset_model_state()
        with patch.object(temporal_relation, "_MODEL_DIR", tmp_path):
            r = classify_relation("event A", "event B")
            assert r is None

    def test_already_loaded(self) -> None:
        """Treat the cached model as ready without reloading files."""
        _reset_model_state()
        temporal_relation._model = MagicMock()
        temporal_relation._tokenizer = MagicMock()
        temporal_relation._config = {"labels": LABELS, "max_seq_len": 128}
        assert temporal_relation._load_model() is True
        _reset_model_state()

    def test_load_model_exception_returns_false(self, tmp_path: Path) -> None:
        """Return ``False`` when model metadata cannot be read."""
        _reset_model_state()
        (tmp_path / "model.pt").write_bytes(b"not-a-real-checkpoint")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        with patch.object(temporal_relation, "_MODEL_DIR", tmp_path):
            with patch("temporal_relation.json.load", side_effect=OSError("corrupt")):
                result = temporal_relation._load_model()
                assert result is False

    def test_load_model_with_fake_external_modules(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Load through the real wrapper path using typed fake dependencies."""
        _reset_model_state()
        _install_fake_training_modules(monkeypatch)
        (tmp_path / "model.pt").write_bytes(b"fake-checkpoint")
        (tmp_path / "config.json").write_text(
            '{"model_name": "fake-bert", "num_classes": 6, '
            '"max_seq_len": 64, "labels": ["before", "after", "same_day", '
            '"overlaps", "contains", "unknown"]}',
            encoding="utf-8",
        )

        with patch.object(temporal_relation, "_MODEL_DIR", tmp_path):
            assert temporal_relation._load_model() is True
            result = classify_relation("event A", "event B")

        assert result is not None
        assert result.relation == "before"
        assert result.confidence == 0.70
        _reset_model_state()

    def test_load_model_rejects_non_object_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject model config JSON that is not an object."""
        _reset_model_state()
        _install_fake_training_modules(monkeypatch)
        (tmp_path / "model.pt").write_bytes(b"fake-checkpoint")
        (tmp_path / "config.json").write_text("[]", encoding="utf-8")
        with patch.object(temporal_relation, "_MODEL_DIR", tmp_path):
            assert temporal_relation._load_model() is False

    def test_classify_returns_none_for_incomplete_cache(self) -> None:
        """Return ``None`` if the lazy-load success state is inconsistent."""
        _reset_model_state()
        with patch("temporal_relation._load_model", return_value=True):
            assert classify_relation("event A", "event B") is None


# ── Classification with mocked model ──────────────────────────


class TestClassifyRelationMocked:
    """Validate classification through the public inference function."""

    def test_returns_highest_prob_label(self) -> None:
        """Return the label with the highest softmax probability."""
        _install_mock_classifier(torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.0]]))
        r = classify_relation("I started yoga", "I visited the dentist")
        assert r is not None
        assert r.relation == "before"
        assert r.confidence > 0.5
        _reset_model_state()

    def test_probabilities_sum_to_one(self) -> None:
        """Return a normalized probability distribution."""
        _install_mock_classifier(torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.0]]))
        r = classify_relation("event A", "event B")
        assert r is not None
        total = sum(r.probabilities.values())
        assert abs(total - 1.0) < 0.01
        _reset_model_state()

    def test_all_labels_in_probabilities(self) -> None:
        """Return one probability entry per configured label."""
        _install_mock_classifier(torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.0]]))
        r = classify_relation("event A", "event B")
        assert r is not None
        for label in LABELS:
            assert label in r.probabilities
        _reset_model_state()

    def test_empty_event_strings(self) -> None:
        """Delegate empty event strings to tokenizer padding."""
        _install_mock_classifier(torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.0]]))
        r = classify_relation("", "")
        assert r is not None  # model handles empty input via tokenizer padding
        _reset_model_state()


# ── Event ordering ─────────────────────────────────────────────


class TestOrderEvents:
    """Validate event ordering through the public ordering function."""

    def test_single_event_unchanged(self) -> None:
        """Preserve a single event without model access."""
        events = [("I went to the gym", "2023-04-10")]
        result = order_events(events)
        assert len(result) == 1
        assert result[0][0] == "I went to the gym"
        assert result[0][2] == ""  # no relation for first

    def test_empty_list(self) -> None:
        """Preserve an empty event list."""
        assert order_events([]) == []

    def test_model_unavailable_preserves_order(self) -> None:
        """Return original order when no classifier is available."""
        _reset_model_state()
        events = [("A", ""), ("B", ""), ("C", "")]
        result = order_events(events)
        assert len(result) == 3
        assert result[0][0] == "A"
        assert result[1][0] == "B"
        assert result[2][0] == "C"

    def test_two_events_before_relation(self) -> None:
        """Keep chronological order when the first event is before the second."""
        _install_mock_classifier(torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        events = [("First event", ""), ("Second event", "")]
        result = order_events(events)
        assert result == [
            ("First event", "", ""),
            ("Second event", "", "before"),
        ]
        _reset_model_state()

    def test_two_events_after_relation(self) -> None:
        """Move the second event first when the classifier returns ``after``."""
        _install_mock_classifier(torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]))
        events = [("Later event", ""), ("Earlier event", "")]
        result = order_events(events)
        assert len(result) == 2
        assert result[0][0] == "Earlier event"
        _reset_model_state()

    def test_three_events_pairwise(self) -> None:
        """Compare all event pairs when ordering more than two events."""
        _install_mock_classifier(torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        events = [("A", ""), ("B", ""), ("C", "")]
        result = order_events(events)
        assert len(result) == 3
        _reset_model_state()
