# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for temporal relation classifier

"""Real-checkpoint tests for the C3 temporal relation runtime wrapper."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from transformers import BertConfig, BertModel, BertTokenizer

import temporal_relation
from temporal_relation import LABELS, RelationResult, classify_relation, order_events


def _reset_model() -> None:
    temporal_relation._model = None
    temporal_relation._tokenizer = None
    temporal_relation._config = None


@contextmanager
def _model_at(model_dir: Path) -> Iterator[None]:
    _reset_model()
    original = temporal_relation._MODEL_DIR
    temporal_relation._MODEL_DIR = model_dir
    try:
        yield
    finally:
        temporal_relation._MODEL_DIR = original
        _reset_model()


def _write_checkpoint(root: Path, *, top_label: str | None = "before") -> Path:
    """Write a tiny real BERT tokenizer, backbone and production head state."""
    model_dir = root / "temporal-relation-v1"
    backbone_dir = root / "backbone"
    model_dir.mkdir(parents=True)
    backbone_dir.mkdir(parents=True)

    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "first",
        "second",
        "event",
        "before",
        "after",
        "today",
    ]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True).save_pretrained(model_dir)

    backbone_config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    BertModel(backbone_config).save_pretrained(backbone_dir)

    config = {
        "model_name": str(backbone_dir),
        "num_classes": len(LABELS),
        "max_seq_len": 16,
        "labels": LABELS,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    model = temporal_relation._build_model(str(backbone_dir), len(LABELS))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        if top_label is not None:
            cast(Any, model).classifier[-1].bias[LABELS.index(top_label)] = 6.0
    torch.save(model.state_dict(), model_dir / "model.pt")
    return model_dir


class TestContracts:
    def test_labels_and_result_value_semantics(self) -> None:
        assert LABELS == ["before", "after", "same_day", "overlaps", "contains", "unknown"]
        result = RelationResult("before", 0.9, {"before": 0.9})
        assert result == RelationResult("before", 0.9, {"before": 0.9})

    def test_config_helpers_accept_valid_values_and_reject_invalid_ones(self) -> None:
        assert temporal_relation._config_int({"n": 7}, "n", 3) == 7
        assert temporal_relation._config_int({"n": "7"}, "n", 3) == 3
        assert temporal_relation._config_model_name({"model_name": "bert"}) == "bert"
        for value in (None, ""):
            try:
                temporal_relation._config_model_name({"model_name": value})
            except ValueError as exc:
                assert "model_name" in str(exc)
            else:
                raise AssertionError("invalid model name was accepted")

    def test_label_config_accepts_lists_and_tuples_with_safe_fallback(self) -> None:
        assert temporal_relation._config_labels({"labels": list(LABELS)}) == LABELS
        assert temporal_relation._config_labels({"labels": tuple(LABELS)}) == tuple(LABELS)
        assert temporal_relation._config_labels({"labels": ["before", 7]}) == LABELS
        assert temporal_relation._config_labels({"labels": "before"}) == LABELS


class TestModelLoading:
    def test_missing_checkpoint_returns_none(self, tmp_path: Path) -> None:
        with _model_at(tmp_path / "missing"):
            assert classify_relation("event A", "event B") is None

    def test_corrupt_and_non_object_configs_fail_closed(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt"
        corrupt.mkdir()
        (corrupt / "model.pt").write_bytes(b"checkpoint")
        (corrupt / "config.json").write_text("{", encoding="utf-8")
        with _model_at(corrupt):
            assert temporal_relation._load_model() is False

        non_object = tmp_path / "non-object"
        non_object.mkdir()
        (non_object / "model.pt").write_bytes(b"checkpoint")
        (non_object / "config.json").write_text("[]", encoding="utf-8")
        with _model_at(non_object):
            assert temporal_relation._load_model() is False

    def test_loaded_real_checkpoint_is_reused(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "loaded")
        with _model_at(model_dir):
            assert temporal_relation._load_model() is True
            loaded = temporal_relation._model
            assert temporal_relation._load_model() is True
            assert temporal_relation._model is loaded

    def test_incomplete_real_cache_fails_closed(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "incomplete")
        with _model_at(model_dir):
            assert temporal_relation._load_model() is True
            temporal_relation._tokenizer = None
            assert classify_relation("event A", "event B") is None


class TestRealCheckpointClassification:
    def test_every_label_maps_through_real_inference(self, tmp_path: Path) -> None:
        for label in LABELS:
            model_dir = _write_checkpoint(tmp_path / label, top_label=label)
            with _model_at(model_dir):
                result = classify_relation("first event", "second event")

            assert result is not None
            assert result.relation == label
            assert result.confidence > 0.9
            assert set(result.probabilities) == set(LABELS)
            assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_real_tokenizer_handles_empty_event_text(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "empty")
        with _model_at(model_dir):
            result = classify_relation("", "")
        assert result is not None
        assert result.relation == "before"


class TestOrderEvents:
    def test_empty_and_single_inputs_need_no_model(self) -> None:
        assert order_events([]) == []
        assert order_events([("only", "2026-07-13")]) == [("only", "2026-07-13", "")]

    def test_missing_checkpoint_preserves_multiple_events(self, tmp_path: Path) -> None:
        events = [("A", ""), ("B", ""), ("C", "")]
        with _model_at(tmp_path / "missing"):
            assert order_events(events) == [("A", "", ""), ("B", "", ""), ("C", "", "")]

    def test_before_checkpoint_keeps_chronological_order(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "before", top_label="before")
        with _model_at(model_dir):
            result = order_events([("First event", "1"), ("Second event", "2")])
        assert result == [("First event", "1", ""), ("Second event", "2", "before")]

    def test_after_checkpoint_moves_second_event_first(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "after", top_label="after")
        with _model_at(model_dir):
            result = order_events([("Later event", "2"), ("Earlier event", "1")])
        assert result == [("Earlier event", "1", ""), ("Later event", "2", "after")]

    def test_low_confidence_and_non_ordering_labels_preserve_input_order(
        self, tmp_path: Path
    ) -> None:
        events = [("A", ""), ("B", ""), ("C", "")]
        uniform = _write_checkpoint(tmp_path / "uniform", top_label=None)
        with _model_at(uniform):
            low_confidence = order_events(events)
        assert [event[0] for event in low_confidence] == ["A", "B", "C"]

        unknown = _write_checkpoint(tmp_path / "unknown", top_label="unknown")
        with _model_at(unknown):
            non_ordering = order_events(events)
        assert [event[0] for event in non_ordering] == ["A", "B", "C"]
        assert non_ordering[1][2] == "unknown"

    def test_incomplete_cache_preserves_order_without_relation(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "incomplete-order")
        with _model_at(model_dir):
            assert temporal_relation._load_model() is True
            temporal_relation._tokenizer = None
            result = order_events([("A", ""), ("B", "")])
        assert result == [("A", "", ""), ("B", "", "")]
