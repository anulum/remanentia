# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for fact validity model

"""Real-checkpoint tests for the C5 fact-validity runtime wrapper."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizer

import fact_validity_model
from fact_validity_model import FACT_TYPES, FactClassification, classify_fact


def _reset_model() -> None:
    fact_validity_model._model = None
    fact_validity_model._tokenizer = None
    fact_validity_model._config = None


@contextmanager
def _model_at(model_dir: Path) -> Iterator[None]:
    _reset_model()
    original = fact_validity_model._MODEL_DIR
    fact_validity_model._MODEL_DIR = model_dir
    try:
        yield
    finally:
        fact_validity_model._MODEL_DIR = original
        _reset_model()


def _write_checkpoint(
    root: Path,
    *,
    type_idx: int = 0,
    supersedes_logit: float = -2.0,
    boundary_logit: float = -2.0,
) -> Path:
    """Write a tiny real BERT tokenizer, backbone and production head state."""
    model_dir = root / "fact-validity-v1"
    backbone_dir = root / "backbone"
    model_dir.mkdir(parents=True)
    backbone_dir.mkdir(parents=True)

    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "i",
        "live",
        "berlin",
        "visited",
        "plan",
        "hiking",
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
        "model_type": "bert",
        "model_name": str(backbone_dir),
        "fact_types": FACT_TYPES,
        "max_seq_len": 16,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    model = fact_validity_model._build_model(str(backbone_dir), len(FACT_TYPES))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.type_head.bias[type_idx] = 5.0
        model.supersedes_head[-1].bias.fill_(supersedes_logit)
        model.boundary_head[-1].bias.fill_(boundary_logit)
    torch.save(model.state_dict(), model_dir / "model.pt")
    return model_dir


class TestConstants:
    def test_expected_types(self) -> None:
        assert FACT_TYPES == ["state", "event", "preference", "plan"]


class TestFactClassification:
    def test_fields_and_equality(self) -> None:
        classification = FactClassification(
            fact_type="state",
            supersedes_prob=0.8,
            has_boundary_prob=0.3,
            confidence=0.95,
        )

        assert classification == FactClassification("state", 0.8, 0.3, 0.95)


class TestModelLoading:
    def test_missing_checkpoint_returns_none(self, tmp_path: Path) -> None:
        with _model_at(tmp_path / "missing"):
            assert classify_fact("I live in Berlin") is None

    def test_corrupt_config_returns_false(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "corrupt"
        model_dir.mkdir()
        (model_dir / "model.pt").write_bytes(b"not a checkpoint")
        (model_dir / "config.json").write_text("{", encoding="utf-8")

        with _model_at(model_dir):
            assert fact_validity_model._load_model() is False

    def test_loaded_checkpoint_is_reused(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(tmp_path / "loaded")

        with _model_at(model_dir):
            assert fact_validity_model._load_model() is True
            loaded = fact_validity_model._model
            assert fact_validity_model._load_model() is True
            assert fact_validity_model._model is loaded


class TestRealCheckpointClassification:
    def test_all_fact_type_heads_map_through_real_inference(self, tmp_path: Path) -> None:
        for type_idx, expected in enumerate(FACT_TYPES):
            model_dir = _write_checkpoint(tmp_path / expected, type_idx=type_idx)
            with _model_at(model_dir):
                result = classify_fact("I live in Berlin")

            assert result is not None
            assert result.fact_type == expected
            assert result.confidence > 0.9

    def test_probability_heads_use_real_sigmoid_outputs(self, tmp_path: Path) -> None:
        model_dir = _write_checkpoint(
            tmp_path / "probabilities",
            supersedes_logit=3.0,
            boundary_logit=-3.0,
        )

        with _model_at(model_dir):
            result = classify_fact("I visited Berlin")
            empty_result = classify_fact("")

        assert result is not None
        assert result.supersedes_prob > 0.9
        assert result.has_boundary_prob < 0.1
        assert empty_result is not None
        assert empty_result.fact_type == "state"
