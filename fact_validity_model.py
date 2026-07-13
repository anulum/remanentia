# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fact validity model (C5 runtime wrapper)

"""Runtime inference wrapper for C5: fact validity model.

Classifies facts by type (state/event/preference/plan) and detects
supersession (state changes) and temporal boundaries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent
_MODEL_DIR = _BASE / "models" / "fact-validity-v1"

_model = None
_tokenizer = None
_config = None

FACT_TYPES = ["state", "event", "preference", "plan"]


@dataclass
class FactClassification:
    """Result of ML-based fact classification.

    Attributes:
        fact_type: Predicted type from :data:`FACT_TYPES`.
        supersedes_prob: Probability that this fact supersedes a prior fact.
        has_boundary_prob: Probability that this fact has a temporal boundary.
        confidence: Softmax probability of the predicted type.
    """

    fact_type: str
    supersedes_prob: float
    has_boundary_prob: float
    confidence: float


def _build_model(model_name: str, num_types: int = 4) -> Any:
    """Build the production backbone and three fact-validity heads."""
    import torch.nn as nn
    from transformers import AutoModel

    class _FactModel(nn.Module):
        """BERT backbone with type + supersedes + boundary heads."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.type_head = nn.Linear(hidden, num_types)
            self.supersedes_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.boundary_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any]:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = out.last_hidden_state[:, 0]
            return (
                self.type_head(cls_emb),
                self.supersedes_head(cls_emb).squeeze(-1),
                self.boundary_head(cls_emb).squeeze(-1),
            )

    return _FactModel()


def _load_model() -> bool:
    """Lazy-load the trained bert-mini fact validity model.

    Returns:
        ``True`` if the model is ready, ``False`` if unavailable or corrupt.
    """
    global _model, _tokenizer, _config

    if _model is not None:
        return True

    if not (_MODEL_DIR / "model.pt").exists():
        log.debug("Fact validity model not found at %s", _MODEL_DIR)
        return False

    try:
        import torch
        from transformers import AutoTokenizer

        with open(_MODEL_DIR / "config.json") as f:
            _config = json.load(f)

        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))

        device = "cpu"
        model = _build_model(
            _config["model_name"],
            len(_config.get("fact_types", FACT_TYPES)),
        )
        state = torch.load(_MODEL_DIR / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        log.info("Fact validity model loaded from %s", _MODEL_DIR)
        return True
    except Exception:
        log.warning("Failed to load fact validity model", exc_info=True)
        return False


def classify_fact(text: str) -> Optional[FactClassification]:
    """Classify a fact sentence.

    Args:
        text: Fact sentence (e.g., "I started working at Google").

    Returns:
        FactClassification with type, supersedes probability, boundary probability.
        None if model is unavailable.
    """
    if not _load_model():
        return None

    import torch

    max_len = _config.get("max_seq_len", 128)
    enc = _tokenizer(
        text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        type_logits, sup_logit, bnd_logit = _model(enc["input_ids"], enc["attention_mask"])
        type_probs = torch.softmax(type_logits, dim=-1).squeeze(0)
        sup_prob = torch.sigmoid(sup_logit).item()
        bnd_prob = torch.sigmoid(bnd_logit).item()

    types = _config.get("fact_types", FACT_TYPES)
    top_idx = type_probs.argmax().item()
    return FactClassification(
        fact_type=types[top_idx],
        supersedes_prob=sup_prob,
        has_boundary_prob=bnd_prob,
        confidence=type_probs[top_idx].item(),
    )
