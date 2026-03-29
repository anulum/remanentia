# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal relation classifier (C3 runtime wrapper)
"""Runtime inference wrapper for C3: temporal relation classifier.

Classifies temporal relations between event pairs:
before, after, same_day, overlaps, contains, unknown.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent
_MODEL_DIR = _BASE / "models" / "temporal-relation-v1"

_model = None
_tokenizer = None
_config = None

LABELS = ["before", "after", "same_day", "overlaps", "contains", "unknown"]


@dataclass
class RelationResult:
    """Result of a pairwise temporal relation classification.

    Attributes:
        relation: Predicted label from :data:`LABELS`.
        confidence: Softmax probability of the predicted label.
        probabilities: Per-class probability distribution.
    """

    relation: str
    confidence: float
    probabilities: dict[str, float]


def _load_model() -> bool:
    """Lazy-load the trained bert-small temporal relation classifier.

    Returns:
        ``True`` if the model is ready, ``False`` if unavailable or corrupt.
    """
    global _model, _tokenizer, _config

    if _model is not None:
        return True

    if not (_MODEL_DIR / "model.pt").exists():
        log.debug("Temporal relation model not found at %s", _MODEL_DIR)
        return False

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        with open(_MODEL_DIR / "config.json") as f:
            _config = json.load(f)

        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))

        class _Classifier(nn.Module):
            """BERT backbone + Linear→ReLU→Dropout→Linear classification head."""

            def __init__(self, model_name, num_classes=6):
                """Build backbone and 2-layer classifier."""
                super().__init__()
                self.backbone = AutoModel.from_pretrained(model_name)
                hidden = self.backbone.config.hidden_size
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes),
                )

            def forward(self, input_ids, attention_mask):
                """Return 6-class logits from [CLS] embedding."""
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = out.last_hidden_state[:, 0]
                return self.classifier(cls_emb)

        device = "cpu"
        model = _Classifier(_config["model_name"], _config.get("num_classes", 6))
        state = torch.load(_MODEL_DIR / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        log.info("Temporal relation classifier loaded from %s", _MODEL_DIR)
        return True
    except Exception:
        log.warning("Failed to load temporal relation classifier", exc_info=True)
        return False


def classify_relation(event_a: str, event_b: str) -> Optional[RelationResult]:
    """Classify the temporal relation between two events.

    Args:
        event_a: Description of event A.
        event_b: Description of event B.

    Returns:
        RelationResult with relation label, confidence, and per-class probabilities.
        None if model is unavailable.
    """
    if not _load_model():
        return None

    import torch

    max_len = _config.get("max_seq_len", 128)
    enc = _tokenizer(
        event_a, event_b,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    labels = _config.get("labels", LABELS)
    prob_dict = {l: probs[i].item() for i, l in enumerate(labels)}
    top_idx = probs.argmax().item()
    return RelationResult(
        relation=labels[top_idx],
        confidence=probs[top_idx].item(),
        probabilities=prob_dict,
    )


def order_events(events: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Order a list of (event_text, date_or_empty) by temporal relation.

    Returns list of (event_text, date, inferred_relation_to_previous).
    Falls back to original order if model unavailable.
    """
    if len(events) <= 1 or not _load_model():
        return [(text, dt, "") for text, dt in events]

    # Build pairwise relation matrix
    n = len(events)
    before_count = [0] * n  # how many events this one is "before"

    for i in range(n):
        for j in range(i + 1, n):
            result = classify_relation(events[i][0], events[j][0])
            if result and result.confidence > 0.6:
                if result.relation == "before":
                    before_count[i] += 1
                elif result.relation == "after":
                    before_count[j] += 1

    # Sort by before_count descending (most "before" = earliest)
    indices = sorted(range(n), key=lambda i: -before_count[i])
    ordered = []
    for rank, idx in enumerate(indices):
        rel = ""
        if rank > 0:
            prev_idx = indices[rank - 1]
            r = classify_relation(events[prev_idx][0], events[idx][0])
            rel = r.relation if r else ""
        ordered.append((events[idx][0], events[idx][1], rel))

    return ordered
