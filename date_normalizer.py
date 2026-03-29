# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Vague date normaliser (C4 runtime wrapper)
"""Runtime inference wrapper for C4: date normaliser.

Converts vague date expressions to ISO dates using the trained model,
with rule-based fallback for common patterns.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent
_MODEL_DIR = _BASE / "models" / "date-normalizer-v1"

# Lazy-loaded model state
_model = None
_tokenizer = None

# ---------------------------------------------------------------------------
# Rule-based normaliser (covers ~60% of vague expressions without ML)
# ---------------------------------------------------------------------------

_QUANTIFIED_RE = re.compile(
    r"\b(?:about\s+|around\s+|roughly\s+|approximately\s+)?"
    r"(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)

_WEEKDAY_RE = re.compile(
    r"\b(?:last|this\s+past)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)

_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_COUPLE_RE = re.compile(
    r"\ba\s+couple\s+of\s+(days?|weeks?|months?)\s+ago\b",
    re.IGNORECASE,
)

_FEW_RE = re.compile(
    r"\ba\s+few\s+(days?|weeks?|months?)\s+ago\b",
    re.IGNORECASE,
)

_SEVERAL_RE = re.compile(
    r"\bseveral\s+(days?|weeks?|months?)\s+ago\b",
    re.IGNORECASE,
)

_SIMPLE_RE = {
    re.compile(r"\byesterday\b", re.I): lambda r: r - timedelta(days=1),
    re.compile(r"\btoday\b", re.I): lambda r: r,
    re.compile(r"\bthe\s+other\s+day\b", re.I): lambda r: r - timedelta(days=3),
    re.compile(r"\bnot\s+long\s+ago\b", re.I): lambda r: r - timedelta(days=7),
    re.compile(r"\brecently\b", re.I): lambda r: r - timedelta(days=5),
    re.compile(r"\bearlier\s+this\s+week\b", re.I): lambda r: r - timedelta(days=max(r.weekday(), 1)),
    re.compile(r"\bearlier\s+this\s+month\b", re.I): lambda r: r.replace(day=max(1, r.day // 2)),
    re.compile(r"\blast\s+week\b", re.I): lambda r: r - timedelta(weeks=1),
    re.compile(r"\blast\s+month\b", re.I): lambda r: _month_delta(r, -1),
    re.compile(r"\blast\s+year\b", re.I): lambda r: r.replace(year=r.year - 1),
    re.compile(r"\bthis\s+week\b", re.I): lambda r: r - timedelta(days=r.weekday()),
    re.compile(r"\bthis\s+month\b", re.I): lambda r: r.replace(day=1),
    re.compile(r"\bthis\s+year\b", re.I): lambda r: r.replace(month=1, day=1),
}


def _month_delta(d: date, months: int) -> date:
    """Shift *d* by *months*, clamping day to target month's maximum."""
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    max_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, max_day))


@dataclass
class DateResult:
    """Result of a vague date normalisation attempt.

    Attributes:
        iso_date: Resolved date in ISO 8601 format (YYYY-MM-DD).
        confidence: Normalisation confidence in [0, 1].
        method: Resolution strategy — ``"rule"`` or ``"model"``.
    """

    iso_date: str
    confidence: float
    method: str  # "rule" or "model"


def _rule_based_normalise(expr: str, ref: date) -> Optional[DateResult]:
    """Attempt rule-based normalisation. Returns None if no rule matches."""
    expr_stripped = expr.strip()

    # Quantified: "N days/weeks/months/years ago"
    m = _QUANTIFIED_RE.search(expr_stripped)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "day":
            target = ref - timedelta(days=n)
        elif unit == "week":
            target = ref - timedelta(weeks=n)
        elif unit == "month":
            target = _month_delta(ref, -n)
        elif unit == "year":
            target = _month_delta(ref, -n * 12)
        else:  # pragma: no cover — regex guarantees valid units
            return None
        return DateResult(iso_date=target.isoformat(), confidence=0.95, method="rule")

    # "a couple of days/weeks/months ago"
    m = _COUPLE_RE.search(expr_stripped)
    if m:
        unit = m.group(1).lower().rstrip("s")
        if unit == "day":
            target = ref - timedelta(days=2)
        elif unit == "week":
            target = ref - timedelta(weeks=2)
        elif unit == "month":
            target = _month_delta(ref, -2)
        else:  # pragma: no cover — regex guarantees valid units
            return None
        return DateResult(iso_date=target.isoformat(), confidence=0.9, method="rule")

    # "a few days/weeks/months ago"
    m = _FEW_RE.search(expr_stripped)
    if m:
        unit = m.group(1).lower().rstrip("s")
        if unit == "day":
            target = ref - timedelta(days=3)
        elif unit == "week":
            target = ref - timedelta(weeks=3)
        elif unit == "month":
            target = _month_delta(ref, -3)
        else:  # pragma: no cover — regex guarantees valid units
            return None
        return DateResult(iso_date=target.isoformat(), confidence=0.8, method="rule")

    # "several days/weeks/months ago"
    m = _SEVERAL_RE.search(expr_stripped)
    if m:
        unit = m.group(1).lower().rstrip("s")
        if unit == "day":
            target = ref - timedelta(days=5)
        elif unit == "week":
            target = ref - timedelta(weeks=5)
        elif unit == "month":
            target = _month_delta(ref, -5)
        else:  # pragma: no cover — regex guarantees valid units
            return None
        return DateResult(iso_date=target.isoformat(), confidence=0.7, method="rule")

    # "last Monday" / "this past Friday"
    m = _WEEKDAY_RE.search(expr_stripped)
    if m:
        day_name = m.group(1).lower()
        day_idx = _WEEKDAY_MAP.get(day_name)
        if day_idx is not None:
            days_back = (ref.weekday() - day_idx) % 7
            if days_back == 0:
                days_back = 7
            target = ref - timedelta(days=days_back)
            return DateResult(iso_date=target.isoformat(), confidence=0.95, method="rule")

    # Simple fixed patterns
    for pattern, resolver in _SIMPLE_RE.items():
        if pattern.search(expr_stripped):
            try:
                target = resolver(ref)
                return DateResult(iso_date=target.isoformat(), confidence=0.85, method="rule")
            except (ValueError, AttributeError):  # pragma: no cover
                continue

    return None


# ---------------------------------------------------------------------------
# ML-based normaliser (fallback for expressions rules can't handle)
# ---------------------------------------------------------------------------

def _load_model():
    """Lazy-load the trained bert-mini date normaliser from *_MODEL_DIR*.

    Returns:
        ``True`` if the model is ready, ``False`` if unavailable or corrupt.
    """
    global _model, _tokenizer

    if _model is not None:
        return True

    if not (_MODEL_DIR / "model.pt").exists():
        log.debug("Date normaliser model not found at %s", _MODEL_DIR)
        return False

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        with open(_MODEL_DIR / "config.json") as f:
            config = json.load(f)

        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))  # pragma: no cover

        # Reconstruct model                              # pragma: no cover
        import torch.nn as nn                             # pragma: no cover

        class _DateNormalizer(nn.Module):                 # pragma: no cover
            """8-digit date predictor: backbone → 8 × softmax-10 + confidence."""

            def __init__(self, model_name, num_digits=8):
                """Build backbone, digit heads, and confidence head."""
                super().__init__()
                self.backbone = AutoModel.from_pretrained(model_name)
                hidden = self.backbone.config.hidden_size
                self.digit_heads = nn.ModuleList([
                    nn.Linear(hidden, 10) for _ in range(num_digits)
                ])
                self.confidence_head = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
                )

            def forward(self, input_ids, attention_mask):
                """Return (list[digit_logits], confidence) from [CLS]."""
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = out.last_hidden_state[:, 0]
                digit_logits = [h(cls_emb) for h in self.digit_heads]
                confidence = self.confidence_head(cls_emb).squeeze(-1)
                return digit_logits, confidence

        device = "cpu"  # inference on CPU is fast for this tiny model  # pragma: no cover
        model = _DateNormalizer(config["model_name"], config.get("num_digits", 8))  # pragma: no cover
        state = torch.load(_MODEL_DIR / "model.pt", map_location=device, weights_only=True)  # pragma: no cover
        model.load_state_dict(state)                      # pragma: no cover
        model.eval()                                      # pragma: no cover
        _model = model                                    # pragma: no cover
        log.info("Date normaliser model loaded from %s", _MODEL_DIR)  # pragma: no cover
        return True                                       # pragma: no cover
    except Exception:
        log.warning("Failed to load date normaliser model", exc_info=True)
        return False


def _model_normalise(expr: str, ref: date) -> Optional[DateResult]:
    """Normalise *expr* using the trained ML model.

    Falls back to ``None`` when the model is unavailable or the predicted
    date is invalid (e.g. month 13).

    Returns:
        :class:`DateResult` with ``method="model"`` or ``None``.
    """
    if not _load_model():
        return None

    import torch

    text = f"reference: {ref.isoformat()} expression: {expr}"
    enc = _tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        digit_logits, confidence = _model(enc["input_ids"], enc["attention_mask"])
        pred_digits = [dl.argmax(dim=-1).item() for dl in digit_logits]
        conf = confidence.item()

    # Reconstruct ISO date
    digits_str = "".join(str(d) for d in pred_digits)
    try:
        y = int(digits_str[:4])
        m = int(digits_str[4:6])
        d = int(digits_str[6:8])
        target = date(y, m, d)
        return DateResult(iso_date=target.isoformat(), confidence=conf, method="model")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Regex to detect vague date expressions in text
VAGUE_DATE_RE = re.compile(
    r"(?:"
    r"\b(?:about\s+|around\s+|roughly\s+|approximately\s+)?\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b|"
    r"\ba\s+(?:few|couple\s+of|several)\s+(?:days?|weeks?|months?)\s+ago\b|"
    r"\b(?:last|this\s+past)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
    r"\b(?:yesterday|today|recently|the\s+other\s+day|not\s+long\s+ago)\b|"
    r"\b(?:earlier\s+this|last|this)\s+(?:week|month|year)\b|"
    r"\bsome\s+time\s+(?:ago|back)\b"
    r")",
    re.IGNORECASE,
)


def normalize_date_expression(expr: str, ref: date) -> Optional[DateResult]:
    """Normalise a vague date expression to an ISO date.

    Tries rule-based first, falls back to trained model.

    Args:
        expr: Vague date expression (e.g., "3 weeks ago", "last Tuesday").
        ref: Reference date to resolve against.

    Returns:
        DateResult with ISO date and confidence, or None if unresolvable.
    """
    # Rule-based first (fast, high confidence)
    result = _rule_based_normalise(expr, ref)
    if result is not None:
        return result

    # ML fallback
    return _model_normalise(expr, ref)


def extract_and_normalise(text: str, ref: date) -> list[DateResult]:
    """Find all vague date expressions in text and normalise them.

    Returns list of DateResult for each matched expression.
    """
    results = []
    for match in VAGUE_DATE_RE.finditer(text):
        result = normalize_date_expression(match.group(0), ref)
        if result is not None:
            results.append(result)
    return results
