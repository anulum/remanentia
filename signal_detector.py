# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Signal detection for confidence updates

"""Detect correction and reinforcement signals in user messages.

Adapted from DeerFlow 2.0 (ByteDance) signal detection pattern.
"""

import re
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Type of detected signal."""

    CORRECTION = "correction"
    REINFORCEMENT = "reinforcement"
    NEUTRAL = "neutral"


@dataclass
class DetectedSignal:
    """A detected correction or reinforcement signal."""

    signal_type: SignalType
    confidence: float  # How confident we are this is a real signal [0.0-1.0]
    matched_pattern: str  # Which pattern matched
    context: str  # Surrounding text


# Correction patterns — user is telling us something was wrong
_CORRECTION_PATTERNS: list[tuple[str, float]] = [
    (r"\bthat(?:'s| is| was) (?:wrong|incorrect|not right|inaccurate)\b", 0.95),
    (r"\bno[,.]?\s+(?:not |it's not |that's not )", 0.85),
    (r"\bactually[,.]?\s+(?:it |the |I |we |that )", 0.80),
    (r"\bcorrection[:\s]", 0.95),
    (r"\bI (?:was|were) wrong\b", 0.90),
    (r"\bmistake[sd]?\b", 0.70),
    (r"\bmisunderstood\b", 0.85),
    (r"\bshould have been\b", 0.80),
    (r"\bin fact[,.]?\s", 0.75),
    (r"\bnot true\b", 0.90),
    (r"\bthat(?:'s| is) (?:outdated|stale|old|obsolete)\b", 0.85),
    (r"\bforget (?:what I said|that|about)\b", 0.90),
    (r"\bignore (?:that|what|my previous)\b", 0.85),
    (r"\bwrong answer\b", 0.95),
    (r"\btry again\b", 0.60),
]

# Reinforcement patterns — user confirms something is correct
_REINFORCEMENT_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:exactly|precisely) (?:right|correct|that)\b", 0.95),
    (r"\bperfect[.!]?\s*$", 0.85),
    (r"\byes[,.]?\s+(?:that's |that is |correct|right|exactly)", 0.90),
    (r"\bkeep (?:doing |it |that |going)\b", 0.75),
    (r"\bgood (?:answer|response|point|catch)\b", 0.80),
    (r"\bthat(?:'s| is) (?:correct|right|accurate|true)\b", 0.90),
    (r"\bconfirmed?\b", 0.70),
    (r"\bspot on\b", 0.85),
    (r"\bnailed it\b", 0.85),
    (r"\bwell done\b", 0.75),
]


def detect_signals(text: str) -> list[DetectedSignal]:
    """Detect correction and reinforcement signals in text.

    Parameters
    ----------
    text : str
        User message text to analyse.

    Returns
    -------
    list[DetectedSignal]
        Detected signals, sorted by confidence descending.
    """
    signals: list[DetectedSignal] = []
    text_lower = text.lower()

    for pattern, conf in _CORRECTION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            signals.append(
                DetectedSignal(
                    signal_type=SignalType.CORRECTION,
                    confidence=conf,
                    matched_pattern=pattern,
                    context=text[max(0, match.start() - 40) : match.end() + 40],
                )
            )

    for pattern, conf in _REINFORCEMENT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            signals.append(
                DetectedSignal(
                    signal_type=SignalType.REINFORCEMENT,
                    confidence=conf,
                    matched_pattern=pattern,
                    context=text[max(0, match.start() - 40) : match.end() + 40],
                )
            )

    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals


def classify_message(text: str) -> SignalType:
    """Classify a user message as correction, reinforcement, or neutral.

    Takes the highest-confidence signal if any are detected.
    """
    signals = detect_signals(text)
    if not signals:
        return SignalType.NEUTRAL
    return signals[0].signal_type
