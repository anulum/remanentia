# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — reader confidence signals for calibrated abstention

"""Two confidence signals per answer, for the calibrated-abstention axis.

Calibrated abstention asks whether a system knows when it does not know. Scoring
it (`coverage_accuracy`) needs a per-answer confidence, and the honest design is
to emit two complementary signals so the scorecard can use either:

1. **Reader self-report** — the reader LLM ends its answer with a
   ``CONFIDENCE: 0.83`` line, parsed off and clamped to [0, 1]. Principled (the
   reader rates its own certainty) but only present when the model complies.
2. **Retrieval-score proxy** — the top reranker/BM25 score squashed to [0, 1].
   Always available, no prompt change; weaker across heterogeneous score scales
   but robust.

This module holds the parsing/normalisation so the benchmark harness only wires
thin calls; the logic is tested here. It performs no model calls.
"""

from __future__ import annotations

import math
import re

_CONFIDENCE_LINE = re.compile(
    r"\n?\s*confidence\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+|\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

_SUFFIX = (
    "\n\nAfter your answer, on a new final line, state your confidence that the "
    "answer is correct as `CONFIDENCE: x` where x is a number between 0 and 1."
)


def confidence_suffix() -> str:
    """Return the instruction appended to the reader prompt to elicit a rating."""
    return _SUFFIX


def parse_confidence(answer: str) -> tuple[str, float | None]:
    """Split a trailing ``CONFIDENCE: x`` rating off *answer*.

    Returns the answer with the rating line removed and the parsed confidence
    clamped to [0, 1], or ``(answer.strip(), None)`` when no rating is present.
    """
    match = _CONFIDENCE_LINE.search(answer)
    if match is None:
        return answer.strip(), None
    value = max(0.0, min(1.0, float(match.group(1))))
    cleaned = answer[: match.start()].strip()
    return cleaned, value


def normalise_score(score: float) -> float:
    """Squash a raw retrieval score into [0, 1] with a logistic function.

    For a cross-encoder logit this is a calibrated relevance probability; for an
    unbounded BM25 score it is a monotone proxy. Guards against overflow.
    """
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(score, 60.0)))
    exp = math.exp(max(score, -60.0))
    return exp / (1.0 + exp)
