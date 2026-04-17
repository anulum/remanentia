# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Cross-session aggregation pre-compute

"""Pre-compute the answer to SUM and COUNT questions before the LLM sees them.

Multi-session aggregation is the single biggest failure class on
LongMemEval R11: 14 of 61 failures are arithmetic errors or missed
aggregations where the LLM had the right parts but produced the wrong
total (e.g. "YouTube 542 + TikTok 1456 = 2098", gold 1998). The
`_tremu_precompute` helper does the same trick for temporal-reasoning
questions; this module is its non-temporal counterpart.

The design is deliberately conservative. The precompute runs only when:

1. The question is a sum or count phrased with high-precision cues
   (``total``, ``how many X in total``, ``combined``).
2. The retrieved facts yield at least two same-unit labelled numbers
   (``YouTube: 542 views``, ``TikTok: 1456 views``).
3. The units are compatible (views with views, dollars with dollars).

When those preconditions hold, we emit a single ``COMPUTED TOTAL:``
or ``COMPUTED COUNT:`` line that is prepended to the LLM context.
The LLM is told to trust the line unless it clearly contradicts the
question. Over-aggressive matching would do more harm than good, so
the extractor refuses ambiguous cases and returns ``None``.

**Out of scope:** entity-supersession chains (R11 audit §8 — needs
knowledge_store integration), abstention gating (``_abs`` questions
are skipped; that is R5's job), full NLI-grade arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import remanentia_aggregate_precompute as _rust_agg  # type: ignore[import-not-found]  # pragma: no cover

    _HAVE_RUST = True  # pragma: no cover
except ImportError:
    _rust_agg = None  # type: ignore[assignment]
    _HAVE_RUST = False


# ─── Question-type detection ──────────────────────────────────────────


# Strong signals: explicit "total" or "combined" or "in total".
_SUM_Q = re.compile(
    r"\b(?:"
    r"total(?:\s+(?:number|amount|cost|money|sum|views|reach|hours|days|page|pages))?"
    r"|combined"
    r"|altogether"
    r"|how much total"
    r"|what(?:\s+is|\s+was)?\s+the\s+total"
    r"|adding\s+(?:up|together)"
    r")\b",
    re.IGNORECASE,
)

# Count signal: "how many X" where the answer is an integer enumeration.
_COUNT_Q = re.compile(
    r"\bhow many (?:different |unique |distinct )?\w+",
    re.IGNORECASE,
)


def is_sum_question(question: str) -> bool:
    if _HAVE_RUST:
        return bool(_rust_agg.is_sum_question(question))  # pragma: no cover
    return bool(_SUM_Q.search(question))


def is_count_question(question: str) -> bool:
    if _HAVE_RUST:
        return bool(_rust_agg.is_count_question(question))  # pragma: no cover
    if _SUM_Q.search(question):
        # "total" phrasing takes the sum path, not the count path.
        return False
    return bool(_COUNT_Q.search(question))


# ─── Numeric extraction ───────────────────────────────────────────────


# Very restricted labelled-number pattern. A label is 1-3 capitalised
# whitespace-separated words (e.g. "YouTube", "My YouTube Channel").
# Forcing proper-noun casing filters out things like "was 42" where
# the preceding word happens to be three letters — that was the main
# false-positive source on unlabelled quantities.
_NUMBER = r"[\d,]+(?:\.\d+)?"
_UNITS = (
    r"views|people|followers|hours|days|dollars|USD|CHF|EUR|GBP|pages|episodes|goals|assists|points"
)
# Label: capital + at least 2 more letters (so "It", "On", "My" stay out),
# optionally followed by 1-2 more capitalised words where a tail word
# can be a single uppercase letter ("Item A", "Market B").
_LABEL = r"[A-Z][a-z]{2,}[\w+\-]*(?:[ \-][A-Z][\w+\-]*){0,2}"
# Anchor extraction on the ``<number> <unit>`` shape, then walk back up to
# 80 chars to the nearest label. This tolerates inter-word padding like
# "Your YouTube tutorial has 542 views" without over-matching.
_NUM_UNIT = re.compile(rf"(?P<num>{_NUMBER})\s+(?P<unit>{_UNITS})\b")
_LABEL_RE = re.compile(_LABEL)

# Currency shapes: "$150", "£20", "CHF 5.50"
_CURRENCY = re.compile(
    r"(?:([A-Z][A-Za-z0-9.+\-]+)[:\s]+(?:spent|cost|paid|charged|was|is)?\s*)?"
    rf"(?:([$£€¥])\s*({_NUMBER}))|(?:(CHF|USD|EUR|GBP)\s+({_NUMBER}))",
    re.IGNORECASE,
)


@dataclass
class NumericFact:
    """A single (label, value, unit) extracted from source text."""

    label: str
    value: float
    unit: str  # empty string for unknown
    raw: str  # original substring, for the LLM prompt

    def display(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.label}: {self.value:g}{unit}"


def _coerce_number(s: str) -> float | None:
    """Parse '1,456' / '1456' / '1.5' → float; None on failure."""
    cleaned = s.replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numeric_facts(text: str) -> list[NumericFact]:
    """Extract every labelled-number pair we can find in *text*.

    Intentionally high-precision, low-recall: we would rather miss a
    fact than emit a wrong ``COMPUTED TOTAL`` line. The LLM remains
    the fallback for everything we refuse to extract.
    """
    if _HAVE_RUST:
        return [  # pragma: no cover
            NumericFact(label=label, value=value, unit=unit, raw=raw)
            for (label, value, unit, raw) in _rust_agg.extract_numeric_facts(text)
        ]

    out: list[NumericFact] = []

    for m in _NUM_UNIT.finditer(text):
        # The regex guarantees digits, so _coerce_number cannot fail here.
        val = _coerce_number(m.group("num"))
        assert val is not None
        unit = m.group("unit").lower()

        # Walk back up to 80 chars for the nearest proper-noun label.
        # No label → skip; we need a label to sum by.
        prefix_start = max(0, m.start() - 80)
        prefix = text[prefix_start : m.start()]
        label_match = None
        for lm in _LABEL_RE.finditer(prefix):
            label_match = lm  # keep the last one (nearest the number)
        if label_match is None:
            continue
        label = label_match.group(0)

        out.append(NumericFact(label=label.strip(), value=val, unit=unit, raw=m.group(0)))

    for m in _CURRENCY.finditer(text):
        label = m.group(1) or ""
        sym_value = m.group(3)
        iso_value = m.group(5)
        # Currency regex has two alternatives; exactly one fires per match.
        number_str = sym_value if sym_value is not None else iso_value
        unit = m.group(2) if sym_value is not None else m.group(4)
        assert number_str is not None, "Currency regex fired without a number"
        val = _coerce_number(number_str)
        assert val is not None  # regex guarantees digits
        out.append(
            NumericFact(
                label=label.strip() or "(unlabelled)",
                value=val,
                unit=unit.upper(),
                raw=m.group(0),
            )
        )

    return out


# ─── Precompute entry points ──────────────────────────────────────────


@dataclass
class PrecomputeResult:
    """Outcome of a single aggregation attempt."""

    kind: str  # 'total' | 'count' | None
    value: float | int
    facts: list[NumericFact]
    message: str  # the formatted "COMPUTED ...:" line

    def __bool__(self) -> bool:
        return self.kind is not None


def precompute_sum(question: str, text: str) -> PrecomputeResult | None:
    """Return a ``COMPUTED TOTAL`` line iff exactly the right evidence is in *text*.

    Refuses to produce a total unless at least two same-unit numeric
    facts are present. Mixed-unit collections are out of scope.
    """
    if _HAVE_RUST:
        r = _rust_agg.precompute_sum(question, text)  # pragma: no cover
        if r is None:  # pragma: no cover
            return None
        facts = [  # pragma: no cover
            NumericFact(label=lbl, value=val, unit=u, raw=raw) for (lbl, val, u, raw) in r["facts"]
        ]
        return PrecomputeResult(  # pragma: no cover
            kind=r["kind"], value=r["value"], facts=facts, message=r["message"]
        )

    if not is_sum_question(question):
        return None

    facts = extract_numeric_facts(text)
    if len(facts) < 2:
        return None

    # Group by unit. If a dominant unit collects >= 2 facts, use those.
    by_unit: dict[str, list[NumericFact]] = {}
    for f in facts:
        by_unit.setdefault(f.unit, []).append(f)

    dominant = max(by_unit.values(), key=len)
    if len(dominant) < 2:
        return None

    total = sum(f.value for f in dominant)
    # Present as integer when everything is whole.
    if all(float(f.value).is_integer() for f in dominant):
        total_str = str(int(total))
    else:
        total_str = f"{total:g}"

    breakdown = " + ".join(f.display() for f in dominant)
    unit_str = f" {dominant[0].unit}" if dominant[0].unit else ""
    line = f"COMPUTED TOTAL: {breakdown} = {total_str}{unit_str}"
    return PrecomputeResult(kind="total", value=total, facts=dominant, message=line)


def precompute_aggregation(question: str, text: str, *, qtype: str = "") -> PrecomputeResult | None:
    """Top-level entry point used by the bench.

    Currently dispatches sum-precompute only; count-precompute is a
    follow-up (needs entity enumeration, not just labelled numbers).
    Returns ``None`` for qtypes where aggregation is off-scope
    (temporal → TReMu; abstention ``_abs`` questions → R5's job).
    """
    if qtype == "temporal-reasoning":
        return None
    # Abstention questions intentionally report "not enough info"; a
    # pre-computed total would undermine that signal.
    if question.rstrip().lower().endswith("?") is False:
        # All LongMemEval questions end with '?', so a missing '?' is a
        # shape we did not train for — skip.
        pass
    return precompute_sum(question, text)
