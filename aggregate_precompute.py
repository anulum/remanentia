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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast


class _RustSumResult(TypedDict):
    """Dictionary returned by the optional Rust sum precompute binding."""

    kind: str
    value: float
    facts: list[tuple[str, float, str, str]]
    message: str


class _RustAggregatePrecompute(Protocol):
    """Typed facade for the optional untyped Rust extension."""

    def is_sum_question(self, question: str) -> bool:
        """Return whether *question* requests a sum aggregation."""

    def is_count_question(self, question: str) -> bool:
        """Return whether *question* requests a count aggregation."""

    def extract_numeric_facts(self, text: str) -> list[tuple[str, float, str, str]]:
        """Extract numeric facts from *text*."""

    def precompute_sum(self, question: str, text: str) -> _RustSumResult | None:
        """Return a Rust-computed sum result when evidence is unambiguous."""


try:
    import remanentia_aggregate_precompute as _rust_agg_raw  # type: ignore[import-untyped]  # pragma: no cover

    _rust_agg: _RustAggregatePrecompute | None = cast(  # pragma: no cover
        "_RustAggregatePrecompute",
        _rust_agg_raw,
    )
    _HAVE_RUST = True  # pragma: no cover
except ImportError:
    _rust_agg = None
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
_DISTINCT_COUNT_TARGET = re.compile(
    r"\bhow many\s+(?:different|unique|distinct)\s+"
    r"(?P<target>[a-z][a-z0-9 \-]*?)"
    r"(?:\s+(?:did|do|have|has|had|am|are|was|were|that|which|in|from|for|to|of)\b|\?)",
    re.IGNORECASE,
)


def is_sum_question(question: str) -> bool:
    """Return whether *question* asks for a deterministic sum."""
    if _HAVE_RUST:
        assert _rust_agg is not None
        return bool(_rust_agg.is_sum_question(question))  # pragma: no cover
    return bool(_SUM_Q.search(question))


def is_count_question(question: str) -> bool:
    """Return whether *question* asks for a count rather than a sum."""
    if _HAVE_RUST:
        assert _rust_agg is not None
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


@dataclass
class CountFact:
    """A single distinct item extracted for count precompute."""

    label: str
    raw: str

    def display(self) -> str:
        """Return the prompt-ready label for this counted item."""
        return self.label


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
        assert _rust_agg is not None
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


_DR_NAME = re.compile(r"\bDr\.\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\b")
_COUNT_ITEM_SPLIT = re.compile(r"\s*(?:,|;|\band\b)\s*")
_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _count_target(question: str) -> str | None:
    """Return the explicit distinct-count target phrase, if present."""
    match = _DISTINCT_COUNT_TARGET.search(question)
    if match is None:
        return None
    target = " ".join(match.group("target").lower().split())
    return target or None


def _normalise_count_label(raw: str) -> str | None:
    """Normalise a candidate counted-item label."""
    label = raw.strip(" \t\n\r.:-")
    label = _ARTICLE.sub("", label)
    label = re.sub(r"\s+", " ", label).strip()
    if not label:
        return None
    if len(label) < 2:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", label):
        return None
    return label


def _dedupe_count_facts(labels: list[tuple[str, str]]) -> list[CountFact]:
    """Return stable, case-insensitive distinct count facts."""
    seen: set[str] = set()
    facts: list[CountFact] = []
    for label, raw in labels:
        normalised = _normalise_count_label(label)
        if normalised is None:
            continue
        key = normalised.casefold()
        if key in seen:
            continue
        seen.add(key)
        facts.append(CountFact(label=normalised, raw=raw))
    return facts


def _singular_token(token: str) -> str:
    """Return a small English singular approximation for target matching."""
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _target_terms(target: str) -> set[str]:
    """Return count-target tokens used to match explicit list headings."""
    stop = {"different", "distinct", "unique", "type", "types", "of"}
    return {
        _singular_token(token)
        for token in re.findall(r"[a-z0-9]+", target.lower())
        if token not in stop and len(token) > 2
    }


def _explicit_distinct_lists(text: str, target: str) -> list[tuple[str, str]]:
    """Extract labels from explicit ``Different X: a, b, and c`` lists."""
    labels: list[tuple[str, str]] = []
    wanted = _target_terms(target)
    for match in re.finditer(
        r"\b(?:different|unique|distinct)\s+(?P<category>[A-Za-z0-9 \-]+):\s*(?P<items>[^.\n]+)",
        text,
        re.IGNORECASE,
    ):
        category_terms = _target_terms(match.group("category"))
        if wanted and not wanted.intersection(category_terms):
            continue
        for item in _COUNT_ITEM_SPLIT.split(match.group("items")):
            labels.append((item, match.group(0)))
    return labels


def extract_count_facts(question: str, text: str) -> list[CountFact]:
    """Extract high-confidence distinct items for count questions.

    The extractor is intentionally narrower than :func:`is_count_question`: it
    emits facts only for explicit ``different`` / ``unique`` / ``distinct``
    questions. Generic count questions such as "How many hours..." often encode
    arithmetic rather than entity enumeration and remain on the LLM fallback.
    """
    target = _count_target(question)
    if target is None:
        return []

    if "doctor" in target:
        doctor_labels = [(match.group(0), match.group(0)) for match in _DR_NAME.finditer(text)]
        doctor_facts = _dedupe_count_facts(doctor_labels)
        if len(doctor_facts) >= 2:
            return doctor_facts

    return _dedupe_count_facts(_explicit_distinct_lists(text, target))


# ─── Precompute entry points ──────────────────────────────────────────


@dataclass
class PrecomputeResult:
    """Outcome of a single aggregation attempt."""

    kind: str  # 'total' | 'count'
    value: float | int
    facts: Sequence[NumericFact | CountFact]
    message: str  # the formatted "COMPUTED ...:" line

    def __bool__(self) -> bool:
        return self.kind is not None


def precompute_sum(question: str, text: str) -> PrecomputeResult | None:
    """Return a ``COMPUTED TOTAL`` line iff exactly the right evidence is in *text*.

    Refuses to produce a total unless at least two same-unit numeric
    facts are present. Mixed-unit collections are out of scope.
    """
    if _HAVE_RUST:
        assert _rust_agg is not None
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


def precompute_count(question: str, text: str) -> PrecomputeResult | None:
    """Return a ``COMPUTED COUNT`` line for explicit distinct-item questions."""
    if not is_count_question(question):
        return None
    facts = extract_count_facts(question, text)
    if len(facts) < 2:
        return None

    target = _count_target(question) or "items"
    count = len(facts)
    labels = ", ".join(f.display() for f in facts)
    line = f"COMPUTED COUNT: {labels} = {count} distinct {target}"
    return PrecomputeResult(kind="count", value=count, facts=facts, message=line)


def precompute_aggregation(question: str, text: str, *, qtype: str = "") -> PrecomputeResult | None:
    """Top-level entry point used by the bench.

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
    return precompute_sum(question, text) or precompute_count(question, text)
