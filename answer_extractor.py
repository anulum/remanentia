# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Answer extraction

"""Extract short answers from retrieved paragraphs.

Given a query and a paragraph, attempts to extract the most likely
answer span using pattern matching. No LLM required.

Handles:
- Dates: "2026-03-15", "March 15"
- Numbers/percentages: "66.4%", "1,986", "42"
- Versions: "v3.9.0", "0.2.0"
- Names: capitalized multi-word spans near query terms
- Yes/no: from negation/affirmation patterns
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from importlib import import_module
from typing import Any


def extract_answer(query: str, paragraph: str) -> str | None:
    """Extract a short answer from a paragraph given a query.

    Returns the best candidate answer string, or None if no answer found.
    Uses Rust engine when available (~11× faster), except for duration
    questions which require Python date arithmetic.
    """
    q = query.lower()
    p = paragraph

    # Duration questions need Python date arithmetic — check before Rust
    if _is_duration_question(q):
        dur = extract_duration(p, query)
        if dur is not None:
            return dur

    try:
        _rust_extract = import_module("remanentia_answer_extractor").extract_answer

        return _rust_extract(query, paragraph)  # pragma: no cover
    except ImportError:
        pass

    # Detect question type and dispatch
    if _is_when_question(q):
        return _extract_date_answer(p, query=query)
    if _is_duration_question(q):
        return _extract_number_answer(p, q)
    if _is_how_many_question(q):
        return _extract_number_answer(p, q)
    if _is_version_question(q):
        return _extract_version_answer(p)
    if _is_who_question(q):
        return _extract_name_answer(p, q)
    if _is_yes_no_question(q):
        return _extract_yes_no(p, q)
    if _is_what_percent_question(q):
        return _extract_percentage_answer(p)

    # Generic: try percentage, then version, then date, then number
    for extractor in [
        _extract_percentage_answer,
        _extract_version_answer,
        _extract_date_answer,
        _extract_number_answer_generic,
    ]:
        result = extractor(p)
        if result:
            return result

    return None


def extract_all_candidates(query: str, paragraph: str) -> list[dict[str, Any]]:
    """Extract all answer candidates with types and scores."""
    candidates: list[dict[str, Any]] = []
    p = paragraph

    for m in re.finditer(r"\d{4}-\d{2}-\d{2}", p):
        candidates.append({"answer": m.group(), "type": "date", "score": 0.8})

    for m in re.finditer(r"\d+\.?\d*%", p):
        candidates.append({"answer": m.group(), "type": "percentage", "score": 0.7})

    for m in re.finditer(r"v\d+\.\d+(?:\.\d+)?", p):
        candidates.append({"answer": m.group(), "type": "version", "score": 0.7})

    for m in re.finditer(r"\b\d[\d,]+(?:\.\d+)?\b", p):
        val = m.group()
        if not re.match(r"\d{4}-", val) and len(val) > 1:
            candidates.append({"answer": val, "type": "number", "score": 0.5})

    for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", p):
        candidates.append({"answer": m.group(), "type": "name", "score": 0.4})

    # Score boost for candidates near query terms
    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    for c in candidates:
        answer = str(c["answer"])
        pos = p.find(answer)
        if pos >= 0:
            window = p[max(0, pos - 80) : pos + len(answer) + 80].lower()
            overlap = sum(1 for t in q_tokens if t in window)
            c["score"] += overlap * 0.1

    candidates.sort(key=lambda x: -float(x["score"]))
    return candidates


# ── Question type detection ──────────────────────────────────────


def _is_when_question(q: str) -> bool:
    return bool(re.search(r"\bwhen\b|\bwhat date\b|\bwhat time\b", q))


def _is_duration_question(q: str) -> bool:
    return bool(
        re.search(
            r"\bhow many days\b|\bhow many weeks\b|\bhow many months\b"
            r"|\bhow long between\b|\bhow long did\b|\bduration\b",
            q,
        )
    )


def _is_how_many_question(q: str) -> bool:
    return bool(re.search(r"\bhow many\b|\bhow much\b|\bcount\b|\bnumber of\b", q))


def _is_version_question(q: str) -> bool:
    return bool(re.search(r"\bversion\b|\brelease\b|\bv\d", q))


def _is_who_question(q: str) -> bool:
    return bool(re.search(r"\bwho\b|\bwhose\b|\bwhom\b", q))


def _is_yes_no_question(q: str) -> bool:
    return bool(re.search(r"^(is|are|was|were|did|does|do|has|have|can|will|should)\b", q))


def _is_what_percent_question(q: str) -> bool:
    return bool(re.search(r"\bpercent\b|\baccuracy\b|\bscore\b|\brate\b|\b%", q))


# ── Answer extractors ────────────────────────────────────────────


def _extract_date_answer(text: str, query: str = "") -> str | None:
    """Extract the most query-relevant date from text.

    When multiple dates exist, scores each by proximity to query terms
    rather than returning the first match.
    """
    candidates = []
    for m in re.finditer(r"\d{4}-\d{2}-\d{2}", text):
        candidates.append((m.group(), m.start()))
    for m in re.finditer(
        r"((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?)",
        text,
    ):
        candidates.append((m.group(), m.start()))
    if not candidates:
        return None
    if len(candidates) == 1 or not query:
        return candidates[0][0]
    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    best_score, best_date = -1, candidates[0][0]
    for date_str, pos in candidates:
        window = text[max(0, pos - 80) : pos + len(date_str) + 80].lower()
        overlap = sum(1 for t in q_tokens if t in window)
        if overlap > best_score:
            best_score = overlap
            best_date = date_str
    return best_date


def _extract_number_answer(text: str, query: str) -> str | None:
    candidates = []
    for m in re.finditer(r"\b(\d[\d,]*(?:\.\d+)?)\b", text):
        val = m.group(1)
        if re.match(r"20\d{2}$", val) or len(val) == 0:
            continue
        candidates.append((val, m.start()))
    if not candidates:
        return None
    if len(candidates) == 1 or not query:
        return candidates[0][0]
    return _best_by_proximity(candidates, text, query)


def _extract_number_answer_generic(text: str) -> str | None:
    return _extract_number_answer(text, "")


def _extract_version_answer(text: str) -> str | None:
    versions = re.findall(r"v\d+\.\d+(?:\.\d+)?", text)
    if versions:
        return versions[0]
    return None


def _extract_percentage_answer(text: str) -> str | None:
    pcts = re.findall(r"\d+\.?\d*%", text)
    if pcts:
        return pcts[0]
    return None


def _extract_name_answer(text: str, query: str) -> str | None:
    candidates = []
    for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text):
        candidates.append((m.group(), m.start()))
    if not candidates:
        q_tokens = set(re.findall(r"\w{3,}", query.lower()))
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
            pos = m.start()
            window = text[max(0, pos - 50) : pos + 50].lower()
            if any(t in window for t in q_tokens):
                candidates.append((m.group(), m.start()))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]
    return _best_by_proximity(candidates, text, query)


def _extract_yes_no(text: str, query: str) -> str | None:
    t = text.lower()
    _negation_markers = [
        "not ",
        "no ",
        "never ",
        "doesn't ",
        "didn't ",
        "isn't ",
        "wasn't ",
        "weren't ",
        "won't ",
        "wouldn't ",
        "couldn't ",
        "shouldn't ",
        "haven't ",
        "hasn't ",
        "can't ",
        "cannot ",
        "unable ",
        "failed to ",
        "stopped ",
        "quit ",
        "gave up ",
    ]
    q_tokens = set(re.findall(r"\w{3,}", query))
    neg_hits = 0
    pos_hits = 0
    for token in q_tokens:
        pos = t.find(token)
        if pos >= 0:
            window = t[max(0, pos - 40) : pos + 40]
            if any(neg in window for neg in _negation_markers):
                neg_hits += 1
            else:
                pos_hits += 1
    if neg_hits > pos_hits:
        return "No"
    return "Yes"


# ── Duration extraction ─────────────────────────────────────────


def _proximity_score(text: str, date_pos: int, date_len: int, q_tokens: set[str]) -> float:
    """Distance-weighted token proximity score for a date in text.

    Each query token that appears inside the 120-char window around
    *date_pos* contributes ``1 / (1 + d/20)`` where ``d`` is the absolute
    distance in characters from the token's start to the date. A match
    right next to the date counts ~1.0; a match 40 chars away counts
    ~0.33; a match 80 chars away counts ~0.20.

    Tighter than the flat 80-char "presence" window used before Task #35,
    and continuous instead of discrete so ties are rare.
    """
    window_radius = 60
    start = max(0, date_pos - window_radius)
    end = min(len(text), date_pos + date_len + window_radius)
    window = text[start:end].lower()
    score = 0.0
    for tok in q_tokens:
        idx = window.find(tok)
        while idx >= 0:
            # Distance from this token to the date within the window
            token_abs = start + idx
            distance = max(0, date_pos - (token_abs + len(tok)), token_abs - (date_pos + date_len))
            score += 1.0 / (1.0 + distance / 20.0)
            idx = window.find(tok, idx + 1)
    return score


def extract_duration(text: str, query: str) -> str | None:
    """Extract a duration answer by finding two dates and computing the difference.

    For "how many days/weeks/months between X and Y" questions, extracts
    all dates from text, picks the pair most relevant to the query, and
    returns the computed difference. This replaces LLM mental arithmetic
    which is error-prone.

    Task #35: uses distance-weighted proximity scoring instead of flat
    window presence count, so the two closest dates to query keywords
    are picked rather than random tied candidates.
    """
    from datetime import date as _date

    dates = []
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            d = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            dates.append((d, m.start(), m.end() - m.start()))
        except ValueError:
            continue

    if len(dates) < 2:
        return None

    # Distance-weighted proximity scoring
    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    scored = [(_proximity_score(text, pos, dl, q_tokens), d, pos) for d, pos, dl in dates]
    scored.sort(key=lambda x: -x[0])

    # Use top 2 scored dates (or first/last if no overlap)
    if scored[0][0] > 0 and scored[1][0] > 0 and scored[0][1] != scored[1][1]:
        d1, d2 = sorted([scored[0][1], scored[1][1]])
    else:
        by_date = sorted(dates, key=lambda x: x[0])
        d1, d2 = by_date[0][0], by_date[-1][0]

    delta_days = (d2 - d1).days
    inclusive_days = delta_days + 1
    q = query.lower()
    if "week" in q:
        weeks = delta_days // 7
        return f"{weeks} weeks ({delta_days} days exclusive, {inclusive_days} inclusive)"
    if "month" in q:
        months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        return f"{months} months ({delta_days} days exclusive, {inclusive_days} inclusive)"
    return f"{delta_days} days (or {inclusive_days} days counting both endpoints)"


# ── Query-proximity scoring ──────────────────────────────────────


def _best_by_proximity(
    candidates: list[tuple[str, int]],
    text: str,
    query: str,
) -> str:
    """Return the candidate closest to query terms in the text."""
    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    best_score, best_val = -1, candidates[0][0]
    for val, pos in candidates:
        window = text[max(0, pos - 80) : pos + len(val) + 80].lower()
        overlap = sum(1 for t in q_tokens if t in window)
        if overlap > best_score:
            best_score = overlap
            best_val = val
    return best_val


# ── Fuzzy matching ───────────────────────────────────────────────


def fuzzy_match(candidate: str, gold: str, threshold: float = 0.7) -> bool:
    """Check if candidate fuzzy-matches gold answer. Rust-accelerated."""
    try:
        _rust_fuzzy = import_module("remanentia_answer_extractor").fuzzy_match

        return _rust_fuzzy(candidate, gold, threshold)  # pragma: no cover
    except ImportError:
        pass
    if not candidate or not gold:
        return False
    c, g = candidate.lower().strip(), gold.lower().strip()
    if c == g or c in g or g in c:
        return True
    ratio = SequenceMatcher(None, c, g).ratio()
    return ratio >= threshold


# ── Number normalization ─────────────────────────────────────────

_WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}


def normalize_number(text: str) -> str | None:
    """Normalize number words and formats to digits. Rust-accelerated."""
    try:
        _rust_num = import_module("remanentia_answer_extractor").normalize_number

        return _rust_num(text)  # pragma: no cover
    except ImportError:
        pass
    t = text.strip().lower()

    # Already numeric
    m = re.match(r"^([\d,]+\.?\d*)%?$", t)
    if m:
        return m.group(1).replace(",", "")

    # Word numbers: "forty-two", "twenty one"
    parts = re.split(r"[-\s]+", t)
    total = 0
    current = 0
    found_any = False
    for part in parts:
        if part in _WORD_TO_NUM:
            val = _WORD_TO_NUM[part]
            if val >= 100:
                current = max(current, 1) * val
            else:
                current += val
            found_any = True
        elif part == "and":
            continue
        else:
            break

    if found_any:
        total += current
        return str(total)

    return None


# ── Sentence extraction ──────────────────────────────────────────


def extract_best_sentence(query: str, paragraph: str) -> str | None:
    """Return the sentence most relevant to the query. Rust-accelerated."""
    try:
        _rust_best = import_module("remanentia_answer_extractor").extract_best_sentence

        return _rust_best(query, paragraph)  # pragma: no cover
    except ImportError:
        pass
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if not sentences:  # pragma: no cover
        return None

    q_tokens = set(re.findall(r"\w{3,}", query.lower()))
    best_score = 0
    best_sent = None

    for sent in sentences:
        s_tokens = set(re.findall(r"\w{3,}", sent.lower()))
        overlap = len(q_tokens & s_tokens)
        if overlap > best_score:
            best_score = overlap
            best_sent = sent

    return best_sent


# ── LLM backend ───────────────────────────────────────────────────

_BACKEND = None


def set_llm_backend(backend) -> None:
    """Set the module-level LLM backend instance."""
    global _BACKEND
    _BACKEND = backend


def get_llm_backend():
    """Return the current LLM backend (or ``None``)."""
    return _BACKEND


# ── LLM-powered extraction ─────────────────────────────────────


def llm_extract_answer(query: str, paragraph: str, model: str = "remote-default") -> str | None:
    """Extract answer via LLM backend. Fallback when regex returns None.

    Returns 1-2 sentence answer or None.
    """
    if _BACKEND is None:
        return None

    prompt = (
        "Given this context, answer the question in 1-2 sentences. "
        "If the context doesn't contain the answer, say 'unknown'.\n\n"
        f"Context: {paragraph[:1000]}\n\n"
        f"Question: {query}"
    )
    try:
        answer = _BACKEND.complete(prompt, max_tokens=100)
        if not answer:
            return None
        if answer.lower() in ("unknown", "i don't know", "not mentioned"):
            return None
        return answer
    except Exception:
        return None


def llm_generate_prospective_queries(
    paragraph: str, doc_name: str, model: str = "remote-default"
) -> list[str]:
    """Generate hypothetical future queries for a paragraph via LLM.

    Kumiho technique: index by "what queries will need this" rather than
    "what does this say". Measured at 98.5% recall in Kumiho (2026).
    """
    if _BACKEND is None:
        return []

    prompt = (
        "Generate 5 short search queries someone might use to find this information. "
        "Include: factual questions, decision lookups, metric requests, temporal questions. "
        "One query per line, no numbering.\n\n"
        f"Source: {doc_name}\n"
        f"Content: {paragraph[:800]}"
    )
    try:
        text = _BACKEND.complete(prompt, max_tokens=200)
        if not text:
            return []
        queries = [q.strip() for q in text.split("\n") if q.strip() and len(q.strip()) > 5]
        return queries[:8]
    except Exception:
        return []


def llm_synthesize_answer(
    query: str, paragraphs: list[str], model: str = "remote-default"
) -> str | None:
    """Synthesize an answer from multiple retrieved paragraphs.

    Unlike llm_extract_answer (single paragraph), this reasons across
    multiple sources to produce a grounded, cited answer.
    Uses question-type-specific prompts for better accuracy.
    """
    if _BACKEND is None:
        return None

    context = "\n\n---\n\n".join(
        f"[Source {i + 1}]: {p[:600]}" for i, p in enumerate(paragraphs[:10])
    )

    # Question-type-specific prompt (improves counterfactual/temporal)
    q_lower = query.lower()
    if any(w in q_lower for w in ["would", "could", "might", "likely"]):
        system_prompt = (
            "Answer the hypothetical question by reasoning about the person's "
            "stated preferences, personality, and past actions from the sources. "
            "Answer with 'Yes' or 'No' followed by a brief reason. "
            "If insufficient information, say 'unknown'."
        )
    elif any(
        w in q_lower
        for w in ["what are", "what does", "list", "hobbies", "interests", "activities"]
    ):
        system_prompt = (
            "List ALL relevant items mentioned across ALL sources. "
            "Combine information from different sources into one complete answer. "
            "If the answer isn't in the sources, say 'unknown'."
        )
    else:
        system_prompt = (
            "Answer the question using ONLY the provided sources. "
            "Be concise (1-3 sentences). "
            "If sources don't contain the answer, say 'unknown'."
        )

    try:
        answer = _BACKEND.complete(
            f"{system_prompt}\n\n{context}\n\nQuestion: {query}",
            max_tokens=200,
        )
        if not answer:
            return None
        if answer.lower() in ("unknown", "i don't know", "not mentioned"):
            return None
        return answer
    except Exception:
        return None
