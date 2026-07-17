# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval query intelligence and prospective queries

"""Classify retrieval text and generate prospective lookup queries."""

from __future__ import annotations

import re
from importlib import import_module
from typing import Any, cast


def tokenize_python(text: str) -> list[str]:
    """Tokenize lowercase three-character BM25 terms without native code."""
    return re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower())


def tokenize(text: str) -> list[str]:
    """Tokenize with the installed native engine when available."""
    try:
        native_tokenize = import_module("remanentia_search").tokenize
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return tokenize_python(text)
    return cast(list[str], native_tokenize(text))


def reciprocal_rank_fusion_python(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse heterogeneous rankings with standard reciprocal-rank weights."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (paragraph_index, _score) in enumerate(ranked):
            scores[paragraph_index] = scores.get(paragraph_index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse rankings with the installed native engine when available."""
    try:
        native_rrf = import_module("remanentia_retrieve").reciprocal_rank_fusion
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return reciprocal_rank_fusion_python(ranked_lists, k)
    return cast(list[tuple[int, float]], native_rrf(ranked_lists, k))  # pragma: no cover


def classify_query(query: str) -> dict[str, Any]:
    """Classify query intent for search routing."""
    normalized = query.lower()
    intent: dict[str, Any] = {
        "type": "general",
        "boost_types": [],
        "date_filter": None,
        "recency": False,
    }

    if any(
        word in normalized for word in ("where is", "find the", "locate", "which file", "what file")
    ):
        intent["type"] = "location"
        intent["boost_types"] = ["function", "code"]
    elif any(
        word in normalized
        for word in ("what did we decide", "decision", "chose", "rejected", "why did we")
    ):
        intent["type"] = "decision"
        intent["boost_types"] = ["decision"]
    elif any(word in normalized for word in ("what went wrong", "failure", "bug", "error", "fix")):
        intent["type"] = "debugging"
        intent["boost_types"] = ["finding", "decision"]
    elif any(word in normalized for word in ("status", "progress", "current", "latest")):
        intent["type"] = "status"
        intent["recency"] = True
    elif any(
        word in normalized for word in ("performance", "benchmark", "accuracy", "score", "percent")
    ):
        intent["type"] = "metric"
        intent["boost_types"] = ["metric"]
    elif any(
        word in normalized
        for word in ("when", "date", "timeline", "before", "after", "first", "last")
    ):
        intent["type"] = "temporal"
        intent["recency"] = any(word in normalized for word in ("latest", "recent", "last"))
    elif any(word in normalized for word in ("how does", "how to", "explain", "what is")):
        intent["type"] = "explanation"
        intent["boost_types"] = ["function", "finding"]

    return intent


def classify_paragraph(text: str, is_code: bool = False) -> str:
    """Return the semantic retrieval type of a paragraph."""
    try:
        rust_classifier = import_module("remanentia_search").classify_paragraph
        return cast(str, rust_classifier(text, is_code))  # pragma: no cover
    except ImportError:  # pragma: no cover - optional extension is installed in CI
        return classify_paragraph_python(text, is_code)


def classify_paragraph_python(text: str, is_code: bool = False) -> str:
    """Classify a paragraph without the optional native extension."""
    normalized = text.lower()

    if is_code:
        if re.match(r"\s*(def |fn |pub fn |class |impl )", text):
            return "function"
        return "code"
    if any(
        word in normalized
        for word in ("decided", "decision", "chose", "rejected", "we will", "the plan")
    ):
        return "decision"
    if any(
        word in normalized
        for word in ("found", "finding", "result", "measured", "shows that", "proved")
    ):
        return "finding"
    if any(
        word in normalized
        for word in ("P@1", "percent", "accuracy", "precision", "score", "benchmark")
    ):
        return "metric"
    if any(
        word in normalized for word in ("version", "v0.", "v1.", "v2.", "v3.", "release", "shipped")
    ):
        return "version"
    return "discussion"


def generate_prospective_queries(text: str, document_name: str, paragraph_type: str) -> list[str]:
    """Generate bounded hypothetical future queries for one paragraph."""
    queries: list[str] = []
    capitalized = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    for phrase in capitalized[:5]:
        if len(phrase) > 3:
            queries.extend((f"what is {phrase}", phrase.lower()))

    functions = re.findall(r"(?:def |fn |class )\s*(\w+)", text)
    for function in functions[:3]:
        queries.extend((f"where is {function}", f"how does {function} work", function))

    person = capitalized[0].lower() if capitalized else "the person"
    for match in re.finditer(
        r"(?:likes?|loves?|enjoys?|prefers?|hates?|dislikes?|"
        r"interested in|passionate about|into)\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        activity = match.group(1).strip().lower()
        queries.extend(
            (
                f"hobbies {activity}",
                f"interests {activity}",
                f"what does {person} like",
                activity,
            )
        )

    for match in re.finditer(
        r"(?:works? (?:as|at|for)|employed (?:at|by)|is a |job (?:is|as))\s+"
        r"(.{3,40}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        role = match.group(1).strip().lower()
        queries.extend((f"where does {person} work", f"job {role}", f"career {role}", role))

    for _match in re.finditer(
        r"(?:married to|dating|friends? with|partner|spouse|sibling|brother|sister)"
        r"\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        queries.extend(("relationship status", f"who is {person} dating"))

    for match in re.finditer(
        r"(?:allergic to|allergy|intolerant|sensitive to|cannot eat|vegetarian|vegan)"
        r"\s*(.{0,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        subject = match.group(1).strip().lower()
        queries.extend((f"allergic {subject}", f"what is {person} allergic to"))

    for match in re.finditer(
        r"(?:went to|visited|trip to|lives? in|moved to|from|travel(?:led|ed)? to)\s+"
        r"(.{3,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        place = match.group(1).strip().lower()
        queries.extend((f"where did {person} go", f"trip {place}", place))

    for match in re.finditer(
        r"(?:learning|studying|started|taking up|practicing)\s+"
        r"(.{3,30}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        skill = match.group(1).strip().lower()
        queries.extend((f"what is {person} learning", skill))

    for match in re.finditer(
        r"(?:favou?rite)\s+(\w+)\s+(?:is|was)\s+(.{3,40}?)(?:[.,;!?\n]|$)",
        text,
        re.I,
    ):
        queries.extend((f"favourite {match.group(1).lower()}", match.group(2).strip().lower()))
    for match in re.finditer(r"(?:favou?rite)\s+(.{3,40}?)(?:[.,;!?\n]|$)", text, re.I):
        queries.append(f"favourite {match.group(1).strip().lower()}")

    if paragraph_type == "decision":
        subjects = re.findall(
            r"(?:decided|chose|rejected|will)\s+(?:to\s+)?(.{10,40}?)(?:\.|,|$)",
            text,
            re.I,
        )
        for subject in subjects[:2]:
            normalized = subject.strip().lower()
            queries.extend((f"why did we {normalized}", f"what did we decide about {normalized}"))
    if paragraph_type == "finding":
        subject = document_name.replace(".md", "").replace("_", " ")
        queries.append(f"what did we find about {subject}")
    if paragraph_type == "metric":
        for number in re.findall(r"\d+\.?\d*%", text)[:2]:
            queries.append(f"what score {number}")

    for version in re.findall(r"v\d+\.\d+(?:\.\d+)?", text)[:2]:
        queries.extend((f"what version {version}", f"when was {version} released"))
    for date in re.findall(r"\d{4}-\d{2}-\d{2}", text)[:2]:
        queries.append(f"what happened on {date}")

    if ".py" in document_name or ".rs" in document_name:
        base_name = document_name.split(".")[0]
        queries.extend((f"what does {base_name} do", f"where is {base_name}"))

    return list(dict.fromkeys(queries))[:20]
