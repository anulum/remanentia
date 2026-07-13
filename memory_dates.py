# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Memory date extraction and ranking

"""Extract and rank dates carried by indexed memory documents."""

from __future__ import annotations

import re
from datetime import date, datetime

DATE_EXPRESSION = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
    r"|\b(?:yesterday|today|last\s+(?:week|month|year))\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b",
    re.IGNORECASE,
)


def has_date_expression(text: str) -> bool:
    """Return whether text contains a supported date expression."""
    return bool(DATE_EXPRESSION.search(text))


def recency_boost(date_str: str) -> float:
    """Return the retrieval multiplier for a document date."""
    try:
        document_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        days_ago = (date.today() - document_date).days
        if days_ago <= 2:
            return 1.8
        if days_ago <= 5:
            return 1.4
        if days_ago <= 14:
            return 1.2
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def extract_date_context(text: str) -> list[tuple[str, str]]:
    """Return ISO dates paired with their surrounding text."""
    results = []
    for match in re.finditer(r"(\d{4}-\d{2}-\d{2})", text):
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 100)
        results.append((match.group(1), text[start:end].strip()))
    return results


def parse_document_date(text: str, filename: str) -> str:
    """Return the first ISO date, preferring the filename over content."""
    filename_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if filename_match:
        return filename_match.group(1)
    content_match = re.search(r"(\d{4}-\d{2}-\d{2})", text[:500])
    return content_match.group(1) if content_match else ""
