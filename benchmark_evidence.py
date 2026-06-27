# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark judge evidence helpers

"""Build row-level judge evidence for benchmark result artefacts."""

from __future__ import annotations

import hashlib
import re
from typing import Mapping, cast

JsonDict = dict[str, object]


def prompt_sha256(prompt: str) -> str:
    """Return the SHA-256 digest for a judge prompt.

    Parameters
    ----------
    prompt
        Exact judge prompt text.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Return a deterministic whitespace-token estimate for report accounting.

    The benchmark APIs can return provider token usage, but older result files
    do not contain it. This estimate is intentionally simple and stable; report
    consumers can distinguish it from exact API usage via the field names.
    """
    return len(re.findall(r"\S+", text))


def _int_from_value(value: object) -> int | None:
    """Coerce integral values to ``int`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _usage_int(usage: object, key: str) -> int | None:
    """Extract an integer usage attribute from mapping or object usage data."""
    if isinstance(usage, Mapping):
        return _int_from_value(usage.get(key))
    return _int_from_value(cast(object, getattr(usage, key, None)))


def build_judge_evidence(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    latency_ms: float,
    usage: object | None = None,
) -> JsonDict:
    """Build row-level judge evidence for a benchmark result record.

    Parameters
    ----------
    prompt
        Exact prompt sent to the judge.
    model
        Judge model identifier.
    max_tokens
        Maximum judge completion tokens requested.
    latency_ms
        Wall-clock judge-call latency in milliseconds.
    usage
        Optional API usage object or mapping with ``prompt_tokens``,
        ``completion_tokens``, and ``total_tokens``.

    Returns
    -------
    dict[str, object]
        JSON-serialisable metadata suitable for merging into a result row.
    """
    evidence: JsonDict = {
        "judge_model": model,
        "judge_max_tokens": max_tokens,
        "judge_prompt_sha256": prompt_sha256(prompt),
        "judge_prompt_chars": len(prompt),
        "judge_prompt_tokens_estimate": estimate_tokens(prompt),
        "judge_latency_ms": round(latency_ms, 3),
    }
    if usage is not None:
        for source_key, output_key in (
            ("prompt_tokens", "judge_prompt_tokens"),
            ("completion_tokens", "judge_completion_tokens"),
            ("total_tokens", "judge_total_tokens"),
        ):
            value = _usage_int(usage, source_key)
            if value is not None:
                evidence[output_key] = value
    return evidence
