# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Guarded-tier bridge to Director-AI

"""Director-AI bridge for Remanentia's Guarded memory tier.

The Guarded tier is the optional grounding layer that sits between a
memory retrieval result and whatever the downstream LLM synthesises
from it. Before an answer leaves the server, we ask Director-AI's
`score(prompt, response, facts=…)` whether the response stays
grounded in the retrieved memory, then either pass, flag, or block
depending on a configurable threshold.

The integration is opt-in:

- ``director-ai`` lives in the ``[guarded]`` extras of
  ``pyproject.toml``; default installs do not pull it.
- :func:`is_available` reports whether the optional dep is importable
  so the MCP server can degrade gracefully when it isn't.
- :class:`GuardedResult` mirrors the bits of Director-AI's
  ``CoherenceScore`` that Remanentia actually uses, so downstream
  code never has to import director_ai directly.

Design notes:

1. **Facts are keyed by memory name or id.** Director-AI's ``facts``
   parameter expects a ``dict[str, str]`` where the key is a stable
   identifier and the value is the evidence text. We map each
   retrieval result to ``{result.name: result.snippet_or_answer}``.
2. **Default threshold 0.3** matches Director-AI's own default. The
   Guarded tier's stronger "block" threshold is policy-configurable
   per caller via :class:`GuardedPolicy`.
3. **No network calls.** Director-AI loads a local cross-encoder /
   NLI model on first use; there's no external API.
4. **Audit-trail ready.** :class:`GuardedResult` carries the full
   per-chunk evidence so downstream audit logs can record *which*
   memory fragment grounded the decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ─── Optional dep probe ──────────────────────────────────────────────


def is_available() -> bool:
    """True iff the ``director-ai`` optional dependency is installed."""
    try:
        import director_ai  # noqa: F401
    except ImportError:
        return False
    return True


# ─── Policy + result shapes ──────────────────────────────────────────


@dataclass(frozen=True)
class GuardedPolicy:
    """What "passing" means for a given caller.

    - ``approve_threshold`` — ``CoherenceScore.approved`` is Director-AI's
      own boolean, but we keep this as a second gate so callers can be
      stricter than Director-AI's default 0.3.
    - ``block_below`` — scores strictly below this are blocked. The
      MCP layer treats a blocked response as if the LLM returned
      nothing, falling back to raw retrieval context.
    - ``use_nli`` — surfacing the underlying NLI model's opinion
      separately from the factual-grounding score. ``None`` means
      Director-AI picks; ``True`` / ``False`` override.
    - ``injection_detection`` — Director-AI 3.12+ emits a prompt-injection
      risk score; on by default so the Guarded tier also catches
      prompt-injected memory fragments. Off in tests.
    """

    approve_threshold: float = 0.3
    block_below: float = 0.15
    use_nli: bool | None = None
    injection_detection: bool = True


DEFAULT_POLICY = GuardedPolicy()


@dataclass
class GuardedResult:
    """Remanentia-facing view of a Director-AI coherence score."""

    score: float
    approved: bool
    blocked: bool
    h_logical: float
    h_factual: float
    injection_risk: float | None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Scoring ─────────────────────────────────────────────────────────


def _evidence_to_dicts(ev: Any) -> list[dict[str, Any]]:
    """Normalise Director-AI ``ScoringEvidence`` into plain dicts."""
    chunks = getattr(ev, "chunks", None) or []
    out: list[dict[str, Any]] = []
    for c in chunks:
        out.append(
            {
                "text": getattr(c, "text", "")[:400],
                "distance": float(getattr(c, "distance", 0.0)),
                "source": getattr(c, "source", "unknown"),
            }
        )
    return out


def score_memory_answer(
    prompt: str,
    response: str,
    memory_facts: dict[str, str],
    *,
    policy: GuardedPolicy | None = None,
) -> GuardedResult | None:
    """Grade *response* against *memory_facts* under *policy*.

    Returns ``None`` if Director-AI is not installed — callers should
    treat that as "Guarded tier unavailable, fall back to ungraded
    retrieval" rather than as an error.

    The ``memory_facts`` dict is keyed by a stable fact identifier
    (memory name / path / id). Director-AI scores the response's
    grounding against the union of those snippets.
    """
    if not is_available():
        return None

    policy = policy or DEFAULT_POLICY

    from director_ai import score as _director_score

    raw = _director_score(
        prompt,
        response,
        facts=memory_facts,
        threshold=policy.approve_threshold,
        use_nli=policy.use_nli,
        injection_detection=policy.injection_detection,
    )

    s = float(raw.score)
    blocked = s < policy.block_below
    reason = ""
    if blocked:
        reason = f"score {s:.3f} < block_below {policy.block_below:.3f}"
    elif not raw.approved:
        reason = f"score {s:.3f} < approve_threshold {policy.approve_threshold:.3f}"

    injection_risk: float | None = None
    if policy.injection_detection:
        injection_risk = float(getattr(raw, "injection_risk", 0.0) or 0.0)

    return GuardedResult(
        score=s,
        approved=bool(raw.approved and not blocked),
        blocked=blocked,
        h_logical=float(getattr(raw, "h_logical", 0.0)),
        h_factual=float(getattr(raw, "h_factual", 0.0)),
        injection_risk=injection_risk,
        evidence=_evidence_to_dicts(getattr(raw, "evidence", None)),
        reason=reason,
    )


def facts_from_results(results: list[Any]) -> dict[str, str]:
    """Build a ``memory_facts`` dict from a list of ``MemoryIndex.search`` hits.

    Prefers the extracted ``answer`` when present (it is the concrete
    span Remanentia believes answers the query); otherwise falls back
    to the first 400 characters of the snippet. Names are
    deduplicated; duplicates keep the first occurrence.
    """
    facts: dict[str, str] = {}
    for r in results:
        name = getattr(r, "name", None)
        if not name:
            continue
        if name in facts:
            continue
        text = getattr(r, "answer", None) or getattr(r, "snippet", "") or ""
        if text:
            facts[name] = text[:400]
    return facts
