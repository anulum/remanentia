# ADR-0005: Prefer rule-based date normalisation over ML

- **Status:** Accepted
- **Date:** 2026-04-17 (retroactive; rules landed 2026-03-29)
- **Module:** `date_normalizer.py`

## Context

LongMemEval's temporal-reasoning category failed at 45.9 % in R8.
Root cause was that vague date expressions ("3 weeks ago", "last
Tuesday") were never resolved against the session timestamp; the
LLM saw "3 weeks ago" and guessed. We designed C4 to fix this.

Two implementations were built:

- **12-pattern rule engine** — deterministic regex + ISO arithmetic
  per pattern class (quantified offset, weekday, vague, absolute,
  relative, quarter).
- **ML companion** — DistilBERT fine-tuned as a day-offset
  regressor. ``training/train_date_normalizer.py``.

## Decision

Ship and wire the rule engine. Keep the ML companion as a training
reproducibility artefact but do **not** route runtime traffic through
it. Document both in `docs/models/date_normalizer.md`.

## Options considered

- **Rules only (chosen).** Deterministic, high confidence on
  observed phrases, benchmark-measured (+66.9 pp date coverage,
  +14.3 pp temporal-reasoning).
- **ML only.** Achieved 24.8 % exact / 65.7 % relaxed (±3 d) — much
  weaker than the rule engine on the patterns that matter.
- **Hybrid.** Rules first, ML fallback for unmatched patterns. No
  committed evidence that the ML component adds signal beyond rules.

## Consequences

- Positive: deterministic output; zero false positives by
  construction; no model download in the hot path; reproducible
  across Python versions.
- Negative: English-only. Non-English date phrases (Slovak, German,
  CJK) require hand-adding pattern families. Tracked as a
  multilingual-support gap.
- Follow-up: extend the pattern set to cover Slovak first, given
  the target user base.
