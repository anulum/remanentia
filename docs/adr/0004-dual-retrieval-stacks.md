# ADR-0004: Keep two retrieval stacks until consolidation (P4-24)

- **Status:** Accepted · superseded-by **P4-24** (not yet done)
- **Date:** 2026-04-17 (retroactive ADR)

## Context

Retrieval goes through two parallel code paths:

1. ``MemoryIndex`` in ``memory_index.py`` (1 959 LoC): BM25 +
   bi-encoder + cross-encoder rerank + 4-stage answer extractor.
2. ``ArcaneRetriever`` in ``arcane_retriever.py`` (~500 LoC): four
   parallel channels (FAST, WORKING, EPISODIC, SEMANTIC) with
   RRF fusion.

The LongMemEval bench routes temporal, multi-session,
knowledge-update and single-session-preference to ``ArcaneRetriever``;
single-session factoid goes to ``MemoryIndex``. The WORKING channel
in ``ArcaneRetriever`` currently calls ``FactIndex.query`` which is
structurally identical to the FAST channel, as recorded in the private v0.4
integration gap audit.

## Decision

Keep both stacks for the current release. Do **not** invest in
convergence until the multi-session audit recommendations (R2-R5)
have landed and stabilised, because moving the stack boundary would
invalidate the per-qtype numbers we are trying to measure.

The long-term plan is one formal owner per qtype. That work is
tracked as **P4-24 ArcaneRetriever / MemoryIndex consolidation** in
the gap audit and will supersede this ADR.

## Options considered

- **Unify now.** Would reshuffle benchmark numbers in the middle of
  the audit cycle; no way to tell whether the new score reflects
  retrieval or unification churn.
- **Keep both, formalise the dispatch.** Chosen. Also explicit in the
  bench's hybrid routing.
- **Delete ``ArcaneRetriever``.** Loses the 4-channel RRF scoring
  that drives temporal and multi-session gains.

## Consequences

- Positive: we can measure each R-recommendation against a stable
  retrieval boundary.
- Negative: maintenance cost of two retrievers; latent WORKING/FAST
  duplication; no single source of truth for "where a fact came
  from".
- Follow-up: P4-24. Until it lands, every retrieval-side ADR must
  declare which stack it modifies.
