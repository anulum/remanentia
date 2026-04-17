# Remanentia Model Cards

Each trained component in Remanentia ships with a model card that
states what the model does, what it was trained on, how it was
evaluated, and — crucially — what it does **not** do. We publish the
cards here, including the ones that document failures, so downstream
users can decide whether to rely on a component or replace it.

The five components covered:

| Card | Status | Summary |
|------|--------|---------|
| [Embedding fine-tune (C1)](embedding.md) | UNCERTAIN | 3 464 triplets on `all-MiniLM-L6-v2`; retrieval A/B unmeasured |
| [Cross-encoder fine-tune (C2)](cross_encoder.md) | UNCERTAIN | AP 84.57 % on own split; real-world reranking effect unmeasured |
| [Temporal relation classifier (C3)](relation.md) | **NON-FUNCTIONAL** | F1-macro 0.178, random baseline 0.167; ships but should not be used |
| [Date normaliser (C4)](date_normalizer.md) | RULE ENGINE SHIPS; ML WEAK | 12 regex patterns deterministic; ML part mediocre (65.7 % relaxed) |
| [Fact-validity model (C5)](fact_validity.md) | OVERFIT TO SYNTHETIC | 100 % on synthetic templates; real conversational performance unknown |

All five cards derive from `training/HONEST_ASSESSMENT.md` (2026-03-29),
the repo's internal post-mortem of the temporal-training programme.
The cards are the public version of that document; users who read only
the README should still know what they're getting.

## Component-level rollup

The rule-based date normaliser (C4 "rules") and the reference-date
pipeline wiring were the two changes that actually moved LongMemEval
temporal-reasoning from 45.9 % to 60.2 % in benchmarking.  The ML
components contributed marginally at best. This is why the current
retrieval pipeline prefers the rule-based path:

- **C4 rules** drive +66.9 pp date coverage (23.3 % → 90.2 %).
- **C1/C2 fine-tunes** have no committed A/B number.
- **C3 is inactive** by default — the classifier's F1 is at chance.
- **C5 is gated** by regex first; ML only overrides catch-all "event".

## Evaluation cadence

We re-evaluate the retrieval path (C1/C2) when we change the fact
pipeline or the cross-encoder rerank stage. Models that ship broken
(C3) are kept in the tree as training reproducibility artefacts but
are not wired into any default code path. If you want to retrain, the
scripts in `training/` are the entry points; please update the
corresponding card with your new numbers before submitting a PR.
