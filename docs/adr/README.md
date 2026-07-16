# Architecture Decision Records

This directory holds the ADR trail for Remanentia. An ADR captures
**one** architectural decision, the forces behind it, the options
considered, and the consequences — in a small fixed-format file that
lives beside the code.

We follow the [MADR 3.0](https://adr.github.io/madr/) convention with
minor adjustments: numbered files (``NNNN-short-title.md``), three
life-cycle states (``Proposed`` · ``Accepted`` · ``Superseded by …``),
and a required **Consequences** section so the trade-offs stay
visible after the fact.

## Why bother

Before this log existed, three non-obvious architectural decisions
were implicit in the codebase:

1. Retrieval runs through *two* parallel stacks (``MemoryIndex`` and
   ``ArcaneRetriever``) with no formal owner per code path.
2. Persistent state serialised with ``pickle`` for months after the
   security-risk was flagged, with a fallback kept "for compat".
3. The pluggable LLM backend Protocol is the only seam available to
   site-specific overrides; there is no plugin-system, and every
   non-backend extension has been a fork.

The first two were resolved (see ADR 0002 and 0004); the third is
still open and documented here so future contributors know the
state of play.

## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-use-adr-log.md) | Use an ADR log for architectural decisions | Accepted |
| [0002](0002-remove-pickle-runtime.md) | Remove pickle at runtime, ship one-shot migrator | Accepted (2026-04-17) |
| [0003](0003-pluggable-llm-backend-protocol.md) | Pluggable LLM backend via Protocol | Accepted |
| [0004](0004-dual-retrieval-stacks.md) | Keep two retrieval stacks until consolidation | Accepted · to be superseded by P4-24 |
| [0005](0005-rule-based-date-normalisation.md) | Prefer rule-based date normalisation over ML | Accepted |
| [0006](0006-preregister-temporal-snn-memory-experiment.md) | Preregister the temporal SNN memory experiment | Accepted |

## How to add an ADR

1. Copy [`0000-template.md`](0000-template.md) and rename to the next
   zero-padded number plus a kebab-case title.
2. Fill in the sections. Keep **Context** to the forces that pushed
   the decision and **Consequences** to what you are now committed to
   (including the negatives).
3. Add a row to the index table above. State should start at
   ``Proposed`` and move to ``Accepted`` once the commit lands.
4. Never edit an ADR's conclusions after acceptance. If the decision
   changes, write a new ADR that supersedes it and add a
   ``Superseded by`` link in the original.
