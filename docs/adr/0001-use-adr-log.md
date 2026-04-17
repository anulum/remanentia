# ADR-0001: Use an ADR log for architectural decisions

- **Status:** Accepted
- **Date:** 2026-04-17

## Context

Three non-obvious architectural decisions (two retrieval stacks,
pickle fallback kept for months, the Protocol-based LLM backend as
the only extension seam) were flagged in the 2026-04-17 comprehensive
gap audit as implicit. Future readers had no way to tell whether
they were design choices, accidents, or tech debt. The fix is a
small ADR log beside the code.

## Decision

Record architectural decisions in `docs/adr/NNNN-title.md`, MADR 3.0
format, with Context / Decision / Options / Consequences sections.
Index maintained in `docs/adr/README.md`.

## Options considered

- **GitHub Discussions.** Low friction, but no versioning with the
  code and hard to grep from a code review.
- **CHANGELOG only.** Captures _what_ shipped, not _why_.
- **ADR log (chosen).** Small, self-contained, evolves with the code,
  survives repository migrations.

## Consequences

- Positive: every non-trivial structural change leaves a paper trail;
  new contributors can read the history in one pass.
- Negative: mild ceremony per decision.
- Follow-up: write ADRs 0002-0005 for the decisions the gap audit
  pulled out of implicit state.
