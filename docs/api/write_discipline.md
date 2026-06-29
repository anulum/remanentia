# write_discipline

The ecosystem write-discipline gate. Remanentia is the memory authority for the
fleet, so it owns — and enforces — the structural contract that governs what may
become retrievable memory. This is the "enforced gate, not goodwill" the MS.0
audit called for.

## Why this module exists

Retrieval quality is bounded by what is stored and how it is structured. The
MS.0 audit found the write side undisciplined: actor/event fields 84–99 % empty,
three timestamp formats, uncontrolled project vocabulary. The normalisation layer
(`feed_normalization`) *defaults* missing fields, which hides the omission — a
record with no producer becomes `actor="synapse"`, a record with no time becomes
`0.0`. Defaulting is not enforcement.

This module inspects a write **before** defaults are applied, so a missing field
is caught as a violation rather than silently back-filled, and attributes every
violation to its producer so the contract can be enforced by evidence.

## Disposition

- **accepted** — satisfies the contract; safe to normalise and index.
- **quarantined** — has usable content but violates discipline; held out of the
  retrievable index (recoverable) and attributed to its producer.
- **rejected** — structurally useless (no/short content); never enters memory.

The default policy is lenient (quarantine + attribute) so a rollout can measure
the violation surface without losing data. Set `strict=True` to reject any
violation at source once producers conform.

## Public surface

```python
from write_discipline import (
    WriteContract, FieldMap, DisciplineVerdict,
    inspect_write, producer_label,
    DisciplineLedger, ProducerRecord, audit_records,
    load_stimulus_records, build_memory_record,
)
```

### `build_memory_record(content, project, actor, *, timestamp=None, entities=None, kind=None, source_ref=None) -> dict`

The writer-side complement to `inspect_write`: the easiest way to emit a
conformant write is to construct it here. Producers must supply real
`content` / `project` / `actor` (each raises `ValueError` if empty); `timestamp`
defaults to the wall clock (real provenance at write time, not a sentinel);
`project` is normalised to an uppercase slug and `actor` to its controlled role.
Fleet writers should adopt this instead of hand-building stimulus dicts.

### `WriteContract`

The contract: `require_project` / `require_actor` / `require_timestamp` /
`require_entities` flags, `min_content_chars`, an optional `known_projects`
allowlist, a `strict` policy flag, and a `FieldMap`.

### `FieldMap`

Which record keys hold each contract field, so the gate is shape-agnostic across
the fleet's many write shapes (SNN stimuli `text`/`source`; feed findings
`statement`/`provenance`; status rows `event`/`summary`). Each field accepts
several candidate keys, tried in order.

### `inspect_write(record, *, contract) -> DisciplineVerdict`

Inspect one raw write. Returns the disposition, the normalised `PROJECT/actor`
producer, and the tuple of violation codes (`missing_content`,
`content_too_short`, `missing_project`, `uncontrolled_project`, `missing_actor`,
`missing_timestamp`, `missing_entities`). The verdict is truthy when accepted.

### `DisciplineLedger` / `audit_records(records, *, contract)`

Accumulate verdicts and attribute violations per producer. `as_report()` returns
a JSON-serialisable accountability report (global counts, conformance, and the
worst producers with their violation breakdown) — the artefact that makes
ecosystem write-discipline enforceable by name.

### `load_stimulus_records(directory)`

Load every `*.json` stimulus mapping under a directory, skipping malformed or
non-object files, for auditing a real firehose.

## CLI

```
remanentia-write-discipline [DIRECTORY] [--require-entities] [--strict] [--worst N]
```

Audits a stimulus directory (default `$REMANENTIA_STIMULI_DIR` or `snn_stimuli`)
and prints the per-producer accountability report as JSON.

## Invariants

- **Catches omission, not just shape.** A missing field is a violation, never a
  silent default — that is the whole point.
- **Attribution.** Every verdict names its producer, so discipline is enforced
  by evidence, not exhortation.
- **Non-destructive by default.** Lenient mode quarantines rather than drops, so
  an enforcement rollout never loses recoverable data.
- **Deterministic.** Same record + same contract → same verdict.

## See also

- `feed_normalization` — the controlled-vocab normaliser the gate builds on.
- `feed_ingest` / `finding_ingest` — live ingest paths the gate guards.
