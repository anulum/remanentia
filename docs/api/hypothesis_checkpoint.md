# hypothesis_checkpoint

Append-only JSONL checkpointing for long benchmark runs — every hypothesis
record is durable the moment it is produced, and an interrupted run resumes
instead of restarting.

## Why this module exists

A full-S LongMemEval pass over a slow local reader takes tens of hours (the
first sovereign run measured ~30.5 h on one card). Holding every hypothesis in
memory and flushing at the end means one interruption — an OOM reap, a power
blip, a mistaken kill — discards the whole run. This module writes each record
as it is produced and reloads the completed ones on resume.

## Public surface

```python
from hypothesis_checkpoint import append_record, completed_ids, load_completed
```

- `append_record(path, record)` — append one hypothesis record as a JSONL line.
- `load_completed(path)` — reload completed records; a blank or half-written
  final line left by an interrupted append is skipped, not fatal.
- `completed_ids(records)` — the question-id set already answered, used to skip
  work on resume.

## Invariants

- **Crash-tolerant load.** Only the final line may be damaged by an interrupted
  append; the loader tolerates exactly that and never silently drops a
  well-formed record.
- **Append-only.** Records are never rewritten in place; resume is a pure
  read-then-skip.

## See also

- `bench_longmemeval` — checkpoints each emitted `hypothesis_record` and
  resumes from the JSONL.
- `benchmark_evidence` — the evidence layer the completed run feeds.
