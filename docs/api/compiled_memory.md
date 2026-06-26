# compiled_memory

`compiled_memory` converts durable evidence into typed fact cards that the
retrieval stack can index and search.

## Seed Facts

Static historical and cross-project facts live in
`data/compiled_seed_facts.jsonl`. The compiler loads that JSONL file before it
adds facts derived from live index snapshots, benchmark reports, source files,
and local operational notes.

This keeps long-lived factual records out of Python control flow. Public release
cleanup can review or replace the seed data without editing the compiler.

## Outputs

`compile_facts()` writes:

- `memory/compiled/facts.jsonl`
- `memory/compiled/compiled_facts.md`

Both outputs are generated runtime memory inputs and stay outside tracked public
source by default.

::: compiled_memory.CompiledFact
    options:
      show_source: true
      members_order: source

::: compiled_memory.compile_facts

::: compiled_memory.load_compiled_facts

::: compiled_memory.search_compiled_facts
