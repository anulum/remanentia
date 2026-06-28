# reflector

Periodic cluster summarisation and gap detection over `KnowledgeStore` notes.

The reflector is Remanentia's no-egress consolidation digest pass. It loads
recent persisted knowledge notes, groups related notes by keyword/entity
overlap, emits heuristic or optional LLM summaries, identifies decisions without
measured outcomes, reports unresolved supersession chains, and writes a
human-readable digest under `memory/digests/`.

## Runtime Contract

- `reflect_once(days=7, use_llm=False)` is deterministic and local-only by
  default.
- `use_llm=True` calls the configured answer-extractor backend for cluster
  summaries and prospective queries; the Python heuristic path remains the
  fallback.
- `remanentia_consolidation.cluster_notes` accelerates clustering when the Rust
  extension is installed. The Python fallback keeps the same cluster-index
  contract and is covered by dedicated tests.
- The result payload uses `reflector.ReflectionResult`, so callers can inspect
  status, note counts, cluster counts, gap counts, contradiction counts, digest
  path, and digest text without parsing Markdown.

## Coverage

`tests/test_reflector.py` exercises the source at 100% isolated line coverage:

- native and Python clustering paths
- heuristic and optional-backend summary generation
- prospective-query generation and backend failure handling
- gap and contradiction detection
- real `KnowledgeStore.save()` / `KnowledgeStore.load()` reflection paths
- digest-file emission under patched storage roots

::: reflector
    options:
      show_source: true
      members_order: source
