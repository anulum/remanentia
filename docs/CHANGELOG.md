# Changelog

All notable changes to Remanentia are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Rust STDP/LIF acceleration (arcane_stdp) — 2–3x speedup, bit-exact with numpy
- Enterprise hardening: ARCHITECTURE, CONTRIBUTING, SECURITY, GOVERNANCE, ROADMAP, VALIDATION, NOTICE, CITATION.cff
- MkDocs documentation site with 4 guides, 9 API refs, benchmarks, research
- GitHub issue/PR templates, dependabot, CodeQL, Scorecard, stale bot workflows
- README badges, header image, branding footer

### Fixed
- CI: PEP 639 license classifier conflict
- CI: mock anthropic import in _get_client tests
- SPDX headers: Sotek → Šotek across all source files

## [0.3.1] - 2026-03-26

### Retrieval Quality (Wave 2)

- **Sentence-level indexing**: paragraphs over 200 chars are split into
  sentences with 1-sentence overlap context windows. Finer granularity means
  more precise BM25 matching — a sentence about "pottery" no longer competes
  with a 500-word paragraph where "pottery" appears once.

- **Query decomposition for multi-hop**: complex queries like "What hobbies
  does the person who works at Google have?" are automatically decomposed into
  sub-queries ["who works at Google?", "what hobbies?"]. Results are combined
  and re-ranked by relevance to the original query.

- **Enhanced prospective queries (12 categories)**: expanded from 5 pattern
  types to 12: named entities, functions, activities/preferences, occupation,
  relationships, allergies/health, travel/location, learning/skills, favourites,
  decisions/findings/metrics, versions/dates, file/code. Cap raised to 20
  queries per paragraph.

- **Confidence scoring**: every SearchResult now includes a `confidence` field
  (0.0-1.0) computed from score normalisation, answer extraction success, and
  absolute score thresholds.

- **Cross-reference answer verification**: when multiple results extract the
  same answer, confidence is boosted proportionally. Disagreements are left
  at base confidence.

### Testing

- 25 new tests covering all wave 2 features (668 total, up from 643).

## [0.3.0] - 2026-03-26

### Retrieval Engine

- **Real term frequency in BM25**: scoring now uses actual token counts per
  paragraph instead of binary presence. Paragraphs that mention a query term
  multiple times rank higher, matching the standard BM25 formula:
  `IDF * tf * (k1+1) / (tf + k1*(1-b+b*dl/avg_dl))`.

- **DF-tracked incremental IDF**: `add_file()` now maintains a global document
  frequency counter (`_df`). New tokens get accurate IDF values computed from
  real corpus statistics instead of the previous `log(1 + N/2)` approximation
  that drifted over time.

- **Normalised score fusion**: when the bi-encoder model is loaded, BM25 and
  embedding scores are normalised to [0,1] before fusion. Previously raw BM25
  scores (unbounded) were mixed with cosine similarities ([-1,1]), producing
  incommensurable fused scores.

- **Reciprocal Rank Fusion (RRF)**: replaces the naive weighted sum for
  BM25+embedding fusion. RRF score = sum(1/(k+rank)) across ranked lists.
  Scale-invariant, well-proven in hybrid retrieval (Cormack et al. 2009, k=60).

### Temporal Reasoning

- **Relative date resolution**: `parse_dates()` now resolves "yesterday",
  "today", "last week", "last month", "last year", "this week", "this month",
  "this year" against a reference date. Previously these expressions were
  detected by regex but never converted to actual calendar dates.

- **O(N) edge building**: `add_events()` replaced the O(N^2) pairwise
  comparison with date-bucketed approach: same_day edges within buckets,
  before/after edges only between adjacent date buckets with capped edges
  per pair. At 10,000 events this reduces edge count from ~5,000 to <500.

- **Query-relevant date extraction**: `_extract_date_answer()` now scores
  all date candidates by proximity to query terms and returns the highest-scored
  match, instead of blindly returning the first date found in the text.

### Entity Graph

- **Typed relations persisted**: `consolidation_engine._update_graph()` now
  extracts typed relations (caused_by, fixed_by, replaced, contradicts,
  version_of, depends_on, improved, produced, used_in, tested_with) from
  trace text and persists them to `relations.jsonl`. Previously only co_occurs
  edges were written despite `entity_extractor.extract_relations()` detecting
  11 typed relation types.

- **Typed relation boost in retrieval**: `_entity_boost_score()` gives 1.5x
  the boost weight to typed relations vs co_occurs, since they carry stronger
  semantic signal for query answering.

### Concurrency

- **Async consolidation**: `handle_remember()` runs consolidation in a
  background thread with 10-second debounce, instead of blocking every write
  with a synchronous `consolidate()` call.

- **Thread-safe singletons**: `_UNIFIED_INDEX`, `_KNOWLEDGE_STORE`, and index
  writes are protected by `threading.Lock`. Safe for concurrent access from
  the FastAPI server.

### Answer Extraction

- **Query-proximity scoring**: all answer extractors (_extract_date_answer,
  _extract_number_answer, _extract_name_answer) now collect ALL candidates
  and score by proximity to query terms in the source text, returning the
  highest-scored match instead of the first regex hit.

- **Improved yes/no detection**: expanded negation markers (15 patterns
  including "couldn't", "unable", "failed to", etc.) with majority-vote
  scoring across query terms.

### Indexing

- **Raised chunking limits**: `MAX_CODE_CHUNK_CHARS` raised from 500 to 1000,
  `MAX_CODE_CHUNKS` raised from 50 to 200. Previously files over ~25K chars
  had content silently discarded. Now indexes up to 200K chars per file.

### Testing

- 45 new tests covering all improvements (643 total, up from 598).
- Dedicated `test_improvements.py` with tests for: _token_counts, RRF,
  relative date resolution, typed relations, async consolidation, thread
  safety, query-proximity scoring, chunking limits.
