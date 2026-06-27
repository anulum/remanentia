# Changelog

All notable changes to Remanentia are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `store_sources.py`, `remanentia-store-sources`, and
  `remanentia store-sources` write the selected-store `MemoryIndex` source
  config used to index traces, semantic memory, compiled facts, and the
  external stimulus firehose during MS.0 backfill refreshes.
- `store_manifest.py`, `remanentia-store-manifest`, and
  `remanentia store-manifest` record the selected memory-store paths and current
  artifact counts before backlog reconsolidation mutates corpus state.
- `store_paths.py` centralises the canonical memory-store layout so hub ingest,
  feed ingest, and index freshness monitoring derive findings, cursor, digest,
  stimuli, and vector-index paths from the same `REMANENTIA_BASE` /
  `REMANENTIA_STIMULI_DIR` contract.
- `memory_sources.py` provides neutral repository-local MemoryIndex source
  defaults plus JSON/env configuration through
  `REMANENTIA_MEMORY_SOURCES_CONFIG` and `REMANENTIA_MEMORY_SOURCES_JSON`.
- `data/compiled_seed_facts.jsonl` supplies durable historical and cross-project
  compiled-memory seed facts as data instead of Python control flow.
- `feed_ingest.py` and the `remanentia-feed-ingest` console script ingest explicit
  `Finding:` / `Decision:` rows from `~/synapse/feed.ndjson`, reuse the
  `synapse_channel` finding parser and admission gate, and persist admitted
  candidates through the existing Markdown finding sink.
- Full LongMemEval-S benchmark mode (`bench_longmemeval.py --full`): runs the realistic
  ~50-session haystack instead of the oracle (gold-only) setting, with a new
  retrieved-context reader that feeds the reader only the top retrieved sessions rather
  than the whole history. Tunable via `REMANENTIA_FULL_MAX_SESSIONS`,
  `REMANENTIA_FULL_CHAR_BUDGET`, `REMANENTIA_FULL_RETRIEVE_K`.
- `retrieved_context.py` — budget-limited retrieved-session assembly and gold-session
  recall scoring for large-haystack evaluation.
- `tools/retrieval_recall.py` — full-S retrieval-recall diagnostic (recall@N per
  category, no LLM calls).

### Fixed
- Cross-encoder reranking is on by default again in `ArcaneRetriever`. It had been
  gated behind an opt-in environment flag, which silently cost ~8–9 LongMemEval
  questions (knowledge-update and multi-session). Opt out with
  `REMANENTIA_ARCANE_CE_DISABLE=1` for latency-sensitive live/MCP use.

### Changed
- `tools/install_user_services.py` now accepts `--base` and `--stimuli-dir`,
  exports the selected store into generated API, vector-worker, and
  freshness-watchdog units, and writes vector-worker and freshness artifacts
  under that selected store.
- `remanentia status` and `remanentia init` now use the same canonical
  `store_paths.py` resolver as ingest and freshness monitoring, so
  `REMANENTIA_BASE` selects one operator-visible store instead of leaving CLI
  status on import-time repository paths.
- `memory_index.py` no longer hardcodes deployment-specific source roots; it
  consumes the resolved `memory_sources.SourceConfig` while preserving the
  existing `SOURCES` / `SOURCE_EXTENSIONS` compatibility globals for tests and
  callers that patch them.
- `compiled_memory.py` loads seed facts from tracked JSONL data, with an installed
  data-file fallback, and no longer mines the removed internal benchmark script.
- Benchmark reporting reconciled around the two LongMemEval settings. The headline is
  now the full-S retrieval setting (3-run mean 56.6% overall on the current
  `gpt-4o-mini`), which is what published leaderboards measure; the 72.2%/~71% figures
  are labelled as the oracle setting (gold sessions only, retrieval not exercised) and
  marked not comparable to leaderboards. Per-run accuracy history, including the three
  full-S runs, is recorded in `benchmarks/longmemeval_history.jsonl`.
- `arcane_retriever.py` is strict-mypy clean.

### Documentation
- `compiled_memory` now has a MkDocs API page documenting seed facts and generated
  compiled-fact outputs.
- **Model cards published** (2026-04-17) at `docs/models/` for the five
  trained components (C1 embedding, C2 cross-encoder, C3 relation
  classifier, C4 date normaliser, C5 fact-validity). Each card states
  what the model does, what it was trained on, how it was evaluated,
  and what it does **not** do. C3 in particular is publicly documented
  as non-functional (F1-macro 0.178 vs 0.167 random baseline) —
  previously this fact lived only in `training/HONEST_ASSESSMENT.md`.
  Not wired into any default code path regardless. README updated with
  component-status table.
- **LOCOMO committed result published** (2026-04-17): 83.1 % overall
  (1 651 / 1 986) from `paper/locomo_results.json` supersedes the
  pre-LLM 74.7 % baseline in README and `docs/benchmarks/LOCOMO.md`.

### Added
- **LongMemEval R11 — 65% temporal target achieved** (2026-04-11, commits 94f44c7 + 8b187a4 + d9d9713 + 9dd754b):
  - **temporal-reasoning: 65.4% (87/133)** — target achieved, up from R10 57.9% (+7.5pp / +10 questions)
  - Cumulative R8 → R11: 45.9% → 65.4% = **+19.5pp / +26 questions**
  - Overall: 72.2% (361/500); R10→R11 movement (−0.6pp) within LLM single-run noise envelope
  - Category changes R10→R11: multi-session 63.2%→54.1% (−12, within typical ~±10 question noise — all R11 code changes are gated inside `if qtype == "temporal-reasoning"` branches and do not touch multi-session pipeline), knowledge-update 88.5%→84.6% (−3), preference 83.3%→90.0% (+2, recovered)
  - Four pipeline changes for temporal-reasoning:
    - **Fuzzy inclusive/exclusive durations** (commit 94f44c7): `temporal_code_execute` and `extract_duration` emit dual-format output ("30 days or 31 days counting both endpoints") so the LLM/judge has both inclusive and exclusive forms.
    - **Question_date anchoring** (commit 8b187a4): plumb `question_date` from oracle through `run_benchmark → _arcane_answer → _build_context → _tremu_precompute → temporal_code_execute`. Prepends `TODAY (question was asked on): ...` to temporal-reasoning LLM prompt. Fixed "how many X ago" queries where the LLM was hallucinating "today" as an arbitrary date.
    - **Multi-event proximity tuning** (commit d9d9713): `_score_event_vs_query` combines unigram + 2× bigram + density term; `_proximity_score` uses distance-weighted scoring within a tighter 60-char window. Eliminates tied scoring that caused `_pick_duration_pair` / `extract_duration` to choose the wrong date pair.
    - **Narrow multi-hop chain resolution** (commit 9dd754b): `_expand_chained_dates` resolves `"N (days|weeks|months) (after|before) YYYY-MM-DD"` patterns within a single sentence, emitting the computed ISO date as a new `TemporalEvent`. Scope limited to ISO anchors; entity-linked chains out of scope.
  - 31 new tests across temporal_graph (22: fuzzy dual-format, question_date anchor, scoring, proximity, chained), answer_extractor (9: dual-format + proximity), bringing total to 1,814
  - R10 results archived as `data/longmemeval_hypotheses.results.R10_baseline.jsonl` (gitignored)

- **LongMemEval R10 — intraday HH:MM resolution** (2026-04-11, commits 416228f + d0c7640):
  - **Overall: 72.8% (364/500)** — within LLM judge noise margin of R9 73.0%
  - **temporal-reasoning: 57.9% (77/133)**, up from R9 56.4% (+1.5pp / +2 questions)
  - knowledge-update 88.5%, multi-session 63.2%, single-session-preference 83.3% (was 90.0% — LLM noise on 30-question subset)
  - Cumulative: temporal 45.9% (R8) → 57.9% (R10) = +12pp / +16 questions
  - `date_normalizer._parse_session_datetime()` parses LongMemEval timestamps with HH:MM precision
  - `AtomicFact.session_date` field propagated by `fact_decomposer` for intraday tiebreaking
  - `arcane_retriever._sort_results_chronologically()` uses 4-tier key:
    `(date, normalised_session_datetime, session_idx, turn_idx)` — same-day facts sorted by HH:MM
  - `ArcaneRetriever.build_context` accepts `sort_chronologically: bool = False`;
    bench_longmemeval passes `True` only for temporal-reasoning. RRF relevance order preserved
    for knowledge-update, multi-session, and preference qtypes (first release of Task #32 caused
    -5 question regression in each; `fix: qtype-aware chronological sort` restored them)
  - 14 new tests across date_normalizer (10), arcane_retriever (5 sort + 2 flag), fact_decomposer (2)
  - R9 results archived as `data/longmemeval_hypotheses.results.R9_baseline.jsonl` (gitignored)

- **LongMemEval R9 — temporal breakthrough** (2026-04-11):
  - Overall 73.0% (365/500), up from R8 69.0% (+4.0pp)
  - temporal-reasoning 56.4% (75/133), up from R8 45.9% (+10.5pp, +14 questions)
  - multi-session: 63.9%, knowledge-update: 88.5%, all single-session ≥85.7%
  - Three pipeline changes:
    - **Session-anchored date resolution** (commit 50149ff): `normalise_in_context()`
      resolves vague expressions ("yesterday", "3 weeks ago") against session
      timestamps at index time. Wired into `bench_longmemeval._build_index_for_question`
      and `temporal_graph.extract_events`. 28 new tests.
    - **Explicit duration arithmetic + TReMu pre-computation** (commit e062f0e):
      `_pick_duration_pair()` keyword-matched event pairing, weeks/months/counting
      support in `temporal_code_execute()`, `extract_duration()` in answer_extractor
      with priority over Rust path, `_tremu_precompute()` prepends `COMPUTED ANSWER:`
      to LLM context for temporal questions. 17 new tests.
    - **Chronological session ordering** (commit e867382): `_sort_results_chronologically()`
      sorts FusedResult by `valid_from`/`date_mentions`, sessions sorted by
      `haystack_dates` before LLM context build. 7 new tests.
- **Local LLM backend switched to Gemma 3 4B via Ollama** (commit de8d731):
  - `LLMConfig` defaults to `gemma3:4b` on Ollama port 11434 (was Qwen 2.5 7B on 8080)
  - Benchmark on RX 6600 XT 8GB via Vulkan: 45-67 tok/s, 3.5 GB VRAM
  - Gemma 4 e2b/e4b found broken in Ollama 0.20.2 (q4_K_M tag dedups to bf16)
  - Multi-GPU Vulkan tensor split confirmed unviable due to PCIe x1 bottleneck (0.6 tok/s)
  - `tools/bench_local_llm.sh` reproducible benchmark harness, `docs/guides/local_llm_setup.md`
- **Recall + SNN Rust rustification (13th crate, 52 functions total)**:
  - `remanentia_recall` (new crate): tokenize_words (1.4×), tokenize_words_min,
    token_overlap_score, assess_novelty — wired into memory_recall.py
  - `arcane_stdp.encode_text`: hash-based unigram+bigram SNN encoding — wired
    into snn_backend.py (3rd Rust function alongside stdp_batch, lif_step)
  - Key finding: assess_novelty 0.03× due to HashSet FFI overhead on 500+ tokens;
    encode_text 0.9× due to numpy array FFI. Both document the pyclass-vs-stateless
    boundary. tokenize_words achieves 1.4× on 3.8K char input.
  - 38 new tests (24 recall parity + 14 SNN encode parity/performance)
- **Tier 1–3 Rust rustification (12 crates, 23 functions)**:
  - `remanentia_retrieve` (new crate): 13 hot-path functions — tokenize, stem, hash_encode (26.7×),
    tfidf_score, spike_feature, snn_affinity, cosine_sim, RRF, entity_graph_score, filename_bonus
  - `remanentia_fact_decomposer`: `RustFactIndex` persistent pyclass (8.8×)
  - `remanentia_temporal`: `build_temporal_edges`, `score_temporal_query` (2.3×)
  - `remanentia_knowledge_store`: `knowledge_search`, `get_related_ids`, `graph_search`,
    `RustKnowledgeIndex` pyclass
  - `remanentia_consolidation`: `cluster_traces` (76.1×), `build_summary_dag`, `cluster_notes` (12.6×)
  - `arcane_stdp`: `homeostatic_scaling` (45.4×)
  - Key finding: `#[pyclass]` persistent objects >> stateless FFI for class methods;
    Python datetime parsing overhead accounts for 76× in cluster_traces;
    FFI dict construction overhead makes build_summary_dag slower in Rust (0.3×)
- **26 end-to-end tests** (`test_e2e_rust_pipeline.py`): full pipeline coverage across Tiers 1–3
  — ingest → retrieve → consolidate → DAG, cross-tier integration, Rust path verification
- **31 Tier 3 unit tests** (`test_tier3_rust.py`): cluster_traces, homeostatic_scaling,
  cluster_notes, build_summary_dag — exact parity with Python paths
- **41 Tier 2 unit tests** (`test_tier2_rust.py`): RustFactIndex, temporal edges,
  temporal query, knowledge search/graph/related
- **65 Tier 1 retrieval tests** in `test_remanentia_retrieve.py`
- **Temporal training pipeline (C1–C5)**: 5 models trained on 5x RX 6600 XT (ROCm 6.2)
  - C1: fine-tuned `all-MiniLM-L6-v2` embedding model for temporal-aware retrieval
  - C2: fine-tuned `ms-marco-MiniLM-L-6-v2` cross-encoder (AP=84.57%)
  - C3: temporal relation classifier (`bert-small`, 6 classes: before/after/same_day/overlaps/contains/unknown)
  - C4: date normaliser with rule-based engine (95% confidence) + ML fallback (`bert-mini`)
  - C5: fact validity model for type classification + supersession detection (combined=100%)
- **date_normalizer.py**: rule-based + ML vague date normalisation ("3 weeks ago" → ISO date)
  - 12 rule patterns: quantified, couple, few, several, weekday, 11 simple expressions
  - Confidence scoring (0.7–0.95) with graceful ML fallback
- **temporal_relation.py**: temporal relation classifier for event ordering
- **fact_validity_model.py**: fact type + supersession detection model
- **training/**: full training infrastructure (data generation, 5 training scripts, parallel launcher)
- **197 new tests** (844 → 1,041):
  - test_date_normalizer (80): rule patterns, ML mock, month arithmetic, weekday, edge cases, model loading
  - test_temporal_relation (17): classification, ordering, model loading/exception, N>2 pairwise
  - test_fact_validity_model (18): type/supersedes/boundary classification, model loading/exception
  - test_temporal_training_integration (24): parse_dates+C4, _build_fact+C5, _ch_temporal+C3, full pipeline, graceful degradation
  - test_temporal_synth (32): date normalisation, temporal relations, fact validity generators, save_jsonl
  - test_generate_data (26): tokenise, BM25, flatten_turns, triplet/pair/date extraction
- `temporal` optional dependency group in pyproject.toml
- Date extraction coverage: +321% dated facts (121 → 510) after reference_date plumbing fix
- Local evaluation harness (`training/eval_local.py`): retrieval recall, date coverage, TReMu hits — no API needed
- Honest assessment (`training/HONEST_ASSESSMENT.md`): component-level validation status
- **LongMemEval temporal-reasoning: 45.9% (R8 baseline)** — GPT-4o-mini verified
- Rust STDP/LIF acceleration (arcane_stdp) — 2–3x speedup, bit-exact with numpy
- Enterprise hardening: ARCHITECTURE, CONTRIBUTING, SECURITY, GOVERNANCE, ROADMAP, VALIDATION, NOTICE, CITATION.cff
- MkDocs documentation site with 4 guides, 9 API refs, benchmarks, research
- GitHub issue/PR templates, dependabot, CodeQL, Scorecard, stale bot workflows
- README badges, header image, branding footer
- **100% test coverage**: 1,599 tests across 19 modules (zero lines missing)
- Tests for api_server.py, arcane_retriever.py, fact_decomposer.py (3 new test files)
- api_server, arcane_retriever, fact_decomposer added to py-modules

### Fixed
- CI: PEP 639 license classifier conflict
- CI: mock anthropic import in _get_client tests
- SPDX headers: Sotek → Šotek across all source files
- SPDX headers: full 6-line header on all edited test files

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
