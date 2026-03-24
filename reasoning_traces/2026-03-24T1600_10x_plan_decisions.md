# Decision Trace: Remanentia 10x Architecture
# Date: 2026-03-24T16:00Z

## Research Basis
- 3 parallel research agents + 6 web searches + Remanentia's own memory
- 15+ competitor systems analyzed (Kumiho, EverMemOS, MemMachine, Memori, Mem0, Letta, Zep, LangMem, TiMem, SuperLocalMemory, MemOS, MAGMA, A-MEM, Observational Memory, Hindsight)
- Prior research: 120+ papers from March 20 investigation marathon

## Key Findings That Drove Decisions

1. Kumiho (93.3% LOCOMO) achieves 98.5% recall via LLM-powered prospective indexing. We already have template-based version. Upgrade to LLM = biggest single improvement.

2. Every system above 81% uses LLM for answer generation. Our retrieval is competitive (92.9% P@1). The gap is answer synthesis, not finding documents.

3. Temporal reasoning (47.9%) is our weakest category by far. Graph-based systems (Mem0g 58.1%, Zep bitemporal) handle this with temporal edges.

4. We are #1 zero-LLM system (81.2% vs SuperLocalMemory 74.8%). This is a genuine market position.

## Decisions Made

- Phase 1: LLM prospective indexing (upgrade existing _generate_prospective_queries)
- Phase 2: Multi-paragraph answer synthesis (upgrade existing llm_extract_answer)
- Phase 3: Temporal knowledge graph (new module)
- LLM remains opt-in. Zero-LLM mode preserved as first-class.
- Target: 90%+ LOCOMO overall, 75%+ temporal
