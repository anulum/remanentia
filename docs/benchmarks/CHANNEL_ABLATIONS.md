# ArcaneRetriever — channel ablations (answer-session recall)

- **Sample**: first 500 LongMemEval oracle items (top_k=10, seed=42).
- **Metric**: fraction of ``answer_session_ids`` appearing among the sessions covered by the retriever's top-K facts.
- **Config**: each ablation removes exactly one channel; ``ALL`` is the baseline with every channel active.
- **Runtime**: 21.2 s.

## Mean recall per qtype

| qtype | ALL | no_bm25 | no_entity | no_temporal | no_session |
|---|---|---|---|---|---|
| knowledge-update | 0.987 | 0.987 (+0.000) | 0.994 (+0.006) | 0.987 (+0.000) | 0.987 (+0.000) |
| multi-session | 0.981 | 0.980 (-0.001) | 0.978 (-0.003) | 0.976 (-0.004) | 0.964 (-0.017) |
| single-session-assistant | 1.000 | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) |
| single-session-preference | 1.000 | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) |
| single-session-user | 1.000 | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) | 1.000 (+0.000) |
| temporal-reasoning | 0.956 | 0.961 (+0.005) | 0.965 (+0.009) | 0.982 (+0.026) | 0.960 (+0.004) |
| **overall** | 0.981 | 0.982 (+0.001) | 0.984 (+0.003) | 0.987 (+0.006) | 0.978 (-0.003) |

## Reading the table

Each ablation cell shows ``mean (delta_vs_ALL)``. A large negative delta means the missing channel was pulling its weight for that qtype; a near-zero delta means the other three channels compensate and the removed channel is redundant on that sample.

This is a **retrieval-side** measurement. A channel that does not move recall might still help the downstream LLM by contributing different facts to the same RRF-fused list; a separate end-to-end ablation under ``bench_longmemeval.py --arcane`` would answer that.

## Findings worth flagging

- **`temporal` hurts `temporal-reasoning`**: recall rises 0.956 → 0.982 (+0.026) when the channel is removed. Expected direction is the opposite; investigate the channel's ranking logic.
- **`session` helps `multi-session`**: removing it drops recall 0.981 → 0.964 (-0.017).
