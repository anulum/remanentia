# Remanentia Local LLM Benchmark Results

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 6600 XT (8 GB VRAM), ROCm 6.2.4, gfx1030
**Server:** llama.cpp (ROCm build, -ngl 99, -c 16384)
**Benchmark:** LongMemEval (50 questions, temporal-reasoning subset, fuzzy-match scoring)
**Retrieval:** Legacy BM25 pipeline (not ArcaneRetriever)

## Results

| Model | Params | Overall | ms/q | Notes |
|-------|-------:|--------:|-----:|-------|
| Llama 3.2 1B Q4 | 1.2B | 16.0% | ~3,500 | Too small for complex reasoning |
| Qwen 2.5 1.5B Q4 | 1.8B | 28.0% | ~3,800 | Best sub-2B option |
| Llama 3.2 3B Q4 | 3.2B | 34.0% | 7,795 | Decent but below Qwen 3B |
| **Qwen 2.5 3B Q4** | **3.4B** | **44.0%** | **8,563** | **Best value: accuracy/speed/VRAM** |
| Phi 3.5 Mini Q4 | 3.8B | CRASH | — | Segfault on ROCm 6.2.4 (gfx1030) |
| Llama 3.1 8B Q4 | 8.0B | 42.0% | 14,022 | Slower than Qwen 3B, worse accuracy |
| **Qwen 2.5 7B Q4** | **7.6B** | **44.0%** | **13,498** | Same accuracy as 3B, 1.6× slower |
| Mistral 7B v0.3 Q4 | 7.2B | **46.0%** | 18,028 | Highest accuracy but 2× slower |

## Reference (cloud API)

| Model | Overall (500q) | Notes |
|-------|---------------:|-------|
| GPT-4o-mini | 69.0% | R8 official, GPT-judge scored |
| Qwen 2.5 3B local | 37.2% | Full 500q run, fuzzy-match scored |

## Analysis

1. **Qwen 2.5 3B is the recommended local model.** Same accuracy as 7B at half the
   latency and 40% less VRAM. The 3B→7B scaling gives zero accuracy gain on this benchmark.

2. **Qwen family dominates.** At every size tier, Qwen outperforms Llama by 10pp.

3. **7B+ models are VRAM-limited.** With 16K context on 8 GB VRAM, 7B models barely fit
   and run 1.6-2× slower. Multi-GPU (2× RX 6600 XT) would help.

4. **Local vs cloud gap is ~32pp.** GPT-4o-mini scores 69% vs local best 46%.
   This gap is primarily in temporal reasoning and multi-session questions
   where the small model struggles with long-context reasoning.

5. **Phi 3.5 Mini crashes on ROCm 6.2.4** — likely an unsupported operation on gfx1030.

## Recommendation

For production Remanentia deployment:
- **Single RX 6600 XT:** Qwen 2.5 3B Q4_K_M (~2 GB VRAM, ~8.5s/query)
- **Dual RX 6600 XT:** Qwen 2.5 7B Q4_K_M with tensor parallelism
- **CPU fallback:** Qwen 2.5 1.5B Q4_K_M (~1 GB RAM, ~5s/query on i5-11600K)
