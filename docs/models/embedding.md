# Model Card — Embedding Fine-Tune (C1)

> **Status: UNCERTAIN.** The fine-tune ran to completion on 3 464
> triplets, but no A/B evaluation was captured. We cannot claim the
> fine-tuned variant is better than the base model on retrieval. By
> default the pipeline loads the **base** `all-MiniLM-L6-v2` unless a
> `models/temporal-embed-v1/` directory is present.

## What this model is for

Encode memory fragments and queries into 384-d vectors for dense
retrieval. Used by `memory_index.MemoryIndex._compute_embeddings`
and `build_memory_standalone.MemoryIndex._compute_embeddings`.

## Architecture

- Base: `sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-d)
- Pooling: mean-of-token-embeddings (provider default)
- Fine-tune objective: triplet margin loss on (anchor, positive,
  negative) over temporally-related memory pairs

Script: `training/train_embedding.py`.

## Training data

- **Size:** 3 464 triplets
- **Generation:** temporal augmentation of LongMemEval sessions
- **Caveat:** 3 464 triplets is **very small** for fine-tuning a
  22 M-parameter encoder. Catastrophic forgetting on non-temporal
  queries is a plausible risk.

## Evaluation

**None of substance.**

- `sentence-transformers.fit()` ran and produced weights.
- **No retrieval A/B eval was performed** comparing base vs fine-tune
  on any committed benchmark.
- Cosine similarity on training positives/negatives is self-referential
  and is not reported.

## What we would need before trusting this

- `recall@5` / `recall@10` on LongMemEval and LOCOMO, base vs fine-tune
- Specific regression test on non-temporal categories (preference,
  knowledge-update) to rule out catastrophic forgetting

Until those exist, treat this model as experimental.

## What currently ships

The production retrieval path loads base `all-MiniLM-L6-v2` unless
the `models/temporal-embed-v1/` directory is present at startup.
If you want to A/B test, place the fine-tune there and compare.

## Reproduction

```bash
cd training
CUDA_VISIBLE_DEVICES=0 python train_embedding.py
# Evaluation A/B is your job.
```
