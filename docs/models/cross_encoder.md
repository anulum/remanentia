# Model Card — Cross-Encoder Fine-Tune (C2)

> **Status: UNCERTAIN.** AP on our own held-out split is 84.57 %, but
> the split is self-referential (drawn from the same synthetic
> process as the training set). Real-world reranking impact on
> LongMemEval or LOCOMO has not been measured. By default the
> pipeline loads the **base** `cross-encoder/ms-marco-MiniLM-L-6-v2`
> unless a local `models/temporal-ce-v1/` directory is present.

## What this model is for

Re-rank the top-k candidates from BM25/embedding retrieval by
pairwise scoring (query, document). Used by:

- `memory_index.MemoryIndex._load_models` (cross_encoder attribute)
- `arcane_retriever.ArcaneRetriever._load_ce_model` (class attribute)
- `bench_locomo.py::_get_cross_encoder`

## Architecture

- Base: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, MS-MARCO
  passage-ranking pre-training)
- Fine-tune: binary relevance classification on
  (query, retrieved_paragraph) pairs
- Cross-entropy loss, AdamW, linear warm-up

Script: `training/train_cross_encoder.py`.

## Training data

- (Query, paragraph) pairs from LongMemEval session decomposition
  with labels derived from the oracle's `answer_session_ids`
- **Caveat:** the split is drawn from the same source as the training
  examples — a held-out *paragraph* from a *seen* session is easier
  than a fresh eval.

## Evaluation

| Metric  | Value   | Notes |
|---------|---------|-------|
| AP      | 84.57 % | Own eval split (self-referential) |
| Recall@10 on LongMemEval vs BM25-only | **unmeasured** | Committed ablation pending — `P2-13` |
| Recall@10 on LOCOMO vs BM25-only | **unmeasured** | Same note |

## What we would need before trusting this

- Committed LongMemEval A/B: BM25-only vs BM25 + fine-tuned CE vs
  BM25 + base CE, reported per qtype
- Same for LOCOMO
- A deliberately out-of-distribution eval set (queries on unseen
  sessions) to detect overfitting

The upcoming `P2-13 Cross-encoder real-world eval` task will produce
those numbers; this card will be updated in place when that lands.

## What currently ships

Production uses the base model unless `models/temporal-ce-v1/` is
present. Given the uncertainty above, deployers are encouraged to
stay on the base model until the A/B is committed.

## Reproduction

```bash
cd training
CUDA_VISIBLE_DEVICES=1 python train_cross_encoder.py
```
