# Model Card — Temporal Relation Classifier (C3)

> **Status: NON-FUNCTIONAL.** The classifier's macro-F1 on the held-out
> split is 0.178. The random baseline for six balanced classes is
> 0.167. The model did not learn the task and is **not wired into any
> default code path**. This card exists so downstream users know
> the weights in `training/` should not be trusted.

## What this model is for

Given a pair of temporal expressions ("yesterday", "last week",
"March 15, 2026") extracted from the same conversation, predict the
Allen-algebra relation between them: BEFORE / AFTER / EQUAL /
OVERLAPS / CONTAINS / MEETS.

Intended downstream use was to feed ArcaneRetriever's SEMANTIC
channel with relation edges so "what happened first" questions could
be answered without LLM reasoning over every pair.

## Architecture

- Base: `distilbert-base-uncased`
- Two expression spans concatenated with `[SEP]`
- Pooled `[CLS]` → 768-d → 6-way softmax classifier head
- Cross-entropy loss, AdamW, linear warm-up

Script: `training/train_relation.py`.

## Training data

- **Size:** 15 000 synthetic pairs
- **Generation:** `training/generate_data.py` with uniform class
  distribution (2 500 examples per class)
- **Vocabulary:** templated expressions over date ranges in
  2022–2026
- **Splits:** 80 / 10 / 10 train / val / test

## Evaluation

| Metric    | Value |
|-----------|------:|
| F1 macro  | 0.178 |
| Accuracy  | 0.181 |
| Random baseline (6 classes, balanced) | 0.167 |

The model is performing **at chance**. It did not separate the classes.

## Why it failed

Self-reported in `training/HONEST_ASSESSMENT.md`:

1. **Too little data.** 15 k uniform synthetic pairs did not supply
   the distributional signal a general relation classifier needs.
2. **Template over-simplification.** Uniform distribution over six
   classes with a fixed template vocabulary trained the model to
   memorise template shapes rather than temporal reasoning.
3. **Domain gap.** LongMemEval conversations use vague and
   context-dependent temporal phrases; our training templates were
   lexically cleaner than any real conversation.

## What to use instead

- **Rule-based date arithmetic** in `temporal_graph.py::temporal_code_execute`
  handles comparison when both expressions resolve to ISO dates via
  the rule-based normaliser (C4 rules).
- **LLM reasoning** on the cleaned context (90 % date coverage after
  C4) recovered most of the task — this is what currently ships.

## Disclosure

Committing a failed model is a deliberate choice: pretending it
works would be worse than admitting it does not. If you find the
weights in `training/` and are tempted to wire them in, please read
this card first.

## Reproduction

```bash
cd training
CUDA_VISIBLE_DEVICES=2 python train_relation.py
python eval_local.py  # retrieval-side — relation model is not wired
```

We do not publish training checkpoints for C3 because they would
be misleading.
