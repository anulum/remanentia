# Model Card — Fact-Validity Model (C5)

> **Status: OVERFITTED TO SYNTHETIC.** The classifier reaches 100 %
> on held-out synthetic templates, which is a sign of template-memorisation,
> not generalisation. Real conversational performance is unmeasured.
> Production gates this model behind a regex-first pipeline so it
> only runs on the catch-all "event" bucket.

## What this model is for

Label an atomic fact with a lifecycle type so the consolidation engine
knows how long it should remain "active":

| Label    | Example                               | Default validity |
|----------|---------------------------------------|------------------|
| plan     | "I plan to visit Berlin next month."  | Until superseded |
| event    | "I went to Berlin on March 15."       | Permanent        |
| opinion  | "I think Python is a clean language." | Long-lived       |
| state    | "I live in Zürich."                   | Until overridden |
| question | "Is Python still maintained?"         | Ephemeral        |

## Architecture

- Base: `distilbert-base-uncased`
- Head: 5-way softmax
- Training script: `training/train_fact_validity.py`

## Training data

- **Source:** synthetic templates covering each label family
  (e.g. "I plan to…" → plan; "I went to…" → event)
- **Size:** ~5 000 templated sentences
- **Caveat:** synthetic templates are **cleanly separable** — each
  label has a unique opening phrase. Real conversations do not.

## Evaluation

| Metric | Synthetic held-out split | Real conversational data |
|--------|--------------------------:|--------------------------:|
| Accuracy | 100 % | **unmeasured** |
| F1 macro | 100 % | **unmeasured** |

100 % on synthetic data is a warning sign: the model learned template
shapes, not semantic fact-type inference. A model that hits 100 % on
its own generation process is usually not learning anything
transferable.

## Production gating

The consolidation engine consults a **regex-first** fact-typer:

- Strong lexical signals (first-person explicit markers) are handled
  by regex.
- **Only the catch-all "event" fallback** is routed to C5 to refine
  (event → plan/state/opinion/question when the regex was ambiguous).

This keeps the weak ML signal out of the high-confidence path while
leaving a recovery route for sentences the regex cannot classify.

## What we would need before trusting this

- Human-labelled validation set over 500–1 000 real LongMemEval /
  LOCOMO sentences
- Confusion matrix on that set
- Per-label precision and recall, not just accuracy
- A committed A/B on consolidation quality (retention of stale vs
  valid memories) when C5 is enabled for all facts, not just
  "event" fallback

None of those exist today.

## Reproduction

```bash
cd training
CUDA_VISIBLE_DEVICES=4 python train_fact_validity.py
```
