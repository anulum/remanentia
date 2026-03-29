# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2: Fine-tune ms-marco cross-encoder on temporal relevance pairs.

Usage:
    CUDA_VISIBLE_DEVICES=1 python training/train_cross_encoder.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_DATA = Path(__file__).resolve().parent / "datasets"
_OUT = _BASE / "models" / "temporal-ce-v1"

# Hyperparameters
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 384
EVAL_SPLIT = 0.1


def load_pairs() -> list[dict]:
    """Load cross-encoder training pairs from JSONL dataset."""
    path = _DATA / "crossencoder_pairs.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Train the temporal cross-encoder (C2) on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("Loading pairs...")
    raw = load_pairs()
    log.info("  Total: %d pairs (pos=%d, neg=%d)",
             len(raw),
             sum(1 for r in raw if r["label"] == 1),
             sum(1 for r in raw if r["label"] == 0))

    # Split
    split_idx = max(1, int(len(raw) * (1 - EVAL_SPLIT)))
    train_raw = raw[:split_idx]
    eval_raw = raw[split_idx:]

    train_examples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in train_raw
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    log.info("Loading base model: %s", MODEL_NAME)
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_SEQ_LEN, device=device)

    # Evaluator
    eval_sentences1 = [r["query"] for r in eval_raw]
    eval_sentences2 = [r["passage"] for r in eval_raw]
    eval_labels = [int(r["label"]) for r in eval_raw]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=list(zip(eval_sentences1, eval_sentences2)),
        labels=eval_labels,
        name="temporal-ce-eval",
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    log.info("Training: %d examples, %d steps, %d warmup", len(train_examples), total_steps, warmup_steps)

    model.fit(
        train_dataloader=train_loader,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_loader),
        output_path=str(_OUT),
        optimizer_params={"lr": LR},
        weight_decay=0.01,
        show_progress_bar=True,
        use_amp=False,
    )

    log.info("Model saved to %s", _OUT)

    # Verify
    loaded = CrossEncoder(str(_OUT), device=device)
    scores = loaded.predict([
        ("When did I buy the car?", "I bought a new car last Tuesday."),
        ("When did I buy the car?", "The weather was nice today."),
    ])
    log.info("Verification scores: relevant=%.3f, irrelevant=%.3f", scores[0], scores[1])
    log.info("C2 training complete.")


if __name__ == "__main__":
    main()
