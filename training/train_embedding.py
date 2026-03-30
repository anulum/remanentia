# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C1: Fine-tune all-MiniLM-L6-v2 on temporal-aware embedding triplets.

Usage:
    CUDA_VISIBLE_DEVICES=0 python training/train_embedding.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_DATA = Path(__file__).resolve().parent / "datasets"
_OUT = _BASE / "models" / "temporal-embed-v1"

# Hyperparameters
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 256
EVAL_SPLIT = 0.1


def load_triplets() -> list[dict]:
    """Load embedding training triplets from JSONL dataset."""
    path = _DATA / "embedding_triplets.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Train the temporal-aware bi-encoder (C1) on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("Loading triplets...")
    raw = load_triplets()
    log.info("  Total: %d triplets", len(raw))

    # Split train / eval
    split_idx = max(1, int(len(raw) * (1 - EVAL_SPLIT)))
    train_raw = raw[:split_idx]
    eval_raw = raw[split_idx:]

    train_examples = [
        InputExample(texts=[r["anchor"], r["positive"], r["negative"]]) for r in train_raw
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    log.info("Loading base model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN

    # Add 4 temporal special tokens
    special_tokens = ["[TEMPORAL_Q]", "[DATE_ANCHOR]", "[BEFORE]", "[AFTER]"]
    word_embedding_model = model[0]
    tokenizer = word_embedding_model.tokenizer
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    log.info("Added %d special tokens", len(special_tokens))

    # Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    eval_anchors = [r["anchor"] for r in eval_raw]
    eval_positives = [r["positive"] for r in eval_raw]
    eval_negatives = [r["negative"] for r in eval_raw]
    evaluator = TripletEvaluator(
        anchors=eval_anchors,
        positives=eval_positives,
        negatives=eval_negatives,
        name="temporal-eval",
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    log.info(
        "Training: %d examples, %d steps, %d warmup", len(train_examples), total_steps, warmup_steps
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_loader),
        output_path=str(_OUT),
        optimizer_params={"lr": LR},
        weight_decay=0.01,
        show_progress_bar=True,
        use_amp=False,  # FP32 for ROCm stability
    )

    log.info("Model saved to %s", _OUT)

    # Verify
    loaded = SentenceTransformer(str(_OUT), device=device)
    test_emb = loaded.encode(["When did I buy the car?"], convert_to_tensor=True)
    log.info("Verification embedding shape: %s", test_emb.shape)
    log.info("C1 training complete.")


if __name__ == "__main__":
    main()
