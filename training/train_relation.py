# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C3: Train temporal relation classifier (before/after/same_day/overlaps/contains/unknown).

Usage:
    CUDA_VISIBLE_DEVICES=2 python training/train_relation.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_DATA = Path(__file__).resolve().parent / "datasets"
_OUT = _BASE / "models" / "temporal-relation-v1"

# Hyperparameters
MODEL_NAME = "prajjwal1/bert-small"
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 128
EVAL_SPLIT = 0.15
HIDDEN = 512
NUM_CLASSES = 6

LABELS = ["before", "after", "same_day", "overlaps", "contains", "unknown"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


class RelationDataset(Dataset):
    """PyTorch dataset for temporal relation event pairs."""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        """Initialise with raw samples, tokenizer, and max sequence length."""
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return tokenised input_ids, attention_mask, and label for *idx*."""
        s = self.samples[idx]
        enc = self.tokenizer(
            s["event_a"],
            s["event_b"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL2ID[s["relation"]], dtype=torch.long),
        }


class TemporalRelationClassifier(nn.Module):
    """BERT-small backbone with 6-class classification head."""

    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES):
        """Build backbone + 2-layer classification head."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        """Return 6-class logits from [CLS] embedding."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0]
        return self.classifier(cls_emb)


def load_data() -> list[dict]:
    """Load temporal relation training data from JSONL dataset."""
    path = _DATA / "temporal_relations_synth.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Train the temporal relation classifier (C3) on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("Loading data...")
    raw = load_data()
    log.info("  Total: %d samples", len(raw))

    # Class distribution
    dist = Counter(s["relation"] for s in raw)
    log.info("  Distribution: %s", dict(dist))

    # Class weights (inverse frequency)
    total = len(raw)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * dist.get(l, 1)) for l in LABELS],
        dtype=torch.float32,
    ).to(device)

    # Split
    split_idx = int(len(raw) * (1 - EVAL_SPLIT))
    train_raw = raw[:split_idx]
    eval_raw = raw[split_idx:]

    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = RelationDataset(train_raw, tokenizer)
    eval_ds = RelationDataset(eval_raw, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    model = TemporalRelationClassifier(MODEL_NAME).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info("Training: %d examples, %d steps, %d epochs", len(train_raw), total_steps, EPOCHS)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        # Macro F1
        from sklearn.metrics import f1_score

        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        log.info(
            "Epoch %d/%d: loss=%.4f, acc=%.3f, F1-macro=%.3f", epoch + 1, EPOCHS, avg_loss, acc, f1
        )

        if f1 > best_f1:
            best_f1 = f1
            _OUT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _OUT / "model.pt")
            tokenizer.save_pretrained(str(_OUT))
            # Save config
            config = {
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "labels": LABELS,
                "max_seq_len": MAX_SEQ_LEN,
                "best_f1": best_f1,
            }
            with open(_OUT / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            log.info("  Saved best model (F1=%.3f)", best_f1)

    log.info("C3 training complete. Best F1-macro: %.3f", best_f1)


if __name__ == "__main__":
    main()
