# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C5: Train fact validity model — fact type + supersession detection.

Architecture: bert-mini + multi-task heads.

Usage:
    CUDA_VISIBLE_DEVICES=4 python training/train_fact_validity.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_DATA = Path(__file__).resolve().parent / "datasets"
_OUT = _BASE / "models" / "fact-validity-v1"

# Hyperparameters
MODEL_NAME = "prajjwal1/bert-mini"
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 128
EVAL_SPLIT = 0.15

FACT_TYPES = ["state", "event", "preference", "plan"]
FTYPE2ID = {t: i for i, t in enumerate(FACT_TYPES)}


class FactValidityDataset(Dataset):
    """PyTorch dataset for fact classification training."""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        """Initialise with raw samples, tokenizer, and max sequence length."""
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return tokenised fact text with type, supersedes, and boundary labels."""
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "fact_type": torch.tensor(FTYPE2ID.get(s["fact_type"], 0), dtype=torch.long),
            "supersedes": torch.tensor(float(s["supersedes"]), dtype=torch.float),
            "has_boundary": torch.tensor(float(s["has_boundary"]), dtype=torch.float),
        }


class FactValidityModel(nn.Module):
    """BERT-mini backbone with type + supersedes + boundary heads."""

    def __init__(self, model_name: str, num_types: int = 4):
        """Build backbone + type classification + supersedes + boundary heads."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.type_head = nn.Linear(hidden, num_types)
        self.supersedes_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask):
        """Return (type_logits, supersedes_logit, boundary_logit) from [CLS]."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0]
        type_logits = self.type_head(cls_emb)
        supersedes_logit = self.supersedes_head(cls_emb).squeeze(-1)
        boundary_logit = self.boundary_head(cls_emb).squeeze(-1)
        return type_logits, supersedes_logit, boundary_logit


def load_data() -> list[dict]:
    """Load fact validity training data from synthetic JSONL."""
    path = _DATA / "fact_validity_synth.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Train the fact validity model (C5) on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("Loading data...")
    raw = load_data()
    log.info("  Total: %d samples", len(raw))
    dist = Counter(s["fact_type"] for s in raw)
    log.info("  Types: %s", dict(dist))
    log.info("  Supersedes: %d", sum(1 for s in raw if s["supersedes"]))

    # Split
    split_idx = int(len(raw) * (1 - EVAL_SPLIT))
    train_raw = raw[:split_idx]
    eval_raw = raw[split_idx:]

    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = FactValidityDataset(train_raw, tokenizer)
    eval_ds = FactValidityDataset(eval_raw, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    model = FactValidityModel(MODEL_NAME).to(device)
    type_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info("Training: %d examples, %d steps, %d epochs", len(train_raw), total_steps, EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            fact_type = batch["fact_type"].to(device)
            supersedes = batch["supersedes"].to(device)
            has_boundary = batch["has_boundary"].to(device)

            type_logits, sup_logit, bnd_logit = model(input_ids, attention_mask)

            loss = (
                type_criterion(type_logits, fact_type)
                + bce_criterion(sup_logit, supersedes)
                + 0.5 * bce_criterion(bnd_logit, has_boundary)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        type_correct = 0
        sup_correct = 0
        bnd_correct = 0
        total_eval = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                fact_type = batch["fact_type"]
                supersedes = batch["supersedes"]
                has_boundary = batch["has_boundary"]

                type_logits, sup_logit, bnd_logit = model(input_ids, attention_mask)
                type_preds = type_logits.argmax(dim=-1).cpu()
                sup_preds = (torch.sigmoid(sup_logit) > 0.5).float().cpu()
                bnd_preds = (torch.sigmoid(bnd_logit) > 0.5).float().cpu()

                total_eval += len(fact_type)
                type_correct += (type_preds == fact_type).sum().item()
                sup_correct += (sup_preds == supersedes).sum().item()
                bnd_correct += (bnd_preds == has_boundary).sum().item()

        type_acc = type_correct / max(total_eval, 1)
        sup_acc = sup_correct / max(total_eval, 1)
        bnd_acc = bnd_correct / max(total_eval, 1)
        combined = (type_acc + sup_acc + bnd_acc) / 3

        log.info(
            "Epoch %d/%d: loss=%.4f, type_acc=%.3f, sup_acc=%.3f, bnd_acc=%.3f, combined=%.3f",
            epoch + 1, EPOCHS, avg_loss, type_acc, sup_acc, bnd_acc, combined,
        )

        if combined > best_acc:
            best_acc = combined
            _OUT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _OUT / "model.pt")
            tokenizer.save_pretrained(str(_OUT))
            config = {
                "model_name": MODEL_NAME,
                "fact_types": FACT_TYPES,
                "max_seq_len": MAX_SEQ_LEN,
                "best_combined_accuracy": best_acc,
            }
            with open(_OUT / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            log.info("  Saved best model (combined=%.3f)", best_acc)

    log.info("C5 training complete. Best combined accuracy: %.3f", best_acc)


if __name__ == "__main__":
    main()
