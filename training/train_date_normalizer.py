# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C4: Train date normaliser — vague expressions to ISO dates.

Architecture: bert-mini + 8-digit classification (YYYY-MM-DD) + confidence head.

Usage:
    CUDA_VISIBLE_DEVICES=3 python training/train_date_normalizer.py
"""

from __future__ import annotations

import json
import logging
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
_OUT = _BASE / "models" / "date-normalizer-v1"

# Hyperparameters
MODEL_NAME = "prajjwal1/bert-mini"
EPOCHS = 15
BATCH_SIZE = 64
LR = 5e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 64
EVAL_SPLIT = 0.1
NUM_DIGITS = 8  # YYYYMMDD


def _date_to_digits(iso_date: str) -> list[int]:
    """Convert ISO date 'YYYY-MM-DD' to list of 8 digit ints."""
    clean = iso_date.replace("-", "")
    if len(clean) != 8:
        return [0] * 8
    return [int(c) for c in clean]


class DateNormDataset(Dataset):
    """PyTorch dataset for date normalisation (expression, reference) pairs."""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        """Initialise with raw samples, tokenizer, and max sequence length."""
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return tokenised reference+expression input and 8-digit target."""
        s = self.samples[idx]
        # Input: "reference_date: YYYY-MM-DD [SEP] expression"
        text = f"reference: {s['ref_date']} expression: {s['expr']}"
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        digits = _date_to_digits(s["target_date"])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "digits": torch.tensor(digits, dtype=torch.long),
        }


class DateNormalizer(nn.Module):
    """BERT-mini backbone with 8-digit classification + confidence head."""

    def __init__(self, model_name: str):
        """Build backbone + 8 digit classification heads + confidence head."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        # 8 digit heads: each predicts one digit (0-9)
        self.digit_heads = nn.ModuleList([nn.Linear(hidden, 10) for _ in range(NUM_DIGITS)])
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        """Return list of 8 digit logit tensors and confidence scalar."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0]  # [CLS]
        digit_logits = [head(cls_emb) for head in self.digit_heads]  # list of (B, 10)
        confidence = self.confidence_head(cls_emb).squeeze(-1)  # (B,)
        return digit_logits, confidence


def load_data() -> list[dict]:
    """Load date normalisation training data from synthetic JSONL."""
    samples = []
    # Synthetic data
    synth_path = _DATA / "date_normalisation_synth.jsonl"
    with open(synth_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    # Natural data (no target_date, skip for training)
    return [s for s in samples if "target_date" in s and s["target_date"]]


def main() -> None:
    """Train the date normaliser (C4) on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("Loading data...")
    raw = load_data()
    log.info("  Total: %d samples", len(raw))

    # Split
    split_idx = int(len(raw) * (1 - EVAL_SPLIT))
    train_raw = raw[:split_idx]
    eval_raw = raw[split_idx:]

    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = DateNormDataset(train_raw, tokenizer)
    eval_ds = DateNormDataset(eval_raw, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    model = DateNormalizer(MODEL_NAME).to(device)
    digit_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info("Training: %d examples, %d steps, %d epochs", len(train_raw), total_steps, EPOCHS)

    best_exact = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digits = batch["digits"].to(device)  # (B, 8)

            digit_logits, confidence = model(input_ids, attention_mask)

            # Digit loss: sum of 8 CE losses
            loss = sum(digit_criterion(digit_logits[i], digits[:, i]) for i in range(NUM_DIGITS))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        exact_match = 0
        relaxed_match = 0  # +/- 3 days
        total_eval = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                digits = batch["digits"]  # (B, 8)

                digit_logits, confidence = model(input_ids, attention_mask)
                pred_digits = torch.stack(
                    [dl.argmax(dim=-1).cpu() for dl in digit_logits], dim=1
                )  # (B, 8)

                for pred, gold in zip(pred_digits, digits):
                    total_eval += 1
                    if torch.equal(pred, gold):
                        exact_match += 1
                    # Relaxed: parse to date and check +/- 3 days
                    try:
                        pred_str = "".join(str(d.item()) for d in pred)
                        gold_str = "".join(str(d.item()) for d in gold)
                        from datetime import date as dt_date

                        pred_date = dt_date(
                            int(pred_str[:4]), int(pred_str[4:6]), int(pred_str[6:8])
                        )
                        gold_date = dt_date(
                            int(gold_str[:4]), int(gold_str[4:6]), int(gold_str[6:8])
                        )
                        if abs((pred_date - gold_date).days) <= 3:
                            relaxed_match += 1
                    except (ValueError, IndexError):
                        pass

        exact_acc = exact_match / max(total_eval, 1)
        relaxed_acc = relaxed_match / max(total_eval, 1)

        log.info(
            "Epoch %d/%d: loss=%.4f, exact=%.3f, relaxed(±3d)=%.3f",
            epoch + 1,
            EPOCHS,
            avg_loss,
            exact_acc,
            relaxed_acc,
        )

        if exact_acc > best_exact:
            best_exact = exact_acc
            _OUT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _OUT / "model.pt")
            tokenizer.save_pretrained(str(_OUT))
            config = {
                "model_name": MODEL_NAME,
                "num_digits": NUM_DIGITS,
                "max_seq_len": MAX_SEQ_LEN,
                "best_exact_accuracy": best_exact,
            }
            with open(_OUT / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            log.info("  Saved best model (exact=%.3f)", best_exact)

    log.info("C4 training complete. Best exact accuracy: %.3f", best_exact)


if __name__ == "__main__":
    main()
