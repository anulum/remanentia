# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Experiment 2: Embedding-only (MiniLM) on LOCOMO
from __future__ import annotations
import re, time, json
import numpy as np
from pathlib import Path

def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))

CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}

from datasets import load_dataset
ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")

import torch
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

results = {}
correct = 0
tested = 0
t0 = time.monotonic()

for ci, conv in enumerate(ds):
    questions = conv["questions"]
    answers = conv["answers"]
    categories = conv["category"]
    turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]

    # Pre-encode all turns for this conversation
    turn_embs = model.encode(
        [t[:512] for t in turns],
        batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )

    for qi in range(len(questions)):
        q, a = questions[qi], answers[qi]
        cat = categories[qi] if qi < len(categories) else 0
        cat_name = CATS.get(cat, f"cat-{cat}")

        q_emb = model.encode(q, normalize_embeddings=True, convert_to_numpy=True)
        sims = turn_embs @ q_emb
        top_idx = np.argsort(-sims)[:20]
        retrieved = [(int(i), float(sims[i])) for i in top_idx if sims[i] > 0]

        a_lower = a.lower().strip()
        a_tokens = tokenize(a)
        hit = False
        for idx, sc in retrieved:
            if idx < len(turns) and a_lower in turns[idx].lower():
                hit = True
                break
        if not hit:
            for idx, sc in retrieved:
                if idx < len(turns):
                    t_tokens = tokenize(turns[idx])
                    if a_tokens and len(a_tokens & t_tokens) / max(len(a_tokens), 1) > 0.3:
                        hit = True
                        break
        if not hit and a_tokens and len(a_tokens) > 2:
            covered = set()
            for idx, sc in retrieved:
                if idx < len(turns):
                    covered |= (a_tokens & tokenize(turns[idx]))
            if len(covered) / max(len(a_tokens), 1) > 0.5:
                hit = True

        if cat_name not in results:
            results[cat_name] = {"correct": 0, "total": 0}
        results[cat_name]["total"] += 1
        results[cat_name]["correct"] += int(hit)
        correct += int(hit)
        tested += 1

elapsed = time.monotonic() - t0
overall = correct / max(tested, 1) * 100

out = {"experiment": "Embedding-only (MiniLM)", "accuracy": round(overall, 1), "elapsed_s": round(elapsed, 1), "by_category": {}}
for cat, s in sorted(results.items()):
    acc = s["correct"] / max(s["total"], 1) * 100
    out["by_category"][cat] = {"correct": s["correct"], "total": s["total"], "accuracy": round(acc, 1)}

Path("exp2_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
