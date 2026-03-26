# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Experiment 3: BM25(0.4) + Embedding(0.6) hybrid on LOCOMO
from __future__ import annotations
import math, re, time, json
import numpy as np
from collections import Counter
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

BM25_W = 0.4

results = {}
correct = 0
tested = 0
t0 = time.monotonic()

for conv in ds:
    questions = conv["questions"]
    answers = conv["answers"]
    categories = conv["category"]
    turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]

    # Pre-encode turns
    turn_embs = model.encode(
        [t[:512] for t in turns],
        batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )

    # Pre-tokenize for BM25
    n = len(turns)
    df = Counter()
    para_tokens = []
    for p in turns:
        t = tokenize(p)
        para_tokens.append(t)
        for tok in t:
            df[tok] += 1
    idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
    avg_dl = sum(len(t) for t in para_tokens) / max(n, 1)
    k1, b = 1.5, 0.75

    for qi in range(len(questions)):
        q, a = questions[qi], answers[qi]
        cat = categories[qi] if qi < len(categories) else 0
        cat_name = CATS.get(cat, f"cat-{cat}")

        q_tokens = tokenize(q)

        # BM25 scores
        bm25_scores = []
        for i, pt in enumerate(para_tokens):
            score = 0.0
            dl = len(pt)
            for qt in q_tokens:
                if qt in pt:
                    score += idf.get(qt, 0) * (1 + k1) / (1 + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            bm25_scores.append(score)

        # Embedding scores
        q_emb = model.encode(q, normalize_embeddings=True, convert_to_numpy=True)
        emb_scores = (turn_embs @ q_emb).tolist()

        # Normalize to [0,1] and fuse
        bm25_max = max(bm25_scores) if bm25_scores else 1
        emb_max = max(emb_scores) if emb_scores else 1
        fused = [
            BM25_W * (bs / max(bm25_max, 1e-6)) + (1 - BM25_W) * (es / max(emb_max, 1e-6))
            for bs, es in zip(bm25_scores, emb_scores)
        ]
        top_idx = sorted(range(len(fused)), key=lambda i: -fused[i])[:20]
        retrieved = [(i, fused[i]) for i in top_idx if fused[i] > 0]

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

out = {"experiment": "BM25(0.4)+Embed(0.6) hybrid", "accuracy": round(overall, 1), "elapsed_s": round(elapsed, 1), "by_category": {}}
for cat, s in sorted(results.items()):
    acc = s["correct"] / max(s["total"], 1) * 100
    out["by_category"][cat] = {"correct": s["correct"], "total": s["total"], "accuracy": round(acc, 1)}

Path("exp3_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
