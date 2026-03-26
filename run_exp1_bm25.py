# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Experiment 1: BM25 baseline on LOCOMO
from __future__ import annotations
import math, re, time, json
from collections import Counter
from pathlib import Path

def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))

CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}

from datasets import load_dataset
ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")

results = {}
correct = 0
tested = 0
t0 = time.monotonic()

for conv in ds:
    questions = conv["questions"]
    answers = conv["answers"]
    categories = conv["category"]
    turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]

    for qi in range(len(questions)):
        q, a = questions[qi], answers[qi]
        cat = categories[qi] if qi < len(categories) else 0
        cat_name = CATS.get(cat, f"cat-{cat}")

        q_tokens = tokenize(q)
        if not q_tokens:
            tested += 1
            if cat_name not in results:
                results[cat_name] = {"correct": 0, "total": 0}
            results[cat_name]["total"] += 1
            continue

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
        scored = []
        for i, pt in enumerate(para_tokens):
            score = 0.0
            dl = len(pt)
            for qt in q_tokens:
                if qt in pt:
                    score += idf.get(qt, 0) * (1 + k1) / (1 + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        retrieved = scored[:20]

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

out = {"experiment": "BM25 baseline", "accuracy": round(overall, 1), "elapsed_s": round(elapsed, 1), "by_category": {}}
for cat, s in sorted(results.items()):
    acc = s["correct"] / max(s["total"], 1) * 100
    out["by_category"][cat] = {"correct": s["correct"], "total": s["total"], "accuracy": round(acc, 1)}

Path("exp1_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
