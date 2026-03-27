# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Failure Analysis
from __future__ import annotations
import math, os, re, time, json
import numpy as np
from collections import Counter
from pathlib import Path

def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))

CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

from datasets import load_dataset
ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")
from answer_extractor import extract_answer, fuzzy_match, extract_best_sentence


def precompute_conv(turns):
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
    turn_embs = embed_model.encode(
        [p[:512] for p in turns], batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True)
    return para_tokens, idf, avg_dl, turn_embs


def hybrid_search(query, turns, para_tokens, idf, avg_dl, turn_embs, top_k=20):
    n = len(turns)
    if n == 0:
        return []
    q_tokens = tokenize(query)
    k1, b = 1.5, 0.75
    bm25_scores = []
    for i, pt in enumerate(para_tokens):
        score = 0.0
        dl = len(pt)
        for qt in q_tokens:
            if qt in pt:
                score += idf.get(qt, 0) * (1 + k1) / (1 + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        bm25_scores.append(score)
    q_emb = embed_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    emb_scores = (turn_embs @ q_emb).tolist()
    bm25_max = max(bm25_scores) if bm25_scores else 1
    emb_max = max(emb_scores) if emb_scores else 1
    scored = []
    for i in range(n):
        fused = (0.4 * bm25_scores[i] / max(bm25_max, 1e-6) +
                 0.6 * emb_scores[i] / max(emb_max, 1e-6))
        if fused > 0:
            scored.append((i, fused))
    scored.sort(key=lambda x: -x[1])
    candidates = scored[:top_k * 3]
    if candidates:
        pairs = [(query, turns[idx][:512]) for idx, _ in candidates]
        try:
            ce_scores = ce_model.predict(pairs, show_progress_bar=False)
            reranked = [(candidates[i][0], float(ce_scores[i])) for i in range(len(candidates))]
            reranked.sort(key=lambda x: -x[1])
            return reranked[:top_k]
        except Exception:
            pass
    return candidates[:top_k]


failures = {"single-hop": [], "multi-hop": [], "temporal": [], "adversarial": [], "open-domain": []}
t0 = time.monotonic()

for ci, conv in enumerate(ds):
    questions = conv["questions"]
    answers = conv["answers"]
    categories = conv["category"]
    turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]
    para_tokens, idf, avg_dl, turn_embs = precompute_conv(turns)

    for qi in range(len(questions)):
        q, a = questions[qi], answers[qi]
        cat = categories[qi] if qi < len(categories) else 0
        cat_name = CATS.get(cat, f"cat-{cat}")

        retrieved = hybrid_search(q, turns, para_tokens, idf, avg_dl, turn_embs)
        a_lower = a.lower().strip()
        a_tokens = tokenize(a)
        hit = False

        for idx, sc in retrieved:
            if idx < len(turns) and a_lower in turns[idx].lower():
                hit = True; break
        if not hit:
            for idx, sc in retrieved:
                if idx < len(turns):
                    t_tokens = tokenize(turns[idx])
                    if a_tokens and len(a_tokens & t_tokens) / max(len(a_tokens), 1) > 0.3:
                        hit = True; break
        if not hit and a_tokens and len(a_tokens) > 2:
            covered = set()
            for idx, sc in retrieved:
                if idx < len(turns):
                    covered |= (a_tokens & tokenize(turns[idx]))
            if len(covered) / max(len(a_tokens), 1) > 0.5:
                hit = True
        if not hit:
            for idx, sc in retrieved[:5]:
                if idx < len(turns):
                    extracted = extract_answer(q, turns[idx])
                    if extracted and a_lower:
                        if fuzzy_match(extracted, a_lower, threshold=0.6):
                            hit = True; break

        if not hit and cat_name in failures:
            top_turn = turns[retrieved[0][0]][:200] if retrieved else "NO RETRIEVAL"
            failures[cat_name].append({
                "q": q[:150], "a": a[:100],
                "top_turn": top_turn,
                "n_retrieved": len(retrieved),
            })

elapsed = time.monotonic() - t0
print(f"Analysis done in {elapsed:.0f}s\n")

for cat, fails in sorted(failures.items()):
    print(f"\n{'='*60}")
    print(f"{cat}: {len(fails)} failures (retrieval+extraction only, no LLM)")
    print(f"{'='*60}")
    for f in fails[:5]:
        print(f"\nQ: {f['q']}")
        print(f"A: {f['a']}")
        print(f"Top: {f['top_turn'][:150]}")
        print(f"Retrieved: {f['n_retrieved']}")

Path("failure_analysis_v2.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
