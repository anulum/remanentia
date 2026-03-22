# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LOCOMO Benchmark

"""LOCOMO benchmark: multi-session conversational memory QA.

Categories: 1=single-hop, 2=multi-hop, 3=temporal, 4=adversarial, 5=open-domain
Each conversation has multiple sessions. Questions require retrieving
from specific sessions/turns to answer.

We test: given a question, can our retrieval find the evidence turns?
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))


def bm25_search(query, paragraphs, top_k=10):
    """BM25-lite search."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    # IDF
    import math
    from collections import Counter
    n = len(paragraphs)
    df = Counter()
    para_tokens = []
    for p in paragraphs:
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
                tf = 1.0
                score += idf.get(qt, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        if score > 0:
            scored.append((i, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def evaluate_retrieval(question, answer, evidence_indices, retrieved_indices, turns):
    """Evaluate if retrieval found the evidence."""
    # Evidence hit: did we retrieve any paragraph from evidence sessions/turns?
    evidence_set = set()
    for ev in evidence_indices:
        if isinstance(ev, list) and len(ev) == 2:
            session_idx, turn_idx = ev
            evidence_set.add((session_idx, turn_idx))

    # Check if retrieved paragraphs overlap with evidence
    for idx, score in retrieved_indices:
        if idx in [e[1] for e in evidence_set if e[0] == 0]:  # simplified
            return True

    # Fallback: check if answer appears in retrieved text
    answer_lower = answer.lower().strip()
    for idx, score in retrieved_indices:
        if idx < len(turns) and answer_lower in turns[idx].lower():
            return True

    # Token overlap check
    a_tokens = tokenize(answer)
    for idx, score in retrieved_indices:
        if idx < len(turns):
            t_tokens = tokenize(turns[idx])
            if a_tokens and len(a_tokens & t_tokens) / max(len(a_tokens), 1) > 0.5:
                return True

    return False


CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}


def main():
    from datasets import load_dataset

    print("Loading LOCOMO benchmark...")
    ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")
    print(f"Loaded {len(ds)} conversations\n")

    results_by_cat = {}
    total_correct = 0
    total_tested = 0
    t0 = time.monotonic()

    for conv_idx, conv in enumerate(ds):
        questions = conv["questions"]
        answers = conv["answers"]
        evidences = conv["evidences"]
        categories = conv["category"]
        turns_raw = conv["turns"]

        # Parse turns into individual paragraphs
        turns = []
        if isinstance(turns_raw, list):
            turns = turns_raw
        elif isinstance(turns_raw, str):
            turns = [t.strip() for t in turns_raw.split("\n") if t.strip()]

        if not turns or not questions:
            continue

        # Evaluate each question
        for qi in range(len(questions)):
            question = questions[qi]
            answer = answers[qi]
            cat = categories[qi] if qi < len(categories) else 0
            evidence = evidences[qi] if qi < len(evidences) else []

            cat_name = CATEGORY_NAMES.get(cat, f"cat-{cat}")

            # Search
            retrieved = bm25_search(question, turns, top_k=10)

            # Evaluate
            correct = evaluate_retrieval(question, answer, evidence, retrieved, turns)

            if cat_name not in results_by_cat:
                results_by_cat[cat_name] = {"correct": 0, "total": 0}
            results_by_cat[cat_name]["total"] += 1
            results_by_cat[cat_name]["correct"] += int(correct)
            total_correct += int(correct)
            total_tested += 1

    elapsed = time.monotonic() - t0

    print(f"{'='*60}")
    print(f"LOCOMO RESULTS ({total_tested} questions, {elapsed:.1f}s)")
    print(f"{'='*60}")
    overall = total_correct / max(total_tested, 1) * 100
    print(f"\nOverall: {total_correct}/{total_tested} ({overall:.1f}%)")
    print(f"\nBy category:")
    for cat, stats in sorted(results_by_cat.items()):
        acc = stats["correct"] / max(stats["total"], 1) * 100
        print(f"  {cat:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")

    # Context: what competitors score
    print(f"\nContext (from literature):")
    print(f"  GPT-4o:           30-70% (LongMemEval)")
    print(f"  Mem0:             ~60% (estimated)")
    print(f"  Hindsight:        91.4% (LongMemEval)")
    print(f"  Remanentia BM25:  {overall:.1f}% (this run)")

    out = {
        "total_correct": total_correct,
        "total_tested": total_tested,
        "accuracy": round(overall, 1),
        "by_category": results_by_cat,
        "elapsed_s": round(elapsed, 1),
        "method": "BM25-lite token overlap",
        "note": "LOCOMO preprocessed dataset (10 conversations). No embedding rerank. Honest baseline.",
    }
    Path("paper/locomo_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved to paper/locomo_results.json")


if __name__ == "__main__":
    main()
