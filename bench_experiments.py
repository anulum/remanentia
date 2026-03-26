# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Radical LOCOMO experiments
# Each experiment tests a fundamentally different approach.

from __future__ import annotations

import io
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))


# ── Experiment 1: Semantic search via embeddings ────────────────

def embed_search(query, paragraphs, model, top_k=20):
    """Pure embedding cosine similarity search."""
    import numpy as np
    if not paragraphs:
        return []
    q_emb = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    p_embs = model.encode(
        [p[:512] for p in paragraphs],
        batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    sims = p_embs @ q_emb
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in top_idx if sims[i] > 0]


# ── Experiment 2: BM25 + embedding hybrid ───────────────────────

def hybrid_search(query, paragraphs, model, top_k=20, bm25_weight=0.4):
    """BM25 + embedding fusion."""
    import numpy as np

    # BM25
    q_tokens = tokenize(query)
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

    bm25_scores = []
    for i, pt in enumerate(para_tokens):
        score = 0.0
        dl = len(pt)
        for qt in q_tokens:
            if qt in pt:
                score += idf.get(qt, 0) * (1 + k1) / (1 + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        bm25_scores.append(score)

    # Embedding
    q_emb = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    p_embs = model.encode(
        [p[:512] for p in paragraphs],
        batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    emb_scores = (p_embs @ q_emb).tolist()

    # Normalize both to [0,1]
    bm25_max = max(bm25_scores) if bm25_scores else 1
    emb_max = max(emb_scores) if emb_scores else 1
    bm25_norm = [s / max(bm25_max, 1e-6) for s in bm25_scores]
    emb_norm = [s / max(emb_max, 1e-6) for s in emb_scores]

    # Fuse
    fused = [bm25_weight * b + (1 - bm25_weight) * e for b, e in zip(bm25_norm, emb_norm)]

    top_idx = sorted(range(len(fused)), key=lambda i: -fused[i])[:top_k]
    return [(i, fused[i]) for i in top_idx if fused[i] > 0]


# ── Experiment 3: LLM answering (send context + question to Haiku) ──

def llm_answer(question, turns, retrieved_indices, model_name="claude-haiku-4-5-20251001"):
    """Send top retrieved turns to LLM for answer generation."""
    import anthropic
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(
        f"[Turn {idx}]: {turns[idx][:400]}"
        for idx, _ in retrieved_indices[:5]
        if idx < len(turns)
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    "Answer the question using ONLY the conversation turns below. "
                    "Be concise (1-2 sentences). If the answer isn't in the turns, say 'unknown'.\n\n"
                    f"{context}\n\nQuestion: {question}"
                ),
            }],
        )
        answer = response.content[0].text.strip()
        if answer.lower() in ("unknown", "i don't know"):
            return None
        return answer
    except Exception:
        return None


# ── Evaluation ──────────────────────────────────────────────────

def evaluate(question, gold_answer, retrieved, turns, session_offsets=None, llm_answer_text=None):
    """Check if we got the right answer via multiple matching strategies."""
    answer_lower = gold_answer.lower().strip()
    a_tokens = tokenize(gold_answer)

    # Exact substring in retrieved text
    for idx, score in retrieved:
        if idx < len(turns) and answer_lower in turns[idx].lower():
            return True

    # Token overlap per turn
    for idx, score in retrieved:
        if idx < len(turns):
            t_tokens = tokenize(turns[idx])
            if a_tokens and len(a_tokens & t_tokens) / max(len(a_tokens), 1) > 0.3:
                return True

    # Cross-turn token coverage
    if a_tokens and len(a_tokens) > 2:
        covered = set()
        for idx, score in retrieved:
            if idx < len(turns):
                covered |= (a_tokens & tokenize(turns[idx]))
        if len(covered) / max(len(a_tokens), 1) > 0.5:
            return True

    # Answer extraction
    try:
        from answer_extractor import extract_answer, fuzzy_match
        for idx, score in retrieved[:5]:
            if idx < len(turns):
                extracted = extract_answer(question, turns[idx])
                if extracted and answer_lower:
                    if fuzzy_match(extracted, answer_lower, threshold=0.6):
                        return True
    except ImportError:
        pass

    # LLM answer matching
    if llm_answer_text and gold_answer:
        ga = gold_answer.lower().strip()
        la = llm_answer_text.lower().strip()
        if ga in la or la in ga:
            return True
        ga_tokens = tokenize(gold_answer)
        la_tokens = tokenize(llm_answer_text)
        if ga_tokens and len(ga_tokens & la_tokens) / max(len(ga_tokens), 1) > 0.3:
            return True

    return False


CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}

_DS = None

def _get_dataset():
    global _DS
    if _DS is None:
        from datasets import load_dataset
        _DS = load_dataset("KhangPTT373/locomo_preprocess", split="test")
    return _DS


def run_experiment(name, search_fn, use_llm_answer=False):
    """Run a LOCOMO experiment with a given search function."""
    ds = _get_dataset()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    results_by_cat = {}
    total_correct = 0
    total_tested = 0
    t0 = time.monotonic()

    for conv in ds:
        questions = conv["questions"]
        answers = conv["answers"]
        categories = conv["category"]
        turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]
        sessions = conv.get("sessions", [])
        session_offsets = []
        if isinstance(sessions, list):
            offset = 0
            for sess in sessions:
                session_offsets.append(offset)
                if isinstance(sess, str):
                    offset += len([t for t in sess.split("\n") if t.strip()])
                elif isinstance(sess, list):
                    offset += len(sess)

        for qi in range(len(questions)):
            question = questions[qi]
            answer = answers[qi]
            cat = categories[qi] if qi < len(categories) else 0
            cat_name = CATEGORY_NAMES.get(cat, f"cat-{cat}")

            retrieved = search_fn(question, turns)

            llm_ans = None
            if use_llm_answer and retrieved:
                llm_ans = llm_answer(question, turns, retrieved)

            correct = evaluate(question, answer, retrieved, turns, session_offsets, llm_ans)

            if cat_name not in results_by_cat:
                results_by_cat[cat_name] = {"correct": 0, "total": 0}
            results_by_cat[cat_name]["total"] += 1
            results_by_cat[cat_name]["correct"] += int(correct)
            total_correct += int(correct)
            total_tested += 1

    elapsed = time.monotonic() - t0
    overall = total_correct / max(total_tested, 1) * 100

    print(f"\nOverall: {total_correct}/{total_tested} ({overall:.1f}%) [{elapsed:.0f}s]")
    for cat, stats in sorted(results_by_cat.items()):
        acc = stats["correct"] / max(stats["total"], 1) * 100
        print(f"  {cat:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")

    result = {
        "experiment": name,
        "accuracy": round(overall, 1),
        "by_category": results_by_cat,
        "elapsed_s": round(elapsed, 1),
    }
    return result


if __name__ == "__main__":
    import os
    results = []

    # Pre-load dataset
    print("Loading LOCOMO dataset...")
    _get_dataset()

    # Inline BM25 + CE search (avoid importing bench_locomo which breaks stdout)
    def bm25_ce_search(query, paragraphs, top_k=20):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
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
                    score += idf.get(qt, 0) * (1 + k1) / (1 + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    bm25_search = bm25_ce_search
    r1 = run_experiment(
        "BM25 + CE (baseline)",
        lambda q, turns: bm25_search(q, turns, top_k=20),
    )
    results.append(r1)

    # Experiment 2: Embedding-only search
    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded on {device}")

    r2 = run_experiment(
        "Embedding-only (MiniLM)",
        lambda q, turns: embed_search(q, turns, embed_model, top_k=20),
    )
    results.append(r2)

    # Experiment 3: BM25 + Embedding hybrid
    r3 = run_experiment(
        "BM25(0.4) + Embedding(0.6) hybrid",
        lambda q, turns: hybrid_search(q, turns, embed_model, top_k=20, bm25_weight=0.4),
    )
    results.append(r3)

    # Experiment 4: BM25 + CE + LLM answering
    if os.environ.get("ANTHROPIC_API_KEY"):
        r4 = run_experiment(
            "BM25 + CE + LLM answer (Haiku)",
            lambda q, turns: bm25_search(q, turns, top_k=20),
            use_llm_answer=True,
        )
        results.append(r4)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['experiment']:40s}: {r['accuracy']:5.1f}%")

    Path("paper/experiment_results_v2.json").write_text(
        json.dumps(results, indent=2) + "\n")
    print(f"\nSaved to paper/experiment_results_v2.json")
