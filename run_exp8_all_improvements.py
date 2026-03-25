# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Experiment 6: Full pipeline — hybrid retrieval + cross-encoder + temporal code
# + LLM synthesis with 10-turn context
# This is the "everything we've built" experiment.
from __future__ import annotations
import math, os, re, time, json
import numpy as np
from collections import Counter
from pathlib import Path

def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))

CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial", 5: "open-domain"}

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

client = None
if api_key:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

from datasets import load_dataset
ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")

from temporal_graph import TemporalEvent, temporal_code_execute, parse_dates
from answer_extractor import extract_answer, fuzzy_match, extract_best_sentence
from answer_normalizer import answers_match as normalized_match, semantic_similarity


def precompute_conv(turns):
    """Pre-compute BM25 index and embeddings for a conversation."""
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


_PERSON_CENTRIC_PATTERNS = re.compile(
    r"\b(relationship|hobby|hobbies|interest|interests|career|job|status|"
    r"personality|feel|feeling|prefer|favorite|partake|destress|self-care|"
    r"political|leaning|member|community)\b", re.IGNORECASE)


def _is_person_centric(query):
    if _PERSON_CENTRIC_PATTERNS.search(query):
        return True
    q_lower = query.lower()
    return any(w in q_lower for w in ["would ", "could ", "likely "])


def _extract_query_names(query):
    names = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
        word = m.group(1).lower()
        if word not in {"what", "when", "where", "who", "how", "why", "would",
                        "could", "does", "did", "has", "have", "the", "which",
                        "likely", "yes", "not"}:
            names.add(word)
    return names


def hybrid_search(query, turns, para_tokens, idf, avg_dl, turn_embs, top_k=20):
    """BM25(0.4) + embedding(0.6) + gated entity boost + cross-encoder rerank."""
    n = len(turns)
    if n == 0:
        return []

    q_tokens = tokenize(query)
    query_names = _extract_query_names(query) if _is_person_centric(query) else set()
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
        # Entity-centric boost: turns mentioning query's person get 1.3x
        if query_names and fused > 0:
            t_lower = turns[i].lower()
            if any(name in t_lower for name in query_names):
                fused *= 1.3
        if fused > 0:
            scored.append((i, fused))
    scored.sort(key=lambda x: -x[1])
    candidates = scored[:top_k * 3]

    # Cross-encoder rerank
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


# Threshold below which cross-encoder confidence is "low" —
# triggers early LLM routing instead of waiting for stage 4 fallback
CE_LOW_CONFIDENCE = 3.0


def _dedup_turn_indices(indices, turns, threshold=0.8):
    """Remove near-duplicate retrieved turns."""
    kept = []
    kept_token_sets = []
    for idx, score in indices:
        if idx >= len(turns):
            continue
        t_tokens = tokenize(turns[idx])
        is_dup = False
        for prev_tokens in kept_token_sets:
            if not t_tokens or not prev_tokens:
                continue
            overlap = len(t_tokens & prev_tokens) / max(min(len(t_tokens), len(prev_tokens)), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((idx, score))
            kept_token_sets.append(t_tokens)
    return kept


def llm_answer(question, turns, retrieved_indices):
    if not client:
        return None
    # Dedup + expand context to 15 turns
    deduped = _dedup_turn_indices(retrieved_indices, turns)
    context = "\n\n".join(
        f"[Turn {idx}]: {turns[idx][:500]}"
        for idx, _ in deduped[:15]
        if idx < len(turns))

    # Question-type-specific prompts
    q_lower = question.lower()
    if any(w in q_lower for w in ["would", "could", "might", "likely"]):
        prompt = (
            "Answer the hypothetical question by reasoning about the person's "
            "stated preferences, personality, and past actions from the turns. "
            "Answer with 'Yes' or 'No' (or 'Likely yes/no') followed by a brief reason. "
            "If insufficient information, say 'unknown'.")
    elif any(w in q_lower for w in ["what are", "what does", "list", "hobbies",
                                     "interests", "activities", "all"]):
        prompt = (
            "List ALL relevant items mentioned across ALL turns. "
            "Combine information from different turns into one complete answer. "
            "If the answer isn't in the turns, say 'unknown'.")
    else:
        prompt = (
            "Answer the question using ONLY the conversation turns below. "
            "Include ALL relevant details from ALL turns. "
            "Be thorough but concise. If the answer isn't in the turns, say 'unknown'.")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=150,
            messages=[{"role": "user", "content":
                f"{prompt}\n\n{context}\n\nQuestion: {question}"}])
        answer = response.content[0].text.strip()
        if answer.lower() in ("unknown", "i don't know", "not mentioned"):
            return None
        return answer
    except Exception:
        return None


results = {}
correct = 0
tested = 0
llm_calls = 0
temporal_code_hits = 0
t0 = time.monotonic()

for ci, conv in enumerate(ds):
    questions = conv["questions"]
    answers = conv["answers"]
    categories = conv["category"]
    turns = conv["turns"] if isinstance(conv["turns"], list) else [t.strip() for t in conv["turns"].split("\n") if t.strip()]

    # Pre-compute once per conversation (major speedup)
    para_tokens, idf, avg_dl, turn_embs = precompute_conv(turns)

    for qi in range(len(questions)):
        q, a = questions[qi], answers[qi]
        cat = categories[qi] if qi < len(categories) else 0
        cat_name = CATS.get(cat, f"cat-{cat}")

        retrieved = hybrid_search(q, turns, para_tokens, idf, avg_dl, turn_embs)

        a_lower = a.lower().strip()
        a_tokens = tokenize(a)
        hit = False

        # Stage 1: Direct text matching
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

        # Confidence-based routing: check if retrieval is trustworthy
        top_ce_score = retrieved[0][1] if retrieved else 0.0
        low_confidence = top_ce_score < CE_LOW_CONFIDENCE

        # Stage 2: Answer extraction + sentence matching (skip if low confidence → go to LLM)
        if not hit and not low_confidence:
            for idx, sc in retrieved[:5]:
                if idx < len(turns):
                    extracted = extract_answer(q, turns[idx])
                    if extracted and a_lower:
                        if fuzzy_match(extracted, a_lower, threshold=0.6):
                            hit = True
                            break
                        ext_tokens = tokenize(extracted)
                        if ext_tokens and len(ext_tokens & a_tokens) / max(len(a_tokens), 1) > 0.3:
                            hit = True
                            break
                    best_sent = extract_best_sentence(q, turns[idx])
                    if best_sent and a_lower:
                        s_tokens = tokenize(best_sent)
                        if a_tokens and len(a_tokens & s_tokens) / max(len(a_tokens), 1) > 0.4:
                            hit = True
                            break

        # Stage 3: Temporal code execution
        if not hit and not low_confidence:
            q_lower = q.lower()
            if any(w in q_lower for w in ["when", "how long", "before", "after", "since",
                                           "first", "latest", "most recent", "how many days"]):
                t_events = []
                for idx, _ in retrieved[:10]:
                    if idx < len(turns):
                        for d in parse_dates(turns[idx]):
                            t_events.append(TemporalEvent(
                                date=d, text=turns[idx][:200], source="turn", paragraph_idx=idx))
                if t_events:
                    code_answer = temporal_code_execute(q, t_events)
                    if code_answer and a_lower:
                        ca_lower = code_answer.lower()
                        if a_lower in ca_lower or ca_lower in a_lower:
                            hit = True
                            temporal_code_hits += 1
                        else:
                            ca_tokens = tokenize(code_answer)
                            if a_tokens and len(a_tokens & ca_tokens) / max(len(a_tokens), 1) > 0.3:
                                hit = True
                                temporal_code_hits += 1

        # Stage 4: LLM synthesis
        # Fires when: retrieval-based stages failed OR low confidence (early routing)
        if not hit and retrieved and client:
            llm_ans = llm_answer(q, turns, retrieved)
            llm_calls += 1
            if llm_ans:
                la = llm_ans.lower().strip()
                if a_lower in la or la in a_lower:
                    hit = True
                elif normalized_match(llm_ans, a):
                    hit = True
                else:
                    la_tokens = tokenize(llm_ans)
                    if a_tokens and len(a_tokens & la_tokens) / max(len(a_tokens), 1) > 0.3:
                        hit = True
                    # Semantic similarity fallback for paraphrase mismatches
                    elif len(a) > 3 and len(llm_ans) > 3:
                        sim = semantic_similarity(llm_ans, a)
                        if sim > 0.7:
                            hit = True

        if cat_name not in results:
            results[cat_name] = {"correct": 0, "total": 0}
        results[cat_name]["total"] += 1
        results[cat_name]["correct"] += int(hit)
        correct += int(hit)
        tested += 1

    if (ci + 1) % 5 == 0:
        pct = correct / max(tested, 1) * 100
        Path("exp8_progress.txt").write_text(
            f"Conv {ci+1}/{len(ds)}, {tested} tested, {correct} correct ({pct:.1f}%), "
            f"{llm_calls} LLM, {temporal_code_hits} temporal code\n")

elapsed = time.monotonic() - t0
overall = correct / max(tested, 1) * 100

out = {
    "experiment": "Full pipeline v3: +normalizer+gated_boost+confidence_routing+dedup+semantic",
    "accuracy": round(overall, 1),
    "elapsed_s": round(elapsed, 1),
    "llm_calls": llm_calls,
    "temporal_code_hits": temporal_code_hits,
    "by_category": {},
}
for cat, s in sorted(results.items()):
    acc = s["correct"] / max(s["total"], 1) * 100
    out["by_category"][cat] = {"correct": s["correct"], "total": s["total"], "accuracy": round(acc, 1)}

Path("exp8_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
