# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extract training data from LongMemEval for C1 (embedding) and C2 (cross-encoder).

Also extracts natural date normalisation examples for C4.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

_RNG = random.Random(42)
_BASE = Path(__file__).resolve().parent.parent
_DATA = _BASE / "data"
_OUT = Path(__file__).resolve().parent / "datasets"

# ---- tokeniser (matches remanentia's own) ----
_TOK_RE = re.compile(r"\w{3,}")


def _tokenise(text: str) -> set[str]:
    """Extract lower-cased tokens of length >= 3 from *text*."""
    return {m.lower() for m in _TOK_RE.findall(text)}


def _bm25_score(
    q_tokens: set[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 relevance score between a query and a document."""
    dl = len(doc_tokens)
    tf_map = Counter(doc_tokens)
    score = 0.0
    for t in q_tokens:
        if t not in tf_map:
            continue
        tf = tf_map[t]
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf.get(t, 0.0) * numerator / denominator
    return score


def load_oracle() -> list[dict]:
    """Load the LongMemEval oracle dataset (500 questions with sessions)."""
    with open(_DATA / "longmemeval_oracle.json", encoding="utf-8") as f:
        return json.load(f)


def _flatten_turns(sessions: list[list[dict]]) -> list[str]:
    """Flatten all turns into a list of content strings."""
    turns = []
    for sess in sessions:
        for turn in sess:
            content = turn.get("content", "").strip()
            if len(content) >= 20:
                turns.append(content)
    return turns


def generate_embedding_triplets(data: list[dict]) -> list[dict]:
    """C1: Generate (anchor, positive, negative) triplets for bi-encoder training.

    For each question:
      anchor = question text
      positive = turn containing the gold answer (substring match)
      hard_negative = highest BM25 turn that does NOT contain the answer
    """
    triplets: list[dict] = []

    for q in data:
        question = q["question"]
        answer = str(q["answer"]).lower()
        turns = _flatten_turns(q.get("haystack_sessions", []))
        if not turns:
            continue

        q_tokens = _tokenise(question)

        # Identify positives (turns containing the answer)
        positives = []
        negatives = []
        for turn in turns:
            if answer in turn.lower():
                positives.append(turn)
            else:
                negatives.append(turn)

        if not positives:
            # Fallback: best token overlap
            overlaps = [
                (turn, len(_tokenise(answer) & _tokenise(turn)))
                for turn in turns
            ]
            overlaps.sort(key=lambda x: -x[1])
            if overlaps and overlaps[0][1] > 0:
                positives = [overlaps[0][0]]
                negatives = [t for t, _ in overlaps[1:]]

        if not positives or not negatives:
            continue

        # Compute IDF for BM25 scoring of negatives
        doc_count = len(negatives)
        df: dict[str, int] = {}
        neg_token_lists = []
        for neg in negatives:
            toks = list(_tokenise(neg))
            neg_token_lists.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1 + doc_count / (1 + c)) for t, c in df.items()}
        avg_dl = sum(len(tl) for tl in neg_token_lists) / max(len(neg_token_lists), 1)

        # Score negatives by BM25 (hardest negatives first)
        scored_negs = []
        for neg, toks in zip(negatives, neg_token_lists):
            sc = _bm25_score(q_tokens, toks, idf, avg_dl)
            scored_negs.append((neg, sc))
        scored_negs.sort(key=lambda x: -x[1])

        # Create triplets: each positive paired with top-5 hard negatives
        for pos in positives[:3]:
            for neg, _ in scored_negs[:5]:
                triplets.append({
                    "anchor": question,
                    "positive": pos[:512],
                    "negative": neg[:512],
                    "qtype": q.get("question_type", ""),
                })

    _RNG.shuffle(triplets)
    return triplets


def generate_cross_encoder_pairs(data: list[dict]) -> list[dict]:
    """C2: Generate (query, passage, label) pairs for cross-encoder training.

    label=1 if passage contains the answer, label=0 otherwise.
    """
    pairs: list[dict] = []

    for q in data:
        question = q["question"]
        answer = str(q["answer"]).lower()
        turns = _flatten_turns(q.get("haystack_sessions", []))
        if not turns:
            continue

        for turn in turns:
            if answer in turn.lower():
                pairs.append({
                    "query": question,
                    "passage": turn[:512],
                    "label": 1,
                    "qtype": q.get("question_type", ""),
                })
            else:
                # Only keep hard negatives (with some keyword overlap)
                q_tokens = _tokenise(question)
                t_tokens = _tokenise(turn)
                overlap = len(q_tokens & t_tokens)
                if overlap >= 2:
                    pairs.append({
                        "query": question,
                        "passage": turn[:512],
                        "label": 0,
                        "qtype": q.get("question_type", ""),
                    })

    # Balance: 1:3 positive:negative ratio
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    target_neg = min(len(neg), len(pos) * 3)
    _RNG.shuffle(neg)
    balanced = pos + neg[:target_neg]
    _RNG.shuffle(balanced)
    return balanced


def extract_natural_date_examples(data: list[dict]) -> list[dict]:
    """C4: Extract natural vague date expressions from temporal questions.

    Uses haystack_dates to anchor the reference date.
    """
    vague_re = re.compile(
        r"\b(\d+\s+(?:days?|weeks?|months?|years?)\s+ago|"
        r"a\s+(?:few|couple\s+of)\s+(?:days?|weeks?|months?)\s+ago|"
        r"last\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
        r"earlier\s+this\s+(?:week|month|year)|"
        r"recently|the\s+other\s+day|not\s+long\s+ago|"
        r"some\s+time\s+(?:ago|back))\b",
        re.IGNORECASE,
    )
    date_fmt_re = re.compile(r"(\d{4})/(\d{2})/(\d{2})")

    samples: list[dict] = []

    for q in data:
        if q.get("question_type") != "temporal-reasoning":
            continue

        haystack_dates = q.get("haystack_dates", [])
        sessions = q.get("haystack_sessions", [])

        for si, sess in enumerate(sessions):
            # Get reference date from haystack_dates
            ref_date = None
            if si < len(haystack_dates):
                m = date_fmt_re.search(haystack_dates[si])
                if m:
                    ref_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

            if not ref_date:
                continue

            for turn in sess:
                content = turn.get("content", "")
                for match in vague_re.finditer(content):
                    samples.append({
                        "expr": match.group(0),
                        "ref_date": ref_date,
                        "source": "longmemeval_natural",
                    })

    return samples


def save_jsonl(data: list[dict], path: Path) -> None:
    """Write *data* as newline-delimited JSON to *path*, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(data)} samples")


def main() -> None:
    """Extract all training data from LongMemEval and save to JSONL."""
    print("Loading LongMemEval oracle...")
    data = load_oracle()
    print(f"  {len(data)} questions loaded")

    print("Generating C1 embedding triplets...")
    triplets = generate_embedding_triplets(data)
    save_jsonl(triplets, _OUT / "embedding_triplets.jsonl")

    print("Generating C2 cross-encoder pairs...")
    pairs = generate_cross_encoder_pairs(data)
    save_jsonl(pairs, _OUT / "crossencoder_pairs.jsonl")

    print("Extracting C4 natural date examples...")
    nat_dates = extract_natural_date_examples(data)
    save_jsonl(nat_dates, _OUT / "date_normalisation_natural.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
