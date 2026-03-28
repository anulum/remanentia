# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LongMemEval Benchmark Runner

"""Run LongMemEval benchmark against Remanentia's retrieval pipeline.

Each question has a set of chat sessions (haystack). We:
1. Index the haystack sessions as documents
2. Search for the question
3. Extract/synthesise an answer
4. Output hypothesis file for GPT-judge evaluation

Usage:
    python bench_longmemeval.py                    # regex-only answers
    python bench_longmemeval.py --llm              # LLM synthesis
    python bench_longmemeval.py --limit 50         # first 50 questions
    python bench_longmemeval.py --evaluate          # run GPT-judge after
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_PATH = Path(__file__).parent / "data" / "longmemeval_oracle.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "longmemeval_hypotheses.jsonl"

_USE_LLM = "--llm" in sys.argv
_EVALUATE = "--evaluate" in sys.argv
_LIMIT = None
for i, arg in enumerate(sys.argv):
    if arg == "--limit" and i + 1 < len(sys.argv):
        _LIMIT = int(sys.argv[i + 1])


def _flatten_sessions(sessions: list[list[dict]]) -> str:
    """Flatten chat sessions into a single searchable text."""
    parts = []
    for sess_idx, session in enumerate(sessions):
        lines = []
        for turn in session:
            role = turn["role"].upper()
            content = turn["content"]
            lines.append(f"[{role}]: {content}")
        parts.append(f"--- Session {sess_idx + 1} ---\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _build_index_for_question(sessions: list[list[dict]]):
    """Build a MemoryIndex over the haystack sessions for one question."""
    from memory_index import MemoryIndex, Document, _tokenize, _token_counts, _classify_paragraph

    idx = MemoryIndex()
    idx.documents = []
    idx.paragraph_index = []
    idx.paragraph_tokens = []
    idx.paragraph_token_counts = []
    idx.paragraph_types = []
    idx._inverted_index = {}
    idx._df = {}
    idx.idf = {}
    idx._para_lengths = __import__("numpy").array([], dtype=__import__("numpy").float32)
    idx._avg_dl = 1.0
    idx._built = True
    idx._rust_bm25_dirty = True

    import numpy as np
    import math

    for sess_idx, session in enumerate(sessions):
        # Each turn becomes a paragraph
        for turn_idx, turn in enumerate(session):
            content = turn["content"]
            if len(content) < 20:
                continue

            role = turn["role"]
            doc_name = f"session_{sess_idx}_turn_{turn_idx}_{role}"
            doc = Document(
                name=doc_name,
                source=f"session_{sess_idx}",
                path=doc_name,
                paragraphs=[content],
                date="",
                doc_type="conversation",
            )
            doc_idx = len(idx.documents)
            idx.documents.append(doc)

            para_idx = len(idx.paragraph_tokens)
            idx.paragraph_index.append((doc_idx, 0))
            token_list = _tokenize(content)
            tokens = set(token_list)
            token_counts = _token_counts(token_list)
            idx.paragraph_tokens.append(tokens)
            idx.paragraph_token_counts.append(token_counts)
            p_type = _classify_paragraph(content, is_code=False)
            idx.paragraph_types.append(p_type)

            n_total = len(idx.paragraph_tokens)
            for t in tokens:
                idx._df[t] = idx._df.get(t, 0) + 1
                idx.idf[t] = math.log(1 + n_total / (1 + idx._df[t]))
                if t not in idx._inverted_index:
                    idx._inverted_index[t] = []
                idx._inverted_index[t].append(para_idx)

    # Compute para lengths
    idx._para_lengths = np.array(
        [len(t) for t in idx.paragraph_tokens], dtype=np.float32
    )
    idx._avg_dl = float(np.mean(idx._para_lengths)) if len(idx._para_lengths) > 0 else 1.0

    return idx


def _answer_from_retrieval(question: str, idx, use_llm: bool = False) -> str:
    """Search index and extract/synthesise answer."""
    results = idx.search(question, top_k=10)

    if not results:
        return "I don't have enough information to answer this question."

    # Gather full paragraphs from top results (not just 300-char snippets)
    paras = []
    for r in results[:5]:
        doc_idx = next(
            (di for di, d in enumerate(idx.documents) if d.name == r.name), None
        )
        if doc_idx is not None:
            full_text = idx.documents[doc_idx].paragraphs[r.paragraph_idx]
            paras.append(full_text[:1500])
        else:
            paras.append(r.snippet)

    # LLM synthesis first (much better for conversational Q&A)
    if use_llm:
        try:
            from answer_extractor import _get_client
            client = _get_client()
            if client:
                context = "\n\n---\n\n".join(
                    f"[Source {i+1}]: {p}" for i, p in enumerate(paras)
                )
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Based ONLY on the following conversation excerpts, answer the question. "
                            "Be concise (1-2 sentences). If the answer requires counting days, "
                            "calculate precisely from the dates mentioned. "
                            "If the information is not in the sources, say 'unknown'.\n\n"
                            f"{context}\n\nQuestion: {question}"
                        ),
                    }],
                )
                answer = response.content[0].text.strip()
                if answer.lower() not in ("unknown", "i don't know"):
                    return answer
        except Exception:
            pass

    # Regex extraction fallback
    if results[0].answer:
        return results[0].answer

    return results[0].snippet[:500]


def run_benchmark():
    """Run LongMemEval and output hypothesis file."""
    print(f"Loading LongMemEval from {DATA_PATH}...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if _LIMIT:
        data = data[:_LIMIT]

    print(f"Questions: {len(data)}")
    print(f"LLM mode: {_USE_LLM}")
    print()

    type_counts = Counter(d["question_type"] for d in data)
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    hypotheses = []
    t0 = time.monotonic()

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold = str(item["answer"])

        # Build per-question index over haystack sessions
        idx = _build_index_for_question(item["haystack_sessions"])

        # Get answer
        hypothesis = _answer_from_retrieval(question, idx, use_llm=_USE_LLM)

        hypotheses.append({
            "question_id": qid,
            "hypothesis": hypothesis,
        })

        type_total[qtype] += 1

        # Quick local check (not authoritative — GPT-judge is the real eval)
        gold_lower = gold.lower().strip()
        hyp_lower = hypothesis.lower().strip()
        local_match = (
            gold_lower in hyp_lower
            or hyp_lower in gold_lower
            or _fuzzy_overlap(gold_lower, hyp_lower) > 0.6
        )
        if local_match:
            type_correct[qtype] += 1

        if (i + 1) % 25 == 0 or i == len(data) - 1:
            elapsed = time.monotonic() - t0
            total_correct = sum(type_correct.values())
            total_done = i + 1
            pct = total_correct / total_done * 100
            print(
                f"[{total_done}/{len(data)}] "
                f"local_match={total_correct}/{total_done} ({pct:.1f}%) "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.monotonic() - t0

    # Save hypotheses
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"\nHypotheses saved to {OUTPUT_PATH}")

    # Print local results (indicative, not authoritative)
    print(f"\n{'='*60}")
    print(f"LongMemEval Local Results (fuzzy match, NOT GPT-judge)")
    print(f"{'='*60}")
    total_correct = sum(type_correct.values())
    total = sum(type_total.values())
    print(f"Overall: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print()
    for qtype in sorted(type_total.keys()):
        c = type_correct[qtype]
        t = type_total[qtype]
        print(f"  {qtype:30s}: {c:3d}/{t:3d} ({c/t*100:.1f}%)")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/len(data)*1000:.0f}ms/question)")
    print(f"\nNOTE: These are local fuzzy-match scores. Run with --evaluate")
    print(f"for authoritative GPT-judge scoring.")


def _fuzzy_overlap(a: str, b: str) -> float:
    """Token overlap ratio between two strings."""
    tokens_a = set(re.findall(r"\w{3,}", a))
    tokens_b = set(re.findall(r"\w{3,}", b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(len(tokens_a), len(tokens_b))


def run_evaluation():
    """Run GPT-judge evaluation on saved hypotheses."""
    try:
        import anthropic
    except ImportError:
        print("anthropic package required for evaluation. pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY required for GPT-judge evaluation")
        return

    client = anthropic.Anthropic(api_key=api_key)

    with open(DATA_PATH, encoding="utf-8") as f:
        references = json.load(f)
    qid_to_ref = {r["question_id"]: r for r in references}

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        hypotheses = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(hypotheses)} hypotheses with Claude as judge...")

    type_scores = defaultdict(list)
    results = []

    for i, hyp in enumerate(hypotheses):
        qid = hyp["question_id"]
        ref = qid_to_ref.get(qid)
        if not ref:
            continue

        qtype = ref["question_type"]
        question = ref["question"]
        gold = str(ref["answer"])
        response = hyp["hypothesis"]

        prompt = _judge_prompt(qtype, question, gold, response)

        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            judge_answer = msg.content[0].text.strip().lower()
            correct = "yes" in judge_answer
        except Exception as e:
            print(f"  Judge error on {qid}: {e}")
            correct = False

        type_scores[qtype].append(1 if correct else 0)
        results.append({**hyp, "judge_label": correct, "question_type": qtype})

        if (i + 1) % 50 == 0:
            total = sum(sum(v) for v in type_scores.values())
            done = sum(len(v) for v in type_scores.values())
            print(f"  [{done}/{len(hypotheses)}] correct={total}/{done} ({total/done*100:.1f}%)")

    # Final results
    print(f"\n{'='*60}")
    print(f"LongMemEval Results (Claude-judge)")
    print(f"{'='*60}")
    all_scores = [s for scores in type_scores.values() for s in scores]
    print(f"Overall: {sum(all_scores)}/{len(all_scores)} ({sum(all_scores)/len(all_scores)*100:.1f}%)")
    print()
    for qtype in sorted(type_scores.keys()):
        scores = type_scores[qtype]
        print(f"  {qtype:30s}: {sum(scores):3d}/{len(scores):3d} ({sum(scores)/len(scores)*100:.1f}%)")

    # Save results
    results_path = OUTPUT_PATH.with_suffix(".results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {results_path}")


def _judge_prompt(task: str, question: str, answer: str, response: str) -> str:
    """Generate judge prompt matching LongMemEval's evaluation protocol."""
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the "
            "required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal "
            "information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        return (
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )


if __name__ == "__main__":
    run_benchmark()
    if _EVALUATE:
        run_evaluation()
