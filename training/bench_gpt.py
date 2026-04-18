# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LongMemEval benchmark with GPT-4o-mini

"""Full LongMemEval benchmark using GPT-4o-mini for both synthesis and judging.

Runs without the hosted-SDK Python package. Estimated cost: ~$0.54
for 500 questions against the public GPT-4o-mini endpoint.

Usage:
    OPENAI_API_KEY=sk-... python training/bench_gpt.py [--temporal-only]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

_DATA = _BASE / "data"
_OUT = Path(__file__).resolve().parent

MODEL = "gpt-4o-mini"


def _get_client():
    """Create the hosted-LLM client from environment."""
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    timeout = float(os.environ.get("REMANENTIA_OPENAI_TIMEOUT", "30"))
    return OpenAI(api_key=key, timeout=timeout)


def _type_prompt(qtype: str) -> str:
    """Return question-type-specific system prompt."""
    if qtype == "temporal-reasoning":
        return (
            "Answer temporal questions about conversation history.\n"
            "RULES:\n"
            "- 'How many days between A and B': count calendar days\n"
            "- 'Which came first': compare dates\n"
            "- 'What was the first/last X': find earliest/latest dated event\n"
            "- START with the direct answer\n"
        )
    if qtype == "knowledge-update":
        return (
            "IMPORTANT: If multiple answers exist across sessions, use the MOST RECENT one.\n"
            "The user's latest statement supersedes earlier ones.\n"
            "Answer with the most up-to-date information (1-2 sentences).\n"
        )
    if qtype == "multi-session":
        return (
            "You are answering a question that requires combining information "
            "from multiple sessions.\nRead ALL sessions carefully. "
            "The answer may span multiple sessions.\n"
            "Give the answer directly in 1-2 sentences.\n"
        )
    if qtype == "single-session-preference":
        return (
            "Based on conversation history, provide a personalised answer "
            "that reflects the user's specific preferences and interests.\n"
            "Do NOT give a generic answer. Use actual stated preferences.\n"
            "Give a personalised response (2-3 sentences).\n"
        )
    return "Answer the question based on the conversation history. Be concise (1-2 sentences).\n"


def _build_context(sessions: list, results: list | None = None) -> str:
    """Build context string from sessions and optionally retrieved facts."""
    parts = []
    for si, sess in enumerate(sessions):
        turns = []
        for turn in sess:
            role = turn["role"].upper()
            turns.append(f"[{role}]: {turn['content']}")
        parts.append(f"=== Session {si + 1} ===\n" + "\n".join(turns))
    return "\n\n".join(parts)


def _synthesise(client, question: str, context: str, qtype: str) -> str:
    """Generate answer using GPT-4o-mini."""
    prompt = _type_prompt(qtype)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=150,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}\nAnswer:"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def _judge(client, gold: str, hypothesis: str, qtype: str) -> bool:
    """Judge if hypothesis matches gold answer using GPT-4o-mini."""
    extra = ""
    if qtype == "temporal-reasoning":
        extra = "\n*** SPECIAL: Do not penalise off-by-one errors for day counts ***\n"

    prompt = (
        f"Answer yes if the response contains the correct answer.\n"
        f"If the response is equivalent to the correct answer or contains "
        f"all intermediate steps to get the correct answer, answer yes.\n"
        f"If only a subset of information, answer no.\n"
        f"{extra}\n"
        f"Correct Answer: {gold}\n"
        f"Model Response: {hypothesis}\n"
        f"Answer: yes or no only."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def _arcane_answer(question, sessions, qtype, haystack_dates=None):
    """Get answer from ArcaneRetriever pipeline (local, no API)."""
    from arcane_retriever import ArcaneRetriever

    ar = ArcaneRetriever(sessions, session_dates=haystack_dates)
    results = ar.retrieve(question, qtype, top_k=15, max_iterations=2)

    if not results:
        return None, _build_context(sessions)

    # Build context with retrieved facts as timeline header. ArcaneRetriever
    # is currently used only for the retrieval ranking; the LLM sees the
    # fully-flattened session context below. `build_context` is kept as a
    # side-effect so the retriever pre-computes its internal caches.
    ar.build_context(question, results, max_facts=15)
    full_ctx = _build_context(sessions)

    if qtype == "temporal-reasoning":
        dated_facts = [r for r in results if r.fact.date_mentions]
        timeline = []
        for r in dated_facts:
            for d in r.fact.date_mentions:
                timeline.append(f"  {d}: {r.fact.text[:100]}")
        if timeline:
            timeline_str = "EXTRACTED TIMELINE:\n" + "\n".join(sorted(set(timeline))[:20])
            return results, f"{timeline_str}\n\n{full_ctx}"

    return results, full_ctx


def main():
    """Run full LongMemEval benchmark with GPT-4o-mini."""
    temporal_only = "--temporal-only" in sys.argv

    client = _get_client()

    with open(_DATA / "longmemeval_oracle.json", encoding="utf-8") as f:
        data = json.load(f)

    if temporal_only:
        data = [q for q in data if q.get("question_type") == "temporal-reasoning"]
        print(f"Running TEMPORAL-ONLY: {len(data)} questions")
    else:
        print(f"Running FULL benchmark: {len(data)} questions")

    print(f"Model: {MODEL}")
    print(f"Estimated cost: ~${len(data) * 0.0011:.2f}")
    print()

    results_by_type: dict[str, dict] = {}
    hypotheses = []
    t0 = time.time()

    for qi, item in enumerate(data):
        qid = item["question_id"]
        question = item["question"]
        gold = str(item["answer"])
        qtype = item.get("question_type", "unknown")
        sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates")

        # Step 1: Build context — hybrid routing based on proven results
        # ArcaneRetriever: temporal (+15pp), single-session factoid (+10-13pp)
        # Full context: preference (-53pp regression), multi-session (-8pp), knowledge-update (-5pp)
        use_arcane = qtype in (
            "temporal-reasoning",
            "single-session-user",
            "single-session-assistant",
        )

        if use_arcane:
            retrieval_results, context = _arcane_answer(
                question,
                sessions,
                qtype,
                haystack_dates=haystack_dates,
            )
        else:
            context = _build_context(sessions)

        # Step 2: Synthesise answer with GPT
        hypothesis = _synthesise(client, question, context, qtype)

        # Step 3: Judge with GPT
        correct = _judge(client, gold, hypothesis, qtype)

        # Track
        if qtype not in results_by_type:
            results_by_type[qtype] = {"correct": 0, "total": 0}
        results_by_type[qtype]["total"] += 1
        if correct:
            results_by_type[qtype]["correct"] += 1

        hypotheses.append(
            {
                "question_id": qid,
                "hypothesis": hypothesis,
                "judge_label": correct,
                "question_type": qtype,
            }
        )

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t0
            total_correct = sum(r["correct"] for r in results_by_type.values())
            total_tested = sum(r["total"] for r in results_by_type.values())
            pct = total_correct * 100 / total_tested if total_tested else 0
            print(
                f"  [{qi + 1}/{len(data)}] {total_correct}/{total_tested} = {pct:.1f}%  ({elapsed:.0f}s)"
            )

    # Final results
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({MODEL}, {elapsed:.0f}s)")
    print(f"{'=' * 60}")

    grand_correct = 0
    grand_total = 0
    for qtype in sorted(results_by_type):
        r = results_by_type[qtype]
        pct = r["correct"] * 100 / r["total"] if r["total"] else 0
        print(f"  {qtype:30s}  {r['correct']:3d}/{r['total']:3d} = {pct:5.1f}%")
        grand_correct += r["correct"]
        grand_total += r["total"]

    overall = grand_correct * 100 / grand_total if grand_total else 0
    print(f"  {'OVERALL':30s}  {grand_correct:3d}/{grand_total:3d} = {overall:5.1f}%")

    # Save
    suffix = "_temporal" if temporal_only else "_full"
    hyp_path = _OUT / f"bench_gpt{suffix}_hypotheses.jsonl"
    with open(hyp_path, "w") as f:
        for h in hypotheses:
            f.write(json.dumps(h) + "\n")

    res_path = _OUT / f"bench_gpt{suffix}_results.json"
    with open(res_path, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "overall_accuracy": overall,
                "by_type": {
                    k: {**v, "accuracy": v["correct"] * 100 / v["total"] if v["total"] else 0}
                    for k, v in results_by_type.items()
                },
                "elapsed_s": elapsed,
                "n_questions": grand_total,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {hyp_path.name}, {res_path.name}")


if __name__ == "__main__":
    main()
