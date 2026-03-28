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


def _answer_from_retrieval(
    question: str, idx, sessions: list, qtype: str, use_llm: bool = False
) -> str:
    """Search index and extract/synthesise answer with type-specific strategy."""
    results = idx.search(question, top_k=10)

    if not results:
        return "I don't have enough information to answer this question."

    if not use_llm:
        if results[0].answer:
            return results[0].answer
        return results[0].snippet[:500]

    # Build context based on question type
    context = _build_context(question, idx, results, sessions, qtype)

    # Get type-specific prompt
    prompt = _type_prompt(question, qtype, context)

    try:
        from answer_extractor import _get_client
        client = _get_client()
        if not client:
            return results[0].snippet[:500]

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()
        if answer.lower() not in ("unknown", "i don't know", "not mentioned"):
            return answer
    except Exception:
        pass

    if results[0].answer:
        return results[0].answer
    return results[0].snippet[:500]


def _build_context(question: str, idx, results, sessions: list, qtype: str) -> str:
    """Build LLM context adapted to question type."""

    if qtype in ("multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"):
        # Cross-session + temporal + knowledge-update: full session content
        parts = []
        for sess_idx, session in enumerate(sessions):
            turns = []
            for turn in session:
                role = turn["role"].upper()
                turns.append(f"[{role}]: {turn['content']}")
            parts.append(f"=== Session {sess_idx + 1} ===\n" + "\n".join(turns))
        context = "\n\n".join(parts)

        # For temporal: prepend extracted dates
        if qtype == "temporal-reasoning":
            dates_info = _extract_temporal_facts(sessions)
            if dates_info:
                context = f"EXTRACTED DATES AND EVENTS:\n{dates_info}\n\n{context}"

        return context

    # Single-session types: focused BM25 results (no noise from surrounding turns)
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

    return "\n\n---\n\n".join(paras)


def _extract_temporal_facts(sessions: list) -> str:
    """Extract dates, parse them, compute day differences between all pairs."""
    import re
    from datetime import datetime

    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    def _parse_date(s: str) -> datetime | None:
        s = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", s.strip())
        for fmt in (
            "%B %d, %Y", "%B %d %Y", "%B %d",
            "%b %d, %Y", "%b %d %Y", "%b %d",
            "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y",
            "%Y-%m-%d", "%Y/%m/%d",
        ):
            try:
                d = datetime.strptime(s, fmt)
                if d.year == 1900:
                    d = d.replace(year=2023)
                return d
            except ValueError:
                continue
        return None

    # Extract all dated events from user turns
    date_patterns = [
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
        r"(\d{1,2}/\d{1,2}/\d{2,4})",
    ]

    events: list[tuple[datetime, str]] = []
    for sess_idx, session in enumerate(sessions):
        for turn in session:
            text = turn["content"]
            for pattern in date_patterns:
                for m in re.finditer(pattern, text):
                    date_str = m.group(1) if m.lastindex else m.group()
                    parsed = _parse_date(date_str)
                    if not parsed:
                        continue
                    start = max(0, m.start() - 80)
                    end = min(len(text), m.end() + 80)
                    ctx = text[start:end].strip().replace("\n", " ")
                    events.append((parsed, f"{date_str}: \"{ctx}\""))

    if not events:
        return ""

    # Deduplicate by date
    seen_dates: dict[str, tuple[datetime, str]] = {}
    for dt, desc in events:
        key = dt.strftime("%Y-%m-%d")
        if key not in seen_dates:
            seen_dates[key] = (dt, desc)

    sorted_events = sorted(seen_dates.values(), key=lambda x: x[0])

    lines = ["TIMELINE (chronological):"]
    for i, (dt, desc) in enumerate(sorted_events):
        lines.append(f"  {i+1}. {dt.strftime('%B %d, %Y')} ({dt.strftime('%A')}) — {desc}")

    # Compute pairwise day differences for adjacent + notable pairs
    if len(sorted_events) >= 2:
        lines.append("\nDAY DIFFERENCES (computed):")
        for i in range(len(sorted_events)):
            for j in range(i + 1, min(i + 4, len(sorted_events))):
                dt_a, desc_a = sorted_events[i]
                dt_b, desc_b = sorted_events[j]
                days = (dt_b - dt_a).days
                lines.append(
                    f"  {dt_a.strftime('%b %d')} → {dt_b.strftime('%b %d')} = {days} days"
                )

    return "\n".join(lines)


def _type_prompt(question: str, qtype: str, context: str) -> str:
    """Generate type-specific LLM prompt for answer synthesis."""

    if qtype == "temporal-reasoning":
        return (
            "Answer this temporal question using the TIMELINE and DAY DIFFERENCES provided.\n\n"
            "The day differences have been PRE-COMPUTED for you — use them directly.\n"
            "For 'how many days' questions: find the two relevant events in the timeline "
            "and look up their day difference in the computed table.\n"
            "For 'which came first' questions: check the chronological order in the timeline.\n\n"
            "START with the direct answer.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    if qtype == "multi-session":
        return (
            "You are answering a question that requires combining information from multiple conversation sessions.\n\n"
            "Read ALL sessions carefully. The answer may span multiple sessions.\n\n"
            f"Conversation sessions:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give the answer directly in 1-2 sentences."
        )

    if qtype == "knowledge-update":
        return (
            "You are answering a question where information may have been updated over time.\n\n"
            "IMPORTANT: If multiple answers exist across sessions, use the MOST RECENT one.\n"
            "The user's latest statement supersedes earlier ones.\n\n"
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with the most up-to-date information. Be concise (1-2 sentences)."
        )

    if qtype == "single-session-preference":
        return (
            "The user is asking a general question. Based on their conversation history, "
            "provide a personalized answer that reflects their specific preferences, "
            "interests, and personal details mentioned in the conversation.\n\n"
            "Do NOT give a generic answer. Use the user's actual stated preferences.\n\n"
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give a personalized response (2-3 sentences) that reflects the user's specific situation."
        )

    # single-session-user, single-session-assistant
    return (
        "Based ONLY on the following conversation excerpts, answer the question.\n"
        "Be concise (1-2 sentences). If the information is not present, say 'unknown'.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )


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

        # Get answer (pass sessions + type for context strategy)
        hypothesis = _answer_from_retrieval(
            question, idx, item["haystack_sessions"], qtype, use_llm=_USE_LLM
        )

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
    """Run judge evaluation on saved hypotheses.

    Uses OpenAI gpt-4o-mini by default (matches official LongMemEval protocol).
    Falls back to Anthropic Claude if OPENAI_API_KEY not set.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    judge_backend = None
    judge_client = None

    if openai_key:
        try:
            from openai import OpenAI
            judge_client = OpenAI(api_key=openai_key)
            judge_backend = "openai"
            print("Judge: OpenAI gpt-4o-mini (official LongMemEval protocol)")
        except ImportError:
            pass

    if not judge_client and anthropic_key:
        try:
            import anthropic
            judge_client = anthropic.Anthropic(api_key=anthropic_key)
            judge_backend = "anthropic"
            print("Judge: Anthropic Claude Haiku")
        except ImportError:
            pass

    if not judge_client:
        print("No API key available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        return

    with open(DATA_PATH, encoding="utf-8") as f:
        references = json.load(f)
    qid_to_ref = {r["question_id"]: r for r in references}

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        hypotheses = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(hypotheses)} hypotheses...")

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
            if judge_backend == "openai":
                resp = judge_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=10,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                judge_answer = resp.choices[0].message.content.strip().lower()
            else:
                msg = judge_client.messages.create(
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
    judge_name = "GPT-4o-mini" if judge_backend == "openai" else "Claude-Haiku"
    print(f"\n{'='*60}")
    print(f"LongMemEval Results ({judge_name} judge)")
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
