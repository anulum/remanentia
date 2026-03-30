# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local temporal evaluation — no API credits required.

Measures retrieval quality on LongMemEval temporal-reasoning questions
using four metrics that do not need an LLM judge:

1. Retrieval Recall@K  — is the gold answer in top-K retrieved facts?
2. Direct Match Rate   — does any retrieved fact contain the answer?
3. Date Coverage       — do top-K results include dated facts?
4. TReMu Hit Rate      — does temporal_code_execute produce the right answer?

Usage:
    python training/eval_local.py [--with-dates] [--without-dates]

    --with-dates:    evaluate with session_dates (C4 pipeline wired)
    --without-dates: evaluate without session_dates (baseline)
    (default: run both and compare)
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_DATA = _BASE / "data"

sys.path.insert(0, str(_BASE))


def _parse_haystack_date(raw: str) -> str:
    """Extract ISO-ish date from haystack_dates format '2023/04/10 (Mon) 17:50'."""
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", raw)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def _token_overlap(text_a: str, text_b: str) -> float:
    """Token-level Jaccard-like overlap between two strings."""
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def run_eval(use_session_dates: bool) -> dict:
    """Run local evaluation on 133 temporal-reasoning questions.

    Returns dict with metric summaries.
    """
    from fact_decomposer import FactIndex, decompose_sessions
    from temporal_graph import TemporalEvent, temporal_code_execute

    with open(_DATA / "longmemeval_oracle.json", encoding="utf-8") as f:
        data = json.load(f)

    temporal = [q for q in data if q.get("question_type") == "temporal-reasoning"]
    print(
        f"\nEvaluating {len(temporal)} temporal questions "
        f"({'WITH' if use_session_dates else 'WITHOUT'} session_dates)"
    )

    recall_at_5 = 0
    recall_at_10 = 0
    direct_match = 0
    date_coverage = 0
    tremu_hits = 0
    tremu_attempts = 0
    total = len(temporal)
    t0 = time.time()

    for qi, q in enumerate(temporal):
        question = q["question"]
        answer = str(q["answer"]).lower().strip()
        sessions = q.get("haystack_sessions", [])

        # Parse session dates
        session_dates = None
        if use_session_dates:
            raw_dates = q.get("haystack_dates", [])
            session_dates = [_parse_haystack_date(d) for d in raw_dates]

        # Decompose sessions into atomic facts
        facts = decompose_sessions(
            sessions,
            default_year=2023,
            session_dates=session_dates,
        )
        if not facts:
            continue

        idx = FactIndex(facts)
        hits = idx.temporal_query(question, top_k=10)

        # Metric 1: Retrieval Recall@K
        found_at_5 = False
        found_at_10 = False
        for rank, (fact, score) in enumerate(hits):
            text_lower = fact.text.lower()
            if answer in text_lower or _token_overlap(answer, text_lower) > 0.5:
                if rank < 5:
                    found_at_5 = True
                found_at_10 = True
                break
        if found_at_5:
            recall_at_5 += 1
        if found_at_10:
            recall_at_10 += 1

        # Metric 2: Direct Match (any retrieved fact contains answer)
        if found_at_10:
            direct_match += 1

        # Metric 3: Date Coverage (top-10 has dated facts)
        dated_in_top10 = sum(1 for f, _ in hits if f.date_mentions)
        if dated_in_top10 >= 1:
            date_coverage += 1

        # Metric 4: TReMu (temporal code execution) for arithmetic questions
        q_lower = question.lower()
        if any(
            w in q_lower
            for w in (
                "how many days",
                "how long",
                "before or after",
                "most recent",
                "first",
                "earliest",
                "latest",
            )
        ):
            tremu_attempts += 1
            events = []
            for fact, _ in hits[:10]:
                for d in fact.date_mentions:
                    events.append(
                        TemporalEvent(
                            date=d,
                            text=fact.text[:200],
                            source="eval",
                        )
                    )
            if events:
                code_answer = temporal_code_execute(question, events)
                if code_answer and answer in code_answer.lower():
                    tremu_hits += 1

        if (qi + 1) % 25 == 0:
            print(f"  [{qi + 1}/{total}] recall@10={recall_at_10}/{qi + 1}")

    elapsed = time.time() - t0
    results = {
        "mode": "with_session_dates" if use_session_dates else "baseline",
        "total": total,
        "recall_at_5": recall_at_5,
        "recall_at_5_pct": round(recall_at_5 * 100 / total, 1),
        "recall_at_10": recall_at_10,
        "recall_at_10_pct": round(recall_at_10 * 100 / total, 1),
        "direct_match": direct_match,
        "direct_match_pct": round(direct_match * 100 / total, 1),
        "date_coverage": date_coverage,
        "date_coverage_pct": round(date_coverage * 100 / total, 1),
        "tremu_hits": tremu_hits,
        "tremu_attempts": tremu_attempts,
        "tremu_pct": round(tremu_hits * 100 / max(tremu_attempts, 1), 1),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n  Recall@5:       {results['recall_at_5']}/{total} = {results['recall_at_5_pct']}%")
    print(f"  Recall@10:      {results['recall_at_10']}/{total} = {results['recall_at_10_pct']}%")
    print(f"  Direct match:   {results['direct_match']}/{total} = {results['direct_match_pct']}%")
    print(f"  Date coverage:  {results['date_coverage']}/{total} = {results['date_coverage_pct']}%")
    print(
        f"  TReMu hits:     {results['tremu_hits']}/{results['tremu_attempts']} = {results['tremu_pct']}%"
    )
    print(f"  Time:           {results['elapsed_s']}s")

    return results


def main() -> None:
    """Run baseline vs wired-pipeline comparison."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "--compare"

    if mode == "--without-dates":
        results = run_eval(use_session_dates=False)
    elif mode == "--with-dates":
        results = run_eval(use_session_dates=True)
    else:
        print("=" * 60)
        print("LOCAL TEMPORAL EVALUATION (no API credits needed)")
        print("=" * 60)

        baseline = run_eval(use_session_dates=False)
        wired = run_eval(use_session_dates=True)

        print(f"\n{'=' * 60}")
        print("COMPARISON: baseline vs wired pipeline")
        print(f"{'=' * 60}")
        for key in (
            "recall_at_5_pct",
            "recall_at_10_pct",
            "direct_match_pct",
            "date_coverage_pct",
            "tremu_pct",
        ):
            label = key.replace("_pct", "").replace("_", " ").title()
            b = baseline[key]
            w = wired[key]
            delta = w - b
            arrow = "+" if delta > 0 else ""
            print(f"  {label:20s}  {b:6.1f}% → {w:6.1f}%  ({arrow}{delta:.1f}pp)")

        # Save results
        out_path = _BASE / "training" / "eval_local_results.json"
        with open(out_path, "w") as f:
            json.dump({"baseline": baseline, "wired": wired}, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
