# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — full-S retrieval-recall diagnostic

"""Measure how often ArcaneRetriever surfaces the gold answer sessions.

On the realistic LongMemEval-S haystack (~50 sessions/question, ~2 gold) the
question that matters is: does retrieval put the gold sessions in front of the
reader? This diagnostic answers it with **no LLM calls** — it runs the retriever
over every question and reports the gold-session recall as a function of how
many top-ranked sessions the reader would be shown (recall@N). Failures with
gold-not-recalled are retrieval misses; failures with gold-recalled are
synthesis. Run before the expensive end-to-end LLM benchmark.

Usage::

    OMP_NUM_THREADS=4 .venv/bin/python tools/retrieval_recall.py [--limit N] [--dataset PATH]
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from retrieved_context import gold_session_recall, rank_ordered_sessions  # noqa: E402

RECALL_NS = (3, 5, 10, 15, 20)
DEFAULT_DATASET = REPO / "data" / "longmemeval_s.json"


class RecallRecord(TypedDict):
    """Per-question recall outcome: qtype, recall@N map, candidate/gold counts."""

    qtype: str
    recall: dict[int, float]
    candidates: int
    gold_n: int


def recall_curve(
    rank_ordered: Sequence[int],
    gold_idxs: Sequence[int | None],
    ns: tuple[int, ...] = RECALL_NS,
) -> dict[int, float]:
    """Gold-session recall when the reader sees the top-N retrieved sessions."""
    return {n: gold_session_recall(rank_ordered[:n], gold_idxs) for n in ns}


def aggregate_recall(
    records: Sequence[RecallRecord], ns: tuple[int, ...] = RECALL_NS
) -> dict[str, dict[str, float]]:
    """Per-qtype (and overall) mean recall@N and full-recall rate.

    Each record is ``{"qtype", "recall": {n: float}, "candidates": int}``.
    Returns, per qtype and ``"overall"``: ``mean@N`` for each N plus
    ``full@<maxN>`` (fraction with recall 1.0 at the largest N) and
    ``mean_candidates``.
    """
    by_type: dict[str, list[RecallRecord]] = defaultdict(list)
    for r in records:
        by_type[r["qtype"]].append(r)
        by_type["overall"].append(r)

    max_n = max(ns)
    out: dict[str, dict[str, float]] = {}
    for qtype, recs in by_type.items():
        n_recs = len(recs)
        row: dict[str, float] = {"n": float(n_recs)}
        for n in ns:
            row[f"mean@{n}"] = sum(r["recall"][n] for r in recs) / n_recs
        row[f"full@{max_n}"] = sum(1 for r in recs if r["recall"][max_n] >= 1.0) / n_recs
        row["mean_candidates"] = sum(r["candidates"] for r in recs) / n_recs
        out[qtype] = row
    return out


def _wait_for_cross_encoder(timeout_s: float = 90.0) -> bool:
    """Trigger and block on the background cross-encoder load (reader parity)."""
    import os

    if os.getenv("REMANENTIA_ARCANE_CE_DISABLE") == "1":
        return False
    from arcane_retriever import ArcaneRetriever

    probe = ArcaneRetriever([[{"role": "user", "content": "warm up the model"}]])
    probe._load_ce()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not ArcaneRetriever._ce_loading and ArcaneRetriever._ce_model is not None:
            return bool(ArcaneRetriever._ce_model)
        time.sleep(0.5)
    return False


def run(dataset: Path, limit: int | None = None) -> list[RecallRecord]:
    """Run retrieval over *dataset* and return per-question recall records."""
    from arcane_retriever import ArcaneRetriever

    ce_ready = _wait_for_cross_encoder()
    print(f"cross-encoder rerank: {'ON' if ce_ready else 'OFF'}", flush=True)

    with open(dataset) as fh:
        data = json.load(fh)
    if limit:
        data = data[:limit]
    print(f"questions: {len(data)} from {dataset.name}", flush=True)

    records: list[RecallRecord] = []
    t0 = time.monotonic()
    for i, item in enumerate(data):
        sessions = item["haystack_sessions"]
        sid_to_idx = {sid: j for j, sid in enumerate(item["haystack_session_ids"])}
        gold = [sid_to_idx.get(s) for s in item.get("answer_session_ids", [])]

        # Retrieve well beyond max(RECALL_NS) facts so enough distinct sessions
        # surface to make recall@20 meaningful (20 facts span only ~10 sessions).
        ar = ArcaneRetriever(sessions, session_dates=item.get("haystack_dates"))
        results = ar.retrieve(item["question"], item["question_type"], top_k=50, max_iterations=2)
        ranked = rank_ordered_sessions(results)

        records.append(
            {
                "qtype": item["question_type"],
                "recall": recall_curve(ranked, gold),
                "candidates": len(ranked),
                "gold_n": len([g for g in gold if g is not None]),
            }
        )
        if (i + 1) % 50 == 0 or i == len(data) - 1:
            el = time.monotonic() - t0
            print(f"  [{i + 1}/{len(data)}] {el:.0f}s ({el / (i + 1) * 1000:.0f}ms/q)", flush=True)
    return records


def _print_table(agg: dict[str, dict[str, float]]) -> None:
    cols = [f"mean@{n}" for n in RECALL_NS] + [f"full@{max(RECALL_NS)}", "mean_candidates"]
    print(f"\n{'qtype':<26}{'n':>5}" + "".join(f"{c:>14}" for c in cols))
    for qtype in sorted(agg, key=lambda k: (k != "overall", k)):
        row = agg[qtype]
        cells = "".join(
            f"{row[c]:>13.1%}" if c.startswith(("mean@", "full@")) else f"{row[c]:>13.1f}"
            for c in cols
        )
        print(f"{qtype:<26}{int(row['n']):>5}{cells}")


def main(argv: list[str]) -> int:
    dataset = DEFAULT_DATASET
    limit: int | None = None
    for i, a in enumerate(argv):
        if a == "--dataset" and i + 1 < len(argv):
            dataset = Path(argv[i + 1])
        if a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
    records = run(dataset, limit)
    agg = aggregate_recall(records)
    _print_table(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
