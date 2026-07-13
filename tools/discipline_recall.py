# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — write-discipline ceiling on retrieval (W4 runner)

"""Measure the recall cost of the dominant write-discipline failure.

Runs GPT-free retrieval recall twice over the same haystack — once with the
canonical per-session timestamps, once with them stripped (``session_dates=None``,
the dominant fleet failure) — and prints the per-question-type recall drop via
:func:`discipline_impact.discipline_impact`. The drop is the write-discipline
ceiling: recall the retriever can never recover because the write omitted its
timestamp. CPU-bound I/O harness (omitted from coverage); the comparison logic is
tested in ``discipline_impact``.

Run from the repo root (NICED, no API):
    nice -n 19 python tools/discipline_recall.py --limit 50
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import cast

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from discipline_impact import discipline_impact, impact_payload, worst_hit  # noqa: E402
from retrieval_recall import (  # noqa: E402
    DEFAULT_DATASET,
    RECALL_NS,
    aggregate_recall,
    rank_ordered_sessions,
    recall_curve,
)


def _recall_run(
    data: list[dict[str, object]], *, strip_timestamps: bool
) -> list[dict[str, object]]:
    """Recall records over *data*; drop the session timestamps when stripping."""
    from arcane_retriever import ArcaneRetriever

    label = "DEGRADED (no timestamps)" if strip_timestamps else "CANONICAL (timestamps)"
    print(f"\n=== {label} ===", flush=True)
    records: list[dict[str, object]] = []
    t0 = time.monotonic()
    for i, item in enumerate(data):
        sessions = cast("list[list[dict[str, str]]]", item["haystack_sessions"])
        session_ids = cast("list[str]", item["haystack_session_ids"])
        sid_to_idx = {sid: j for j, sid in enumerate(session_ids)}
        answer_ids = cast("list[str]", item.get("answer_session_ids", []))
        gold = [sid_to_idx.get(s) for s in answer_ids]
        dates = None if strip_timestamps else cast("list[str] | None", item.get("haystack_dates"))
        ar = ArcaneRetriever(sessions, session_dates=dates)
        results = ar.retrieve(
            str(item["question"]), str(item["question_type"]), top_k=50, max_iterations=2
        )
        ranked = rank_ordered_sessions(results)
        records.append(
            {
                "qtype": item["question_type"],
                "recall": recall_curve(ranked, gold),
                "candidates": len(ranked),
                "gold_n": len([g for g in gold if g is not None]),
            }
        )
        if (i + 1) % 25 == 0 or i == len(data) - 1:
            el = time.monotonic() - t0
            print(f"  [{i + 1}/{len(data)}] {el:.0f}s ({el / (i + 1) * 1000:.0f}ms/q)", flush=True)
    return records


def main(argv: list[str]) -> int:
    """Run the canonical-vs-degraded recall comparison and print the ceiling."""
    dataset = DEFAULT_DATASET
    limit: int | None = None
    per_type: int | None = None
    json_out: Path | None = None
    args = list(argv)
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--per-type" in args:
        per_type = int(args[args.index("--per-type") + 1])
    if "--dataset" in args:
        dataset = Path(args[args.index("--dataset") + 1])
    if "--json-out" in args:
        json_out = Path(args[args.index("--json-out") + 1])

    with open(dataset) as fh:
        data = json.load(fh)
    if per_type:
        # Sample evenly across question types — the dataset is grouped by type, so
        # a plain head would only cover the first type (e.g. single-session-user,
        # where timestamps are irrelevant). Timestamp sensitivity lives in the
        # temporal-reasoning / multi-session / knowledge-update types.
        seen: dict[str, int] = {}
        sampled: list[dict[str, object]] = []
        for item in data:
            qt = str(item["question_type"])
            if seen.get(qt, 0) < per_type:
                sampled.append(item)
                seen[qt] = seen.get(qt, 0) + 1
        data = sampled
    elif limit:
        data = data[:limit]
    print(f"questions: {len(data)} from {Path(dataset).name}", flush=True)

    canonical = aggregate_recall(_recall_run(data, strip_timestamps=False))
    degraded = aggregate_recall(_recall_run(data, strip_timestamps=True))

    impacts = discipline_impact(canonical, degraded, ns=RECALL_NS)
    print(f"\n{'qtype':<26}{'canon@10':>12}{'degr@10':>12}{'drop@10':>12}")
    for imp in impacts:
        print(
            f"{imp.qtype:<26}{imp.canonical[10]:>11.1%}{imp.degraded[10]:>11.1%}{imp.delta_at(10):>11.1%}"
        )
    worst = worst_hit(impacts, n=10)
    if worst is not None:
        print(
            f"\nworst hit by missing timestamps: {worst.qtype} (-{worst.delta_at(10):.1%} recall@10)"
        )
    if json_out is not None:
        payload = impact_payload(
            impacts,
            metadata={
                "dataset": Path(dataset).name,
                "questions": len(data),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "retrieval": "GPT-free ArcaneRetriever, canonical vs stripped timestamps",
                "ce_rerank_sync": os.environ.get("REMANENTIA_ARCANE_CE_SYNC", "") == "1",
                "ce_rerank_disabled": os.environ.get("REMANENTIA_ARCANE_CE_DISABLE", "") == "1",
            },
        )
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote discipline-ceiling artefact -> {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
