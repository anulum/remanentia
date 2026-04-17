# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — ArcaneRetriever channel-ablation harness

"""Measure each ArcaneRetriever channel's retrieval contribution.

For a fixed sample of LongMemEval oracle items, run
:class:`ArcaneRetriever.retrieve` under five configurations:

- ``ALL``              — every channel active
- ``no_bm25``          — FAST channel off
- ``no_entity``        — WORKING channel off
- ``no_temporal``      — TEMPORAL channel off
- ``no_session``       — DEEP / cross-session channel off

For each configuration we compute **answer-session recall@K**: the
fraction of oracle ``answer_session_ids`` that appear among the
sessions covered by the retriever's top-K facts.

Recall@K is a purely retrieval-side metric; no LLM, no API cost, no
GPU required. The harness runs CPU-only and is seeded from
``REMANENTIA_SEED`` so each run is reproducible.

Usage:

    python ablation_channels.py --limit 100 --top-k 10 \\
        --out docs/benchmarks/CHANNEL_ABLATIONS.md

The output markdown is ready to commit to ``docs/benchmarks/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from arcane_retriever import ArcaneRetriever
from seed_utils import seed_everything, seed_from_env


# ─── Ablation configurations ──────────────────────────────────────────


ALL_CHANNELS: tuple[str, ...] = ("bm25", "entity", "temporal", "session")

CONFIGS: dict[str, tuple[str, ...]] = {
    "ALL": ALL_CHANNELS,
    "no_bm25": tuple(c for c in ALL_CHANNELS if c != "bm25"),
    "no_entity": tuple(c for c in ALL_CHANNELS if c != "entity"),
    "no_temporal": tuple(c for c in ALL_CHANNELS if c != "temporal"),
    "no_session": tuple(c for c in ALL_CHANNELS if c != "session"),
}


# ─── Core measurement ────────────────────────────────────────────────


@dataclass
class ItemResult:
    """Per-item recall for one configuration."""

    question_id: str
    qtype: str
    config: str
    recall: float  # 0.0 – 1.0


def _covered_sessions(
    retriever: ArcaneRetriever,
    question: str,
    qtype: str,
    top_k: int,
    allowed_channels: tuple[str, ...],
) -> set[int]:
    """Run retrieval under an ablated channel set; return covered session indices."""
    original_gate = retriever._gate

    def _ablated_gate(q: str, t: str) -> list[str]:
        default = original_gate(q, t)
        return [c for c in default if c in allowed_channels] or list(allowed_channels)

    retriever._gate = _ablated_gate  # type: ignore[method-assign]
    try:
        hits = retriever.retrieve(question, qtype=qtype, top_k=top_k)
    finally:
        retriever._gate = original_gate  # type: ignore[method-assign]

    return {h.fact.session_idx for h in hits}


def run_item(
    item: dict,
    top_k: int,
    reference_date: str | None = None,
) -> list[ItemResult]:
    """Run every configuration on a single oracle item."""
    sessions = item["haystack_sessions"]
    session_dates = item.get("haystack_dates")
    session_ids = item["haystack_session_ids"]
    answer_sids = set(item["answer_session_ids"])
    qtype = item["question_type"]

    retriever = ArcaneRetriever(
        sessions,
        session_dates=session_dates,
        reference_date=reference_date or item.get("question_date"),
    )

    results: list[ItemResult] = []
    for config_name, channels in CONFIGS.items():
        covered_idx = _covered_sessions(retriever, item["question"], qtype, top_k, channels)
        covered_sids = {session_ids[i] for i in covered_idx if i < len(session_ids)}
        hit = covered_sids & answer_sids
        recall = len(hit) / max(1, len(answer_sids))
        results.append(
            ItemResult(
                question_id=item["question_id"],
                qtype=qtype,
                config=config_name,
                recall=recall,
            )
        )
    return results


# ─── Aggregation + report ────────────────────────────────────────────


def aggregate(rows: list[ItemResult]) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{config: {qtype: {mean, n}}}`` plus an ``_overall`` qtype."""
    from collections import defaultdict

    by_config_qtype: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_config_qtype[r.config][r.qtype].append(r.recall)
        by_config_qtype[r.config]["_overall"].append(r.recall)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for config, qtypes in by_config_qtype.items():
        out[config] = {}
        for qtype, vals in qtypes.items():
            out[config][qtype] = {
                "mean": statistics.mean(vals) if vals else 0.0,
                "n": len(vals),
            }
    return out


def format_report(
    agg: dict[str, dict[str, dict[str, float]]],
    *,
    top_k: int,
    seed: int,
    limit: int,
    runtime_s: float,
) -> str:
    """Build the markdown report body."""
    qtypes = sorted({q for conf in agg.values() for q in conf if q != "_overall"}) + ["_overall"]
    configs = ["ALL", "no_bm25", "no_entity", "no_temporal", "no_session"]

    lines: list[str] = []
    lines.append("# ArcaneRetriever — channel ablations (answer-session recall)")
    lines.append("")
    lines.append(
        f"- **Sample**: first {limit} LongMemEval oracle items (top_k={top_k}, seed={seed})."
    )
    lines.append(
        "- **Metric**: fraction of ``answer_session_ids`` appearing among "
        "the sessions covered by the retriever's top-K facts."
    )
    lines.append(
        "- **Config**: each ablation removes exactly one channel; ``ALL`` is the "
        "baseline with every channel active."
    )
    lines.append(f"- **Runtime**: {runtime_s:.1f} s.")
    lines.append("")
    lines.append("## Mean recall per qtype")
    lines.append("")
    header = "| qtype | " + " | ".join(configs) + " |"
    sep = "|---|" + "|".join(["---"] * len(configs)) + "|"
    lines.append(header)
    lines.append(sep)
    for qtype in qtypes:
        cells = []
        baseline = agg.get("ALL", {}).get(qtype, {}).get("mean", 0.0)
        for config in configs:
            m = agg.get(config, {}).get(qtype, {}).get("mean", 0.0)
            delta = m - baseline
            if config == "ALL":
                cells.append(f"{m:.3f}")
            else:
                sign = "+" if delta >= 0 else ""
                cells.append(f"{m:.3f} ({sign}{delta:.3f})")
        label = qtype if qtype != "_overall" else "**overall**"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append(
        "Each ablation cell shows ``mean (delta_vs_ALL)``. A large "
        "negative delta means the missing channel was pulling its weight "
        "for that qtype; a near-zero delta means the other three channels "
        "compensate and the removed channel is redundant on that sample."
    )
    lines.append("")
    lines.append(
        "This is a **retrieval-side** measurement. A channel that does not "
        "move recall might still help the downstream LLM by contributing "
        "different facts to the same RRF-fused list; a separate end-to-end "
        "ablation under ``bench_longmemeval.py --arcane`` would answer that."
    )
    lines.append("")
    lines.append("## Findings worth flagging")
    lines.append("")
    lines.append(_derive_findings(agg))
    return "\n".join(lines) + "\n"


def _derive_findings(agg: dict[str, dict[str, dict[str, float]]]) -> str:
    """Scan the aggregate table for deltas worth calling out in prose."""
    lines: list[str] = []

    # Detect configurations where removing a channel *increased* recall
    # on a qtype — that channel is actively hurting on that qtype.
    for config in ("no_bm25", "no_entity", "no_temporal", "no_session"):
        removed = config.removeprefix("no_")
        for qtype, ablated in agg.get(config, {}).items():
            if qtype == "_overall":
                continue
            baseline = agg.get("ALL", {}).get(qtype, {}).get("mean", 0.0)
            delta = ablated["mean"] - baseline
            if delta > 0.015:
                lines.append(
                    f"- **`{removed}` hurts `{qtype}`**: recall rises "
                    f"{baseline:.3f} → {ablated['mean']:.3f} "
                    f"(+{delta:.3f}) when the channel is removed. "
                    "Expected direction is the opposite; investigate the "
                    "channel's ranking logic."
                )

    # Detect channels that carry real weight (>= 0.01 drop when ablated).
    for config in ("no_bm25", "no_entity", "no_temporal", "no_session"):
        removed = config.removeprefix("no_")
        for qtype, ablated in agg.get(config, {}).items():
            if qtype == "_overall":
                continue
            baseline = agg.get("ALL", {}).get(qtype, {}).get("mean", 0.0)
            delta = ablated["mean"] - baseline
            if delta < -0.01:
                lines.append(
                    f"- **`{removed}` helps `{qtype}`**: removing it drops "
                    f"recall {baseline:.3f} → {ablated['mean']:.3f} "
                    f"({delta:+.3f})."
                )

    if not lines:
        lines.append(
            "- No individual channel changes recall by more than ±0.01 on "
            "any qtype in this sample. Either every channel is pulling "
            "roughly equal weight, or the retriever is saturating the "
            "recall ceiling for this top_k."
        )
    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ArcaneRetriever channel ablations on LongMemEval.")
    p.add_argument(
        "--oracle",
        type=Path,
        default=Path("data/longmemeval_oracle.json"),
        help="Path to the LongMemEval oracle JSON.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of oracle items to process (default: 50).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieve this many facts per question (default: 10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/benchmarks/CHANNEL_ABLATIONS.md"),
        help="Markdown report output path.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: $REMANENTIA_SEED or 42).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional raw per-item results as JSONL.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N items.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seed = seed_everything(args.seed if args.seed is not None else seed_from_env())
    print(f"Seed: {seed}")

    if not args.oracle.exists():
        print(f"Oracle not found: {args.oracle}", file=sys.stderr)
        return 1

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    items = oracle[: args.limit]
    print(f"Running ablation on {len(items)} items, top_k={args.top_k}")

    t0 = time.perf_counter()
    all_rows: list[ItemResult] = []
    for i, item in enumerate(items, start=1):
        all_rows.extend(run_item(item, top_k=args.top_k))
        if i % args.progress_every == 0:  # pragma: no cover
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(items) - i)
            print(f"  [{i}/{len(items)}] elapsed {elapsed:.1f}s  eta {eta:.0f}s")

    runtime_s = time.perf_counter() - t0
    agg = aggregate(all_rows)

    # Write markdown report.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        format_report(agg, top_k=args.top_k, seed=seed, limit=len(items), runtime_s=runtime_s),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")

    # Optional JSONL dump.
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            for r in all_rows:
                fh.write(json.dumps(r.__dict__) + "\n")
        print(f"Wrote {args.json_out}")

    # Print the table to stdout as well.
    print()
    print(format_report(agg, top_k=args.top_k, seed=seed, limit=len(items), runtime_s=runtime_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
