# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ablation evaluation harness for temporal training components.

Runs the LongMemEval benchmark with different combinations of trained models
to measure the contribution of each component.

Usage:
    python training/eval_temporal.py [--config N]

Configs:
    0 = Baseline (no fine-tuned models)
    1 = +C4 only (date normaliser)
    2 = +C4+C1 (+ embedding)
    3 = +C4+C1+C2 (+ cross-encoder)
    4 = All 5 components
    5 = Run all configs sequentially
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
_MODELS = _BASE / "models"

# Component model directories
_COMPONENTS = {
    "C1": _MODELS / "temporal-embed-v1",
    "C2": _MODELS / "temporal-ce-v1",
    "C3": _MODELS / "temporal-relation-v1",
    "C4": _MODELS / "date-normalizer-v1",
    "C5": _MODELS / "fact-validity-v1",
}

CONFIGS = {
    0: {"name": "Baseline", "enabled": set()},
    1: {"name": "+C4 (date normaliser)", "enabled": {"C4"}},
    2: {"name": "+C4+C1 (+ embedding)", "enabled": {"C4", "C1"}},
    3: {"name": "+C4+C1+C2 (+ cross-encoder)", "enabled": {"C4", "C1", "C2"}},
    4: {"name": "All components", "enabled": {"C1", "C2", "C3", "C4", "C5"}},
}


def _toggle_models(enabled: set[str]) -> dict[str, bool]:
    """Enable/disable model directories by renaming to .disabled suffix."""
    state = {}
    for comp, path in _COMPONENTS.items():
        disabled_path = path.parent / (path.name + ".disabled")
        if comp in enabled:
            # Enable: rename .disabled -> original
            if disabled_path.exists() and not path.exists():
                disabled_path.rename(path)
            state[comp] = path.exists()
        else:
            # Disable: rename original -> .disabled
            if path.exists():
                path.rename(disabled_path)
                state[comp] = True
            else:
                state[comp] = False
    return state


def _restore_all() -> None:
    """Restore all model directories to their original names."""
    for comp, path in _COMPONENTS.items():
        disabled_path = path.parent / (path.name + ".disabled")
        if disabled_path.exists() and not path.exists():
            disabled_path.rename(path)


def _run_evaluation(config_id: int) -> dict:
    """Run temporal retrieval evaluation for a given ablation config."""
    config = CONFIGS[config_id]
    print(f"\n{'=' * 60}")
    print(f"Config {config_id}: {config['name']}")
    print(f"Enabled: {config['enabled'] or 'none (baseline)'}")
    print(f"{'=' * 60}")

    # Toggle models
    _toggle_models(config["enabled"])

    # Force reimport of modules to pick up model changes
    for mod in [
        "date_normalizer",
        "temporal_relation",
        "fact_validity_model",
        "temporal_graph",
        "fact_decomposer",
    ]:
        if mod in sys.modules:
            # Reset lazy-loaded model state
            m = sys.modules[mod]
            for attr in ("_model", "_tokenizer", "_config"):
                if hasattr(m, attr):
                    setattr(m, attr, None)

    # Load dataset
    with open(_DATA / "longmemeval_oracle.json", encoding="utf-8") as f:
        data = json.load(f)

    temporal = [q for q in data if q.get("question_type") == "temporal-reasoning"]
    print(f"Evaluating {len(temporal)} temporal questions...")

    # Import evaluation components
    sys.path.insert(0, str(_BASE))
    from fact_decomposer import FactIndex, decompose_sessions

    results = {"correct": 0, "total": 0, "details": []}
    t0 = time.time()

    date_fmt_re = re.compile(r"(\d{4})/(\d{2})/(\d{2})")

    for qi, q in enumerate(temporal):
        question = q["question"]
        answer = str(q["answer"]).lower()
        sessions = q.get("haystack_sessions", [])
        haystack_dates = q.get("haystack_dates", [])

        # Set reference date from question_date
        ref_date = date.today()
        qd = q.get("question_date", "")
        m = date_fmt_re.search(qd)
        if m:
            try:
                ref_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        # Decompose sessions into atomic facts
        facts = decompose_sessions(sessions, default_year=ref_date.year)
        if not facts:
            results["total"] += 1
            results["details"].append(
                {
                    "id": q["question_id"],
                    "correct": False,
                    "reason": "no_facts",
                }
            )
            continue

        # Build fact index and retrieve
        fact_index = FactIndex(facts)
        hits = fact_index.temporal_query(question, top_k=10)

        # Check if any retrieved fact contains the answer
        hit = False
        for fact, score in hits:
            if answer in fact.text.lower():
                hit = True
                break
            # Token overlap check
            a_tokens = set(answer.split())
            f_tokens = set(fact.text.lower().split())
            if len(a_tokens) > 0 and len(a_tokens & f_tokens) / len(a_tokens) > 0.5:
                hit = True
                break

        results["total"] += 1
        if hit:
            results["correct"] += 1
        results["details"].append(
            {
                "id": q["question_id"],
                "correct": hit,
            }
        )

        if (qi + 1) % 25 == 0:
            pct = results["correct"] / results["total"] * 100
            print(f"  [{qi + 1}/{len(temporal)}] accuracy: {pct:.1f}%")

    elapsed = time.time() - t0
    accuracy = results["correct"] / max(results["total"], 1) * 100

    print(f"\nConfig {config_id} ({config['name']}):")
    print(f"  Temporal accuracy: {results['correct']}/{results['total']} = {accuracy:.1f}%")
    print(f"  Time: {elapsed:.1f}s")

    results["accuracy"] = accuracy
    results["elapsed"] = elapsed
    results["config"] = config["name"]

    return results


def main() -> None:
    """Run ablation study across all configs or a single specified config."""
    config_arg = (
        int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].startswith("-") is False else 5
    )

    if config_arg == 5:
        # Run all configs
        all_results = {}
        for cid in sorted(CONFIGS):
            if not CONFIGS[cid]["enabled"] or all(
                _COMPONENTS[c].exists()
                or (_COMPONENTS[c].parent / (_COMPONENTS[c].name + ".disabled")).exists()
                for c in CONFIGS[cid]["enabled"]
            ):
                all_results[cid] = _run_evaluation(cid)
            else:
                missing = [
                    c
                    for c in CONFIGS[cid]["enabled"]
                    if not _COMPONENTS[c].exists()
                    and not (_COMPONENTS[c].parent / (_COMPONENTS[c].name + ".disabled")).exists()
                ]
                print(f"\nSkipping config {cid} — missing models: {missing}")

        _restore_all()

        # Summary
        print(f"\n{'=' * 60}")
        print("ABLATION SUMMARY")
        print(f"{'=' * 60}")
        for cid, res in sorted(all_results.items()):
            print(f"  Config {cid} ({res['config']}): {res['accuracy']:.1f}%")

        # Save results
        out_path = _BASE / "training" / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
    else:
        _run_evaluation(config_arg)
        _restore_all()


if __name__ == "__main__":
    main()
