# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — local-judge agreement runner (sovereign judge validation)

"""Re-judge a hosted-judge results file with a local model and measure agreement.

Reads a judged LongMemEval results JSONL (rows carrying the reference judge's
``judge_label``), rebuilds the EXACT judge prompt the reference pass used
(``bench_longmemeval._judge_prompt`` — same protocol, same yes/no parsing),
sends it to a local Ollama-compatible server, and folds the label pairs into
:func:`judge_agreement.agreement_stats`. The per-question pairs land in a
JSONL (``--pairs-out``) for qualitative disagreement review and the summary
in a committable JSON artefact (``--json-out``).

A local judge may replace the hosted one only on the evidence this run
produces — never by assumption. CPU/GPU-bound I/O harness (omitted from
coverage); the statistics live in the tested :mod:`judge_agreement`.

Run from the repo root (server must be up; judge prompts are short):
    python tools/local_judge_agreement.py \
        --results data/longmemeval_full_s_seed42.results.jsonl \
        --model gemma3:12b-ctx8k --url http://localhost:11434/v1 \
        --json-out benchmarks/judge_agreement_seed42.json \
        --pairs-out data/judge_agreement_seed42_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from judge_agreement import JudgePair, agreement_payload  # noqa: E402


def _load_references(dataset: Path) -> dict[str, dict[str, str]]:
    """Map question_id → {question_type, question, answer} from the dataset."""
    with open(dataset, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        str(item["question_id"]): {
            "question_type": str(item["question_type"]),
            "question": str(item["question"]),
            "answer": str(item["answer"]),
        }
        for item in data
    }


def main(argv: list[str] | None = None) -> int:
    """Run the local re-judge pass and write the agreement artefact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path, help="judged results JSONL")
    parser.add_argument("--dataset", type=Path, default=_REPO / "data" / "longmemeval_s.json")
    parser.add_argument("--model", default="gemma3:12b-ctx8k", help="local judge model tag")
    parser.add_argument("--url", default="http://localhost:11434/v1", help="local server URL")
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N rows")
    parser.add_argument("--json-out", type=Path, default=None, help="agreement artefact path")
    parser.add_argument("--pairs-out", type=Path, default=None, help="per-question pairs JSONL")
    args = parser.parse_args(argv)

    from bench_longmemeval import _judge_prompt  # heavy import deferred to runtime
    from llm_backend import LocalLLMBackend

    refs = _load_references(args.dataset)
    backend = LocalLLMBackend(base_url=args.url, model=args.model, timeout=120.0)

    rows: list[dict[str, object]] = []
    with open(args.results, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "judge_label" in row and str(row.get("question_id", "")) in refs:
                    rows.append(row)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"re-judging {len(rows)} rows with {args.model} @ {args.url}", flush=True)

    pairs: list[JudgePair] = []
    pair_rows: list[dict[str, object]] = []
    reference_judges: set[str] = set()
    t0 = time.monotonic()
    for i, row in enumerate(rows):
        qid = str(row["question_id"])
        ref = refs[qid]
        prompt = _judge_prompt(
            ref["question_type"], ref["question"], ref["answer"], str(row["hypothesis"])
        )
        raw = backend.complete(prompt, max_tokens=10)
        # Same decision rule as the reference pass: "yes" anywhere in the
        # lowered answer = correct; an unusable/empty answer is recorded as
        # unanswered, never coerced to a label.
        candidate: bool | None = ("yes" in raw.strip().lower()) if raw else None
        reference = bool(row["judge_label"])
        model = row.get("judge_model")
        if isinstance(model, str) and model:
            reference_judges.add(model)
        pairs.append((reference, candidate))
        pair_rows.append(
            {
                "question_id": qid,
                "question_type": ref["question_type"],
                "reference_label": reference,
                "candidate_label": candidate,
                "candidate_raw": (raw or "")[:80],
            }
        )
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            el = time.monotonic() - t0
            answered = sum(1 for _, c in pairs if c is not None)
            agree = sum(1 for r, c in pairs if c is not None and r == c)
            print(
                f"  [{i + 1}/{len(rows)}] agree={agree}/{answered} {el:.0f}s",
                flush=True,
            )

    payload = agreement_payload(
        pairs,
        metadata={
            "reference_judges": sorted(reference_judges),
            "candidate_judge": args.model,
            "candidate_url": args.url,
            "results_path": str(args.results),
            "dataset": args.dataset.name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(payload["stats"], indent=2))

    if args.pairs_out is not None:
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pairs_out, "w", encoding="utf-8") as fh:
            for pair_row in pair_rows:
                fh.write(json.dumps(pair_row, ensure_ascii=False) + "\n")
        print(f"wrote pairs -> {args.pairs_out}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote judge-agreement artefact -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
