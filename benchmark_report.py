# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark evidence report generator

"""Build auditable benchmark evidence reports from persisted result files.

The report layer is deliberately post-run: it consumes benchmark artefacts that
already exist on disk and computes score, prompt-hash, token, and latency
summaries without invoking readers, judges, or retrieval backends.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, cast

from benchmark_evidence import estimate_tokens, prompt_sha256

JsonDict = dict[str, object]

SCHEMA_VERSION = 1


def _pct(correct: int, total: int) -> float:
    """Return a one-decimal percentage for ``correct / total``."""
    return round(100.0 * correct / total, 1) if total else 0.0


def _score(correct: int, total: int) -> JsonDict:
    """Build a score dictionary shared by overall and category summaries."""
    return {"correct": correct, "total": total, "accuracy": _pct(correct, total)}


def _number(value: object) -> float | None:
    """Coerce numeric JSON values to ``float`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_from_value(value: object) -> int | None:
    """Coerce integral JSON values to ``int`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _field_int(row: Mapping[str, object], key: str) -> int | None:
    """Return integer field ``key`` from a mapping, if present and integral."""
    return _int_from_value(row.get(key))


def _load_jsonl(path: Path) -> list[JsonDict]:
    """Load non-empty JSONL rows from ``path`` and require object rows."""
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(cast(JsonDict, payload))
    return rows


def load_benchmark_payload(path: Path) -> list[JsonDict] | JsonDict:
    """Load a benchmark result payload from JSONL rows or supported JSON summary.

    Parameters
    ----------
    path
        Result file. ``*.jsonl`` is treated as row-wise judged output; ``*.json``
        accepts either a list of row objects or the committed LOCOMO summary
        shape with ``total_correct``, ``total_tested``, and ``by_category``.

    Returns
    -------
    list[dict[str, object]] | dict[str, object]
        Parsed row list or supported summary object.

    Raises
    ------
    ValueError
        If the file shape is not a supported benchmark artefact.
    """
    if path.suffix == ".jsonl":
        return _load_jsonl(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows: list[JsonDict] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"JSON item {index} is not an object")
            rows.append(cast(JsonDict, item))
        return rows
    if isinstance(payload, dict):
        obj = cast(JsonDict, payload)
        if {"total_correct", "total_tested", "by_category"}.issubset(obj):
            return obj
    raise ValueError("Unsupported benchmark JSON shape")


def _quantile_nearest(values: Sequence[float], q: float) -> float | None:
    """Return nearest-rank quantile for non-empty values; otherwise ``None``."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return round(ordered[rank - 1], 3)


def _median(values: Sequence[float]) -> float | None:
    """Return the median for non-empty values; otherwise ``None``."""
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 3)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 3)


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean for non-empty values; otherwise ``None``."""
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _summarise_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    benchmark: str,
    source_path: Path,
    generated_at: str,
    source_format: str,
) -> JsonDict:
    """Summarise row-wise judged benchmark output.

    Rows without a ``judge_label`` are unjudged: they carry no correctness
    evidence, so they are excluded from the score rather than silently counted
    as wrong (which would deflate accuracy and diverge from
    ``scorecard_report`` on the same artefact). Their count is reported as
    ``n_unjudged`` so a partially judged artefact stays visible.
    """
    correct_by: dict[str, int] = defaultdict(int)
    total_by: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    prompt_hashes: set[str] = set()
    models: set[str] = set()
    seeds: set[int] = set()
    prompt_token_estimate = 0
    completion_tokens = 0
    unjudged = 0

    for row in rows:
        # Seed provenance is run metadata, not correctness evidence, so it is
        # collected from every row — judged or not.
        seed = _int_from_value(row.get("seed"))
        if seed is not None:
            seeds.add(seed)
        if "judge_label" not in row:
            unjudged += 1
            continue
        qtype = str(row.get("question_type") or row.get("category") or "unknown")
        total_by[qtype] += 1
        if bool(row.get("judge_label")):
            correct_by[qtype] += 1

        latency = _number(row.get("judge_latency_ms"))
        if latency is not None:
            latencies.append(latency)

        prompt_hash = row.get("judge_prompt_sha256")
        if isinstance(prompt_hash, str) and prompt_hash:
            prompt_hashes.add(prompt_hash)
        else:
            prompt = row.get("judge_prompt")
            if isinstance(prompt, str) and prompt:
                prompt_hashes.add(prompt_sha256(prompt))

        model = row.get("judge_model")
        if isinstance(model, str) and model:
            models.add(model)

        exact_prompt_tokens = _field_int(row, "judge_prompt_tokens")
        estimated_prompt_tokens = _field_int(row, "judge_prompt_tokens_estimate")
        if exact_prompt_tokens is not None:
            prompt_token_estimate += exact_prompt_tokens
        elif estimated_prompt_tokens is not None:
            prompt_token_estimate += estimated_prompt_tokens
        else:
            prompt = row.get("judge_prompt")
            if isinstance(prompt, str):
                prompt_token_estimate += estimate_tokens(prompt)

        completion = _field_int(row, "judge_completion_tokens")
        if completion is not None:
            completion_tokens += completion

    overall_correct = sum(correct_by.values())
    overall_total = sum(total_by.values())
    by_category = {qtype: _score(correct_by[qtype], total_by[qtype]) for qtype in sorted(total_by)}
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "source_path": str(source_path),
        "source_format": source_format,
        "generated_at": generated_at,
        "n_records": overall_total,
        "n_unjudged": unjudged,
        "score": _score(overall_correct, overall_total),
        "by_category": by_category,
        "runtime": {
            "latency_ms": {
                "mean": _mean(latencies),
                "p50": _median(latencies),
                "p95": _quantile_nearest(latencies, 0.95),
            }
        },
        "tokens": {
            "judge_prompt_estimate": prompt_token_estimate,
            "judge_completion": completion_tokens,
        },
        "judge": {
            "models": sorted(models),
            "prompt_sha256": sorted(prompt_hashes),
        },
        "seeds": sorted(seeds),
    }


def _seeds_from_payload(payload: Mapping[str, object]) -> list[int]:
    """Extract distinct RNG seeds from a JSON summary payload.

    Accepts either a ``seeds`` list (multi-run summary) or a single ``seed``
    value; anything non-integral is ignored rather than fabricated.
    """
    seeds: set[int] = set()
    raw_list = payload.get("seeds")
    if isinstance(raw_list, list):
        for value in raw_list:
            parsed = _int_from_value(value)
            if parsed is not None:
                seeds.add(parsed)
    single = _int_from_value(payload.get("seed"))
    if single is not None:
        seeds.add(single)
    return sorted(seeds)


def _summarise_json_summary(
    payload: Mapping[str, object],
    *,
    benchmark: str,
    source_path: Path,
    generated_at: str,
) -> JsonDict:
    """Summarise the committed LOCOMO-style benchmark JSON payload."""
    correct = _int_from_value(payload.get("total_correct"))
    total = _int_from_value(payload.get("total_tested"))
    if correct is None or total is None:
        raise ValueError("JSON summary must contain integer total_correct and total_tested")

    raw_categories = payload.get("by_category")
    if not isinstance(raw_categories, Mapping):
        raise ValueError("JSON summary must contain object by_category")

    by_category: dict[str, JsonDict] = {}
    for name, raw_stats in raw_categories.items():
        if not isinstance(raw_stats, Mapping):
            raise ValueError(f"Category {name!s} is not an object")
        cat_correct = _int_from_value(raw_stats.get("correct"))
        cat_total = _int_from_value(raw_stats.get("total"))
        if cat_correct is None or cat_total is None:
            raise ValueError(f"Category {name!s} lacks integer correct/total")
        by_category[str(name)] = _score(cat_correct, cat_total)

    elapsed_s = _number(payload.get("elapsed_s"))
    runtime: JsonDict = {"elapsed_s": elapsed_s}
    if elapsed_s is not None and total:
        runtime["mean_per_record_ms"] = round(elapsed_s * 1000.0 / total, 3)
    else:
        runtime["mean_per_record_ms"] = None

    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "source_path": str(source_path),
        "source_format": "json",
        "generated_at": generated_at,
        "n_records": total,
        "n_unjudged": 0,
        "score": _score(correct, total),
        "by_category": by_category,
        "runtime": runtime,
        "tokens": {"judge_prompt_estimate": 0, "judge_completion": 0},
        "judge": {"models": [], "prompt_sha256": []},
        "seeds": _seeds_from_payload(payload),
    }
    method = payload.get("method")
    if isinstance(method, str):
        report["method"] = method
    llm_enabled = payload.get("llm_enabled")
    if isinstance(llm_enabled, bool):
        report["llm_enabled"] = llm_enabled
    return report


def report_from_path(
    path: Path,
    *,
    benchmark: str,
    generated_at: str | None = None,
) -> JsonDict:
    """Build a benchmark report from a JSONL result file or supported JSON summary.

    Parameters
    ----------
    path
        Benchmark artefact path.
    benchmark
        Human-readable benchmark identifier, such as ``"longmemeval"`` or
        ``"locomo"``.
    generated_at
        Optional ISO-8601 timestamp. Defaults to current UTC time.

    Returns
    -------
    dict[str, object]
        Machine-readable benchmark evidence report.
    """
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    payload = load_benchmark_payload(path)
    if isinstance(payload, list):
        source_format = "jsonl" if path.suffix == ".jsonl" else "json"
        return _summarise_rows(
            payload,
            benchmark=benchmark,
            source_path=path,
            generated_at=stamp,
            source_format=source_format,
        )
    return _summarise_json_summary(
        payload,
        benchmark=benchmark,
        source_path=path,
        generated_at=stamp,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the benchmark-report command-line parser."""
    parser = argparse.ArgumentParser(
        prog="remanentia-benchmark-report",
        description="Build a benchmark evidence report from persisted result files.",
    )
    parser.add_argument("results", type=Path, help="benchmark JSONL rows or JSON summary")
    parser.add_argument("--benchmark", required=True, help="benchmark name")
    parser.add_argument("--output", type=Path, required=True, help="report JSON destination")
    parser.add_argument("--generated-at", default=None, help="ISO-8601 timestamp override")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark-report CLI."""
    args = _build_parser().parse_args(argv)
    report = report_from_path(
        args.results,
        benchmark=args.benchmark,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    score = cast(Mapping[str, object], report["score"])
    print(
        f"wrote {args.benchmark} report "
        f"{score['accuracy']}% ({score['correct']}/{score['total']}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
