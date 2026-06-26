# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — recall calibration and abstention gate

"""Calibrate recall abstention from the persisted recall ledger.

The calibration surface consumes the existing :mod:`recall_ledger` stream:
recall queries with a top retrieval score and a downstream correctness verdict.
It fits a score threshold that maximises accepted-query coverage while keeping
empirical error at or below the requested target on labelled examples. The same
gate then produces per-query abstention decisions and a local correctness
estimate for score-neighbourhood monitoring.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from recall_ledger import RecallLedger, RecallQuery, default_ledger


DEFAULT_TARGET_ERROR_RATE = 0.1
DEFAULT_MIN_LABELLED = 30
DEFAULT_HOLDOUT_FRACTION = 0.25


@dataclass(frozen=True)
class CalibrationExample:
    """One recall query with the fields needed for calibration.

    Parameters
    ----------
    event_id
        Stable recall event identifier from the ledger.
    query
        Query text used for operator diagnostics.
    score
        Top retrieval score recorded at recall time.
    was_correct
        Downstream verifier verdict for the recalled memories.
    """

    event_id: str
    query: str
    score: float
    was_correct: bool


@dataclass(frozen=True)
class RecallCalibrationDecision:
    """Abstention decision for one recall score.

    Parameters
    ----------
    score
        Score being judged, or ``None`` when nothing was returned.
    threshold
        Fitted threshold. ``None`` means the gate is in cold start.
    abstain
        Whether the caller should withhold the recall result.
    estimated_correctness
        Empirical correctness estimate for recalls at least as strong as this
        score. A high score beyond the calibration range is estimated as 1.0.
    reason
        Machine-readable decision reason.
    n_neighbours
        Number of calibration examples used for the local estimate.
    n_labelled
        Number of labelled examples available to the gate.
    """

    score: float | None
    threshold: float | None
    abstain: bool
    estimated_correctness: float
    reason: str
    n_neighbours: int
    n_labelled: int


@dataclass(frozen=True)
class HoldoutCalibrationReport:
    """Held-out calibration metrics for a fitted gate.

    Parameters
    ----------
    n_total
        Number of held-out labelled recalls.
    n_accepted
        Number of held-out recalls the gate would answer.
    n_abstained
        Number of held-out recalls the gate would withhold.
    coverage
        Fraction of held-out recalls accepted.
    accuracy
        Correctness rate among accepted held-out recalls, or ``None`` if the
        gate accepted none.
    mean_estimated_correctness
        Mean per-query estimate among accepted recalls, or ``None`` if none
        were accepted.
    calibration_error
        Absolute difference between accepted accuracy and mean estimated
        correctness, or ``None`` if no accepted recall can be evaluated.
    error_rate
        Error rate among accepted held-out recalls, or ``None`` if none were
        accepted.
    """

    n_total: int
    n_accepted: int
    n_abstained: int
    coverage: float
    accuracy: float | None
    mean_estimated_correctness: float | None
    calibration_error: float | None
    error_rate: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable report dictionary."""

        return {
            "n_total": self.n_total,
            "n_accepted": self.n_accepted,
            "n_abstained": self.n_abstained,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "mean_estimated_correctness": self.mean_estimated_correctness,
            "calibration_error": self.calibration_error,
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True)
class CalibratedRecallGate:
    """Empirical threshold gate fitted from labelled recall examples.

    Parameters
    ----------
    threshold
        Minimum score needed to answer. ``None`` means cold start.
    target_error_rate
        Maximum accepted-query error rate requested by the operator.
    examples
        Labelled examples used for fitting and local score estimates.
    min_labelled
        Minimum examples required before a threshold may be trusted.
    """

    threshold: float | None
    target_error_rate: float
    examples: tuple[CalibrationExample, ...]
    min_labelled: int

    @classmethod
    def fit(
        cls,
        examples: Iterable[CalibrationExample],
        *,
        target_error_rate: float = DEFAULT_TARGET_ERROR_RATE,
        min_labelled: int = DEFAULT_MIN_LABELLED,
    ) -> CalibratedRecallGate:
        """Fit a threshold that maximises coverage under the error target.

        Parameters
        ----------
        examples
            Labelled recall examples with scores and correctness verdicts.
        target_error_rate
            Accepted-query error-rate ceiling in ``[0, 1]``.
        min_labelled
            Minimum labelled examples required to leave cold start.

        Returns
        -------
        CalibratedRecallGate
            Fitted gate. If there are too few labels, ``threshold`` is ``None``.

        Raises
        ------
        ValueError
            If the target or minimum count is outside its valid range.
        """

        if not 0.0 <= target_error_rate <= 1.0:
            raise ValueError("target_error_rate must be between 0 and 1")
        if min_labelled < 1:
            raise ValueError("min_labelled must be at least 1")

        labelled = tuple(examples)
        if len(labelled) < min_labelled:
            return cls(None, target_error_rate, labelled, min_labelled)

        best_threshold: float | None = None
        best_accepted = -1
        for threshold in sorted({example.score for example in labelled}):
            accepted = [example for example in labelled if example.score >= threshold]
            errors = sum(1 for example in accepted if not example.was_correct)
            error_rate = errors / len(accepted)
            if error_rate <= target_error_rate and len(accepted) > best_accepted:
                best_threshold = threshold
                best_accepted = len(accepted)

        if best_threshold is None:
            best_threshold = max(example.score for example in labelled)
        return cls(best_threshold, target_error_rate, labelled, min_labelled)

    @property
    def n_labelled(self) -> int:
        """Return the number of labelled calibration examples."""

        return len(self.examples)

    def decide(self, score: float | None) -> RecallCalibrationDecision:
        """Return the abstention decision for one recall score.

        Parameters
        ----------
        score
            Top retrieval score, or ``None`` when no memory was returned.

        Returns
        -------
        RecallCalibrationDecision
            Decision, local correctness estimate, and machine-readable reason.
        """

        if score is None:
            return RecallCalibrationDecision(
                score=None,
                threshold=self.threshold,
                abstain=True,
                estimated_correctness=0.0,
                reason="no_score",
                n_neighbours=0,
                n_labelled=self.n_labelled,
            )
        estimate, n_neighbours = self.estimate_correctness(score)
        if self.threshold is None:
            return RecallCalibrationDecision(
                score=score,
                threshold=None,
                abstain=True,
                estimated_correctness=estimate,
                reason="insufficient_labels",
                n_neighbours=n_neighbours,
                n_labelled=self.n_labelled,
            )
        if score < self.threshold:
            return RecallCalibrationDecision(
                score=score,
                threshold=self.threshold,
                abstain=True,
                estimated_correctness=estimate,
                reason="score_below_threshold",
                n_neighbours=n_neighbours,
                n_labelled=self.n_labelled,
            )
        return RecallCalibrationDecision(
            score=score,
            threshold=self.threshold,
            abstain=False,
            estimated_correctness=estimate,
            reason="accepted",
            n_neighbours=n_neighbours,
            n_labelled=self.n_labelled,
        )

    def estimate_correctness(self, score: float) -> tuple[float, int]:
        """Estimate correctness from examples at least as strong as ``score``.

        A score higher than the observed calibration range receives estimate
        ``1.0`` with zero neighbours: no lower-scoring miss should reduce a
        stronger-than-observed query. Scores inside the observed range use all
        examples with ``example.score >= score``. If no such examples exist for
        another reason, accepted-threshold examples provide the fallback prior.
        """

        if not self.examples:
            return 0.0, 0
        max_score = max(example.score for example in self.examples)
        if score > max_score:
            return 1.0, 0

        neighbours = [example for example in self.examples if example.score >= score]
        correct = sum(1 for example in neighbours if example.was_correct)
        return correct / len(neighbours), len(neighbours)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable gate summary."""

        accepted = (
            [example for example in self.examples if self.threshold is not None and example.score >= self.threshold]
            if self.threshold is not None
            else []
        )
        error_rate = (
            sum(1 for example in accepted if not example.was_correct) / len(accepted)
            if accepted
            else None
        )
        return {
            "threshold": self.threshold,
            "target_error_rate": self.target_error_rate,
            "min_labelled": self.min_labelled,
            "n_labelled": self.n_labelled,
            "n_accepted_calibration": len(accepted),
            "calibration_coverage": len(accepted) / self.n_labelled if self.n_labelled else 0.0,
            "calibration_error_rate": error_rate,
        }


def calibration_examples_from_ledger(ledger: RecallLedger) -> tuple[CalibrationExample, ...]:
    """Load scored, correctness-labelled calibration examples from ``ledger``.

    Parameters
    ----------
    ledger
        Recall ledger containing query and outcome records.

    Returns
    -------
    tuple[CalibrationExample, ...]
        Examples in ledger query order. Unscored recalls and recalls without a
        correctness verdict are ignored.
    """

    examples: list[CalibrationExample] = []
    for query in ledger.queries():
        example = _example_from_query(query)
        if example is not None:
            examples.append(example)
    return tuple(examples)


def fit_gate_from_ledger(
    ledger: RecallLedger,
    *,
    target_error_rate: float = DEFAULT_TARGET_ERROR_RATE,
    min_labelled: int = DEFAULT_MIN_LABELLED,
) -> CalibratedRecallGate:
    """Fit a recall gate directly from a persisted ledger."""

    return CalibratedRecallGate.fit(
        calibration_examples_from_ledger(ledger),
        target_error_rate=target_error_rate,
        min_labelled=min_labelled,
    )


def split_examples(
    examples: Sequence[CalibrationExample],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[tuple[CalibrationExample, ...], tuple[CalibrationExample, ...]]:
    """Split examples into calibration and held-out measurement sets.

    The split is deterministic and range-aware for small local ledgers: it
    reserves the highest-scored example plus lower-tail examples for holdout,
    keeping both high-confidence and abstention-zone checks in the report.
    """

    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    ordered = tuple(examples)
    if not ordered or holdout_fraction == 0.0:
        return ordered, ()
    holdout_count = max(1, int(round(len(ordered) * holdout_fraction)))
    holdout_count = min(holdout_count, max(len(ordered) - 1, 0))
    if holdout_count == 0:
        return ordered, ()

    holdout_indices = {0}
    lower_start = max(1, len(ordered) - holdout_count)
    holdout_indices.update(range(lower_start, len(ordered)))
    while len(holdout_indices) > holdout_count:
        holdout_indices.remove(max(holdout_indices))
    train = tuple(example for index, example in enumerate(ordered) if index not in holdout_indices)
    holdout = tuple(example for index, example in enumerate(ordered) if index in holdout_indices)
    return train, holdout


def evaluate_holdout(
    gate: CalibratedRecallGate,
    holdout: Iterable[CalibrationExample],
) -> HoldoutCalibrationReport:
    """Measure coverage, accuracy, and calibration error on held-out examples."""

    examples = tuple(holdout)
    accepted: list[tuple[CalibrationExample, RecallCalibrationDecision]] = []
    for example in examples:
        decision = gate.decide(example.score)
        if not decision.abstain:
            accepted.append((example, decision))

    n_total = len(examples)
    n_accepted = len(accepted)
    n_abstained = n_total - n_accepted
    coverage = n_accepted / n_total if n_total else 0.0
    if not accepted:
        return HoldoutCalibrationReport(
            n_total=n_total,
            n_accepted=0,
            n_abstained=n_abstained,
            coverage=coverage,
            accuracy=None,
            mean_estimated_correctness=None,
            calibration_error=None,
            error_rate=None,
        )
    accuracy = sum(1 for example, _decision in accepted if example.was_correct) / n_accepted
    mean_estimate = sum(decision.estimated_correctness for _example, decision in accepted) / n_accepted
    return HoldoutCalibrationReport(
        n_total=n_total,
        n_accepted=n_accepted,
        n_abstained=n_abstained,
        coverage=coverage,
        accuracy=accuracy,
        mean_estimated_correctness=mean_estimate,
        calibration_error=abs(accuracy - mean_estimate),
        error_rate=1.0 - accuracy,
    )


def _example_from_query(query: RecallQuery) -> CalibrationExample | None:
    """Convert a ledger query into a calibration example when fully labelled."""

    if query.score is None or query.was_correct is None:
        return None
    return CalibrationExample(
        event_id=query.event_id,
        query=query.query,
        score=float(query.score),
        was_correct=bool(query.was_correct),
    )


def _ledger_from_path(path: Path | None) -> RecallLedger:
    """Return an explicit-path ledger or the default runtime ledger."""

    if path is None:
        return default_ledger()
    return RecallLedger(path)


def build_report(
    ledger: RecallLedger,
    *,
    target_error_rate: float = DEFAULT_TARGET_ERROR_RATE,
    min_labelled: int = DEFAULT_MIN_LABELLED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> dict[str, object]:
    """Build a JSON-serialisable calibration report from a ledger."""

    examples = calibration_examples_from_ledger(ledger)
    train, holdout = split_examples(examples, holdout_fraction=holdout_fraction)
    fit_examples = train if len(train) >= min_labelled else examples
    gate = CalibratedRecallGate.fit(
        fit_examples,
        target_error_rate=target_error_rate,
        min_labelled=min_labelled,
    )
    report = gate.to_dict()
    report["holdout"] = evaluate_holdout(gate, holdout).to_dict()
    return report


def _format_text_report(report: dict[str, object]) -> str:
    """Format a report for operators running the CLI interactively."""

    lines = [
        "Recall calibration",
        f"  labelled examples: {report['n_labelled']}",
        f"  target error rate: {report['target_error_rate']}",
        f"  threshold: {report['threshold']}",
        f"  calibration coverage: {report['calibration_coverage']}",
        f"  calibration error rate: {report['calibration_error_rate']}",
    ]
    holdout = report.get("holdout")
    if isinstance(holdout, dict):
        lines.extend(
            [
                "  holdout:",
                f"    examples: {holdout['n_total']}",
                f"    accepted: {holdout['n_accepted']}",
                f"    coverage: {holdout['coverage']}",
                f"    accuracy: {holdout['accuracy']}",
                f"    calibration error: {holdout['calibration_error']}",
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recall calibration CLI.

    Parameters
    ----------
    argv
        Optional argument sequence. ``None`` reads process arguments.

    Returns
    -------
    int
        Process exit code. ``0`` means a report was emitted successfully.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=None, help="Recall ledger JSONL path.")
    parser.add_argument(
        "--target-error-rate",
        type=float,
        default=DEFAULT_TARGET_ERROR_RATE,
        help="Maximum accepted-query error rate.",
    )
    parser.add_argument(
        "--min-labelled",
        type=int,
        default=DEFAULT_MIN_LABELLED,
        help="Minimum correctness-labelled recalls required before answering.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Fraction of labelled recalls reserved for held-out measurement.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = build_report(
        _ledger_from_path(args.ledger),
        target_error_rate=args.target_error_rate,
        min_labelled=args.min_labelled,
        holdout_fraction=args.holdout_fraction,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
