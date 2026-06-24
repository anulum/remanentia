# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — index-freshness gate (MS.0: detect a stalled consolidation)

"""Measure how far the retrieval index lags behind its source firehose.

The memory pipeline is a chain: incoming **stimuli** are written as
**findings**, periodically **consolidated** into digests, and folded into the
**vector index** the retriever actually searches. Each stage can stall
independently and silently — and one did: a 2026-05-12 stall left the vector
index frozen on an April worldview for ~8 weeks while stimuli kept arriving,
so recall answered from a corpus that no longer reflected reality. Nothing
alerted, because nothing watched the drift.

This module is that watch. It probes each stage for its newest artifact and
reports the **drift** between the source and the sink — newest stimulus minus
newest indexed artifact. Past a configured tolerance the pipeline is declared
stale and the gate fails closed, so a cron or CI job catches a stall in days,
not weeks. It reads only mtimes; it never mutates the corpus and makes no
decision about *how* to re-consolidate — it only makes the lag visible and
loud.

Drift is measured on file modification time, not on dates parsed out of
filenames: mtime is uniform across every stage (stimuli, findings, digests,
binary index blobs) and reflects when an artifact was actually produced, which
is the question a freshness gate asks. A stage whose directory is missing or
empty reads as "no artifact" (``newest is None``) rather than an error, so the
gate degrades to a clear signal instead of a crash on a fresh checkout.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SECONDS_PER_DAY = 86400.0

DEFAULT_MAX_DRIFT_DAYS = 7.0
"""Tolerated lag between the newest source and the newest index artifact.
A weekly consolidation cadence makes a drift beyond a week a real stall, not
ordinary scheduling jitter."""


@dataclass(frozen=True)
class StageFreshness:
    """The newest-artifact reading for one pipeline stage.

    Attributes
    ----------
    name:
        Human-readable stage label (e.g. ``"vector-index"``).
    newest:
        Modification time of the most recently changed artifact, in epoch
        seconds, or ``None`` when the stage holds no artifact.
    count:
        Number of artifacts matched.
    root:
        The directory probed, as a string, for reporting.
    """

    name: str
    newest: float | None
    count: int
    root: str

    @property
    def present(self) -> bool:
        """Whether the stage holds at least one artifact."""
        return self.newest is not None


@dataclass(frozen=True)
class PipelineFreshness:
    """Drift verdict across an ordered source→sink chain of stages."""

    stages: tuple[StageFreshness, ...]
    max_drift_seconds: float

    @property
    def source(self) -> StageFreshness | None:
        """The newest-bearing upstream stage (the firehose end)."""
        for stage in self.stages:
            if stage.present:
                return stage
        return None

    @property
    def sink(self) -> StageFreshness | None:
        """The newest-bearing downstream stage (the index end)."""
        for stage in reversed(self.stages):
            if stage.present:
                return stage
        return None

    @property
    def drift_seconds(self) -> float | None:
        """Newest source minus newest sink, in seconds.

        ``None`` when fewer than two stages carry an artifact (no chain to
        measure). Positive means the sink lags the source; a value at or below
        zero means the sink is at least as fresh as the source.
        """
        source, sink = self.source, self.sink
        if source is None or sink is None or source is sink:
            return None
        assert source.newest is not None and sink.newest is not None
        return source.newest - sink.newest

    @property
    def drift_days(self) -> float | None:
        """:attr:`drift_seconds` expressed in days, or ``None``."""
        drift = self.drift_seconds
        return None if drift is None else drift / SECONDS_PER_DAY

    @property
    def stale(self) -> bool:
        """Whether the pipeline has fallen behind beyond tolerance.

        Stale when the source→sink drift exceeds the tolerance, and also when
        a source carries artifacts but no downstream sink exists at all — an
        index that was never built lags infinitely, which the drift figure
        alone cannot express.
        """
        source, sink = self.source, self.sink
        if source is not None and (sink is None or sink is source):
            return True
        drift = self.drift_seconds
        return drift is not None and drift > self.max_drift_seconds

    def summary(self) -> str:
        """A one-line-per-stage report ending in the drift verdict."""
        lines = []
        for stage in self.stages:
            if stage.present:
                age = f"newest {_iso(stage.newest)} · {stage.count} artifact(s)"
            else:
                age = "no artifact"
            lines.append(f"  {stage.name:<16} {age}")
        verdict = "STALE" if self.stale else "fresh"
        drift = self.drift_days
        drift_str = "n/a" if drift is None else f"{drift:.1f}d"
        lines.append(
            f"  → drift (source→sink) {drift_str} "
            f"(tolerance {self.max_drift_seconds / SECONDS_PER_DAY:.1f}d) :: {verdict}"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """A JSON-serialisable view for cron/CI consumers."""
        return {
            "stages": [
                {
                    "name": s.name,
                    "newest": s.newest,
                    "newest_iso": _iso(s.newest),
                    "count": s.count,
                    "root": s.root,
                }
                for s in self.stages
            ],
            "drift_seconds": self.drift_seconds,
            "drift_days": self.drift_days,
            "max_drift_seconds": self.max_drift_seconds,
            "stale": self.stale,
        }


def _iso(epoch: float | None) -> str:
    """Format an epoch time as a UTC ISO-8601 string, or ``"—"`` if absent."""
    if epoch is None:
        return "—"
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe_stage(name: str, root: str | Path, patterns: Sequence[str]) -> StageFreshness:
    """Read the newest mtime and artifact count for one stage.

    Parameters
    ----------
    name:
        Stage label for the report.
    root:
        Directory to scan; a missing directory yields an empty stage.
    patterns:
        One or more glob patterns (relative to *root*) selecting the stage's
        artifacts. Patterns may recurse (e.g. ``"**/*.md"``).
    """
    root_path = Path(root)
    newest: float | None = None
    count = 0
    if root_path.is_dir():
        for pattern in patterns:
            for match in root_path.glob(pattern):
                if not match.is_file():
                    continue
                count += 1
                mtime = match.stat().st_mtime
                if newest is None or mtime > newest:
                    newest = mtime
    return StageFreshness(name=name, newest=newest, count=count, root=str(root_path))


def assess_pipeline(
    stages: Iterable[StageFreshness],
    *,
    max_drift_days: float = DEFAULT_MAX_DRIFT_DAYS,
) -> PipelineFreshness:
    """Combine ordered stage readings into a drift verdict.

    *stages* must be ordered source→sink (firehose first, index last); the
    drift is measured between the newest-bearing stage from each end.
    """
    return PipelineFreshness(
        stages=tuple(stages),
        max_drift_seconds=max_drift_days * SECONDS_PER_DAY,
    )


def default_stages(base: str | Path | None = None) -> list[StageFreshness]:
    """Probe Remanentia's canonical source→sink pipeline.

    Resolves the repository base from *base* or ``REMANENTIA_BASE`` (falling
    back to this file's directory), and the stimuli firehose from
    ``REMANENTIA_STIMULI_DIR`` (the live store lives outside the repo, so it is
    env-pointed; an unset or absent directory simply reads as empty). The
    ordered chain is stimuli → findings → digests → vector-index.
    """
    if base is None:
        base = os.environ.get("REMANENTIA_BASE")
    root = Path(base) if base is not None else Path(__file__).parent
    stimuli_dir = os.environ.get("REMANENTIA_STIMULI_DIR", str(root / "snn_stimuli"))
    return [
        probe_stage("stimuli", stimuli_dir, ["*.json"]),
        probe_stage("findings", root / "memory" / "semantic", ["**/*.md"]),
        probe_stage("digests", root / "memory" / "digests", ["*.md"]),
        probe_stage("vector-index", root / "snn_state" / "vector_index", ["*.npz", "*.sqlite"]),
    ]


def assess_default(
    base: str | Path | None = None,
    *,
    max_drift_days: float = DEFAULT_MAX_DRIFT_DAYS,
) -> PipelineFreshness:
    """Convenience: probe the canonical stages and return their drift verdict."""
    return assess_pipeline(default_stages(base), max_drift_days=max_drift_days)


def main() -> int:  # pragma: no cover — CLI/cron entry point
    """Print the freshness report; exit non-zero when the pipeline is stale."""
    import json
    import sys

    as_json = "--json" in sys.argv[1:]
    report = assess_default()
    if as_json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print("Remanentia index freshness:")
        print(report.summary())
    return 1 if report.stale else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
