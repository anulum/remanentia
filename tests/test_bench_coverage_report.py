# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for R3 answer-session coverage diagnostic

"""Coverage diagnostic for `bench_longmemeval.compute_coverage_buckets`.

Exercises the R3 post-hoc analyser via subprocess so the module-level
``sys.argv`` probes in `bench_longmemeval` do not taint the test
interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(results: list[dict], oracle_list: list[dict]) -> dict[str, list[float]]:
    """Invoke compute_coverage_buckets in a subprocess and return the result."""
    oracle_map = {q["question_id"]: q for q in oracle_list}
    code = (
        "import sys, json\n"
        f"sys.argv = ['bench_longmemeval.py']\n"
        "import bench_longmemeval as b\n"
        f"results = {results!r}\n"
        f"oracle = {oracle_map!r}\n"
        "buckets = b.compute_coverage_buckets(results, oracle)\n"
        "print(json.dumps({k: v for k, v in buckets.items()}))\n"
    )
    r = subprocess.run(
        [PY, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": str(REPO / ".venv" / "bin") + ":/usr/bin:/bin"},
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout.strip())


@pytest.fixture
def sample_oracle():
    return [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "answer_session_ids": ["s1", "s2"],
            "haystack_session_ids": ["s1", "s2", "s3"],
            "haystack_sessions": [
                [{"role": "user", "content": "YouTube tutorial about Python programming language"}],
                [{"role": "user", "content": "TikTok clip showing cats playing piano music"}],
                [{"role": "user", "content": "Unrelated session about weather reports"}],
            ],
        },
        {
            "question_id": "q2",
            "question_type": "temporal-reasoning",
            "answer_session_ids": ["t1"],
            "haystack_session_ids": ["t1", "t2"],
            "haystack_sessions": [
                [{"role": "user", "content": "Meeting on Monday March fifteenth"}],
                [{"role": "user", "content": "Coffee shop trip yesterday"}],
            ],
        },
    ]


class TestComputeCoverageBuckets:
    def test_empty_returns_empty_dict(self, sample_oracle):
        assert _run([], sample_oracle) == {}

    def test_passing_question_excluded(self, sample_oracle):
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "YouTube tutorial about Python",
                "judge_label": True,
            }
        ]
        assert _run(results, sample_oracle) == {}

    def test_full_coverage_failure(self, sample_oracle):
        """Hypothesis mentions both answer sessions → coverage 1.0."""
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "YouTube Python tutorial and TikTok piano cats together",
                "judge_label": False,
            }
        ]
        buckets = _run(results, sample_oracle)
        assert buckets == {"multi-session": [1.0]}

    def test_half_coverage_failure(self, sample_oracle):
        """Hypothesis only mentions session 1 content → coverage 0.5."""
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "YouTube Python tutorial content only mentioned",
                "judge_label": False,
            }
        ]
        buckets = _run(results, sample_oracle)
        assert buckets == {"multi-session": [0.5]}

    def test_zero_coverage_failure(self, sample_oracle):
        """Hypothesis mentions nothing from any answer session."""
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "Sorry I have no information about that topic",
                "judge_label": False,
            }
        ]
        buckets = _run(results, sample_oracle)
        assert buckets == {"multi-session": [0.0]}

    def test_unknown_question_id_skipped(self, sample_oracle):
        results = [
            {
                "question_id": "q-missing",
                "question_type": "multi-session",
                "hypothesis": "any",
                "judge_label": False,
            }
        ]
        assert _run(results, sample_oracle) == {}

    def test_empty_hypothesis_skipped(self, sample_oracle):
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "",
                "judge_label": False,
            }
        ]
        assert _run(results, sample_oracle) == {}

    def test_qtype_grouping(self, sample_oracle):
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "YouTube Python tutorial",
                "judge_label": False,
            },
            {
                "question_id": "q2",
                "question_type": "temporal-reasoning",
                "hypothesis": "Meeting Monday March fifteenth discussed",
                "judge_label": False,
            },
        ]
        buckets = _run(results, sample_oracle)
        assert set(buckets) == {"multi-session", "temporal-reasoning"}
        assert len(buckets["multi-session"]) == 1
        assert len(buckets["temporal-reasoning"]) == 1
