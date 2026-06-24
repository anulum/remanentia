# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for R3 answer-session coverage diagnostic

"""Coverage diagnostic for `bench_longmemeval.compute_coverage_buckets`.

The function is pure (no dependency on the module-level ``sys.argv`` /
``os.environ`` probes), so it is imported and called in-process — the
same direct-import pattern other bench tests use via ``conftest``'s
``sys.path`` insertion. The argv-sensitive constants that genuinely need
a clean interpreter are exercised separately in
``test_bench_longmemeval_cli.py``.
"""

from __future__ import annotations

import pytest

from bench_longmemeval import compute_coverage_buckets


def _buckets(results: list[dict], oracle_list: list[dict]) -> dict[str, list[float]]:
    """Call compute_coverage_buckets with an oracle keyed by question_id."""
    oracle = {q["question_id"]: q for q in oracle_list}
    return compute_coverage_buckets(results, oracle)


@pytest.fixture
def sample_oracle() -> list[dict]:
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
        assert _buckets([], sample_oracle) == {}

    def test_passing_question_excluded(self, sample_oracle):
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "YouTube tutorial about Python",
                "judge_label": True,
            }
        ]
        assert _buckets(results, sample_oracle) == {}

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
        assert _buckets(results, sample_oracle) == {"multi-session": [1.0]}

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
        assert _buckets(results, sample_oracle) == {"multi-session": [0.5]}

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
        assert _buckets(results, sample_oracle) == {"multi-session": [0.0]}

    def test_unknown_question_id_skipped(self, sample_oracle):
        results = [
            {
                "question_id": "q-missing",
                "question_type": "multi-session",
                "hypothesis": "any",
                "judge_label": False,
            }
        ]
        assert _buckets(results, sample_oracle) == {}

    def test_empty_hypothesis_skipped(self, sample_oracle):
        results = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "hypothesis": "",
                "judge_label": False,
            }
        ]
        assert _buckets(results, sample_oracle) == {}

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
        buckets = _buckets(results, sample_oracle)
        assert set(buckets) == {"multi-session", "temporal-reasoning"}
        assert len(buckets["multi-session"]) == 1
        assert len(buckets["temporal-reasoning"]) == 1

    def test_no_answer_sessions_skipped(self, sample_oracle):
        """A gold entry with no answer_session_ids contributes nothing."""
        oracle = sample_oracle + [
            {
                "question_id": "q3",
                "question_type": "multi-session",
                "answer_session_ids": [],
                "haystack_session_ids": ["x1"],
                "haystack_sessions": [[{"role": "user", "content": "anything at all here"}]],
            }
        ]
        results = [
            {
                "question_id": "q3",
                "question_type": "multi-session",
                "hypothesis": "anything at all here matches",
                "judge_label": False,
            }
        ]
        assert _buckets(results, oracle) == {}

    def test_answer_session_missing_from_haystack(self, sample_oracle):
        """An answer session absent from the haystack counts as not covered."""
        oracle = [
            {
                "question_id": "q4",
                "question_type": "multi-session",
                "answer_session_ids": ["present", "absent"],
                "haystack_session_ids": ["present"],
                "haystack_sessions": [
                    [{"role": "user", "content": "Saturn rings telescope observation"}],
                ],
            }
        ]
        results = [
            {
                "question_id": "q4",
                "question_type": "multi-session",
                "hypothesis": "Saturn rings telescope observation noted",
                "judge_label": False,
            }
        ]
        # 'present' covered (>=2 overlapping tokens), 'absent' uncovered → 1/2.
        assert _buckets(results, oracle) == {"multi-session": [0.5]}
