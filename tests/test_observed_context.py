# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the LLM observe→reflect reader context

"""Verify the LLM Observer distils retrieved sessions into a lean, dated set.

The mechanism only helps if it (a) hands the reader an observation block instead
of the transcript when the completion is good, (b) falls back to the raw dump —
never an empty answer — when the completion is blank, failed, or unparsable, and
(c) chunks the sessions so a short-context local model is never handed the whole
retrieved dump in one call. All three, plus the parse discipline (dating,
supersession, dedup, caps), are pinned here against a fake completer so no model
runs in the suite.
"""

from __future__ import annotations

from observed_context import Completer, ObservedContext, build_observed_context

SESSIONS = [
    "=== Session 1 (2023-05-01) ===\n[USER]: I adopted a dog named Max.",
    "=== Session 2 (2023-06-12) ===\n[USER]: Max is now on a grain-free diet.",
]


def _fixed(reply: str | None) -> Completer:
    """A completer that always returns *reply*, ignoring the prompt."""

    def _complete(prompt: str, max_tokens: int) -> str | None:
        return reply

    return _complete


class TestFallbackToRawDump:
    """A bad observation pass must yield an empty context, not a starved reader."""

    def test_empty_sessions_short_circuits(self):
        called = False

        def _complete(prompt: str, max_tokens: int) -> str | None:
            nonlocal called
            called = True
            return "- [2023] something"

        ctx = build_observed_context("q", [], _complete)
        assert not ctx
        assert ctx.rendered == ""
        # No chunk to observe → no model call is worth making.
        assert called is False

    def test_whitespace_only_sessions_short_circuit(self):
        ctx = build_observed_context("q", ["   ", "\n\t"], _fixed("- [2023] x"))
        assert not ctx

    def test_none_completion(self):
        ctx = build_observed_context("q", SESSIONS, _fixed(None))
        assert not ctx
        assert ctx.rendered == ""

    def test_empty_completion(self):
        ctx = build_observed_context("q", SESSIONS, _fixed(""))
        assert not ctx

    def test_completer_exception_is_swallowed(self):
        def _boom(prompt: str, max_tokens: int) -> str | None:
            raise RuntimeError("backend down")

        ctx = build_observed_context("q", SESSIONS, _boom)
        assert not ctx
        assert ctx.rendered == ""

    def test_whitespace_only_completion_parses_to_empty(self):
        ctx = build_observed_context("q", SESSIONS, _fixed("\n   \n\t\n"))
        assert not ctx

    def test_date_only_lines_have_no_body(self):
        ctx = build_observed_context("q", SESSIONS, _fixed("[2023-05-01]\n[2023-06-12]"))
        assert not ctx


class TestDistillation:
    def test_parses_dated_observations_in_authored_order(self):
        reply = (
            "- [2023-06-12] Max is on a grain-free diet.\n"
            "- [2023-05-01] User adopted a dog named Max.\n"
        )
        ctx = build_observed_context("What does Max eat?", SESSIONS, _fixed(reply))
        assert ctx
        assert isinstance(ctx, ObservedContext)
        assert len(ctx.observations) == 2
        assert ctx.observations[0].date == "2023-06-12"
        assert ctx.observations[0].text == "Max is on a grain-free diet."
        assert ctx.rendered.startswith("OBSERVATIONS (")
        assert "[2023-06-12] Max is on a grain-free diet." in ctx.rendered

    def test_undated_line_renders_as_undated(self):
        ctx = build_observed_context("q", SESSIONS, _fixed("User adopted a dog named Max."))
        assert ctx.observations[0].date == ""
        assert "[undated] User adopted a dog named Max." in ctx.rendered

    def test_superseded_marker_is_stripped_and_flagged(self):
        reply = "- [2023-05-01] Max eats kibble. [superseded: switched later]"
        ctx = build_observed_context("q", SESSIONS, _fixed(reply))
        assert ctx.observations[0].superseded is True
        assert "superseded: switched later" not in ctx.observations[0].text
        assert ctx.observations[0].text == "Max eats kibble."
        assert ctx.rendered.endswith("[superseded]")

    def test_duplicate_bodies_collapse_case_insensitively(self):
        reply = "[2023] Max is a dog.\n[2024] MAX IS A DOG.\n[2025] Max sleeps a lot."
        ctx = build_observed_context("q", SESSIONS, _fixed(reply))
        bodies = [o.text for o in ctx.observations]
        assert bodies == ["Max is a dog.", "Max sleeps a lot."]

    def test_bullet_and_whitespace_normalised(self):
        reply = "*    [2023-05-01]    Max   likes    walks.   "
        ctx = build_observed_context("q", SESSIONS, _fixed(reply))
        assert ctx.observations[0].text == "Max likes walks."
        assert ctx.observations[0].date == "2023-05-01"


class TestCaps:
    def test_count_cap_truncates(self):
        reply = "\n".join(f"[2023] fact number {i}" for i in range(10))
        ctx = build_observed_context("q", SESSIONS, _fixed(reply), max_observations=3)
        assert len(ctx.observations) == 3

    def test_char_budget_stops_after_first(self):
        reply = "[2023] first observation here\n[2024] second observation here"
        # One chunk (default per_call budget), so arrival order is authored order;
        # the global char budget admits the header + first line but not the second.
        ctx = build_observed_context("q", SESSIONS, _fixed(reply), char_budget=1)
        assert len(ctx.observations) == 1
        assert ctx.observations[0].text == "first observation here"

    def test_max_tokens_is_forwarded_to_completer(self):
        seen: dict[str, int] = {}

        def _complete(prompt: str, max_tokens: int) -> str | None:
            seen["max_tokens"] = max_tokens
            return "[2023] a fact"

        build_observed_context("q", SESSIONS, _complete, max_tokens=123)
        assert seen["max_tokens"] == 123

    def test_prompt_carries_question_and_sessions(self):
        seen: dict[str, str] = {}

        def _complete(prompt: str, max_tokens: int) -> str | None:
            seen["prompt"] = prompt
            return "[2023] a fact"

        build_observed_context("Where does Max live?", SESSIONS, _complete)
        assert "Where does Max live?" in seen["prompt"]
        assert "Max is now on a grain-free diet." in seen["prompt"]


class TestChunking:
    """Sessions are packed into size-bounded chunks so a short-context local model
    is never handed the whole retrieved dump in one call."""

    def test_small_budget_makes_one_call_per_session(self):
        calls: list[str] = []

        def _complete(prompt: str, max_tokens: int) -> str | None:
            calls.append(prompt)
            # Emit an observation that echoes which session this call saw.
            if "adopted a dog" in prompt:
                return "[2023-05-01] User adopted Max."
            return "[2023-06-12] Max is on a grain-free diet."

        # Each SESSIONS block is > 20 chars, so a tiny budget splits them.
        ctx = build_observed_context("q", SESSIONS, _complete, per_call_char_budget=40)
        assert len(calls) == 2  # one call per session
        # Each prompt carried only its own session, not the other.
        assert "adopted a dog" in calls[0] and "grain-free" not in calls[0]
        assert "grain-free" in calls[1] and "adopted a dog" not in calls[1]
        # Observations from both chunks merged.
        texts = [o.text for o in ctx.observations]
        assert texts == ["User adopted Max.", "Max is on a grain-free diet."]

    def test_large_budget_is_a_single_call(self):
        calls: list[str] = []

        def _complete(prompt: str, max_tokens: int) -> str | None:
            calls.append(prompt)
            return "[2023] one fact"

        build_observed_context("q", SESSIONS, _complete, per_call_char_budget=1_000_000)
        assert len(calls) == 1  # both sessions in one chunk — the validated cloud shape

    def test_dedup_across_chunks(self):
        def _complete(prompt: str, max_tokens: int) -> str | None:
            # Both chunks report the same fact — it must appear once.
            return "[2023] Max is a dog."

        ctx = build_observed_context("q", SESSIONS, _complete, per_call_char_budget=40)
        assert len(ctx.observations) == 1

    def test_partial_chunk_failure_still_yields(self):
        def _complete(prompt: str, max_tokens: int) -> str | None:
            if "adopted a dog" in prompt:
                return None  # first chunk fails
            return "[2023-06-12] Max is on a grain-free diet."

        ctx = build_observed_context("q", SESSIONS, _complete, per_call_char_budget=40)
        assert len(ctx.observations) == 1
        assert ctx.observations[0].text == "Max is on a grain-free diet."

    def test_session_larger_than_budget_gets_its_own_chunk(self):
        calls = 0

        def _complete(prompt: str, max_tokens: int) -> str | None:
            nonlocal calls
            calls += 1
            return f"[2023] fact {calls}"

        big = ["x" * 500, "y" * 500, "z" * 500]
        ctx = build_observed_context("q", big, _complete, per_call_char_budget=100)
        assert calls == 3  # each oversized session observed alone
        assert len(ctx.observations) == 3
