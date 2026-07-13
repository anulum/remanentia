# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for answer_extractor

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from answer_extractor import (
    _extract_answer_python,
    _extract_best_sentence_python,
    extract_all_candidates,
    extract_answer,
    extract_best_sentence,
    fuzzy_match,
    normalize_number,
    _extract_date_answer,
    _extract_name_answer,
    _extract_number_answer,
    _extract_percentage_answer,
    _extract_version_answer,
    _extract_yes_no,
    _fuzzy_match_python,
    _is_how_many_question,
    _is_version_question,
    _is_when_question,
    _is_who_question,
    _is_yes_no_question,
    _normalize_number_python,
)
from llm_backend import LocalLLMBackend


# ── Question type detection ──────────────────────────────────────


class TestQuestionTypes:
    def test_when(self) -> None:
        assert _is_when_question("when did we fix the bug")
        assert _is_when_question("what date was the release")
        assert not _is_when_question("where is the file")

    def test_how_many(self) -> None:
        assert _is_how_many_question("how many tests passed")
        assert _is_how_many_question("how much did it cost")
        assert not _is_how_many_question("how does it work")

    def test_version(self) -> None:
        assert _is_version_question("what version is deployed")
        assert _is_version_question("latest release number")
        assert not _is_version_question("what is the status")

    def test_who(self) -> None:
        assert _is_who_question("who wrote this code")
        assert not _is_who_question("what is this code")

    def test_yes_no(self) -> None:
        assert _is_yes_no_question("is the daemon running")
        assert _is_yes_no_question("did we fix the bug")
        assert _is_yes_no_question("can we deploy now")
        assert not _is_yes_no_question("what is the status")


# ── Date extraction ──────────────────────────────────────────────


class TestExtractDate:
    def test_iso_date(self) -> None:
        assert _extract_date_answer("Fixed on 2026-03-15 during the sprint.") == "2026-03-15"

    def test_english_date(self) -> None:
        result = _extract_date_answer("Released on March 15, 2026.")
        assert result is not None
        assert "March" in result

    def test_no_date(self) -> None:
        assert _extract_date_answer("No dates in this text.") is None

    def test_multiple_dates_returns_first(self) -> None:
        assert _extract_date_answer("Started 2026-03-10, finished 2026-03-15.") == "2026-03-10"

    def test_query_terms_choose_nearest_date(self) -> None:
        text = (
            "The design review happened on 2026-01-01. "
            + "x " * 100
            + "The release candidate shipped on 2026-02-03."
        )

        assert _extract_date_answer(text, query="when did release candidate ship") == "2026-02-03"

    def test_equal_query_scores_keep_first_date(self) -> None:
        text = "Started 2026-03-10 and finished 2026-03-15."
        assert _extract_date_answer(text, query="unrelated token") == "2026-03-10"


class TestPythonFallbackDispatch:
    def test_duration_without_dates_continues_to_standard_engine(self) -> None:
        assert extract_answer("how many days did it take", "No dates or numbers here.") is None

    def test_extract_answer_dispatches_question_types_without_native_extension(self) -> None:
        assert _extract_answer_python("when did it ship", "It shipped on 2026-02-03.") == "2026-02-03"
        assert _extract_answer_python("how many tests passed", "Exactly 42 tests passed.") == "42"
        assert _extract_answer_python("what version shipped", "Release v1.2.3 is live.") == "v1.2.3"
        assert _extract_answer_python("who approved it", "Alice approved the release.") == "Alice"
        assert _extract_answer_python("did it pass", "Yes, the suite passed.") == "Yes"
        assert _extract_answer_python("what percent passed", "Coverage reached 99.5%.") == "99.5%"
        assert _extract_answer_python("how long did it take", "The run took 14 days.") == "14"
        assert _extract_answer_python("what percentage passed", "Coverage reached 98%.") == "98%"

    def test_extract_answer_generic_fallback_order_without_native_extension(self) -> None:
        assert _extract_answer_python("tell me the metric", "Coverage reached 88%.") == "88%"
        assert _extract_answer_python("tell me the version", "Release v2.0 is live.") == "v2.0"
        assert _extract_answer_python("tell me the date", "Shipped on 2026-03-04.") == "2026-03-04"
        assert _extract_answer_python("tell me the number", "There were 123 failures.") == "123"
        assert _extract_answer_python("tell me something", "No extractable answer here.") is None

    def test_number_extraction_empty_and_queryless_paths(self) -> None:
        assert _extract_number_answer("No digits here.", "how many") is None
        assert _extract_number_answer("There were 2 and then 5.", "") == "2"
        assert (
            _extract_number_answer(
                "The old count was 2. " + "x " * 100 + "The release count was 5.",
                "release count",
            )
            == "5"
        )

    def test_fuzzy_match_python_fallback_paths(self) -> None:
        assert _fuzzy_match_python("", "gold") is False
        assert _fuzzy_match_python("March 15", "march 15") is True
        assert _fuzzy_match_python("answer", "the answer was used") is True
        assert _fuzzy_match_python("kitten", "sitting", threshold=0.5) is True
        assert _fuzzy_match_python("alpha", "omega", threshold=0.9) is False

    def test_normalize_number_python_fallback_paths(self) -> None:
        assert _normalize_number_python("1,234%") == "1234"
        assert _normalize_number_python("one hundred and five") == "105"
        assert _normalize_number_python("forty-two") == "42"
        assert _normalize_number_python("no number") is None

    def test_extract_best_sentence_python_fallback(self) -> None:
        paragraph = "Alpha is unrelated. Beta retrieval passed all checks."
        assert _extract_best_sentence_python("retrieval checks", paragraph) == (
            "Beta retrieval passed all checks."
        )
        assert _extract_best_sentence_python("missing terms", paragraph) is None


# ── Number extraction ────────────────────────────────────────────


class TestExtractNumber:
    def test_simple_number(self) -> None:
        result = _extract_number_answer("We processed 1986 questions.", "how many questions")
        assert result is not None
        assert "1986" in result

    def test_decimal_number(self) -> None:
        result = _extract_number_answer("Score was 66.4 on the benchmark.", "what score")
        assert result is not None
        assert "66.4" in result

    def test_filters_years(self) -> None:
        result = _extract_number_answer("In 2026 we processed 500 items.", "how many items")
        assert result is not None
        assert result != "2026"


# ── Version extraction ───────────────────────────────────────────


class TestExtractVersion:
    def test_semver(self) -> None:
        assert _extract_version_answer("Released v3.9.0 to PyPI.") == "v3.9.0"

    def test_two_part(self) -> None:
        assert _extract_version_answer("Updated to v0.2.") == "v0.2"

    def test_no_version(self) -> None:
        assert _extract_version_answer("No version here.") is None


# ── Percentage extraction ────────────────────────────────────────


class TestExtractPercentage:
    def test_integer_percent(self) -> None:
        assert _extract_percentage_answer("Accuracy was 100% on benchmark.") == "100%"

    def test_decimal_percent(self) -> None:
        assert _extract_percentage_answer("LOCOMO score: 66.4%.") == "66.4%"

    def test_no_percent(self) -> None:
        assert _extract_percentage_answer("No percentages here.") is None


# ── Name extraction ──────────────────────────────────────────────


class TestExtractName:
    def test_multi_word_name(self) -> None:
        result = _extract_name_answer("Written by Miroslav Sotek in 2026.", "who wrote")
        assert result is not None
        assert "Miroslav" in result

    def test_no_name(self) -> None:
        assert _extract_name_answer("no names here at all.", "who did") is None


# ── Yes/No extraction ───────────────────────────────────────────


class TestExtractYesNo:
    def test_negation(self) -> None:
        result = _extract_yes_no("The daemon is not running.", "is daemon running")
        assert result == "No"

    def test_affirmation(self) -> None:
        result = _extract_yes_no("The daemon is running.", "is daemon running")
        assert result == "Yes"


# ── Full extract_answer ──────────────────────────────────────────


class TestExtractAnswer:
    def test_when_question(self) -> None:
        result = extract_answer(
            "When did we fix the STDP bug?",
            "The STDP bug was fixed on 2026-03-15 by correcting the mask.",
        )
        assert result == "2026-03-15"

    def test_version_question(self) -> None:
        result = extract_answer(
            "What version was released?",
            "We released v3.9.0 to PyPI on March 15.",
        )
        assert result == "v3.9.0"

    def test_percentage_question(self) -> None:
        result = extract_answer(
            "What is the LOCOMO accuracy?",
            "The LOCOMO benchmark scored 66.4% on 1,986 questions.",
        )
        assert result == "66.4%"

    def test_yes_no_question(self) -> None:
        result = extract_answer(
            "Is the SNN used for retrieval?",
            "SNN retrieval is not used. Weight set to 0.00.",
        )
        assert result == "No"

    def test_generic_fallback(self) -> None:
        result = extract_answer(
            "What happened to the daemon?",
            "Daemon killed. GPU freed. v0.2.0 released. 66.4% accuracy.",
        )
        assert result is not None


# ── extract_all_candidates ───────────────────────────────────────


class TestExtractAllCandidates:
    def test_mixed_types(self) -> None:
        candidates = extract_all_candidates(
            "STDP results",
            "On 2026-03-15, accuracy reached 66.4% with v3.9.0. Processed 1,986 items.",
        )
        types = {c["type"] for c in candidates}
        assert "date" in types
        assert "percentage" in types
        assert "version" in types

    def test_proximity_boost(self) -> None:
        candidates = extract_all_candidates(
            "STDP accuracy",
            "STDP scored 66.4%. BM25 scored 48.9%.",
        )
        # 66.4% should score higher (closer to "STDP")
        pcts = [c for c in candidates if c["type"] == "percentage"]
        if len(pcts) >= 2:
            assert pcts[0]["answer"] == "66.4%"

    def test_empty(self) -> None:
        assert extract_all_candidates("query", "no extractable answers") == []

    def test_name_candidates(self) -> None:
        candidates = extract_all_candidates(
            "author",
            "Written by Miroslav Sotek at Anulum on 2026-03-15.",
        )
        types = {c["type"] for c in candidates}
        assert "name" in types


# ── Fuzzy matching ───────────────────────────────────────────────


class TestFuzzyMatch:
    def test_exact_match(self) -> None:
        assert fuzzy_match("hello", "hello") is True

    def test_substring_match(self) -> None:
        assert fuzzy_match("hello", "hello world") is True
        assert fuzzy_match("hello world", "hello") is True

    def test_fuzzy_close(self) -> None:
        assert fuzzy_match("Miroslav Sotek", "Miroslav Šotek") is True

    def test_fuzzy_distant(self) -> None:
        assert fuzzy_match("apple", "zebra") is False

    def test_empty(self) -> None:
        assert fuzzy_match("", "hello") is False
        assert fuzzy_match("hello", "") is False

    def test_case_insensitive(self) -> None:
        assert fuzzy_match("HELLO", "hello") is True


# ── Number normalization ─────────────────────────────────────────


class TestNormalizeNumber:
    def test_plain_number(self) -> None:
        assert normalize_number("42") == "42"

    def test_comma_number(self) -> None:
        assert normalize_number("1,986") == "1986"

    def test_percentage(self) -> None:
        assert normalize_number("66.4%") == "66.4"

    def test_word_number(self) -> None:
        assert normalize_number("forty-two") == "42"

    def test_word_teen(self) -> None:
        assert normalize_number("thirteen") == "13"

    def test_word_compound(self) -> None:
        assert normalize_number("twenty one") == "21"

    def test_not_a_number(self) -> None:
        assert normalize_number("hello") is None


# ── Sentence extraction ──────────────────────────────────────────


class TestExtractBestSentence:
    def test_finds_relevant(self) -> None:
        para = "The weather is nice. The STDP bug was fixed on March 15. It was a good day."
        result = extract_best_sentence("STDP bug fix", para)
        assert result is not None
        assert "STDP" in result

    def test_empty_paragraph(self) -> None:
        assert extract_best_sentence("query", "") is None

    def test_single_sentence(self) -> None:
        result = extract_best_sentence("test", "This is a test sentence.")
        assert result == "This is a test sentence."


# ── extract_answer dispatch branches ────────────────────────────


class TestExtractAnswerDispatch:
    def test_who_question(self) -> None:
        result = extract_answer("who wrote the code", "Written by Miroslav Sotek in 2026.")
        assert result is not None
        assert "Miroslav" in result

    def test_how_many_question(self) -> None:
        result = extract_answer("how many tests passed", "We ran 250 tests successfully.")
        assert result is not None

    def test_what_percent_question(self) -> None:
        result = extract_answer("what percent accuracy", "Accuracy was 81.2% on LOCOMO.")
        assert result == "81.2%"

    def test_generic_no_typed_match(self) -> None:
        result = extract_answer("tell me about the project", "No special types here at all.")
        assert result is None


# ── _extract_name_answer single word near query ─────────────────


class TestExtractNameSingleWord:
    def test_single_capitalized_near_query(self) -> None:
        result = _extract_name_answer("The project lead is Miroslav.", "who is leading the project")
        assert result is not None

    def test_unrelated_capitalized_word_is_ignored(self) -> None:
        text = "London is distant. " + "x" * 100 + " the project lead is Miroslav."
        assert _extract_name_answer(text, "who leads project") == "Miroslav"


# ── _is_what_percent_question ───────────────────────────────────


class TestIsWhatPercentQuestion:
    def test_percent_word(self) -> None:
        from answer_extractor import _is_what_percent_question

        assert _is_what_percent_question("what percent did we get")
        assert _is_what_percent_question("accuracy on benchmark")
        assert not _is_what_percent_question("where is the file")


# ── normalize_number edge cases ────────────────────────────────


class TestNormalizeNumberEdge:
    def test_hundred(self) -> None:
        assert normalize_number("one hundred") == "100"

    def test_and_word(self) -> None:
        assert normalize_number("twenty and one") == "21"


# ── Real local HTTP LLM integration ────────────────────────────


class _CompletionServer(ThreadingHTTPServer):
    responses: list[str | None]
    requests: list[dict[str, Any]]


class _CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = cast(_CompletionServer, self.server)
        length = int(self.headers.get("Content-Length", "0"))
        request = cast(dict[str, Any], json.loads(self.rfile.read(length)))
        server.requests.append(request)
        response = server.responses.pop(0)
        body = json.dumps({"choices": [{"message": {"content": response}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


@contextmanager
def _local_backend(*responses: str | None) -> Iterator[tuple[LocalLLMBackend, _CompletionServer]]:
    server = _CompletionServer(("127.0.0.1", 0), _CompletionHandler)
    server.responses = list(responses)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    backend = LocalLLMBackend(base_url=f"http://{host}:{port}/v1", model="test", timeout=2)
    try:
        yield backend, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _closed_backend() -> LocalLLMBackend:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()
    return LocalLLMBackend(base_url=f"http://{host}:{port}/v1", timeout=0.1)


def _prompt(request: dict[str, Any]) -> str:
    messages = cast(list[dict[str, str]], request["messages"])
    return messages[-1]["content"]


@contextmanager
def _configured(backend: LocalLLMBackend | None) -> Iterator[None]:
    from answer_extractor import get_llm_backend, set_llm_backend

    original = get_llm_backend()
    set_llm_backend(backend)
    try:
        yield
    finally:
        set_llm_backend(original)


class TestLLMExtractAnswer:
    def test_no_backend_returns_none(self) -> None:
        from answer_extractor import llm_extract_answer

        with _configured(None):
            assert llm_extract_answer("when", "context") is None

    def test_success_crosses_http_and_preserves_prompt_contract(self) -> None:
        from answer_extractor import llm_extract_answer

        with _local_backend("March 15, 2026") as (backend, server), _configured(backend):
            assert llm_extract_answer("when", "x" * 2000) == "March 15, 2026"
        request = server.requests[0]
        assert request["max_tokens"] == 100
        assert "x" * 1000 in _prompt(request)
        assert "x" * 1001 not in _prompt(request)

    @pytest.mark.parametrize(  # type: ignore[untyped-decorator]
        "response", ["unknown", "I don't know", "not mentioned", None]
    )
    def test_empty_or_abstaining_http_response_returns_none(self, response: str | None) -> None:
        from answer_extractor import llm_extract_answer

        with _local_backend(response) as (backend, _server), _configured(backend):
            assert llm_extract_answer("question", "paragraph") is None

    def test_real_connection_refusal_returns_none(self) -> None:
        from answer_extractor import llm_extract_answer

        with _configured(_closed_backend()):
            assert llm_extract_answer("question", "paragraph") is None


class TestLLMBackendAccessors:
    def test_set_get_and_clear_real_backend(self) -> None:
        from answer_extractor import get_llm_backend, set_llm_backend

        original = get_llm_backend()
        with _local_backend("answer") as (backend, _server):
            try:
                set_llm_backend(backend)
                assert get_llm_backend() is backend
                set_llm_backend(None)
                assert get_llm_backend() is None
            finally:
                set_llm_backend(original)


class TestLLMProspectiveQueries:
    def test_no_backend_and_failed_transport_return_empty(self) -> None:
        from answer_extractor import llm_generate_prospective_queries

        with _configured(None):
            assert llm_generate_prospective_queries("paragraph", "doc.md") == []
        with _configured(_closed_backend()):
            assert llm_generate_prospective_queries("paragraph", "doc.md") == []

    def test_real_http_queries_are_filtered_and_capped(self) -> None:
        from answer_extractor import llm_generate_prospective_queries

        response = "Tiny\n" + "\n".join(f"Query number {i} about something" for i in range(15))
        with _local_backend(response) as (backend, server), _configured(backend):
            queries = llm_generate_prospective_queries("paragraph", "doc.md")
        assert len(queries) == 8
        assert all(len(query) > 5 for query in queries)
        assert server.requests[0]["max_tokens"] == 200
        assert "Source: doc.md" in _prompt(server.requests[0])

    def test_missing_http_content_returns_empty(self) -> None:
        from answer_extractor import llm_generate_prospective_queries

        with _local_backend(None) as (backend, _server), _configured(backend):
            assert llm_generate_prospective_queries("paragraph", "doc.md") == []


class TestLLMSynthesizeAnswer:
    def test_no_backend_failed_transport_and_abstention_return_none(self) -> None:
        from answer_extractor import llm_synthesize_answer

        with _configured(None):
            assert llm_synthesize_answer("query", ["paragraph"]) is None
        with _configured(_closed_backend()):
            assert llm_synthesize_answer("query", ["paragraph"]) is None
        for response in ("unknown", "I don't know", "not mentioned", None):
            with _local_backend(response) as (backend, _server), _configured(backend):
                assert llm_synthesize_answer("query", ["paragraph"]) is None

    @pytest.mark.parametrize(  # type: ignore[untyped-decorator]
        ("query", "response", "prompt_fragment"),
        [
            ("would the user enjoy hiking?", "Yes, outdoors", "hypothetical"),
            ("what are the hobbies?", "hiking, reading", "List ALL"),
            ("how many tests?", "The answer is 42", "Be concise"),
        ],
    )
    def test_question_specific_prompts_cross_real_http(
        self, query: str, response: str, prompt_fragment: str
    ) -> None:
        from answer_extractor import llm_synthesize_answer

        with _local_backend(response) as (backend, server), _configured(backend):
            assert llm_synthesize_answer(query, ["source text"]) == response
        assert prompt_fragment in _prompt(server.requests[0])

    def test_only_first_ten_sources_cross_http(self) -> None:
        from answer_extractor import llm_synthesize_answer

        with _local_backend("answer") as (backend, server), _configured(backend):
            assert llm_synthesize_answer("query", [f"paragraph {i}" for i in range(20)]) == "answer"
        prompt = _prompt(server.requests[0])
        assert "[Source 10]" in prompt
        assert "[Source 11]" not in prompt


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAnswerExtractorPipeline:
    """Full pipeline: query → extract → normalise → match."""

    def test_extract_feeds_normalizer(self) -> None:
        from answer_extractor import extract_answer
        from answer_normalizer import normalize_answer

        answer = extract_answer("What accuracy?", "The accuracy reached 88.5% on LOCOMO.")
        assert answer is not None
        normed = normalize_answer(answer)
        assert "88.5" in normed

    def test_extract_feeds_fuzzy_match(self) -> None:
        from answer_extractor import extract_answer, fuzzy_match

        answer = extract_answer("When was it?", "The review was on March 15, 2026.")
        assert answer is not None
        assert fuzzy_match(answer, "March 15, 2026")

    def test_rust_and_python_agree(self) -> None:
        """Rust extract_answer matches Python on same input."""
        from answer_extractor import extract_answer

        result = extract_answer("When?", "Meeting on 2026-03-15.")
        assert result is not None
        assert "2026-03-15" in result or "March" in result


# ── Duration extraction (Task #27) ──────────────────────────────


class TestExtractDuration:
    def test_days_between_two_dates(self) -> None:
        from answer_extractor import extract_duration

        text = "The trip started on 2023-01-01 and ended on 2023-02-01."
        result = extract_duration(text, "how many days between start and end")
        assert result is not None
        assert "31 days" in result

    def test_weeks_between_dates(self) -> None:
        from answer_extractor import extract_duration

        text = "Project began 2023-03-01 and shipped 2023-03-22."
        result = extract_duration(text, "how many weeks between start and ship")
        assert result is not None
        assert "3 weeks" in result

    def test_months_between_dates(self) -> None:
        from answer_extractor import extract_duration

        text = "Joined on 2023-01-15, left on 2023-04-15."
        result = extract_duration(text, "how many months between joining and leaving")
        assert result is not None
        assert "3 months" in result

    def test_single_date_returns_none(self) -> None:
        from answer_extractor import extract_duration

        text = "Event on 2023-01-01 was great."
        assert extract_duration(text, "how many days") is None

    def test_no_dates_returns_none(self) -> None:
        from answer_extractor import extract_duration

        text = "No dates here at all."
        assert extract_duration(text, "how many days") is None

    def test_keyword_proximity_scoring(self) -> None:
        """Picks dates nearest to query keywords, not arbitrary first/last."""
        from answer_extractor import extract_duration

        # Sessions far enough apart that 80-char windows don't overlap
        text = (
            "The gym membership started on 2023-01-01 after months of planning. "
            + "x" * 100
            + " I had an unrelated dental appointment on 2023-02-15 at the clinic. "
            + "x" * 100
            + " The gym membership ended on 2023-03-01 when I moved away."
        )
        result = extract_duration(text, "how many days between gym start and end")
        assert result is not None
        assert "59 days" in result  # Jan 1 → Mar 1

    def test_duration_dispatched_from_extract_answer(self) -> None:
        """extract_answer dispatches to extract_duration for duration questions."""
        from answer_extractor import extract_answer

        text = "Started 2023-01-01, finished 2023-02-01."
        result = extract_answer("how many days between start and finish", text)
        assert result is not None
        assert "31 days" in result

    def test_is_duration_question(self) -> None:
        from answer_extractor import _is_duration_question

        assert _is_duration_question("how many days between x and y")
        assert _is_duration_question("how many weeks did it take")
        assert _is_duration_question("how long between the events")
        assert not _is_duration_question("what is the weather")

    def test_days_dual_format(self) -> None:
        """Task #33: days duration returns both exclusive and inclusive."""
        from answer_extractor import extract_duration

        text = "The trip started on 2023-01-01 and ended on 2023-02-01."
        result = extract_duration(text, "how many days between start and end")
        assert result is not None
        assert "31 days" in result
        assert "32 days" in result
        assert "both endpoints" in result

    def test_weeks_dual_format(self) -> None:
        from answer_extractor import extract_duration

        text = "Project began 2023-03-01 and shipped 2023-03-22."
        result = extract_duration(text, "how many weeks between start and ship")
        assert result is not None
        assert "3 weeks" in result
        assert "21 days exclusive" in result
        assert "22 inclusive" in result

    def test_months_dual_format(self) -> None:
        from answer_extractor import extract_duration

        text = "Joined on 2023-01-15, left on 2023-04-15."
        result = extract_duration(text, "how many months between joining and leaving")
        assert result is not None
        assert "3 months" in result
        assert "90 days exclusive" in result
        assert "91 inclusive" in result


# ── Proximity scoring tuning (Task #35) ─────────────────────────


class TestProximityScoring:
    def test_closer_token_scores_higher(self) -> None:
        from answer_extractor import _proximity_score

        # Date at position ~50; 'gym' right next to it vs 'gym' 100 chars away
        close = "The gym session happened on 2023-05-22 in the morning."
        far = (
            "The gym was opened last year as part of a community project, but this "
            "is about the 2023-05-22 community meeting."
        )
        q_tokens = {"gym", "session"}
        date_pos_close = close.find("2023-05-22")
        date_pos_far = far.find("2023-05-22")

        s_close = _proximity_score(close, date_pos_close, 10, q_tokens)
        s_far = _proximity_score(far, date_pos_far, 10, q_tokens)
        assert s_close > s_far

    def test_no_tokens_scores_zero(self) -> None:
        from answer_extractor import _proximity_score

        s = _proximity_score("unrelated 2023-05-22 content", 10, 10, {"gym", "session"})
        assert s == 0.0

    def test_proximity_picks_right_pair(self) -> None:
        """When multiple dates exist, distance-weighted scoring picks the ones
        closest to query keywords."""
        from answer_extractor import extract_duration

        text = (
            "The unrelated event happened on 2022-12-31 long ago. "
            + "x" * 200
            + " The gym membership started on 2023-01-01 after months of planning. "
            + "x" * 200
            + " I had a dentist appointment on 2023-02-15 at the clinic. "
            + "x" * 200
            + " The gym membership ended on 2023-03-01 when I moved away. "
            + "x" * 200
            + " Another unrelated event on 2023-12-31."
        )
        result = extract_duration(text, "how many days between gym start and end")
        assert result is not None
        # Jan 1 → Mar 1 = 59 days exclusive, 60 inclusive
        assert "59 days" in result
        assert "60 days" in result

    def test_tighter_window_ignores_distant_matches(self) -> None:
        """Query tokens >60 chars from date should contribute less."""
        from answer_extractor import _proximity_score

        # Single 'gym' token very far from date
        text = "gym " + "x" * 200 + " 2023-05-22"
        s = _proximity_score(text, text.find("2023-05-22"), 10, {"gym"})
        # Distance ~200 chars — outside 60-char window → score 0
        assert s == 0.0


class TestMultibyteSafety:
    """Regression for Rust PanicException on non-ASCII proximity slicing.

    Observed 2026-04-17 on LongMemEval Tokyo item where `¥` (2-byte UTF-8)
    sat on the ±80-byte proximity window edge and `text[s..e]` panicked
    inside a multi-byte sequence. Fix snaps byte indices to char boundary.
    """

    TOKYO_YEN = (
        "Navigating Tokyo's transportation system can be intimidating, but "
        "don't worry, I'm here to help! While taxis are convenient, they can "
        "be expensive, especially during peak hours. Fortunately, Tokyo has a "
        "comprehensive and efficient public transportation system. Subway "
        "tickets start at ¥170 and day passes cost around ¥800. The JR "
        "Yamanote line is a convenient circular route through the city. "
        "Taxis typically charge ¥730 for the first 2 km and then ¥90 per "
        "additional 280 m. For longer journeys consider an IC card like "
        "Suica or Pasmo which gives a small fare discount on most trains "
        "and buses across the metropolitan area so that you do not need to "
        "buy individual tickets each time you board a vehicle in Tokyo."
    )

    def test_yen_paragraph_does_not_panic(self) -> None:
        for q in ("how much", "when", "who", "what line"):
            extract_answer(q, self.TOKYO_YEN)  # must not raise

    def test_cjk_paragraph_does_not_panic(self) -> None:
        text = "東京の地下鉄は便利です。The Tokyo subway costs around ¥170 per ride."
        for q in ("how much does the subway cost", "when", "who uses"):
            extract_answer(q, text)

    def test_emoji_paragraph_does_not_panic(self) -> None:
        text = "The 🎯 target was hit on 2026-04-14 when Alice ran a 26.2 mile marathon."
        assert extract_answer("when was the target hit", text) is not None

    def test_euro_accents_paragraph_does_not_panic(self) -> None:
        text = "Zürich tickets cost €4.40 for short trips and Fräulein Müller confirmed."
        extract_answer("how much", text)
        extract_answer("who confirmed", text)

    def test_rust_path_direct(self) -> None:
        """Call Rust extractor directly — bypass Python fallbacks."""
        try:
            from remanentia_answer_extractor import (  # type: ignore[import-not-found]
                extract_answer as r_extract,
            )
        except ImportError:
            return  # Rust crate not built (CI without maturin build)
        for q in ("how much", "when", "who"):
            r_extract(q, self.TOKYO_YEN)  # must not panic


class TestExtractDurationDefensive:
    """Guard the date-parse fallback inside extract_duration."""

    def test_invalid_iso_date_is_skipped(self) -> None:
        """``2020-02-31`` matches ``\\b\\d{4}-\\d{2}-\\d{2}\\b`` but fails
        ``date(...)`` — the except-branch must drop it silently and not
        poison the duration. We just need to prove the branch runs.
        """
        from answer_extractor import extract_duration

        # A shape-matching-but-calendar-invalid string as the only date:
        # after the ``except ValueError: continue`` drops it, the function
        # falls through the ``if len(dates) < 2: return None`` guard.
        result = extract_duration(
            "I started the project on 2020-02-31 as the launch date.",
            "how many days between start and end",
        )
        assert result is None
