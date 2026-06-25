# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for answer_extractor

from __future__ import annotations

from unittest.mock import patch


from answer_extractor import (
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
    _is_how_many_question,
    _is_version_question,
    _is_when_question,
    _is_who_question,
    _is_yes_no_question,
)


def _without_native():
    return patch("answer_extractor.import_module", side_effect=ImportError)


# ── Question type detection ──────────────────────────────────────


class TestQuestionTypes:
    def test_when(self):
        assert _is_when_question("when did we fix the bug")
        assert _is_when_question("what date was the release")
        assert not _is_when_question("where is the file")

    def test_how_many(self):
        assert _is_how_many_question("how many tests passed")
        assert _is_how_many_question("how much did it cost")
        assert not _is_how_many_question("how does it work")

    def test_version(self):
        assert _is_version_question("what version is deployed")
        assert _is_version_question("latest release number")
        assert not _is_version_question("what is the status")

    def test_who(self):
        assert _is_who_question("who wrote this code")
        assert not _is_who_question("what is this code")

    def test_yes_no(self):
        assert _is_yes_no_question("is the daemon running")
        assert _is_yes_no_question("did we fix the bug")
        assert _is_yes_no_question("can we deploy now")
        assert not _is_yes_no_question("what is the status")


# ── Date extraction ──────────────────────────────────────────────


class TestExtractDate:
    def test_iso_date(self):
        assert _extract_date_answer("Fixed on 2026-03-15 during the sprint.") == "2026-03-15"

    def test_english_date(self):
        result = _extract_date_answer("Released on March 15, 2026.")
        assert result is not None
        assert "March" in result

    def test_no_date(self):
        assert _extract_date_answer("No dates in this text.") is None

    def test_multiple_dates_returns_first(self):
        assert _extract_date_answer("Started 2026-03-10, finished 2026-03-15.") == "2026-03-10"

    def test_query_terms_choose_nearest_date(self):
        text = (
            "The design review happened on 2026-01-01. "
            + "x " * 100
            + "The release candidate shipped on 2026-02-03."
        )

        assert _extract_date_answer(text, query="when did release candidate ship") == "2026-02-03"


class TestPythonFallbackDispatch:
    def test_extract_answer_dispatches_question_types_without_native_extension(self):
        with _without_native():
            assert extract_answer("when did it ship", "It shipped on 2026-02-03.") == "2026-02-03"
            assert extract_answer("how many tests passed", "Exactly 42 tests passed.") == "42"
            assert extract_answer("what version shipped", "Release v1.2.3 is live.") == "v1.2.3"
            assert extract_answer("who approved it", "Alice approved the release.") == "Alice"
            assert extract_answer("did it pass", "Yes, the suite passed.") == "Yes"
            assert extract_answer("what percent passed", "Coverage reached 99.5%.") == "99.5%"
            assert extract_answer("how long did it take", "The run took 14 days.") == "14"
            assert extract_answer("what percentage passed", "Coverage reached 98%.") == "98%"

    def test_extract_answer_generic_fallback_order_without_native_extension(self):
        with _without_native():
            assert extract_answer("tell me the metric", "Coverage reached 88%.") == "88%"
            assert extract_answer("tell me the version", "Release v2.0 is live.") == "v2.0"
            assert extract_answer("tell me the date", "Shipped on 2026-03-04.") == "2026-03-04"
            assert extract_answer("tell me the number", "There were 123 failures.") == "123"
            assert extract_answer("tell me something", "No extractable answer here.") is None

    def test_number_extraction_empty_and_queryless_paths(self):
        assert _extract_number_answer("No digits here.", "how many") is None
        assert _extract_number_answer("There were 2 and then 5.", "") == "2"
        assert (
            _extract_number_answer(
                "The old count was 2. " + "x " * 100 + "The release count was 5.",
                "release count",
            )
            == "5"
        )

    def test_fuzzy_match_python_fallback_paths(self):
        with _without_native():
            assert fuzzy_match("", "gold") is False
            assert fuzzy_match("March 15", "march 15") is True
            assert fuzzy_match("answer", "the answer was used") is True
            assert fuzzy_match("kitten", "sitting", threshold=0.5) is True
            assert fuzzy_match("alpha", "omega", threshold=0.9) is False

    def test_normalize_number_python_fallback_paths(self):
        with _without_native():
            assert normalize_number("1,234%") == "1234"
            assert normalize_number("one hundred and five") == "105"
            assert normalize_number("forty-two") == "42"
            assert normalize_number("no number") is None

    def test_extract_best_sentence_python_fallback(self):
        paragraph = "Alpha is unrelated. Beta retrieval passed all checks."
        with _without_native():
            assert extract_best_sentence("retrieval checks", paragraph) == (
                "Beta retrieval passed all checks."
            )
            assert extract_best_sentence("missing terms", paragraph) is None


# ── Number extraction ────────────────────────────────────────────


class TestExtractNumber:
    def test_simple_number(self):
        result = _extract_number_answer("We processed 1986 questions.", "how many questions")
        assert result is not None
        assert "1986" in result

    def test_decimal_number(self):
        result = _extract_number_answer("Score was 66.4 on the benchmark.", "what score")
        assert result is not None
        assert "66.4" in result

    def test_filters_years(self):
        result = _extract_number_answer("In 2026 we processed 500 items.", "how many items")
        assert result is not None
        assert result != "2026"


# ── Version extraction ───────────────────────────────────────────


class TestExtractVersion:
    def test_semver(self):
        assert _extract_version_answer("Released v3.9.0 to PyPI.") == "v3.9.0"

    def test_two_part(self):
        assert _extract_version_answer("Updated to v0.2.") == "v0.2"

    def test_no_version(self):
        assert _extract_version_answer("No version here.") is None


# ── Percentage extraction ────────────────────────────────────────


class TestExtractPercentage:
    def test_integer_percent(self):
        assert _extract_percentage_answer("Accuracy was 100% on benchmark.") == "100%"

    def test_decimal_percent(self):
        assert _extract_percentage_answer("LOCOMO score: 66.4%.") == "66.4%"

    def test_no_percent(self):
        assert _extract_percentage_answer("No percentages here.") is None


# ── Name extraction ──────────────────────────────────────────────


class TestExtractName:
    def test_multi_word_name(self):
        result = _extract_name_answer("Written by Miroslav Sotek in 2026.", "who wrote")
        assert result is not None
        assert "Miroslav" in result

    def test_no_name(self):
        assert _extract_name_answer("no names here at all.", "who did") is None


# ── Yes/No extraction ───────────────────────────────────────────


class TestExtractYesNo:
    def test_negation(self):
        result = _extract_yes_no("The daemon is not running.", "is daemon running")
        assert result == "No"

    def test_affirmation(self):
        result = _extract_yes_no("The daemon is running.", "is daemon running")
        assert result == "Yes"


# ── Full extract_answer ──────────────────────────────────────────


class TestExtractAnswer:
    def test_when_question(self):
        result = extract_answer(
            "When did we fix the STDP bug?",
            "The STDP bug was fixed on 2026-03-15 by correcting the mask.",
        )
        assert result == "2026-03-15"

    def test_version_question(self):
        result = extract_answer(
            "What version was released?",
            "We released v3.9.0 to PyPI on March 15.",
        )
        assert result == "v3.9.0"

    def test_percentage_question(self):
        result = extract_answer(
            "What is the LOCOMO accuracy?",
            "The LOCOMO benchmark scored 66.4% on 1,986 questions.",
        )
        assert result == "66.4%"

    def test_yes_no_question(self):
        result = extract_answer(
            "Is the SNN used for retrieval?",
            "SNN retrieval is not used. Weight set to 0.00.",
        )
        assert result == "No"

    def test_generic_fallback(self):
        result = extract_answer(
            "What happened to the daemon?",
            "Daemon killed. GPU freed. v0.2.0 released. 66.4% accuracy.",
        )
        assert result is not None


# ── extract_all_candidates ───────────────────────────────────────


class TestExtractAllCandidates:
    def test_mixed_types(self):
        candidates = extract_all_candidates(
            "STDP results",
            "On 2026-03-15, accuracy reached 66.4% with v3.9.0. Processed 1,986 items.",
        )
        types = {c["type"] for c in candidates}
        assert "date" in types
        assert "percentage" in types
        assert "version" in types

    def test_proximity_boost(self):
        candidates = extract_all_candidates(
            "STDP accuracy",
            "STDP scored 66.4%. BM25 scored 48.9%.",
        )
        # 66.4% should score higher (closer to "STDP")
        pcts = [c for c in candidates if c["type"] == "percentage"]
        if len(pcts) >= 2:
            assert pcts[0]["answer"] == "66.4%"

    def test_empty(self):
        assert extract_all_candidates("query", "no extractable answers") == []

    def test_name_candidates(self):
        candidates = extract_all_candidates(
            "author",
            "Written by Miroslav Sotek at Anulum on 2026-03-15.",
        )
        types = {c["type"] for c in candidates}
        assert "name" in types


# ── Fuzzy matching ───────────────────────────────────────────────


class TestFuzzyMatch:
    def test_exact_match(self):
        assert fuzzy_match("hello", "hello") is True

    def test_substring_match(self):
        assert fuzzy_match("hello", "hello world") is True
        assert fuzzy_match("hello world", "hello") is True

    def test_fuzzy_close(self):
        assert fuzzy_match("Miroslav Sotek", "Miroslav Šotek") is True

    def test_fuzzy_distant(self):
        assert fuzzy_match("apple", "zebra") is False

    def test_empty(self):
        assert fuzzy_match("", "hello") is False
        assert fuzzy_match("hello", "") is False

    def test_case_insensitive(self):
        assert fuzzy_match("HELLO", "hello") is True


# ── Number normalization ─────────────────────────────────────────


class TestNormalizeNumber:
    def test_plain_number(self):
        assert normalize_number("42") == "42"

    def test_comma_number(self):
        assert normalize_number("1,986") == "1986"

    def test_percentage(self):
        assert normalize_number("66.4%") == "66.4"

    def test_word_number(self):
        assert normalize_number("forty-two") == "42"

    def test_word_teen(self):
        assert normalize_number("thirteen") == "13"

    def test_word_compound(self):
        assert normalize_number("twenty one") == "21"

    def test_not_a_number(self):
        assert normalize_number("hello") is None


# ── Sentence extraction ──────────────────────────────────────────


class TestExtractBestSentence:
    def test_finds_relevant(self):
        para = "The weather is nice. The STDP bug was fixed on March 15. It was a good day."
        result = extract_best_sentence("STDP bug fix", para)
        assert result is not None
        assert "STDP" in result

    def test_empty_paragraph(self):
        assert extract_best_sentence("query", "") is None

    def test_single_sentence(self):
        result = extract_best_sentence("test", "This is a test sentence.")
        assert result == "This is a test sentence."


# ── extract_answer dispatch branches ────────────────────────────


class TestExtractAnswerDispatch:
    def test_who_question(self):
        result = extract_answer("who wrote the code", "Written by Miroslav Sotek in 2026.")
        assert result is not None
        assert "Miroslav" in result

    def test_how_many_question(self):
        result = extract_answer("how many tests passed", "We ran 250 tests successfully.")
        assert result is not None

    def test_what_percent_question(self):
        result = extract_answer("what percent accuracy", "Accuracy was 81.2% on LOCOMO.")
        assert result == "81.2%"

    def test_generic_no_typed_match(self):
        result = extract_answer("tell me about the project", "No special types here at all.")
        assert result is None


# ── _extract_name_answer single word near query ─────────────────


class TestExtractNameSingleWord:
    def test_single_capitalized_near_query(self):
        result = _extract_name_answer("The project lead is Miroslav.", "who is leading the project")
        assert result is not None


# ── _is_what_percent_question ───────────────────────────────────


class TestIsWhatPercentQuestion:
    def test_percent_word(self):
        from answer_extractor import _is_what_percent_question

        assert _is_what_percent_question("what percent did we get")
        assert _is_what_percent_question("accuracy on benchmark")
        assert not _is_what_percent_question("where is the file")


# ── normalize_number edge cases ────────────────────────────────


class TestNormalizeNumberEdge:
    def test_hundred(self):
        assert normalize_number("one hundred") == "100"

    def test_and_word(self):
        assert normalize_number("twenty and one") == "21"


# ── FakeBackend for LLM tests ─────────────────────────────────


class _FakeBackend:
    """Simple LLM backend stub for testing."""

    def __init__(self, response):
        self._response = response
        self.last_prompt = None
        self.last_kwargs = {}

    def complete(self, prompt, *, max_tokens=200, system=""):
        self.last_prompt = prompt
        self.last_kwargs = {"max_tokens": max_tokens, "system": system}
        if callable(self._response):
            return self._response(prompt)
        return self._response


class _ErrorBackend:
    """Backend that always raises."""

    def complete(self, prompt, **kwargs):
        raise RuntimeError("backend error")


# ── llm_extract_answer ─────────────────────────────────────────


class TestLLMExtractAnswer:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND
        answer_extractor._BACKEND = None

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_none(self):
        from answer_extractor import llm_extract_answer

        assert llm_extract_answer("when", "context about dates") is None

    def test_successful_extraction(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        backend = _FakeBackend("March 15, 2026")
        answer_extractor._BACKEND = backend
        result = llm_extract_answer("when", "context")
        assert result == "March 15, 2026"
        assert "context" in backend.last_prompt
        assert backend.last_kwargs["max_tokens"] == 100

    def test_unknown_answer_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        answer_extractor._BACKEND = _FakeBackend("unknown")
        assert llm_extract_answer("when did X happen", "some context") is None

    def test_i_dont_know_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        answer_extractor._BACKEND = _FakeBackend("I don't know")
        assert llm_extract_answer("question", "paragraph") is None

    def test_none_response_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        answer_extractor._BACKEND = _FakeBackend(None)
        assert llm_extract_answer("question", "paragraph") is None

    def test_backend_error_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        answer_extractor._BACKEND = _ErrorBackend()
        assert llm_extract_answer("question", "paragraph") is None

    def test_paragraph_truncated_to_1000(self):
        import answer_extractor
        from answer_extractor import llm_extract_answer

        backend = _FakeBackend("answer")
        answer_extractor._BACKEND = backend
        llm_extract_answer("q", "x" * 2000)
        # The prompt should contain at most 1000 chars of paragraph
        assert "x" * 1001 not in backend.last_prompt


# ── LLM backend getter/setter ────────────────────────────────


class TestLLMBackendAccessors:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_set_and_get(self):
        from answer_extractor import set_llm_backend, get_llm_backend

        backend = _FakeBackend("test")
        set_llm_backend(backend)
        assert get_llm_backend() is backend

    def test_set_none(self):
        from answer_extractor import set_llm_backend, get_llm_backend

        set_llm_backend(None)
        assert get_llm_backend() is None


# ── LLM prospective queries ───────────────────────────────────


class TestLLMProspectiveQueries:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND
        answer_extractor._BACKEND = None

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_empty(self):
        from answer_extractor import llm_generate_prospective_queries

        assert llm_generate_prospective_queries("paragraph text", "doc.md") == []

    def test_generates_queries(self):
        import answer_extractor
        from answer_extractor import llm_generate_prospective_queries

        answer_extractor._BACKEND = _FakeBackend(
            "What is the LOCOMO score?\nWhen was the benchmark run?\nTiny"
        )
        result = llm_generate_prospective_queries("paragraph", "doc.md")
        assert len(result) == 2  # "Tiny" is <= 5 chars, filtered out

    def test_none_response_returns_empty(self):
        import answer_extractor
        from answer_extractor import llm_generate_prospective_queries

        answer_extractor._BACKEND = _FakeBackend(None)
        assert llm_generate_prospective_queries("paragraph", "doc.md") == []

    def test_backend_error_returns_empty(self):
        import answer_extractor
        from answer_extractor import llm_generate_prospective_queries

        answer_extractor._BACKEND = _ErrorBackend()
        assert llm_generate_prospective_queries("paragraph", "doc.md") == []

    def test_max_8_queries(self):
        import answer_extractor
        from answer_extractor import llm_generate_prospective_queries

        answer_extractor._BACKEND = _FakeBackend(
            "\n".join(f"Query number {i} about something" for i in range(15))
        )
        result = llm_generate_prospective_queries("paragraph", "doc.md")
        assert len(result) <= 8


# ── LLM synthesis ──────────────────────────────────────────────


class TestLLMSynthesizeAnswer:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND
        answer_extractor._BACKEND = None

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_none(self):
        from answer_extractor import llm_synthesize_answer

        assert llm_synthesize_answer("query", ["para1", "para2"]) is None

    def test_synthesizes_answer(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        backend = _FakeBackend("The score was 69.0%")
        answer_extractor._BACKEND = backend
        result = llm_synthesize_answer("what was the score?", ["para about scores"])
        assert result == "The score was 69.0%"

    def test_unknown_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        answer_extractor._BACKEND = _FakeBackend("unknown")
        assert llm_synthesize_answer("query", ["para"]) is None

    def test_none_response_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        answer_extractor._BACKEND = _FakeBackend(None)
        assert llm_synthesize_answer("query", ["para"]) is None

    def test_backend_error_returns_none(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        answer_extractor._BACKEND = _ErrorBackend()
        assert llm_synthesize_answer("query", ["para"]) is None

    def test_hypothetical_prompt(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        backend = _FakeBackend("Yes, they would enjoy it")
        answer_extractor._BACKEND = backend
        result = llm_synthesize_answer("would the user enjoy hiking?", ["likes outdoors"])
        assert result == "Yes, they would enjoy it"
        assert "hypothetical" in backend.last_prompt.lower()

    def test_list_prompt(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        backend = _FakeBackend("hiking, reading, cooking")
        answer_extractor._BACKEND = backend
        result = llm_synthesize_answer("what are the hobbies?", ["hobby info"])
        assert result == "hiking, reading, cooking"
        assert "list all" in backend.last_prompt.lower()

    def test_default_prompt(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        backend = _FakeBackend("The answer is 42")
        answer_extractor._BACKEND = backend
        result = llm_synthesize_answer("how many tests?", ["test count info"])
        assert result == "The answer is 42"
        assert "concise" in backend.last_prompt.lower()

    def test_max_10_paragraphs(self):
        import answer_extractor
        from answer_extractor import llm_synthesize_answer

        backend = _FakeBackend("answer")
        answer_extractor._BACKEND = backend
        paras = [f"paragraph {i}" for i in range(20)]
        llm_synthesize_answer("query", paras)
        # Only first 10 should be in prompt
        assert "[Source 10]" in backend.last_prompt
        assert "[Source 11]" not in backend.last_prompt


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAnswerExtractorPipeline:
    """Full pipeline: query → extract → normalise → match."""

    def test_extract_feeds_normalizer(self):
        from answer_extractor import extract_answer
        from answer_normalizer import normalize_answer

        answer = extract_answer("What accuracy?", "The accuracy reached 88.5% on LOCOMO.")
        assert answer is not None
        normed = normalize_answer(answer)
        assert "88.5" in normed

    def test_extract_feeds_fuzzy_match(self):
        from answer_extractor import extract_answer, fuzzy_match

        answer = extract_answer("When was it?", "The review was on March 15, 2026.")
        assert answer is not None
        assert fuzzy_match(answer, "March 15, 2026")

    def test_rust_and_python_agree(self):
        """Rust extract_answer matches Python on same input."""
        from answer_extractor import extract_answer

        result = extract_answer("When?", "Meeting on 2026-03-15.")
        assert result is not None
        assert "2026-03-15" in result or "March" in result


# ── Duration extraction (Task #27) ──────────────────────────────


class TestExtractDuration:
    def test_days_between_two_dates(self):
        from answer_extractor import extract_duration

        text = "The trip started on 2023-01-01 and ended on 2023-02-01."
        result = extract_duration(text, "how many days between start and end")
        assert result is not None
        assert "31 days" in result

    def test_weeks_between_dates(self):
        from answer_extractor import extract_duration

        text = "Project began 2023-03-01 and shipped 2023-03-22."
        result = extract_duration(text, "how many weeks between start and ship")
        assert result is not None
        assert "3 weeks" in result

    def test_months_between_dates(self):
        from answer_extractor import extract_duration

        text = "Joined on 2023-01-15, left on 2023-04-15."
        result = extract_duration(text, "how many months between joining and leaving")
        assert result is not None
        assert "3 months" in result

    def test_single_date_returns_none(self):
        from answer_extractor import extract_duration

        text = "Event on 2023-01-01 was great."
        assert extract_duration(text, "how many days") is None

    def test_no_dates_returns_none(self):
        from answer_extractor import extract_duration

        text = "No dates here at all."
        assert extract_duration(text, "how many days") is None

    def test_keyword_proximity_scoring(self):
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

    def test_duration_dispatched_from_extract_answer(self):
        """extract_answer dispatches to extract_duration for duration questions."""
        from answer_extractor import extract_answer

        text = "Started 2023-01-01, finished 2023-02-01."
        result = extract_answer("how many days between start and finish", text)
        assert result is not None
        assert "31 days" in result

    def test_is_duration_question(self):
        from answer_extractor import _is_duration_question

        assert _is_duration_question("how many days between x and y")
        assert _is_duration_question("how many weeks did it take")
        assert _is_duration_question("how long between the events")
        assert not _is_duration_question("what is the weather")

    def test_days_dual_format(self):
        """Task #33: days duration returns both exclusive and inclusive."""
        from answer_extractor import extract_duration

        text = "The trip started on 2023-01-01 and ended on 2023-02-01."
        result = extract_duration(text, "how many days between start and end")
        assert result is not None
        assert "31 days" in result
        assert "32 days" in result
        assert "both endpoints" in result

    def test_weeks_dual_format(self):
        from answer_extractor import extract_duration

        text = "Project began 2023-03-01 and shipped 2023-03-22."
        result = extract_duration(text, "how many weeks between start and ship")
        assert result is not None
        assert "3 weeks" in result
        assert "21 days exclusive" in result
        assert "22 inclusive" in result

    def test_months_dual_format(self):
        from answer_extractor import extract_duration

        text = "Joined on 2023-01-15, left on 2023-04-15."
        result = extract_duration(text, "how many months between joining and leaving")
        assert result is not None
        assert "3 months" in result
        assert "90 days exclusive" in result
        assert "91 inclusive" in result


# ── Proximity scoring tuning (Task #35) ─────────────────────────


class TestProximityScoring:
    def test_closer_token_scores_higher(self):
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

    def test_no_tokens_scores_zero(self):
        from answer_extractor import _proximity_score

        s = _proximity_score("unrelated 2023-05-22 content", 10, 10, {"gym", "session"})
        assert s == 0.0

    def test_proximity_picks_right_pair(self):
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

    def test_tighter_window_ignores_distant_matches(self):
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

    def test_yen_paragraph_does_not_panic(self):
        for q in ("how much", "when", "who", "what line"):
            extract_answer(q, self.TOKYO_YEN)  # must not raise

    def test_cjk_paragraph_does_not_panic(self):
        text = "東京の地下鉄は便利です。The Tokyo subway costs around ¥170 per ride."
        for q in ("how much does the subway cost", "when", "who uses"):
            extract_answer(q, text)

    def test_emoji_paragraph_does_not_panic(self):
        text = "The 🎯 target was hit on 2026-04-14 when Alice ran a 26.2 mile marathon."
        assert extract_answer("when was the target hit", text) is not None

    def test_euro_accents_paragraph_does_not_panic(self):
        text = "Zürich tickets cost €4.40 for short trips and Fräulein Müller confirmed."
        extract_answer("how much", text)
        extract_answer("who confirmed", text)

    def test_rust_path_direct(self):
        """Call Rust extractor directly — bypass Python fallbacks."""
        try:
            from remanentia_answer_extractor import extract_answer as r_extract
        except ImportError:
            return  # Rust crate not built (CI without maturin build)
        for q in ("how much", "when", "who"):
            r_extract(q, self.TOKYO_YEN)  # must not panic


class TestExtractDurationDefensive:
    """Guard the date-parse fallback inside extract_duration."""

    def test_invalid_iso_date_is_skipped(self):
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
