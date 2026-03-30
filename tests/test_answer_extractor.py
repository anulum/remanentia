# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for answer_extractor

from __future__ import annotations


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
