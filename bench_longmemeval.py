# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LongMemEval benchmark runner

"""Run LongMemEval benchmark against Remanentia's retrieval pipeline.

Each question has a set of chat sessions (haystack). We:
1. Index the haystack sessions as documents
2. Search for the question
3. Extract/synthesise an answer
4. Output hypothesis file for GPT-judge evaluation

Usage:
    python bench_longmemeval.py                    # regex-only answers
    python bench_longmemeval.py --llm              # LLM synthesis
    python bench_longmemeval.py --limit 50         # first 50 questions
    python bench_longmemeval.py --evaluate          # run GPT-judge after
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

if __name__ == "__main__":  # pragma: no cover — avoid breaking pytest capture
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --full selects the realistic LongMemEval-S haystack (~50 sessions/question,
# ~2 gold) instead of the oracle setting (gold sessions only). On full-S the
# arcane reader cannot dump the whole history (~123K tokens) — it switches to a
# retrieved-context reader (top-ranked sessions only), so retrieval is actually
# exercised. Default stays oracle for backward-comparable numbers.
_USE_FULL = "--full" in sys.argv
_DATA_FILE = "longmemeval_s.json" if _USE_FULL else "longmemeval_oracle.json"
DATA_PATH = Path(__file__).parent / "data" / _DATA_FILE
OUTPUT_PATH = Path(__file__).parent / "data" / "longmemeval_hypotheses.jsonl"

# Retrieved-context reader budget (full-S only). The reader sees the top
# REMANENTIA_FULL_MAX_SESSIONS retrieved sessions, capped at
# REMANENTIA_FULL_CHAR_BUDGET characters of transcript. full-S sessions run
# ~10K chars, so the char budget must fit max_sessions of them (~120K chars
# ≈ 30K tokens, well inside gpt-4o-mini's 128K window) or it silently drops
# retrieved gold sessions before the reader sees them.
_RETRIEVED_CONTEXT = _USE_FULL
_FULL_MAX_SESSIONS = int(os.environ.get("REMANENTIA_FULL_MAX_SESSIONS", "10"))
_FULL_CHAR_BUDGET = int(os.environ.get("REMANENTIA_FULL_CHAR_BUDGET", "120000"))
_FULL_RETRIEVE_K = int(os.environ.get("REMANENTIA_FULL_RETRIEVE_K", "50"))
# Cross-session entity-summary synthesis (P1.3). Off by default: the 2-seed
# full-S ablation (2026-06-29) found no reliable accuracy effect (multi-session
# +2.5 within reader-noise; seed42 +7 did not reproduce). Opt in to experiment.
_SYNTHESIS_ENABLE = os.environ.get("REMANENTIA_SYNTHESIS_ENABLE", "") == "1"
# Lean bi-temporal observe→reflect reader context (W2). Off by default until the
# full-S ablation validates it; when on, the reader sees a lean, dated,
# supersession-resolved observation set INSTEAD of the raw-session dump
# (Mastra OM / Engram mechanism). Opt in: REMANENTIA_LEAN_CONTEXT=1.
_LEAN_CONTEXT = os.environ.get("REMANENTIA_LEAN_CONTEXT", "") == "1"

_USE_LLM = "--llm" in sys.argv
_EVALUATE = "--evaluate" in sys.argv
_USE_ARCANE = "--arcane" in sys.argv
_USE_LOCAL_LLM = "--local-llm" in sys.argv
_LIMIT = None
_PROGRESS_EVERY = 25  # questions between progress prints
_SEED: int | None = None
for i, arg in enumerate(sys.argv):
    if arg == "--limit" and i + 1 < len(sys.argv):
        _LIMIT = int(sys.argv[i + 1])
    if arg == "--progress-every" and i + 1 < len(sys.argv):
        _PROGRESS_EVERY = max(1, int(sys.argv[i + 1]))
    if arg == "--seed" and i + 1 < len(sys.argv):
        _SEED = int(sys.argv[i + 1])

# Hard per-request timeout for every hosted-LLM call. Prevents the
# bench from stalling forever on a hung HTTPS connection (2026-04-17
# incident: poll_schedule_timeout for 25+ min because no timeout was
# set on the client constructor).
_OPENAI_TIMEOUT = float(os.environ.get("REMANENTIA_OPENAI_TIMEOUT", "30"))

# Pin every randomness source Remanentia pulls in. REMANENTIA_SEED env
# or --seed flag overrides the 42 default; result banners print the
# effective seed so the noise envelope can be measured rather than
# guessed. Does not affect hosted-LLM sampling (controlled per-request
# via the ``seed`` and ``temperature`` parameters).
from seed_utils import seed_everything, seed_from_env  # noqa: E402

_EFFECTIVE_SEED = seed_everything(_SEED if _SEED is not None else seed_from_env())


def _hypothesis_complete(prompt: str, max_tokens: int = 400) -> str | None:
    """Generate a hypothesis answer using the configured LLM backend.

    When ``--local-llm`` is set (or ``OPENAI_API_KEY`` is missing),
    routes through :class:`llm_backend.LocalLLMBackend` (default Ollama
    at localhost:11434 running ``gemma3:4b``). Otherwise uses the
    hosted ``gpt-4o-mini`` endpoint directly. Returns ``None`` on
    backend failure.

    The GPT-judge in :func:`run_evaluation` is NOT affected — it stays
    on ``gpt-4o-mini`` via the hosted API so that scores remain
    comparable with previous rounds.
    """
    if _USE_LOCAL_LLM or not os.environ.get("OPENAI_API_KEY"):
        from llm_backend import LocalLLMBackend

        backend = LocalLLMBackend(timeout=120.0)
        if backend.is_available():
            try:
                return cast(str | None, backend.complete(prompt, max_tokens=max_tokens))
            except Exception:  # pragma: no cover — network / runtime errors
                return None
        return None

    try:
        from openai import OpenAI

        client = OpenAI(timeout=_OPENAI_TIMEOUT)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else None
    except Exception as e:  # pragma: no cover — surface the cause
        print(f"[openai error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


def _flatten_sessions(sessions: list[list[dict[str, Any]]]) -> str:
    """Flatten chat sessions into a single searchable text."""
    parts: list[str] = []
    for sess_idx, session in enumerate(sessions):
        lines: list[str] = []
        for turn in session:
            role = str(turn.get("role", "")).upper()
            content = str(turn.get("content", ""))
            lines.append(f"[{role}]: {content}")
        parts.append(f"--- Session {sess_idx + 1} ---\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _build_index_for_question(
    sessions: list[list[dict[str, Any]]],
    haystack_dates: list[str] | None = None,
) -> Any:
    """Build a MemoryIndex over the haystack sessions for one question.

    When *haystack_dates* are provided, each document carries the session
    timestamp and vague date expressions in turn content are resolved
    against that timestamp via :func:`date_normalizer.normalise_in_context`.
    """
    from memory_index import MemoryIndex, Document, _tokenize, _token_counts, _classify_paragraph

    idx = MemoryIndex()
    idx.documents = []
    idx.paragraph_index = []
    idx.paragraph_tokens = []
    idx.paragraph_token_counts = []
    idx.paragraph_types = []
    idx._inverted_index = {}
    idx._df = {}
    idx.idf = {}
    idx._para_lengths = __import__("numpy").array([], dtype=__import__("numpy").float32)
    idx._avg_dl = 1.0
    idx._built = True
    idx._rust_bm25_dirty = True

    import numpy as np
    import math
    from date_normalizer import normalise_in_context

    for sess_idx, session in enumerate(sessions):
        sess_date = (
            haystack_dates[sess_idx] if haystack_dates and sess_idx < len(haystack_dates) else ""
        )

        # Each turn becomes a paragraph
        for turn_idx, turn in enumerate(session):
            content = str(turn.get("content", ""))
            if len(content) < 20:
                continue

            # Resolve vague dates against session timestamp
            if sess_date:
                content = normalise_in_context(content, sess_date)

            role = str(turn.get("role", ""))
            doc_name = f"session_{sess_idx}_turn_{turn_idx}_{role}"
            doc = Document(
                name=doc_name,
                source=f"session_{sess_idx}",
                path=doc_name,
                paragraphs=[content],
                date=sess_date,
                doc_type="conversation",
            )
            doc_idx = len(idx.documents)
            idx.documents.append(doc)

            para_idx = len(idx.paragraph_tokens)
            idx.paragraph_index.append((doc_idx, 0))
            token_list = _tokenize(content)
            tokens = set(token_list)
            token_counts = _token_counts(token_list)
            idx.paragraph_tokens.append(tokens)
            idx.paragraph_token_counts.append(token_counts)
            p_type = _classify_paragraph(content, is_code=False)
            idx.paragraph_types.append(p_type)

            n_total = len(idx.paragraph_tokens)
            for t in tokens:
                idx._df[t] = idx._df.get(t, 0) + 1
                idx.idf[t] = math.log(1 + n_total / (1 + idx._df[t]))
                if t not in idx._inverted_index:
                    idx._inverted_index[t] = []
                idx._inverted_index[t].append(para_idx)

    # Compute para lengths
    idx._para_lengths = np.array([len(t) for t in idx.paragraph_tokens], dtype=np.float32)
    idx._avg_dl = float(np.mean(idx._para_lengths)) if len(idx._para_lengths) > 0 else 1.0

    return idx


def _answer_from_retrieval(
    question: str,
    idx: Any,
    sessions: list[list[dict[str, Any]]],
    qtype: str,
    use_llm: bool = False,
    haystack_dates: list[str] | None = None,
    question_date: str = "",
) -> str:
    """Search index and extract/synthesise answer with type-specific strategy."""
    results = idx.search(question, top_k=10)

    if not results:
        return "I don't have enough information to answer this question."

    if not use_llm:
        if results[0].answer:
            return str(results[0].answer)
        return str(results[0].snippet[:500])

    # Build context based on question type
    context = _build_context(
        question,
        idx,
        results,
        sessions,
        qtype,
        haystack_dates=haystack_dates,
        question_date=question_date,
    )

    # Get type-specific prompt
    prompt = _type_prompt(question, qtype, context)
    answer = _hypothesis_complete(prompt, max_tokens=400)
    if answer and answer.lower() not in ("unknown", "i don't know", "not mentioned"):
        return answer

    if results[0].answer:
        return str(results[0].answer)
    return str(results[0].snippet[:500])


def _build_context(
    question: str,
    idx: Any,
    results: list[Any],
    sessions: list[list[dict[str, Any]]],
    qtype: str,
    haystack_dates: list[str] | None = None,
    question_date: str = "",
) -> str:
    """Build LLM context adapted to question type."""

    if qtype in (
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    ):
        # Sort sessions chronologically so LLM sees oldest first
        indexed_sessions = list(enumerate(sessions))
        if haystack_dates and len(haystack_dates) == len(sessions):
            indexed_sessions.sort(
                key=lambda x: haystack_dates[x[0]] if x[0] < len(haystack_dates) else ""
            )

        # Cross-session + temporal + knowledge-update: full session content
        parts: list[str] = []
        for order, (orig_idx, session) in enumerate(indexed_sessions):
            sess_date = (
                haystack_dates[orig_idx]
                if haystack_dates and orig_idx < len(haystack_dates)
                else ""
            )
            header = f"=== Session {order + 1}"
            if sess_date:
                header += f" ({sess_date})"
            header += " ==="
            turns: list[str] = []
            for turn in session:
                role = str(turn.get("role", "")).upper()
                turns.append(f"[{role}]: {turn.get('content', '')}")
            parts.append(header + "\n" + "\n".join(turns))
        context = "\n\n".join(parts)

        # For temporal: prepend extracted dates + TReMu computed answer + TODAY anchor
        if qtype == "temporal-reasoning":
            dates_info = _extract_temporal_facts(sessions)
            if dates_info:
                context = f"EXTRACTED DATES AND EVENTS:\n{dates_info}\n\n{context}"

            # TReMu: pre-compute temporal arithmetic (Task #34: pass question_date)
            tremu_answer = _tremu_precompute(
                question, sessions, haystack_dates, question_date=question_date
            )
            if tremu_answer:
                context = f"COMPUTED ANSWER: {tremu_answer}\n\n{context}"

            # Task #34: anchor "today" so LLM doesn't hallucinate reference
            if question_date:
                context = f"TODAY (question was asked on): {question_date}\n\n{context}"

        return context

    # Single-session types: focused BM25 results (no noise from surrounding turns)
    paras: list[str] = []
    for r in results[:5]:
        doc_idx = next((di for di, d in enumerate(idx.documents) if d.name == r.name), None)
        if doc_idx is not None:
            full_text = idx.documents[doc_idx].paragraphs[r.paragraph_idx]
            paras.append(full_text[:1500])
        else:
            paras.append(str(r.snippet))

    return "\n\n---\n\n".join(paras)


def _extract_temporal_facts(sessions: list[list[dict[str, Any]]]) -> str:
    """Extract dates and temporal references from sessions for pre-computation."""
    import re

    events: list[str] = []
    date_patterns = [
        r"(?:on\s+|since\s+|from\s+)?(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
        r"(\d{1,2}/\d{1,2}/\d{2,4})",
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
    ]

    for sess_idx, session in enumerate(sessions):
        for turn in session:
            if turn.get("role") != "user":
                continue
            text = str(turn.get("content", ""))
            for pattern in date_patterns:
                for m in re.finditer(pattern, text):
                    date_str = m.group(1) if m.lastindex else m.group()
                    # Get surrounding context
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 60)
                    context = text[start:end].strip()
                    events.append(f'Session {sess_idx + 1}: "{context}" [date: {date_str}]')

    if not events:
        return ""

    return "\n".join(events[:20])


def _tremu_precompute(
    question: str,
    sessions: list[list[dict[str, Any]]],
    haystack_dates: list[str] | None = None,
    question_date: str = "",
) -> str | None:
    """Run TReMu temporal arithmetic on session events before LLM.

    Extracts events from all sessions using the temporal graph, then
    executes :func:`temporal_code_execute` to compute durations,
    ordering, and counts via Python arithmetic instead of LLM guessing.
    When *question_date* is provided, it is used as the "today" reference
    for "how many X ago" queries (Task #34).
    """
    from temporal_graph import TemporalGraph, temporal_code_execute

    tg = TemporalGraph()
    for sess_idx, session in enumerate(sessions):
        sess_date = ""
        if haystack_dates and sess_idx < len(haystack_dates):
            sess_date = haystack_dates[sess_idx]
        text = " ".join(str(t.get("content", "")) for t in session if t.get("role") == "user")
        events = tg.extract_events(text, f"session_{sess_idx}", reference_date=sess_date)
        tg.add_events(events)

    if not tg.events:
        return None

    return temporal_code_execute(question, tg.events, question_date=question_date)


def _type_prompt(question: str, qtype: str, context: str) -> str:
    """Generate type-specific LLM prompt for answer synthesis."""

    if qtype == "temporal-reasoning":
        return (
            "Answer this temporal question about a conversation history.\n\n"
            "RULES:\n"
            "- If the context contains 'TODAY (question was asked on): ...', "
            "use THAT date as today. NEVER invent a different 'today'.\n"
            "- If the context contains a 'COMPUTED ANSWER' line, trust it unless "
            "it clearly contradicts the question.\n"
            "- 'How many days between A and B': count calendar days from A to B "
            "(exclusive of start, inclusive of end; both forms may be acceptable)\n"
            "- 'How many days/weeks/months ago': measure from the event date to TODAY\n"
            "- 'Which came first': compare the dates\n"
            "- 'What was the first/last X': find the earliest/latest dated event matching X\n"
            "- If counting days: show the two dates and the subtraction\n"
            "- START with the direct answer\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    if qtype == "multi-session":
        return (
            "You are answering a question that requires combining information from multiple conversation sessions.\n\n"
            "Read ALL sessions carefully. The answer may span multiple sessions.\n\n"
            "ABSTAIN WHEN APPROPRIATE. Before answering, identify every "
            "specific noun phrase in the question (product names, places, "
            "numbers-with-units like '30-gallon tank', named devices like "
            "'iPad'). If any such noun does NOT actually appear in the "
            "conversation context, do not guess or approximate — reply "
            "with 'The information provided is not enough' and state what "
            "was missing. Fabricating a plausible answer is always wrong.\n\n"
            "If the context contains a 'COMPUTED TOTAL:' or 'COMPUTED COUNT:' "
            "line, trust it unless it clearly contradicts the question — the "
            "aggregation has been double-checked by deterministic code.\n"
            "If the context contains an 'ENTITY SUMMARY' block, it is the "
            "consolidated, deduplicated cross-session record for each named "
            "entity (oldest→newest); prefer it when combining facts across "
            "sessions.\n"
            "If the question asks 'how many' or 'total', give a single integer "
            "as the first sentence. Do not hedge with 'at least'.\n\n"
            f"Conversation sessions:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give the answer directly in 1-2 sentences."
        )

    if qtype == "knowledge-update":
        return (
            "You are answering a question where information may have been updated over time.\n\n"
            "IMPORTANT: If multiple answers exist across sessions, use the MOST RECENT one.\n"
            "The user's latest statement supersedes earlier ones.\n\n"
            "ABSTAIN WHEN APPROPRIATE. If the question mentions a specific "
            "noun (product, place, device, named entity) that does NOT "
            "appear in the conversation, reply with 'The information "
            "provided is not enough' instead of fabricating.\n\n"
            "If the context contains a 'COMPUTED TOTAL:' or 'COMPUTED COUNT:' "
            "line, trust it unless it clearly contradicts the question.\n"
            "If the context contains an 'ENTITY SUMMARY' block, the statement "
            "marked '(most recent)' is the latest value for that entity — use "
            "it as the up-to-date answer.\n\n"
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with the most up-to-date information. Be concise (1-2 sentences)."
        )

    if qtype == "single-session-preference":
        return (
            "The user is asking a general question. Based on their conversation history, "
            "provide a personalized answer that reflects their specific preferences, "
            "interests, and personal details mentioned in the conversation.\n\n"
            "Do NOT give a generic answer. Use the user's actual stated preferences.\n\n"
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give a personalized response (2-3 sentences) that reflects the user's specific situation."
        )

    # single-session-user, single-session-assistant
    return (
        "Based ONLY on the following conversation excerpts, answer the question.\n"
        "Be concise (1-2 sentences). If the information is not present, say 'unknown'.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )


def _arcane_answer(
    question: str,
    sessions: list[list[dict[str, Any]]],
    qtype: str,
    haystack_dates: list[str] | None = None,
    question_date: str = "",
    haystack_session_ids: list[str] | None = None,
    answer_session_ids: list[str] | None = None,
    retrieval_diagnostics: dict[str, object] | None = None,
) -> str:
    """Answer using ArcaneRetriever pipeline (v0.4.0 architecture).

    When *haystack_dates* are provided, the C4 date normaliser resolves
    vague expressions ("3 weeks ago") against each session's timestamp.
    When *question_date* is provided, it is anchored as "today" in the
    LLM prompt for temporal-reasoning questions (Task #34). When full-S
    retrieved-context mode is active, *retrieval_diagnostics* is populated with
    the selected/missing answer-session evidence written to the hypothesis
    artefact.
    """
    from arcane_retriever import ArcaneRetriever

    ar = ArcaneRetriever(sessions, session_dates=haystack_dates)
    # On full-S the reader is fed only the top retrieved sessions, so retrieve
    # enough facts for the session selector to have candidates beyond the few
    # the oracle setting needs.
    _top_k = _FULL_RETRIEVE_K if _RETRIEVED_CONTEXT else 15
    results = ar.retrieve(question, qtype, top_k=_top_k, max_iterations=2)

    if not results:
        return "I don't have enough information to answer this question."

    # Build context from retrieved facts
    # Chronological sort only for temporal questions (Task #32 fix).
    # Relevance order (RRF) wins for multi-session, knowledge-update, preference.
    arcane_context = ar.build_context(
        question,
        results,
        max_facts=15,
        sort_chronologically=(qtype == "temporal-reasoning"),
    )

    # For temporal/multi-session/knowledge-update: also include full sessions
    # (the LLM needs full context to reason over, facts provide ranking signal)
    if qtype in (
        "temporal-reasoning",
        "multi-session",
        "knowledge-update",
        "single-session-preference",
    ):
        if _RETRIEVED_CONTEXT:
            # Full-S: feed only the top retrieved sessions (the whole haystack
            # is ~123K tokens and cannot be dumped) — this is where retrieval
            # actually matters. select_sessions renders them chronologically.
            from retrieved_context import build_selection_diagnostics, select_sessions

            retrieved = select_sessions(
                results,
                sessions,
                session_dates=haystack_dates,
                max_sessions=_FULL_MAX_SESSIONS,
                char_budget=_FULL_CHAR_BUDGET,
            )
            if retrieval_diagnostics is not None:
                diagnostics = build_selection_diagnostics(
                    retrieved,
                    haystack_session_ids=haystack_session_ids,
                    answer_session_ids=answer_session_ids,
                )
                retrieval_diagnostics.update(diagnostics.as_json_dict())
            full_context = retrieved.session_text
        else:
            # Oracle: dump every haystack session (gold-only, tiny) oldest-first.
            indexed_sessions = list(enumerate(sessions))
            if haystack_dates and len(haystack_dates) == len(sessions):
                indexed_sessions.sort(
                    key=lambda x: haystack_dates[x[0]] if x[0] < len(haystack_dates) else ""
                )

            session_parts: list[str] = []
            for order, (orig_idx, session) in enumerate(indexed_sessions):
                sess_date = (
                    haystack_dates[orig_idx]
                    if haystack_dates and orig_idx < len(haystack_dates)
                    else ""
                )
                header = f"=== Session {order + 1}"
                if sess_date:
                    header += f" ({sess_date})"
                header += " ==="
                turns: list[str] = []
                for turn in session:
                    role = str(turn.get("role", "")).upper()
                    turns.append(f"[{role}]: {turn.get('content', '')}")
                session_parts.append(header + "\n" + "\n".join(turns))
            full_context = "\n\n".join(session_parts)

        if qtype == "temporal-reasoning":
            # Prepend date-rich facts from retrieval + timeline
            dated_facts = [r for r in results if r.fact.date_mentions]
            timeline = []
            for r in dated_facts:
                for d in r.fact.date_mentions:
                    timeline.append(f"  {d}: {r.fact.text[:150]}")
            timeline.sort()
            timeline_str = "\n".join(timeline) if timeline else "(no explicit dates found)"

            # Task #34: anchor "today" so LLM doesn't hallucinate reference
            today_header = ""
            if question_date:
                today_header = f"TODAY (question was asked on): {question_date}\n\n"

            context = (
                f"{today_header}"
                f"TIMELINE OF DATED EVENTS:\n{timeline_str}\n\n"
                f"RETRIEVED FACTS (ranked by relevance):\n{arcane_context}\n\n"
                f"FULL CONVERSATION HISTORY:\n{full_context}"
            )
        else:
            # Non-temporal arcane-path qtypes (multi-session /
            # knowledge-update / single-session-preference). Try to
            # pre-compute a TOTAL: if the retrieved facts contain at
            # least two same-unit labelled numbers and the question
            # asks for a sum, we prepend a COMPUTED TOTAL line that
            # the LLM is told to trust. Addresses R11 multi-session
            # ARITHMETIC_ERROR and MISSING_AGGREGATION failures.
            from aggregate_precompute import precompute_aggregation

            agg_context = f"{arcane_context}\n\n{full_context}"
            agg = precompute_aggregation(question, agg_context, qtype=qtype)
            precompute_header = f"{agg.message}\n\n" if agg else ""

            # Reflection / entity-summary synthesis (P1.3): consolidate the
            # retrieved facts about the question's named entities into one
            # deduplicated, chronologically-ordered block so the reader does
            # not have to re-aggregate them from ~10 raw sessions. Operates
            # only over already-retrieved facts → no extra API call.
            synthesis_header = ""
            if _SYNTHESIS_ENABLE:
                from cross_session_synthesis import synthesise

                synth = synthesise(question, [r.fact for r in results], qtype=qtype)
                if synth:
                    synthesis_header = f"{synth.message}\n\n"

            # W2: when lean mode is on, the reader sees a lean, dated,
            # supersession-resolved observation set INSTEAD of the raw-session
            # dump (Mastra OM / Engram). Falls back to the raw history when the
            # lean set is empty, so it never starves the reader.
            history_block = f"FULL CONVERSATION HISTORY:\n{full_context}"
            if _LEAN_CONTEXT:
                from lean_context import build_lean_context

                lean = build_lean_context([r.fact for r in results])
                if lean:
                    history_block = lean.rendered

            context = (
                f"{precompute_header}"
                f"{synthesis_header}"
                f"RETRIEVED FACTS (ranked by relevance):\n{arcane_context}\n\n"
                f"{history_block}"
            )
    else:
        # Single-session: use retrieved facts as focused context
        context = arcane_context

    prompt = _type_prompt(question, qtype, context)
    answer = _hypothesis_complete(prompt, max_tokens=400)
    if answer and answer.lower() not in ("unknown", "i don't know", "not mentioned"):
        return answer

    # Fallback: return top fact text
    return results[0].fact.text[:500] if results else "Unknown"


def run_benchmark() -> None:
    """Run LongMemEval and output hypothesis file."""
    print(f"Loading LongMemEval from {DATA_PATH}...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = cast(list[dict[str, Any]], json.load(f))

    if _LIMIT:
        data = data[:_LIMIT]

    print(f"Questions: {len(data)}")
    print(f"Dataset: {_DATA_FILE} ({'full-S retrieval' if _USE_FULL else 'oracle'})")
    if _RETRIEVED_CONTEXT:
        print(
            f"Reader: retrieved-context (top {_FULL_MAX_SESSIONS} sessions, "
            f"{_FULL_CHAR_BUDGET} char budget, retrieve k={_FULL_RETRIEVE_K})"
        )
    print(f"LLM mode: {_USE_LLM}")
    print(f"Arcane mode: {_USE_ARCANE}")
    print(f"Local LLM mode: {_USE_LOCAL_LLM} (Ollama gemma3:4b)")
    print(f"Progress every: {_PROGRESS_EVERY} questions")
    print(f"Hosted-LLM timeout: {_OPENAI_TIMEOUT:.0f} s per request")
    print(f"Seed: {_EFFECTIVE_SEED}")
    print()

    type_correct: defaultdict[str, int] = defaultdict(int)
    type_total: defaultdict[str, int] = defaultdict(int)

    hypotheses: list[dict[str, object]] = []
    t0 = time.monotonic()

    for i, item in enumerate(data):
        qid = str(item["question_id"])
        qtype = str(item["question_type"])
        question = str(item["question"])
        gold = str(item["answer"])

        item_dates = cast(list[str] | None, item.get("haystack_dates"))
        item_question_date = str(item.get("question_date", ""))
        item_session_ids = cast(list[str] | None, item.get("haystack_session_ids"))
        item_answer_session_ids = cast(list[str] | None, item.get("answer_session_ids"))
        haystack_sessions = cast(list[list[dict[str, Any]]], item["haystack_sessions"])

        q_start = time.monotonic()
        retrieval_diagnostics: dict[str, object] = {}
        if _USE_ARCANE:
            # Hybrid: ArcaneRetriever for hard categories, legacy for single-session
            if qtype in (
                "temporal-reasoning",
                "multi-session",
                "knowledge-update",
                "single-session-preference",
            ):
                hypothesis = _arcane_answer(
                    question,
                    haystack_sessions,
                    qtype,
                    haystack_dates=item_dates,
                    question_date=item_question_date,
                    haystack_session_ids=item_session_ids,
                    answer_session_ids=item_answer_session_ids,
                    retrieval_diagnostics=retrieval_diagnostics,
                )
            else:
                # Single-session factoid: legacy BM25 pipeline (full paragraphs) + GPT-4o-mini
                idx = _build_index_for_question(haystack_sessions, haystack_dates=item_dates)
                hypothesis = _answer_from_retrieval(
                    question,
                    idx,
                    haystack_sessions,
                    qtype,
                    use_llm=True,
                    haystack_dates=item_dates,
                    question_date=item_question_date,
                )
        else:
            # Legacy pipeline
            idx = _build_index_for_question(haystack_sessions, haystack_dates=item_dates)
            hypothesis = _answer_from_retrieval(
                question,
                idx,
                haystack_sessions,
                qtype,
                use_llm=_USE_LLM,
                haystack_dates=item_dates,
                question_date=item_question_date,
            )

        hypothesis_record: dict[str, object] = {
            "question_id": qid,
            "hypothesis": hypothesis,
        }
        if retrieval_diagnostics:
            hypothesis_record["retrieval_diagnostics"] = retrieval_diagnostics
        hypotheses.append(hypothesis_record)

        # Loud per-question heartbeat so a slow item is visible before the
        # next progress milestone.  (2026-04-17T0711: a single stalled call
        # hid 25+ min of "no output" behind the default per-25 print.)
        q_elapsed = time.monotonic() - q_start
        if q_elapsed > max(_OPENAI_TIMEOUT, 10.0):
            print(
                f"  [slow Q] {qid} ({qtype}) took {q_elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        type_total[qtype] += 1

        # Quick local check (not authoritative — GPT-judge is the real eval)
        gold_lower = gold.lower().strip()
        hyp_lower = hypothesis.lower().strip()
        local_match = (
            gold_lower in hyp_lower
            or hyp_lower in gold_lower
            or _fuzzy_overlap(gold_lower, hyp_lower) > 0.6
        )
        if local_match:
            type_correct[qtype] += 1

        if (i + 1) % _PROGRESS_EVERY == 0 or i == len(data) - 1:
            elapsed = time.monotonic() - t0
            total_correct = sum(type_correct.values())
            total_done = i + 1
            pct = total_correct / total_done * 100
            eta = (elapsed / total_done) * (len(data) - total_done)
            print(
                f"[{total_done}/{len(data)}] "
                f"local_match={total_correct}/{total_done} ({pct:.1f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t0

    # Save hypotheses
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"\nHypotheses saved to {OUTPUT_PATH}")

    # Print local results (indicative, not authoritative)
    print(f"\n{'=' * 60}")
    print(f"LongMemEval Local Results (fuzzy match, NOT GPT-judge)")
    print(f"{'=' * 60}")
    total_correct = sum(type_correct.values())
    total = sum(type_total.values())
    print(f"Overall: {total_correct}/{total} ({total_correct / total * 100:.1f}%)")
    print()
    for qtype in sorted(type_total.keys()):
        c = type_correct[qtype]
        t = type_total[qtype]
        print(f"  {qtype:30s}: {c:3d}/{t:3d} ({c / t * 100:.1f}%)")
    print(f"\nTime: {elapsed:.1f}s ({elapsed / len(data) * 1000:.0f}ms/question)")
    print(f"\nNOTE: These are local fuzzy-match scores. Run with --evaluate")
    print(f"for authoritative GPT-judge scoring.")


def _fuzzy_overlap(a: str, b: str) -> float:
    """Token overlap ratio between two strings."""
    tokens_a = set(re.findall(r"\w{3,}", a))
    tokens_b = set(re.findall(r"\w{3,}", b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(len(tokens_a), len(tokens_b))


def run_evaluation() -> None:
    """Run GPT-judge evaluation on saved hypotheses (GPT-4o-mini judge)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY required for GPT-judge evaluation")
        return

    client = OpenAI(api_key=api_key, timeout=_OPENAI_TIMEOUT)

    with open(DATA_PATH, encoding="utf-8") as f:
        references = cast(list[dict[str, Any]], json.load(f))
    qid_to_ref: dict[str, dict[str, Any]] = {str(r["question_id"]): r for r in references}

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        hypotheses = cast(list[dict[str, Any]], [json.loads(line) for line in f if line.strip()])

    judge_model = "gpt-4o-mini"
    judge_max_tokens = 10
    print(f"Evaluating {len(hypotheses)} hypotheses with {judge_model} as judge...")

    type_scores: defaultdict[str, list[int]] = defaultdict(list)
    results: list[dict[str, Any]] = []

    for i, hyp in enumerate(hypotheses):
        qid = str(hyp["question_id"])
        ref = qid_to_ref.get(qid)
        if not ref:
            continue

        qtype = str(ref["question_type"])
        question = str(ref["question"])
        gold = str(ref["answer"])
        response = str(hyp["hypothesis"])

        prompt = _judge_prompt(qtype, question, gold, response)

        usage: object | None = None
        started = time.perf_counter()
        try:
            msg = client.chat.completions.create(
                model=judge_model,
                max_tokens=judge_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = msg.choices[0].message.content
            judge_answer = content.strip().lower() if isinstance(content, str) else ""
            correct = "yes" in judge_answer
            usage = getattr(msg, "usage", None)
        except Exception as e:
            print(f"  Judge error on {qid}: {e}")
            correct = False
        judge_latency_ms = (time.perf_counter() - started) * 1000.0

        from benchmark_evidence import build_judge_evidence

        type_scores[qtype].append(1 if correct else 0)
        results.append(
            {
                **hyp,
                "judge_label": correct,
                "question_type": qtype,
                **build_judge_evidence(
                    prompt,
                    model=judge_model,
                    max_tokens=judge_max_tokens,
                    latency_ms=judge_latency_ms,
                    usage=usage,
                ),
            }
        )

        if (i + 1) % max(_PROGRESS_EVERY, 25) == 0:
            total = sum(sum(v) for v in type_scores.values())
            done = sum(len(v) for v in type_scores.values())
            print(
                f"  [{done}/{len(hypotheses)}] correct={total}/{done} ({total / done * 100:.1f}%)",
                flush=True,
            )

    # Final results
    print(f"\n{'=' * 60}")
    print("LongMemEval Results (LLM judge)")
    print(f"{'=' * 60}")
    all_scores = [s for scores in type_scores.values() for s in scores]
    print(
        f"Overall: {sum(all_scores)}/{len(all_scores)} ({sum(all_scores) / len(all_scores) * 100:.1f}%)"
    )
    print()
    for qtype in sorted(type_scores.keys()):
        scores = type_scores[qtype]
        print(
            f"  {qtype:30s}: {sum(scores):3d}/{len(scores):3d} ({sum(scores) / len(scores) * 100:.1f}%)"
        )

    # Save results
    results_path = OUTPUT_PATH.with_suffix(".results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {results_path}")

    # Answer-session coverage diagnostic (audit R3). For each failed
    # multi-session / knowledge-update / temporal question, approximate
    # how many of the gold answer sessions actually surfaced in the
    # model hypothesis. Over-approximates (a session is "covered" if
    # any of its 4+ letter content tokens appear in the hypothesis)
    # but the histogram tells us whether failures concentrate on low-
    # coverage retrieval vs high-coverage-but-wrong synthesis.
    _coverage_report(results, qid_to_ref)
    from full_s_diagnostics import print_full_s_diagnostic_report

    print_full_s_diagnostic_report(results)


def _coverage_tokens(text: str) -> set[str]:
    """Lowercased 4+ letter content tokens from *text* (coverage diagnostic)."""
    import re

    return {t.lower() for t in re.findall(r"\b[A-Za-z]{4,}\b", text)}


def compute_coverage_buckets(
    results: list[dict[str, Any]], oracle: dict[str, dict[str, Any]]
) -> dict[str, list[float]]:
    """Per-qtype answer-session coverage ratios for failed questions.

    A session is counted as "covered" when at least two of its 4+
    letter content tokens also appear in the hypothesis. Pure
    approximation — the number tells you whether failures are
    retrieval-side (low coverage) or synthesis-side (high coverage).
    """
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get("judge_label"):
            continue
        ref = oracle.get(str(r["question_id"]))
        if not ref:
            continue
        answer_sessions = ref.get("answer_session_ids", [])
        if not answer_sessions:
            continue
        hypothesis_toks = _coverage_tokens(str(r.get("hypothesis", "")))
        if not hypothesis_toks:
            continue

        sessions = ref.get("haystack_sessions", [])
        session_id_to_idx = {
            str(sid): i for i, sid in enumerate(ref.get("haystack_session_ids", []))
        }
        covered = 0
        for ans_sid in answer_sessions:
            sess_idx = session_id_to_idx.get(str(ans_sid))
            if sess_idx is None or sess_idx >= len(sessions):
                continue
            sess_toks = _coverage_tokens(
                " ".join(str(turn.get("content", "")) for turn in sessions[sess_idx])
            )
            overlap = len(sess_toks & hypothesis_toks)
            if overlap >= 2:
                covered += 1
        buckets[str(r["question_type"])].append(covered / len(answer_sessions))
    return buckets


def _coverage_report(results: list[dict[str, Any]], oracle: dict[str, dict[str, Any]]) -> None:
    """Print the R3 answer-session coverage histogram per qtype."""
    buckets = compute_coverage_buckets(results, oracle)
    if not buckets:
        return
    print(f"\n{'=' * 60}")
    print("Answer-session coverage on failures (audit R3)")
    print(f"{'=' * 60}")
    print(f"{'qtype':<32}{'n':>4}{'mean':>8}{'<=50%':>8}")
    for qtype in sorted(buckets):
        cov = buckets[qtype]
        mean = sum(cov) / len(cov)
        low = sum(1 for c in cov if c <= 0.5)
        print(f"{qtype:<32}{len(cov):>4}{mean:>8.1%}{low:>8}")


def _judge_prompt(task: str, question: str, answer: str, response: str) -> str:
    """Generate judge prompt matching LongMemEval's evaluation protocol."""
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the "
            "required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal "
            "information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        return (
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )


def _parse_cli() -> None:
    """Parse argv via argparse when invoked as ``__main__``.

    The module-level ``sys.argv in`` probes at lines 41-54 stay as the
    source of truth so existing subprocess-based tests keep working,
    but this function gives ``python bench_longmemeval.py --help``
    a real help banner instead of silently running the benchmark
    (2026-04-17T0711 incident: ``--help`` fell through to a no-flag
    bench run that crashed on the first multi-byte paragraph).

    Flag semantics match the legacy membership checks exactly; any
    divergence would silently disagree with the tests. Validation
    (positive limits, non-negative seeds) happens here so operators
    get a clear error rather than a later exception.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="bench_longmemeval.py",
        description="Run the LongMemEval benchmark or judge prior outputs.",
    )
    p.add_argument(
        "--llm", action="store_true", help="use the hosted GPT-4o-mini endpoint for synthesis"
    )
    p.add_argument(
        "--evaluate", action="store_true", help="after generation, run GPT-judge on outputs"
    )
    p.add_argument("--arcane", action="store_true", help="use ArcaneRetriever hybrid pipeline")
    p.add_argument("--local-llm", action="store_true", help="route synthesis through local Ollama")
    p.add_argument("--limit", type=int, metavar="N", help="cap question count at N")
    p.add_argument(
        "--progress-every",
        type=int,
        metavar="N",
        help=f"print progress every N questions (default {_PROGRESS_EVERY})",
    )
    p.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help=f"pin RNG seed (default {_EFFECTIVE_SEED} or $REMANENTIA_SEED)",
    )
    # argparse walks sys.argv[1:]; unknown args (e.g. from test subprocess
    # probes) are tolerated so the legacy flag set keeps working.
    p.parse_known_args()


if __name__ == "__main__":
    _parse_cli()
    run_benchmark()
    if _EVALUATE:
        run_evaluation()
