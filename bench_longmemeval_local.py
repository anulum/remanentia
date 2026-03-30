#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: Remanentia — persistent AI memory
# Repository: https://github.com/anulum/remanentia
"""LongMemEval benchmark with local LLM backend (no cloud API needed).

Patches bench_longmemeval to route all LLM calls through LocalLLMBackend.

Usage:
    python bench_longmemeval_local.py --llm --limit 25
    python bench_longmemeval_local.py --llm
    python bench_longmemeval_local.py --arcane --llm
"""

from __future__ import annotations

import sys
import time

from llm_backend import LocalLLMBackend
from answer_extractor import set_llm_backend

# Verify server is running
backend = LocalLLMBackend(timeout=120.0)
if not backend.is_available():
    print("ERROR: Local LLM server not running on localhost:8080")
    sys.exit(1)

set_llm_backend(backend)
print(f"Backend: {backend._base_url} model={backend._model}")

# Ensure --llm is in sys.argv so bench module picks it up
if "--llm" not in sys.argv:
    sys.argv.append("--llm")

# Import benchmark (does NOT auto-run, only __main__ does)
import bench_longmemeval as bench

assert bench._USE_LLM, "bench_longmemeval._USE_LLM must be True"

# Patch: replace OpenAI calls with our local backend
_orig_answer = bench._answer_from_retrieval


def _local_answer_from_retrieval(question, idx, sessions, qtype, use_llm=False):
    """Replacement that uses local LLM backend instead of OpenAI."""
    results = idx.search(question, top_k=10)
    if not results:
        return "I don't have enough information to answer this question."

    if not use_llm:
        if results[0].answer:
            return results[0].answer
        return results[0].snippet[:500]

    context = bench._build_context(question, idx, results, sessions, qtype)
    prompt = bench._type_prompt(question, qtype, context)

    answer = backend.complete(prompt, max_tokens=400)
    if answer and answer.lower() not in ("unknown", "i don't know", "not mentioned"):
        return answer

    # Fallback to regex extraction
    if results[0].answer:
        return results[0].answer
    return results[0].snippet[:500]


def _local_arcane_answer(question, sessions, qtype, haystack_dates=None):
    """Replacement that uses local LLM backend instead of OpenAI."""
    from arcane_retriever import ArcaneRetriever

    ar = ArcaneRetriever(sessions, session_dates=haystack_dates)
    results = ar.retrieve(question, qtype, top_k=15, max_iterations=2)
    if not results:
        return "I don't have enough information to answer this question."

    arcane_context = ar.build_context(question, results, max_facts=15)

    if qtype in (
        "temporal-reasoning",
        "multi-session",
        "knowledge-update",
        "single-session-preference",
    ):
        session_parts = []
        for sess_idx, session in enumerate(sessions):
            turns = [f"[{t['role'].upper()}]: {t['content']}" for t in session]
            session_parts.append(f"=== Session {sess_idx + 1} ===\n" + "\n".join(turns))
        full_context = "\n\n".join(session_parts)

        if qtype == "temporal-reasoning":
            dated_facts = [r for r in results if r.fact.date_mentions]
            timeline = []
            for r in dated_facts:
                for d in r.fact.date_mentions:
                    timeline.append(f"  {d}: {r.fact.text[:150]}")
            timeline.sort()
            timeline_str = "\n".join(timeline) if timeline else "(no dates found)"
            context = (
                f"TIMELINE:\n{timeline_str}\n\nFACTS:\n{arcane_context}\n\nHISTORY:\n{full_context}"
            )
        else:
            context = f"FACTS:\n{arcane_context}\n\nHISTORY:\n{full_context}"
    else:
        context = arcane_context

    prompt = bench._type_prompt(question, qtype, context)
    answer = backend.complete(prompt, max_tokens=400)
    if answer and answer.lower() not in ("unknown", "i don't know", "not mentioned"):
        return answer
    return results[0].fact.text[:500] if results else "Unknown"


# Apply patches
bench._answer_from_retrieval = _local_answer_from_retrieval
bench._arcane_answer = _local_arcane_answer

if __name__ == "__main__":
    t0 = time.time()
    bench.run_benchmark()
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
