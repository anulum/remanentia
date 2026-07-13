# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Recall command handler

"""Execute and render CLI recall requests."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable


def run_recall_command(
    args: argparse.Namespace,
    *,
    setup_llm_backend: Callable[[str], None],
) -> None:
    """Run filtered index search or structured recall for CLI arguments."""
    has_filters = (
        getattr(args, "project", "")
        or getattr(args, "after", "")
        or getattr(args, "before", "")
    )
    if has_filters:
        _run_filtered_recall(args, setup_llm_backend)
        return
    _run_structured_recall(args)


def _run_filtered_recall(
    args: argparse.Namespace,
    setup_llm_backend: Callable[[str], None],
) -> None:
    from memory_index import auto_rebuild_if_needed

    index = auto_rebuild_if_needed(use_gpu=False)
    use_llm = getattr(args, "llm", False) or bool(os.environ.get("REMANENTIA_LLM_ANSWERS"))
    if use_llm:
        setup_llm_backend(getattr(args, "llm_backend", "auto"))
    results = index.search(
        args.query,
        top_k=args.top,
        project=getattr(args, "project", ""),
        after=getattr(args, "after", ""),
        before=getattr(args, "before", ""),
        use_llm=use_llm,
    )
    for result in results:
        print(f"[{result.source}] {result.name} (score={result.score:.3f})")
        if result.answer:
            print(f"  Answer: {result.answer}")
        print(f"  {result.snippet[:200]}")
        print()


def _run_structured_recall(args: argparse.Namespace) -> None:
    from memory_recall import recall

    context = recall(args.query, top_k=args.top, include_content=args.content)
    if args.format == "summary":
        print(context.summary)
        return
    if args.format == "context":
        print(context.to_llm_context())
        return
    if args.format != "json":
        raise ValueError(f"unsupported recall format: {args.format}")
    print(
        json.dumps(
            {
                "query": context.query,
                "trace": context.trace,
                "score": context.trace_score,
                "entities": context.entities,
                "related": context.related_entities,
                "semantic_memories": len(context.semantic_memories),
                "before": context.before,
                "after": context.after,
                "cross_project": context.cross_project,
                "novelty": context.novelty_score,
                "elapsed_ms": context.elapsed_ms,
            },
            indent=2,
        )
    )
