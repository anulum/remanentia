# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Arcane Sapience — Persistent neural memory for AI agents.

A spiking neural network (LIF + STDP) that accumulates experience
across sessions. Not a vector database — a living network whose
synaptic weights encode associative relationships.

Quick start::

    from arcane_sapience import retrieve, drop_stimulus, heartbeat

    # Query the memory
    results = retrieve("disruption prediction plasma")

    # Feed new information
    drop_stimulus("Fixed ruff lint in scpn-control", source="claude")

    # Register agent presence
    heartbeat("my-agent", project="my-project")
"""

__version__ = "0.1.0"

try:  # pragma: no cover
    from .retrieve import retrieve, retrieve_context, retrieval_history
    from .retrieve import related_traces, query_suggestions, trace_summaries, chunk_traces
    from .snn_daemon import drop_stimulus, heartbeat
    from .skill_extractor import extract_skills, query_skills, load_skills
    from .cognitive_snapshot import snapshot_save, snapshot_load, continuity_score, reconstruction_prompt
    from .active_retrieval import consult_memory, decision_guard
except ImportError:
    from retrieve import retrieve, retrieve_context, retrieval_history
    from retrieve import related_traces, query_suggestions, trace_summaries, chunk_traces
    from snn_daemon import drop_stimulus, heartbeat
    from skill_extractor import extract_skills, query_skills, load_skills
    from cognitive_snapshot import snapshot_save, snapshot_load, continuity_score, reconstruction_prompt
    from active_retrieval import consult_memory, decision_guard

__all__ = [
    "retrieve",
    "retrieve_context",
    "retrieval_history",
    "related_traces",
    "query_suggestions",
    "trace_summaries",
    "chunk_traces",
    "drop_stimulus",
    "heartbeat",
    "extract_skills",
    "query_skills",
    "load_skills",
    "snapshot_save",
    "snapshot_load",
    "continuity_score",
    "reconstruction_prompt",
    "consult_memory",
    "decision_guard",
]
