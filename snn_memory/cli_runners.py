# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory run adapter

"""Pinned-model run adapter: encode a corpus, train, probe and orchestrate controls.

Every entry point here loads the git-ignored pinned sentence-embedding checkpoint
(or spawns fresh processes that do), so this module is the model adapter/process
responsibility split out of :mod:`snn_memory.cli`. It is excluded from the default
CI-core coverage denominator (the model is absent in CI) and is covered by its own
    required model-provisioned report — the explicit
    ``tests/model_gates/snn_memory_cli_runners_gate.py`` and installed/source-tree
    model gates — never by omission.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from file_utils import atomic_write_json
from snn_memory.checkpoint import load_checkpoint, save_checkpoint
from snn_memory.cli_io import load_corpus, read_json
from snn_memory.contracts import EncoderConfig, ModelConfig, ProbeConfig, TrainConfig
from snn_memory.experiment import decision_report, evaluate_condition
from snn_memory.probe import probe_checkpoint
from snn_memory.sentence_encoder import LocalSentenceEncoder, encode_trace
from snn_memory.state import FloatArray
from snn_memory.trainer import train_memories


def _encode_corpus(
    manifest_path: Path,
    model: ModelConfig,
    encoder_config: EncoderConfig,
    input_current: float,
) -> tuple[list[str], list[FloatArray], LocalSentenceEncoder, str]:
    """Encode every corpus source through the pinned model into spike currents."""
    labels, texts, checkpoint_path, expected_digest, corpus_digest = load_corpus(manifest_path)
    sentence_encoder = LocalSentenceEncoder(checkpoint_path)
    if sentence_encoder.digest != expected_digest:
        raise ValueError("corpus manifest encoder digest mismatch")
    sequences = [
        encode_trace(
            text,
            sentence_encoder,
            model,
            encoder_config,
            input_current=input_current,
        ).currents
        for text in texts
    ]
    return labels, sequences, sentence_encoder, corpus_digest


def run_train(args: argparse.Namespace) -> int:
    """Encode the corpus, train online and persist a frozen checkpoint bundle."""
    config = read_json(args.config)
    model = ModelConfig(**config.get("model", {}))
    encoder_config = EncoderConfig(**config.get("encoder", {}))
    train = TrainConfig(**config.get("train", {}))
    labels, sequences, sentence_encoder, corpus_digest = _encode_corpus(
        args.corpus_manifest, model, encoder_config, train.input_current
    )
    resume = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume is not None:
        metadata = resume.manifest["metadata"]
        if resume.model != model or resume.labels != tuple(labels):
            raise ValueError("resume checkpoint model or labels differ")
        if metadata["encoder"] != encoder_config.__dict__:
            raise ValueError("resume checkpoint encoder configuration differs")
        if metadata["encoder_digest"] != sentence_encoder.digest:
            raise ValueError("resume checkpoint encoder digest differs")
        if metadata["corpus_digest"] != corpus_digest:
            raise ValueError("resume checkpoint corpus digest differs")
        if metadata["seed"] != train.seed or metadata["input_current"] != train.input_current:
            raise ValueError("resume checkpoint training seed or input current differs")
        completed = int(metadata["epochs_completed"])
        if train.epochs <= completed:
            raise ValueError("resume target epochs must exceed completed epochs")
    else:
        completed = 0
    result = train_memories(
        sequences,
        labels,
        model,
        train,
        initial_weights=None if resume is None else resume.weights,
        initial_topology=None if resume is None else resume.topology,
        start_epoch=completed,
    )
    save_checkpoint(
        args.output,
        result.weights,
        result.topology,
        result.signatures,
        labels,
        model,
        {
            "seed": train.seed,
            "epochs_completed": train.epochs,
            "input_current": train.input_current,
            "encoder": encoder_config.__dict__,
            "encoder_digest": sentence_encoder.digest,
            "corpus_digest": corpus_digest,
            "python": platform.python_version(),
        },
        ([] if resume is None else list(resume.training_events)) + result.events,
    )
    return 0


def run_probe(args: argparse.Namespace) -> int:
    """Recall from a frozen checkpoint using only an encoded cue."""
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint.manifest["metadata"]
    encoder_config = EncoderConfig(**metadata["encoder"])
    sentence_encoder = LocalSentenceEncoder(args.encoder_checkpoint)
    if sentence_encoder.digest != metadata["encoder_digest"]:
        raise ValueError("probe encoder digest differs from training manifest")
    cue = encode_trace(
        args.cue.read_text(encoding="utf-8"),
        sentence_encoder,
        checkpoint.model,
        encoder_config,
        input_current=args.input_current,
    ).currents
    result = probe_checkpoint(
        checkpoint,
        cue,
        ProbeConfig(
            seed=args.seed,
            cue_fraction=args.cue_fraction,
            completion_steps=args.completion_steps,
            input_current=args.input_current,
        ),
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "label": result.label,
            "scores": result.scores.tolist(),
            "completion_spikes": result.completion_spikes,
            "recurrence_input_ratio": result.recurrence_input_ratio,
            "weight_digest_unchanged": result.weight_digest_unchanged,
        },
        indent=2,
        sort_keys=True,
    )
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """Score every matched control for every seed in isolated fresh processes."""
    checkpoint = load_checkpoint(args.checkpoint)
    _, _, _, _, corpus_digest = load_corpus(args.corpus_manifest)
    if checkpoint.manifest["metadata"]["corpus_digest"] != corpus_digest:
        raise ValueError("benchmark corpus digest differs from training manifest")
    seeds = [int(value) for value in args.seeds.split(",")]
    conditions = ("trained", "shuffled", "random", "zero", "encoder-only")
    scores: dict[str, dict[int, list[float]]] = {name: {} for name in conditions}
    details: dict[str, dict[int, list[dict[str, Any]]]] = {name: {} for name in conditions}
    with tempfile.TemporaryDirectory(prefix="remanentia-snn-benchmark-") as temporary:
        for run_seed in seeds:
            for condition in conditions:
                output = Path(temporary) / f"{condition}-{run_seed}.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "snn_memory.cli",
                        "condition",
                        "--checkpoint",
                        str(args.checkpoint),
                        "--corpus-manifest",
                        str(args.corpus_manifest),
                        "--condition",
                        condition,
                        "--seed",
                        str(run_seed),
                        "--cue-fraction",
                        str(args.cue_fraction),
                        "--completion-steps",
                        str(args.completion_steps),
                        "--input-current",
                        str(args.input_current),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                result = read_json(output)
                scores[condition][run_seed] = [float(value) for value in result["scores"]]
                details[condition][run_seed] = list(result["details"])
    report = decision_report(scores, details, seeds)
    atomic_write_json(args.output, report, indent=2, sort_keys=True)
    return 0


def run_condition(args: argparse.Namespace) -> int:
    """Score one matched control against the encoded corpus for one seed."""
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint.manifest["metadata"]
    encoder_config = EncoderConfig(**metadata["encoder"])
    labels, sequences, sentence_encoder, corpus_digest = _encode_corpus(
        args.corpus_manifest, checkpoint.model, encoder_config, args.input_current
    )
    if sentence_encoder.digest != metadata["encoder_digest"]:
        raise ValueError("condition encoder digest differs from training manifest")
    if corpus_digest != metadata["corpus_digest"]:
        raise ValueError("condition corpus digest differs from training manifest")
    scores, details = evaluate_condition(
        checkpoint,
        sequences,
        labels,
        args.condition,
        args.seed,
        ProbeConfig(
            seed=args.seed,
            cue_fraction=args.cue_fraction,
            completion_steps=args.completion_steps,
            input_current=args.input_current,
        ),
    )
    atomic_write_json(
        args.output,
        {"scores": scores, "details": details},
        indent=2,
        sort_keys=True,
    )
    return 0
