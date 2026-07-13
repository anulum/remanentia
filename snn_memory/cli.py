# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory command line

"""Standalone train, probe, benchmark, inspect and manifest verification CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from file_utils import atomic_write_json
from snn_memory.checkpoint import load_checkpoint, save_checkpoint, verify_run_directory
from snn_memory.contracts import EncoderConfig, ModelConfig, ProbeConfig, TrainConfig
from snn_memory.sentence_encoder import LocalSentenceEncoder, encode_trace
from snn_memory.experiment import decision_report, evaluate_condition
from snn_memory.probe import probe_checkpoint
from snn_memory.trainer import train_memories


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def _load_corpus(manifest_path: Path) -> tuple[list[str], list[str], Path, str | None]:
    manifest = _read_json(manifest_path)
    entries = manifest.get("entries")
    checkpoint = manifest.get("encoder_checkpoint")
    encoder_digest = manifest.get("encoder_digest")
    if not isinstance(entries, list) or not isinstance(checkpoint, str):
        raise ValueError("corpus manifest requires entries and encoder_checkpoint")
    if encoder_digest is not None and not isinstance(encoder_digest, str):
        raise ValueError("encoder_digest must be a SHA-256 string")
    labels: list[str] = []
    texts: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise ValueError("each corpus entry requires label, path, and SHA-256")
        source = (manifest_path.parent / entry["path"]).resolve()
        labels.append(entry["label"])
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"corpus source digest mismatch: {source}")
        texts.append(content.decode("utf-8"))
    encoder_path = (manifest_path.parent / checkpoint).resolve()
    return labels, texts, encoder_path, encoder_digest


def _encode_corpus(
    manifest_path: Path,
    model: ModelConfig,
    encoder_config: EncoderConfig,
    input_current: float,
) -> tuple[list[str], list[Any], LocalSentenceEncoder]:
    labels, texts, checkpoint_path, expected_digest = _load_corpus(manifest_path)
    sentence_encoder = LocalSentenceEncoder(checkpoint_path)
    if expected_digest is not None and sentence_encoder.digest != expected_digest:
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
    return labels, sequences, sentence_encoder


def _train(args: argparse.Namespace) -> int:
    config = _read_json(args.config)
    model = ModelConfig(**config.get("model", {}))
    encoder_config = EncoderConfig(**config.get("encoder", {}))
    train = TrainConfig(**config.get("train", {}))
    labels, sequences, sentence_encoder = _encode_corpus(
        args.corpus_manifest, model, encoder_config, train.input_current
    )
    resume = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume is not None:
        if resume.model != model or resume.labels != tuple(labels):
            raise ValueError("resume checkpoint model or labels differ")
        completed = int(resume.manifest["metadata"]["epochs_completed"])
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
            "encoder": encoder_config.__dict__,
            "encoder_digest": sentence_encoder.digest,
            "python": platform.python_version(),
        },
        result.events,
    )
    return 0


def _probe(args: argparse.Namespace) -> int:
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


def _benchmark(args: argparse.Namespace) -> int:
    seeds = [int(value) for value in args.seeds.split(",")]
    conditions = ("trained", "shuffled", "random", "zero", "encoder-only")
    scores: dict[str, dict[int, list[float]]] = {name: {} for name in conditions}
    details: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {} for name in conditions
    }
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
                result = _read_json(output)
                scores[condition][run_seed] = [float(value) for value in result["scores"]]
                details[condition][run_seed] = list(result["details"])
    report = decision_report(scores, details, seeds)
    atomic_write_json(args.output, report, indent=2, sort_keys=True)
    return 0


def _condition(args: argparse.Namespace) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint.manifest["metadata"]
    encoder_config = EncoderConfig(**metadata["encoder"])
    labels, sequences, sentence_encoder = _encode_corpus(
        args.corpus_manifest, checkpoint.model, encoder_config, args.input_current
    )
    if sentence_encoder.digest != metadata["encoder_digest"]:
        raise ValueError("condition encoder digest differs from training manifest")
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


def _inspect(args: argparse.Namespace) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    print(json.dumps(checkpoint.manifest, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    print(verify_run_directory(args.run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone research parser without product CLI imports."""
    parser = argparse.ArgumentParser(prog="remanentia-snn-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--corpus-manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume-checkpoint", type=Path)
    train.set_defaults(handler=_train)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--cue", type=Path, required=True)
    probe.add_argument("--encoder-checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--seed", type=int, default=11)
    probe.add_argument("--cue-fraction", type=float, default=0.5)
    probe.add_argument("--completion-steps", type=int, default=40)
    probe.add_argument("--input-current", type=float, default=18.0)
    probe.set_defaults(handler=_probe)
    bench = subparsers.add_parser("benchmark")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--corpus-manifest", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--seeds", default="11,29,47,71,101,131,167,211,257,307")
    bench.add_argument("--cue-fraction", type=float, default=0.5)
    bench.add_argument("--completion-steps", type=int, default=40)
    bench.add_argument("--input-current", type=float, default=18.0)
    bench.set_defaults(handler=_benchmark)
    condition = subparsers.add_parser("condition")
    condition.add_argument("--checkpoint", type=Path, required=True)
    condition.add_argument("--corpus-manifest", type=Path, required=True)
    condition.add_argument(
        "--condition",
        choices=("trained", "shuffled", "random", "zero", "encoder-only"),
        required=True,
    )
    condition.add_argument("--seed", type=int, required=True)
    condition.add_argument("--cue-fraction", type=float, default=0.5)
    condition.add_argument("--completion-steps", type=int, default=40)
    condition.add_argument("--input-current", type=float, default=18.0)
    condition.add_argument("--output", type=Path, required=True)
    condition.set_defaults(handler=_condition)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.set_defaults(handler=_inspect)
    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one standalone temporal-memory command."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
