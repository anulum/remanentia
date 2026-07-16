# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory command line

"""Model-free CLI surface: argument parsing, checkpoint inspection and dispatch.

Parsing, ``inspect`` and ``verify-manifest`` need no embedding model and pull in no
run adapter or scientific graph (experiment, trainer, controls, sentence-embedding
model) — only the checkpoint reader they genuinely require — and are covered in the
default (CI) measurement. The pinned-model run adapter lives in
:mod:`snn_memory.cli_runners`; :func:`main` imports it lazily, only when a model
command is actually dispatched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from snn_memory.checkpoint import load_checkpoint, verify_run_directory

_MODEL_COMMANDS = ("train", "probe", "benchmark", "condition")


def _inspect(args: argparse.Namespace) -> int:
    """Print the manifest of a frozen checkpoint bundle."""
    checkpoint = load_checkpoint(args.checkpoint)
    print(json.dumps(checkpoint.manifest, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Validate a run directory and print its deterministic array digest."""
    print(verify_run_directory(args.run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone research parser (pure argparse, no run-adapter import)."""
    parser = argparse.ArgumentParser(prog="remanentia-snn-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--corpus-manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume-checkpoint", type=Path)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--cue", type=Path, required=True)
    probe.add_argument("--encoder-checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--seed", type=int, default=11)
    probe.add_argument("--cue-fraction", type=float, default=0.5)
    probe.add_argument("--completion-steps", type=int, default=40)
    probe.add_argument("--input-current", type=float, default=18.0)
    bench = subparsers.add_parser("benchmark")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--corpus-manifest", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--seeds", default="11,29,47,71,101,131,167,211,257,307")
    bench.add_argument("--cue-fraction", type=float, default=0.5)
    bench.add_argument("--completion-steps", type=int, default=40)
    bench.add_argument("--input-current", type=float, default=18.0)
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
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one standalone temporal-memory command, loading the model only if needed."""
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "verify-manifest":
        return _verify(args)
    from snn_memory import cli_runners

    runners = {
        "train": cli_runners.run_train,
        "probe": cli_runners.run_probe,
        "benchmark": cli_runners.run_benchmark,
        "condition": cli_runners.run_condition,
    }
    return int(runners[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
