# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pinned-model run-adapter tests

"""In-process model-provisioned coverage for the pinned-model CLI run adapter.

These exercise ``snn_memory.cli_runners`` against the real ``.snn_models`` checkpoint
and real tracked corpora, so this is the required model report for the module that
the CI-core coverage denominator excludes. Pinned-model gated (local, not CI).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from snn_memory import cli_runners
from snn_memory.checkpoint import load_checkpoint
from tests.model_gates.model_precondition import MODEL, ROOT, require_pinned_model

PINNED_DIGEST = require_pinned_model()
SNN_DOC = ROOT / "docs/research/snn_consolidation.md"
RETRIEVAL_DOC = ROOT / "docs/adr/0004-dual-retrieval-stacks.md"


def _entry(name: str, path: Path) -> dict[str, str]:
    return {
        "label": name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _inputs(
    tmp_path: Path,
    *,
    epochs: int = 1,
    encoder_digest: str | None = None,
    encoder_overrides: dict[str, object] | None = None,
    train_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    encoder: dict[str, object] = {"feature_dim": 24, "packet_ms": 5, "silent_ms": 1}
    encoder.update(encoder_overrides or {})
    train: dict[str, object] = {"seed": 11, "epochs": epochs, "input_current": 18.0}
    train.update(train_overrides or {})
    config = tmp_path / f"config-{epochs}.json"
    config.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 24, "connectivity": 0.25},
                "encoder": encoder,
                "train": train,
            }
        )
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "encoder_checkpoint": str(MODEL),
        "encoder_digest": PINNED_DIGEST if encoder_digest is None else encoder_digest,
        "entries": [_entry("snn", SNN_DOC), _entry("retrieval", RETRIEVAL_DOC)],
    }
    corpus = tmp_path / f"corpus-{epochs}.json"
    corpus.write_text(json.dumps(manifest))
    return config, corpus


def _ns(**fields: object) -> argparse.Namespace:
    return argparse.Namespace(**fields)


def _corrupt_manifest_encoder_digest(checkpoint: Path) -> None:
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["metadata"]["encoder_digest"] = "1" * 64
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))


def _drift_corpus(corpus: Path, tmp_path: Path) -> Path:
    """Create a valid same-label manifest whose first verified source byte differs."""
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    source = tmp_path / "drifted-source.md"
    source.write_text(SNN_DOC.read_text(encoding="utf-8") + "\ncorpus drift\n", encoding="utf-8")
    payload["entries"][0]["path"] = str(source)
    payload["entries"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    drifted = tmp_path / "drifted-corpus.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    return drifted


def test_run_train_probe_condition_and_benchmark(tmp_path: Path) -> None:
    config, corpus = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    assert (
        cli_runners.run_train(
            _ns(config=config, corpus_manifest=corpus, output=checkpoint, resume_checkpoint=None)
        )
        == 0
    )
    assert load_checkpoint(checkpoint).labels == ("snn", "retrieval")

    probe_out = tmp_path / "probe.json"
    assert (
        cli_runners.run_probe(
            _ns(
                checkpoint=checkpoint,
                cue=SNN_DOC,
                encoder_checkpoint=MODEL,
                output=probe_out,
                seed=11,
                cue_fraction=0.5,
                completion_steps=8,
                input_current=18.0,
            )
        )
        == 0
    )
    probe_payload = json.loads(probe_out.read_text())
    assert probe_payload["weight_digest_unchanged"] is True
    assert probe_payload["label"] is None

    condition_out = tmp_path / "condition.json"
    assert (
        cli_runners.run_condition(
            _ns(
                checkpoint=checkpoint,
                corpus_manifest=corpus,
                condition="trained",
                seed=11,
                cue_fraction=0.5,
                completion_steps=8,
                input_current=18.0,
                output=condition_out,
            )
        )
        == 0
    )
    assert len(json.loads(condition_out.read_text())["scores"]) == 2

    zero_condition_out = tmp_path / "zero-condition.json"
    assert (
        cli_runners.run_condition(
            _ns(
                checkpoint=checkpoint,
                corpus_manifest=corpus,
                condition="zero",
                seed=11,
                cue_fraction=0.5,
                completion_steps=8,
                input_current=18.0,
                output=zero_condition_out,
            )
        )
        == 0
    )
    assert all(
        row["prediction"] is None for row in json.loads(zero_condition_out.read_text())["details"]
    )

    report_out = tmp_path / "report.json"
    assert (
        cli_runners.run_benchmark(
            _ns(
                checkpoint=checkpoint,
                corpus_manifest=corpus,
                output=report_out,
                seeds="11",
                cue_fraction=0.5,
                completion_steps=6,
                input_current=18.0,
            )
        )
        == 0
    )
    report = json.loads(report_out.read_text())
    assert set(report["conditions"]) == {"trained", "shuffled", "random", "zero", "encoder-only"}


def test_run_train_resumes_from_a_prior_checkpoint(tmp_path: Path) -> None:
    config_one, corpus = _inputs(tmp_path, epochs=1)
    config_three, _ = _inputs(tmp_path, epochs=3)
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    cli_runners.run_train(
        _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
    )
    assert (
        cli_runners.run_train(
            _ns(
                config=config_three, corpus_manifest=corpus, output=resumed, resume_checkpoint=first
            )
        )
        == 0
    )
    resumed_checkpoint = load_checkpoint(resumed)
    assert resumed_checkpoint.manifest["metadata"]["epochs_completed"] == 3
    direct = tmp_path / "direct"
    cli_runners.run_train(
        _ns(config=config_three, corpus_manifest=corpus, output=direct, resume_checkpoint=None)
    )
    direct_checkpoint = load_checkpoint(direct)
    assert resumed_checkpoint.manifest["event_digest"] == direct_checkpoint.manifest["event_digest"]
    assert resumed_checkpoint.training_events == direct_checkpoint.training_events


def test_run_train_rejects_a_mismatched_resume_checkpoint(tmp_path: Path) -> None:
    config_one, corpus = _inputs(tmp_path, epochs=1)
    first = tmp_path / "first"
    cli_runners.run_train(
        _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
    )
    wider = tmp_path / "wider.json"
    wider.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 32, "connectivity": 0.25},
                "encoder": {"feature_dim": 24, "packet_ms": 5, "silent_ms": 1},
                "train": {"seed": 11, "epochs": 3, "input_current": 18.0},
            }
        )
    )
    with pytest.raises(ValueError, match="resume checkpoint model or labels differ"):
        cli_runners.run_train(
            _ns(
                config=wider, corpus_manifest=corpus, output=tmp_path / "x", resume_checkpoint=first
            )
        )


def test_run_train_rejects_resume_contract_drift(tmp_path: Path) -> None:
    cases: tuple[tuple[dict[str, object] | None, dict[str, object] | None, str], ...] = (
        ({"active_fraction": 0.1}, None, "encoder configuration differs"),
        (None, {"seed": 29}, "training seed or input current differs"),
        (None, {"input_current": 19.0}, "training seed or input current differs"),
    )
    for index, (encoder_overrides, train_overrides, match) in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        config_one, corpus = _inputs(case, epochs=1)
        first = case / "first"
        cli_runners.run_train(
            _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
        )
        config_three, _ = _inputs(
            case,
            epochs=3,
            encoder_overrides=encoder_overrides,
            train_overrides=train_overrides,
        )
        with pytest.raises(ValueError, match=match):
            cli_runners.run_train(
                _ns(
                    config=config_three,
                    corpus_manifest=corpus,
                    output=case / "x",
                    resume_checkpoint=first,
                )
            )


def test_run_train_rejects_resume_encoder_digest_drift(tmp_path: Path) -> None:
    config_one, corpus = _inputs(tmp_path, epochs=1)
    first = tmp_path / "first"
    cli_runners.run_train(
        _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
    )
    _corrupt_manifest_encoder_digest(first)
    config_three, _ = _inputs(tmp_path, epochs=3)
    with pytest.raises(ValueError, match="encoder digest differs"):
        cli_runners.run_train(
            _ns(
                config=config_three,
                corpus_manifest=corpus,
                output=tmp_path / "x",
                resume_checkpoint=first,
            )
        )


def test_run_train_rejects_same_label_corpus_byte_drift(tmp_path: Path) -> None:
    config_one, corpus = _inputs(tmp_path, epochs=1)
    first = tmp_path / "first"
    cli_runners.run_train(
        _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
    )
    config_three, _ = _inputs(tmp_path, epochs=3)
    drifted = _drift_corpus(corpus, tmp_path)
    with pytest.raises(ValueError, match="corpus digest differs"):
        cli_runners.run_train(
            _ns(
                config=config_three,
                corpus_manifest=drifted,
                output=tmp_path / "x",
                resume_checkpoint=first,
            )
        )


def test_run_train_rejects_non_forward_resume_target(tmp_path: Path) -> None:
    for index, (completed_epochs, target_epochs) in enumerate(((1, 1), (3, 2))):
        case = tmp_path / str(index)
        case.mkdir()
        config_one, corpus = _inputs(case, epochs=completed_epochs)
        first = case / "first"
        cli_runners.run_train(
            _ns(config=config_one, corpus_manifest=corpus, output=first, resume_checkpoint=None)
        )
        target, _ = _inputs(case, epochs=target_epochs)
        with pytest.raises(ValueError, match="target epochs must exceed"):
            cli_runners.run_train(
                _ns(
                    config=target,
                    corpus_manifest=corpus,
                    output=case / "x",
                    resume_checkpoint=first,
                )
            )


def test_encode_corpus_rejects_a_manifest_encoder_digest_mismatch(tmp_path: Path) -> None:
    config, corpus = _inputs(tmp_path, encoder_digest="1" * 64)
    with pytest.raises(ValueError, match="corpus manifest encoder digest mismatch"):
        cli_runners.run_train(
            _ns(
                config=config, corpus_manifest=corpus, output=tmp_path / "x", resume_checkpoint=None
            )
        )


def test_run_probe_rejects_an_encoder_digest_mismatch(tmp_path: Path) -> None:
    config, corpus = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    cli_runners.run_train(
        _ns(config=config, corpus_manifest=corpus, output=checkpoint, resume_checkpoint=None)
    )
    _corrupt_manifest_encoder_digest(checkpoint)
    with pytest.raises(ValueError, match="probe encoder digest differs"):
        cli_runners.run_probe(
            _ns(
                checkpoint=checkpoint,
                cue=SNN_DOC,
                encoder_checkpoint=MODEL,
                output=tmp_path / "p.json",
                seed=11,
                cue_fraction=0.5,
                completion_steps=8,
                input_current=18.0,
            )
        )


def test_run_condition_rejects_an_encoder_digest_mismatch(tmp_path: Path) -> None:
    config, corpus = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    cli_runners.run_train(
        _ns(config=config, corpus_manifest=corpus, output=checkpoint, resume_checkpoint=None)
    )
    _corrupt_manifest_encoder_digest(checkpoint)
    with pytest.raises(ValueError, match="condition encoder digest differs"):
        cli_runners.run_condition(
            _ns(
                checkpoint=checkpoint,
                corpus_manifest=corpus,
                condition="trained",
                seed=11,
                cue_fraction=0.5,
                completion_steps=8,
                input_current=18.0,
                output=tmp_path / "c.json",
            )
        )


def test_condition_and_benchmark_reject_same_label_corpus_byte_drift(tmp_path: Path) -> None:
    config, corpus = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    cli_runners.run_train(
        _ns(config=config, corpus_manifest=corpus, output=checkpoint, resume_checkpoint=None)
    )
    drifted = _drift_corpus(corpus, tmp_path)
    condition_args = _ns(
        checkpoint=checkpoint,
        corpus_manifest=drifted,
        condition="trained",
        seed=11,
        cue_fraction=0.5,
        completion_steps=8,
        input_current=18.0,
        output=tmp_path / "condition.json",
    )
    with pytest.raises(ValueError, match="condition corpus digest differs"):
        cli_runners.run_condition(condition_args)
    benchmark_args = _ns(
        checkpoint=checkpoint,
        corpus_manifest=drifted,
        output=tmp_path / "benchmark.json",
        seeds="11",
        cue_fraction=0.5,
        completion_steps=8,
        input_current=18.0,
    )
    with pytest.raises(ValueError, match="benchmark corpus digest differs"):
        cli_runners.run_benchmark(benchmark_args)
