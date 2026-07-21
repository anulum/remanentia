# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed-wheel cue-materializer D2 gate

"""Exercise D2 only through real Git repositories, real files and the real encoder.

Every fixture repository is driven by the ``git`` CLI, every cue-set and bundle
forgery is recomputed with an independent local reimplementation of the
canonical serialisation, opaque identifier derivations, corruption position
ranking and the domain-separated self digests, and every encoder claim is
proven against the real pinned local sentence-transformer checkpoint. No mock,
fake inference, or in-memory substitute stands in for a production surface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import os
import runpy
import stat
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from snn_memory import cue_materializer
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.cue_materializer import (
    CueMaterializerError,
    materialize_cue_set,
    read_cue_bundle,
    read_cue_set,
    validate_cue_bundle_bytes,
    validate_cue_set_bytes,
    verify_cue_bundle_with_encoder,
    verify_cue_set_against_sources,
)
from snn_memory.encoder import embeddings_to_currents
from snn_memory.sentence_encoder import LocalSentenceEncoder
from snn_memory.source_universe import (
    SourceUniverseError,
    validate_source_universe_bytes,
    write_source_universe,
)

CUE_SET_DOMAIN = b"remanentia:snn-v2-cue-set:v1\0"
BUNDLE_DOMAIN = b"remanentia:snn-v2-cue-bundle:v1\0"
UNIVERSE_DOMAIN = b"remanentia:snn-v2-source-universe:v1\0"
CUE_ID_DOMAIN = b"remanentia:snn-v2-cue-id:v1\0"
VARIANT_ID_DOMAIN = b"remanentia:snn-v2-variant-id:v1\0"
CALIBRATION_DOMAIN = b"remanentia:snn-v2-calibration-block:v1\0"
PERMUTATION_DOMAIN = b"remanentia:snn-v2-family-permutation:v1\0"
POSITION_DOMAIN = b"remanentia:snn-v2-corruption-positions:v1\0"
NOISE_DOMAIN = b"remanentia:snn-v2-noise-token:v1\0"
FAMILIES = ("truncation", "deletion", "masking", "sparse_noise")
PERMUTATIONS = tuple(itertools.permutations(FAMILIES))
PERCENTS = (0, 10, 25, 40)
BASE_TIMESTAMP = 1_700_000_000
WORD_A = (
    "ledger",
    "orchard",
    "harbour",
    "granite",
    "willow",
    "meadow",
    "signal",
    "quarry",
    "lantern",
    "compass",
    "thicket",
    "estuary",
    "paddock",
    "furnace",
    "gable",
    "cistern",
)
WORD_B = ("morning", "evening", "quiet", "sudden", "steady", "distant", "narrow")
WORD_C = (
    "ridge",
    "valley",
    "coast",
    "plain",
    "summit",
    "delta",
    "basin",
    "moor",
    "heath",
    "fen",
    "loch",
    "strand",
    "combe",
    "tor",
    "glen",
    "wold",
)

MODEL = ModelConfig(n_neurons=64)
ENCODER_CONFIG = EncoderConfig(feature_dim=16, packet_ms=5, silent_ms=2, active_fraction=0.1)
INPUT_CURRENT = 18.0


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(payload: dict[str, Any], domain: bytes) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    canonical = _canonical(unsigned)
    framed = domain + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _norm_sha(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return _sha(normalized.encode("utf-8"))


def _tree_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = "symlink:" + os.readlink(path)
        elif path.is_dir():
            snapshot[relative] = "dir"
        else:
            snapshot[relative] = "file:" + _sha(path.read_bytes())
    return snapshot


def _cue_id(text_sha: str) -> str:
    return "cue-" + hashlib.sha256(CUE_ID_DOMAIN + text_sha.encode()).hexdigest()[:32]


def _variant_id(base_sha: str, family: str, percent: int) -> str:
    material = f"{base_sha}\0{family}\0{percent}".encode()
    return "variant-" + hashlib.sha256(VARIANT_ID_DOMAIN + material).hexdigest()[:32]


def _digest_index(domain: bytes, record_sha: str, modulo: int) -> int:
    return int(hashlib.sha256(domain + record_sha.encode()).hexdigest(), 16) % modulo


def _positions(family: str, base_sha: str, universe: int, count: int) -> list[int]:
    def rank(position: int) -> str:
        material = (
            POSITION_DOMAIN
            + family.encode()
            + b"\0"
            + base_sha.encode()
            + b"\0"
            + position.to_bytes(8, "big")
        )
        return hashlib.sha256(material).hexdigest()

    return sorted(sorted(range(universe), key=rank)[:count])


def _selected(family: str, base_sha: str, token_count: int, percent: int) -> list[int]:
    affected = (token_count * percent + 50) // 100
    if family == "truncation":
        return list(range(token_count - affected, token_count))
    universe = token_count + 1 if family == "sparse_noise" else token_count
    return _positions(family, base_sha, universe, affected)


def _noise_token(base_sha: str, gap: int, lexicon: Sequence[str]) -> str:
    material = NOISE_DOMAIN + base_sha.encode() + b"\0" + gap.to_bytes(8, "big")
    return lexicon[int(hashlib.sha256(material).hexdigest(), 16) % len(lexicon)]


def _block_text(events: Sequence[str]) -> str:
    return "\n".join(" ".join(event.split()) for event in events)


def _apply(
    family: str, base_text: str, selected: Sequence[int], base_sha: str, lexicon: Sequence[str]
) -> str:
    lines = [line.split(" ") for line in base_text.split("\n")]
    flat = [(index, token) for index, line in enumerate(lines) for token in line]
    rebuilt: list[list[str]] = [[] for _ in lines]
    if family == "sparse_noise":
        by_gap = {gap: _noise_token(base_sha, gap, lexicon) for gap in selected}
        for position, (line_index, token) in enumerate(flat):
            if position in by_gap:
                rebuilt[line_index].append(by_gap[position])
            rebuilt[line_index].append(token)
        if len(flat) in by_gap:
            rebuilt[flat[-1][0]].append(by_gap[len(flat)])
    else:
        chosen = frozenset(selected)
        for position, (line_index, token) in enumerate(flat):
            if position in chosen:
                if family == "masking":
                    rebuilt[line_index].append("[MASK]")
                continue
            rebuilt[line_index].append(token)
    return "\n".join(" ".join(line) for line in rebuilt if line)


def _blocks(event_count: int) -> list[list[int]]:
    bounds = [index * event_count // 5 for index in range(6)]
    return [list(range(bounds[index], bounds[index + 1])) for index in range(5)]


def _split_events(text: str) -> list[str]:
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _expect(substring: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except (CueMaterializerError, SourceUniverseError) as error:
        if substring not in str(error):
            raise AssertionError(f"unexpected error {error!r}, wanted {substring!r}") from error
        return
    raise AssertionError(f"expected a fail-closed error containing {substring!r}")


def _run(
    arguments: list[str],
    cwd: Path,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments, cwd=cwd, capture_output=True, check=check, env=env, input=input_bytes
    )


def _git(repo: Path, *arguments: str) -> bytes:
    return _run(["git", *arguments], repo).stdout


def _commit(repo: Path, message: str, timestamp: int) -> None:
    _git(repo, "add", "-A")
    environment = dict(os.environ)
    environment["GIT_AUTHOR_DATE"] = f"@{timestamp} +0000"
    environment["GIT_COMMITTER_DATE"] = f"@{timestamp} +0000"
    _run(["git", "commit", "-q", "-m", message], repo, env=environment)


def _init(parent: Path, name: str, *extra: str) -> Path:
    repo = parent / name
    repo.mkdir(parents=True)
    _run(["git", "init", "-q", *extra], repo)
    _git(repo, "config", "user.name", "D2 Gate")
    _git(repo, "config", "user.email", "d2@example.invalid")
    return repo


def _write(repo: Path, relative: str, raw: bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _standard_doc(record: int) -> bytes:
    lines = []
    for index in range(52):
        lines.append(
            f"Record {record} event {index:03d} explores {WORD_A[record]} "
            f"{WORD_B[index % 7]} phenomena beside {WORD_C[record]} landmark "
            f"mark-{record}-{index}."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _unicode_doc(record: int) -> bytes:
    lines = []
    for index in range(52):
        lines.append(
            f"Récord {record} café entry {index:03d} keeps naïve señal notes — "
            f"beside {WORD_C[record]} quéue mark-{record}-{index} … fine."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mask_collision_doc(record: int) -> bytes:
    lines = []
    for index in range(52):
        lines.append(
            f"Masked {record} record keeps [MASK] token alpha basalt fjord literal "
            f"{WORD_B[index % 7]} probe mark-{record}-{index}; punctuation , stays."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _wide_doc(record: int) -> bytes:
    lines = []
    for index in range(50):
        extras = " ".join(f"filler-{record}-{index}-{position}" for position in range(14))
        lines.append(
            f"Wide {record} record event {index:03d} stretches across {extras} "
            f"and closes mark-{record}-{index}."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _near_duplicate_doc() -> bytes:
    shared = (
        "Shared vocabulary sentence keeps twenty stable tokens covering harbour "
        "ledger orchard granite willow meadow signal quarry lantern compass notes."
    )
    lines = []
    for index in range(50):
        if index % 10 == 5:
            lines.append(shared + f" Distinct block {index // 10} marker.")
        else:
            lines.append(shared)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _uniform_doc() -> bytes:
    line = "Uniform sentence repeats identical deterministic tokens for every block."
    return ("\n".join([line] * 50) + "\n").encode("utf-8")


def _nfkc_collision_doc() -> bytes:
    nfc_line = "Café résumé entry keeps stable text mark-{i:02d}."
    nfd_line = "Café résumé entry keeps stable text mark-{i:02d}."
    lines = [nfc_line.format(i=index) for index in range(10)]
    lines += [nfd_line.format(i=index) for index in range(10)]
    lines += [
        f"Distinct tail sentence {index:02d} carries unrelated closing text." for index in range(30)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_manifests(repo: Path, source_doc: bytes) -> None:
    extra = _write(repo, "docs/public/manifest-source.md", source_doc)
    development = {
        "schema_version": 1,
        "split": "development",
        "encoder_checkpoint": "../../.snn_models/model",
        "encoder_digest": "1" * 64,
        "entries": [
            {
                "label": "v1-dev-0",
                "path": "../../docs/public/manifest-source.md",
                "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
            }
        ],
    }
    _write(
        repo,
        "experiments/snn_memory/development_corpus.json",
        json.dumps(development, sort_keys=True).encode("utf-8"),
    )
    locked = {
        "schema_version": 1,
        "split": "locked-evaluation",
        "locked": True,
        "encoder_checkpoint": "../../.snn_models/model",
        "encoder_digest": "1" * 64,
        "entries": [],
    }
    _write(
        repo,
        "experiments/snn_memory/locked_evaluation_corpus.json",
        json.dumps(locked, sort_keys=True).encode("utf-8"),
    )


def _fixture_repo(parent: Path, name: str, docs: dict[int, bytes] | None = None) -> Path:
    repo = _init(parent, name)
    overrides = docs or {}
    for record in range(16):
        raw = overrides.get(record, _standard_doc(record))
        _write(repo, f"docs/public/record-{record:02d}.md", raw)
    _write_manifests(repo, _standard_doc(0).replace(b"Record 0", b"Source X"))
    _commit(repo, "fixture", BASE_TIMESTAMP)
    return repo


def _universe(repo: Path, workspace: Path, name: str) -> tuple[Path, str]:
    output = workspace / f"{name}-universe.json"
    result = write_source_universe(repo, output)
    return output, result.file_sha256


def _expected_record(
    item: dict[str, Any], events: list[str], lexicon: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    texts: dict[str, str] = {}
    record_sha = item["content_sha256"]
    blocks = _blocks(item["event_count"])
    calibration_index = _digest_index(CALIBRATION_DOMAIN, record_sha, 5)
    permutation = PERMUTATIONS[_digest_index(PERMUTATION_DOMAIN, record_sha, 24)]
    calibration_text = _block_text([events[index] for index in blocks[calibration_index]])
    calibration_id = _cue_id(_sha(calibration_text.encode("utf-8")))
    texts[f"cues/{calibration_id}.txt"] = calibration_text
    entry: dict[str, Any] = {
        "record_id": item["record_id"],
        "event_count": item["event_count"],
        "calibration_block_index": calibration_index,
        "family_permutation_index": _digest_index(PERMUTATION_DOMAIN, record_sha, 24),
        "calibration_cue": {
            "cue_id": calibration_id,
            "path": f"cues/{calibration_id}.txt",
            "sha256": _sha(calibration_text.encode("utf-8")),
            "normalized_text_sha256": _norm_sha(calibration_text),
            "event_indices": blocks[calibration_index],
        },
        "evaluation_base_cues": [],
    }
    evaluation = [blocks[index] for index in range(5) if index != calibration_index]
    for family, indices in zip(permutation, evaluation, strict=True):
        base_text = _block_text([events[index] for index in indices])
        base_sha = _sha(base_text.encode("utf-8"))
        token_count = sum(len(line.split(" ")) for line in base_text.split("\n"))
        variants = []
        for percent in PERCENTS:
            selected = _selected(family, base_sha, token_count, percent)
            text = _apply(family, base_text, selected, base_sha, lexicon)
            variant_id = _variant_id(base_sha, family, percent)
            texts[f"cues/{variant_id}.txt"] = text
            variants.append(
                {
                    "variant_id": variant_id,
                    "path": f"cues/{variant_id}.txt",
                    "sha256": _sha(text.encode("utf-8")),
                    "normalized_text_sha256": _norm_sha(text),
                    "requested_percent": percent,
                    "affected_count": len(selected),
                    "realized_fraction": len(selected) / token_count,
                    "selected_positions": selected,
                    "tokenizer_digest": cue_materializer.TOKENIZER_SHA256,
                }
            )
        entry["evaluation_base_cues"].append(
            {
                "cue_id": _cue_id(base_sha),
                "task": "record_recall",
                "transform_family": family,
                "event_indices": indices,
                "token_count": token_count,
                "variants": variants,
            }
        )
    return entry, texts


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or type(value).__name__ == "mappingproxy":
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _check_happy_manifest(
    repo: Path, universe_path: Path, payload: dict[str, Any], lexicon: Sequence[str]
) -> None:
    universe = json.loads(universe_path.read_bytes())
    assert payload["source_universe"]["file_sha256"] == _sha(universe_path.read_bytes())
    assert payload["source_universe"]["repository_head"] == universe["repository"]["head"]
    assert payload["source_universe"]["selected_record_ids"] == universe["selected_record_ids"]
    assert [record["record_id"] for record in payload["records"]] == universe["selected_record_ids"]
    calibration_indices = set()
    permutation_indices = set()
    families_seen = set()
    gap_at_end = False
    for record, item in zip(payload["records"], universe["selected"], strict=True):
        raw = (repo / item["path"]).read_bytes()
        assert _sha(raw) == item["content_sha256"]
        events = _split_events(raw.decode("utf-8"))
        expected, _texts = _expected_record(item, events, lexicon)
        actual = _thaw(record)
        for base_cue in actual["evaluation_base_cues"]:
            for variant in base_cue["variants"]:
                assert set(variant.pop("bundle")) == {"path", "sha256"}
        assert actual == expected, f"record {item['record_id']} derivation differs"
        calibration_indices.add(record["calibration_block_index"])
        permutation_indices.add(record["family_permutation_index"])
        record_hash = item["content_sha256"]
        for base_cue in record["evaluation_base_cues"]:
            families_seen.add(base_cue["transform_family"])
            assert record_hash[:32] not in base_cue["cue_id"]
            token_count = base_cue["token_count"]
            for variant in base_cue["variants"]:
                assert record_hash[:32] not in variant["variant_id"]
                assert record_hash[:32] not in variant["path"]
                if (
                    base_cue["transform_family"] == "sparse_noise"
                    and variant["selected_positions"]
                    and variant["selected_positions"][-1] == token_count
                ):
                    gap_at_end = True
        calibration = record["calibration_cue"]
        assert record_hash[:32] not in calibration["cue_id"]
        assert record_hash[:32] not in calibration["path"]
    assert families_seen == set(FAMILIES)
    assert len(calibration_indices) > 1, "fixture must exercise several calibration blocks"
    assert len(permutation_indices) > 1, "fixture must exercise several permutations"
    assert gap_at_end, "fixture must select the terminal insertion gap at least once"


def _check_texts_on_disk(output: Path, payload: dict[str, Any]) -> None:
    lexicon_path = output / payload["noise_lexicon"]["path"]
    assert _sha(lexicon_path.read_bytes()) == payload["noise_lexicon"]["sha256"]
    tokens = lexicon_path.read_text(encoding="utf-8").splitlines()
    assert len(tokens) == payload["noise_lexicon"]["token_count"] == 64
    assert len(set(tokens)) == 64 and all(token and " " not in token for token in tokens)
    dropped = False
    mask_literal = False
    noise_collision = False
    for record in payload["records"]:
        calibration = record["calibration_cue"]
        raw = (output / calibration["path"]).read_bytes()
        assert _sha(raw) == calibration["sha256"]
        for base_cue in record["evaluation_base_cues"]:
            base_text = (output / base_cue["variants"][0]["path"]).read_text(encoding="utf-8")
            if "[MASK]" in base_text:
                mask_literal = True
            if {"alpha", "basalt", "fjord"} & set(base_text.split()):
                noise_collision = True
            for variant in base_cue["variants"]:
                raw = (output / variant["path"]).read_bytes()
                assert _sha(raw) == variant["sha256"]
                assert _norm_sha(raw.decode("utf-8")) == variant["normalized_text_sha256"]
                stored = raw.decode("utf-8")
                if base_cue["transform_family"] in ("truncation", "deletion") and len(
                    stored.split("\n")
                ) < len(base_cue["event_indices"]):
                    dropped = True
    assert dropped, "fixture must drop at least one fully-corrupted line"
    assert mask_literal, "fixture must contain a literal [MASK] token collision"
    assert noise_collision, "fixture must contain noise-lexicon token collisions"


def _check_immutability(artifact: Any, bundle: Any) -> None:
    try:
        artifact.payload["records"] = ()
        raise AssertionError("cue-set payload accepted a mutation")
    except TypeError:
        pass
    try:
        artifact.payload["records"][0]["record_id"] = "sha256:" + "0" * 64
        raise AssertionError("nested cue-set payload accepted a mutation")
    except TypeError:
        pass
    assert isinstance(artifact.payload["records"], tuple)
    assert not bundle.embedding.flags.writeable
    assert not bundle.current_rows.flags.writeable
    try:
        bundle.embedding[0, 0] = 0.0
        raise AssertionError("bundle embedding accepted a mutation")
    except ValueError:
        pass


def _check_bundles(
    output: Path,
    payload: dict[str, Any],
    repo: Path,
    universe_path: Path,
    encoder_checkpoint: Path,
    encoder_digest: str,
    adapter: LocalSentenceEncoder,
    lexicon: Sequence[str],
) -> tuple[Any, int]:
    universe = json.loads(universe_path.read_bytes())
    sampled = None
    verified = 0
    per_family: set[str] = set()
    for record, item in zip(payload["records"], universe["selected"], strict=True):
        events = _split_events((repo / item["path"]).read_bytes().decode("utf-8"))
        _entry, texts = _expected_record(item, events, lexicon)
        for base_cue in record["evaluation_base_cues"]:
            family = base_cue["transform_family"]
            for variant in base_cue["variants"]:
                bundle_path = output / variant["bundle"]["path"]
                bundle = read_cue_bundle(bundle_path, variant["bundle"]["sha256"])
                assert bundle.payload["cue_id"] == variant["variant_id"]
                expected_text = texts[variant["path"]]
                expected_raw = expected_text.encode("utf-8")
                assert bundle.text == expected_text, f"{variant['variant_id']} cue bytes differ"
                assert bundle.payload["text_sha256"] == _sha(expected_raw) == variant["sha256"]
                assert bundle.payload["normalized_text_sha256"] == _norm_sha(expected_text)
                embedding = adapter.encode(_split_events(expected_text))
                assert bundle.embedding.shape == embedding.shape
                assert bundle.embedding.tobytes() == embedding.tobytes(), (
                    f"{variant['variant_id']} embedding differs from the pinned encoder"
                )
                currents = embeddings_to_currents(
                    embedding, MODEL, ENCODER_CONFIG, input_current=INPUT_CURRENT
                )
                rows, columns = np.nonzero(currents)
                assert bundle.currents_shape == (currents.shape[0], currents.shape[1])
                assert bundle.current_rows.tobytes() == rows.astype(np.int64).tobytes()
                assert bundle.current_columns.tobytes() == columns.astype(np.int64).tobytes()
                assert bundle.current_values.tobytes() == currents[rows, columns].tobytes()
                decoded = np.frombuffer(
                    base64.b64decode(bundle.payload["embedding"]["data_base64"]), dtype="<f8"
                )
                assert decoded.reshape(embedding.shape).tobytes() == embedding.tobytes()
                verified += 1
                if family not in per_family and variant["requested_percent"] == 25:
                    verify_cue_bundle_with_encoder(bundle, encoder_checkpoint, encoder_digest)
                    per_family.add(family)
                    sampled = bundle
    assert per_family == set(FAMILIES)
    assert verified == 256, f"verified only {verified} of 256 bundles"
    return sampled, verified


def _check_atomicity(
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    output: Path,
    workspace: Path,
) -> None:
    def attempt(target: Path) -> Callable[[], object]:
        return lambda: materialize_cue_set(
            repo,
            universe_path,
            universe_sha,
            encoder_checkpoint,
            encoder_digest,
            target,
            model=MODEL,
            encoder_config=ENCODER_CONFIG,
            input_current=INPUT_CURRENT,
        )

    _expect("output root must be freshly created", attempt(output))
    leftovers = [path for path in output.rglob(".*tmp*") if path.is_file()]
    assert not leftovers, f"temporary artifacts leaked: {leftovers}"
    _expect(
        "output parent cannot be resolved",
        attempt(workspace / "missing-parent/deep/output"),
    )
    outside_root = workspace / "escape-root-target"
    outside_root.mkdir()
    root_link = workspace / "symlinked-root"
    root_link.symlink_to(outside_root)
    _expect("output root must be freshly created", attempt(root_link))
    assert not any(outside_root.iterdir()), "symlinked root escaped the declared output"
    dangling = workspace / "dangling-root"
    dangling.symlink_to(workspace / "never-created-target")
    _expect("output root must be freshly created", attempt(dangling))
    assert not (workspace / "never-created-target").exists()
    outside_parent = workspace / "escape-parent-target"
    outside_parent.mkdir()
    parent_link = workspace / "symlinked-parent"
    parent_link.symlink_to(outside_parent)
    _expect(
        "output parent must be an absolute canonical path",
        attempt(parent_link / "output"),
    )
    assert not any(outside_parent.iterdir()), "symlinked parent escaped the declared output"
    wrong_digest_root = workspace / "inproc-wrong-digest"
    _expect(
        "pinned encoder directory digest mismatch",
        lambda: materialize_cue_set(
            repo,
            universe_path,
            universe_sha,
            encoder_checkpoint,
            "0" * 64,
            wrong_digest_root,
            model=MODEL,
            encoder_config=ENCODER_CONFIG,
            input_current=INPUT_CURRENT,
        ),
    )
    assert not wrong_digest_root.exists(), "in-process wrong-digest run left a partial tree"
    _check_inprocess_residual_policy(
        workspace, repo, universe_path, universe_sha, encoder_checkpoint, encoder_digest, output
    )


def _run_attacked_materialisation(
    workspace: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    target: Path,
    swap: Callable[[Path], None],
) -> None:
    import threading
    import time

    swapped = threading.Event()
    stop = threading.Event()

    def watcher() -> None:
        while not stop.is_set():
            if (target / "cues").is_dir() and (target / "bundles").is_dir():
                os.rename(target, workspace / (target.name + "-moved"))
                swap(target)
                swapped.set()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=watcher)
    thread.start()
    try:
        _expect(
            "output root pathname was replaced during materialisation",
            lambda: materialize_cue_set(
                repo,
                universe_path,
                universe_sha,
                encoder_checkpoint,
                encoder_digest,
                target,
                model=MODEL,
                encoder_config=ENCODER_CONFIG,
                input_current=INPUT_CURRENT,
            ),
        )
    finally:
        stop.set()
        thread.join()
    assert swapped.is_set(), "materialisation finished before the fresh root could be attacked"


def _check_inprocess_residual_policy(
    workspace: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    valid_tree: Path,
) -> None:
    import shutil

    def moved_of(target: Path) -> Path:
        return workspace / (target.name + "-moved")

    # Attack A: replace the root pathname with an unowned, otherwise-valid
    # deterministic tree carrying a sentinel. Success must fail closed on the
    # inode binding, the whole unowned replacement tree must be byte-identical
    # afterwards, and only the emptied captured inode may remain as residual.
    replacement = workspace / "residual-valid-replacement"
    shutil.copytree(valid_tree, replacement)
    (replacement / "SENTINEL-do-not-touch").write_bytes(b"unowned replacement sentinel\n")
    before_a = _tree_snapshot(replacement)
    target_a = workspace / "residual-attack-a"
    _run_attacked_materialisation(
        workspace,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        target_a,
        lambda path: os.rename(replacement, path),
    )
    assert target_a.is_dir() and not target_a.is_symlink()
    assert _tree_snapshot(target_a) == before_a, (
        "the unowned valid replacement tree was altered by cleanup"
    )
    moved_a = moved_of(target_a)
    assert moved_a.is_dir() and not any(moved_a.iterdir()), (
        "captured inode residual must remain empty, not a partial tree"
    )

    # Attack B: replace the root pathname with an unowned EMPTY directory. A blind
    # rmdir of the declared name would succeed and delete this unowned directory;
    # the inode-gated policy must leave it in place. Record its no-follow identity
    # and require the exact same inode/device to survive, empty.
    empty_replacement = workspace / "residual-empty-replacement"
    empty_replacement.mkdir()
    empty_stat = os.stat(empty_replacement, follow_symlinks=False)
    target_b = workspace / "residual-attack-b"
    _run_attacked_materialisation(
        workspace,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        target_b,
        lambda path: os.rename(empty_replacement, path),
    )
    survivor = os.stat(target_b, follow_symlinks=False)
    assert stat.S_ISDIR(survivor.st_mode) and not target_b.is_symlink(), (
        "unowned empty replacement directory was removed by blind cleanup"
    )
    assert (survivor.st_dev, survivor.st_ino) == (empty_stat.st_dev, empty_stat.st_ino), (
        "unowned empty replacement was deleted and recreated instead of preserved"
    )
    assert not any(target_b.iterdir()), "unowned empty replacement was written into"
    moved_b = moved_of(target_b)
    assert moved_b.is_dir() and not any(moved_b.iterdir()), (
        "captured inode residual must remain empty next to an unowned empty replacement"
    )

    # Attack C: rename the root away leaving no replacement, exercising the
    # stat-raises path of the inode binding and the residual policy.
    target_c = workspace / "residual-attack-c"
    _run_attacked_materialisation(
        workspace,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        target_c,
        lambda path: None,
    )
    assert not target_c.exists(), "renamed-away pathname must not be recreated"
    moved_c = moved_of(target_c)
    assert moved_c.is_dir() and not any(moved_c.iterdir()), (
        "captured inode residual must remain empty after a rename-away attack"
    )


def _check_write_transactions(
    workspace: Path,
    install_target: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    valid_tree: Path,
    model_config: Path,
    encoder_config: Path,
) -> list[int]:
    import shutil

    script = _console_script(install_target)
    site = Path(cue_materializer.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    pids: list[int] = []

    def materialize_argv(target: Path, digest: str) -> list[str]:
        # Match the exact model/encoder/current used to build valid_tree so the
        # substituted replacement is genuinely byte-valid for this installed call;
        # the negative proof must rest on the captured-inode binding, never on an
        # incidental config/digest mismatch.
        return [
            str(script),
            "materialize",
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            digest,
            "--output-dir",
            str(target),
            "--model-config",
            str(model_config),
            "--encoder-config",
            str(encoder_config),
            "--input-current",
            str(INPUT_CURRENT),
        ]

    # Real installed wrong-digest cleanup: the owned pathname still binds, so the
    # complete call-owned root is removed with no partial tree.
    cleanup_root = workspace / "cli-wrong-digest-out"
    failed = _run(materialize_argv(cleanup_root, "0" * 64), workspace, check=False, env=environment)
    assert failed.returncode == 2, failed.stderr.decode()
    assert b"pinned encoder directory digest mismatch" in failed.stderr
    assert not cleanup_root.exists(), "failed installed run left a partial output tree"

    # Real installed concurrent attack: replace the root pathname with an unowned,
    # otherwise-valid deterministic tree (a copy of a genuine materialisation plus
    # a sentinel), built for the SAME model/encoder/current so it is byte-valid for
    # this call. Success must fail closed on the captured-inode binding, the whole
    # unowned replacement tree must be byte-identical afterwards, and the captured
    # inode must be emptied and left only as a residual.
    replacement = workspace / "cli-valid-replacement"
    shutil.copytree(valid_tree, replacement)
    (replacement / "SENTINEL-do-not-touch").write_bytes(b"installed unowned replacement sentinel\n")
    before = _tree_snapshot(replacement)
    target = workspace / "concurrent-root"
    process = subprocess.Popen(
        materialize_argv(target, encoder_digest),
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pids.append(process.pid)
    swapped = False
    while process.poll() is None:
        if (target / "cues").is_dir() and (target / "bundles").is_dir():
            os.rename(target, workspace / "concurrent-moved")
            os.rename(replacement, target)
            swapped = True
            break
        _sleep_briefly()
    _stdout, stderr = process.communicate()
    assert swapped, "materialisation finished before the fresh root could be attacked"
    assert process.returncode == 2, (
        f"pathname-swapped run must fail closed, got {process.returncode}: {stderr.decode()}"
    )
    assert b"output root pathname was replaced during materialisation" in stderr
    assert target.is_dir() and not target.is_symlink()
    assert _tree_snapshot(target) == before, (
        "installed run altered the unowned valid replacement tree"
    )
    moved = workspace / "concurrent-moved"
    assert moved.is_dir() and not any(moved.iterdir()), (
        "captured inode residual must remain empty, not a partial tree"
    )
    return pids


def _sleep_briefly() -> None:
    import time

    time.sleep(0.02)


def _module_cli(arguments: list[str], expected_code: int) -> tuple[str, str]:
    import contextlib
    import io

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cue_materializer.main(arguments)
    assert code == expected_code, f"CLI exit {code}, wanted {expected_code}: {stderr.getvalue()}"
    return stdout.getvalue(), stderr.getvalue()


def _runpy_main(arguments: list[str], expected_code: int) -> None:
    previous = sys.argv
    sys.argv = ["remanentia-snn-cue-materializer", *arguments]
    try:
        runpy.run_module("snn_memory.cue_materializer", run_name="__main__", alter_sys=False)
        raise AssertionError("module __main__ did not exit")
    except SystemExit as exit_info:
        assert exit_info.code == expected_code, f"__main__ exit {exit_info.code}"
    finally:
        sys.argv = previous


def _console_script(install_target: Path) -> Path:
    script = install_target / "bin/remanentia-snn-cue-materializer"
    return script.resolve(strict=True)


def _check_cli(
    workspace: Path,
    install_target: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    api_output: Path,
    payload: dict[str, Any],
    set_sha: str,
) -> tuple[Path, list[int], Path, Path]:
    script = _console_script(install_target)
    site = Path(cue_materializer.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    model_config = workspace / "model-config.json"
    model_config.write_text(
        json.dumps({name: getattr(MODEL, name) for name in MODEL.__dataclass_fields__})
    )
    encoder_config = workspace / "encoder-config.json"
    encoder_config.write_text(
        json.dumps(
            {name: getattr(ENCODER_CONFIG, name) for name in ENCODER_CONFIG.__dataclass_fields__}
        )
    )
    cli_output = workspace / "cli-output"
    pids: list[int] = []
    process = subprocess.Popen(
        [
            str(script),
            "materialize",
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
            "--output-dir",
            str(cli_output),
            "--model-config",
            str(model_config),
            "--encoder-config",
            str(encoder_config),
            "--input-current",
            "18.0",
        ],
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    pids.append(process.pid)
    assert process.pid != os.getpid()
    assert process.returncode == 0, stderr.decode()
    report = json.loads(stdout)
    assert report["file_sha256"] == set_sha, "cross-process materialisation differs"
    api_files = sorted(path for path in api_output.rglob("*") if path.is_file())
    for api_file in api_files:
        cli_file = cli_output / api_file.relative_to(api_output)
        assert api_file.read_bytes() == cli_file.read_bytes(), f"{cli_file} differs"
    cli_files = sorted(path for path in cli_output.rglob("*") if path.is_file())
    assert len(cli_files) == len(api_files)
    validate = subprocess.Popen(
        [
            str(script),
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            set_sha,
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
        ],
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = validate.communicate()
    pids.append(validate.pid)
    assert validate.returncode == 0, stderr.decode()
    assert json.loads(stdout)["source_verified"] is True
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    bundle_ref = variant["bundle"]
    check_bundle = subprocess.Popen(
        [
            str(script),
            "validate-bundle",
            "--bundle",
            str(api_output / bundle_ref["path"]),
            "--bundle-sha256",
            bundle_ref["sha256"],
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
        ],
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = check_bundle.communicate()
    pids.append(check_bundle.pid)
    assert check_bundle.returncode == 0, stderr.decode()
    bundle_report = json.loads(stdout)
    assert bundle_report["encoder_verified"] is True
    assert bundle_report["cue_id"] == variant["variant_id"]
    failure = _run(
        [
            str(script),
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            "0" * 64,
        ],
        workspace,
        check=False,
        env=environment,
    )
    assert failure.returncode == 2
    assert b"SHA-256 mismatch" in failure.stderr
    partial = _run(
        [
            str(script),
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            set_sha,
            "--repo-root",
            str(repo),
        ],
        workspace,
        check=False,
        env=environment,
    )
    assert partial.returncode == 2
    assert b"together" in partial.stderr
    return cli_output, pids, model_config, encoder_config


def _check_cli_inprocess(
    workspace: Path,
    api_output: Path,
    set_sha: str,
    bundle_path: Path,
    bundle_sha: str,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    near_duplicate: tuple[Path, Path, str],
    model_config: Path,
    encoder_config: Path,
) -> None:
    stdout, _ = _module_cli(
        [
            "materialize",
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
            "--output-dir",
            str(workspace / "inprocess-output"),
            "--model-config",
            str(model_config),
            "--encoder-config",
            str(encoder_config),
        ],
        0,
    )
    assert json.loads(stdout)["file_sha256"] == set_sha
    near_repo, near_universe_path, near_universe_sha = near_duplicate
    _, stderr = _module_cli(
        [
            "materialize",
            "--repo-root",
            str(near_repo),
            "--source-universe",
            str(near_universe_path),
            "--source-universe-sha256",
            near_universe_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
            "--output-dir",
            str(workspace / "inprocess-near-dup"),
        ],
        2,
    )
    assert "near-duplicate cues" in stderr
    stdout, _ = _module_cli(
        [
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            set_sha,
        ],
        0,
    )
    assert json.loads(stdout)["source_verified"] is False
    stdout, _ = _module_cli(
        [
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            set_sha,
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
        ],
        0,
    )
    assert json.loads(stdout)["source_verified"] is True
    _, stderr = _module_cli(
        [
            "validate-set",
            "--cue-set",
            str(api_output / "cue_set.json"),
            "--cue-set-sha256",
            set_sha,
            "--repo-root",
            str(repo),
        ],
        2,
    )
    assert "together" in stderr
    stdout, _ = _module_cli(
        ["validate-bundle", "--bundle", str(bundle_path), "--bundle-sha256", bundle_sha],
        0,
    )
    assert json.loads(stdout)["encoder_verified"] is False
    stdout, _ = _module_cli(
        [
            "validate-bundle",
            "--bundle",
            str(bundle_path),
            "--bundle-sha256",
            bundle_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
        ],
        0,
    )
    assert json.loads(stdout)["encoder_verified"] is True
    _, stderr = _module_cli(
        ["validate-bundle", "--bundle", str(bundle_path), "--encoder-digest", "0" * 64],
        2,
    )
    assert "together" in stderr
    _runpy_main(
        ["validate-bundle", "--bundle", str(bundle_path), "--bundle-sha256", bundle_sha],
        0,
    )
    _runpy_main(["validate-bundle", "--bundle", str(workspace / "missing.json")], 2)


def _check_repo_failures(
    workspace: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> None:
    def attempt(target_repo: Path, name: str, **overrides: Any) -> Callable[[], object]:
        return lambda: materialize_cue_set(
            target_repo,
            overrides.get("universe_path", universe_path),
            overrides.get("universe_sha", universe_sha),
            overrides.get("encoder_checkpoint", encoder_checkpoint),
            overrides.get("encoder_digest", encoder_digest),
            workspace / name,
            model=overrides.get("model", MODEL),
            encoder_config=ENCODER_CONFIG,
            input_current=overrides.get("input_current", INPUT_CURRENT),
        )

    link = workspace / "repo-link"
    link.symlink_to(repo)
    _expect("absolute canonical path", attempt(link, "symlinked-repo"))
    _expect("is not the Git top level", attempt(repo / "docs", "subdir-repo"))
    non_git = workspace / "not-a-git-repo"
    non_git.mkdir()
    _expect("git rev-parse --show-toplevel failed", attempt(non_git, "non-git-repo"))
    sha256_repo = _init(workspace, "sha256-repo", "--object-format=sha256")
    _write(sha256_repo, "note.md", b"Sha256 object format repository.\n")
    _commit(sha256_repo, "fixture", BASE_TIMESTAMP)
    _expect("HEAD is not a SHA-1 commit", attempt(sha256_repo, "sha256-repo-out"))
    moved = workspace / "moved-head"
    _run(["git", "clone", "-q", str(repo), str(moved)], workspace)
    _run(["git", "config", "user.name", "D2 Gate"], moved)
    _run(["git", "config", "user.email", "d2@example.invalid"], moved)
    _write(moved, "docs/public/new-note.md", b"Moved head content.\n")
    _commit(moved, "move head", BASE_TIMESTAMP + 60)
    _expect("HEAD differs from the source-universe artifact", attempt(moved, "moved-out"))
    _expect(
        "input current must be a finite positive number",
        attempt(repo, "zero-current", input_current=0.0),
    )
    _expect(
        "input current must be a finite positive number",
        attempt(repo, "nan-current", input_current=float("nan")),
    )
    _expect(
        "pinned encoder directory digest mismatch",
        attempt(repo, "wrong-encoder", encoder_digest="0" * 64),
    )
    _expect(
        "pinned encoder checkpoint directory is missing",
        attempt(repo, "missing-encoder", encoder_checkpoint=workspace / "absent-encoder"),
    )
    _expect(
        "source-universe file SHA-256 mismatch",
        attempt(repo, "wrong-universe-sha", universe_sha="0" * 64),
    )


def _check_replace_attack(
    workspace: Path,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> None:
    selected = json.loads(universe_path.read_bytes())["selected"][0]
    original = (repo / selected["path"]).read_bytes()
    forged = (
        _run(
            ["git", "hash-object", "-w", "--stdin"],
            repo,
            input_bytes=original.replace(b"Record", b"Forged"),
        )
        .stdout.decode()
        .strip()
    )
    _git(repo, "replace", selected["blob_oid"], forged)
    try:
        _expect(
            "selected source bytes differ from the artifact",
            lambda: materialize_cue_set(
                repo,
                universe_path,
                universe_sha,
                encoder_checkpoint,
                encoder_digest,
                workspace / "replaced-out",
                model=MODEL,
                encoder_config=ENCODER_CONFIG,
            ),
        )
    finally:
        _git(repo, "replace", "-d", selected["blob_oid"])


def _craft_universe(base_bytes: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    payload = json.loads(base_bytes.decode("utf-8"))
    mutate(payload)
    payload["self_sha256"] = _self_digest(payload, UNIVERSE_DOMAIN)
    return _canonical(payload)


def _check_forged_universe(
    workspace: Path,
    repo: Path,
    universe_path: Path,
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> None:
    base_bytes = universe_path.read_bytes()

    def attempt(name: str, mutate: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        forged = _craft_universe(base_bytes, mutate)
        forged_path = workspace / f"forged-{name}.json"
        forged_path.write_bytes(forged)
        validate_source_universe_bytes(forged, expected_file_sha256=_sha(forged))
        return lambda: materialize_cue_set(
            repo,
            forged_path,
            _sha(forged),
            encoder_checkpoint,
            encoder_digest,
            workspace / f"forged-{name}-out",
            model=MODEL,
            encoder_config=ENCODER_CONFIG,
        )

    def forge_blob_oid(payload: dict[str, Any]) -> None:
        target = payload["selected"][0]
        path = target["normalized_path"]
        for item in (*payload["considered"], *payload["selected"]):
            if item["normalized_path"] == path:
                item["blob_oid"] = "f" * 40

    _expect("selected blob identity differs", attempt("blob-oid", forge_blob_oid))

    def forge_event_count(payload: dict[str, Any]) -> None:
        target = payload["selected"][1]
        path = target["normalized_path"]
        for item in (*payload["considered"], *payload["selected"]):
            if item["normalized_path"] == path:
                item["event_count"] = item["event_count"] + 1
                item["event_sha256"] = [*item["event_sha256"], "0" * 64]

    _expect("event count differs", attempt("event-count", forge_event_count))

    def forge_event_order(payload: dict[str, Any]) -> None:
        target = payload["selected"][2]
        path = target["normalized_path"]
        for item in (*payload["considered"], *payload["selected"]):
            if item["normalized_path"] == path:
                hashes = list(item["event_sha256"])
                hashes[0], hashes[1] = hashes[1], hashes[0]
                item["event_sha256"] = hashes

    _expect("event order differs", attempt("event-order", forge_event_order))


def _check_pathological_repos(
    workspace: Path,
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> tuple[Path, Path, str]:
    cases = (
        ("near-dup", _near_duplicate_doc(), "near-duplicate cues"),
        ("uniform", _uniform_doc(), "duplicate cue text across the materialised set"),
        ("nfkc", _nfkc_collision_doc(), "duplicate normalized cue text across the set"),
    )
    near_duplicate: tuple[Path, Path, str] | None = None
    for name, doc, message in cases:
        repo = _fixture_repo(workspace, f"{name}-repo", {7: doc})
        universe_path, universe_sha = _universe(repo, workspace, name)
        if name == "near-dup":
            near_duplicate = (repo, universe_path, universe_sha)
        _expect(
            message,
            lambda repo=repo, universe_path=universe_path, universe_sha=universe_sha, name=name: (
                materialize_cue_set(
                    repo,
                    universe_path,
                    universe_sha,
                    encoder_checkpoint,
                    encoder_digest,
                    workspace / f"{name}-out",
                )
            ),
        )
    assert near_duplicate is not None
    return near_duplicate


def _check_cli_default_configs(
    workspace: Path,
    install_target: Path,
    near_duplicate: tuple[Path, Path, str],
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> int:
    repo, universe_path, universe_sha = near_duplicate
    script = _console_script(install_target)
    site = Path(cue_materializer.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    process = subprocess.Popen(
        [
            str(script),
            "materialize",
            "--repo-root",
            str(repo),
            "--source-universe",
            str(universe_path),
            "--source-universe-sha256",
            universe_sha,
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--encoder-digest",
            encoder_digest,
            "--output-dir",
            str(workspace / "cli-default-out"),
        ],
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _stdout, stderr = process.communicate()
    assert process.returncode == 2, stderr.decode()
    assert b"near-duplicate cues" in stderr
    return process.pid


def _check_module_identity(
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    encoder_checkpoint: Path,
    encoder_digest: str,
    workspace: Path,
) -> None:
    import compileall
    import importlib

    from snn_memory import encoder as installed_encoder

    def attempt(name: str) -> object:
        return materialize_cue_set(
            repo,
            universe_path,
            universe_sha,
            encoder_checkpoint,
            encoder_digest,
            workspace / f"identity-{name}",
            model=MODEL,
            encoder_config=ENCODER_CONFIG,
        )

    module_path = Path(str(cue_materializer.__file__))
    encoder_path = Path(str(installed_encoder.__file__))
    module_bytes = module_path.read_bytes()
    encoder_bytes = encoder_path.read_bytes()
    try:
        encoder_path.write_bytes(b"\xff\xfe")
        _expect("implementation source is not strict UTF-8", lambda: attempt("utf8"))
    finally:
        encoder_path.write_bytes(encoder_bytes)
    try:
        module_path.unlink()
        _expect("implementation source cannot be read", lambda: attempt("unlink"))
    finally:
        module_path.write_bytes(module_bytes)
    bytecode = encoder_path.with_suffix(".pyc")
    try:
        assert compileall.compile_file(str(encoder_path), quiet=1, legacy=True)
        encoder_path.unlink()
        importlib.invalidate_caches()
        importlib.reload(installed_encoder)
        _expect("not a real non-shadowed source file", lambda: attempt("sourceless"))
    finally:
        bytecode.unlink(missing_ok=True)
        encoder_path.write_bytes(encoder_bytes)
        importlib.invalidate_caches()
        importlib.reload(installed_encoder)
    package_dir = module_path.parent
    relocated = package_dir.with_name("snn_memory_relocated")
    try:
        package_dir.rename(relocated)
        package_dir.symlink_to(relocated.name)
        _expect("implementation source traverses a symlink", lambda: attempt("symlink"))
    finally:
        if package_dir.is_symlink():
            package_dir.unlink()
        relocated.rename(package_dir)


def _craft_set(base_bytes: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    payload = json.loads(base_bytes.decode("utf-8"))
    mutate(payload)
    payload["self_sha256"] = _self_digest(payload, CUE_SET_DOMAIN)
    return _canonical(payload)


def _check_crafted_manifests(base_bytes: bytes) -> None:
    plain = json.loads(base_bytes.decode("utf-8"))
    assert validate_cue_set_bytes(base_bytes).payload_self_sha256 == plain["self_sha256"]
    _expect(
        "cue-set file SHA-256 mismatch",
        lambda: validate_cue_set_bytes(base_bytes, expected_file_sha256="0" * 64),
    )
    _expect(
        "duplicate JSON key",
        lambda: validate_cue_set_bytes(
            b'{"artifact_type":"snn-memory-cue-set","artifact_type":"snn-memory-cue-set"}'
        ),
    )
    _expect(
        "non-finite JSON constant",
        lambda: validate_cue_set_bytes(b'{"schema_version":Infinity}'),
    )
    _expect(
        "non-finite JSON number",
        lambda: validate_cue_set_bytes(b'{"schema_version":1e999}'),
    )
    _expect("is not strict UTF-8 JSON", lambda: validate_cue_set_bytes(b"\xff\xfe"))
    _expect("root must be an object", lambda: validate_cue_set_bytes(b"[]\n"))
    _expect(
        "schema validation failed",
        lambda: validate_cue_set_bytes(_craft_set(base_bytes, lambda p: p.update(extra=1))),
    )
    _expect(
        "not canonical",
        lambda: validate_cue_set_bytes(base_bytes[:-1] + b" \n"),
    )

    unsigned = json.loads(base_bytes.decode("utf-8"))
    unsigned["self_sha256"] = "0" * 64
    _expect("self digest mismatch", lambda: validate_cue_set_bytes(_canonical(unsigned)))

    def craft(mutate: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        return lambda: validate_cue_set_bytes(_craft_set(base_bytes, mutate))

    def record(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
        return payload["records"][index]

    def base_cue(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
        return record(payload)["evaluation_base_cues"][index]

    def rotate_records(payload: dict[str, Any]) -> None:
        payload["records"].append(payload["records"].pop(0))

    _expect("record order differs from the selected record IDs", craft(rotate_records))
    _expect(
        "cue-set encoder configuration digest mismatch",
        craft(lambda p: p["encoder"].update(config_digest="0" * 64)),
    )

    def invalid_encoder_config(payload: dict[str, Any]) -> None:
        payload["encoder"]["config"]["feature_dim"] = float(
            payload["encoder"]["config"]["feature_dim"]
        )
        payload["encoder"]["config_digest"] = _config_digest(payload["encoder"]["config"])

    _expect("cue-set encoder configuration is invalid", craft(invalid_encoder_config))
    _expect(
        "cue-set model configuration digest mismatch",
        craft(lambda p: p["model"].update(config_digest="0" * 64)),
    )

    def invalid_model_config(payload: dict[str, Any]) -> None:
        payload["model"]["config"]["n_neurons"] = float(payload["model"]["config"]["n_neurons"])
        payload["model"]["config_digest"] = _config_digest(payload["model"]["config"])

    _expect("cue-set model configuration is invalid", craft(invalid_model_config))
    _expect(
        "cue-set neuron count must be a JSON integer",
        craft(lambda p: p["model"].update(n_neurons=64.0)),
    )

    def neuron_count_mismatch(payload: dict[str, Any]) -> None:
        payload["model"]["config"]["n_neurons"] = 66
        payload["model"]["config_digest"] = _config_digest(payload["model"]["config"])
        payload["model"]["n_neurons"] = 68

    _expect(
        "cue-set model neuron count differs from its configuration",
        craft(neuron_count_mismatch),
    )
    _expect(
        "implementation source base64 is invalid",
        craft(lambda p: p["implementations"]["split_events"].update(bytes_base64="?!")),
    )
    _expect(
        "implementation source byte binding differs",
        craft(lambda p: p["implementations"]["split_events"].update(sha256="0" * 64)),
    )
    _expect(
        "record event count must be a JSON integer",
        craft(lambda p: record(p).update(event_count=float(record(p)["event_count"]))),
    )
    _expect(
        "calibration block index differs from its record digest",
        craft(
            lambda p: record(p).update(
                calibration_block_index=(record(p)["calibration_block_index"] + 1) % 5
            )
        ),
    )
    _expect(
        "family permutation index differs from its record digest",
        craft(
            lambda p: record(p).update(
                family_permutation_index=(record(p)["family_permutation_index"] + 1) % 24
            )
        ),
    )

    def leak_calibration(payload: dict[str, Any]) -> None:
        target = record(payload)
        leaked = "cue-" + target["record_id"].removeprefix("sha256:")[:32]
        target["calibration_cue"]["cue_id"] = leaked
        target["calibration_cue"]["path"] = f"cues/{leaked}.txt"

    _expect("cue identity leaks its record ID", craft(leak_calibration))

    def leak_variant(payload: dict[str, Any]) -> None:
        target = base_cue(payload)["variants"][1]
        leaked = "variant-" + record(payload)["record_id"].removeprefix("sha256:")[:32]
        target["variant_id"] = leaked
        target["path"] = f"cues/{leaked}.txt"
        target["bundle"]["path"] = f"bundles/{leaked}.json"

    _expect("variant identity leaks its record ID", craft(leak_variant))
    _expect(
        "calibration event indices differ from the frozen partition",
        craft(
            lambda p: record(p)["calibration_cue"].update(
                event_indices=[index + 1 for index in record(p)["calibration_cue"]["event_indices"]]
            )
        ),
    )
    _expect(
        "calibration cue ID differs from its derivation",
        craft(lambda p: record(p)["calibration_cue"].update(sha256="f" * 64)),
    )

    def calibration_path(payload: dict[str, Any]) -> None:
        other = record(payload, 1)["calibration_cue"]["cue_id"]
        record(payload)["calibration_cue"]["path"] = f"cues/{other}.txt"

    _expect("calibration path differs from its opaque cue ID", craft(calibration_path))

    def duplicate_calibration(payload: dict[str, Any]) -> None:
        source = record(payload)["calibration_cue"]
        target = record(payload, 1)["calibration_cue"]
        for field in ("cue_id", "path", "sha256", "normalized_text_sha256"):
            target[field] = source[field]

    _expect("duplicate cue ID across the cue set", craft(duplicate_calibration))

    def duplicate_family(payload: dict[str, Any]) -> None:
        base_cue(payload)["transform_family"] = base_cue(payload, 1)["transform_family"]

    _expect("transform family differs from the frozen permutation", craft(duplicate_family))
    _expect(
        "evaluation event indices differ from the partition",
        craft(
            lambda p: base_cue(p).update(
                event_indices=[index + 1 for index in base_cue(p)["event_indices"]]
            )
        ),
    )
    _expect(
        "base cue ID differs from its derivation",
        craft(lambda p: base_cue(p).update(cue_id=_cue_id("f" * 64))),
    )
    _expect(
        "base cue token count must be a JSON integer",
        craft(lambda p: base_cue(p).update(token_count=float(base_cue(p)["token_count"]))),
    )

    def variant_id_forgery(payload: dict[str, Any]) -> None:
        target = base_cue(payload)["variants"][1]
        target["variant_id"] = _variant_id("f" * 64, base_cue(payload)["transform_family"], 10)
        target["path"] = f"cues/{target['variant_id']}.txt"
        target["bundle"]["path"] = f"bundles/{target['variant_id']}.json"

    _expect("variant ID differs from its derivation", craft(variant_id_forgery))

    def variant_path_forgery(payload: dict[str, Any]) -> None:
        other = base_cue(payload, 1)["variants"][1]["variant_id"]
        base_cue(payload)["variants"][1]["path"] = f"cues/{other}.txt"

    _expect("variant path differs from its opaque variant ID", craft(variant_path_forgery))

    def bundle_path_forgery(payload: dict[str, Any]) -> None:
        other = base_cue(payload, 1)["variants"][1]["variant_id"]
        base_cue(payload)["variants"][1]["bundle"]["path"] = f"bundles/{other}.json"

    _expect("bundle path differs from its opaque variant ID", craft(bundle_path_forgery))
    _expect(
        "selected positions differ from the frozen derivation",
        craft(
            lambda p: base_cue(p)["variants"][1].update(
                selected_positions=base_cue(p)["variants"][2]["selected_positions"]
            )
        ),
    )
    _expect(
        "affected count differs from the frozen rounding rule",
        craft(
            lambda p: base_cue(p)["variants"][1].update(
                affected_count=base_cue(p)["variants"][1]["affected_count"] + 1
            )
        ),
    )
    _expect(
        "realized fraction differs from its counts",
        craft(
            lambda p: base_cue(p)["variants"][1].update(
                realized_fraction=base_cue(p)["variants"][1]["realized_fraction"] / 2
            )
        ),
    )

    def duplicate_variant_text(payload: dict[str, Any]) -> None:
        left = base_cue(payload)["variants"][1]
        right = base_cue(payload, 1)["variants"][1]
        right["sha256"] = left["sha256"]

    _expect("duplicate cue text digest across the cue set", craft(duplicate_variant_text))

    def duplicate_normalized(payload: dict[str, Any]) -> None:
        left = base_cue(payload)["variants"][1]
        right = base_cue(payload, 1)["variants"][1]
        right["normalized_text_sha256"] = left["normalized_text_sha256"]

    _expect("duplicate normalized cue digest across the cue set", craft(duplicate_normalized))

    def duplicate_bundle(payload: dict[str, Any]) -> None:
        left = base_cue(payload)["variants"][1]
        right = base_cue(payload, 1)["variants"][1]
        right["bundle"]["sha256"] = left["bundle"]["sha256"]

    _expect("duplicate bundle digest across the cue set", craft(duplicate_bundle))


def _config_digest(config: dict[str, Any]) -> str:
    raw = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return _sha(raw)


def _clone_set(source: Path, workspace: Path, name: str) -> tuple[Path, dict[str, Any]]:
    import shutil

    destination = workspace / name
    shutil.copytree(source, destination)
    payload = json.loads((destination / "cue_set.json").read_bytes())
    return destination, payload


def _reseal_set(destination: Path, payload: dict[str, Any]) -> str:
    payload["self_sha256"] = _self_digest(payload, CUE_SET_DOMAIN)
    raw = _canonical(payload)
    (destination / "cue_set.json").unlink()
    (destination / "cue_set.json").write_bytes(raw)
    return _sha(raw)


def _craft_bundle(base_bytes: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    payload = json.loads(base_bytes.decode("utf-8"))
    mutate(payload)
    payload["self_sha256"] = _self_digest(payload, BUNDLE_DOMAIN)
    return _canonical(payload)


def _swap_bundle_into_set(destination: Path, payload: dict[str, Any], forged_bundle: bytes) -> str:
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    bundle_path = destination / variant["bundle"]["path"]
    bundle_path.unlink()
    bundle_path.write_bytes(forged_bundle)
    variant["bundle"]["sha256"] = _sha(forged_bundle)
    return _reseal_set(destination, payload)


def _retarget_calibration(destination: Path, calibration: dict[str, Any], raw: bytes) -> None:
    calibration["sha256"] = _sha(raw)
    calibration["cue_id"] = _cue_id(calibration["sha256"])
    calibration["path"] = f"cues/{calibration['cue_id']}.txt"
    (destination / calibration["path"]).write_bytes(raw)


def _check_file_level_attacks(
    workspace: Path, happy_output: Path, set_sha: str, lexicon: list[str]
) -> None:
    def read(destination: Path, digest: str) -> Callable[[], object]:
        return lambda: read_cue_set(destination / "cue_set.json", digest)

    destination, payload = _clone_set(happy_output, workspace, "attack-lexicon")
    with (destination / "noise_lexicon.txt").open("ab") as handle:
        handle.write(b"extra\n")
    _expect("noise lexicon digest mismatch", read(destination, set_sha))

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-bytes")
    calibration = payload["records"][0]["calibration_cue"]
    with (destination / calibration["path"]).open("ab") as handle:
        handle.write(b"!")
    _expect("calibration cue digest mismatch", read(destination, set_sha))

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-utf8")
    calibration = payload["records"][0]["calibration_cue"]
    _retarget_calibration(destination, calibration, b"\xff\xfe")
    _expect(
        "calibration cue is not strict UTF-8",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-normalized")
    calibration = payload["records"][0]["calibration_cue"]
    calibration["normalized_text_sha256"] = "f" * 64
    _expect(
        "calibration cue normalized-text digest mismatch",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-canonical")
    calibration = payload["records"][0]["calibration_cue"]
    original = (destination / calibration["path"]).read_text(encoding="utf-8")
    tampered = original.replace(" ", "  ", 1)
    _retarget_calibration(destination, calibration, tampered.encode("utf-8"))
    calibration["normalized_text_sha256"] = _norm_sha(tampered)
    _expect(
        "calibration cue text is not canonical",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-lines")
    calibration = payload["records"][0]["calibration_cue"]
    original = (destination / calibration["path"]).read_text(encoding="utf-8")
    merged = original.replace("\n", " ", 1)
    _retarget_calibration(destination, calibration, merged.encode("utf-8"))
    calibration["normalized_text_sha256"] = _norm_sha(merged)
    _expect(
        "calibration cue lines differ from its event indices",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-token-count")
    base_cue = payload["records"][0]["evaluation_base_cues"][0]
    base_cue["token_count"] += 1
    base_sha = base_cue["variants"][0]["sha256"]
    family = base_cue["transform_family"]
    for variant in base_cue["variants"]:
        selected = _selected(
            family, base_sha, base_cue["token_count"], variant["requested_percent"]
        )
        variant["selected_positions"] = selected
        variant["affected_count"] = len(selected)
        variant["realized_fraction"] = len(selected) / base_cue["token_count"]
    _expect(
        "base cue token count differs from its text",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-variant-derivation")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][2]
    forged_text = "Forged replacement text stays canonical.\nSecond forged line remains."
    (destination / variant["path"]).unlink()
    (destination / variant["path"]).write_text(forged_text, encoding="utf-8")
    variant["sha256"] = _sha(forged_text.encode("utf-8"))
    variant["normalized_text_sha256"] = _norm_sha(forged_text)
    _expect(
        "variant cue text differs from its derivation",
        read(destination, _reseal_set(destination, payload)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-symlink")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][0]
    target = destination / variant["path"]
    aside = target.with_name("aside.txt")
    target.rename(aside)
    target.symlink_to(aside.name)
    _expect("base cue path traverses a symlink", read(destination, set_sha))

    destination, payload = _clone_set(happy_output, workspace, "attack-missing-bundle")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    (destination / variant["bundle"]["path"]).unlink()
    _expect("cue bundle path escapes or is missing", read(destination, set_sha))

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-tamper")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    with (destination / variant["bundle"]["path"]).open("ab") as handle:
        handle.write(b" ")
    _expect("cue bundle digest mismatch", read(destination, set_sha))

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-cueid")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    other = payload["records"][0]["evaluation_base_cues"][0]["variants"][2]["variant_id"]
    bundle_bytes = (destination / variant["bundle"]["path"]).read_bytes()
    forged = _craft_bundle(bundle_bytes, lambda p: p.update(cue_id=other))
    _expect(
        "bundle cue ID differs from its variant",
        read(destination, _swap_bundle_into_set(destination, payload, forged)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-encoder")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    bundle_bytes = (destination / variant["bundle"]["path"]).read_bytes()
    forged = _craft_bundle(bundle_bytes, lambda p: p["encoder"].update(identity="other-model"))
    _expect(
        "bundle encoder differs from the cue set",
        read(destination, _swap_bundle_into_set(destination, payload, forged)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-model")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    bundle_bytes = (destination / variant["bundle"]["path"]).read_bytes()
    forged = _craft_bundle(bundle_bytes, lambda p: p["model"].update(config_digest="f" * 64))
    _expect(
        "bundle model differs from the cue set",
        read(destination, _swap_bundle_into_set(destination, payload, forged)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-impl")
    variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    bundle_bytes = (destination / variant["bundle"]["path"]).read_bytes()
    forged = _craft_bundle(
        bundle_bytes, lambda p: p["implementations"]["split_events"].update(sha256="f" * 64)
    )
    _expect(
        "bundle implementations differ from the cue set",
        read(destination, _swap_bundle_into_set(destination, payload, forged)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-bundle-text")
    base_cue = None
    for candidate in payload["records"][0]["evaluation_base_cues"]:
        if candidate["transform_family"] == "masking":
            base_cue = candidate
    assert base_cue is not None
    variant = base_cue["variants"][1]
    donor = base_cue["variants"][2]
    donor_text = (destination / donor["path"]).read_text(encoding="utf-8")
    bundle_bytes = (destination / variant["bundle"]["path"]).read_bytes()

    def swap_text(bundle_payload: dict[str, Any]) -> None:
        raw = donor_text.encode("utf-8")
        bundle_payload["text_utf8_base64"] = base64.b64encode(raw).decode("ascii")
        bundle_payload["text_sha256"] = _sha(raw)
        bundle_payload["normalized_text_sha256"] = _norm_sha(donor_text)

    forged = _craft_bundle(bundle_bytes, swap_text)
    _expect(
        "bundle text differs from its variant cue",
        read(destination, _swap_bundle_into_set(destination, payload, forged)),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-near-dup")
    events = [line for line in _near_duplicate_doc().decode("utf-8").splitlines() if line]
    record = payload["records"][3]
    item = {
        "record_id": record["record_id"],
        "content_sha256": record["record_id"].removeprefix("sha256:"),
        "event_count": len(events),
    }
    entry, texts = _expected_record(item, events, lexicon)
    for relative, text in texts.items():
        path = destination / relative
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    for forged_base in entry["evaluation_base_cues"]:
        for forged_variant in forged_base["variants"]:
            forged_variant["bundle"] = {
                "path": f"bundles/{forged_variant['variant_id']}.json",
                "sha256": _sha(forged_variant["variant_id"].encode("ascii")),
            }
    payload["records"][3] = entry
    _expect("near-duplicate cues", read(destination, _reseal_set(destination, payload)))

    destination, payload = _clone_set(happy_output, workspace, "attack-cal-empty")
    calibration = payload["records"][0]["calibration_cue"]
    _retarget_calibration(destination, calibration, b"")
    calibration["normalized_text_sha256"] = _norm_sha("")
    _expect(
        "calibration cue text is empty",
        read(destination, _reseal_set(destination, payload)),
    )


def _check_crafted_bundles(base_bytes: bytes, input_current: float) -> None:
    base_payload = json.loads(base_bytes.decode("utf-8"))
    assert validate_cue_bundle_bytes(base_bytes).payload_self_sha256 == base_payload["self_sha256"]
    _expect(
        "cue-bundle file SHA-256 mismatch",
        lambda: validate_cue_bundle_bytes(base_bytes, expected_file_sha256="0" * 64),
    )
    _expect(
        "duplicate JSON key",
        lambda: validate_cue_bundle_bytes(b'{"cue_id":"a","cue_id":"b"}'),
    )
    _expect(
        "schema validation failed",
        lambda: validate_cue_bundle_bytes(_craft_bundle(base_bytes, lambda p: p.update(extra=1))),
    )
    _expect("not canonical", lambda: validate_cue_bundle_bytes(base_bytes[:-1] + b" \n"))
    unsigned = json.loads(base_bytes.decode("utf-8"))
    unsigned["self_sha256"] = "0" * 64
    _expect(
        "cue-bundle self digest mismatch",
        lambda: validate_cue_bundle_bytes(_canonical(unsigned)),
    )

    def craft(mutate: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        return lambda: validate_cue_bundle_bytes(_craft_bundle(base_bytes, mutate))

    _expect(
        "cue-bundle text base64 is invalid",
        craft(lambda p: p.update(text_utf8_base64="?!")),
    )
    _expect(
        "cue-bundle text digest mismatch",
        craft(lambda p: p.update(text_sha256="f" * 64)),
    )
    _expect(
        "cue-bundle normalized-text digest mismatch",
        craft(lambda p: p.update(normalized_text_sha256="f" * 64)),
    )

    def double_space(payload: dict[str, Any]) -> None:
        text = base64.b64decode(payload["text_utf8_base64"]).decode("utf-8")
        tampered = text.replace(" ", "  ", 1).encode("utf-8")
        payload["text_utf8_base64"] = base64.b64encode(tampered).decode("ascii")
        payload["text_sha256"] = _sha(tampered)
        payload["normalized_text_sha256"] = _norm_sha(tampered.decode("utf-8"))

    _expect("cue-bundle cue text is not canonical", craft(double_space))
    _expect(
        "cue-bundle encoder configuration digest mismatch",
        craft(lambda p: p["encoder"].update(config_digest="f" * 64)),
    )

    def invalid_encoder_config(payload: dict[str, Any]) -> None:
        payload["encoder"]["config"]["feature_dim"] = float(
            payload["encoder"]["config"]["feature_dim"]
        )
        payload["encoder"]["config_digest"] = _config_digest(payload["encoder"]["config"])

    _expect("cue-bundle encoder configuration is invalid", craft(invalid_encoder_config))
    _expect(
        "cue-bundle neuron count must be a JSON integer",
        craft(lambda p: p["model"].update(n_neurons=float(p["model"]["n_neurons"]))),
    )
    _expect(
        "cue-bundle array shape must be a JSON integer",
        craft(lambda p: p["embedding"]["shape"].__setitem__(1, float(p["embedding"]["shape"][1]))),
    )
    _expect(
        "cue-bundle embedding rows differ from cue events",
        craft(lambda p: p["embedding"]["shape"].__setitem__(0, p["embedding"]["shape"][0] + 1)),
    )
    _expect(
        "cue-bundle embedding byte length differs from its declared shape",
        craft(lambda p: p["embedding"]["shape"].__setitem__(1, p["embedding"]["shape"][1] + 1)),
    )

    def non_finite_embedding(payload: dict[str, Any]) -> None:
        raw = bytearray(base64.b64decode(payload["embedding"]["data_base64"]))
        raw[0:8] = np.array([np.nan], dtype="<f8").tobytes()
        payload["embedding"]["data_base64"] = base64.b64encode(bytes(raw)).decode("ascii")

    _expect("cue-bundle embedding contains non-finite values", craft(non_finite_embedding))
    _expect(
        "cue-bundle embedding base64 payload is invalid",
        craft(lambda p: p["embedding"].update(data_base64="?!")),
    )
    _expect(
        "cue-bundle current shape differs from its cue and model",
        craft(lambda p: p["currents"]["shape"].__setitem__(0, p["currents"]["shape"][0] + 7)),
    )

    def short_rows(payload: dict[str, Any]) -> None:
        raw = base64.b64decode(payload["currents"]["row_base64"])
        payload["currents"]["row_base64"] = base64.b64encode(raw[:-8]).decode("ascii")

    _expect("cue-bundle rows byte length differs from its declared shape", craft(short_rows))

    def wrong_value(payload: dict[str, Any]) -> None:
        values = np.frombuffer(
            base64.b64decode(payload["currents"]["value_base64"]), dtype="<f8"
        ).copy()
        values[0] = input_current + 1.0
        payload["currents"]["value_base64"] = base64.b64encode(values.tobytes()).decode("ascii")

    _expect("cue-bundle current values differ from the input current", craft(wrong_value))

    def _currents(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
        rows = np.frombuffer(base64.b64decode(payload["currents"]["row_base64"]), dtype="<i8")
        columns = np.frombuffer(base64.b64decode(payload["currents"]["column_base64"]), dtype="<i8")
        values = np.frombuffer(base64.b64decode(payload["currents"]["value_base64"]), dtype="<f8")
        return rows.copy(), columns.copy(), values.copy()

    def _store(payload: dict[str, Any], rows: Any, columns: Any, values: Any) -> None:
        payload["currents"]["row_base64"] = base64.b64encode(rows.tobytes()).decode("ascii")
        payload["currents"]["column_base64"] = base64.b64encode(columns.tobytes()).decode("ascii")
        payload["currents"]["value_base64"] = base64.b64encode(values.tobytes()).decode("ascii")

    def out_of_range_row(payload: dict[str, Any]) -> None:
        rows, columns, values = _currents(payload)
        rows[-1] = payload["currents"]["shape"][0] + 3
        _store(payload, rows, columns, values)

    _expect("cue-bundle current rows are out of range", craft(out_of_range_row))

    def out_of_range_column(payload: dict[str, Any]) -> None:
        rows, columns, values = _currents(payload)
        columns[-1] = payload["currents"]["shape"][1]
        _store(payload, rows, columns, values)

    _expect("cue-bundle current columns are out of range", craft(out_of_range_column))

    def unsorted_entries(payload: dict[str, Any]) -> None:
        rows, columns, values = _currents(payload)
        rows[[0, 1]] = rows[[1, 0]]
        columns[[0, 1]] = columns[[1, 0]]
        _store(payload, rows, columns, values)

    _expect("cue-bundle current entries are not strictly sorted", craft(unsorted_entries))

    def silent_window_row(payload: dict[str, Any]) -> None:
        rows, columns, values = _currents(payload)
        period = (
            base_payload["encoder"]["config"]["packet_ms"]
            + base_payload["encoder"]["config"]["silent_ms"]
        )
        packet = base_payload["encoder"]["config"]["packet_ms"]
        event_end = rows[-1] - (rows[-1] % period) + packet
        rows[-1] = event_end
        _store(payload, rows, columns, values)

    _expect("cue-bundle current entries fall in a silent window", craft(silent_window_row))

    def unbalanced_events(payload: dict[str, Any]) -> None:
        rows, columns, values = _currents(payload)
        period = (
            base_payload["encoder"]["config"]["packet_ms"]
            + base_payload["encoder"]["config"]["silent_ms"]
        )
        packet = base_payload["encoder"]["config"]["packet_ms"]
        width = payload["currents"]["shape"][1]
        boundary = int(np.searchsorted(rows, period))
        occupied = set(zip(rows.tolist(), columns.tolist()))
        previous = (int(rows[boundary - 1]), int(columns[boundary - 1]))
        for row in range(packet - 1, -1, -1):
            for column in range(width - 1, -1, -1):
                candidate = (row, column)
                if candidate in occupied or candidate <= previous:
                    continue
                rows[boundary] = row
                columns[boundary] = column
                _store(payload, rows, columns, values)
                return
        raise AssertionError("fixture bundle does not admit the event-move forgery")

    _expect("cue-bundle per-event active counts differ from the contract", craft(unbalanced_events))


def _check_bundle_read_surface(workspace: Path, bundle_path: Path, bundle_sha: str) -> None:
    missing = workspace / "no-bundle.json"
    _expect("cue-bundle file cannot be opened safely", lambda: read_cue_bundle(missing))
    link = workspace / "bundle-link.json"
    link.symlink_to(bundle_path)
    _expect("cue-bundle file cannot be opened safely", lambda: read_cue_bundle(link))
    fifo = workspace / "bundle-fifo.json"
    os.mkfifo(fifo)
    _expect("cue-bundle file is not a regular file", lambda: read_cue_bundle(fifo))
    artifact = read_cue_bundle(bundle_path, bundle_sha)
    assert _sha(artifact.canonical_bytes) == bundle_sha


def _check_encoder_forgeries(
    workspace: Path,
    bundle_bytes: bytes,
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> None:
    bundle = validate_cue_bundle_bytes(bundle_bytes)
    _expect(
        "pinned encoder checkpoint directory is missing",
        lambda: verify_cue_bundle_with_encoder(
            bundle, workspace / "absent-encoder", encoder_digest
        ),
    )
    _expect(
        "pinned encoder directory digest mismatch",
        lambda: verify_cue_bundle_with_encoder(bundle, encoder_checkpoint, "0" * 64),
    )
    forged_pin = _craft_bundle(
        bundle_bytes, lambda p: p["encoder"].update(directory_sha256="f" * 64)
    )
    _expect(
        "cue-bundle encoder directory digest differs from the pin",
        lambda: verify_cue_bundle_with_encoder(
            validate_cue_bundle_bytes(forged_pin), encoder_checkpoint, encoder_digest
        ),
    )

    def swap_embedding_rows(payload: dict[str, Any]) -> None:
        shape = payload["embedding"]["shape"]
        matrix = (
            np.frombuffer(base64.b64decode(payload["embedding"]["data_base64"]), dtype="<f8")
            .reshape(shape[0], shape[1])
            .copy()
        )
        matrix[[0, 1]] = matrix[[1, 0]]
        payload["embedding"]["data_base64"] = base64.b64encode(matrix.tobytes()).decode("ascii")

    resealed = _craft_bundle(bundle_bytes, swap_embedding_rows)
    forged_bundle = validate_cue_bundle_bytes(resealed)
    _expect(
        "cue-bundle embedding differs from the pinned encoder output",
        lambda: verify_cue_bundle_with_encoder(forged_bundle, encoder_checkpoint, encoder_digest),
    )

    def move_latency(payload: dict[str, Any]) -> None:
        period = (
            payload["encoder"]["config"]["packet_ms"] + payload["encoder"]["config"]["silent_ms"]
        )
        packet = payload["encoder"]["config"]["packet_ms"]
        rows = np.frombuffer(
            base64.b64decode(payload["currents"]["row_base64"]), dtype="<i8"
        ).copy()
        columns = np.frombuffer(
            base64.b64decode(payload["currents"]["column_base64"]), dtype="<i8"
        ).copy()
        entries = list(zip(rows.tolist(), columns.tolist()))
        occupied = set(entries)
        for index, (row, column) in enumerate(entries):
            for delta in (1, -1):
                moved = row + delta
                if moved // period != row // period or moved % period >= packet or moved < 0:
                    continue
                if (moved, column) in occupied:
                    continue
                candidate = sorted(entries[:index] + [(moved, column)] + entries[index + 1 :])
                if candidate == entries[:index] + [(moved, column)] + entries[index + 1 :]:
                    rows[index] = moved
                    payload["currents"]["row_base64"] = base64.b64encode(rows.tobytes()).decode(
                        "ascii"
                    )
                    return
        raise AssertionError("fixture bundle does not admit a latency-move forgery")

    moved = _craft_bundle(bundle_bytes, move_latency)
    moved_bundle = validate_cue_bundle_bytes(moved)
    _expect(
        "cue-bundle currents differ from the pinned derivation",
        lambda: verify_cue_bundle_with_encoder(moved_bundle, encoder_checkpoint, encoder_digest),
    )


def _forge_bundle_bytes(
    template: dict[str, Any], variant_id: str, text: str, adapter: LocalSentenceEncoder
) -> bytes:
    events = _split_events(text)
    embedding = adapter.encode(events)
    currents = embeddings_to_currents(embedding, MODEL, ENCODER_CONFIG, input_current=INPUT_CURRENT)
    rows, columns = np.nonzero(currents)
    payload = json.loads(json.dumps(template))
    raw_text = text.encode("utf-8")
    payload["cue_id"] = variant_id
    payload["text_sha256"] = _sha(raw_text)
    payload["normalized_text_sha256"] = _norm_sha(text)
    payload["text_utf8_base64"] = base64.b64encode(raw_text).decode("ascii")
    payload["embedding"] = {
        "dtype": "<f8",
        "shape": [int(embedding.shape[0]), int(embedding.shape[1])],
        "data_base64": base64.b64encode(np.ascontiguousarray(embedding).tobytes()).decode("ascii"),
    }
    payload["currents"] = {
        "encoding": "coo-sorted-v1",
        "dtype": "<f8",
        "index_dtype": "<i8",
        "shape": [int(currents.shape[0]), int(currents.shape[1])],
        "row_base64": base64.b64encode(rows.astype("<i8").tobytes()).decode("ascii"),
        "column_base64": base64.b64encode(columns.astype("<i8").tobytes()).decode("ascii"),
        "value_base64": base64.b64encode(currents[rows, columns].astype("<f8").tobytes()).decode(
            "ascii"
        ),
    }
    payload["self_sha256"] = _self_digest(payload, BUNDLE_DOMAIN)
    return _canonical(payload)


def _check_against_sources(
    workspace: Path,
    happy_output: Path,
    set_sha: str,
    repo: Path,
    universe_path: Path,
    universe_sha: str,
    other_universe_path: Path,
    other_universe_sha: str,
    other_repo: Path,
    adapter: LocalSentenceEncoder,
    lexicon: list[str],
) -> None:
    happy = read_cue_set(happy_output / "cue_set.json", set_sha)
    universe = validate_source_universe_bytes(
        universe_path.read_bytes(), expected_file_sha256=universe_sha
    )
    other_universe = validate_source_universe_bytes(
        other_universe_path.read_bytes(), expected_file_sha256=other_universe_sha
    )
    verify_cue_set_against_sources(happy, universe, repo)
    _expect(
        "bound to a different source-universe artifact",
        lambda: verify_cue_set_against_sources(happy, other_universe, other_repo),
    )
    _expect(
        "repository HEAD differs from the cue-set binding",
        lambda: verify_cue_set_against_sources(happy, universe, other_repo),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-binding-head")
    payload["source_universe"]["repository_head"] = "f" * 40
    forged_sha = _reseal_set(destination, payload)
    forged = read_cue_set(destination / "cue_set.json", forged_sha)
    _expect(
        "cue-set repository head differs from the artifact",
        lambda: verify_cue_set_against_sources(forged, universe, repo),
    )

    destination, payload = _clone_set(happy_output, workspace, "attack-derivation")
    selected = json.loads(universe_path.read_bytes())["selected"][5]
    raw = (repo / selected["path"]).read_bytes()
    events = _split_events(raw.decode("utf-8"))[:-1]
    item = {
        "record_id": selected["record_id"],
        "content_sha256": selected["content_sha256"],
        "event_count": len(events),
    }
    entry, texts = _expected_record(item, events, lexicon)
    for relative, text in texts.items():
        path = destination / relative
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    template_ref = payload["records"][0]["evaluation_base_cues"][0]["variants"][0]["bundle"]
    template = json.loads((destination / template_ref["path"]).read_bytes())
    for forged_base in entry["evaluation_base_cues"]:
        for forged_variant in forged_base["variants"]:
            text = texts[forged_variant["path"]]
            bundle_bytes = _forge_bundle_bytes(
                template, forged_variant["variant_id"], text, adapter
            )
            bundle_relative = f"bundles/{forged_variant['variant_id']}.json"
            bundle_file = destination / bundle_relative
            if not bundle_file.exists():
                bundle_file.write_bytes(bundle_bytes)
            forged_variant["bundle"] = {"path": bundle_relative, "sha256": _sha(bundle_bytes)}
    payload["records"][5] = entry
    forged_sha = _reseal_set(destination, payload)
    forged = read_cue_set(destination / "cue_set.json", forged_sha)
    _expect(
        "record cue derivation differs from its sources",
        lambda: verify_cue_set_against_sources(forged, universe, repo),
    )


def _check_shifted_repo(
    workspace: Path,
    happy_payload: dict[str, Any],
    encoder_checkpoint: Path,
    encoder_digest: str,
) -> tuple[Path, Path, str]:
    shifted_repo = _init(workspace, "shifted-repo")
    for record in range(16):
        _write(shifted_repo, f"docs/public/record-{record:02d}.md", _standard_doc(record))
    _write_manifests(shifted_repo, _standard_doc(0).replace(b"Record 0", b"Source X"))
    _commit(shifted_repo, "fixture", BASE_TIMESTAMP + 7_200)
    universe_path, universe_sha = _universe(shifted_repo, workspace, "shifted")
    shifted_output = workspace / "shifted-output"
    result = materialize_cue_set(
        shifted_repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        shifted_output,
        model=MODEL,
        encoder_config=ENCODER_CONFIG,
        input_current=INPUT_CURRENT,
    )
    shifted_payload = json.loads((shifted_output / "cue_set.json").read_bytes())
    assert result.file_sha256 == _sha((shifted_output / "cue_set.json").read_bytes())
    assert (
        shifted_payload["source_universe"]["repository_head"]
        != happy_payload["source_universe"]["repository_head"]
    )

    shared_ids = {record["record_id"] for record in happy_payload["records"]} & {
        record["record_id"] for record in shifted_payload["records"]
    }
    assert len(shared_ids) == 13, f"expected 13 shared standard records, got {len(shared_ids)}"

    def cue_identity(payload: dict[str, Any]) -> list[tuple[str, ...]]:
        identity: list[tuple[str, ...]] = []
        for record in payload["records"]:
            if record["record_id"] not in shared_ids:
                continue
            identity.append((record["record_id"], record["calibration_cue"]["cue_id"]))
            for base_cue in record["evaluation_base_cues"]:
                for variant in base_cue["variants"]:
                    identity.append((record["record_id"], variant["variant_id"], variant["sha256"]))
        return identity

    shared_happy = cue_identity(happy_payload)
    shared_shifted = cue_identity(shifted_payload)
    assert shared_happy and sorted(shared_happy) == sorted(shared_shifted), (
        "cue derivation must not depend on commit time"
    )
    return shifted_repo, universe_path, universe_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--install-target", type=Path, required=True)
    parser.add_argument("--public-cue-set-schema", type=Path, required=True)
    parser.add_argument("--public-cue-set-license", type=Path, required=True)
    parser.add_argument("--public-cue-bundle-schema", type=Path, required=True)
    parser.add_argument("--public-cue-bundle-license", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-digest", required=True)
    arguments = parser.parse_args()
    module_origin = Path(str(cue_materializer.__file__)).resolve()
    assert module_origin.is_relative_to(arguments.install_target.resolve())
    schema_dir = module_origin.parent / "schema"
    for public, packaged in (
        (arguments.public_cue_set_schema, schema_dir / "snn_memory_cue_set_v2.schema.json"),
        (
            arguments.public_cue_set_license,
            schema_dir / "snn_memory_cue_set_v2.schema.json.license",
        ),
        (arguments.public_cue_bundle_schema, schema_dir / "snn_memory_cue_bundle_v2.schema.json"),
        (
            arguments.public_cue_bundle_license,
            schema_dir / "snn_memory_cue_bundle_v2.schema.json.license",
        ),
    ):
        assert public.read_bytes() == packaged.read_bytes(), f"schema copy drift: {public}"
    workspace: Path = arguments.workspace
    encoder_checkpoint: Path = arguments.encoder_checkpoint.resolve(strict=True)
    encoder_digest: str = arguments.encoder_digest

    repo = _fixture_repo(
        workspace,
        "happy-repo",
        {4: _unicode_doc(4), 9: _mask_collision_doc(9), 13: _wide_doc(13)},
    )
    universe_path, universe_sha = _universe(repo, workspace, "happy")
    output = workspace / "happy-output"
    result = materialize_cue_set(
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        output,
        model=MODEL,
        encoder_config=ENCODER_CONFIG,
        input_current=INPUT_CURRENT,
    )
    set_sha = result.file_sha256
    base_bytes = (output / "cue_set.json").read_bytes()
    assert set_sha == _sha(base_bytes)
    payload = json.loads(base_bytes.decode("utf-8"))
    lexicon = (output / "noise_lexicon.txt").read_text(encoding="utf-8").splitlines()

    _check_happy_manifest(repo, universe_path, payload, lexicon)
    _check_texts_on_disk(output, payload)
    adapter = LocalSentenceEncoder(encoder_checkpoint)
    assert adapter.digest == encoder_digest
    sampled, verified_bundles = _check_bundles(
        output, payload, repo, universe_path, encoder_checkpoint, encoder_digest, adapter, lexicon
    )
    artifact = read_cue_set(output / "cue_set.json", set_sha)
    _check_immutability(artifact, sampled)
    _check_atomicity(
        repo, universe_path, universe_sha, encoder_checkpoint, encoder_digest, output, workspace
    )
    cli_output, pids, model_config, encoder_config = _check_cli(
        workspace,
        arguments.install_target,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        output,
        payload,
        set_sha,
    )
    sample_variant = payload["records"][0]["evaluation_base_cues"][0]["variants"][1]
    sample_bundle_path = output / sample_variant["bundle"]["path"]
    sample_bundle_sha = sample_variant["bundle"]["sha256"]
    transaction_pids = _check_write_transactions(
        workspace,
        arguments.install_target,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        output,
        model_config,
        encoder_config,
    )
    pids.extend(transaction_pids)
    _check_repo_failures(
        workspace, repo, universe_path, universe_sha, encoder_checkpoint, encoder_digest
    )
    _check_replace_attack(
        workspace, repo, universe_path, universe_sha, encoder_checkpoint, encoder_digest
    )
    _check_forged_universe(workspace, repo, universe_path, encoder_checkpoint, encoder_digest)
    near_duplicate = _check_pathological_repos(workspace, encoder_checkpoint, encoder_digest)
    _check_cli_inprocess(
        workspace,
        output,
        set_sha,
        sample_bundle_path,
        sample_bundle_sha,
        repo,
        universe_path,
        universe_sha,
        encoder_checkpoint,
        encoder_digest,
        near_duplicate,
        model_config,
        encoder_config,
    )
    default_config_pid = _check_cli_default_configs(
        workspace, arguments.install_target, near_duplicate, encoder_checkpoint, encoder_digest
    )
    pids.append(default_config_pid)
    _check_module_identity(
        repo, universe_path, universe_sha, encoder_checkpoint, encoder_digest, workspace
    )
    _check_crafted_manifests(base_bytes)
    _check_file_level_attacks(workspace, output, set_sha, lexicon)
    sample_bundle_bytes = sample_bundle_path.read_bytes()
    _check_crafted_bundles(sample_bundle_bytes, INPUT_CURRENT)
    _check_bundle_read_surface(workspace, sample_bundle_path, sample_bundle_sha)
    _check_encoder_forgeries(workspace, sample_bundle_bytes, encoder_checkpoint, encoder_digest)
    shifted_repo, shifted_universe_path, shifted_universe_sha = _check_shifted_repo(
        workspace, payload, encoder_checkpoint, encoder_digest
    )
    _check_against_sources(
        workspace,
        output,
        set_sha,
        repo,
        universe_path,
        universe_sha,
        shifted_universe_path,
        shifted_universe_sha,
        shifted_repo,
        adapter,
        lexicon,
    )
    print(
        json.dumps(
            {
                "cli_output": str(cli_output),
                "cue_set_sha256": set_sha,
                "gate_pid": os.getpid(),
                "module_origin": str(module_origin),
                "module_sha256": _sha(module_origin.read_bytes()),
                "status": "pass",
                "subprocess_pids": pids,
                "verified_bundles": verified_bundles,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
