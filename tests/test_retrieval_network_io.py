# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface retrieval checkpoint tests

"""Exercise production checkpoint resolution through real JSON and NPZ files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from REMANENTIA.retrieval_network_io import (
    NetworkCacheKey,
    NetworkPaths,
    NetworkPayload,
    infer_backend_from_path,
    load_checkpoint,
    load_network,
    normalize_backend,
    read_json,
    resolve_network_config,
    resolve_path,
    write_json_atomic,
)


def _paths(base_dir: Path) -> NetworkPaths:
    state_dir = base_dir / "snn_state"
    return NetworkPaths(
        base_dir=base_dir,
        state_dir=state_dir,
        retrieval_state_path=state_dir / "retrieval_state.json",
        default_network_path=state_dir / "identity_net.pkl",
        embedding_network_path=state_dir / "identity_net_embedding_trained.pkl",
    )


def _save_checkpoint(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as checkpoint_file:
        np.savez(checkpoint_file, **arrays)


def test_json_surfaces_reject_absent_malformed_and_non_object_data(tmp_path: Path) -> None:
    """Only a readable JSON object is accepted as state metadata."""
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    array = tmp_path / "array.json"
    valid = tmp_path / "valid.json"
    malformed.write_text("{broken", encoding="utf-8")
    array.write_text("[1, 2]", encoding="utf-8")
    valid.write_text('{"backend": "lsh"}', encoding="utf-8")

    assert read_json(missing) is None
    assert read_json(malformed) is None
    assert read_json(array) is None
    assert read_json(tmp_path) is None
    assert read_json(valid) == {"backend": "lsh"}


def test_atomic_json_write_replaces_a_real_file(tmp_path: Path) -> None:
    """Atomic persistence leaves the complete destination and no temporary file."""
    destination = tmp_path / "nested" / "state.json"
    destination.parent.mkdir()
    destination.write_text('{"old": true}', encoding="utf-8")

    write_json_atomic(destination, {"generation": 2, "project": "Remanentia"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": 2,
        "project": "Remanentia",
    }
    assert not destination.with_suffix(".json.tmp").exists()


def test_backend_and_path_normalization_contract(tmp_path: Path) -> None:
    """Backend names and relative checkpoint paths follow the production contract."""
    absolute = tmp_path / "absolute.npz"

    assert [normalize_backend(name) for name in ("hash", "lsh", "embedding")] == [
        "hash",
        "lsh",
        "embedding",
    ]
    assert normalize_backend("unknown") == "lsh"
    assert normalize_backend(None) == "lsh"
    assert resolve_path(str(absolute), base_dir=tmp_path) == absolute
    assert resolve_path("states/network.npz", base_dir=tmp_path) == (
        tmp_path / "states" / "network.npz"
    ).resolve()
    assert infer_backend_from_path(Path("identity_embedding_v2.npz")) == "embedding"
    assert infer_backend_from_path(Path("identity_net.npz")) == "lsh"


def test_explicit_and_manifest_checkpoint_resolution(tmp_path: Path) -> None:
    """Explicit paths win; otherwise a valid retrieval manifest is authoritative."""
    paths = _paths(tmp_path)
    explicit = tmp_path / "manual_embedding_state.npz"
    manifested = tmp_path / "checkpoints" / "retrieval.npz"
    _save_checkpoint(explicit, weights=np.eye(2))
    _save_checkpoint(manifested, weights=np.ones((2, 2)))
    write_json_atomic(
        paths.retrieval_state_path,
        {
            "checkpoint_path": "checkpoints/retrieval.npz",
            "encoding_backend": "hash",
            "source": "trained_eval_winner",
        },
    )

    explicit_config = resolve_network_config(paths, explicit)
    manifest_config = resolve_network_config(paths)

    assert explicit_config == {
        "checkpoint_path": explicit,
        "encoding_backend": "embedding",
        "source": "explicit",
    }
    assert manifest_config == {
        "checkpoint_path": manifested.resolve(),
        "encoding_backend": "hash",
        "source": "trained_eval_winner",
    }
    with pytest.raises(FileNotFoundError, match="No SNN state"):
        resolve_network_config(paths, tmp_path / "absent.npz")


def test_invalid_manifests_fall_through_to_current_state(tmp_path: Path) -> None:
    """Incomplete and stale manifests cannot mask a valid current-state checkpoint."""
    paths = _paths(tmp_path)
    checkpoint = tmp_path / "current" / "network.npz"
    _save_checkpoint(checkpoint, weights=np.eye(3))
    paths.retrieval_state_path.parent.mkdir(parents=True)
    paths.retrieval_state_path.write_text('{"source": "missing-path"}', encoding="utf-8")
    write_json_atomic(
        paths.state_dir / "current_state.json",
        {
            "retrieval_checkpoint_path": "current/network.npz",
            "retrieval_backend": "embedding",
        },
    )

    assert resolve_network_config(paths) == {
        "checkpoint_path": checkpoint.resolve(),
        "encoding_backend": "embedding",
        "source": "current_state",
    }

    write_json_atomic(
        paths.retrieval_state_path,
        {"checkpoint_path": "stale/missing.npz", "encoding_backend": "hash"},
    )
    assert resolve_network_config(paths)["source"] == "current_state"


def test_legacy_fallback_priority_and_backend_metadata(tmp_path: Path) -> None:
    """Embedding fallback wins, then legacy metadata chooses its declared backend."""
    paths = _paths(tmp_path)
    paths.state_dir.mkdir(parents=True)
    paths.embedding_network_path.touch()
    paths.default_network_path.touch()

    embedding = resolve_network_config(paths)
    assert embedding["checkpoint_path"] == paths.embedding_network_path.resolve()
    assert embedding["encoding_backend"] == "embedding"

    paths.embedding_network_path.unlink()
    write_json_atomic(paths.state_dir / "current_state.json", {"encoding_backend": "hash"})
    legacy = resolve_network_config(paths)
    assert legacy["checkpoint_path"] == paths.default_network_path.resolve()
    assert legacy["encoding_backend"] == "hash"
    assert legacy["source"] == "legacy_checkpoint_fallback"

    write_json_atomic(paths.state_dir / "current_state.json", {"retrieval_backend": "embedding"})
    assert resolve_network_config(paths)["encoding_backend"] == "embedding"

    (paths.state_dir / "current_state.json").unlink()
    assert resolve_network_config(paths)["encoding_backend"] == "lsh"


def test_resolution_reports_when_no_compatible_checkpoint_exists(tmp_path: Path) -> None:
    """A stale current-state path still ends in the operator-facing missing-state error."""
    paths = _paths(tmp_path)
    write_json_atomic(
        paths.state_dir / "current_state.json",
        {"retrieval_checkpoint_path": "missing.npz"},
    )

    with pytest.raises(FileNotFoundError, match="No compatible SNN checkpoint"):
        resolve_network_config(paths)


def test_npz_checkpoint_loading_supports_migrated_and_direct_paths(tmp_path: Path) -> None:
    """Both migrated ``.pkl`` names and direct NPZ paths load numeric arrays."""
    legacy_name = tmp_path / "identity_net.pkl"
    migrated = legacy_name.with_suffix(".npz")
    direct_nonstandard = tmp_path / "checkpoint.bin"
    _save_checkpoint(migrated, weights=np.arange(4).reshape(2, 2))
    _save_checkpoint(direct_nonstandard, thresholds=np.array([0.1, 0.2]))

    migrated_payload = load_checkpoint(legacy_name)
    direct_payload = load_checkpoint(direct_nonstandard)

    np.testing.assert_array_equal(migrated_payload["weights"], np.arange(4).reshape(2, 2))
    np.testing.assert_allclose(direct_payload["thresholds"], [0.1, 0.2])


def test_checkpoint_loader_rejects_pickle_and_missing_files(tmp_path: Path) -> None:
    """Unsafe pickle bytes are rejected and missing paths remain distinguishable."""
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"not a zip checkpoint")

    with pytest.raises(ValueError, match="unsupported legacy pickle checkpoint"):
        load_checkpoint(legacy)
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        load_checkpoint(tmp_path / "missing.pkl")


def test_network_loading_tags_and_reuses_the_real_checkpoint(tmp_path: Path) -> None:
    """A loaded NPZ is tagged once and reused while its filesystem identity is stable."""
    paths = _paths(tmp_path)
    checkpoint = tmp_path / "trained_lsh.npz"
    weights = np.arange(9, dtype=np.float64).reshape(3, 3)
    _save_checkpoint(checkpoint, weights=weights, threshold=np.array([-55.0]))
    stale_key: NetworkCacheKey = ("stale", 0, 0, "hash")
    cache: dict[NetworkCacheKey, NetworkPayload] = {
        stale_key: {"_checkpoint_path": "stale"}
    }

    first = load_network(paths, cache, checkpoint)
    second = load_network(paths, cache, checkpoint)

    assert first is second
    assert stale_key not in cache
    assert len(cache) == 1
    np.testing.assert_array_equal(first["weights"], weights)
    assert first["_checkpoint_path"] == str(checkpoint)
    assert first["_encoding_backend"] == "lsh"
    assert first["_retrieval_source"] == "explicit"
    assert len(str(first["_state_signature"])) == 32


def test_retrieve_wrapper_activates_the_checkpoint_backend(tmp_path: Path) -> None:
    """The legacy retrieval facade loads a real checkpoint and activates encoding.py."""
    import encoding
    import retrieve

    checkpoint = tmp_path / "production_embedding_checkpoint.npz"
    _save_checkpoint(checkpoint, weights=np.eye(2))
    original_backend = encoding.get_backend()
    retrieve._NETWORK_CACHE.clear()
    encoding.set_backend("hash")
    try:
        payload = retrieve._load_network(checkpoint)
        assert payload["_encoding_backend"] == "embedding"
        assert encoding.get_backend() == "embedding"
    finally:
        encoding.set_backend(original_backend)
        retrieve._NETWORK_CACHE.clear()
