# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval network checkpoint resolution and loading

"""Resolve and load retrieval checkpoints through real filesystem surfaces."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, TypedDict, cast

import numpy as np
import numpy.typing as npt

NetworkArray = npt.NDArray[np.generic]
NetworkPayload = dict[str, NetworkArray | str]
NetworkCacheKey = tuple[str, int, int, str]


class NetworkConfig(TypedDict):
    """Resolved retrieval checkpoint and its encoding contract."""

    checkpoint_path: Path
    encoding_backend: str
    source: str


@dataclass(frozen=True)
class NetworkPaths:
    """Filesystem locations used to resolve a retrieval checkpoint."""

    base_dir: Path
    state_dir: Path
    retrieval_state_path: Path
    default_network_path: Path
    embedding_network_path: Path


def read_json(path: Path) -> dict[str, object] | None:
    """Read a JSON object, returning ``None`` for absent or invalid input."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def normalize_backend(backend: object) -> str:
    """Return a supported encoding backend, defaulting to LSH."""
    return str(backend) if backend in {"hash", "lsh", "embedding"} else "lsh"


def resolve_path(path_text: str, *, base_dir: Path) -> Path:
    """Resolve a checkpoint path relative to the repository base directory."""
    path = Path(path_text)
    return path if path.is_absolute() else (base_dir / path).resolve()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Replace a JSON file atomically after writing it beside the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def infer_backend_from_path(path: Path) -> str:
    """Infer the legacy checkpoint backend from its filename."""
    return "embedding" if "embedding" in path.stem else "lsh"


def _existing_manifest_config(
    manifest: dict[str, object] | None,
    *,
    paths: NetworkPaths,
) -> NetworkConfig | None:
    if not manifest:
        return None
    raw_path = manifest.get("checkpoint_path")
    if not raw_path:
        return None
    checkpoint_path = resolve_path(str(raw_path), base_dir=paths.base_dir)
    if not checkpoint_path.exists():
        return None
    return {
        "checkpoint_path": checkpoint_path,
        "encoding_backend": normalize_backend(manifest.get("encoding_backend")),
        "source": str(manifest.get("source", "retrieval_state")),
    }


def resolve_network_config(
    paths: NetworkPaths,
    state_path: Path | None = None,
) -> NetworkConfig:
    """Resolve the explicit, manifested, current, or legacy checkpoint config."""
    if state_path is not None:
        checkpoint_path = resolve_path(str(state_path), base_dir=paths.base_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No SNN state at {checkpoint_path}.")
        return {
            "checkpoint_path": checkpoint_path,
            "encoding_backend": infer_backend_from_path(checkpoint_path),
            "source": "explicit",
        }

    manifest_config = _existing_manifest_config(
        read_json(paths.retrieval_state_path), paths=paths
    )
    if manifest_config is not None:
        return manifest_config

    current_state = read_json(paths.state_dir / "current_state.json")
    if current_state:
        raw_path = current_state.get("retrieval_checkpoint_path")
        if raw_path:
            checkpoint_path = resolve_path(str(raw_path), base_dir=paths.base_dir)
            if checkpoint_path.exists():
                return {
                    "checkpoint_path": checkpoint_path,
                    "encoding_backend": normalize_backend(
                        current_state.get("retrieval_backend")
                    ),
                    "source": "current_state",
                }

    if paths.embedding_network_path.exists():
        return {
            "checkpoint_path": paths.embedding_network_path.resolve(),
            "encoding_backend": "embedding",
            "source": "embedding_checkpoint_fallback",
        }

    if paths.default_network_path.exists():
        current_state = current_state or {}
        return {
            "checkpoint_path": paths.default_network_path.resolve(),
            "encoding_backend": normalize_backend(
                current_state.get("retrieval_backend")
                or current_state.get("encoding_backend")
                or infer_backend_from_path(paths.default_network_path)
            ),
            "source": "legacy_checkpoint_fallback",
        }

    raise FileNotFoundError(
        "No compatible SNN checkpoint found. Expected retrieval_state.json, "
        "identity_net_embedding_trained.pkl, or identity_net.pkl."
    )


def load_checkpoint(path: Path) -> dict[str, NetworkArray]:
    """Load an NPZ checkpoint and reject unsafe legacy pickle payloads."""
    npz_path = path.with_suffix(".npz") if path.suffix != ".npz" else path
    for candidate in (npz_path, path):
        if candidate.exists() and zipfile.is_zipfile(candidate):
            with np.load(candidate, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}
    if path.exists():
        raise ValueError(
            f"{path}: unsupported legacy pickle checkpoint. "
            f"Run `python tools/migrate_pickle_to_npz.py --path {path.parent}` "
            "to convert pre-0.4 checkpoints to npz."
        )
    raise FileNotFoundError(f"No checkpoint at {npz_path} or {path}")


def load_network(
    paths: NetworkPaths,
    cache: MutableMapping[NetworkCacheKey, NetworkPayload],
    state_path: Path | None = None,
) -> NetworkPayload:
    """Load and metadata-tag a checkpoint, reusing an unchanged cached payload."""
    config = resolve_network_config(paths, state_path)
    checkpoint_path = config["checkpoint_path"]
    stat = checkpoint_path.stat()
    cache_key = (
        str(checkpoint_path),
        stat.st_mtime_ns,
        stat.st_size,
        config["encoding_backend"],
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    payload: NetworkPayload = dict(load_checkpoint(checkpoint_path))
    signature = hashlib.md5(
        (
            f"{checkpoint_path}:{stat.st_mtime_ns}:{stat.st_size}:"
            f"{config['encoding_backend']}"
        ).encode()
    ).hexdigest()
    payload.update(
        {
            "_checkpoint_path": str(checkpoint_path),
            "_encoding_backend": config["encoding_backend"],
            "_state_signature": signature,
            "_retrieval_source": config["source"],
        }
    )
    cache.clear()
    cache[cache_key] = payload
    return payload
