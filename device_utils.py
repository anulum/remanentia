# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Safe torch device selection

"""Resolve the right torch device, accounting for GPU compute-capability.

`torch.cuda.is_available()` returns True whenever CUDA is compiled and a
CUDA-capable device is visible — even when PyTorch was not compiled with
kernels for that device's compute capability (e.g. a GTX 1060 sm_61
with a modern torch build that drops everything below sm_75). Such
devices accept `.cuda()` tensors but throw
``torch.AcceleratorError: no kernel image is available`` on the first
forward pass.

:func:`safe_device` closes that trap:

1. If CUDA is not available, return ``"cpu"``.
2. If a specific GPU index is requested (via ``CUDA_VISIBLE_DEVICES`` or
   the ``index`` argument), look up its compute capability and verify it
   appears in :func:`torch.cuda.get_arch_list`. On mismatch, emit a
   one-line warning to stderr and fall back to ``"cpu"``.
3. Otherwise return ``"cuda"`` (or ``f"cuda:{index}"`` if index given).

The check is cached per-process so repeated callers do not re-probe.
"""

from __future__ import annotations

import os
import sys

# Per-process cache: (cuda_available, device_string, warning_emitted).
_CACHE: dict[int | None, str] = {}


def _cap_to_arch(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def _torch_arch_list() -> list[str]:
    import torch

    fn = getattr(torch.cuda, "get_arch_list", None)
    if fn is None:
        return []  # pragma: no cover — torch CUDA build without get_arch_list
    try:
        return list(fn())
    except Exception:  # pragma: no cover — defensive against torch internal changes
        return []


def safe_device(index: int | None = None, *, warn: bool = True) -> str:
    """Return ``"cuda"``, ``f"cuda:{i}"`` or ``"cpu"`` given current hardware.

    Parameters
    ----------
    index:
        Optional GPU index (0, 1, ...). ``None`` means "whatever torch
        picks by default". When ``CUDA_VISIBLE_DEVICES`` is set, the
        index is relative to that list.
    warn:
        Emit a one-line stderr warning the first time a GPU is rejected
        for compute-capability mismatch. Repeat calls are silent.

    Notes
    -----
    The result is cached per process keyed by ``index``. Call
    :func:`clear_cache` from tests if you mutate the environment.
    """
    if index in _CACHE:
        return _CACHE[index]

    forced = os.environ.get("REMANENTIA_FORCE_DEVICE", "").strip()
    if forced:
        # Operator override: trust the named device and bypass the arch-list
        # guard. Use when a GPU runs despite its arch not being listed — e.g. a
        # GTX 1060 (sm_61) on a torch build that ships sm_60 cubins, verified to
        # execute. The operator takes responsibility for the choice.
        _CACHE[index] = forced
        return forced

    import torch

    if not torch.cuda.is_available():
        _CACHE[index] = "cpu"
        return "cpu"

    target = 0 if index is None else index
    try:
        cap = torch.cuda.get_device_capability(target)
    except Exception:  # pragma: no cover — defensive
        _CACHE[index] = "cpu"
        return "cpu"

    arch = _cap_to_arch(cap)
    archs = _torch_arch_list()
    if archs and arch not in archs and f"{arch}+PTX" not in archs:
        if warn:
            name = torch.cuda.get_device_name(target)
            print(
                f"[device] {name} ({arch}) not in torch arch list {archs}; falling back to CPU.",
                file=sys.stderr,
                flush=True,
            )
        _CACHE[index] = "cpu"
        return "cpu"

    device = "cuda" if index is None else f"cuda:{index}"
    _CACHE[index] = device
    return device


def clear_cache() -> None:
    """Drop the per-process device cache. Tests use this when mutating env."""
    _CACHE.clear()


def torch_device_env_disabled() -> bool:
    """True if `CUDA_VISIBLE_DEVICES=""` explicitly disables CUDA."""
    val = os.environ.get("CUDA_VISIBLE_DEVICES")
    return val is not None and val.strip() == ""
