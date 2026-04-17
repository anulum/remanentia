# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Cross-library seed pinning

"""Pin every randomness source Remanentia pulls in so benchmarks reproduce.

Pre-2026-04-17 no Remanentia entry point set a seed. Every bench run
produced a slightly different number, and the ±10 q "noise envelope"
quoted in the LongMemEval docs was a guess rather than a measurement.
This module is the one call every reproducible entry point now makes.

One level of indirection lets us pin the following chain atomically:

1. Python ``random``
2. ``numpy.random`` (legacy + default_rng)
3. ``torch.manual_seed`` (CPU + all CUDA devices)
4. Environment ``PYTHONHASHSEED`` (for inter-process reproducibility
   — applies only to child processes launched after this call)

Libraries we do NOT pin (out of scope today):

- ``sentence-transformers`` / ``transformers`` internal dropout:
  honour ``torch.manual_seed`` already.
- LLM provider RNG (OpenAI, Anthropic): controlled by ``temperature``
  and ``seed`` request parameters on each call, not library state.
- Rust crates: deterministic by construction — no seed needed.

:func:`seed_everything` returns the seed it used so callers can print
it in the banner and write it to result files.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int | None = None, *, torch_cuda_deterministic: bool = False) -> int:
    """Pin random / numpy / torch / hash seeds.

    Parameters
    ----------
    seed:
        Integer seed. ``None`` means "pick a deterministic default" —
        42 is used so two fresh processes that both pass ``None``
        agree. For truly random behaviour (e.g. multi-run variance
        studies) pass an explicit value each time.
    torch_cuda_deterministic:
        Also set ``torch.backends.cudnn.deterministic = True`` and
        ``torch.use_deterministic_algorithms(True)`` for numerically
        reproducible GPU kernels. This can noticeably slow training;
        default off. Accuracy benchmarks rarely need it.

    Returns
    -------
    int
        The seed value actually used.
    """
    if seed is None:
        seed = 42

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        # The newer default_rng API keeps its own state; instantiate it
        # with the same seed so callers that prefer it stay aligned.
        np.random.default_rng(seed)
    except ImportError:  # pragma: no cover — numpy is a hard dep
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover — CI has no CUDA
            torch.cuda.manual_seed_all(seed)
        if torch_cuda_deterministic:  # pragma: no cover — opt-in slow path
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def seed_from_env(var: str = "REMANENTIA_SEED", default: int = 42) -> int:
    """Read a seed from an env var, falling back to *default* on failure.

    Shell integration shortcut: set ``REMANENTIA_SEED=123`` before
    running any bench script to lock the randomness.
    """
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
