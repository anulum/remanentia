# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pinned local sentence-embedding adapter tests

"""Real pinned-local-model adapter tests over tracked Markdown corpora.

The model-dependent surfaces here require the git-ignored ``.snn_models/``
checkpoint and are therefore local integration evidence; the deterministic
encoding maths this adapter composes is unit-tested in
``tests/test_snn_memory_encoder.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.sentence_encoder import LocalSentenceEncoder, encode_trace

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"


def test_missing_local_checkpoint_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        LocalSentenceEncoder(tmp_path / "missing")


@pytest.mark.skipif(  # type: ignore[untyped-decorator] # Pytest decorator.
    not MODEL.is_dir(), reason="pinned local encoder not provisioned"
)
def test_real_markdown_encodes_ordered_deterministic_packets() -> None:
    encoder = LocalSentenceEncoder(MODEL)
    text = (ROOT / "docs/research/snn_consolidation.md").read_text(encoding="utf-8")
    model = ModelConfig(n_neurons=32, connectivity=0.2)
    config = EncoderConfig(feature_dim=32, packet_ms=6, silent_ms=2)
    first = encode_trace(text, encoder, model, config, input_current=18.0)
    second = encode_trace(text, encoder, model, config, input_current=18.0)
    assert first.event_count > 2
    assert first.currents.shape[1] == model.n_neurons
    np.testing.assert_array_equal(first.currents, second.currents)
    assert first.checkpoint_digest == second.checkpoint_digest
