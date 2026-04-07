# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for signal detection (Phase 2)

from signal_detector import detect_signals, classify_message, SignalType
from knowledge_store import KnowledgeStore


def test_detect_signals_correction():
    text = "That is wrong, my birthday is actually June 5th."
    signals = detect_signals(text)
    assert len(signals) > 0
    assert signals[0].signal_type == SignalType.CORRECTION
    assert signals[0].confidence >= 0.8


def test_detect_signals_reinforcement():
    text = "Exactly right, that is my favorite color."
    signals = detect_signals(text)
    assert len(signals) > 0
    assert signals[0].signal_type == SignalType.REINFORCEMENT
    assert signals[0].confidence >= 0.9


def test_classify_message():
    assert classify_message("No, that's incorrect.") == SignalType.CORRECTION
    assert classify_message("Perfect!") == SignalType.REINFORCEMENT
    assert classify_message("The weather is nice today.") == SignalType.NEUTRAL


def test_signal_integration_in_store():
    ks = KnowledgeStore()

    # Add a fact
    content = "My favorite color is blue."
    note = ks.add_note(content, source="session1")
    initial_conf = note.confidence

    # Reinforce it - use EXACT same content to guarantee search hit
    ks.add_note("Exactly right, my favorite color is blue.", source="session2")
    assert ks.notes[note.id].confirmation_count == 1
    assert ks.notes[note.id].confidence > initial_conf

    # Correct it - use EXACT same content to guarantee search hit
    ks.add_note("That is wrong, my favorite color is blue.", source="session3")
    assert ks.notes[note.id].contradiction_count >= 1
