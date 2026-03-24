# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Entity Extraction (GLiNER + typed relations)

"""Entity extraction via GLiNER2 (205M params, GPU) + typed relation detection.

GLiNER: zero-shot NER — describe entity types in natural language,
model finds them. No training needed.

Typed relations: pattern matching on connecting text between entities.
"because" → caused_by, "fixed" → fixed_by, "replaced" → replaced,
"contradicts" → contradicts, "version" → version_of.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

_GLINER_MODEL = None

ENTITY_LABELS = [
    "person", "project", "software tool", "hardware",
    "algorithm", "file path", "metric value", "version number",
    "neural network model", "benchmark", "mathematical concept",
    "programming language", "scientific concept",
]

RELATION_PATTERNS = [
    (r"\bbecause\b|\bcaused by\b|\bdue to\b|\broot cause\b", "caused_by"),
    (r"\bfixed\b|\brepaired\b|\bcorrected\b|\bpatched\b", "fixed_by"),
    (r"\breplaced\b|\bsuperseded\b|\binstead of\b", "replaced"),
    (r"\bcontradicts?\b|\binconsistent with\b|\bconflicts? with\b", "contradicts"),
    (r"\bv\d+\.\d+|\bversion\b", "version_of"),
    (r"\bdepends on\b|\brequires\b|\bneeds\b", "depends_on"),
    (r"\bimproved\b|\bfrom .+ to\b|\bincreased\b|\bdecreased\b", "improved"),
    (r"\bproduced\b|\bcreated\b|\bgenerated\b|\bwrote\b", "produced"),
    (r"\bused in\b|\bpart of\b|\bcomponent of\b", "used_in"),
    (r"\btested\b|\bbenchmarked\b|\bevaluated\b|\bmeasured\b", "tested_with"),
]


@dataclass
class Entity:
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    evidence: str


def _load_gliner():  # pragma: no cover
    global _GLINER_MODEL
    if _GLINER_MODEL is not None:
        return _GLINER_MODEL
    try:
        from gliner import GLiNER
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        if device == "cuda":
            _GLINER_MODEL = _GLINER_MODEL.to(device)
        return _GLINER_MODEL
    except ImportError:
        return None


def extract_entities(text: str, labels: list[str] | None = None) -> list[Entity]:
    """Extract entities via GLiNER. Falls back to regex if GLiNER unavailable."""
    labels = labels or ENTITY_LABELS
    model = _load_gliner()

    if model is not None:
        # GLiNER extraction — process in chunks (max 512 tokens)
        entities = []
        chunks = [text[i:i + 1500] for i in range(0, len(text), 1500)]
        for chunk in chunks[:20]:  # cap at 20 chunks per document
            try:
                preds = model.predict_entities(chunk, labels, threshold=0.4)
                for p in preds:
                    entities.append(Entity(
                        text=p["text"],
                        label=p["label"],
                        score=p["score"],
                        start=p.get("start", 0),
                        end=p.get("end", 0),
                    ))
            except Exception:  # pragma: no cover
                continue
        # Deduplicate by text
        seen = set()
        unique = []
        for e in entities:
            key = e.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    # Fallback: regex extraction (same as consolidation_engine.py expanded list)
    return _regex_entities(text)


def _regex_entities(text: str) -> list[Entity]:
    """Fallback regex entity extraction."""
    entities = []
    text_lower = text.lower()

    patterns = {
        "project": ["director-ai", "sc-neurocore", "scpn-control", "scpn-fusion",
                     "scpn-phase-orchestrator", "scpn-quantum-control", "remanentia"],
        "algorithm": ["stdp", "lif", "kuramoto", "hopfield", "bcpnn", "csdp",
                       "tf-idf", "bm25", "stuart-landau"],
        "hardware": ["gpu", "cuda", "loihi", "gtx 1060"],
        "software tool": ["pytorch", "numpy", "scipy", "fastapi", "docker"],
    }
    for label, terms in patterns.items():
        for term in terms:
            if term in text_lower:
                entities.append(Entity(text=term, label=label, score=0.5))

    # Version numbers
    for m in re.finditer(r"v\d+\.\d+(?:\.\d+)?", text):
        entities.append(Entity(text=m.group(), label="version number", score=0.6))

    # File paths
    for m in re.finditer(r"[\w/\\]+\.(?:py|rs|md|json|yaml|toml)\b", text):
        name = m.group().split("/")[-1].split("\\")[-1]
        if len(name) > 3:
            entities.append(Entity(text=name, label="file path", score=0.5))

    return entities


def extract_relations(text: str, entities: list[Entity]) -> list[Relation]:
    """Extract typed relations between entities from connecting text."""
    relations = []
    entity_texts = [(e.text, e) for e in entities]

    for i, (t1, e1) in enumerate(entity_texts):
        for t2, e2 in entity_texts[i + 1:]:
            # Find text between the two entities
            pos1 = text.lower().find(t1.lower())
            pos2 = text.lower().find(t2.lower())
            if pos1 < 0 or pos2 < 0:
                continue
            start, end = min(pos1, pos2), max(pos1 + len(t1), pos2 + len(t2))
            between = text[start:end]

            for pattern, rel_type in RELATION_PATTERNS:
                if re.search(pattern, between, re.IGNORECASE):
                    relations.append(Relation(
                        source=t1,
                        target=t2,
                        relation_type=rel_type,
                        evidence=between[:200],
                    ))
                    break
            else:
                # Default: co_occurs if within 500 chars
                if abs(pos1 - pos2) < 500:
                    relations.append(Relation(
                        source=t1, target=t2,
                        relation_type="co_occurs",
                        evidence="",
                    ))

    return relations


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Miroslav Sotek fixed the STDP bug in snn_backend.py because the LTD mask was wrong."
    print(f"Extracting from: {text[:100]}...")

    t0 = time.monotonic()
    entities = extract_entities(text)
    ent_ms = (time.monotonic() - t0) * 1000
    print(f"\nEntities ({ent_ms:.0f}ms):")
    for e in entities:
        print(f"  {e.label:20s} | {e.text:30s} | {e.score:.2f}")

    relations = extract_relations(text, entities)
    print(f"\nRelations:")
    for r in relations:
        print(f"  {r.source:20s} --[{r.relation_type}]--> {r.target}")
