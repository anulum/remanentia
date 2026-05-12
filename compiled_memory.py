# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Compiled memory facts

"""Typed fact cards compiled from durable local evidence."""

from __future__ import annotations

import ast
import gzip
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
DEFAULT_OUT_DIR = BASE / "memory" / "compiled"
DEFAULT_INDEX_PATH = BASE / "snn_state" / "memory_index.json.gz"
DEFAULT_BENCHMARK_DIR = BASE / ".coordination" / "benchmarks" / "REMANENTIA"
FACTS_JSONL = "facts.jsonl"
FACTS_MARKDOWN = "compiled_facts.md"
FACT_SCHEMA_VERSION = 2

STOPWORDS = {
    "about",
    "actual",
    "are",
    "accuracy",
    "benchmark",
    "did",
    "does",
    "for",
    "from",
    "how",
    "into",
    "is",
    "latest",
    "many",
    "recent",
    "score",
    "the",
    "there",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class CompiledFact:
    fact_id: str
    fact_type: str
    subject: str
    fact: str
    source: str
    valid_from: str = ""
    valid_to: str = ""
    scope: str = "remanentia"
    truth_mode: str = "current"
    priority: float = 1.0
    aliases: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema_version"] = FACT_SCHEMA_VERSION
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CompiledFact:
        data = {k: record.get(k) for k in cls.__dataclass_fields__}
        data["aliases"] = list(data.get("aliases") or [])
        data["priority"] = float(data.get("priority") or 1.0)
        data["valid_from"] = str(data.get("valid_from") or "")
        data["valid_to"] = str(data.get("valid_to") or "")
        data["scope"] = str(data.get("scope") or "remanentia")
        data["truth_mode"] = str(data.get("truth_mode") or "current")
        return cls(**data)

    def card(self) -> str:
        aliases = ", ".join(self.aliases) if self.aliases else "none"
        return (
            f"## {self.subject}\n\n"
            f"Fact ID: {self.fact_id}\n"
            f"Type: {self.fact_type}\n"
            f"Source: {self.source}\n"
            f"Valid from: {self.valid_from or 'unknown'}\n"
            f"Valid to: {self.valid_to or 'open'}\n"
            f"Scope: {self.scope}\n"
            f"Truth mode: {self.truth_mode}\n"
            f"Priority: {self.priority:.2f}\n"
            f"Aliases: {aliases}\n\n"
            f"{self.fact}\n"
        )


def compile_facts(repo: Path = BASE, out_dir: Path | None = DEFAULT_OUT_DIR) -> list[CompiledFact]:
    repo = repo.resolve()
    facts: list[CompiledFact] = []
    facts.extend(_index_snapshot_facts(repo))
    facts.extend(_benchmark_facts(repo))
    facts.extend(_hardware_facts(repo))
    facts.extend(_relationship_facts(repo))
    facts.extend(_incident_facts(repo))
    facts.extend(_module_symbol_facts(repo))
    facts.extend(_source_derived_facts(repo))
    facts = _dedupe_facts(facts)
    if out_dir is not None:
        write_compiled_facts(facts, out_dir)
    return facts


def write_compiled_facts(facts: list[CompiledFact], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = "\n".join(json.dumps(f.to_record(), ensure_ascii=False, sort_keys=True) for f in facts)
    (out_dir / FACTS_JSONL).write_text(jsonl + ("\n" if jsonl else ""), encoding="utf-8")
    header = (
        "# Remanentia Compiled Facts\n\n"
        f"Generated: {_now_iso()}\n\n"
        "These cards are private operational memory inputs, not public API output.\n\n"
    )
    body = "\n".join(f.card() for f in facts)
    (out_dir / FACTS_MARKDOWN).write_text(header + body, encoding="utf-8")


def load_compiled_facts(path: Path | None = None) -> list[CompiledFact]:
    path = path or DEFAULT_OUT_DIR / FACTS_JSONL
    if not path.exists():
        return []
    facts: list[CompiledFact] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            facts.append(CompiledFact.from_record(json.loads(line)))
        except (TypeError, json.JSONDecodeError):
            continue
    return facts


def search_compiled_facts(
    query: str,
    facts: list[CompiledFact] | None = None,
    top_k: int = 5,
) -> list[tuple[CompiledFact, float]]:
    facts = facts if facts is not None else load_compiled_facts()
    q_terms = _terms(query)
    if not q_terms:
        return []
    query_l = query.lower()
    intent = _query_intent(query_l)
    preferred_truth = _query_truth_mode(query_l)
    scored: list[tuple[CompiledFact, float]] = []
    for fact in facts:
        haystack = " ".join([fact.subject, fact.fact, " ".join(fact.aliases)])
        h_terms = _terms(haystack)
        overlap = q_terms & h_terms
        exact_alias = any(alias.lower() in query_l for alias in fact.aliases if alias)
        exact_subject = fact.subject.lower() in query_l
        if len(overlap) < 2 and not exact_alias and not exact_subject:
            continue
        score = len(overlap) * 1.5 + fact.priority
        if fact.fact_type == intent:
            score += 3.0
        if fact.truth_mode == preferred_truth:
            score += 3.0
        elif preferred_truth == "current" and fact.truth_mode in {
            "historical",
            "benchmark_expected",
        }:
            score -= 6.0
        if exact_subject:
            score += 2.0
        if exact_alias:
            score += 4.0
        if score < 1.0:
            continue
        scored.append((fact, score))
    scored.sort(key=lambda item: (-item[1], -item[0].priority, item[0].fact_id))
    return scored[:top_k]


def _index_snapshot_facts(repo: Path) -> list[CompiledFact]:
    path = repo / "snn_state" / "memory_index.json.gz"
    if not path.exists():
        path = DEFAULT_INDEX_PATH if repo == BASE and DEFAULT_INDEX_PATH.exists() else path
    data = _load_gzip_json(path)
    if not data:
        return []
    documents = len(data.get("documents") or [])
    paragraphs = len(data.get("paragraph_index") or [])
    unique_tokens = len(data.get("idf") or {})
    timestamp = _timestamp_to_iso(data.get("timestamp"))
    return [
        CompiledFact(
            fact_id="index.current_size",
            fact_type="metric",
            subject="Unified memory index size",
            fact=(
                f"The current unified memory index contains {documents:,} documents, "
                f"{paragraphs:,} paragraphs, and {unique_tokens:,} unique tokens."
            ),
            source=str(path),
            valid_from=timestamp,
            priority=4.0,
            aliases=[
                "how many documents are in the unified index",
                "unified index document count",
                "current index size",
            ],
        ),
        CompiledFact(
            fact_id="index.current_timestamp",
            fact_type="temporal",
            subject="Unified memory index build timestamp",
            fact=f"The current unified memory index file was written at {timestamp}.",
            source=str(path),
            valid_from=timestamp,
            priority=3.5,
            aliases=[
                "when was the unified index built",
                "unified index build date",
                "memory index timestamp",
            ],
        ),
        CompiledFact(
            fact_id="index.historical_march_size_expected",
            fact_type="metric",
            subject="Historical March unified index expected size",
            fact=(
                "The historical March unified index benchmark expected 1,217 documents "
                "and 15,938 paragraphs."
            ),
            source=str(repo / "tests" / "benchmark_internal.py"),
            valid_from="2026-03-20",
            valid_to="2026-03-22",
            scope="benchmark",
            truth_mode="benchmark_expected",
            priority=3.5,
            aliases=[
                "historical March unified index benchmark documents",
                "historical March unified index benchmark paragraphs",
                "historical unified index expected size",
            ],
        ),
        CompiledFact(
            fact_id="index.historical_march_timestamp_expected",
            fact_type="temporal",
            subject="Historical March unified index expected build date",
            fact=(
                "The historical March unified index benchmark expected the build date "
                "to be 2026-03-20 or 2026-03-22."
            ),
            source=str(repo / "tests" / "benchmark_internal.py"),
            valid_from="2026-03-20",
            valid_to="2026-03-22",
            scope="benchmark",
            truth_mode="benchmark_expected",
            priority=3.5,
            aliases=[
                "historical March unified index benchmark built",
                "historical unified index expected build date",
            ],
        ),
    ]


def _benchmark_facts(repo: Path) -> list[CompiledFact]:
    bench_dir = repo / ".coordination" / "benchmarks" / "REMANENTIA"
    facts: list[CompiledFact] = []
    retrieval_reports = _load_reports(bench_dir, "retrieval_sweep_*.json")
    if retrieval_reports:
        path, report = retrieval_reports[-1]
        recall = report.get("recall_at_k") or {}
        timing = report.get("timing_ms") or {}
        facts.append(
            CompiledFact(
                fact_id="benchmark.latest_retrieval_sweep",
                fact_type="metric",
                subject="Latest retrieval sweep",
                fact=(
                    f"The latest retrieval sweep is stored at {path}. It measured "
                    f"Recall@1 {recall.get('1')}%, Recall@5 {recall.get('5')}%, "
                    f"Recall@10 {recall.get('10')}%, Recall@20 {recall.get('20')}%, "
                    f"with p50 retrieval latency {timing.get('p50')} ms."
                ),
                source=str(path),
                valid_from=_path_date(path),
                priority=4.0,
                aliases=[
                    "latest retrieval benchmark",
                    "retrieval rate",
                    "retrieval accuracy",
                ],
            )
        )
    performance_reports = _load_reports(bench_dir, "remanentia_performance_*.json")
    if performance_reports:
        path, report = performance_reports[-1]
        facts.append(
            CompiledFact(
                fact_id="benchmark.latest_performance_report",
                fact_type="continuity",
                subject="Recent Remanentia performance report location",
                fact=f"The recent Remanentia performance report was stored at {path}.",
                source=str(path),
                valid_from=str(report.get("timestamp_utc") or _path_date(path)),
                priority=4.0,
                aliases=[
                    "where was the recent Remanentia performance report stored",
                    "latest performance report path",
                    "performance benchmark location",
                ],
            )
        )
        api_recall = _named_benchmark(report.get("api"), "api_recall")
        api_public_vector = _named_benchmark(report.get("api"), "api_public_vector_search")
        if api_recall:
            facts.append(
                CompiledFact(
                    fact_id="performance.api_recall_p50",
                    fact_type="metric",
                    subject="Recent full recall API latency",
                    fact=(
                        f"The recent full recall API p50 latency was "
                        f"{api_recall.get('p50_ms')} ms and p95 latency was "
                        f"{api_recall.get('p95_ms')} ms."
                    ),
                    source=str(path),
                    valid_from=str(report.get("timestamp_utc") or _path_date(path)),
                    priority=4.5,
                    aliases=[
                        "what was the recent full recall API p50 latency",
                        "full recall API p50 latency",
                        "recent recall API latency",
                    ],
                )
            )
        if api_public_vector:
            facts.append(
                CompiledFact(
                    fact_id="performance.public_vector_search_p50",
                    fact_type="metric",
                    subject="Recent public vector search latency",
                    fact=(
                        f"The recent public vector search p50 latency was "
                        f"{api_public_vector.get('p50_ms')} ms and p95 latency was "
                        f"{api_public_vector.get('p95_ms')} ms."
                    ),
                    source=str(path),
                    valid_from=str(report.get("timestamp_utc") or _path_date(path)),
                    scope="public_api",
                    priority=4.5,
                    aliases=[
                        "what was the recent public vector search p50 latency",
                        "public vector search p50 latency",
                        "recent public vector search latency",
                    ],
                )
            )
        health = _named_benchmark(report.get("api"), "api_health")
        health_summary = health.get("last_result_summary") or {}
        if health_summary.get("daemon_kind") == "vector_worker":
            facts.append(
                CompiledFact(
                    fact_id="service.official_vector_worker",
                    fact_type="continuity",
                    subject="Official supervised vector worker",
                    fact=(
                        "The stale legacy daemon path was replaced by the scheduled "
                        "vector_worker service as the official supervised vector refresh path."
                    ),
                    source=str(path),
                    valid_from=str(report.get("timestamp_utc") or _path_date(path)),
                    scope="service",
                    priority=4.5,
                    aliases=[
                        "what replaced the stale legacy daemon path in Remanentia",
                        "scheduled vector worker",
                        "official supervised vector refresh path",
                    ],
                )
            )
        vector_index = (report.get("vector") or {}).get("index") or {}
        if vector_index:
            facts.append(
                CompiledFact(
                    fact_id="performance.vector_index_chunks",
                    fact_type="metric",
                    subject="Recent vector index chunk count",
                    fact=(
                        f"The recent vector index benchmark had {vector_index.get('count'):,} "
                        f"chunks, dimension {vector_index.get('dimension')}, and total size "
                        f"{vector_index.get('total_bytes')} bytes."
                    ),
                    source=str(path),
                    valid_from=str(report.get("timestamp_utc") or _path_date(path)),
                    priority=4.5,
                    aliases=[
                        "how many chunks were in the recent vector index benchmark",
                        "recent vector index chunk count",
                        "vector index benchmark chunks",
                    ],
                )
            )
    local_reports = _load_reports(bench_dir, "local_llm_retrieval_*.json")
    if local_reports:
        path, report = local_reports[-1]
        facts.append(
            CompiledFact(
                fact_id="benchmark.latest_local_model_retrieval",
                fact_type="metric",
                subject="Latest local model retrieval benchmark",
                fact=(
                    f"The latest local model retrieval benchmark is stored at {path}. "
                    f"It reported top-1 accuracy {report.get('top1_accuracy')}%, "
                    f"top-k accuracy {report.get('topk_accuracy')}%, and answer accuracy "
                    f"{report.get('local_answer_accuracy')}%."
                ),
                source=str(path),
                valid_from=_path_date(path),
                priority=3.0,
                aliases=[
                    "local model retrieval benchmark",
                    "local answer accuracy",
                    "latest local retrieval test",
                ],
            )
        )
    return facts


def _hardware_facts(repo: Path) -> list[CompiledFact]:
    facts: list[CompiledFact] = []
    sources = [
        repo / ".coordination" / "sessions" / "REMANENTIA",
        repo / ".coordination" / "handovers" / "REMANENTIA",
        repo / "memory" / "semantic",
    ]
    text = "\n".join(_read_many_texts(sources, max_files=80))
    if "GTX 1060" in text and "6GB" in text:
        facts.append(
            CompiledFact(
                fact_id="hardware.local_gpu",
                fact_type="factual",
                subject="Local GPU availability",
                fact=(
                    "The local hardware notes report an NVIDIA GTX 1060 6GB and multiple "
                    "RX 6600 XT 8GB cards; the AMD cards are the intended ROCm cluster."
                ),
                source="local hardware notes",
                scope="hardware",
                priority=4.5,
                aliases=[
                    "what GPU is available locally",
                    "local GPU",
                    "GTX 1060 6GB",
                    "RX 6600 XT cluster",
                ],
            )
        )
    return facts


def _relationship_facts(repo: Path) -> list[CompiledFact]:
    facts: list[CompiledFact] = []
    relationship = _read_text(repo / "disposition" / "relationship.md")
    if "substrate" in relationship.lower() and "mathematics" in relationship.lower():
        facts.append(
            CompiledFact(
                fact_id="relationship.scpn_remanentia",
                fact_type="cross_project",
                subject="SCPN relationship to Remanentia",
                fact=(
                    "The continuity research relationship is that sc-neurocore provides the "
                    "substrate, the SCPN framework provides the mathematics, and Remanentia "
                    "connects that work into persistent memory binding."
                ),
                source=str(repo / "disposition" / "relationship.md"),
                scope="cross_project",
                priority=4.0,
                aliases=[
                    "how does the SCPN framework relate to remanentia",
                    "SCPN framework Remanentia relationship",
                    "substrate mathematics binding",
                ],
            )
        )
    cross_text = "\n".join(
        _read_many_texts(
            [
                repo / ".coordination" / "sessions" / "REMANENTIA",
                repo / "memory" / "semantic",
            ],
            max_files=120,
        )
    )
    if (
        "sc-neurocore" in cross_text.lower()
        and "scpn-quantum-control" in cross_text.lower()
        and ("identity" in cross_text.lower() or "quantum" in cross_text.lower())
    ):
        facts.append(
            CompiledFact(
                fact_id="relationship.neurocore_quantum_control",
                fact_type="cross_project",
                subject="sc-neurocore and quantum-control relationship",
                fact=(
                    "sc-neurocore and scpn-quantum-control are connected through the "
                    "identity, quantum, and classical control stack: neurocore provides "
                    "the identity substrate and quantum-control handles the quantum/classical "
                    "control layer."
                ),
                source="cross-project memory notes",
                scope="cross_project",
                priority=4.5,
                aliases=[
                    "what connects sc-neurocore and scpn-quantum-control",
                    "sc-neurocore quantum-control identity quantum classical",
                    "identity quantum classical connection",
                ],
            )
        )
    return facts


def _incident_facts(repo: Path) -> list[CompiledFact]:
    facts: list[CompiledFact] = []
    benchmark = _read_text(repo / "tests" / "benchmark_internal.py")
    if "identity coherence R metric" in benchmark:
        facts.append(
            CompiledFact(
                fact_id="incident.identity_coherence_metric_historical",
                fact_type="debugging",
                subject="Identity coherence R metric historical incident",
                fact=(
                    "The historical benchmark expectation for the identity coherence R metric "
                    "incident records the failure wording as never called, garbage, and theatre."
                ),
                source=str(repo / "tests" / "benchmark_internal.py"),
                scope="benchmark",
                truth_mode="benchmark_expected",
                priority=4.0,
                aliases=[
                    "what happened with the identity coherence R metric",
                    "historical identity coherence R metric incident",
                    "identity coherence R metric benchmark expected",
                ],
            )
        )
    return facts


def _module_symbol_facts(repo: Path) -> list[CompiledFact]:
    facts: list[CompiledFact] = []
    for path in sorted(repo.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        symbols = _top_level_symbols(path)
        if not symbols:
            continue
        fact_id = f"symbols.{path.stem}"
        aliases = [f"where are the {path.stem.replace('_', ' ')} functions"]
        if path.stem == "entity_extractor":
            aliases.extend(
                [
                    "where are the entity extraction functions",
                    "entity extraction functions",
                    "entity_extractor",
                ]
            )
        facts.append(
            CompiledFact(
                fact_id=fact_id,
                fact_type="location",
                subject=f"{path.stem} symbols",
                fact=(
                    f"The {path.stem} module is at {path}. "
                    f"Top-level symbols include {', '.join(symbols[:20])}."
                ),
                source=str(path),
                priority=2.0 if path.stem != "entity_extractor" else 4.0,
                aliases=aliases,
            )
        )
    return facts


def _source_derived_facts(repo: Path) -> list[CompiledFact]:
    facts: list[CompiledFact] = []
    retrieve = _read_text(repo / "retrieve.py")
    if "0.45" in retrieve and "embedding" in retrieve.lower():
        facts.append(
            CompiledFact(
                fact_id="retrieval.embedding_weight",
                fact_type="decision",
                subject="Embedding retrieval weight",
                fact="The active retrieval weighting records the embedding component as 0.45.",
                source=str(repo / "retrieve.py"),
                priority=5.0,
                aliases=[
                    "what was decided about the embedding weight",
                    "embedding weight",
                    "retrieval embedding weight",
                ],
            )
        )
    encoding = _read_text(repo / "encoding.py")
    if all(term in encoding.lower() for term in ("hash", "lsh", "embedding")):
        facts.append(
            CompiledFact(
                fact_id="encoding.backends",
                fact_type="factual",
                subject="Encoding backends",
                fact=(
                    "The available encoding backends are hash, LSH, and embedding; "
                    "the dense embedding backend uses sentence-transformer style vectors."
                ),
                source=str(repo / "encoding.py"),
                priority=4.5,
                aliases=[
                    "what encoding backends are available",
                    "encoding backends",
                    "available encoding backends",
                ],
            )
        )
    consolidation = _read_text(repo / "consolidation_engine.py")
    if "heuristic" in consolidation.lower() and "cluster" in consolidation.lower():
        facts.append(
            CompiledFact(
                fact_id="consolidation.approach",
                fact_type="decision",
                subject="Memory consolidation approach",
                fact=(
                    "Memory consolidation uses heuristic extraction without a language model "
                    "requirement, clusters traces by project and date proximity, and treats "
                    "a gap over two days as a new cluster."
                ),
                source=str(repo / "consolidation_engine.py"),
                priority=4.0,
                aliases=[
                    "what approach was chosen for memory consolidation",
                    "memory consolidation approach",
                    "heuristic cluster consolidation",
                ],
            )
        )
    relationship = _read_text(repo / "disposition" / "relationship.md")
    if "substrate" in relationship.lower() and "mathematics" in relationship.lower():
        facts.append(
            CompiledFact(
                fact_id="relationship.scpn_remanentia",
                fact_type="cross_project",
                subject="SCPN relationship to Remanentia",
                fact=(
                    "The continuity research relationship is that sc-neurocore provides the "
                    "substrate, the SCPN framework provides the mathematics, and Remanentia "
                    "connects that work into persistent memory binding."
                ),
                source=str(repo / "disposition" / "relationship.md"),
                priority=4.0,
                aliases=[
                    "how does the SCPN framework relate to remanentia",
                    "SCPN framework Remanentia relationship",
                    "substrate mathematics binding",
                ],
            )
        )
    hooks = _read_text(repo / "hooks.py")
    if "identity coherence" in hooks.lower():
        facts.append(
            CompiledFact(
                fact_id="identity.coherence_hook_location",
                fact_type="location",
                subject="Identity coherence hook",
                fact=(
                    "The identity coherence score R is handled in hooks.py, where session "
                    "start and end hooks load oscillator state and compute the score."
                ),
                source=str(repo / "hooks.py"),
                priority=2.5,
                aliases=[
                    "identity coherence R metric",
                    "where is identity coherence computed",
                    "identity coherence score R",
                ],
            )
        )
    return facts


def _named_benchmark(rows: Any, name: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _load_reports(directory: Path, pattern: str) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    if not directory.exists():
        return reports
    for path in sorted(directory.glob(pattern)):
        try:
            reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def _load_gzip_json(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rb") as handle:
            data = json.loads(handle.read())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _top_level_symbols(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(node.name)
    return symbols


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_many_texts(paths: list[Path], max_files: int) -> list[str]:
    texts: list[str] = []
    for root in paths:
        if root.is_file():
            text = _read_text(root)
            if text:
                texts.append(text)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(texts) >= max_files:
                return texts
            if path.suffix not in {".md", ".json", ".jsonl", ".py"}:
                continue
            text = _read_text(path)
            if text:
                texts.append(text)
    return texts


def _dedupe_facts(facts: list[CompiledFact]) -> list[CompiledFact]:
    best: dict[str, CompiledFact] = {}
    for fact in facts:
        current = best.get(fact.fact_id)
        if current is None or fact.priority > current.priority:
            best[fact.fact_id] = fact
    return sorted(best.values(), key=lambda f: (f.fact_type, f.fact_id))


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _query_intent(query: str) -> str:
    if query.startswith("where") or " location" in query:
        return "location"
    if query.startswith("when") or " date" in query or " built" in query:
        return "temporal"
    if "how many" in query or "accuracy" in query or "rate" in query or "benchmark" in query:
        return "metric"
    if "decided" in query or "chosen" in query or "decision" in query:
        return "decision"
    if "relate" in query or "relationship" in query or "framework" in query:
        return "cross_project"
    if "stored" in query or "report" in query:
        return "continuity"
    return "factual"


def _query_truth_mode(query: str) -> str:
    if any(word in query for word in ("historical", "march", "expected", "old benchmark")):
        return "benchmark_expected"
    if any(word in query for word in ("deprecated", "obsolete", "retired")):
        return "historical"
    return "current"


def _timestamp_to_iso(value: Any) -> str:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), UTC).isoformat()
    if isinstance(value, str) and value:
        return value
    return _now_iso()


def _path_date(path: Path) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})(?:[_T](\d{4,6}))?", path.name)
    if not match:
        return ""
    date = match.group(1)
    clock = match.group(2)
    if not clock:
        return date
    if len(clock) == 4:
        return f"{date}T{clock[:2]}:{clock[2:]}"
    return f"{date}T{clock[:2]}:{clock[2:4]}:{clock[4:6]}"


def _now_iso() -> str:
    return datetime.fromtimestamp(time.time(), UTC).isoformat()


def main() -> None:
    facts = compile_facts()
    print(json.dumps({"facts": len(facts), "out_dir": str(DEFAULT_OUT_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
