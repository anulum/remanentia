# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Command-line interface

"""Remanentia command-line interface.

Usage::
    remanentia recall "Dimits shift convergence"
    remanentia recall "quantum control" --format context
    remanentia consolidate
    remanentia consolidate --force
    remanentia status
    remanentia graph --top 10
    remanentia entities
    remanentia daemon start
    remanentia daemon stop
    remanentia daemon status
    remanentia serve --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

from store_paths import StorePaths, resolve_store_paths

BASE = Path(__file__).parent
STATE_DIR = BASE / "snn_state"
GRAPH_DIR = BASE / "memory" / "graph"
_HOSTED_BACKEND = "".join(("anth", "ropic"))
VECTOR_WORKER_SERVICE = "remanentia-vector-worker.service"
Command = Callable[[argparse.Namespace], None]


def _runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load optional runtime integrations without making CLI typing transitive."""
    return getattr(import_module(module_name), attr_name)


def _ensure_utf8() -> None:  # pragma: no cover
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _setup_llm_backend(name: str = "auto") -> None:
    """Resolve and install the LLM backend for this session."""
    resolve_backend = _runtime_attr("llm_backend", "resolve_backend")
    set_llm_backend = _runtime_attr("answer_extractor", "set_llm_backend")

    backend_name = _HOSTED_BACKEND if name == "hosted" else name
    set_llm_backend(resolve_backend(backend_name))


def _cli_store_paths() -> StorePaths:
    """Resolve the store paths visible to operator-facing CLI commands."""
    if os.environ.get("REMANENTIA_BASE") or os.environ.get("REMANENTIA_STIMULI_DIR"):
        return resolve_store_paths()
    return resolve_store_paths(base=BASE)


def cmd_recall(args: argparse.Namespace) -> None:
    """Deep memory recall."""
    # Use filtered search if filters are specified
    has_filters = (
        getattr(args, "project", "") or getattr(args, "after", "") or getattr(args, "before", "")
    )
    if has_filters:
        auto_rebuild_if_needed = _runtime_attr("memory_index", "auto_rebuild_if_needed")

        idx = auto_rebuild_if_needed(use_gpu=False)
        use_llm = getattr(args, "llm", False) or bool(os.environ.get("REMANENTIA_LLM_ANSWERS"))
        if use_llm:
            _setup_llm_backend(getattr(args, "llm_backend", "auto"))
        results = idx.search(
            args.query,
            top_k=args.top,
            project=getattr(args, "project", ""),
            after=getattr(args, "after", ""),
            before=getattr(args, "before", ""),
            use_llm=use_llm,
        )
        for r in results:
            print(f"[{r.source}] {r.name} (score={r.score:.3f})")
            if r.answer:
                print(f"  Answer: {r.answer}")
            print(f"  {r.snippet[:200]}")
            print()
        return

    recall = _runtime_attr("memory_recall", "recall")

    ctx = recall(args.query, top_k=args.top, include_content=args.content)

    if args.format == "summary":
        print(ctx.summary)
    elif args.format == "context":
        print(ctx.to_llm_context())
    elif args.format == "json":
        print(
            json.dumps(
                {
                    "query": ctx.query,
                    "trace": ctx.trace,
                    "score": ctx.trace_score,
                    "entities": ctx.entities,
                    "related": ctx.related_entities,
                    "semantic_memories": len(ctx.semantic_memories),
                    "before": ctx.before,
                    "after": ctx.after,
                    "cross_project": ctx.cross_project,
                    "novelty": ctx.novelty_score,
                    "elapsed_ms": ctx.elapsed_ms,
                },
                indent=2,
            )
        )


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Run memory consolidation."""
    consolidate = _runtime_attr("consolidation_engine", "consolidate")

    print("Running consolidation...")
    t0 = time.monotonic()
    result = consolidate(force=args.force)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s")
    print(json.dumps(result, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    paths = _cli_store_paths()

    # Vector worker — the maintained background path.
    worker = _read_vector_worker_state(paths.state_dir)
    if worker["state"] == "alive":
        print("Vector worker: ALIVE")
        print(f"  Cycle: {worker.get('cycle', '?')}")
        print(f"  Last action: {worker.get('last_action', '?')}")
        print(f"  PID: {worker.get('pid', '?')}")
    elif worker["state"] == "missing":
        print("Vector worker: NOT RUNNING")
    else:
        print(f"Vector worker: {str(worker['state']).upper()}")

    # Index freshness — the watchdog that catches a silently stalled index.
    freshness = _read_freshness_report(paths.state_dir)
    if freshness["state"] == "missing":
        print("Index freshness: NO CHECK YET")
    elif freshness["state"] == "unreadable":
        print("Index freshness: UNREADABLE")
    else:
        verdict = "STALE" if freshness["stale"] else "fresh"
        drift = freshness["drift_days"]
        drift_str = "n/a" if drift is None else f"{drift:.1f}d"
        print(f"Index freshness: {verdict.upper()} (drift {drift_str})")
        print(f"  Checked: {freshness['checked_age']}")

    # Daemon
    state_path = paths.state_dir / "current_state.json"
    if state_path.exists():
        s = json.loads(state_path.read_text(encoding="utf-8"))
        age = time.time() - s.get("timestamp", 0)
        status = "ALIVE" if age < 120 else f"STALE ({age:.0f}s ago)"
        print(f"Legacy daemon: {status}")
        print(f"  Cycle: {s.get('cycle', '?')}")
        print(f"  Neurons: {s.get('n_neurons', '?')}")
        print(f"  VRAM: {s.get('vram_mb', '?')} MB")
        print(f"  Live retrieval: {s.get('live_retrieval_available', False)}")
        console = s.get("last_consolidation")
        if console:
            print(
                f"  Last consolidation: {console.get('memories_written', 0)} memories, "
                f"{console.get('entities_found', 0)} entities"
            )
    else:
        print("Legacy daemon: NOT RUNNING")

    # Dashboard
    try:
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", 8888, timeout=2)
        try:
            conn.request("GET", "/api/health")
            resp = conn.getresponse()
            health = json.loads(resp.read().decode("utf-8"))  # pragma: no cover
        finally:
            conn.close()
        print(
            f"Dashboard: UP (port {health.get('port')}, uptime {health.get('uptime_s', 0):.0f}s)"
        )  # pragma: no cover
    except (OSError, TimeoutError, json.JSONDecodeError):
        print("Dashboard: DOWN")

    # Memory stats
    traces_dir = paths.traces_dir
    semantic_dir = paths.semantic_dir
    n_traces = len(list(traces_dir.glob("*.md"))) if traces_dir.exists() else 0
    n_semantic = len(list(semantic_dir.rglob("*.md"))) if semantic_dir.exists() else 0

    entities_path = paths.graph_dir / "entities.jsonl"
    relations_path = paths.graph_dir / "relations.jsonl"
    n_entities = 0
    n_relations = 0
    if entities_path.exists():
        n_entities = sum(1 for l in entities_path.read_text().strip().split("\n") if l.strip())
    if relations_path.exists():
        n_relations = sum(1 for l in relations_path.read_text().strip().split("\n") if l.strip())

    print(f"\nMemory:")
    print(f"  Episodic traces: {n_traces}")
    print(f"  Semantic memories: {n_semantic}")
    print(f"  Entities: {n_entities}")
    print(f"  Relations: {n_relations}")

    # Capacity report per category
    try:
        capacity_report = _runtime_attr("consolidation_engine", "capacity_report")

        report = capacity_report()
        if report:
            print("\n  Capacity:")
            for cat, info in sorted(report.items()):
                bar = "!" if info["needs_consolidation"] else " "
                state_str = ", ".join(f"{s}={c}" for s, c in sorted(info["state_counts"].items()))
                print(
                    f"   {bar} {cat:15s} {info['usage_pct']:5.1f}% "
                    f"({info['chars']:,} / {info['limit']:,} chars, "
                    f"{info['file_count']} files) [{state_str}]"
                )
    except Exception as exc:  # pragma: no cover
        print(f"\n  Capacity: unavailable ({type(exc).__name__})")

    # Disk usage
    total = 0
    for d in [paths.state_dir, traces_dir, semantic_dir, paths.graph_dir]:
        if d.exists():
            total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"  Disk: {total / 1024 / 1024:.1f} MB")


def cmd_store_manifest(args: argparse.Namespace) -> None:
    """Show or persist the resolved memory-store selection manifest."""
    from store_manifest import build_store_manifest, render_store_manifest, write_store_manifest

    manifest = build_store_manifest(base=args.base, stimuli_dir=args.stimuli_dir)
    if args.write:
        write_store_manifest(manifest, args.output)
    if args.json:
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_store_manifest(manifest))


def cmd_graph(args: argparse.Namespace) -> None:
    """Show top entity relationships."""
    relations_path = GRAPH_DIR / "relations.jsonl"
    if not relations_path.exists():
        print("No relations. Run: remanentia consolidate")
        return
    rels = [json.loads(l) for l in relations_path.read_text().strip().split("\n") if l.strip()]
    top = sorted(rels, key=lambda r: -r.get("weight", 0))[: args.top]
    print(f"Top {len(top)} entity relationships:\n")
    for r in top:
        evidence = len(r.get("evidence", []))
        print(
            f"  {r['source']:25s} <-> {r['target']:25s}  weight={r['weight']:2d}  ({evidence} traces)"
        )


def cmd_entities(args: argparse.Namespace) -> None:
    """List all known entities."""
    entities_path = GRAPH_DIR / "entities.jsonl"
    if not entities_path.exists():
        print("No entities. Run: remanentia consolidate")
        return
    entities = [json.loads(l) for l in entities_path.read_text().strip().split("\n") if l.strip()]
    entities.sort(key=lambda e: -e.get("trace_count", 0))
    print(f"{len(entities)} entities:\n")
    for e in entities:
        print(f"  {e['id']:30s}  type={e.get('type', '?'):10s}  traces={e.get('trace_count', 0)}")


def cmd_init(args: argparse.Namespace) -> None:
    """Create memory directory structure."""
    paths = _cli_store_paths()
    dirs = [
        paths.traces_dir,
        paths.semantic_dir,
        paths.graph_dir,
        paths.consolidation_dir,
        paths.state_dir,
    ]
    created = 0
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True)
            created += 1
            print(f"  Created {d.relative_to(paths.base)}")
    if created == 0:
        print("All directories already exist.")
    else:
        print(f"\n{created} directories created. Ready to use.")
        print("Add reasoning traces to reasoning_traces/ then run: remanentia consolidate")


def cmd_daemon(args: argparse.Namespace) -> None:
    """Background worker management."""
    if args.action == "start":
        if _systemd_user_unit_available(VECTOR_WORKER_SERVICE):
            _systemctl_user("start", VECTOR_WORKER_SERVICE)
            print(f"Vector worker service start requested: {VECTOR_WORKER_SERVICE}")
            return
        subprocess.Popen(
            [sys.executable, "-m", "vector_pipeline", "watch", "--interval-s", "900"],
            cwd=BASE,
        )
        print("Vector worker start requested")
    elif args.action == "stop":
        if _systemd_user_unit_available(VECTOR_WORKER_SERVICE):
            _systemctl_user("stop", VECTOR_WORKER_SERVICE)
            print(f"Vector worker service stop requested: {VECTOR_WORKER_SERVICE}")
            return
        worker = _read_vector_worker_state()
        pid = worker.get("pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to vector worker PID {pid}")
            except OSError as e:
                print(f"Failed to stop vector worker PID {pid}: {e}")
        else:
            print("No vector worker PID found")
    elif args.action == "status":
        cmd_status(args)


def _read_vector_worker_state(state_dir: Path | None = None) -> dict[str, object]:
    root = STATE_DIR if state_dir is None else state_dir
    heartbeat = root / "vector_refresh_worker.json"
    if not heartbeat.exists():
        return {"state": "missing"}
    try:
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unreadable"}
    age = time.time() - float(payload.get("timestamp_unix", 0))
    result = payload.get("result")
    return {
        "age_s": round(age),
        "cycle": payload.get("cycle"),
        "last_action": result.get("action") if isinstance(result, dict) else None,
        "pid": payload.get("pid"),
        "state": "alive" if age < 1800 else "stale",
        "status": payload.get("status"),
    }


def _read_freshness_report(state_dir: Path | None = None) -> dict[str, object]:
    """Read the index-freshness watchdog's last verdict for the status view.

    The daily watchdog writes ``snn_state/index_freshness.json``; surfacing it
    here is the whole point of the watchdog — a stalled index was invisible for
    eight weeks because no status line ever reported the drift.
    """
    root = STATE_DIR if state_dir is None else state_dir
    report = root / "index_freshness.json"
    if not report.exists():
        return {"state": "missing"}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unreadable"}
    checked = payload.get("checked_at_unix")
    if isinstance(checked, (int, float)):
        age_s = round(time.time() - float(checked))
        checked_age = f"{age_s}s ago" if age_s < 3600 else f"{age_s / 86400:.1f}d ago"
    else:
        checked_age = "unknown"
    return {
        "state": "present",
        "stale": bool(payload.get("stale")),
        "drift_days": payload.get("drift_days"),
        "checked_age": checked_age,
    }


def _systemctl_user(action: str, unit: str) -> None:
    subprocess.run(["systemctl", "--user", action, unit], check=True)


def _systemd_user_unit_available(unit: str) -> bool:
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "cat", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def cmd_observe(args: argparse.Namespace) -> None:  # pragma: no cover
    """Watch filesystem for changes, auto-create knowledge notes."""
    ObserverState = _runtime_attr("observer", "ObserverState")
    observe_once = _runtime_attr("observer", "observe_once")
    observe_loop = _runtime_attr("observer", "observe_loop")

    if args.once:
        state = ObserverState()
        state.load()
        result = observe_once(state)
        print(json.dumps(result, indent=2))
        state.save()
    else:
        observe_loop(interval=args.interval)


def cmd_reflect(args: argparse.Namespace) -> None:  # pragma: no cover
    """Run deep consolidation via reflector."""
    reflect_once = _runtime_attr("reflector", "reflect_once")

    if args.llm:
        _setup_llm_backend(getattr(args, "llm_backend", "auto"))

    print(f"Reflecting on last {args.days} days...")
    result = reflect_once(days=args.days, use_llm=args.llm)
    if result.get("digest"):
        print(result["digest"])
    else:
        print(json.dumps(result, indent=2))


def cmd_setup_llm(args: argparse.Namespace) -> None:  # pragma: no cover
    """Detect hardware and configure local LLM."""
    _setup = _runtime_attr("llm_setup", "cmd_setup_llm")

    _setup(args)


def cmd_serve(args: argparse.Namespace) -> None:  # pragma: no cover
    """Start the FastAPI REST server."""
    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port)


def cmd_serve_llm(args: argparse.Namespace) -> None:  # pragma: no cover
    """Start local LLM server."""
    _serve = _runtime_attr("llm_setup", "cmd_serve_llm")

    _serve(args)


def cmd_notes(args: argparse.Namespace) -> None:  # pragma: no cover
    """Show knowledge notes."""
    KnowledgeStore = _runtime_attr("knowledge_store", "KnowledgeStore")

    store = KnowledgeStore()
    if not store.load():
        print("No knowledge notes. Run: remanentia observe --once")
        return
    s = store.stats
    print(
        f"Knowledge Store: {s['notes']} notes, {s['links']} links, "
        f"{s['contradictions']} contradictions, {s['triggers_active']} active triggers\n"
    )
    notes = sorted(store.notes.values(), key=lambda n: n.updated, reverse=True)
    for n in notes[: args.top]:
        status = ""
        if n.superseded_by:
            status = " [SUPERSEDED]"
        elif n.supersedes:
            status = " [SUPERSEDES older]"
        print(f"  [{n.id}] {n.title}{status}")
        print(f"    Source: {n.source} | Updated: {n.updated} | Links: {len(n.links)}")
        if n.entities:
            print(f"    Entities: {', '.join(n.entities[:8])}")
        print()


def main() -> None:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="remanentia",
        description="Persistent AI memory with SNN-orchestrated consolidation",
    )
    sub = parser.add_subparsers(dest="command")

    # recall / search
    for name in ("recall", "search"):
        p_recall = sub.add_parser(name, help="Memory recall / search")
        p_recall.add_argument("query", help="Query text")
        p_recall.add_argument("--top", type=int, default=3, help="Number of results")
        p_recall.add_argument("--format", choices=["summary", "context", "json"], default="summary")
        p_recall.add_argument("--content", action="store_true", help="Include trace content")
        p_recall.add_argument("--project", default="", help="Filter by project/source")
        p_recall.add_argument("--after", default="", help="Filter: docs after date (YYYY-MM-DD)")
        p_recall.add_argument("--before", default="", help="Filter: docs before date (YYYY-MM-DD)")
        p_recall.add_argument("--llm", action="store_true", help="Use LLM for answer extraction")
        p_recall.add_argument(
            "--llm-backend",
            choices=["auto", "local", "hosted", "none"],
            default="auto",
            help="LLM backend to use (default: auto)",
        )

    # consolidate
    p_console = sub.add_parser("consolidate", help="Run memory consolidation")
    p_console.add_argument("--force", action="store_true", help="Reconsolidate all traces")

    # status
    sub.add_parser("status", help="System status")

    # store manifest
    p_store = sub.add_parser("store-manifest", help="Show the selected memory store")
    p_store.add_argument("--base", default=None, help="Override REMANENTIA_BASE for this run")
    p_store.add_argument(
        "--stimuli-dir",
        default=None,
        help="Override REMANENTIA_STIMULI_DIR for this run",
    )
    p_store.add_argument("--write", action="store_true", help="Write the manifest JSON")
    p_store.add_argument("--output", default=None, help="Manifest output path when writing")
    p_store.add_argument("--json", action="store_true", help="Print JSON instead of text")

    # graph
    p_graph = sub.add_parser("graph", help="Show entity relationships")
    p_graph.add_argument("--top", type=int, default=15, help="Number of relationships")

    # entities
    sub.add_parser("entities", help="List known entities")

    # daemon
    p_daemon = sub.add_parser("daemon", help="Daemon management")
    p_daemon.add_argument("action", choices=["start", "stop", "status"])

    # init
    sub.add_parser("init", help="Create memory directory structure")

    # observe
    p_observe = sub.add_parser(
        "observe", help="Watch filesystem for changes, auto-create knowledge notes"
    )
    p_observe.add_argument("--once", action="store_true", help="Run once and exit")
    p_observe.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

    # reflect
    p_reflect = sub.add_parser("reflect", help="Run deep consolidation (LLM-powered reflection)")
    p_reflect.add_argument("--days", type=int, default=7, help="Process notes from last N days")
    p_reflect.add_argument(
        "--llm", action="store_true", help="Use LLM for summaries and prospective queries"
    )
    p_reflect.add_argument(
        "--llm-backend",
        choices=["auto", "local", "hosted", "none"],
        default="auto",
        help="LLM backend to use (default: auto)",
    )

    # setup-llm
    p_setup_llm = sub.add_parser("setup-llm", help="Detect hardware and configure local LLM")
    p_setup_llm.add_argument("--model", default=None, help="Model name override")
    p_setup_llm.add_argument("--quant", default="q4_k_m", help="Quantisation level")
    p_setup_llm.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu)")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8001, help="Bind port")

    # serve-llm
    p_serve_llm = sub.add_parser("serve-llm", help="Start local LLM server")
    p_serve_llm.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve_llm.add_argument("--port", type=int, default=8080, help="Server port")

    # notes
    p_notes = sub.add_parser("notes", help="Show knowledge notes")
    p_notes.add_argument("--top", type=int, default=10, help="Number of notes to show")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmd_map: dict[str, Command] = {
        "recall": cmd_recall,
        "search": cmd_recall,
        "consolidate": cmd_consolidate,
        "status": cmd_status,
        "store-manifest": cmd_store_manifest,
        "graph": cmd_graph,
        "entities": cmd_entities,
        "daemon": cmd_daemon,
        "init": cmd_init,
        "observe": cmd_observe,
        "reflect": cmd_reflect,
        "notes": cmd_notes,
        "setup-llm": cmd_setup_llm,
        "serve": cmd_serve,
        "serve-llm": cmd_serve_llm,
    }
    command = str(args.command)
    cmd_map[command](args)


if __name__ == "__main__":
    main()
