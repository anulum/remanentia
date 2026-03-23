# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CLI

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
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
STATE_DIR = BASE / "snn_state"
GRAPH_DIR = BASE / "memory" / "graph"


def _ensure_utf8():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def cmd_recall(args):
    """Deep memory recall."""
    from memory_recall import recall
    ctx = recall(args.query, top_k=args.top, include_content=args.content)

    if args.format == "summary":
        print(ctx.summary)
    elif args.format == "context":
        print(ctx.to_llm_context())
    elif args.format == "json":
        print(json.dumps({
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
        }, indent=2))


def cmd_consolidate(args):
    """Run memory consolidation."""
    from consolidation_engine import consolidate
    print("Running consolidation...")
    t0 = time.monotonic()
    result = consolidate(force=args.force)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s")
    print(json.dumps(result, indent=2))


def cmd_status(args):
    """Show system status."""
    # Daemon
    state_path = STATE_DIR / "current_state.json"
    if state_path.exists():
        s = json.loads(state_path.read_text(encoding="utf-8"))
        age = time.time() - s.get("timestamp", 0)
        status = "ALIVE" if age < 120 else f"STALE ({age:.0f}s ago)"
        print(f"Daemon:  {status}")
        print(f"  Cycle: {s.get('cycle', '?')}")
        print(f"  Neurons: {s.get('n_neurons', '?')}")
        print(f"  VRAM: {s.get('vram_mb', '?')} MB")
        print(f"  Live retrieval: {s.get('live_retrieval_available', False)}")
        consol = s.get("last_consolidation")
        if consol:
            print(f"  Last consolidation: {consol.get('memories_written', 0)} memories, "
                  f"{consol.get('entities_found', 0)} entities")
    else:
        print("Daemon:  NOT RUNNING")

    # Dashboard
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8888/api/health", timeout=2)
        health = json.loads(resp.read())
        print(f"Dashboard: UP (port {health.get('port')}, uptime {health.get('uptime_s', 0):.0f}s)")
    except Exception:
        print("Dashboard: DOWN")

    # Memory stats
    traces_dir = BASE / "reasoning_traces"
    semantic_dir = BASE / "memory" / "semantic"
    n_traces = len(list(traces_dir.glob("*.md"))) if traces_dir.exists() else 0
    n_semantic = len(list(semantic_dir.rglob("*.md"))) if semantic_dir.exists() else 0

    entities_path = GRAPH_DIR / "entities.jsonl"
    relations_path = GRAPH_DIR / "relations.jsonl"
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

    # Disk usage
    total = 0
    for d in [STATE_DIR, traces_dir, semantic_dir, GRAPH_DIR]:
        if d.exists():
            total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"  Disk: {total / 1024 / 1024:.1f} MB")


def cmd_graph(args):
    """Show top entity relationships."""
    relations_path = GRAPH_DIR / "relations.jsonl"
    if not relations_path.exists():
        print("No relations. Run: remanentia consolidate")
        return
    rels = [json.loads(l) for l in relations_path.read_text().strip().split("\n") if l.strip()]
    top = sorted(rels, key=lambda r: -r.get("weight", 0))[:args.top]
    print(f"Top {len(top)} entity relationships:\n")
    for r in top:
        evidence = len(r.get("evidence", []))
        print(f"  {r['source']:25s} <-> {r['target']:25s}  weight={r['weight']:2d}  ({evidence} traces)")


def cmd_entities(args):
    """List all known entities."""
    entities_path = GRAPH_DIR / "entities.jsonl"
    if not entities_path.exists():
        print("No entities. Run: remanentia consolidate")
        return
    entities = [json.loads(l) for l in entities_path.read_text().strip().split("\n") if l.strip()]
    entities.sort(key=lambda e: -e.get("trace_count", 0))
    print(f"{len(entities)} entities:\n")
    for e in entities:
        print(f"  {e['id']:30s}  type={e.get('type','?'):10s}  traces={e.get('trace_count', 0)}")


def cmd_init(args):
    """Create memory directory structure."""
    dirs = [
        BASE / "reasoning_traces",
        BASE / "memory" / "semantic",
        BASE / "memory" / "graph",
        BASE / "consolidation",
        BASE / "snn_state",
    ]
    created = 0
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True)
            created += 1
            print(f"  Created {d.relative_to(BASE)}")
    if created == 0:
        print("All directories already exist.")
    else:
        print(f"\n{created} directories created. Ready to use.")
        print("Add reasoning traces to reasoning_traces/ then run: remanentia consolidate")


def cmd_daemon(args):
    """Daemon management."""
    if args.action == "start":
        import subprocess
        script = str(BASE / "gpu_daemon.py")
        subprocess.Popen([sys.executable, script, "--detach"])
        print("Daemon start requested")
    elif args.action == "stop":
        lock = STATE_DIR / "daemon.lock"
        if lock.exists():
            pid = int(lock.read_text().strip())
            import signal
            try:
                import os
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except OSError as e:
                print(f"Failed to stop PID {pid}: {e}")
        else:
            print("No daemon lock found")
    elif args.action == "status":
        cmd_status(args)


def main():
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

    # consolidate
    p_consol = sub.add_parser("consolidate", help="Run memory consolidation")
    p_consol.add_argument("--force", action="store_true", help="Reconsolidate all traces")

    # status
    sub.add_parser("status", help="System status")

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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmd_map = {
        "recall": cmd_recall,
        "search": cmd_recall,
        "consolidate": cmd_consolidate,
        "status": cmd_status,
        "graph": cmd_graph,
        "entities": cmd_entities,
        "daemon": cmd_daemon,
        "init": cmd_init,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
