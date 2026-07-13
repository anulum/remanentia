# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — SNN daemon monitor dashboard

"""Arcane Sapience SNN Daemon Monitor — live dashboard.

Usage::
    python monitor.py
    # Opens http://localhost:8888 in browser
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from write_discipline import resolve_content

PORT = 8888
BASE = Path(__file__).parent
STATE_DIR = BASE / "snn_state"
STIMULI_DIR = BASE / "snn_stimuli"
HEARTBEAT_DIR = BASE / "heartbeats"
TRACES_DIR = BASE / "reasoning_traces"
SESSION_STATES_DIR = BASE / "session_states"

sys.path.insert(0, str(BASE))


def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    try:
        from retrieve import retrieve

        return retrieve(query, top_k=top_k)
    except Exception as e:
        return [{"error": str(e)}]


def _retrieval_history() -> list[dict]:
    try:
        from retrieve import retrieval_history

        return retrieval_history(limit=50)
    except Exception as e:
        return [{"error": str(e)}]


def _trace_summaries() -> list[dict]:
    try:
        from retrieve import trace_summaries

        return trace_summaries()
    except Exception as e:
        return [{"error": str(e)}]


def _load_network_data() -> tuple[dict | None, str | None]:
    try:
        from retrieve import _load_network

        return _load_network(), None
    except Exception as e:
        return None, str(e)


def _weight_stats() -> dict:
    data, error = _load_network_data()
    if data is None:
        return {"error": error or "no retrieval checkpoint"}
    try:
        import numpy as np

        w = data["w"]
        w_norm = w / (w.max() + 1e-12)
        out_strength = w_norm.sum(axis=1)
        hub_idx = np.argsort(out_strength)[-10:][::-1]
        counts, edges = np.histogram(w[w > 0], bins=10)
        return {
            "weight_mean": float(np.mean(w)),
            "weight_std": float(np.std(w)),
            "sparsity": float((w < 0.01).sum() / w.size),
            "n_neurons": int(w.shape[0]),
            "checkpoint_path": data.get("_checkpoint_path"),
            "encoding_backend": data.get("_encoding_backend"),
            "hub_neurons": hub_idx.tolist(),
            "hub_strengths": [round(float(out_strength[i]), 4) for i in hub_idx],
            "hist_counts": counts.tolist(),
            "hist_edges": [round(float(e), 5) for e in edges],
        }
    except Exception as e:
        return {"error": str(e)}


def _weight_heatmap(bins: int = 50) -> dict:
    """Downsample W to bins×bins for canvas heatmap."""
    data, error = _load_network_data()
    if data is None:
        return {"error": error or "no retrieval checkpoint"}
    try:
        w = data["w"]
        n = w.shape[0]
        block = max(1, n // bins)
        actual_bins = n // block
        # Downsample by block-mean
        trimmed = w[: actual_bins * block, : actual_bins * block]
        reshaped = trimmed.reshape(actual_bins, block, actual_bins, block)
        grid = reshaped.mean(axis=(1, 3))
        return {
            "grid": grid.tolist(),
            "min": float(grid.min()),
            "max": float(grid.max()),
            "size": actual_bins,
        }
    except Exception as e:
        return {"error": str(e)}


def _hub_graph(top_n: int = 20, edges_per_hub: int = 5) -> dict:
    """Top hub neurons and their strongest connections as a graph."""
    data, error = _load_network_data()
    if data is None:
        return {"error": error or "no retrieval checkpoint"}
    try:
        import numpy as np

        w = data["w"]
        w_norm = w / (w.max() + 1e-12)
        out_strength = w_norm.sum(axis=1)
        hub_idx = np.argsort(out_strength)[-top_n:][::-1]
        hub_set = set(hub_idx.tolist())

        nodes = [{"id": int(i), "strength": round(float(out_strength[i]), 4)} for i in hub_idx]
        edges = []
        seen = set()
        for i in hub_idx:
            row = w_norm[i]
            targets = np.argsort(row)[-edges_per_hub:][::-1]
            for t in targets:
                t = int(t)
                if t == int(i) or row[t] < 0.001:
                    continue
                edge_key = (min(int(i), t), max(int(i), t))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                if t not in hub_set:
                    nodes.append({"id": t, "strength": round(float(out_strength[t]), 4)})
                    hub_set.add(t)
                edges.append({"from": int(i), "to": t, "weight": round(float(row[t]), 4)})

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"error": str(e)}


def _session_timeline() -> list[dict]:
    """Build a timeline from session state files."""
    if not SESSION_STATES_DIR.exists():
        return []
    entries = []
    for f in sorted(SESSION_STATES_DIR.glob("*.md"), key=lambda p: p.name):
        name = f.stem
        # Parse date and project from filename: YYYY-MM-DD_project_state
        parts = name.split("_", 2)
        date_str = parts[0] if parts else name
        project = parts[1] if len(parts) > 1 else "unknown"
        entries.append(
            {
                "date": date_str,
                "project": project,
                "file": f.name,
                "size": f.stat().st_size,
            }
        )
    return entries


def _list_traces() -> list[dict]:
    if not TRACES_DIR.exists():
        return []
    result = []
    for f in sorted(TRACES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = f.stat()
        result.append(
            {
                "name": f.name,
                "size": st.st_size,
                "mtime": round(st.st_mtime),
                "mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
            }
        )
    return result


def _related_traces(trace_name: str) -> list[dict]:
    try:
        from retrieve import related_traces

        return related_traces(trace_name)
    except Exception as e:
        return [{"error": str(e)}]


def _query_suggestions(prefix: str) -> list[str]:
    try:
        from retrieve import query_suggestions

        return query_suggestions(prefix)
    except Exception:
        return []


def _session_detail(filename: str) -> dict:
    """Read a session state file and return its content."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.md", filename):
        return {"error": "forbidden"}
    path = (SESSION_STATES_DIR / filename).resolve()
    if not path.is_relative_to(SESSION_STATES_DIR.resolve()):
        return {"error": "forbidden"}
    if not path.exists():
        return {"error": "not found"}
    try:
        content = path.read_text(encoding="utf-8")
        return {"file": filename, "content": content, "size": path.stat().st_size}
    except Exception as e:
        return {"error": str(e)}


_START_TIME = time.time()


def _health() -> dict:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _START_TIME),
        "version": "4.0",
        "port": PORT,
    }


def _consolidation_status() -> dict:
    """Get consolidation pipeline status."""
    graph_dir = BASE / "memory" / "graph"
    semantic_dir = BASE / "memory" / "semantic"
    console_dir = BASE / "consolidation"

    n_semantic = len(list(semantic_dir.rglob("*.md"))) if semantic_dir.exists() else 0
    n_entities = 0
    n_relations = 0
    top_relations = []

    entities_path = graph_dir / "entities.jsonl"
    relations_path = graph_dir / "relations.jsonl"
    if entities_path.exists():
        n_entities = sum(1 for l in entities_path.read_text().strip().split("\n") if l.strip())
    if relations_path.exists():
        rels = [json.loads(l) for l in relations_path.read_text().strip().split("\n") if l.strip()]
        n_relations = len(rels)
        top_relations = sorted(rels, key=lambda r: -r.get("weight", 0))[:10]

    last_run = {}
    last_path = console_dir / "last_consolidation.json"
    if last_path.exists():
        try:
            last_run = json.loads(last_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "semantic_memories": n_semantic,
        "entities": n_entities,
        "relations": n_relations,
        "top_relations": [
            {"source": r["source"], "target": r["target"], "weight": r["weight"]}
            for r in top_relations
        ],
        "last_run": last_run,
    }


def _inject_stimulus(text: str, source: str = "dashboard") -> dict:
    """Inject a stimulus from the dashboard."""
    try:
        from snn_daemon import drop_stimulus

        path = drop_stimulus(text, source=source)
        return {"ok": True, "path": path}
    except Exception as e:
        return {"error": str(e)}


class MonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._serve_dashboard()
        elif path == "/api/current":
            self._serve_json(STATE_DIR / "current_state.json")
        elif path == "/api/history":
            self._serve_history()
        elif path == "/api/stimuli":
            self._serve_stimuli_log()
        elif path == "/api/heartbeats":
            self._serve_heartbeats()
        elif path == "/api/retrieve":
            q = qs.get("q", [""])[0].strip()
            if not q:
                self._json_response({"error": "empty query"}, 400)
            else:
                self._json_response(_retrieve(q))
        elif path == "/api/weight_stats":
            self._json_response(_weight_stats())
        elif path == "/api/weight_heatmap":
            bins = int(qs.get("bins", ["50"])[0])
            self._json_response(_weight_heatmap(bins))
        elif path == "/api/hub_graph":
            self._json_response(_hub_graph())
        elif path == "/api/traces":
            self._json_response(_list_traces())
        elif path == "/api/trace_summaries":
            self._json_response(_trace_summaries())
        elif path == "/api/retrieval_history":
            self._json_response(_retrieval_history())
        elif path == "/api/session_timeline":
            self._json_response(_session_timeline())
        elif path == "/api/health":
            self._json_response(_health())
        elif path == "/api/consolidation":
            self._json_response(_consolidation_status())
        elif path == "/api/session_detail":
            f = qs.get("file", [""])[0].strip()
            if not f:
                self._json_response({"error": "missing file param"}, 400)
            else:
                self._json_response(_session_detail(f))
        elif path == "/api/related":
            t = qs.get("trace", [""])[0].strip()
            if not t:
                self._json_response({"error": "missing trace param"}, 400)
            else:
                self._json_response(_related_traces(t))
        elif path == "/api/suggest":
            q = qs.get("q", [""])[0].strip()
            self._json_response(_query_suggestions(q))
        elif path == "/sw.js":
            self._serve_sw()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/inject_stimulus":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._json_response({"error": "invalid JSON"}, 400)
                return
            text = resolve_content(data)  # canonical 'content' or legacy 'text'
            source = data.get("source", "dashboard")
            if not text:
                self._json_response({"error": "empty text"}, 400)
                return
            self._json_response(_inject_stimulus(text, source))
        else:
            self.send_error(404)

    def _serve_json(self, p: Path):
        if not p.exists():
            self._json_response({"error": "not found"}, 404)
            return
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _serve_history(self):
        p = STATE_DIR / "history.jsonl"
        rows = []
        if p.exists():
            for line in p.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._json_response(rows[-500:])

    def _serve_heartbeats(self):
        agents = []
        if HEARTBEAT_DIR.exists():
            now = time.time()
            for f in sorted(HEARTBEAT_DIR.glob("*.json")):
                try:
                    d = json.loads(f.read_text())
                    age = now - d.get("timestamp", 0)
                    d["age_seconds"] = round(age)
                    d["alive"] = age < 180
                    agents.append(d)
                except (json.JSONDecodeError, OSError):
                    pass
        self._json_response(agents)

    def _serve_stimuli_log(self):
        entries = []
        if STIMULI_DIR.exists():
            for f in sorted(STIMULI_DIR.glob("*.json")):
                try:
                    d = json.loads(f.read_text())
                    entries.append(
                        {
                            "file": f.name,
                            "source": d.get("source", "unknown"),
                            "text": resolve_content(d)[:200],
                            "timestamp": d.get("timestamp", 0),
                            "project": d.get("project", ""),
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    pass
        self._json_response(entries[-100:])

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_dashboard(self):
        html = DASHBOARD_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html)

    def _serve_sw(self):
        sw = b"""self.addEventListener('install',e=>e.waitUntil(caches.open('arcsap-v4').then(c=>c.addAll(['/',
'https://cdn.jsdelivr.net/npm/chart.js@4']))));
self.addEventListener('fetch',e=>{if(e.request.url.includes('/api/'))return;
e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(resp=>{
const c=resp.clone();caches.open('arcsap-v4').then(cache=>cache.put(e.request,c));return resp;})))});"""
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript")
        self.send_header("Service-Worker-Allowed", "/")
        self.end_headers()
        self.wfile.write(sw)

    def log_message(self, format, *args):
        pass


DASHBOARD_PATH = BASE / "monitor_dashboard.html"


MONITOR_LOCK = STATE_DIR / "monitor.lock"


def _is_port_in_use() -> bool:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", PORT))
        sock.close()
        return False
    except OSError:
        return True


def _acquire_monitor_lock() -> bool:
    """Singleton: only one monitor at a time. Checks both lock file AND port."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if _is_port_in_use():
        print(f"Port {PORT} already in use. Another monitor is running.")
        return False

    if MONITOR_LOCK.exists():
        try:
            pid = int(MONITOR_LOCK.read_text().strip())
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                print(
                    f"Monitor already running (PID {pid}). Kill it first or delete {MONITOR_LOCK}"
                )
                return False
        except (ValueError, OSError):
            pass
    MONITOR_LOCK.write_text(str(os.getpid()))
    return True


def _release_monitor_lock():
    try:
        MONITOR_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


def spawn_detached():
    """Spawn monitor as a fully detached process that survives parent exit.

    Uses CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB
    so the monitor is not a child of the launching shell.
    """
    import subprocess

    python = sys.executable
    script = str(Path(__file__).resolve())

    if sys.platform == "win32":
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
        )
        subprocess.Popen(
            [python, script, "--serve"],
            creationflags=flags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            [python, script, "--serve"],
            start_new_session=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def ensure_running() -> dict:
    """Start the monitor if not already running. Safe to call from hooks."""
    if _is_port_in_use():
        return {"status": "already_running", "port": PORT}
    spawn_detached()
    return {"status": "started", "port": PORT}


if __name__ == "__main__":
    if "--serve" not in sys.argv and "--no-detach" not in sys.argv:
        result = ensure_running()
        print(f"Monitor: {result['status']} on port {PORT}")
        if result["status"] == "started":
            time.sleep(1)
            webbrowser.open(f"http://localhost:{PORT}")
        sys.exit(0)

    if not _acquire_monitor_lock():
        sys.exit(1)
    print(f"Arcane Sapience Monitor -> http://localhost:{PORT} (PID {os.getpid()})")
    if "--serve" not in sys.argv:
        webbrowser.open(f"http://localhost:{PORT}")
    server = HTTPServer(("127.0.0.1", PORT), MonitorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        server.server_close()
    finally:
        _release_monitor_lock()
