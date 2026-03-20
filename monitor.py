# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Arcane Sapience SNN Daemon Monitor — live dashboard.

Usage::
    python 04_ARCANE_SAPIENCE/monitor.py
    # Opens http://localhost:8888 in browser
"""
from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
        import numpy as np
        w = data["w"]
        n = w.shape[0]
        block = max(1, n // bins)
        actual_bins = n // block
        # Downsample by block-mean
        trimmed = w[:actual_bins * block, :actual_bins * block]
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
        entries.append({
            "date": date_str,
            "project": project,
            "file": f.name,
            "size": f.stat().st_size,
        })
    return entries


def _list_traces() -> list[dict]:
    if not TRACES_DIR.exists():
        return []
    result = []
    for f in sorted(TRACES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = f.stat()
        result.append({
            "name": f.name,
            "size": st.st_size,
            "mtime": round(st.st_mtime),
            "mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
        })
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
    except Exception as e:
        return []


def _session_detail(filename: str) -> dict:
    """Read a session state file and return its content."""
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
    consol_dir = BASE / "consolidation"

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
    last_path = consol_dir / "last_consolidation.json"
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
            text = data.get("text", "").strip()
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
                    entries.append({
                        "file": f.name,
                        "source": d.get("source", "unknown"),
                        "text": d.get("text", "")[:200],
                        "timestamp": d.get("timestamp", 0),
                        "project": d.get("project", ""),
                    })
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
        html = DASHBOARD_HTML.encode()
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


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arcane Sapience — SNN Monitor</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><polygon points='16,2 30,16 16,30 2,16' fill='%236060c0'/><circle cx='16' cy='16' r='5' fill='%2340ff40' opacity='0.8'/></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0a0a0f; --bg-card: #14141e; --bg-input: #0e0e18;
    --border: #2a2a3a; --border-focus: #6060a0;
    --text: #c8c8d0; --text-dim: #8888a0; --text-muted: #505070; --text-dark: #404060;
    --accent: #8080ff; --accent-green: #40ff40; --accent-red: #ff4040;
    --grid-line: #1a1a2a; --row-border: #1a1a2a;
  }
  [data-theme="light"] {
    --bg: #f0f0f5; --bg-card: #ffffff; --bg-input: #f5f5fa;
    --border: #d0d0e0; --border-focus: #6060a0;
    --text: #202030; --text-dim: #606080; --text-muted: #8888a0; --text-dark: #a0a0b0;
    --accent: #4040c0; --accent-green: #208020; --accent-red: #c03030;
    --grid-line: #e0e0f0; --row-border: #e8e8f0;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Cascadia Code','Fira Code',monospace; font-size:13px; transition:background 0.3s, color 0.3s; }

  .header { padding:10px 20px; background:var(--bg-card); border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  .header h1 { font-size:15px; color:var(--accent); font-weight:normal; white-space:nowrap; }
  .header .status { font-size:12px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .status .alive { color:#40ff40; }
  .status .dead  { color:#ff4040; }
  .kbd-hint { color:#404060; font-size:10px; cursor:pointer; border:1px solid #2a2a3a; padding:1px 6px; border-radius:3px; }
  .kbd-hint:hover { color:#6060a0; border-color:#4040a0; }
  .pulse-dot { width:7px; height:7px; border-radius:50%; background:#40ff40; display:inline-block; margin-right:4px; }
  .pulse-dot.active { animation: pulse 1.4s ease-in-out infinite; }
  .pulse-dot.stale  { background:#ff4040; animation:none; }
  @keyframes pulse {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(64,255,64,0.6); }
    50%      { opacity:0.7; box-shadow:0 0 0 5px rgba(64,255,64,0); }
  }

  .full-spike { padding:0 12px 0; }
  .grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; padding:12px; }
  @media(max-width:1100px) { .grid { grid-template-columns:1fr 1fr; } }
  @media(max-width:700px) {
    .grid { grid-template-columns:1fr; }
    .header { flex-direction:column; align-items:flex-start; }
    .header h1 { font-size:13px; }
    .search-row input, .inject-row textarea { font-size:16px; min-height:44px; }
    .search-row button, .inject-row button { min-height:44px; padding:8px 14px; }
  }
  .full-width { grid-column:1/-1; }

  .card { background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:12px; transition:background 0.3s; }
  .card h2 { font-size:12px; color:var(--accent); margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; }

  .metric { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #1a1a2a; }
  .metric:last-child { border:none; }
  .metric .label { color:#8888a0; }
  .metric .value { color:#e0e0ff; font-weight:bold; }

  .chart-container      { height:150px; position:relative; }
  .chart-container-full { height:130px; position:relative; }
  canvas { width:100% !important; }

  .instance { display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #1a1a2a; flex-wrap:wrap; }
  .instance:last-child { border:none; }
  .dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .dot.alive { background:#40ff40; box-shadow:0 0 6px #40ff40; }
  .dot.dead  { background:#ff4040; }
  .inst-name    { color:#c0c0ff; font-weight:bold; min-width:100px; }
  .inst-project { color:#a0a080; font-size:11px; }
  .inst-detail  { color:#606080; font-size:11px; }
  .inst-age     { color:#606060; font-size:10px; margin-left:auto; white-space:nowrap; }

  .sources { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
  .source-tag { padding:2px 8px; border-radius:3px; font-size:11px; }
  .source-tag.arcane    { background:#2a1a4a; color:#a080ff; }
  .source-tag.codex     { background:#1a3a1a; color:#40c040; }
  .source-tag.gemini    { background:#3a2a1a; color:#c0a040; }
  .source-tag.unknown   { background:#2a2a2a; color:#808080; }
  .source-tag.dashboard { background:#1a2a3a; color:#4080c0; }
  .source-tag.git       { background:#1a2a1a; color:#60a060; }
  .stimuli-list { max-height:130px; overflow-y:auto; }
  .stim-entry { padding:3px 0; border-bottom:1px solid #1a1a2a; }
  .stim-entry .txt { color:#8888a0; font-size:11px; }

  .search-row { display:flex; gap:6px; margin-bottom:8px; }
  .search-row input {
    flex:1; background:#0e0e18; border:1px solid #3a3a5a; border-radius:4px;
    color:#e0e0ff; padding:5px 8px; font-family:inherit; font-size:12px; outline:none;
  }
  .search-row input:focus { border-color:#6060a0; }
  .search-row button, .inject-row button {
    background:#1e1e3a; border:1px solid #3a3a5a; border-radius:4px;
    color:#a0a0ff; padding:5px 10px; cursor:pointer; font-family:inherit; font-size:12px;
  }
  .search-row button:hover, .inject-row button:hover { background:#2a2a4a; }
  .search-results { max-height:200px; overflow-y:auto; }
  .search-result { padding:5px 0; border-bottom:1px solid #1a1a2a; }
  .search-result .trace-name { color:#8080ff; font-size:12px; }
  .search-result .sr-summary { color:#505070; font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .search-result .scores { color:#606080; font-size:11px; margin-top:2px; }
  .search-result .score-bar { height:3px; background:#8080ff; border-radius:2px; margin-top:3px; }
  .search-empty { color:#505070; font-size:11px; padding:6px 0; }

  .inject-row { display:flex; gap:6px; margin-bottom:6px; }
  .inject-row textarea {
    flex:1; background:#0e0e18; border:1px solid #3a3a5a; border-radius:4px;
    color:#e0e0ff; padding:5px 8px; font-family:inherit; font-size:12px;
    outline:none; resize:vertical; min-height:36px; max-height:120px;
  }
  .inject-row textarea:focus { border-color:#60a060; }
  .inject-status { font-size:10px; color:#505070; min-height:14px; }

  .traces-list { max-height:220px; overflow-y:auto; }
  .trace-row { display:flex; align-items:baseline; gap:8px; padding:4px 0; border-bottom:1px solid #1a1a2a; font-size:11px; }
  .trace-row:last-child { border:none; }
  .trace-row .tname { color:#8888cc; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .trace-row .tsummary { color:#606080; font-size:10px; flex:2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .trace-row .tsize { color:#505070; white-space:nowrap; }
  .trace-row .tdate { color:#404060; white-space:nowrap; }

  .hub-row { display:flex; align-items:center; gap:6px; padding:2px 0; font-size:11px; }
  .hub-row .hub-id  { color:#6060a0; width:40px; text-align:right; flex-shrink:0; }
  .hub-bar-wrap { flex:1; background:#1a1a2a; border-radius:2px; height:8px; }
  .hub-bar      { height:8px; border-radius:2px; background:#6060c0; }
  .hub-val      { color:#8080b0; width:50px; text-align:right; flex-shrink:0; }

  .depth-list { max-height:none; overflow:visible; }
  .depth-row { display:flex; align-items:center; gap:5px; padding:1px 0; font-size:10px; }
  .depth-id   { color:#505070; width:22px; text-align:right; flex-shrink:0; }
  .depth-bar-wrap { flex:1; background:#1a1a2a; border-radius:1px; height:6px; }
  .depth-bar       { height:6px; border-radius:1px; }
  .depth-val  { color:#8080b0; width:58px; text-align:right; flex-shrink:0; font-size:10px; }

  .heatmap-wrap { display:flex; flex-direction:column; align-items:center; position:relative; }
  .heatmap-wrap canvas { image-rendering:pixelated; border:1px solid #2a2a3a; cursor:crosshair; }
  .heatmap-legend { display:flex; justify-content:space-between; font-size:10px; color:#505070; margin-top:4px; width:100%; }
  .hm-info { text-align:center; font-size:10px; color:#6060a0; min-height:14px; margin-top:2px; }

  .graph-wrap { position:relative; }
  .graph-wrap canvas { border:1px solid #2a2a3a; border-radius:4px; width:100%; height:300px; cursor:grab; }
  .graph-wrap canvas.dragging { cursor:grabbing; }

  .timeline-wrap { overflow-x:auto; padding:4px 0; }
  .timeline-bar { display:flex; gap:3px; align-items:flex-end; min-height:50px; }
  .tl-item { display:flex; flex-direction:column; align-items:center; min-width:30px; cursor:default; }
  .tl-block { width:24px; border-radius:3px 3px 0 0; min-height:8px; transition:opacity 0.15s; }
  .tl-item:hover .tl-block { opacity:0.7; }
  .tl-date { font-size:7px; color:#303050; margin-top:1px; }

  .rh-list { max-height:200px; overflow-y:auto; }
  .rh-entry { padding:4px 0; border-bottom:1px solid #1a1a2a; cursor:pointer; }
  .rh-entry:hover { background:#1a1a2a; }
  .rh-entry .rh-query { color:#8080cc; font-size:12px; }
  .rh-entry .rh-meta { color:#505070; font-size:10px; }

  /* Animated metric values */
  .metric .value { transition: color 0.3s; }

  /* Raster heat strip */
  .raster-strip { height:24px; width:100%; margin-top:4px; border-radius:2px; overflow:hidden; }
  .raster-strip canvas { width:100%; height:24px; image-rendering:pixelated; }

  /* Autocomplete */
  .ac-wrap { position:relative; }
  .ac-list { position:absolute; top:100%; left:0; right:50px; background:var(--bg-card); border:1px solid var(--border); border-top:none; border-radius:0 0 4px 4px; z-index:50; display:none; max-height:150px; overflow-y:auto; }
  .ac-list.visible { display:block; }
  .ac-item { padding:4px 8px; font-size:11px; color:var(--text-dim); cursor:pointer; }
  .ac-item:hover { background:var(--grid-line); color:var(--text); }

  /* Session detail modal */
  .detail-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:200; display:none; justify-content:center; align-items:center; padding:20px; }
  .detail-overlay.visible { display:flex; }
  .detail-content { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:16px; max-width:700px; width:100%; max-height:80vh; overflow-y:auto; }
  .detail-content h3 { color:var(--accent); font-size:14px; margin-bottom:8px; }
  .detail-content pre { color:var(--text-dim); font-size:11px; white-space:pre-wrap; word-break:break-word; line-height:1.5; }
  .detail-close { float:right; cursor:pointer; color:var(--text-muted); font-size:16px; padding:4px; }
  .detail-close:hover { color:var(--accent-red); }

  /* Theme toggle */
  .theme-btn { background:none; border:1px solid var(--border); border-radius:3px; color:var(--text-muted); font-size:14px; cursor:pointer; padding:1px 6px; line-height:1.2; }
  .theme-btn:hover { border-color:var(--accent); color:var(--accent); }

  /* Floating tooltip */
  #tooltip { position:fixed; background:var(--bg-card); border:1px solid var(--accent); border-radius:4px; padding:4px 8px; font-size:10px; color:var(--text); pointer-events:none; z-index:100; display:none; max-width:300px; white-space:nowrap; }

  /* Shortcuts modal */
  .modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:200; display:none; justify-content:center; align-items:center; }
  .modal-overlay.visible { display:flex; }
  .modal { background:#14141e; border:1px solid #3a3a5a; border-radius:8px; padding:20px; max-width:340px; width:90%; }
  .modal h3 { color:#8080ff; font-size:14px; margin-bottom:12px; }
  .modal .shortcut { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #1a1a2a; }
  .modal .shortcut:last-child { border:none; }
  .modal .key { background:#1a1a2a; color:#a0a0ff; padding:1px 6px; border-radius:3px; font-size:11px; }
  .modal .desc { color:#8888a0; font-size:12px; }

  ::-webkit-scrollbar       { width:4px; }
  ::-webkit-scrollbar-track { background:#0e0e18; }
  ::-webkit-scrollbar-thumb { background:#2a2a4a; border-radius:2px; }
</style>
</head>
<body>

<div id="tooltip"></div>

<div class="modal-overlay" id="shortcuts-modal" onclick="this.classList.remove('visible')">
  <div class="modal" onclick="event.stopPropagation()">
    <h3>Keyboard Shortcuts</h3>
    <div class="shortcut"><span class="key">/</span><span class="desc">Focus search</span></div>
    <div class="shortcut"><span class="key">Enter</span><span class="desc">Execute search / inject</span></div>
    <div class="shortcut"><span class="key">?</span><span class="desc">Show this help</span></div>
    <div class="shortcut"><span class="key">Esc</span><span class="desc">Close modal</span></div>
    <div class="shortcut"><span class="key">Shift+Enter</span><span class="desc">Newline in inject box</span></div>
    <div style="margin-top:12px;color:#404060;font-size:10px;text-align:center">Heatmap: hover for cell values | Graph: drag nodes</div>
  </div>
</div>

<div class="detail-overlay" id="detail-modal" onclick="this.classList.remove('visible')">
  <div class="detail-content" onclick="event.stopPropagation()">
    <span class="detail-close" onclick="document.getElementById('detail-modal').classList.remove('visible')">&times;</span>
    <h3 id="detail-title">Session Detail</h3>
    <pre id="detail-body">Loading...</pre>
  </div>
</div>

<div class="header">
  <h1>&#9670; Arcane Sapience — SNN Daemon Monitor</h1>
  <div class="status">
    <span class="pulse-dot stale" id="pulse"></span>
    Daemon: <span id="daemon-status" class="dead">checking...</span>
    &nbsp;|&nbsp; Last update: <span id="last-update">—</span>
    &nbsp; <span class="kbd-hint" onclick="document.getElementById('shortcuts-modal').classList.add('visible')">?</span>
    &nbsp; <button class="theme-btn" onclick="toggleTheme()" title="Toggle dark/light">&#9681;</button>
  </div>
</div>

<div class="full-spike">
  <div class="card" style="margin-top:12px">
    <h2>Spike Activity Timeline</h2>
    <div class="chart-container-full"><canvas id="chart-spikes-full"></canvas></div>
    <div class="raster-strip"><canvas id="raster-canvas"></canvas></div>
  </div>
</div>

<div class="grid">
  <div class="card full-width">
    <h2>Session Timeline</h2>
    <div class="timeline-wrap" id="session-timeline"></div>
  </div>

  <div class="card full-width">
    <h2>Connected Instances</h2>
    <div id="instances">Checking...</div>
  </div>

  <div class="card">
    <h2>Current State</h2>
    <div id="metrics"></div>
  </div>

  <div class="card">
    <h2>Agent Contributions</h2>
    <div id="sources"></div>
    <div class="stimuli-list" id="stimuli-list"></div>
  </div>

  <div class="card">
    <h2>Memory Retrieval</h2>
    <div class="search-row ac-wrap">
      <input id="search-input" type="text" placeholder="Query SNN memory..." autocomplete="off" oninput="onSearchInput(this.value)">
      <button onclick="doSearch()">Search</button>
      <div class="ac-list" id="ac-list"></div>
    </div>
    <div class="search-results" id="search-results">
      <div class="search-empty">/ to focus, Enter to search, ? for help</div>
    </div>
  </div>

  <div class="card">
    <h2>Inject Stimulus</h2>
    <div class="inject-row">
      <textarea id="inject-text" placeholder="Enter stimulus text..." rows="2"></textarea>
      <button onclick="injectStimulus()">Inject</button>
    </div>
    <div class="inject-status" id="inject-status"></div>
  </div>

  <div class="card">
    <h2>Retrieval History</h2>
    <div class="rh-list" id="retrieval-history"></div>
  </div>

  <div class="card">
    <h2>Weight Distribution</h2>
    <div class="chart-container"><canvas id="chart-wdist"></canvas></div>
  </div>

  <div class="card">
    <h2>Weight Matrix Heatmap</h2>
    <div class="heatmap-wrap">
      <canvas id="heatmap-canvas" width="400" height="400"></canvas>
      <div class="hm-info" id="hm-info"></div>
    </div>
    <div class="heatmap-legend"><span id="hm-min">0</span><span>low &mdash; high</span><span id="hm-max">1</span></div>
  </div>

  <div class="card">
    <h2>Network Graph (Hub Connections)</h2>
    <div class="graph-wrap"><canvas id="graph-canvas"></canvas></div>
  </div>

  <div class="card">
    <h2>Network Hub Neurons</h2>
    <div id="weight-stats-meta" style="color:#505070;font-size:11px;margin-bottom:6px"></div>
    <div id="hub-bars"></div>
  </div>

  <div class="card">
    <h2>Identity Depth (v_deep) Over Time</h2>
    <div class="chart-container"><canvas id="chart-depth"></canvas></div>
  </div>

  <div class="card">
    <h2>Network Membrane (v_mean)</h2>
    <div class="chart-container"><canvas id="chart-vmean"></canvas></div>
  </div>

  <div class="card">
    <h2>Traces &amp; Stimuli Processed</h2>
    <div class="chart-container"><canvas id="chart-traces"></canvas></div>
  </div>

  <div class="card">
    <h2>Memory Storage</h2>
    <div id="mem-current" style="font-size:13px;color:var(--text-dim);margin-bottom:8px"></div>
    <div id="mem-breakdown"></div>
    <div class="chart-container" style="margin-top:8px"><canvas id="chart-memory"></canvas></div>
  </div>

  <div class="card">
    <h2>ArcaneNeuron Identity Gauge (v_deep x 35)</h2>
    <div id="depth-gauge" class="depth-list"></div>
  </div>

  <div class="card full-width">
    <h2>Reasoning Traces (<span id="trace-count">—</span>)</h2>
    <div class="traces-list" id="traces-list"></div>
  </div>
</div>

<script>
const POLL_MS = 5000;
const MAX_CHART_POINTS = 120;
const tip = document.getElementById('tooltip');

function showTip(x, y, text) {
  tip.textContent = text;
  tip.style.left = (x + 12) + 'px';
  tip.style.top = (y - 10) + 'px';
  tip.style.display = 'block';
}
function hideTip() { tip.style.display = 'none'; }

// -- Charts --
if (typeof Chart === 'undefined') {
  document.body.innerHTML = '<div style="color:#ff4040;padding:40px;font-size:14px">Chart.js failed to load (offline?). Reload with internet access.</div>';
}

const lineOpts = (label, color) => ({
  type:'line',
  data:{labels:[], datasets:[{label, data:[], borderColor:color, backgroundColor:color+'18', fill:true, tension:0.3, pointRadius:0, borderWidth:1.5}]},
  options:{responsive:true, maintainAspectRatio:false, animation:false,
    scales:{x:{display:false}, y:{ticks:{color:'#606080',font:{size:10}}, grid:{color:'#1a1a2a'}}},
    plugins:{legend:{display:false}}}
});

const spikeFullChart = new Chart(document.getElementById('chart-spikes-full'), {
  type:'bar',
  data:{labels:[], datasets:[{label:'spikes', data:[], backgroundColor:'#ff404060', borderColor:'#ff6060', borderWidth:1}]},
  options:{responsive:true, maintainAspectRatio:false, animation:false,
    scales:{x:{display:false}, y:{ticks:{color:'#606080',font:{size:10}}, grid:{color:'#1a1a2a'}}},
    plugins:{legend:{display:false}}}
});
const depthChart = new Chart(document.getElementById('chart-depth'), lineOpts('v_deep','#8080ff'));
const vmeanChart = new Chart(document.getElementById('chart-vmean'), lineOpts('v_mean','#40c0c0'));
const traceChart = new Chart(document.getElementById('chart-traces'), {
  type:'line',
  data:{labels:[], datasets:[
    {label:'traces', data:[], borderColor:'#c0a040', pointRadius:0, borderWidth:1.5, tension:0.3, backgroundColor:'#c0a04018', fill:true},
    {label:'stimuli',data:[], borderColor:'#40c040', pointRadius:0, borderWidth:1.5, tension:0.3}
  ]},
  options:{responsive:true, maintainAspectRatio:false, animation:false,
    scales:{x:{display:false}, y:{ticks:{color:'#606080',font:{size:10}}, grid:{color:'#1a1a2a'}}},
    plugins:{legend:{labels:{color:'#8888a0',font:{size:10}}}}}
});

const memChart = new Chart(document.getElementById('chart-memory'), {
  type:'line',
  data:{labels:[], datasets:[
    {label:'total MB', data:[], borderColor:'#e06040', pointRadius:0, borderWidth:1.5, tension:0.3, backgroundColor:'#e0604018', fill:true},
  ]},
  options:{responsive:true, maintainAspectRatio:false, animation:false,
    scales:{x:{display:false}, y:{ticks:{color:'#606080',font:{size:10}, callback:v=>v.toFixed(1)+'MB'}, grid:{color:'#1a1a2a'}}},
    plugins:{legend:{display:false}}}
});

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(2) + ' MB';
}

const MEM_COLORS = {weights:'#e06040',traces:'#c0a040',stimuli:'#40c040',sessions:'#6080e0',heartbeats:'#808090'};

function renderMemory(d) {
  const mb = d.memory_bytes;
  const bd = d.memory_breakdown;
  if (!mb && mb !== 0) { document.getElementById('mem-current').textContent = 'No data yet'; return; }
  document.getElementById('mem-current').innerHTML = '<b>' + fmtBytes(mb) + '</b> total on disk';
  if (bd) {
    const sorted = Object.entries(bd).sort((a,b) => b[1]-a[1]);
    const total = Math.max(mb, 1);
    document.getElementById('mem-breakdown').innerHTML =
      '<div style="display:flex;height:14px;border-radius:3px;overflow:hidden;margin-bottom:4px">' +
      sorted.map(([k,v]) => {
        const pct = (v/total*100);
        if (pct < 0.5) return '';
        return `<div style="width:${pct}%;background:${MEM_COLORS[k]||'#606080'}" title="${k}: ${fmtBytes(v)}"></div>`;
      }).join('') + '</div>' +
      '<div style="font-size:11px;color:var(--text-dim)">' +
      sorted.map(([k,v]) => `<span style="color:${MEM_COLORS[k]||'#606080'}">\u25CF</span> ${k}: ${fmtBytes(v)}`).join(' &nbsp; ') +
      '</div>';
  }
}

function updateCharts(history) {
  const h = history.slice(-MAX_CHART_POINTS);
  const labels = h.map(r => r.cycle);
  const updateLine = (chart, fn) => {
    chart.data.labels = labels;
    chart.data.datasets[0].data = h.map(fn);
    chart.update();
  };
  spikeFullChart.data.labels = labels;
  spikeFullChart.data.datasets[0].data = h.map(r => r.spikes);
  spikeFullChart.update();
  updateLine(depthChart, r => r.mean_v_deep || 0);
  updateLine(vmeanChart, r => r.v_mean);
  traceChart.data.labels = labels;
  traceChart.data.datasets[0].data = h.map(r => r.traces);
  traceChart.data.datasets[1].data = h.map(r => r.stimuli || 0);
  traceChart.update();
  // Memory growth
  const memData = h.filter(r => r.memory_bytes);
  if (memData.length) {
    memChart.data.labels = memData.map(r => r.cycle);
    memChart.data.datasets[0].data = memData.map(r => r.memory_bytes / 1048576);
    memChart.update();
  }
}

// -- Metrics --
function renderMetrics(d) {
  const depths = d.identity_depth || [];
  const minD = depths.length ? Math.min(...depths).toExponential(2) : '—';
  const maxD = depths.length ? Math.max(...depths).toExponential(2) : '—';
  document.getElementById('metrics').innerHTML = [
    ['Cycle',           d.cycle],
    ['Simulated Time',  (d.t?.toFixed(0) ?? '—') + 's'],
    ['Neurons (LIF)',   d.n_neurons],
    ['ArcaneNeurons',  d.arcane_neurons],
    ['Spikes/burst',   d.spikes_this_burst?.toLocaleString()],
    ['v_mean',         (d.v_mean?.toFixed(1) ?? '—') + ' mV'],
    ['Weights (mean)', d.weights_mean?.toFixed(4)],
    ['v_deep range',   minD + ' .. ' + maxD],
    ['mean v_work',    d.mean_v_work?.toExponential(3)],
    ['Traces',         d.traces_processed],
    ['Stimuli',        d.stimuli_processed || 0],
  ].map(([l,v]) => `<div class="metric"><span class="label">${l}</span><span class="value">${v}</span></div>`).join('');
}

// -- Sources --
function sourceClass(s) {
  if (s.startsWith('git-')) return 'git';
  return s;
}
function renderSources(stimuli) {
  const srcMap = {};
  stimuli.forEach(s => { (srcMap[s.source] ??= []).push(s); });
  const tags = Object.keys(srcMap).map(s =>
    `<span class="source-tag ${sourceClass(s)}">${s} (${srcMap[s].length})</span>`
  ).join('');
  document.getElementById('sources').innerHTML =
    '<div class="sources">' + (tags || '<span style="color:#505070">No stimuli yet</span>') + '</div>';
  document.getElementById('stimuli-list').innerHTML = stimuli.slice(-20).reverse().map(s => {
    const t = s.timestamp ? new Date(s.timestamp*1000).toLocaleTimeString() : '';
    return `<div class="stim-entry"><span class="src source-tag ${sourceClass(s.source)}">${s.source}</span> <span style="color:#505070">${t}</span>${s.project ? ' ('+s.project+')' : ''}<div class="txt">${s.text}</div></div>`;
  }).join('');
}

function renderInstances(agents) {
  if (!agents.length) {
    document.getElementById('instances').innerHTML = '<span style="color:#505070">No heartbeats detected</span>';
    return;
  }
  const typeColors = {'snn-daemon':'#8080ff','claude':'#c080ff','codex':'#40c040','gemini':'#c0a040'};
  document.getElementById('instances').innerHTML = agents.map(a => {
    const age = a.age_seconds;
    const ageStr = age < 60 ? age+'s ago' : age < 3600 ? Math.floor(age/60)+'m ago' : Math.floor(age/3600)+'h ago';
    const color = typeColors[a.agent.split('-')[0]] || typeColors[a.agent] || '#808080';
    return `<div class="instance">
      <div class="dot ${a.alive?'alive':'dead'}"></div>
      <span class="inst-name" style="color:${color}">${a.agent}</span>
      <span class="inst-project">${a.project||''}</span>
      <span class="inst-detail">${a.detail||''}</span>
      <span class="inst-age">${a.status} | ${ageStr}</span>
    </div>`;
  }).join('');
}

function renderDepthGauge(depths) {
  if (!depths || !depths.length) {
    document.getElementById('depth-gauge').innerHTML = '<span style="color:#505070">No identity_depth data</span>';
    return;
  }
  const maxAbs = Math.max(...depths.map(Math.abs), 1e-12);
  const hue = v => v < 0 ? 0 : (v / maxAbs) * 180 + 160;
  document.getElementById('depth-gauge').innerHTML = depths.map((v, i) => {
    const pct = Math.abs(v) / maxAbs * 100;
    const h = hue(v);
    return `<div class="depth-row">
      <span class="depth-id">${i}</span>
      <div class="depth-bar-wrap"><div class="depth-bar" style="width:${pct.toFixed(1)}%;background:hsl(${h},70%,50%)"></div></div>
      <span class="depth-val">${v.toExponential(2)}</span>
    </div>`;
  }).join('');
}

// -- Hub neurons + weight distribution --
let _hubLoaded = false;
let _wdistChart = null;
async function loadWeightStats() {
  if (_hubLoaded) return;
  try {
    const d = await fetch('/api/weight_stats').then(r => r.json());
    if (d.error) {
      document.getElementById('hub-bars').innerHTML = `<span style="color:#505070">${d.error}</span>`;
      return;
    }
    document.getElementById('weight-stats-meta').textContent =
      `n=${d.n_neurons}  \u03bc=${d.weight_mean.toFixed(4)}  \u03c3=${d.weight_std.toFixed(4)}  sparse=${(d.sparsity*100).toFixed(1)}%`;
    const maxS = Math.max(...d.hub_strengths, 1e-12);
    document.getElementById('hub-bars').innerHTML = d.hub_neurons.map((nid, i) => {
      const pct = (d.hub_strengths[i] / maxS * 100).toFixed(1);
      return `<div class="hub-row">
        <span class="hub-id">#${nid}</span>
        <div class="hub-bar-wrap"><div class="hub-bar" style="width:${pct}%"></div></div>
        <span class="hub-val">${d.hub_strengths[i].toFixed(1)}</span>
      </div>`;
    }).join('');

    // Weight distribution histogram
    if (d.hist_counts && d.hist_edges) {
      const labels = d.hist_edges.slice(0, -1).map((e, i) =>
        e.toFixed(3) + '-' + d.hist_edges[i+1].toFixed(3)
      );
      if (_wdistChart) _wdistChart.destroy();
      _wdistChart = new Chart(document.getElementById('chart-wdist'), {
        type:'bar',
        data:{labels, datasets:[{data:d.hist_counts, backgroundColor:'#6060c060', borderColor:'#8080ff', borderWidth:1}]},
        options:{responsive:true, maintainAspectRatio:false, animation:false,
          scales:{x:{ticks:{color:'#404060',font:{size:8},maxRotation:45}, grid:{display:false}},
                  y:{ticks:{color:'#606080',font:{size:10}}, grid:{color:'#1a1a2a'}}},
          plugins:{legend:{display:false}}}
      });
    }
    _hubLoaded = true;
  } catch(e) {
    document.getElementById('hub-bars').innerHTML = `<span style="color:#505070">Error: ${e.message}</span>`;
  }
}

// -- Traces with summaries --
let _tracesLoaded = false;
let _traceSummaries = {};
async function loadTraces() {
  if (_tracesLoaded) return;
  try {
    const [traces, sums] = await Promise.all([
      fetch('/api/traces').then(r => r.json()),
      fetch('/api/trace_summaries').then(r => r.json()),
    ]);
    if (Array.isArray(sums)) sums.forEach(s => { _traceSummaries[s.name] = s.summary; });
    document.getElementById('trace-count').textContent = traces.length;
    const fmt = sz => sz < 1024 ? sz+'B' : (sz/1024).toFixed(1)+'K';
    document.getElementById('traces-list').innerHTML = traces.map(t => {
      const summary = _traceSummaries[t.name] || '';
      return `<div class="trace-row">
        <span class="tname" title="${t.name}">${t.name.replace(/\.md$/,'')}</span>
        <span class="tsummary" title="${summary}">${summary}</span>
        <span class="tsize">${fmt(t.size)}</span>
        <span class="tdate">${t.mtime_str}</span>
      </div>`;
    }).join('') || '<span style="color:#505070">No traces found</span>';
    _tracesLoaded = true;
  } catch(e) {
    document.getElementById('traces-list').innerHTML = `<span style="color:#505070">Error: ${e.message}</span>`;
  }
}

// -- Heatmap with hover --
let _heatmapLoaded = false;
let _hmData = null;
async function loadHeatmap() {
  if (_heatmapLoaded) return;
  try {
    const d = await fetch('/api/weight_heatmap').then(r => r.json());
    if (d.error) return;
    _hmData = d;
    const canvas = document.getElementById('heatmap-canvas');
    const ctx = canvas.getContext('2d');
    const sz = d.size;
    canvas.width = sz;
    canvas.height = sz;
    canvas.style.width = Math.min(350, sz * 7) + 'px';
    canvas.style.height = Math.min(350, sz * 7) + 'px';
    const range = d.max - d.min || 1;
    const img = ctx.createImageData(sz, sz);
    for (let r = 0; r < sz; r++) {
      for (let c = 0; c < sz; c++) {
        const val = (d.grid[r][c] - d.min) / range;
        const idx = (r * sz + c) * 4;
        if (val < 0.25) {
          img.data[idx]=0; img.data[idx+1]=Math.floor(val*4*180); img.data[idx+2]=Math.floor(80+val*4*175);
        } else if (val < 0.5) {
          const t=(val-0.25)*4; img.data[idx]=0; img.data[idx+1]=Math.floor(180+t*75); img.data[idx+2]=Math.floor(255-t*255);
        } else if (val < 0.75) {
          const t=(val-0.5)*4; img.data[idx]=Math.floor(t*255); img.data[idx+1]=255; img.data[idx+2]=0;
        } else {
          const t=(val-0.75)*4; img.data[idx]=255; img.data[idx+1]=Math.floor(255-t*255); img.data[idx+2]=0;
        }
        img.data[idx+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    document.getElementById('hm-min').textContent = d.min.toFixed(4);
    document.getElementById('hm-max').textContent = d.max.toFixed(4);
    _heatmapLoaded = true;
  } catch(e) {}
}

// Heatmap hover
document.getElementById('heatmap-canvas').addEventListener('mousemove', function(e) {
  if (!_hmData) return;
  const rect = this.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / rect.width * _hmData.size);
  const row = Math.floor((e.clientY - rect.top) / rect.height * _hmData.size);
  if (row >= 0 && row < _hmData.size && col >= 0 && col < _hmData.size) {
    const val = _hmData.grid[row][col];
    const blockSz = Math.floor(2000 / _hmData.size);
    document.getElementById('hm-info').textContent =
      `block [${row*blockSz}-${(row+1)*blockSz}, ${col*blockSz}-${(col+1)*blockSz}] = ${val.toFixed(5)}`;
  }
});
document.getElementById('heatmap-canvas').addEventListener('mouseleave', function() {
  document.getElementById('hm-info').textContent = '';
});

// -- Network graph with drag + hover --
let _graphLoaded = false;
function loadGraph() {
  if (_graphLoaded) return;
  fetch('/api/hub_graph').then(r => r.json()).then(d => {
    if (d.error || !d.nodes) return;
    const canvas = document.getElementById('graph-canvas');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = Math.floor(rect.width - 2);
    canvas.height = 300;
    const W = canvas.width, H = canvas.height;

    const maxStr = Math.max(...d.nodes.map(n => n.strength), 0.01);
    const nodes = d.nodes.map(n => ({
      ...n,
      x: 40 + Math.random() * (W - 80),
      y: 40 + Math.random() * (H - 80),
      vx: 0, vy: 0,
      r: 3 + (n.strength / maxStr) * 7,
      conns: 0,
    }));
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });
    d.edges.forEach(e => {
      if (nodeMap[e.from]) nodeMap[e.from].conns++;
      if (nodeMap[e.to]) nodeMap[e.to].conns++;
    });

    let hoverNode = null, dragNode = null;

    function step() {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx*dx + dy*dy) + 1;
          const f = 800 / (dist * dist);
          nodes[i].vx -= f*dx/dist; nodes[i].vy -= f*dy/dist;
          nodes[j].vx += f*dx/dist; nodes[j].vy += f*dy/dist;
        }
      }
      for (const e of d.edges) {
        const a = nodeMap[e.from], b = nodeMap[e.to];
        if (!a || !b) continue;
        const dx = b.x-a.x, dy = b.y-a.y;
        const dist = Math.sqrt(dx*dx + dy*dy) + 1;
        const f = dist * 0.005 * (e.weight + 0.1);
        a.vx += f*dx/dist; a.vy += f*dy/dist;
        b.vx -= f*dx/dist; b.vy -= f*dy/dist;
      }
      for (const n of nodes) {
        if (n === dragNode) continue;
        n.vx += (W/2 - n.x) * 0.001;
        n.vy += (H/2 - n.y) * 0.001;
        n.vx *= 0.85; n.vy *= 0.85;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(n.r+5, Math.min(W-n.r-5, n.x));
        n.y = Math.max(n.r+5, Math.min(H-n.r-5, n.y));
      }
    }

    function draw() {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      const hoverEdges = new Set();
      if (hoverNode) {
        d.edges.forEach((e,i) => { if (e.from === hoverNode.id || e.to === hoverNode.id) hoverEdges.add(i); });
      }
      d.edges.forEach((e, i) => {
        const a = nodeMap[e.from], b = nodeMap[e.to];
        if (!a || !b) return;
        const highlight = hoverEdges.has(i);
        ctx.strokeStyle = highlight
          ? `rgba(160,160,255,0.9)`
          : `rgba(100,100,200,${Math.min(e.weight*0.6+0.1, 0.5)})`;
        ctx.lineWidth = highlight ? Math.max(1.5, e.weight*3) : Math.max(0.5, e.weight*1.5);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      });
      for (const n of nodes) {
        const isHover = n === hoverNode;
        const isNeighbor = hoverNode && d.edges.some(e =>
          (e.from === hoverNode.id && e.to === n.id) || (e.to === hoverNode.id && e.from === n.id));
        const dim = hoverNode && !isHover && !isNeighbor;
        const hue = 220 + (n.strength / maxStr) * 60;
        const light = dim ? 25 : (40 + (n.strength / maxStr) * 25);
        const r = isHover ? n.r + 3 : n.r;
        ctx.fillStyle = `hsl(${hue}, ${dim?30:60}%, ${light}%)`;
        ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI*2); ctx.fill();
        if (isHover) {
          ctx.strokeStyle = '#a0a0ff'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(n.x, n.y, r+2, 0, Math.PI*2); ctx.stroke();
        }
        if (!dim || isHover) {
          ctx.fillStyle = dim ? '#404060' : '#8888cc';
          ctx.font = (isHover ? '10' : '8') + 'px monospace';
          ctx.fillText('#'+n.id, n.x+r+2, n.y+3);
        }
      }
    }

    function findNode(mx, my) {
      for (const n of nodes) {
        if (Math.hypot(n.x-mx, n.y-my) < n.r + 4) return n;
      }
      return null;
    }

    canvas.addEventListener('mousedown', e => {
      const rect = canvas.getBoundingClientRect();
      const n = findNode(e.clientX-rect.left, e.clientY-rect.top);
      if (n) { dragNode = n; n.vx = 0; n.vy = 0; canvas.classList.add('dragging'); }
    });
    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX-rect.left, my = e.clientY-rect.top;
      if (dragNode) {
        dragNode.x = Math.max(dragNode.r+5, Math.min(W-dragNode.r-5, mx));
        dragNode.y = Math.max(dragNode.r+5, Math.min(H-dragNode.r-5, my));
        draw();
      } else {
        const n = findNode(mx, my);
        if (n !== hoverNode) {
          hoverNode = n;
          draw();
          if (n) showTip(e.clientX, e.clientY, `#${n.id} | strength: ${n.strength.toFixed(3)} | connections: ${n.conns}`);
          else hideTip();
        }
      }
    });
    canvas.addEventListener('mouseup', () => { dragNode = null; canvas.classList.remove('dragging'); });
    canvas.addEventListener('mouseleave', () => { hoverNode = null; dragNode = null; canvas.classList.remove('dragging'); hideTip(); draw(); });

    for (let i = 0; i < 120; i++) step();
    draw();
    let frame = 0;
    function animate() {
      if (frame < 200 && !dragNode) { step(); draw(); frame++; requestAnimationFrame(animate); }
    }
    requestAnimationFrame(animate);
    _graphLoaded = true;
  }).catch(() => {});
}

// -- Session timeline (hover, no vertical labels) --
let _timelineLoaded = false;
async function loadTimeline() {
  if (_timelineLoaded) return;
  try {
    const entries = await fetch('/api/session_timeline').then(r => r.json());
    if (!entries.length) {
      document.getElementById('session-timeline').innerHTML = '<span style="color:#505070">No session states found</span>';
      return;
    }
    const projColors = {};
    const palette = ['#8080ff','#40c040','#c0a040','#c080ff','#ff6060','#40c0c0','#c06060','#60c060','#a060c0','#80c080','#c08080'];
    let ci = 0;
    entries.forEach(e => { if (!projColors[e.project]) projColors[e.project] = palette[ci++ % palette.length]; });
    const maxSize = Math.max(...entries.map(e => e.size), 1);
    document.getElementById('session-timeline').innerHTML =
      '<div class="timeline-bar">' + entries.map(e => {
        const h = Math.max(14, (e.size / maxSize) * 55);
        const color = projColors[e.project];
        return `<div class="tl-item" title="${e.project}\n${e.date}\n${e.file}\n${(e.size/1024).toFixed(1)}K"
          onmouseover="showTip(event.clientX,event.clientY,'${e.project} | ${e.date} | ${(e.size/1024).toFixed(1)}K')"
          onmouseout="hideTip()"
          onclick="showSessionDetail('${e.file}')" style="cursor:pointer">
          <div class="tl-block" style="height:${h}px;background:${color}"></div>
          <div class="tl-date">${e.date.slice(5)}</div>
        </div>`;
      }).join('') + '</div>' +
      '<div style="margin-top:6px;display:flex;gap:10px;flex-wrap:wrap;font-size:10px">' +
      Object.entries(projColors).map(([p,c]) =>
        `<span style="color:${c}">\u25cf ${p}</span>`
      ).join('') + '</div>';
    _timelineLoaded = true;
  } catch(e) {}
}

// -- Retrieval history --
let _rhLoaded = false;
async function loadRetrievalHistory() {
  if (_rhLoaded) return;
  try {
    const entries = await fetch('/api/retrieval_history').then(r => r.json());
    if (!entries.length || entries[0]?.error) {
      document.getElementById('retrieval-history').innerHTML = '<span style="color:#505070">No retrieval history</span>';
      return;
    }
    document.getElementById('retrieval-history').innerHTML = entries.slice().reverse().map(e => {
      const ts = e.timestamp ? new Date(e.timestamp*1000).toLocaleTimeString() : '';
      const top = e.results?.[0];
      const topStr = top ? `${top.trace.replace(/\.md$/,'').substring(0,30)} (${top.score.toFixed(3)})` : '';
      return `<div class="rh-entry" onclick="replayQuery('${e.query.replace(/'/g,"\\'")}')">
        <div class="rh-query">${e.query}</div>
        <div class="rh-meta">${ts} | top: ${topStr}</div>
      </div>`;
    }).join('');
    _rhLoaded = true;
  } catch(e) {}
}

function replayQuery(q) {
  document.getElementById('search-input').value = q;
  doSearch();
}

// -- Search (with trace summaries) --
let _searching = false;
async function doSearch() {
  const q = document.getElementById('search-input').value.trim();
  if (!q || _searching) return;
  _searching = true;
  document.getElementById('search-results').innerHTML = '<span style="color:#505070">Searching...</span>';
  try {
    const results = await fetch('/api/retrieve?q=' + encodeURIComponent(q)).then(r => r.json());
    if (!results.length || results[0]?.error) {
      document.getElementById('search-results').innerHTML =
        `<div class="search-empty">${results[0]?.error || 'No results'}</div>`;
      return;
    }
    const maxScore = Math.max(...results.map(r => r.score), 0.01);
    document.getElementById('search-results').innerHTML = results.map(r => {
      const pct = (r.score / maxScore * 100).toFixed(0);
      const name = r.trace.replace(/\.md$/, '');
      const summary = _traceSummaries[r.trace] || '';
      const expanded = r.expanded ? ' (expanded)' : '';
      return `<div class="search-result">
        <div class="trace-name">${name}</div>
        ${summary ? `<div class="sr-summary">${summary}</div>` : ''}
        <div class="scores">score ${r.score.toFixed(3)} | kw ${r.kw_score?.toFixed(3)} | snn ${r.snn_score?.toFixed(3)}${expanded}</div>
        <div class="score-bar" style="width:${pct}%"></div>
      </div>`;
    }).join('');
    _rhLoaded = false;
    loadRetrievalHistory();
  } catch(e) {
    document.getElementById('search-results').innerHTML = `<div class="search-empty">Error: ${e.message}</div>`;
  } finally {
    _searching = false;
  }
}

document.getElementById('search-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

// -- Stimulus injection (refreshes sources after inject) --
async function injectStimulus() {
  const text = document.getElementById('inject-text').value.trim();
  if (!text) return;
  const statusEl = document.getElementById('inject-status');
  statusEl.textContent = 'Injecting...';
  statusEl.style.color = '#505070';
  try {
    const resp = await fetch('/api/inject_stimulus', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, source: 'dashboard'}),
    });
    const data = await resp.json();
    if (data.ok) {
      statusEl.textContent = 'Injected: ' + data.path.split('/').pop().split('\\').pop();
      statusEl.style.color = '#40c040';
      document.getElementById('inject-text').value = '';
      setTimeout(poll, 1000);
    } else {
      statusEl.textContent = 'Error: ' + (data.error || 'unknown');
      statusEl.style.color = '#ff4040';
    }
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.style.color = '#ff4040';
  }
}

document.getElementById('inject-text').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); injectStimulus(); }
});

// -- Keyboard shortcuts --
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') {
    if (e.key === 'Escape') document.activeElement.blur();
    return;
  }
  if (e.key === '/') { e.preventDefault(); document.getElementById('search-input').focus(); }
  if (e.key === '?') { document.getElementById('shortcuts-modal').classList.add('visible'); }
  if (e.key === 'Escape') { document.getElementById('shortcuts-modal').classList.remove('visible'); }
});

// -- Poll --
async function poll() {
  try {
    const [cur, hist, stim, hb] = await Promise.all([
      fetch('/api/current').then(r => r.json()),
      fetch('/api/history').then(r => r.json()),
      fetch('/api/stimuli').then(r => r.json()),
      fetch('/api/heartbeats').then(r => r.json()),
    ]);
    const age = Date.now()/1000 - (cur.timestamp || 0);
    const alive = age < 120;
    document.getElementById('pulse').className = 'pulse-dot ' + (alive ? 'active' : 'stale');
    document.getElementById('daemon-status').className = alive ? 'alive' : 'dead';
    document.getElementById('daemon-status').textContent = alive
      ? 'ALIVE (cycle ' + cur.cycle + ')'
      : 'STALE (' + Math.floor(age) + 's ago)';
    document.getElementById('last-update').textContent =
      new Date((cur.timestamp||0)*1000).toLocaleTimeString();
    renderMetrics(cur);
    renderMemory(cur);
    updateCharts(hist);
    renderSources(stim);
    renderInstances(hb);
    renderDepthGauge(cur.identity_depth || []);
  } catch(e) {
    document.getElementById('pulse').className = 'pulse-dot stale';
    document.getElementById('daemon-status').className = 'dead';
    document.getElementById('daemon-status').textContent = 'ERROR: ' + e.message;
  }
}

poll();
loadWeightStats();
loadTraces();
loadHeatmap();
loadGraph();
loadTimeline();
loadRetrievalHistory();

setInterval(poll, POLL_MS);
setInterval(() => { _hubLoaded = false; loadWeightStats(); }, 60000);
setInterval(() => { _heatmapLoaded = false; loadHeatmap(); }, 120000);
setInterval(() => { _tracesLoaded = false; loadTraces(); }, 30000);

// -- Raster heat strip --
const rasterCanvas = document.getElementById('raster-canvas');
const rasterCtx = rasterCanvas.getContext('2d');
let rasterHistory = [];

function updateRaster(history) {
  const data = history.slice(-120);
  rasterHistory = data.filter(h => h.raster);
  if (!rasterHistory.length) return;
  const cols = rasterHistory.length;
  const rows = rasterHistory[0].raster.length;
  rasterCanvas.width = cols;
  rasterCanvas.height = rows;
  const img = rasterCtx.createImageData(cols, rows);
  let maxVal = 1;
  rasterHistory.forEach(h => h.raster.forEach(v => { if (v > maxVal) maxVal = v; }));
  for (let c = 0; c < cols; c++) {
    const r = rasterHistory[c].raster;
    for (let row = 0; row < rows; row++) {
      const val = (r[row] || 0) / maxVal;
      const idx = ((rows - 1 - row) * cols + c) * 4;
      if (val < 0.3) {
        img.data[idx] = 0; img.data[idx+1] = Math.floor(val*3.3*100); img.data[idx+2] = Math.floor(40 + val*3.3*100);
      } else if (val < 0.7) {
        const t = (val-0.3)*2.5;
        img.data[idx] = Math.floor(t*255); img.data[idx+1] = Math.floor(100+t*155); img.data[idx+2] = Math.floor(140-t*140);
      } else {
        const t = (val-0.7)*3.3;
        img.data[idx] = 255; img.data[idx+1] = Math.floor(255-t*200); img.data[idx+2] = 0;
      }
      img.data[idx+3] = 255;
    }
  }
  rasterCtx.putImageData(img, 0, 0);
}

// -- Heatmap click-to-zoom --
let _hmZoom = null;
document.getElementById('heatmap-canvas').addEventListener('click', function(e) {
  if (!_hmData) return;
  const rect = this.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / rect.width * 2);
  const row = Math.floor((e.clientY - rect.top) / rect.height * 2);
  if (_hmZoom && _hmZoom.row === row && _hmZoom.col === col) {
    _hmZoom = null;
  } else {
    _hmZoom = {row, col};
  }
  renderHeatmapZoom();
});

function renderHeatmapZoom() {
  if (!_hmData) return;
  const canvas = document.getElementById('heatmap-canvas');
  const ctx = canvas.getContext('2d');
  const sz = _hmData.size;
  const range = _hmData.max - _hmData.min || 1;

  let startR = 0, startC = 0, endR = sz, endC = sz;
  if (_hmZoom) {
    const half = Math.floor(sz / 2);
    startR = _hmZoom.row * half; endR = startR + half;
    startC = _hmZoom.col * half; endC = startC + half;
  }
  const viewSz = endR - startR;
  canvas.width = viewSz;
  canvas.height = viewSz;

  const img = ctx.createImageData(viewSz, viewSz);
  for (let r = 0; r < viewSz; r++) {
    for (let c = 0; c < viewSz; c++) {
      const val = (_hmData.grid[startR+r][startC+c] - _hmData.min) / range;
      const idx = (r * viewSz + c) * 4;
      if (val < 0.25) { img.data[idx]=0; img.data[idx+1]=Math.floor(val*4*180); img.data[idx+2]=Math.floor(80+val*4*175); }
      else if (val < 0.5) { const t=(val-0.25)*4; img.data[idx]=0; img.data[idx+1]=Math.floor(180+t*75); img.data[idx+2]=Math.floor(255-t*255); }
      else if (val < 0.75) { const t=(val-0.5)*4; img.data[idx]=Math.floor(t*255); img.data[idx+1]=255; img.data[idx+2]=0; }
      else { const t=(val-0.75)*4; img.data[idx]=255; img.data[idx+1]=Math.floor(255-t*255); img.data[idx+2]=0; }
      img.data[idx+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  document.getElementById('hm-info').textContent = _hmZoom
    ? `Zoomed: quadrant [${_hmZoom.row},${_hmZoom.col}] (click to zoom out)`
    : 'Click quadrant to zoom';
}

// -- Graph responsive resize --
window.addEventListener('resize', () => {
  if (_graphLoaded) { _graphLoaded = false; loadGraph(); }
});

// -- Autocomplete --
let _acTimer = null;
function onSearchInput(val) {
  clearTimeout(_acTimer);
  const list = document.getElementById('ac-list');
  if (val.length < 2) { list.classList.remove('visible'); return; }
  _acTimer = setTimeout(async () => {
    try {
      const suggestions = await fetch('/api/suggest?q=' + encodeURIComponent(val)).then(r => r.json());
      if (!suggestions.length) { list.classList.remove('visible'); return; }
      list.innerHTML = suggestions.map(s =>
        `<div class="ac-item" onmousedown="document.getElementById('search-input').value='${s.replace(/'/g,"\\'")}';document.getElementById('ac-list').classList.remove('visible');doSearch()">${s}</div>`
      ).join('');
      list.classList.add('visible');
    } catch(e) { list.classList.remove('visible'); }
  }, 200);
}

document.getElementById('search-input').addEventListener('blur', () => {
  setTimeout(() => document.getElementById('ac-list').classList.remove('visible'), 150);
});

// -- Session detail --
async function showSessionDetail(filename) {
  const modal = document.getElementById('detail-modal');
  document.getElementById('detail-title').textContent = filename;
  document.getElementById('detail-body').textContent = 'Loading...';
  modal.classList.add('visible');
  try {
    const d = await fetch('/api/session_detail?file=' + encodeURIComponent(filename)).then(r => r.json());
    document.getElementById('detail-body').textContent = d.content || d.error || 'Empty';
  } catch(e) {
    document.getElementById('detail-body').textContent = 'Error: ' + e.message;
  }
}

// -- Theme toggle --
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'light' ? '' : 'light');
  localStorage.setItem('arcsap-theme', current === 'light' ? 'dark' : 'light');
}
// Restore saved theme
if (localStorage.getItem('arcsap-theme') === 'light') {
  document.documentElement.setAttribute('data-theme', 'light');
}

// -- Service worker --
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

// -- Hook raster into poll --
const _origUpdateCharts = updateCharts;
updateCharts = function(history) { _origUpdateCharts(history); updateRaster(history); };
</script>
</body>
</html>"""


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
                print(f"Monitor already running (PID {pid}). Kill it first or delete {MONITOR_LOCK}")
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
    so the monitor is not a child of any Claude bash shell.
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
