#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""GPU SNN Daemon — 20K neurons on CUDA with dashboard lifecycle.

Requires Python 3.12 (.venv312) with PyTorch + sentence-transformers.

Usage::

    cd 04_ARCANE_SAPIENCE
    .venv312/Scripts/python.exe gpu_daemon.py              # foreground
    .venv312/Scripts/python.exe gpu_daemon.py --detach      # background + dashboard
    .venv312/Scripts/python.exe gpu_daemon.py --neurons 5000

The daemon owns the dashboard: starts it on launch, doesn't care which
agent (Claude/Codex/Gemini) is connected. Agents are memory contributors,
not dashboard runners.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
TRACES = BASE / "reasoning_traces"
STATE_DIR = BASE / "snn_state"
STIMULI_DIR = BASE / "snn_stimuli"
HEARTBEAT_DIR = BASE / "heartbeats"
RETRIEVAL_STATE_PATH = STATE_DIR / "retrieval_state.json"
LIVE_REQUESTS_DIR = STATE_DIR / "live_retrieval_requests"
LIVE_RESPONSES_DIR = STATE_DIR / "live_retrieval_responses"
LOCK_FILE = STATE_DIR / "daemon.lock"
DAEMON_STDOUT_LOG = STATE_DIR / "gpu_daemon_stdout.log"
DAEMON_STDERR_LOG = STATE_DIR / "gpu_daemon_stderr.log"
DAEMON_RUNTIME_LOG = STATE_DIR / "gpu_daemon_runtime.log"
LIVE_REQUEST_TIMEOUT_S = 30.0
LIVE_RESPONSE_TTL_S = 300.0
LIVE_POLL_INTERVAL_S = 0.5
LIVE_SNN_CANDIDATES = 8
LIVE_SNN_FACTOR = 2
LIVE_FEATURE_STEPS = 8
LIVE_V_REST = -65.0
LIVE_V_THRESH = -55.0
LIVE_V_RESET = -70.0
LIVE_TAU_M = 10.0

MEMORY_DIRS = {
    "weights": STATE_DIR,
    "traces": TRACES,
    "stimuli": STIMULI_DIR,
    "sessions": BASE / "session_states",
    "heartbeats": HEARTBEAT_DIR,
}

_running = True


def _measure_memory() -> dict:
    """Measure disk footprint of all memory components."""
    breakdown = {}
    total = 0
    for label, d in MEMORY_DIRS.items():
        if not d.exists():
            breakdown[label] = 0
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        breakdown[label] = size
        total += size
    return {"total_bytes": total, "breakdown": breakdown}


def _read_retrieval_state() -> dict | None:
    if not RETRIEVAL_STATE_PATH.exists():
        return None
    try:
        return json.loads(RETRIEVAL_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _heartbeat_stale(max_age_s: float = 120.0) -> bool:
    heartbeat_path = HEARTBEAT_DIR / "snn-daemon.json"
    if not heartbeat_path.exists():
        return False
    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    timestamp = float(payload.get("timestamp", 0.0) or 0.0)
    if timestamp <= 0:
        return False
    return (time.time() - timestamp) > max_age_s


def _stop(sig, frame):
    global _running
    _running = False


def _acquire_lock():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stale_heartbeat = _heartbeat_stale()
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if not stale_heartbeat:
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, old_pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    print(f"GPU daemon already running (PID {old_pid})")
                    return False
        except (ValueError, OSError, AttributeError):
            pass
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock():
    LOCK_FILE.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_runtime_log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    with open(DAEMON_RUNTIME_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _publish_status(
    status: str,
    detail: str,
    *,
    pid: int | None = None,
    phase: str | None = None,
    n_neurons: int | None = None,
    live_retrieval_available: bool = False,
) -> None:
    timestamp = time.time()
    heartbeat = {
        "agent": "snn-daemon",
        "project": "arcane-sapience",
        "status": status,
        "pid": int(pid or 0),
        "detail": detail,
        "timestamp": timestamp,
    }
    _write_json_atomic(HEARTBEAT_DIR / "snn-daemon.json", heartbeat)

    state = {
        "status": status,
        "phase": phase,
        "detail": detail,
        "timestamp": timestamp,
        "backend": "gpu",
        "n_neurons": n_neurons,
        "live_retrieval_available": live_retrieval_available,
    }
    _write_json_atomic(STATE_DIR / "current_state.json", state)


def _start_dashboard():
    """Start dashboard as detached process. Daemon owns it."""
    monitor = BASE / "monitor.py"
    if not monitor.exists():
        return
    try:
        from monitor import ensure_running
        result = ensure_running()
        print(f"Dashboard: {result['status']} on port {result['port']}")
    except Exception as e:
        print(f"Dashboard start failed: {e}")


def _spawn_detached():
    """Re-exec self as a fully detached process."""
    python = sys.executable
    script = str(Path(__file__).resolve())
    args = [a for a in sys.argv[1:] if a != "--detach"]
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    with open(DAEMON_STDOUT_LOG, "a", encoding="utf-8") as stdout_log, open(
        DAEMON_STDERR_LOG, "a", encoding="utf-8"
    ) as stderr_log:
        if sys.platform == "win32":
            flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
            )
            subprocess.Popen(
                [python, script, "--serve"] + args,
                creationflags=flags,
                close_fds=True,
                cwd=str(BASE),
                stdin=subprocess.DEVNULL,
                stdout=stdout_log,
                stderr=stderr_log,
            )
        else:
            subprocess.Popen(
                [python, script, "--serve"] + args,
                start_new_session=True,
                close_fds=True,
                cwd=str(BASE),
                stdin=subprocess.DEVNULL,
                stdout=stdout_log,
                stderr=stderr_log,
            )
    print("GPU daemon spawned in background")


class _LiveRetrievalService:
    """Filesystem-backed query service over the live in-memory network."""

    def __init__(
        self,
        net,
        encode_text_fn,
        initial_trace_texts: dict[str, str] | None = None,
        initial_trace_stimuli: dict[str, np.ndarray] | None = None,
    ):
        from retrieve import (
            _EMBED_RERANK_FACTOR,
            _EMBED_RERANK_MIN,
            _WEIGHT_EMB,
            _WEIGHT_KW,
            _WEIGHT_NAME,
            _WEIGHT_SNN,
            _build_idf,
            _cosine_sim,
            _embedding_similarity,
            _filename_bonus,
            _tfidf_score,
            _tier_boost,
            _trace_tier,
        )

        self.net = net
        self.encode_text = encode_text_fn
        self.embed_rerank_factor = _EMBED_RERANK_FACTOR
        self.embed_rerank_min = _EMBED_RERANK_MIN
        self.weight_emb = _WEIGHT_EMB
        self.weight_kw = _WEIGHT_KW
        self.weight_name = _WEIGHT_NAME
        self.weight_snn = _WEIGHT_SNN
        self.build_idf = _build_idf
        self.cosine_sim = _cosine_sim
        self.embedding_similarity = _embedding_similarity
        self.filename_bonus = _filename_bonus
        self.tfidf_score = _tfidf_score
        self.tier_boost = _tier_boost
        self.trace_tier = _trace_tier
        self.request_dir = LIVE_REQUESTS_DIR
        self.response_dir = LIVE_RESPONSES_DIR
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self._cycle = -1
        self.trace_texts: dict[str, str] = {}
        self.trace_stimuli: dict[str, np.ndarray] = {}
        self.trace_names_lower: dict[str, str] = {}
        self.idf: dict[str, float] = {}
        self.trace_spikes: dict[tuple[int, str], np.ndarray] = {}
        self.query_spikes: dict[tuple[int, str], np.ndarray] = {}
        self._warm_trace_names: list[str] = []
        self._warm_index = 0
        if initial_trace_texts and initial_trace_stimuli:
            self.trace_texts = dict(initial_trace_texts)
            self.trace_stimuli = dict(initial_trace_stimuli)
            self.trace_names_lower = {
                name: name.lower().replace("-", " ").replace("_", " ")
                for name in self.trace_texts
            }
            self.idf = self.build_idf(self.trace_texts)
            self._warm_trace_names = sorted(self.trace_texts)
        else:
            self.sync_traces(force=True)

    def sync_traces(self, force: bool = False) -> None:
        trace_files = sorted(TRACES.glob("*.md"))
        live_names = {f.name for f in trace_files}
        changed = force

        for name in list(self.trace_texts):
            if name not in live_names:
                self.trace_texts.pop(name, None)
                self.trace_stimuli.pop(name, None)
                self.trace_names_lower.pop(name, None)
                changed = True

        for trace_file in trace_files:
            text = trace_file.read_text(encoding="utf-8")
            if force or self.trace_texts.get(trace_file.name) != text:
                self.trace_texts[trace_file.name] = text
                self.trace_stimuli[trace_file.name] = self.encode_text(text, self.net.n)
                self.trace_names_lower[trace_file.name] = trace_file.name.lower().replace("-", " ").replace("_", " ")
                changed = True

        if changed:
            self.idf = self.build_idf(self.trace_texts)
            self.trace_spikes.clear()
            self.query_spikes.clear()
            self._warm_trace_names = sorted(self.trace_texts)
            self._warm_index = 0

    def set_cycle(self, cycle: int) -> None:
        if cycle != self._cycle:
            self._cycle = cycle
            self.trace_spikes.clear()
            self.query_spikes.clear()
            self._warm_trace_names = sorted(self.trace_texts)
            self._warm_index = 0

    def warm_cycle_cache(self, max_new: int = 1) -> int:
        warmed = 0
        while warmed < max_new and self._warm_index < len(self._warm_trace_names):
            trace_name = self._warm_trace_names[self._warm_index]
            self._warm_index += 1
            self._trace_feature(trace_name)
            warmed += 1
        return warmed

    def _probe_feature(self, stimulus: np.ndarray) -> np.ndarray:
        if hasattr(self.net, "torch") and hasattr(self.net, "device"):
            torch = self.net.torch
            device = self.net.device
            v_gpu = self.net.v_gpu.clone()
            i_ext_gpu = 0.3 + torch.from_numpy(stimulus.astype(np.float32)).to(device) * 2.0
            spike_count = torch.zeros(self.net.n, dtype=torch.float32, device=device)

            for _ in range(LIVE_FEATURE_STEPS):
                fired = (v_gpu >= LIVE_V_THRESH).float()
                i_syn = torch.mv(self.net.w, fired)
                dv = (
                    -(v_gpu - LIVE_V_REST) / LIVE_TAU_M
                    + i_ext_gpu
                    + i_syn * 0.5
                ) * self.net.dt_ms
                v_gpu += dv
                spiked = v_gpu >= LIVE_V_THRESH
                spike_count += spiked.float()
                v_gpu[spiked] = LIVE_V_RESET

            return spike_count.cpu().numpy()

        v = self.net.v.astype(np.float32).copy()
        stim = stimulus.astype(np.float32)
        i_ext = 0.3 + stim * 2.0
        spike_count = np.zeros(self.net.n, dtype=np.float32)

        for _ in range(LIVE_FEATURE_STEPS):
            fired = (v >= LIVE_V_THRESH).astype(np.float32)
            i_syn = self.net.w.dot(fired) if hasattr(self.net.w, "dot") else (self.net.w @ fired)
            dv = (
                -(v - LIVE_V_REST) / LIVE_TAU_M
                + i_ext
                + i_syn * 0.5
            ) * self.net.dt_ms
            v += dv
            spiked = v >= LIVE_V_THRESH
            spike_count += spiked.astype(np.float32)
            v[spiked] = LIVE_V_RESET

        return spike_count

    def _query_feature(self, query: str) -> np.ndarray:
        key = (self._cycle, query)
        cached = self.query_spikes.get(key)
        if cached is None:
            cached = self._probe_feature(self.encode_text(query, self.net.n))
            self.query_spikes[key] = cached
        return cached

    def _trace_feature(self, trace_name: str) -> np.ndarray:
        key = (self._cycle, trace_name)
        cached = self.trace_spikes.get(key)
        if cached is None:
            cached = self._probe_feature(self.trace_stimuli[trace_name])
            self.trace_spikes[key] = cached
        return cached

    def _score(self, query: str, top_k: int, include_content: bool) -> list[dict]:
        if not self.trace_texts:
            return []

        lexical_scored = []
        for trace_name, text in self.trace_texts.items():
            kw = self.tfidf_score(query, trace_name, text, self.idf)
            name_bonus = self.filename_bonus(query, self.trace_names_lower[trace_name], self.idf)
            tier = self.trace_tier(TRACES / trace_name)
            tier_boost = self.tier_boost(tier)
            lexical_base = (
                self.weight_kw * kw
                + self.weight_name * name_bonus
            ) * tier_boost
            lexical_scored.append({
                "trace": trace_name,
                "score": round(lexical_base, 4),
                "kw_score": round(kw, 4),
                "snn_score": 0.0,
                "emb_score": 0.0,
                "tier": tier,
                "_kw_score": kw,
                "_name_bonus": name_bonus,
                "_base_score": lexical_base,
                "_tier_boost": tier_boost,
            })

        lexical_scored.sort(key=lambda item: item["score"], reverse=True)
        if self._warm_index >= len(self._warm_trace_names):
            snn_candidates = {entry["trace"] for entry in lexical_scored}
        else:
            candidate_count = min(
                len(lexical_scored),
                max(top_k * LIVE_SNN_FACTOR, LIVE_SNN_CANDIDATES),
            )
            snn_candidates = {entry["trace"] for entry in lexical_scored[:candidate_count]}

        query_spikes = self._query_feature(query)
        scored = []
        for entry in lexical_scored:
            snn = 0.0
            if entry["trace"] in snn_candidates:
                snn = self.cosine_sim(query_spikes, self._trace_feature(entry["trace"]))
            entry["snn_score"] = round(snn, 4)
            entry["_base_score"] = (
                self.weight_kw * entry["_kw_score"]
                + self.weight_snn * snn
                + self.weight_name * entry["_name_bonus"]
            ) * entry["_tier_boost"]
            entry["score"] = round(entry["_base_score"], 4)
            scored.append(entry)

        scored.sort(key=lambda item: item["score"], reverse=True)
        rerank_k = min(len(scored), max(top_k * self.embed_rerank_factor, self.embed_rerank_min))
        for entry in scored[:rerank_k]:
            emb_sim = self.embedding_similarity(query, self.trace_texts[entry["trace"]])
            entry["emb_score"] = round(emb_sim, 4)
            entry["score"] = round(
                entry["_base_score"] + self.weight_emb * emb_sim * entry["_tier_boost"],
                4,
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        results = scored[:top_k]
        for entry in results:
            entry.pop("_kw_score", None)
            entry.pop("_base_score", None)
            entry.pop("_tier_boost", None)
            entry.pop("_name_bonus", None)
            if include_content:
                entry["content"] = self.trace_texts[entry["trace"]]
        return results

    def _cleanup(self) -> None:
        cutoff = time.time() - LIVE_RESPONSE_TTL_S
        for directory in (self.request_dir, self.response_dir):
            for path in directory.glob("*.json"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue
            for tmp_path in directory.glob("*.tmp"):
                try:
                    if tmp_path.stat().st_mtime < cutoff:
                        tmp_path.unlink(missing_ok=True)
                except OSError:
                    continue

    def process_pending(self, cycle: int) -> int:
        self.set_cycle(cycle)
        self.sync_traces()
        self._cleanup()
        processed = 0

        for request_path in sorted(self.request_dir.glob("*.json")):
            if request_path.name.endswith(".tmp"):
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            request_id = str(request.get("id") or request_path.stem)
            response_path = self.response_dir / f"{request_id}.json"
            try:
                results = self._score(
                    query=str(request.get("query", "")),
                    top_k=max(int(request.get("top_k", 3)), 1),
                    include_content=bool(request.get("include_content", False)),
                )
                payload = {
                    "id": request_id,
                    "status": "ok",
                    "source": "gpu_live_service",
                    "cycle": cycle,
                    "n_neurons": self.net.n,
                    "t": self.net.t,
                    "results": results,
                }
            except Exception as exc:
                payload = {
                    "id": request_id,
                    "status": "error",
                    "source": "gpu_live_service",
                    "error": str(exc),
                }

            _write_json_atomic(response_path, payload)
            request_path.unlink(missing_ok=True)
            processed += 1
        return processed


def main(n_neurons: int = 20000):
    import torch
    sys.path.insert(0, str(BASE))
    from snn_backend import create_network
    from encoding import set_backend, encode_text

    set_backend("embedding")
    pid = os.getpid()
    final_status: tuple[str, str, str, int | None] | None = None
    _append_runtime_log(f"startup requested pid={pid} neurons={n_neurons}")

    if not _acquire_lock():
        _append_runtime_log("startup aborted: lock held by active daemon")
        sys.exit(1)

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        _publish_status(
            "starting",
            "lock acquired; initializing GPU network",
            pid=pid,
            phase="network",
            n_neurons=n_neurons,
        )

        net = create_network(n_neurons=n_neurons, backend="gpu")
        _append_runtime_log(f"network ready pid={pid} neurons={net.n}")
        print(f"GPU Daemon: {net.n} neurons, VRAM={torch.cuda.memory_allocated()/1e6:.0f}MB")
        _publish_status(
            "starting",
            "GPU network initialized",
            pid=pid,
            phase="dashboard",
            n_neurons=net.n,
        )

        # Dashboard is ours to manage
        _start_dashboard()
        _publish_status(
            "starting",
            "encoding initial traces",
            pid=pid,
            phase="ingest",
            n_neurons=net.n,
        )

        initial_trace_texts: dict[str, str] = {}
        initial_trace_stimuli: dict[str, np.ndarray] = {}
        trace_files = sorted(TRACES.glob("*.md"))
        total_traces = len(trace_files)
        for idx, f in enumerate(trace_files, start=1):
            text = f.read_text(encoding="utf-8")
            stimulus = encode_text(text, net.n)
            net.inject_stimulus(stimulus)
            initial_trace_texts[f.name] = text
            initial_trace_stimuli[f.name] = stimulus
            if idx == 1 or idx == total_traces or idx % 5 == 0:
                _publish_status(
                    "starting",
                    f"encoded {idx}/{total_traces} traces",
                    pid=pid,
                    phase="ingest",
                    n_neurons=net.n,
                )

        live_retrieval = _LiveRetrievalService(
            net,
            encode_text,
            initial_trace_texts=initial_trace_texts,
            initial_trace_stimuli=initial_trace_stimuli,
        )
        _append_runtime_log(
            f"live retrieval ready traces={len(initial_trace_texts)} pid={pid}"
        )

        processed_stimuli = set()
        processed_traces = set(initial_trace_texts)
        stimulus_sources = set()
        cycle = 0

        print(f"Running. Traces={len(processed_traces)}, PID={pid}")
        _append_runtime_log(f"entered main loop pid={pid}")

        while _running:
            cycle += 1

            # New traces
            for f in sorted(TRACES.glob("*.md")):
                if f.name not in processed_traces:
                    text = f.read_text(encoding="utf-8")
                    stimulus = encode_text(text, net.n)
                    net.inject_stimulus(stimulus)
                    processed_traces.add(f.name)
                    live_retrieval.sync_traces()
                    print(f"  New trace: {f.name}")

            # New stimuli
            if STIMULI_DIR.exists():
                for f in sorted(STIMULI_DIR.glob("*.json")):
                    if f.name not in processed_stimuli:
                        try:
                            d = json.loads(f.read_text())
                            if "text" in d:
                                net.inject_stimulus(encode_text(d["text"], net.n))
                                stimulus_sources.add(d.get("source", "unknown"))
                        except Exception:
                            pass
                        processed_stimuli.add(f.name)

            # Burst
            spikes = net.run(duration=1.0)
            net.clear_stimulus()

            # Replay every 3 cycles
            if cycle % 3 == 0:
                rng = np.random.default_rng()
                files = list(TRACES.glob("*.md"))
                if files:
                    for f in rng.choice(files, size=min(3, len(files)), replace=False):
                        net.inject_stimulus(
                            encode_text(f.read_text(encoding="utf-8"), net.n) * 0.3
                        )

            # Membrane histogram for raster
            v_cpu = net.v if isinstance(net.v, np.ndarray) else net.v
            v_hist, _ = np.histogram(v_cpu, bins=50, range=(-70.0, -55.0))

            # Memory consolidation every 10 cycles
            consolidation_result = None
            if cycle % 10 == 0 and cycle > 0:
                try:
                    from consolidation_engine import consolidate
                    consolidation_result = consolidate()
                    _append_runtime_log(
                        f"consolidation cycle={cycle}: "
                        f"status={consolidation_result.get('status','?')}, "
                        f"memories={consolidation_result.get('memories_written',0)}, "
                        f"entities={consolidation_result.get('entities_found',0)}"
                    )
                except Exception as ce:
                    _append_runtime_log(f"consolidation error: {ce}")

            # Memory footprint
            mem = _measure_memory()
            retrieval_state = _read_retrieval_state() or {}

            # State files
            state = net.get_state()
            summary = {
                "t": state["t"],
                "n_neurons": state["n_neurons"],
                "encoding_backend": "embedding",
                "checkpoint_available": False,
                "live_checkpoint_path": None,
                "live_retrieval_available": True,
                "live_retrieval_transport": "filesystem",
                "live_retrieval_request_dir": str(LIVE_REQUESTS_DIR.resolve()),
                "live_retrieval_response_dir": str(LIVE_RESPONSES_DIR.resolve()),
                "live_retrieval_timeout_s": LIVE_REQUEST_TIMEOUT_S,
                "v_mean": float(state["membrane_potentials"].mean()),
                "v_std": float(state["membrane_potentials"].std()),
                "weights_mean": state["weights_mean"],
                "weights_std": state.get("weights_std", 0),
                "spikes_this_burst": int(spikes),
                "cycle": cycle,
                "traces_processed": len(processed_traces),
                "stimuli_processed": len(processed_stimuli),
                "arcane_neurons": 0,
                "identity_depth": [],
                "mean_v_deep": 0.0,
                "mean_v_work": 0.0,
                "stimulus_sources": list(stimulus_sources),
                "raster": v_hist.tolist(),
                "backend": "gpu",
                "gpu": torch.cuda.get_device_name(0),
                "vram_mb": int(torch.cuda.memory_allocated() / 1e6),
                "memory_bytes": mem["total_bytes"],
                "memory_breakdown": mem["breakdown"],
                "timestamp": time.time(),
            }
            if retrieval_state:
                summary["retrieval_checkpoint_path"] = retrieval_state.get("checkpoint_path")
                summary["retrieval_backend"] = retrieval_state.get("encoding_backend")
                summary["retrieval_neurons"] = retrieval_state.get("n_neurons")
            if consolidation_result:
                summary["last_consolidation"] = consolidation_result
            _write_json_atomic(STATE_DIR / "current_state.json", summary)
            with open(STATE_DIR / "history.jsonl", "a", encoding="utf-8") as hf:
                hf.write(json.dumps({
                    "t": summary["t"], "cycle": cycle, "spikes": int(spikes),
                    "v_mean": summary["v_mean"], "mean_v_deep": 0.0, "mean_v_work": 0.0,
                    "traces": len(processed_traces), "stimuli": len(processed_stimuli),
                    "sources": list(stimulus_sources), "raster": v_hist.tolist(),
                    "memory_bytes": mem["total_bytes"],
                    "memory_breakdown": {k: v for k, v in mem["breakdown"].items()},
                    "ts": summary["timestamp"],
                }) + "\n")

            # Heartbeat
            HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(HEARTBEAT_DIR / "snn-daemon.json", {
                "agent": "snn-daemon", "project": "arcane-sapience",
                "status": "running", "pid": pid,
                "detail": f"GPU cycle={cycle} n={net.n} spikes={spikes} vram={summary['vram_mb']}MB",
                "timestamp": time.time(),
            })

            if cycle % 5 == 0:
                print(
                    f"  Cycle {cycle}: {spikes} spikes, "
                    f"v={state['membrane_potentials'].mean():.1f}mV, "
                    f"VRAM={summary['vram_mb']}MB"
                )

            sleep_deadline = time.time() + 30.0
            while _running and time.time() < sleep_deadline:
                live_retrieval.process_pending(cycle)
                time.sleep(LIVE_POLL_INTERVAL_S)

        final_status = ("stopped", "GPU daemon stopped", "stopped", net.n)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _append_runtime_log(detail)
        _append_runtime_log(traceback.format_exc())
        final_status = ("error", detail, "crashed", n_neurons)
        raise
    finally:
        _release_lock()
        if final_status is not None:
            status, detail, phase, final_neurons = final_status
            _publish_status(
                status,
                detail,
                pid=pid,
                phase=phase,
                n_neurons=final_neurons,
            )
            _append_runtime_log(f"shutdown status={status} pid={pid} detail={detail}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arcane Sapience GPU SNN Daemon")
    parser.add_argument("--neurons", type=int, default=20000)
    parser.add_argument("--detach", action="store_true", help="Spawn detached and exit")
    parser.add_argument("--serve", action="store_true", help="Run in serve mode (used by --detach)")
    args = parser.parse_args()

    if args.detach:
        _spawn_detached()
    else:
        main(n_neurons=args.neurons)
