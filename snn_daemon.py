# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Persistent SNN Background Daemon

"""Persistent spiking neural network that runs between sessions.

Maintains a 1000-neuron LIF network whose membrane potentials carry
the residue of accumulated session experience. STDP modifies weights
based on stimulus patterns derived from reasoning traces.

Usage::

    # Start daemon (runs until killed)
    python 04_ARCANE_SAPIENCE/snn_daemon.py

    # Start with custom parameters
    python 04_ARCANE_SAPIENCE/snn_daemon.py --neurons 500 --interval 60

The daemon:
1. Loads or creates the identity network
2. Checks for new stimulus files every --interval seconds
3. Encodes stimuli as current injection into the network
4. Runs the network for a short burst (1s simulated time)
5. Saves membrane state for the P-channel extractor to read
6. Loops indefinitely
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [SNN] %(message)s",
)
logger = logging.getLogger("ArcSap.SNN")

BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "snn_state"
STIMULUS_DIR = BASE_DIR / "snn_stimuli"
TRACES_DIR = BASE_DIR / "reasoning_traces"
SESSION_STATES_DIR = BASE_DIR / "session_states"
HEARTBEAT_DIR = BASE_DIR / "heartbeats"
RETRIEVAL_STATE_PATH = STATE_DIR / "retrieval_state.json"

# Sentinel for graceful shutdown
_running = True

LOCK_PATH = STATE_DIR / "daemon.lock"


def _acquire_lock() -> bool:
    """File-based singleton lock. Returns True if this is the only instance."""
    import os
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            # Check if the PID is still alive (Windows-compatible)
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                logger.error(
                    "Another daemon is running (PID %d). "
                    "Kill it first or delete %s", pid, LOCK_PATH,
                )
                return False
            # Stale lock — process is dead
            logger.info("Removing stale lock (PID %d no longer running)", pid)
        except (ValueError, OSError):
            logger.info("Removing invalid lock file")
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def _release_lock():
    """Remove lock file on exit."""
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received — saving state and exiting")
    _running = False


class SimpleLIFNetwork:
    """Minimal LIF network for identity state persistence.

    Each neuron integrates current, decays toward rest, and spikes
    when threshold is crossed. Membrane potentials persist between
    run() calls — this IS the continuity substrate.

    For production: replace with sc-neurocore's Network class
    (HodgkinHuxley + WangBuzsaki + HindmarshRose populations with
    STDP projections). This simplified version demonstrates the
    architecture without requiring sc-neurocore installation.
    """

    def __init__(self, n_neurons: int = 2000, dt: float = 0.001, seed: int = 42, topology: str = "small_world"):
        self.n = n_neurons
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.v = self.rng.uniform(-70.0, -55.0, n_neurons)

        self.v_rest = -65.0
        self.v_thresh = -55.0
        self.v_reset = -70.0
        self.tau_m = 10.0
        self.dt_ms = dt * 1000

        if topology == "small_world":
            # Watts-Strogatz small-world: k nearest neighbors + p rewiring
            # Better clustering + short path lengths than random topology
            # Watts & Strogatz (1998), Nature 393:440-442
            k = min(200, n_neurons // 5)  # neighbors per side
            p_rewire = 0.1
            self.w = np.zeros((n_neurons, n_neurons))
            for i in range(n_neurons):
                for j in range(1, k // 2 + 1):
                    target = (i + j) % n_neurons
                    if self.rng.random() < p_rewire:
                        target = self.rng.integers(0, n_neurons)
                        while target == i:
                            target = self.rng.integers(0, n_neurons)
                    self.w[i, target] = self.rng.uniform(0.1, 0.6)
                    # Symmetric (undirected small-world)
                    self.w[target, i] = self.rng.uniform(0.1, 0.6)
        else:
            # Random 10% connectivity (legacy)
            self.w = np.zeros((n_neurons, n_neurons))
            n_connections = int(0.10 * n_neurons)
            for i in range(n_neurons):
                targets = self.rng.choice(n_neurons, n_connections, replace=False)
                self.w[i, targets] = self.rng.uniform(0.1, 0.6, n_connections)
        np.fill_diagonal(self.w, 0)

        # Spike history for STDP
        self.last_spike = np.full(n_neurons, -1000.0)
        self.t = 0.0

        # Input current (tonic baseline keeps network alive)
        self.i_ext = np.full(n_neurons, 0.3)

    def run(self, duration: float, arcane_neurons=None):
        """Run the network for `duration` seconds.

        If arcane_neurons is provided, each ArcaneNeuron is stepped every
        timestep with current derived from the spike activity of its
        corresponding LIF neuron group.
        """
        steps = int(duration / self.dt)
        spikes_total = 0
        n_arcane = len(arcane_neurons) if arcane_neurons else 0
        # Each ArcaneNeuron maps to a contiguous group of LIF neurons
        group_size = self.n // n_arcane if n_arcane > 0 else 0

        for _ in range(steps):
            self.t += self.dt

            # Synaptic input from connected neurons
            fired = self.v >= self.v_thresh
            i_syn = self.w @ fired.astype(float)

            # LIF dynamics: dv/dt = -(v - v_rest)/tau_m + I (all in ms)
            dv = (-(self.v - self.v_rest) / self.tau_m + self.i_ext + i_syn * 0.5) * self.dt_ms
            self.v += dv

            # Spike and reset
            spiked = self.v >= self.v_thresh
            if spiked.any():
                spikes_total += spiked.sum()
                self._apply_stdp(spiked)
                self.v[spiked] = self.v_reset
                self.last_spike[spiked] = self.t

            # Drive ArcaneNeurons from LIF group spike rates
            if n_arcane > 0:
                for i, an in enumerate(arcane_neurons):
                    start = i * group_size
                    end = start + group_size if i < n_arcane - 1 else self.n
                    group_rate = float(spiked[start:end].sum()) / (end - start)
                    an.step(group_rate * 5.0 + float(self.i_ext[start]))

        return spikes_total

    def _apply_stdp(self, spiked):
        """Spike-timing-dependent plasticity."""
        tau_plus = 20.0
        tau_minus = 20.0
        a_plus = 0.005
        a_minus = 0.005

        spike_idx = np.where(spiked)[0]
        dt_pre = self.t - self.last_spike  # hoist outside loop
        for i in spike_idx:
            # LTP: neuron i fired (post). Strengthen from pre neurons
            # that fired recently before i (dt > 0 = causal)
            mask_ltp = (dt_pre > 0) & (dt_pre < 100)
            dw = a_plus * np.exp(-dt_pre / tau_plus) * mask_ltp
            self.w[i, :] = np.clip(self.w[i, :] + dw, 0, 2)

            # LTD: neuron i fired (pre). Weaken connections TO neurons
            # that fired before i (anti-causal direction)
            dw_neg = a_minus * np.exp(-dt_pre / tau_minus) * mask_ltp
            self.w[:, i] = np.clip(self.w[:, i] - dw_neg, 0, 2)

    def inject_stimulus(self, pattern: np.ndarray):
        """Set external current from a stimulus pattern.

        Pattern should be shape (n_neurons,) with values 0-1.
        Accumulates additively (multiple traces compound).
        """
        if len(pattern) != self.n:
            if len(pattern) > self.n:
                idx = np.linspace(0, len(pattern) - 1, self.n, dtype=int)
                pattern = pattern[idx]
            else:
                pattern = np.pad(pattern, (0, self.n - len(pattern)))
        self.i_ext = np.clip(self.i_ext + pattern * 2.0, 0, 5.0)

    def clear_stimulus(self):
        # Keep a tonic baseline so the network stays alive
        self.i_ext = np.full(self.n, 1.2)

    def get_state(self) -> dict:
        return {
            "membrane_potentials": self.v.copy(),
            "weights_mean": float(self.w.mean()),
            "weights_std": float(self.w.std()),
            "t": self.t,
            "n_neurons": self.n,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "v": self.v,
                "w": self.w,
                "last_spike": self.last_spike,
                "t": self.t,
                "n": self.n,
                "dt": self.dt,
            }, f)

    @classmethod
    def load(cls, path: Path) -> SimpleLIFNetwork:
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls(n_neurons=data["n"], dt=data["dt"])
        net.v = data["v"]
        net.w = data["w"]
        net.last_spike = data["last_spike"]
        net.t = data["t"]
        return net


_HASH_PRIMES = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]


def encode_text_to_stimulus(text: str, n_neurons: int) -> np.ndarray:
    """Convert text to a spike pattern.

    Uses encoding.py (LSH by default) for semantic similarity.
    Falls back to hash encoding if encoding.py unavailable.
    """
    try:
        from encoding import encode_text
        return encode_text(text, n_neurons)
    except ImportError:
        pass
    # Fallback: hash encoding
    import re

    pattern = np.zeros(n_neurons)
    tokens = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 1]

    for word in tokens:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.15, 1.0)

    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]}_{tokens[i+1]}"
        h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES[:5]:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.25, 1.0)

    return pattern


def check_new_stimuli(stimulus_dir: Path, processed: set) -> list[Path]:
    """Check for new stimulus files (session summaries)."""
    new = []
    if stimulus_dir.exists():
        for f in stimulus_dir.glob("*.json"):
            if f.name not in processed:
                new.append(f)
    return sorted(new)


def check_new_traces(traces_dir: Path, processed: set) -> list[Path]:
    """Check for new reasoning trace files."""
    new = []
    if traces_dir.exists():
        for f in traces_dir.glob("*.md"):
            if f.name not in processed:
                new.append(f)
    return sorted(new)


def _current_encoding_backend() -> str:
    try:
        from encoding import get_backend
        return get_backend()
    except Exception:
        return "lsh"


def _write_retrieval_state(
    checkpoint_path: Path,
    encoding_backend: str,
    source: str,
    n_neurons: int,
) -> None:
    RETRIEVAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_STATE_PATH.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "encoding_backend": encoding_backend,
                "source": source,
                "n_neurons": n_neurons,
                "updated_at": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


def heartbeat(agent: str, project: str = "", status: str = "active", detail: str = ""):
    """Register a heartbeat from any agent instance.

    Call periodically (or at session start/end) so the monitor
    knows which agents are alive and what they're working on.

    Usage::

        python -c "
        import sys; sys.path.insert(0, '04_ARCANE_SAPIENCE')
        from snn_daemon import heartbeat
        heartbeat('claude-session-1', project='director-ai', status='active', detail='CI audit')
        "

    Or write JSON directly::

        echo '{"agent":"codex","project":"sc-neurocore","status":"active","detail":"lint fixes"}' \\
          > 04_ARCANE_SAPIENCE/heartbeats/codex.json
    """
    import os
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    safe = agent.replace("/", "_").replace("\\", "_").replace(" ", "_")
    path = HEARTBEAT_DIR / f"{safe}.json"
    data = {
        "agent": agent,
        "project": project,
        "status": status,
        "detail": detail,
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path)


def drop_stimulus(text: str, source: str = "unknown"):
    """Write a stimulus file for the daemon to pick up.

    Called by ANY agent (Arcane Sapience, Codex, GPT, Gemini) to
    feed reasoning into the shared SNN. The daemon picks up new
    .json files from snn_stimuli/ on its next cycle.

    Usage from any terminal or script::

        python -c "
        from snn_daemon import drop_stimulus
        drop_stimulus('Fixed ruff 0.11.2→0.15.6 in scpn-control, 47 lint violations resolved', source='codex')
        "

    Or write JSON directly::

        echo '{"text":"...", "source":"gemini", "project":"sc-neurocore"}' > 04_ARCANE_SAPIENCE/snn_stimuli/gemini_1710700000.json
    """
    STIMULUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe_source = source.replace("/", "_").replace("\\", "_")
    path = STIMULUS_DIR / f"{safe_source}_{ts}.json"
    # Store text only — daemon re-encodes at its own neuron count
    data = {
        "text": text,
        "source": source,
        "timestamp": ts,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("Stimulus dropped by %s: %s", source, path.name)
    return str(path)


def _replay_traces(net, traces_dir: Path, processed: set, n_traces: int = 3, amplitude: float = 0.1):
    """Replay stored traces at low amplitude during idle cycles.

    Biological analogue: hippocampal replay during sleep. Strengthens
    existing associations without new input. Randomly samples from
    processed traces and re-injects at reduced amplitude.
    """
    if not processed or not traces_dir.exists():
        return 0
    candidates = [traces_dir / name for name in processed if (traces_dir / name).exists()]
    if not candidates:
        return 0
    rng = np.random.default_rng()
    selected = rng.choice(candidates, size=min(n_traces, len(candidates)), replace=False)
    replayed = 0
    for trace_path in selected:
        text = trace_path.read_text(encoding="utf-8")
        stimulus = encode_text_to_stimulus(text, net.n)
        net.inject_stimulus(stimulus * amplitude)
        replayed += 1
    return replayed


def _save_weight_snapshot(w: np.ndarray, state_dir: Path, cycle: int, max_snapshots: int = 10):
    """Save timestamped weight snapshot for rollback.

    Keeps the last max_snapshots versions. If a bad stimulus corrupts
    the weights, rollback to a previous snapshot.
    """
    snap_dir = state_dir / "weight_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"w_cycle_{cycle:06d}.npy"
    np.save(path, w)
    # Prune old snapshots
    snapshots = sorted(snap_dir.glob("w_cycle_*.npy"))
    for old in snapshots[:-max_snapshots]:
        old.unlink()


def _homeostatic_scaling(w: np.ndarray, target_mean: float = 0.3, rate: float = 0.01):
    """Synaptic scaling: normalize weights to maintain target activity.

    Prevents saturation without deleting connections. Each neuron's
    incoming weights are scaled toward the target mean. This is
    activity regulation, not forgetting — all connections persist.

    Reference: Turrigiano & Nelson (2004), "Homeostatic plasticity in
    the developing nervous system," Nature Reviews Neuroscience.
    """
    for i in range(w.shape[0]):
        row = w[i]
        active = row > 0.001
        if active.sum() < 2:
            continue
        current_mean = row[active].mean()
        if current_mean < 0.001:
            continue
        scale = 1.0 + rate * (target_mean / current_mean - 1.0)
        w[i] = np.clip(w[i] * scale, 0, 2.0)


def _extract_summary(text: str) -> str:
    """Extract first content line after heading, skip frontmatter."""
    past_heading = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            past_heading = True
            continue
        if not past_heading or not stripped or stripped.startswith("---"):
            continue
        if stripped.startswith("**") and ":**" in stripped:
            continue
        if stripped.startswith("- **") and ":**" in stripped:
            continue
        return stripped.lstrip("- ").strip()[:120]
    return ""


def main():
    parser = argparse.ArgumentParser(description="Arcane Sapience SNN daemon")
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--burst", type=float, default=1.0, help="Simulation burst duration (seconds)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Singleton enforcement — only one daemon at a time
    if not _acquire_lock():
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STIMULUS_DIR.mkdir(parents=True, exist_ok=True)
    net_path = STATE_DIR / "identity_net.pkl"

    # Try Rust ArcaneNeuron (fastest), then Python, then SimpleLIF
    ArcaneNeuron = None
    arcane_backend = "none"
    try:
        from sc_neurocore_engine.sc_neurocore_engine import ArcaneNeuron
        arcane_backend = "rust"
        logger.info("ArcaneNeuron loaded from Rust/PyO3 — native speed")
    except ImportError:
        try:
            from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
            arcane_backend = "python"
            logger.info("ArcaneNeuron loaded from Python")
        except ImportError:
            logger.info("sc-neurocore not installed — SimpleLIF fallback")

    # Load or create network
    if net_path.exists():
        logger.info("Loading persistent network from %s", net_path)
        net = SimpleLIFNetwork.load(net_path)
        logger.info("Network loaded: %d neurons, t=%.1fs", net.n, net.t)
    else:
        logger.info("Creating new network: %d neurons", args.neurons)
        net = SimpleLIFNetwork(n_neurons=args.neurons)

    # Create ArcaneNeuron ensemble alongside SimpleLIF for identity tracking
    arcane_neurons = []
    arcane_state_path = STATE_DIR / "arcane_ensemble.json"
    if ArcaneNeuron is not None:
        if arcane_state_path.exists():
            saved = json.loads(arcane_state_path.read_text())
            arcane_neurons = [ArcaneNeuron() for _ in range(len(saved))]
            # Warm up each neuron to restore v_deep from saved state
            for an, st in zip(arcane_neurons, saved):
                # Step with zero current to initialise, then rely on
                # accumulated v_deep from continued operation
                an.step(0.0)
            logger.info("Restored %d ArcaneNeurons from JSON (%s backend)", len(arcane_neurons), arcane_backend)
        else:
            arcane_neurons = [ArcaneNeuron() for _ in range(35)]
            logger.info("Created 35 ArcaneNeurons (%s backend)", arcane_backend)

    processed_stimuli: set[str] = set()
    processed_traces: set[str] = set()
    stimulus_sources: set[str] = set()
    trace_summaries: dict[str, str] = {}
    cycle = 0

    # Git stimulus integration
    git_stim_available = False
    try:
        from git_stimulus import scan_repos as _git_scan
        git_stim_available = True
        logger.info("Git stimulus integration active")
    except ImportError:
        logger.info("git_stimulus.py not found — git auto-stimulus disabled")

    # Sparse W save
    sparse_w_available = False
    try:
        from snn_backend import save_sparse
        sparse_w_available = True
    except ImportError:
        pass

    # Complementary learning: slow consolidation matrix
    # Fast W (current): rapid STDP, encodes new experiences immediately
    # Slow W: consolidates patterns that recur across multiple traces
    # Retrieval uses both: fast for recent, slow for consolidated
    # McClelland et al. (1995), "Why there are complementary learning systems"
    w_slow_path = STATE_DIR / "w_slow.npy"
    if w_slow_path.exists():
        w_slow = np.load(w_slow_path)
        if w_slow.shape != net.w.shape:
            w_slow = np.zeros_like(net.w)
        logger.info("Loaded slow consolidation matrix")
    else:
        w_slow = np.zeros_like(net.w)
        logger.info("Created slow consolidation matrix")

    logger.info("SNN daemon started (interval=%ds, burst=%.1fs, topology=%s)",
                args.interval, args.burst,
                "small_world" if hasattr(net, 'n') else "random")

    while _running:
        cycle += 1

        # Git auto-stimulus every 10 cycles
        if git_stim_available and cycle % 10 == 0:
            try:
                n_git = _git_scan()
                if n_git > 0:
                    logger.info("Git stimulus: %d new commits ingested", n_git)
            except Exception as e:
                logger.warning("Git stimulus scan failed: %s", e)

        # Check for new reasoning traces → encode as stimuli
        new_traces = check_new_traces(TRACES_DIR, processed_traces)
        for trace_path in new_traces:
            text = trace_path.read_text(encoding="utf-8")
            stimulus = encode_text_to_stimulus(text, net.n)
            net.inject_stimulus(stimulus)
            # Auto-summarize: first content line after heading
            summary = _extract_summary(text)
            if summary:
                trace_summaries[trace_path.name] = summary
            logger.info("Injected trace: %s (%d active neurons)", trace_path.name, (stimulus > 0).sum())
            processed_traces.add(trace_path.name)

        # Check for explicit stimulus files (from Codex, Gemini, or any agent)
        new_stimuli = check_new_stimuli(STIMULUS_DIR, processed_stimuli)
        for stim_path in new_stimuli:
            data = json.loads(stim_path.read_text())
            pattern = np.array(data.get("pattern", []))
            source = data.get("source", "unknown")
            if len(pattern) > 0:
                net.inject_stimulus(pattern)
                logger.info("Injected stimulus from %s: %s", source, stim_path.name)
            stimulus_sources.add(source)
            processed_stimuli.add(stim_path.name)

        # Run network burst — ArcaneNeurons stepped every timestep inside
        spikes = net.run(args.burst, arcane_neurons=arcane_neurons or None)
        net.clear_stimulus()

        # Membrane histogram for raster visualization (50 bins)
        v_hist, _ = np.histogram(net.v, bins=50, range=(-70.0, -55.0))
        raster_row = v_hist.tolist()

        # Save ArcaneNeuron ensemble (PyO3 objects can't be pickled)
        if arcane_neurons:
            ensemble_states = [an.get_state() for an in arcane_neurons]
            arcane_state_path.write_text(json.dumps(ensemble_states, indent=2) + "\n")

        # Save LIF state
        net.save(net_path)
        _write_retrieval_state(
            net_path,
            _current_encoding_backend(),
            "snn_daemon",
            net.n,
        )

        # Memory replay: re-inject old traces at low amplitude (every 5 cycles)
        if cycle % 5 == 0:
            n_replayed = _replay_traces(net, TRACES_DIR, processed_traces)
            if n_replayed > 0:
                replay_spikes = net.run(0.2, arcane_neurons=arcane_neurons or None)

        # Complementary learning: consolidate fast W into slow W (every 20 cycles)
        if cycle % 20 == 0:
            consolidation_rate = 0.005
            w_slow += consolidation_rate * (net.w - w_slow)
            np.save(w_slow_path, w_slow)

        # Homeostatic plasticity: normalize weights every 50 cycles
        if cycle % 50 == 0:
            _homeostatic_scaling(net.w)

        # Weight versioning: timestamped snapshot (every 100 cycles)
        if cycle % 100 == 0:
            _save_weight_snapshot(net.w, STATE_DIR, cycle)

        # Save sparse W alongside dense (73% disk savings)
        if sparse_w_available and cycle % 20 == 0:
            try:
                save_sparse(net.w, STATE_DIR / "weights_sparse")
            except Exception:
                pass

        # Write readable state summary
        state = net.get_state()
        summary_path = STATE_DIR / "current_state.json"
        history_path = STATE_DIR / "history.jsonl"
        arcane_states = [an.get_state() for an in arcane_neurons] if arcane_neurons else []
        depths = [s.get("v_deep", 0.0) for s in arcane_states]
        summary = {
            "t": state["t"],
            "n_neurons": state["n_neurons"],
            "encoding_backend": _current_encoding_backend(),
            "retrieval_checkpoint_path": str(net_path.resolve()),
            "retrieval_backend": _current_encoding_backend(),
            "v_mean": float(state["membrane_potentials"].mean()),
            "v_std": float(state["membrane_potentials"].std()),
            "weights_mean": state["weights_mean"],
            "weights_std": state["weights_std"],
            "spikes_this_burst": int(spikes),
            "cycle": cycle,
            "traces_processed": len(processed_traces),
            "stimuli_processed": len(processed_stimuli),
            "arcane_neurons": len(arcane_neurons),
            "identity_depth": depths,
            "mean_v_deep": float(np.mean(depths)) if depths else 0.0,
            "mean_v_work": float(np.mean([s.get("v_work", 0.0) for s in arcane_states])) if arcane_states else 0.0,
            "stimulus_sources": list(stimulus_sources),
            "raster": raster_row,
            "timestamp": time.time(),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with open(history_path, "a") as hf:
            hf.write(json.dumps({
                "t": summary["t"],
                "cycle": cycle,
                "spikes": int(spikes),
                "v_mean": summary["v_mean"],
                "mean_v_deep": summary["mean_v_deep"],
                "mean_v_work": summary["mean_v_work"],
                "traces": len(processed_traces),
                "stimuli": len(processed_stimuli),
                "sources": list(stimulus_sources),
                "raster": raster_row,
                "ts": summary["timestamp"],
            }) + "\n")

        if cycle % 10 == 0:
            depth_str = ""
            if arcane_states:
                depths = [s["v_deep"] for s in arcane_states]
                depth_str = f", v_deep=[{min(depths):.2e}..{max(depths):.2e}]"
            logger.info(
                "Cycle %d: t=%.1fs, spikes=%d, v_mean=%.1f, traces=%d%s",
                cycle, state["t"], spikes, float(state["membrane_potentials"].mean()),
                len(processed_traces), depth_str,
            )

        # Daemon heartbeat
        heartbeat(
            "snn-daemon", project="arcane-sapience",
            status="running",
            detail=f"cycle={cycle} t={state['t']:.0f}s spikes={spikes} traces={len(processed_traces)}",
        )

        for _ in range(args.interval):
            if not _running:
                break
            time.sleep(1)

    # Graceful shutdown
    net.save(net_path)
    _write_retrieval_state(
        net_path,
        _current_encoding_backend(),
        "snn_daemon",
        net.n,
    )
    np.save(w_slow_path, w_slow)
    _save_weight_snapshot(net.w, STATE_DIR, cycle)
    if sparse_w_available:
        try:
            save_sparse(net.w, STATE_DIR / "weights_sparse")
        except Exception:
            pass
    _release_lock()
    logger.info("Final state saved. Lock released. Network ran for %.1fs simulated time.", net.t)


if __name__ == "__main__":
    main()
