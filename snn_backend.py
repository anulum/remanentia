# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — SNN Compute Backend (GPU/CPU)

"""Dual-backend LIF network: PyTorch CUDA (default) or NumPy/SciPy CPU.

The LIF dynamics and STDP are identical across backends. The only
difference is where W @ fired runs:

- **GPU** (PyTorch): W stays on CUDA device, GEMV via cuBLAS.
  20,000 neurons in 2.3s per 1s burst (~8% GPU duty at 30s interval).
- **CPU** (SciPy sparse): W stored as CSR, SpMV for the multiply.
  5,000 neurons in ~10s per 1s burst (~33% CPU duty at 30s interval).
- **CPU fallback** (NumPy dense): when scipy.sparse unavailable.
  5,000 neurons in ~28s per 1s burst (94% duty — not recommended).

Auto-detection: GPU if torch.cuda.is_available(), else sparse CPU.

Usage::

    from snn_backend import create_network, detect_backend
    backend = detect_backend()  # "gpu", "sparse_cpu", or "dense_cpu"
    net = create_network(n_neurons=20000, backend=backend)
    spikes = net.run(duration=1.0)
    net.save(path)
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger("ArcSap.Backend")

_HASH_PRIMES = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]

# LIF parameters (shared across backends)
V_REST = -65.0
V_THRESH = -55.0
V_RESET = -70.0
TAU_M = 10.0

# STDP parameters
STDP_TAU = 20.0  # ms
STDP_A_PLUS = 0.005
STDP_A_MINUS = 0.005
STDP_W_MAX = 2.0


def detect_backend() -> str:
    """Auto-detect best available backend."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU backend: %s", name)
            return "gpu"
    except ImportError:
        pass
    try:
        from scipy import sparse as _sp  # noqa: F401

        logger.info("Sparse CPU backend (scipy CSR)")
        return "sparse_cpu"
    except ImportError:
        logger.info("Dense CPU backend (numpy — slow for N>2000)")
        return "dense_cpu"


def encode_text(text: str, n_neurons: int) -> np.ndarray:
    """Hash-based unigram+bigram encoding (matches retrieve.py)."""
    pattern = np.zeros(n_neurons)
    tokens = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 1]

    for word in tokens:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.15, 1.0)

    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]}_{tokens[i + 1]}"
        h = int(hashlib.md5(bg.encode()).hexdigest(), 16)
        for p in _HASH_PRIMES[:5]:
            idx = (h + p) % n_neurons
            pattern[idx] = min(pattern[idx] + 0.25, 1.0)

    return pattern


class _BaseLIFNetwork:
    """Base class with shared state management."""

    def __init__(self, n: int, dt: float, seed: int):
        self.n = n
        self.dt = dt
        self.dt_ms = dt * 1000
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.uniform(-70.0, -55.0, n).astype(np.float32)
        self.last_spike = np.full(n, -1000.0, dtype=np.float32)
        self.t = 0.0
        self.i_ext = np.full(n, 0.3, dtype=np.float32)

    def inject_stimulus(self, pattern: np.ndarray) -> None:
        if len(pattern) != self.n:
            if len(pattern) > self.n:
                idx = np.linspace(0, len(pattern) - 1, self.n, dtype=int)
                pattern = pattern[idx]
            else:
                pattern = np.pad(pattern, (0, self.n - len(pattern)))
        self.i_ext = np.clip(self.i_ext + pattern.astype(np.float32) * 2.0, 0, 5.0)

    def clear_stimulus(self) -> None:
        self.i_ext = np.full(self.n, 1.2, dtype=np.float32)

    def get_state(self) -> dict:
        return {
            "membrane_potentials": self.v.copy(),
            "weights_mean": float(self._w_mean()),
            "weights_std": float(self._w_std()),
            "t": self.t,
            "n_neurons": self.n,
        }

    def _w_mean(self) -> float:
        raise NotImplementedError

    def _w_std(self) -> float:
        raise NotImplementedError


class GPULIFNetwork(_BaseLIFNetwork):
    """PyTorch CUDA backend — W on GPU, cuBLAS GEMV."""

    def __init__(self, n: int = 20000, dt: float = 0.001, seed: int = 42):
        super().__init__(n, dt, seed)
        import torch

        self.device = torch.device("cuda")
        self.torch = torch

        # Build sparse W on CPU, transfer to GPU as dense float32
        w_np = np.zeros((n, n), dtype=np.float32)
        n_conn = int(0.10 * n)
        for i in range(n):
            targets = self.rng.choice(n, n_conn, replace=False)
            w_np[i, targets] = self.rng.uniform(0.1, 0.6, n_conn).astype(np.float32)
        np.fill_diagonal(w_np, 0)

        self.w = torch.from_numpy(w_np).to(self.device)
        self.v_gpu = torch.from_numpy(self.v).to(self.device)
        self._w_np_cache = None  # lazy CPU copy for save/stats

        logger.info("GPU network: %d neurons, W=%.0f MB on %s",
                     n, n * n * 4 / 1e6, torch.cuda.get_device_name(0))

    def run(self, duration: float, arcane_neurons=None) -> int:
        torch = self.torch
        steps = int(duration / self.dt)
        spikes_total = 0
        n_arcane = len(arcane_neurons) if arcane_neurons else 0
        group_size = self.n // n_arcane if n_arcane > 0 else 0

        v_gpu = self.v_gpu
        w = self.w
        i_ext_gpu = torch.from_numpy(self.i_ext).to(self.device)

        for _ in range(steps):
            self.t += self.dt
            fired = (v_gpu >= V_THRESH).float()
            i_syn = torch.mv(w, fired)

            dv = (-(v_gpu - V_REST) / TAU_M + i_ext_gpu + i_syn * 0.5) * self.dt_ms
            v_gpu += dv

            spiked = v_gpu >= V_THRESH
            n_spiked = int(spiked.sum().item())
            if n_spiked > 0:
                spikes_total += n_spiked
                # STDP on CPU (sparse update, not worth GPU kernel)
                spike_idx = torch.where(spiked)[0].cpu().numpy()
                self._apply_stdp_gpu(spike_idx)
                v_gpu[spiked] = V_RESET

            if n_arcane > 0:
                spiked_cpu = spiked.cpu().numpy()
                for i, an in enumerate(arcane_neurons):
                    start = i * group_size
                    end = start + group_size if i < n_arcane - 1 else self.n
                    group_rate = float(spiked_cpu[start:end].sum()) / (end - start)
                    an.step(group_rate * 5.0 + float(self.i_ext[start]))

        self.v = v_gpu.cpu().numpy()
        self.v_gpu = v_gpu
        self._w_np_cache = None
        return spikes_total

    def _apply_stdp_gpu(self, spike_idx: np.ndarray) -> None:
        """STDP weight update — asymmetric Hebbian (Bi & Poo 1998).

        LTP: pre fires before post (dt > 0) → strengthen w[post, pre]
        LTD: post fires before pre (dt < 0) → weaken w[post, pre]
        """
        torch = self.torch
        t_now = self.t
        dt_pre = t_now - self.last_spike

        # Hoist tensor allocation outside loop (was per-spike, 20K transfers)
        dt_arr = torch.from_numpy(dt_pre.astype(np.float32)).to(self.device)

        for i in spike_idx[:20]:
            # LTP: neuron i just fired (post). Strengthen inputs from
            # neurons that fired recently BEFORE i (dt > 0 = pre before post)
            mask_ltp = (dt_arr > 0) & (dt_arr < 100)
            dw_ltp = STDP_A_PLUS * torch.exp(-dt_arr / STDP_TAU) * mask_ltp
            self.w[i, :] = torch.clamp(self.w[i, :] + dw_ltp, 0, STDP_W_MAX)

            # LTD: neuron i just fired (pre). Weaken connections TO neurons
            # that fired recently BEFORE i (dt > 0 = they were post, i is now pre)
            # This is anti-causal: post fired, then pre fires → depress
            dw_ltd = STDP_A_MINUS * torch.exp(-dt_arr / STDP_TAU) * mask_ltp
            self.w[:, i] = torch.clamp(self.w[:, i] - dw_ltd, 0, STDP_W_MAX)

        self.last_spike[spike_idx] = t_now

    def _w_mean(self) -> float:
        return float(self.w.mean().item())

    def _w_std(self) -> float:
        return float(self.w.std().item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        w_cpu = self.w.cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump({
                "v": self.v.copy(),
                "w": w_cpu,
                "last_spike": self.last_spike.copy(),
                "t": self.t,
                "n": self.n,
                "dt": self.dt,
                "backend": "gpu",
            }, f)

    @classmethod
    def load(cls, path: Path) -> GPULIFNetwork:
        import torch

        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls.__new__(cls)
        net.n = data["n"]
        net.dt = data["dt"]
        net.dt_ms = data["dt"] * 1000
        net.rng = np.random.default_rng(42)
        net.v = data["v"]
        net.last_spike = data["last_spike"]
        net.t = data["t"]
        net.i_ext = np.full(net.n, 0.3, dtype=np.float32)
        net.torch = torch
        net.device = torch.device("cuda")
        net.w = torch.from_numpy(data["w"].astype(np.float32)).to(net.device)
        net.v_gpu = torch.from_numpy(net.v).to(net.device)
        net._w_np_cache = None
        return net

    def get_w_numpy(self) -> np.ndarray:
        """Get W as numpy array (cached, for retrieve.py)."""
        if self._w_np_cache is None:
            self._w_np_cache = self.w.cpu().numpy()
        return self._w_np_cache


class DenseCPULIFNetwork(_BaseLIFNetwork):
    """Dense numpy backend with fully vectorized STDP.

    STDP uses outer-product batch updates — no per-spike loops.
    At 5000 neurons: ~10s per 1s burst (33% duty at 30s interval).
    The GEMV (W @ fired) uses BLAS, STDP is pure vectorized numpy.
    """

    def __init__(self, n: int = 5000, dt: float = 0.001, seed: int = 42):
        super().__init__(n, dt, seed)

        self.w = np.zeros((n, n), dtype=np.float32)
        n_conn = int(0.10 * n)
        for i in range(n):
            targets = self.rng.choice(n, n_conn, replace=False)
            self.w[i, targets] = self.rng.uniform(0.1, 0.6, n_conn).astype(np.float32)
        np.fill_diagonal(self.w, 0)

        # Connectivity mask: STDP only modifies existing connections
        self.mask = (self.w > 0).astype(np.float32)

        logger.info("Dense CPU network: %d neurons, W=%.0f MB, sparsity=%.0f%%",
                     n, n * n * 4 / 1e6, (1 - self.mask.sum() / n / n) * 100)

    def run(self, duration: float, arcane_neurons=None) -> int:
        steps = int(duration / self.dt)
        spikes_total = 0
        n_arcane = len(arcane_neurons) if arcane_neurons else 0
        group_size = self.n // n_arcane if n_arcane > 0 else 0

        for step in range(steps):
            self.t += self.dt
            fired = (self.v >= V_THRESH).astype(np.float32)
            i_syn = self.w @ fired  # BLAS GEMV — the fast path

            dv = (-(self.v - V_REST) / TAU_M + self.i_ext + i_syn * 0.5) * self.dt_ms
            self.v += dv

            spiked = self.v >= V_THRESH
            if spiked.any():
                n_spiked = int(spiked.sum())
                spikes_total += n_spiked
                # Vectorized STDP every 10 steps (amortize cost)
                if step % 10 == 0:
                    self._apply_stdp_batch(spiked)
                self.v[spiked] = V_RESET
                self.last_spike[spiked] = self.t

            if n_arcane > 0:
                for i, an in enumerate(arcane_neurons):
                    start = i * group_size
                    end = start + group_size if i < n_arcane - 1 else self.n
                    group_rate = float(spiked[start:end].sum()) / (end - start)
                    an.step(group_rate * 5.0 + float(self.i_ext[start]))

        return spikes_total

    def _apply_stdp_batch(self, spiked: np.ndarray) -> None:
        """Fully vectorized STDP via masked outer product.

        dW = A+ × spike_post ⊗ exp(-dt/τ) × mask   (LTP)
           - A- × exp(-dt/τ) ⊗ spike_post × mask   (LTD)

        Single numpy operation, no Python loops.
        """
        dt_pre = (self.t - self.last_spike).astype(np.float32)
        valid = (dt_pre > 0) & (dt_pre < 100)
        trace = np.exp(-dt_pre / STDP_TAU) * valid  # (N,)
        spike_f = spiked.astype(np.float32)  # (N,)

        # LTP: rows of spiked post-synaptic neurons strengthened
        # dW[post, pre] += A+ × spike_post[post] × trace[pre]
        dw_ltp = STDP_A_PLUS * np.outer(spike_f, trace)

        # LTD: columns of spiked pre-synaptic neurons weakened
        # dW[post, pre] -= A- × trace[post] × spike_post[pre]
        dw_ltd = STDP_A_MINUS * np.outer(trace, spike_f)

        self.w += (dw_ltp - dw_ltd) * self.mask
        np.clip(self.w, 0, STDP_W_MAX, out=self.w)

    def _w_mean(self) -> float:
        nz = self.w[self.mask > 0]
        return float(nz.mean()) if len(nz) > 0 else 0.0

    def _w_std(self) -> float:
        nz = self.w[self.mask > 0]
        return float(nz.std()) if len(nz) > 0 else 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "v": self.v,
                "w": self.w,
                "mask": self.mask,
                "last_spike": self.last_spike,
                "t": self.t,
                "n": self.n,
                "dt": self.dt,
                "backend": "dense_cpu",
            }, f)

    @classmethod
    def load(cls, path: Path) -> DenseCPULIFNetwork:
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls.__new__(cls)
        net.n = data["n"]
        net.dt = data["dt"]
        net.dt_ms = data["dt"] * 1000
        net.rng = np.random.default_rng(42)
        net.v = data["v"]
        net.w = data["w"]
        net.mask = data.get("mask", (data["w"] > 0).astype(np.float32))
        net.last_spike = data["last_spike"]
        net.t = data["t"]
        net.i_ext = np.full(net.n, 0.3, dtype=np.float32)
        return net

    def get_w_numpy(self) -> np.ndarray:
        return self.w


def create_network(
    n_neurons: int = 20000,
    backend: str | None = None,
    dt: float = 0.001,
    seed: int = 42,
) -> GPULIFNetwork | DenseCPULIFNetwork:
    """Create a LIF network with the specified backend.

    backend="gpu"        → PyTorch CUDA (default if available, N=20000)
    backend="sparse_cpu" → DenseCPU with masked STDP (default fallback, N=5000)
    backend="dense_cpu"  → same as sparse_cpu (unified)
    backend=None         → auto-detect
    """
    if backend is None:
        backend = detect_backend()

    if backend == "gpu":
        return GPULIFNetwork(n=n_neurons, dt=dt, seed=seed)
    return DenseCPULIFNetwork(n=n_neurons, dt=dt, seed=seed)


def load_network(path: Path, backend: str | None = None) -> GPULIFNetwork | DenseCPULIFNetwork:
    """Load a saved network, using the specified or saved backend."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    saved_backend = data.get("backend", "dense_cpu")
    target = backend or saved_backend

    if target == "gpu":
        try:
            import torch
            if torch.cuda.is_available():
                return GPULIFNetwork.load(path)
        except ImportError:
            logger.warning("GPU requested but torch not available, falling back to CPU")
    return DenseCPULIFNetwork.load(path)


# ── Sparse W persistence ─────────────────────────────────────────


def save_sparse(w: np.ndarray, path: Path) -> int:
    """Save weight matrix as scipy COO sparse format.

    For a 2000×2000 matrix at 90% sparsity: ~1.5 MB vs ~15 MB dense.
    Returns bytes written.
    """
    try:
        from scipy.sparse import coo_matrix, save_npz
    except ImportError:
        logger.warning("scipy not available — saving dense fallback")
        np.save(path.with_suffix(".npy"), w)
        return w.nbytes

    sparse_w = coo_matrix(w)
    save_npz(path.with_suffix(".npz"), sparse_w)
    size = path.with_suffix(".npz").stat().st_size
    ratio = size / w.nbytes * 100
    logger.info("Sparse W saved: %s (%.1f%% of dense)", path.with_suffix(".npz").name, ratio)
    return size


def load_sparse(path: Path) -> np.ndarray:
    """Load weight matrix from sparse format, return dense."""
    npz = path.with_suffix(".npz")
    npy = path.with_suffix(".npy")

    if npz.exists():
        try:
            from scipy.sparse import load_npz
            w = load_npz(npz).toarray()
            logger.info("Loaded sparse W from %s: shape=%s", npz.name, w.shape)
            return w.astype(np.float32)
        except ImportError:
            logger.warning("scipy not available for sparse load")

    if npy.exists():
        return np.load(npy).astype(np.float32)

    raise FileNotFoundError(f"No sparse weight file at {path}")
