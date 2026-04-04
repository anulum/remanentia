# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Local LLM setup utilities

# Repository: https://github.com/anulum/remanentia
"""Local LLM setup utilities — GPU detection, model management, server control."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".remanentia"
_REPO_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _REPO_DIR / "models"


# ── Data types ────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Recommended model configuration."""

    name: str
    quant: str
    device: str
    download_gb: float
    vram_gb: float


# ── Hardware detection ────────────────────────────────────────────


def detect_gpu_vram() -> float:  # pragma: no cover
    """Return available VRAM in GB (0 if no GPU detected)."""
    # Try torch first
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem
            return round(total / (1024**3), 1)
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        if out:
            return round(int(out.split("\n")[0]) / 1024, 1)
    except Exception:
        pass

    return 0.0


def detect_system_ram() -> float:  # pragma: no cover
    """Return system RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
    except Exception:
        pass

    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip()
        return round(int(out) / (1024**3), 1)
    except Exception:
        pass

    return 0.0


# ── Model recommendation ─────────────────────────────────────────


def recommend_model(vram: float = 0, ram: float = 0) -> ModelConfig | None:
    """Recommend a model based on available hardware."""
    if vram >= 8:
        return ModelConfig("qwen2.5-7b-instruct", "q4_k_m", "gpu", 4.5, 5.0)
    if vram >= 4:
        return ModelConfig("qwen2.5-3b-instruct", "q4_k_m", "gpu", 1.8, 3.0)
    if ram >= 16:
        return ModelConfig("qwen2.5-3b-instruct", "q4_k_m", "cpu", 1.8, 0.0)
    if ram >= 8:
        return ModelConfig("qwen2.5-1.5b-instruct", "q4_k_m", "cpu", 1.0, 0.0)
    return None


# ── Config writing ────────────────────────────────────────────────


def write_config(
    backend: str = "local",
    local_url: str = "http://localhost:8080/v1",
    local_model: str = "qwen2.5-7b-instruct",
    path: Path | None = None,
) -> Path:
    """Write LLM configuration to TOML file."""
    if path is None:
        path = _DEFAULT_CONFIG_DIR / "llm.toml"
    path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "[llm]\n"
        f'backend = "{backend}"\n'
        f'local_url = "{local_url}"\n'
        f'local_model = "{local_model}"\n'
        "\n"
        "[llm.tokens]\n"
        "extract = 100\n"
        "generate = 200\n"
        "synthesise = 200\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


# ── CLI entry points ──────────────────────────────────────────────


def cmd_setup_llm(args) -> None:  # pragma: no cover
    """Interactive hardware detection and model setup."""
    print("Detecting hardware...")
    vram = detect_gpu_vram()
    ram = detect_system_ram()

    if vram > 0:
        print(f"  GPU VRAM: {vram:.1f} GB")
    else:
        print("  GPU: not detected")
    print(f"  System RAM: {ram:.1f} GB")

    rec = recommend_model(vram, ram)
    if rec is None:
        print("\nInsufficient hardware for local LLM. Use cloud backend instead:")
        print("  remanentia recall 'query' --llm --llm-backend anthropic")
        return

    print(f"\n  Recommended: {rec.name} {rec.quant} ({rec.device} mode)")
    print(f"  Download: ~{rec.download_gb:.1f} GB")

    model_name = getattr(args, "model", None) or rec.name

    # Write config
    local_url = "http://localhost:8080/v1"
    cfg_path = write_config(
        backend="local",
        local_url=local_url,
        local_model=model_name,
    )
    print(f"\nConfiguration saved to {cfg_path}")
    print(f"\nNext: start a local LLM server (llama.cpp or Ollama) on port 8080,")
    print(f"then run: remanentia recall 'query' --llm --llm-backend local")


def cmd_serve_llm(args) -> None:  # pragma: no cover
    """Start llama.cpp server with configured model."""
    port = getattr(args, "port", 8080)

    # Find model file
    model_dir = _DEFAULT_MODEL_DIR
    gguf_files = list(model_dir.glob("*.gguf")) if model_dir.exists() else []

    if not gguf_files:
        print(f"No .gguf model files found in {model_dir}")
        print("Download a model first: remanentia setup-llm")
        return

    model_path = gguf_files[0]
    print(f"Starting llama.cpp server...")
    print(f"  Model: {model_path.name}")
    print(f"  Port: {port}")

    try:
        subprocess.run(
            ["llama-server", "-m", str(model_path), "--port", str(port), "--host", "0.0.0.0"],
            check=True,
        )
    except FileNotFoundError:
        print("\nllama-server not found. Install llama.cpp:")
        print("  https://github.com/ggerganov/llama.cpp#build")
    except KeyboardInterrupt:
        print("\nServer stopped.")
