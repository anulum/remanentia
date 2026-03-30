#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: Remanentia — persistent AI memory
# Repository: https://github.com/anulum/remanentia
"""Benchmark ALL local GGUF models on LongMemEval.

Automatically restarts llama-server for each model and runs the benchmark.
Results saved to data/local_model_comparison.json.

Usage:
    python bench_all_local_models.py               # 50 questions per model
    python bench_all_local_models.py --full         # 500 questions on best
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Models to test (ordered small → large)
MODELS = [
    # ("qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen 2.5 1.5B"),  # DONE: 28.0%
    # ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama 3.2 1B"),  # DONE: 16.0%
    ("qwen2.5-3b-instruct-q4_k_m.gguf", "Qwen 2.5 3B"),
    ("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Llama 3.2 3B"),
    ("Phi-3.5-mini-instruct-Q4_K_M.gguf", "Phi 3.5 Mini"),
    ("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "Qwen 2.5 7B"),
    ("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "Mistral 7B"),
    ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Llama 3.1 8B"),
]

MODEL_DIR = Path(__file__).parent / "models"
SERVER_BIN = "/tmp/llama-cpp-build/build-rocm/bin/llama-server"
SERVER_PORT = 8080
LIMIT = 50  # questions per model for comparison
GPU_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "HIP_VISIBLE_DEVICES": "4",
}


_SERVER_PROC = None
_SERVER_LOG = None


def _kill_server():
    """Kill the running llama-server process."""
    global _SERVER_PROC, _SERVER_LOG
    if _SERVER_PROC is not None:
        try:
            _SERVER_PROC.terminate()
            _SERVER_PROC.wait(timeout=5)
        except Exception:
            try:
                _SERVER_PROC.kill()
                _SERVER_PROC.wait(timeout=3)
            except Exception:
                pass
        _SERVER_PROC = None
    if _SERVER_LOG is not None:
        try:
            _SERVER_LOG.close()
        except Exception:
            pass
        _SERVER_LOG = None
    time.sleep(2)


def _start_server(model_file: str) -> subprocess.Popen | None:
    """Start llama-server with a model, return process or None on failure."""
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None

    env = {**os.environ, **GPU_ENV}
    global _SERVER_PROC, _SERVER_LOG
    _SERVER_LOG = open("/tmp/llama-server-bench.log", "w")
    proc = subprocess.Popen(
        [
            SERVER_BIN,
            "-m", str(model_path),
            "--port", str(SERVER_PORT),
            "--host", "127.0.0.1",
            "-ngl", "99",
            "-c", "16384",
            "-np", "1",
            "-fit", "off",
            "--no-warmup",
        ],
        env=env,
        stdout=_SERVER_LOG,
        stderr=_SERVER_LOG,
    )

    # Wait for server to be ready
    import urllib.request
    for i in range(90):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SERVER_PORT}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    _SERVER_PROC = proc
                    return proc
        except Exception:
            pass
        time.sleep(2)

    print("  Server failed to start within 180s")
    proc.kill()
    return None


def _run_benchmark(limit: int) -> dict:
    """Run LongMemEval benchmark and return results dict."""
    from llm_backend import LocalLLMBackend
    from answer_extractor import set_llm_backend

    backend = LocalLLMBackend(timeout=120.0)
    set_llm_backend(backend)

    # Force --llm in sys.argv for bench module
    if "--llm" not in sys.argv:
        sys.argv.append("--llm")

    # Reload bench module to pick up patched backend
    import importlib
    if "bench_longmemeval" in sys.modules:
        importlib.reload(sys.modules["bench_longmemeval"])
    import bench_longmemeval as bench

    # Apply patches
    import bench_longmemeval_local
    bench._answer_from_retrieval = bench_longmemeval_local._local_answer_from_retrieval
    bench._arcane_answer = bench_longmemeval_local._local_arcane_answer

    # Load data
    with open(bench.DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    from collections import Counter, defaultdict
    import re

    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    t0 = time.monotonic()

    for i, item in enumerate(data):
        qtype = item["question_type"]
        question = item["question"]
        gold = str(item["answer"])

        idx = bench._build_index_for_question(item["haystack_sessions"])
        hypothesis = bench._answer_from_retrieval(
            question, idx, item["haystack_sessions"], qtype, use_llm=True
        )

        type_total[qtype] += 1
        gold_lower = gold.lower().strip()
        hyp_lower = hypothesis.lower().strip()
        local_match = (
            gold_lower in hyp_lower
            or hyp_lower in gold_lower
            or bench._fuzzy_overlap(gold_lower, hyp_lower) > 0.6
        )
        if local_match:
            type_correct[qtype] += 1

    elapsed = time.monotonic() - t0
    total_correct = sum(type_correct.values())
    total = sum(type_total.values())

    results = {
        "overall": f"{total_correct}/{total} ({total_correct/total*100:.1f}%)" if total else "0/0",
        "overall_pct": round(total_correct / total * 100, 1) if total else 0,
        "per_type": {},
        "time_s": round(elapsed, 1),
        "ms_per_q": round(elapsed / total * 1000) if total else 0,
        "n_questions": total,
    }
    for qtype in sorted(type_total.keys()):
        c = type_correct[qtype]
        t = type_total[qtype]
        results["per_type"][qtype] = f"{c}/{t} ({c/t*100:.1f}%)" if t else "0/0"

    return results


def main():
    full_run = "--full" in sys.argv

    print("=" * 70)
    print("Remanentia Local Model Comparison — LongMemEval Benchmark")
    print(f"Questions per model: {LIMIT}")
    print(f"Server: {SERVER_BIN}")
    print(f"GPU: RX 6600 XT (ROCm, HIP_VISIBLE_DEVICES=4)")
    print("=" * 70)

    all_results = {}

    for model_file, model_name in MODELS:
        print(f"\n{'─' * 50}")
        print(f"Model: {model_name} ({model_file})")
        print(f"{'─' * 50}")

        _kill_server()
        print("  Starting server...")
        proc = _start_server(model_file)
        if proc is None:
            all_results[model_name] = {"error": "server failed to start"}
            continue

        print("  Server ready. Running benchmark...")
        try:
            results = _run_benchmark(LIMIT)
            all_results[model_name] = results
            print(f"  Result: {results['overall']} in {results['time_s']}s "
                  f"({results['ms_per_q']}ms/q)")
            for qtype, score in results["per_type"].items():
                print(f"    {qtype:35s}: {score}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = {"error": str(e)}
        finally:
            _kill_server()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    # Save results
    output_path = Path(__file__).parent / "data" / "local_model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Overall':>10} {'Time':>8} {'ms/q':>8}")
    print("-" * 55)
    for model_name in [m[1] for m in MODELS]:
        r = all_results.get(model_name, {})
        if "error" in r:
            print(f"{model_name:<25} {'ERROR':>10}")
        else:
            print(f"{model_name:<25} {r.get('overall_pct', 0):>9.1f}% "
                  f"{r.get('time_s', 0):>7.1f}s {r.get('ms_per_q', 0):>7d}")
    print("-" * 55)


if __name__ == "__main__":
    main()
