# Local LLM Setup (Ollama + Gemma 3 4B)

Remanentia ships with a pluggable LLM backend (`llm_backend.py`) that defaults
to a local Ollama server running `gemma3:4b`. This guide covers installation,
GPU acceleration on AMD and NVIDIA, and verification.

## Why Gemma 3 4B?

After benchmarking Gemma 3 (1B, 4B, 12B), Gemma 4 (e2b, e4b), and the
previously-default Qwen 2.5 7B on an RX 6600 XT (8 GB VRAM), `gemma3:4b` was
the best speed/quality tradeoff:

- 45-67 tok/s on a single 8 GB GPU
- 3.3 GB on disk, ~3.5 GB VRAM, leaves 4.5 GB headroom
- Correct math and factoid answers
- Date arithmetic is delegated to Python (`temporal_code_execute`) so the LLM
  never has to count days

Full evaluation: `docs/internal/benchmark_2026-04-10_local_llm_evaluation.md`.

## 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

The installer auto-detects NVIDIA (CUDA) GPUs. AMD GPUs need an additional
step (see §3).

## 2. Pull the model

```bash
ollama pull gemma3:4b
```

Verify with a smoke test:

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -d '{"model":"gemma3:4b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}' \
  | jq -r '.choices[0].message.content'
```

## 3. AMD GPU acceleration (Vulkan)

Ollama 0.20.x ships CUDA, MLX, and Vulkan runners. **No bundled ROCm runner.**
For RX 5000/6000/7000 series cards, the Vulkan path via the open-source RADV
driver is the simplest route — performance is 70-80% of native ROCm.

Create a systemd drop-in to enable Vulkan and pin a specific GPU:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_VULKAN=1"
Environment="CUDA_VISIBLE_DEVICES="           # disable any NVIDIA cards
Environment="GGML_VK_VISIBLE_DEVICES=0"       # 0 = first Vulkan device — adjust!
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_KEEP_ALIVE=2m"
Environment="OLLAMA_NUM_THREAD=2"
User=anulum
Group=anulum
CPUQuota=50%
MemoryMax=4G
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Find the right `GGML_VK_VISIBLE_DEVICES` index by inspecting the journal after
restart:

```bash
journalctl -u ollama --since "1 minute ago" | grep "inference compute"
```

You will see entries like `name=Vulkan0 description="AMD Radeon RX 6600 XT"`.
Use the index after "Vulkan" — e.g. `Vulkan2` → `GGML_VK_VISIBLE_DEVICES=2`.

### CPU/RAM caps

`CPUQuota=50%` and `MemoryMax=4G` prevent Ollama from saturating CPU/RAM when
the model partially offloads to host memory. Adjust if you want more headroom
for batch jobs.

## 4. NVIDIA GPU acceleration

The default install handles CUDA automatically. To pin a specific GPU:

```bash
sudo systemctl edit ollama
# Add:
[Service]
Environment="CUDA_VISIBLE_DEVICES=0"
```

## 5. Verify with Remanentia

```python
from llm_backend import LocalLLMBackend, AutoBackend, LLMConfig

# Direct local
b = LocalLLMBackend()
assert b.is_available(), "Ollama not reachable on localhost:11434"
print(b.complete("What is the capital of Slovakia?", max_tokens=20))
# Expected: 'Bratislava' (or similar single word)

# Auto: prefers local, falls back to the hosted backend, then Null
auto = AutoBackend()
print(auto.complete("Hello"))
```

## 6. Configuration overrides

Create `~/.remanentia/llm.toml`:

```toml
[llm]
backend = "auto"                              # auto | local | anthropic | none
local_url = "http://localhost:11434/v1"
local_model = "gemma3:4b"
local_timeout = 60
anthropic_model = "claude-haiku-4-5-20251001"

[llm.tokens]
extract = 100
generate = 200
synthesise = 200
```

Or set the environment variable `REMANENTIA_LLM_CONFIG=/path/to/llm.toml`.

`remanentia serve-llm` starts `llama-server` on `127.0.0.1` by default. Use
`--host` only for deliberately isolated network deployments with an external
firewall/authentication boundary.

## 6a. ML350 32B coder route

ML350 exposes a hot Qwen 2.5 Coder 32B Q4 route for higher-quality local
coding and memory-synthesis work. It is intentionally kept behind SSH/UFW.

Open a tunnel from the workstation or notebook:

```bash
ssh -N -L 11438:127.0.0.1:11438 anulum@192.168.1.30
```

Then point Remanentia at the tunneled llama.cpp endpoint:

```toml
[llm]
backend = "local"
local_url = "http://127.0.0.1:11438/v1"
local_model = "qwen2.5-coder-32b-instruct-q4"
local_timeout = 300

[llm.tokens]
extract = 160
generate = 320
synthesise = 320
```

Use the config explicitly:

```bash
export REMANENTIA_LLM_CONFIG="$HOME/.remanentia/ml350-code32b.toml"
remanentia recall "query" --llm --llm-backend local
```

The same model is registered in ML350 LiteLLM as `local-code` and `local-best`.
Use LiteLLM only when the caller can pass the configured bearer token; the
direct SSH tunnel above needs no API key and is simpler for local Remanentia
runs.

## 6b. Using the local LLM through the MCP server

The MCP server is the usual way an MCP-compatible tool (Cursor and
others) talks to Remanentia. Two flags flip the backend selection:

```bash
# LLM-synthesised recall answers, local-Ollama backend pinned
python mcp_server.py --llm --local-llm
```

Under the hood the flags set two environment variables the
``handle_recall`` path already reads:

- ``REMANENTIA_LLM_ANSWERS=1`` — call ``resolve_backend`` and
  synthesise an answer rather than return raw retrieval context.
- ``REMANENTIA_LLM_BACKEND=local`` — ``resolve_backend`` builds
  a ``LocalLLMBackend`` from the ``llm.toml`` configuration.

If you prefer the env-var form (e.g. inside a systemd unit or an
MCP config `env` block) it is identical:

```ini
[Service]
Environment="REMANENTIA_LLM_ANSWERS=1"
Environment="REMANENTIA_LLM_BACKEND=local"
```

With ``REMANENTIA_LLM_BACKEND=auto`` (the default), ``AutoBackend``
tries local → hosted → Null in that order, so a running Ollama
at the configured URL wins automatically.

## 7. Re-running the benchmark

```bash
tools/bench_local_llm.sh /tmp/results.md gemma3:4b gemma3:12b
```

This produces the same Markdown table format as the canonical evaluation.

## Troubleshooting

**Ollama returns `None` from `complete()`** — increase the `LocalLLMBackend`
timeout (default 60 s). Cold model loads can take 30-60 s on the first call;
warm cache is sub-second.

**Wrong GPU is being used** — check
`journalctl -u ollama | grep "inference compute"` after restart and adjust
`GGML_VK_VISIBLE_DEVICES`.

**Model doesn't fit in VRAM** — Ollama partially offloads to CPU/RAM. You will
see speed drop from ~50 tok/s to 1-5 tok/s. Switch to a smaller quant or a
smaller model. `gemma3:1b` always fits in 1 GB; `gemma3:4b` always fits in 4 GB.

**Gemma 4 produces garbage output** — the current Ollama registry has tagging
issues for `gemma4:e4b-it-q4_K_M` (the manifest points to bf16). Stick with
Gemma 3 until Google or Ollama publishes proper Q4 builds.

## Suitability Matrix — what local LLM is good for

The Gemma 3 4B Q4 model running on a single RX 6600 XT 8GB (Vulkan) has
measured limits:

| Task | Typical prompt | Suitable? | Notes |
|------|----------------|-----------|-------|
| Single-paragraph Q&A | 200-500 tokens | ✅ | 45-67 tok/s, correct answers |
| Answer extraction / normalization | < 300 tokens | ✅ | Fast, reliable |
| Signal detection / classification | < 200 tokens | ✅ | Short outputs |
| Memory recall (short context) | < 1K tokens | ✅ | Workable |
| **LongMemEval benchmark** | **10K+ tokens** | ❌ | Context overflows default 4-8K window; 4B model underperforms on complex multi-step temporal reasoning |
| Multi-session cross-referencing | 5K+ tokens | ⚠ | Quality degrades sharply above ~4K tokens |
| Code generation | varies | ⚠ | Limited instruction-following |

**Measured 2026-04-11** on `bench_longmemeval.py --llm --arcane --local-llm --limit 10`:
- Speed: ~110 s/question (vs ~5 s with GPT-4o-mini)
- Quality: 0/10 temporal-reasoning correct (vs 7/10 with GPT-4o-mini)
- Root cause: prompt length ~11K tokens; Gemma 3 4B Q4 default context 4-8K on 8GB VRAM; most of the context is truncated and the model answers from nothing

**Recommendation:** use local LLM for short, single-turn Remanentia operations
(recall, extraction, classification). Keep GPT-4o-mini (or any cloud model with
≥32K context and strong reasoning) for the LongMemEval benchmark and other
long-context workloads. The `--local-llm` flag on `bench_longmemeval.py` is
retained as an opt-in for experimentation, not production evaluation.
