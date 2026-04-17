#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Local LLM benchmark harness
#
# Usage: tools/bench_local_llm.sh <output.md> <model1> [model2 ...]
#
# Probes an Ollama-compatible server (default localhost:11434) with three
# canonical prompt categories (math, RAG date arithmetic, factoid) and
# emits a Markdown table with tok/s, latency, and aggregate VRAM use.
#
# Reproduces the 2026-04-10 evaluation in
# docs/internal/benchmark_2026-04-10_local_llm_evaluation.md
set -uo pipefail

OUTPUT="${1:-/tmp/llm_benchmark_results.md}"
shift || true
MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "usage: $0 <output.md> <model1> [model2 ...]" >&2
  exit 2
fi

declare -A PROMPTS=(
  [math]="What is 25 * 4? Reply with one number only."
  [rag]="Context: The user mentioned they joined the gym on 2023-01-15 and quit on 2023-03-20. Question: How many days did the user stay at the gym? Reply with the number followed by 'days'."
  [factoid]="What is the capital of Slovakia? One word."
)

api_call() {
  local model="$1"
  local prompt="$2"
  curl -sS http://localhost:11434/api/generate \
    -d "{\"model\":\"$model\",\"prompt\":$(jq -Rs . <<<"$prompt"),\"stream\":false,\"options\":{\"num_predict\":50,\"temperature\":0.1}}" \
    2>&1
}

vram_used() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showmeminfo vram 2>&1 \
      | awk '/Used Memory/ {sum+=$NF} END {printf "%.2f GB", sum/1024/1024/1024}'
  else
    echo "n/a"
  fi
}

echo "# Remanentia Local LLM Benchmark — $(date -u +%Y-%m-%dT%H:%MZ)" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "| Model | Prompt | Response | tok/s | Latency | VRAM |" >> "$OUTPUT"
echo "|-------|--------|----------|-------|---------|------|" >> "$OUTPUT"

for model in "${MODELS[@]}"; do
  for category in math rag factoid; do
    prompt="${PROMPTS[$category]}"
    result=$(api_call "$model" "$prompt")

    response=$(echo "$result" | jq -r '.response // "ERROR"' | tr '\n' ' ' | head -c 80)
    eval_count=$(echo "$result" | jq -r '.eval_count // 0')
    eval_dur=$(echo "$result" | jq -r '.eval_duration // 1')
    total_dur=$(echo "$result" | jq -r '.total_duration // 1')

    if [ "$eval_count" != "0" ] && [ "$eval_dur" != "0" ]; then
      tok_per_s=$(awk "BEGIN {printf \"%.1f\", $eval_count*1000000000/$eval_dur}")
    else
      tok_per_s="N/A"
    fi
    latency_s=$(awk "BEGIN {printf \"%.2fs\", $total_dur/1000000000}")
    vram_after=$(vram_used)

    printf "| %s | %s | %s | %s | %s | %s |\n" \
      "$model" "$category" "$response" "$tok_per_s" "$latency_s" "$vram_after" \
      >> "$OUTPUT"
    echo "  $model/$category: $tok_per_s tok/s, $latency_s, $vram_after VRAM"
  done
done

echo "" >> "$OUTPUT"
echo "Benchmark complete. Results in $OUTPUT"
