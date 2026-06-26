#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Launch all 5 training jobs in parallel on 5 AMD RX 6600 XT GPUs.
# Each job gets its own GPU via CUDA_VISIBLE_DEVICES (ROCm HIP respects this).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${VENV_PYTHON:-python}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Remanentia Temporal Training ==="
echo "Python: $PYTHON"
echo "Base:   $BASE_DIR"
echo "Logs:   $LOG_DIR"
echo ""

cd "$BASE_DIR"

# Phase 1: Generate data (if not already generated)
if [ ! -f "$SCRIPT_DIR/datasets/embedding_triplets.jsonl" ]; then
    echo "[Phase 1] Generating synthetic data..."
    "$PYTHON" "$SCRIPT_DIR/temporal_synth.py"
    echo "[Phase 1] Extracting LongMemEval data..."
    "$PYTHON" "$SCRIPT_DIR/generate_data.py"
else
    echo "[Phase 1] Training data already exists, skipping generation."
fi

echo ""
echo "[Phase 2] Launching 5 training jobs in parallel..."
echo ""

# C1: Bi-encoder on GPU 0
echo "  GPU 0: C1 Temporal Embedding..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT_DIR/train_embedding.py" \
    > "$LOG_DIR/c1_embedding.log" 2>&1 &
PID_C1=$!

# C2: Cross-encoder on GPU 1
echo "  GPU 1: C2 Temporal Cross-Encoder..."
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT_DIR/train_cross_encoder.py" \
    > "$LOG_DIR/c2_crossencoder.log" 2>&1 &
PID_C2=$!

# C3: Relation classifier on GPU 2
echo "  GPU 2: C3 Temporal Relation Classifier..."
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$SCRIPT_DIR/train_relation.py" \
    > "$LOG_DIR/c3_relation.log" 2>&1 &
PID_C3=$!

# C4: Date normaliser on GPU 3
echo "  GPU 3: C4 Date Normaliser..."
CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$SCRIPT_DIR/train_date_normalizer.py" \
    > "$LOG_DIR/c4_date_normalizer.log" 2>&1 &
PID_C4=$!

# C5: Fact validity on GPU 4
echo "  GPU 4: C5 Fact Validity Model..."
CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$SCRIPT_DIR/train_fact_validity.py" \
    > "$LOG_DIR/c5_fact_validity.log" 2>&1 &
PID_C5=$!

echo ""
echo "All jobs launched. PIDs: C1=$PID_C1, C2=$PID_C2, C3=$PID_C3, C4=$PID_C4, C5=$PID_C5"
echo "Waiting for completion..."
echo ""

# Wait and report
FAILED=0
for name_pid in "C1:$PID_C1" "C2:$PID_C2" "C3:$PID_C3" "C4:$PID_C4" "C5:$PID_C5"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if wait "$pid"; then
        echo "  $name: OK"
    else
        echo "  $name: FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All 5 components trained successfully ==="
else
    echo "=== $FAILED component(s) failed — check logs in $LOG_DIR ==="
    exit 1
fi
