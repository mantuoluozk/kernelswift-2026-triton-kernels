#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR="$ROOT_DIR/../../evaluator/auto_bench.py"
DEVICE_ID="${DEVICE_ID:-0}"
WARMUP="${WARMUP:-200}"
REPEAT="${REPEAT:-500}"

TASKS=(
    task01_grouped_topk
    task02_fused_moe
    task03_flex_attention
    task04_splade_sparse_pooler
    task05_music_flamingo_rotary_embedding
    task06_mm_encoder_attention
    task07_mhc_post
    task08_hc_split_sinkhorn
    task09_centre_random_augmentation
    task10_head_compute_mix_bwd
)

export HIP_VISIBLE_DEVICES="$DEVICE_ID"

for task in "${TASKS[@]}"; do
    echo "===== $task ====="
    python3 "$EVALUATOR" \
        --v0_file "$ROOT_DIR/$task/reference.py" \
        --v1_file "$ROOT_DIR/$task/solution.py" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT"
done
