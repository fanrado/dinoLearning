#!/bin/bash
# =============================================================================
# Iterative Two-Pass DINO Training Script
#
# Pass 1 — Self-supervised pre-training with full augmentations (shape learning)
# Pass 2 — Supervised fine-tuning with minimal augmentations (color/detail learning)
#
# Usage:
#   bash run_dino_iterative.sh              # run both passes
#   bash run_dino_iterative.sh --pass1-only # run Pass 1 only
#   bash run_dino_iterative.sh --pass2-only # run Pass 2 only (requires Pass 1 checkpoint)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to dataset root (must contain train/ and val/ in ImageFolder format)
DATA_PATH="/nfs/data/1/rrazakami/work/dino/PetImages"

# Root output directory; sub-dirs per pass are created automatically
OUTPUT_ROOT="./OUTPUT_DINO_ITERATIVE"

# Number of GPUs available
NUM_GPUS=1

# Architecture — kept consistent across both passes
ARCH="vit_small"
PATCH_SIZE=16
OUT_DIM=65536
NUM_WORKERS=10

# ---------------------------------------------------------------------------
# Pass 1 hyperparameters — full augmentations, shape learning
# ---------------------------------------------------------------------------
P1_EPOCHS=200
P1_BATCH_SIZE=64
P1_LR=0.000125         # 0.0005 base (batch 256) scaled to batch 32: *32/256
P1_MIN_LR=1e-6
P1_WEIGHT_DECAY=0.04
P1_WEIGHT_DECAY_END=0.4
P1_WARMUP_EPOCHS=10
P1_TEACHER_TEMP=0.04
P1_WARMUP_TEACHER_TEMP=0.04
P1_WARMUP_TEACHER_TEMP_EPOCHS=30
P1_MOMENTUM_TEACHER=0.996
P1_NORM_LAST_LAYER=false
P1_GLOBAL_CROPS_SCALE="0.4 1.0"
P1_LOCAL_CROPS_SCALE="0.05 0.4"
P1_LOCAL_CROPS_NUMBER=8

# ---------------------------------------------------------------------------
# Pass 2 hyperparameters — minimal augmentations, color/detail learning
# ---------------------------------------------------------------------------
P2_EPOCHS=100
P2_BATCH_SIZE=64
P2_LR=0.0000125          # 10x lower than Pass 1 to preserve learned features
P2_MIN_LR=1e-7
P2_WEIGHT_DECAY=0.04
P2_WEIGHT_DECAY_END=0.04  # kept flat, no extra regularization needed
P2_WARMUP_EPOCHS=5
P2_TEACHER_TEMP=0.04
P2_WARMUP_TEACHER_TEMP=0.04
P2_WARMUP_TEACHER_TEMP_EPOCHS=0
P2_MOMENTUM_TEACHER=0.9996
P2_NORM_LAST_LAYER=true
P2_GLOBAL_CROPS_SCALE="0.8 1.0"  # conservative crops, preserve color context
P2_LOCAL_CROPS_SCALE="0.5 0.8"   # larger local patches, less extreme viewpoints
P2_LOCAL_CROPS_NUMBER=2           # fewer local crops, less distortion per image

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
RUN_PASS1=true
RUN_PASS2=true

for arg in "$@"; do
    case "$arg" in
        --pass1-only) RUN_PASS2=false ;;
        --pass2-only) RUN_PASS1=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
PASS1_OUTPUT="${OUTPUT_ROOT}/pass1_shape"
PASS2_OUTPUT="${OUTPUT_ROOT}/pass2_color"

mkdir -p "$PASS1_OUTPUT" "$PASS2_OUTPUT"

# ---------------------------------------------------------------------------
# Pass 1 — Self-supervised pre-training (shape learning)
# ---------------------------------------------------------------------------
if [ "$RUN_PASS1" = true ]; then
    echo "============================================================"
    echo "  PASS 1: Self-supervised pre-training (shape learning)"
    echo "  Epochs         : $P1_EPOCHS"
    echo "  Batch size/GPU : $P1_BATCH_SIZE"
    echo "  LR             : $P1_LR"
    echo "  Global crops   : $P1_GLOBAL_CROPS_SCALE"
    echo "  Local crops    : $P1_LOCAL_CROPS_SCALE  x$P1_LOCAL_CROPS_NUMBER"
    echo "  Augmentations  : FULL (ColorJitter, Grayscale, Blur, Solarization)"
    echo "  Output         : $PASS1_OUTPUT"
    echo "============================================================"

    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        main_dino.py \
        --arch "$ARCH" \
        --patch_size "$PATCH_SIZE" \
        --out_dim "$OUT_DIM" \
        --num_workers "$NUM_WORKERS" \
        --epochs "$P1_EPOCHS" \
        --warmup_epochs "$P1_WARMUP_EPOCHS" \
        --batch_size_per_gpu "$P1_BATCH_SIZE" \
        --lr "$P1_LR" \
        --min_lr "$P1_MIN_LR" \
        --weight_decay "$P1_WEIGHT_DECAY" \
        --weight_decay_end "$P1_WEIGHT_DECAY_END" \
        --teacher_temp "$P1_TEACHER_TEMP" \
        --warmup_teacher_temp "$P1_WARMUP_TEACHER_TEMP" \
        --warmup_teacher_temp_epochs "$P1_WARMUP_TEACHER_TEMP_EPOCHS" \
        --momentum_teacher "$P1_MOMENTUM_TEACHER" \
        --norm_last_layer "$P1_NORM_LAST_LAYER" \
        --global_crops_scale $P1_GLOBAL_CROPS_SCALE \
        --local_crops_scale $P1_LOCAL_CROPS_SCALE \
        --local_crops_number "$P1_LOCAL_CROPS_NUMBER" \
        --data_path "$DATA_PATH" \
        --output_dir "$PASS1_OUTPUT"

    echo "Pass 1 complete. Checkpoint saved to: $PASS1_OUTPUT"
fi

# ---------------------------------------------------------------------------
# Locate the Pass 1 teacher checkpoint for Pass 2 initialization
# ---------------------------------------------------------------------------
PASS1_CHECKPOINT="${PASS1_OUTPUT}/checkpoint.pth"

if [ "$RUN_PASS2" = true ] && [ ! -f "$PASS1_CHECKPOINT" ]; then
    echo "ERROR: Pass 1 checkpoint not found at $PASS1_CHECKPOINT"
    echo "Run Pass 1 first or verify the checkpoint path."
    exit 1
fi

# ---------------------------------------------------------------------------
# Pass 2 — Supervised fine-tuning (color and detail learning)
# ---------------------------------------------------------------------------
if [ "$RUN_PASS2" = true ]; then
    echo ""
    echo "============================================================"
    echo "  PASS 2: Supervised fine-tuning (color/detail learning)"
    echo "  Pretrained     : $PASS1_CHECKPOINT"
    echo "  Epochs         : $P2_EPOCHS"
    echo "  Batch size/GPU : $P2_BATCH_SIZE"
    echo "  LR             : $P2_LR  (10x lower than Pass 1)"
    echo "  Global crops   : $P2_GLOBAL_CROPS_SCALE"
    echo "  Local crops    : $P2_LOCAL_CROPS_SCALE  x$P2_LOCAL_CROPS_NUMBER"
    echo "  Augmentations  : MINIMAL (crop + flip only; ColorJitter/Grayscale/Solarization disabled)"
    echo "  Output         : $PASS2_OUTPUT"
    echo "============================================================"

    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        main_dino.py \
        --arch "$ARCH" \
        --patch_size "$PATCH_SIZE" \
        --out_dim "$OUT_DIM" \
        --num_workers "$NUM_WORKERS" \
        --epochs "$P2_EPOCHS" \
        --warmup_epochs "$P2_WARMUP_EPOCHS" \
        --batch_size_per_gpu "$P2_BATCH_SIZE" \
        --lr "$P2_LR" \
        --min_lr "$P2_MIN_LR" \
        --weight_decay "$P2_WEIGHT_DECAY" \
        --weight_decay_end "$P2_WEIGHT_DECAY_END" \
        --teacher_temp "$P2_TEACHER_TEMP" \
        --warmup_teacher_temp "$P2_WARMUP_TEACHER_TEMP" \
        --warmup_teacher_temp_epochs "$P2_WARMUP_TEACHER_TEMP_EPOCHS" \
        --momentum_teacher "$P2_MOMENTUM_TEACHER" \
        --norm_last_layer "$P2_NORM_LAST_LAYER" \
        --global_crops_scale $P2_GLOBAL_CROPS_SCALE \
        --local_crops_scale $P2_LOCAL_CROPS_SCALE \
        --local_crops_number "$P2_LOCAL_CROPS_NUMBER" \
        --color_aug false \
        --pretrained_weights "$PASS1_CHECKPOINT" \
        --data_path "$DATA_PATH" \
        --output_dir "$PASS2_OUTPUT"

    echo "Pass 2 complete. Checkpoint saved to: $PASS2_OUTPUT"
fi

# ---------------------------------------------------------------------------
# Optional: k-NN evaluation after both passes
# ---------------------------------------------------------------------------
if [ "$RUN_PASS1" = true ] && [ "$RUN_PASS2" = true ]; then
    echo ""
    echo "============================================================"
    echo "  k-NN Evaluation — Pass 1 vs Pass 2"
    echo "============================================================"

    PASS1_FEATURES="${PASS1_OUTPUT}/features"
    PASS2_FEATURES="${PASS2_OUTPUT}/features"
    mkdir -p "$PASS1_FEATURES" "$PASS2_FEATURES"

    echo ""
    echo "--- k-NN: Pass 1 (shape backbone) ---"
    torchrun \
        --nproc_per_node=1 \
        eval_knn.py \
        --arch "$ARCH" \
        --patch_size "$PATCH_SIZE" \
        --pretrained_weights "${PASS1_OUTPUT}/checkpoint.pth" \
        --checkpoint_key teacher \
        --nb_knn 5 10 20 \
        --data_path "$DATA_PATH" \
        --dump_features "$PASS1_FEATURES" \
        2>&1 | tee "${PASS1_FEATURES}/knn_results.log"

    echo ""
    echo "--- k-NN: Pass 2 (color+detail backbone) ---"
    torchrun \
        --nproc_per_node=1 \
        eval_knn.py \
        --arch "$ARCH" \
        --patch_size "$PATCH_SIZE" \
        --pretrained_weights "${PASS2_OUTPUT}/checkpoint.pth" \
        --checkpoint_key teacher \
        --nb_knn 5 10 20 \
        --data_path "$DATA_PATH" \
        --dump_features "$PASS2_FEATURES" \
        2>&1 | tee "${PASS2_FEATURES}/knn_results.log"

    echo ""
    echo "--- Feature plots: Pass 1 ---"
    python plot_features.py --features_dir "$PASS1_FEATURES"

    echo ""
    echo "--- Feature plots: Pass 2 ---"
    python plot_features.py --features_dir "$PASS2_FEATURES"
fi

echo ""
echo "All done."
