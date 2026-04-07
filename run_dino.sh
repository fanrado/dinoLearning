#!/bin/bash

# ===== DINO Training Script =====

# Path to your ImageNet (or custom) training data
DATA_PATH="/nfs/data/1/rrazakami/work/dino/PetImages"

# Output directory for checkpoints and logs
OUTPUT_DIR="./OUTPUT_DINO"

# Number of GPUs to use
NUM_GPUS=1

# --- Model parameters ---
ARCH="vit_small"
PATCH_SIZE=16
OUT_DIM=65536

# --- Training parameters ---
EPOCHS=100
BATCH_SIZE_PER_GPU=16
LR=0.0005
NUM_WORKERS=10

mkdir -p "$OUTPUT_DIR"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    main_dino.py \
    --arch "$ARCH" \
    --patch_size "$PATCH_SIZE" \
    --out_dim "$OUT_DIM" \
    --epochs "$EPOCHS" \
    --batch_size_per_gpu "$BATCH_SIZE_PER_GPU" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
