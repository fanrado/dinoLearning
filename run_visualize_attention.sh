#!/bin/bash

# ===== DINO Attention Visualization Script =====

# Path to the pretrained checkpoint (teacher weights)
PRETRAINED_WEIGHTS="./OUTPUT_DINO_ITERATIVE/pass1_shape/checkpoint0040.pth"

# Architecture and patch size must match the checkpoint
ARCH="vit_small"
PATCH_SIZE=16

# Key inside the checkpoint dict to load weights from
CHECKPOINT_KEY="teacher"

# Path to the image to visualize
IMAGE_PATH="./img.png"
# IMAGE_PATH="./PetImages/Cat/167.jpg"


# Resize the image to this size before processing
IMAGE_SIZE="960 960"

# Output directory for saved attention maps
OUTPUT_DIR="./OUTPUT_DINO_ITERATIVE/pass1_shape/attention_visualization"
mkdir -p "$OUTPUT_DIR"
# Optional: threshold to keep top xx% of attention mass (e.g. 0.6); leave empty to disable
THRESHOLD="0.3"

mkdir -p "$OUTPUT_DIR"

THRESHOLD_ARG=""
if [ -n "$THRESHOLD" ]; then
    THRESHOLD_ARG="--threshold $THRESHOLD"
fi

python visualize_attention.py \
    --arch "$ARCH" \
    --patch_size "$PATCH_SIZE" \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    --checkpoint_key "$CHECKPOINT_KEY" \
    --image_path "$IMAGE_PATH" \
    --image_size $IMAGE_SIZE \
    --output_dir "$OUTPUT_DIR" \
    $THRESHOLD_ARG
