#!/bin/bash
# Run linear classification evaluation on top of a frozen DINO model.
# The dataset directory should follow ImageFolder structure:
#   DATA_PATH/
#       class_a/img1.jpg ...
#       class_b/img1.jpg ...

# ---- paths ----------------------------------------------------------------
PRETRAINED_WEIGHTS="/nfs/data/1/rrazakami/work/dino/OUTPUT_DINO/checkpoint.pth"
DATA_PATH="/nfs/data/1/rrazakami/work/dino/PetImages/"
OUTPUT_DIR="./OUTPUT_DINO/linear_eval_output"
# ---------------------------------------------------------------------------

ARCH="vit_small"          # vit_tiny | vit_small | vit_base
PATCH_SIZE=16
CHECKPOINT_KEY="teacher"

NUM_LABELS=10             # number of classes in your dataset
N_LAST_BLOCKS=4           # 4 for vit_small, 1 for vit_base
AVGPOOL=false             # false for vit_small, true for vit_base

EPOCHS=100
LR=0.001
BATCH_SIZE=128
NUM_WORKERS=8
VAL_SPLIT=0.2             # 20% held out for validation

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nproc_per_node=1 \
    eval_linear.py \
    --arch "${ARCH}" \
    --patch_size "${PATCH_SIZE}" \
    --pretrained_weights "${PRETRAINED_WEIGHTS}" \
    --checkpoint_key "${CHECKPOINT_KEY}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_labels "${NUM_LABELS}" \
    --n_last_blocks "${N_LAST_BLOCKS}" \
    --avgpool_patchtokens "${AVGPOOL}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size_per_gpu "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --val_split "${VAL_SPLIT}" \
    --val_freq 5
