#!/bin/bash

source /nfs/data/1/rrazakami/work/venv/bin/activate

# Path to pretrained DINO weights
PRETRAINED_WEIGHTS="/nfs/data/1/rrazakami/work/dino/OUTPUT_DINO/checkpoint.pth"

# Path to ImageNet dataset (must contain train/ and val/ subdirs)
DATA_PATH="/nfs/data/1/rrazakami/work/dino/PetImages/"

# Optional: directory to save/load extracted features
DUMP_FEATURES="/nfs/data/1/rrazakami/work/dino/OUTPUT_DINO/features"      # set to a path to save features, e.g. "./features"
LOAD_FEATURES=""      # set to a path to load precomputed features

# Number of GPUs
NUM_GPUS=1

torchrun --nproc_per_node=${NUM_GPUS} eval_knn.py \
    --arch vit_small \
    --patch_size 16 \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --checkpoint_key teacher \
    --data_path ${DATA_PATH} \
    --batch_size_per_gpu 32 \
    --nb_knn 10 20 100 200 \
    --temperature 0.07 \
    --num_workers 10 \
    --use_cuda true \
    ${DUMP_FEATURES:+--dump_features ${DUMP_FEATURES}} \
    ${LOAD_FEATURES:+--load_features ${LOAD_FEATURES}}
