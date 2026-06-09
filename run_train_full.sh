#!/bin/bash
# Run full finetuning on the H100 GPU server.
# Usage: bash run_train_full.sh
#
# Results (model + head) are saved to /mnt/data/<your_name>/exsl_full/
# Adjust OUTPUT_DIR to your personal folder on the network share.

OUTPUT_DIR="/mnt/data/paul_waldhoff/exsl_full_results"
IMAGE_NAME="exsl-full-ft"
HF_CACHE="${HOME}/.cache/huggingface"   # reuse cached model weights across runs

mkdir -p "$OUTPUT_DIR" "$HF_CACHE"

# Build the Docker image (only needed once / after code changes)
docker build -t "$IMAGE_NAME" .

# Run training
docker run --rm \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --shm-size=16g \
    -v "$OUTPUT_DIR":/app/output \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e OUTPUT_DIR=/app/output \
    "$IMAGE_NAME" \
    python3 train_full.py
