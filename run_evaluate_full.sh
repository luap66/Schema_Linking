#!/bin/bash
# Run evaluation of the fully finetuned ExSL model on the H100 GPU server.
# Usage: bash run_evaluate_full.sh

OUTPUT_DIR="/mnt/data/paul_waldhoff/exsl_full_results"
IMAGE_NAME="exsl-full-ft"
HF_CACHE="${HOME}/.cache/huggingface"

# Run evaluation
docker run --rm \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
    --shm-size=16g \
    -v "$(pwd)":/app \
    -v "$OUTPUT_DIR":/app/output \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e OUTPUT_DIR=/app/output \
    "$IMAGE_NAME" \
    python3 evaluate_full.py "$@"
