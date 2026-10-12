#!/usr/bin/env bash
set -e

# Default to 4 GPUs if NUM_GPUS is not explicitly set
NUM_GPUS=${1:-4}

echo "Launching distributed trainer on ${NUM_GPUS} GPUs..."
torchrun --nproc_per_node=${NUM_GPUS} src/train.py