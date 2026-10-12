# High-Throughput Open-Source LLM Trainer

A minimal, hyper-optimized PyTorch pre-training harness featuring Megatron-style Tensor Parallelism (TP) and NVIDIA Transformer Engine FP8 precision execution.

## Features

- **Megatron-style Tensor Parallelism**: Custom CUDA/NCCL autograd functions (`_CopyToModelParallelRegion` and `_ReduceFromModelParallelRegion`) implementing split column and row parallel linear projections.
- **FP8 Mixed Precision**: Integrated `transformer_engine.pytorch` using HYBRID FP8 format (`E4M3` forward, `E5M2` backward) with delayed scaling.
- **Scaled Dot-Product Attention**: Native FlashAttention implementation via PyTorch Fused Kernels.

## Getting Started

### Prerequisites

Ensure you have NVIDIA drivers and CUDA Toolkit installed on Hopper/Ada architecture GPUs (e.g., H100, L40S) for native FP8 support.

```bash
pip install -r requirements.txt
```
## Launch Training

Make the launcher script executable and start pre-training across available GPUs:
```bash
chmod +x launch.sh
./launch.sh 4  # Launches on 4 GPUs
