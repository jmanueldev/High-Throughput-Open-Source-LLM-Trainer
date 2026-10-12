# High-Throughput-Open-Source-LLM-Trainer
High-Throughput Open-Source LLM Trainer: Built a minimal, hyper-optimized PyTorch pre-training harness supporting Megatron-style tensor parallelism and FP8 precision 

A self-contained, production-grade PyTorch pre-training harness featuring Megatron-style Tensor Parallelism (TP), Transformer Engine FP8 autocast, and a custom distributed training loop.

Key System Architecture Highlights

    Megatron-Style Parallel Operators:

        ColumnParallelLinear: Splits weights along columns and duplicates activation tensor forward passes while accumulating gradients in backward passes via all_reduce.

        RowParallelLinear: Splits weights along rows and performs spatial all_reduce summation across ranks to sync output states.

    NVIDIA Transformer Engine (TE) FP8 Integration:

        Utilizes te.Linear and te.fp8_autocast using HYBRID FP8 Format (E4M3 forward pass for precision, E5M2 backward pass for gradient dynamic range). Includes delayed scaling for overflow prevention.

    Optimized Attention Kernel:

        Integrates PyTorch's native scaled_dot_product_attention for scaled FlashAttention execution under reduced precision.

Execution Command

Launch across 4 GPUs via torchrun:
torchrun --nproc_per_node=4 train.py