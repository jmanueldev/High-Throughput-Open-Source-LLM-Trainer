import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Transformer Engine provides accelerated FP8 GEMMs with dynamic scaling
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


# =====================================================================
# 1. Megatron-Style Distributed Primitives (TP Column & Row Parallel)
# =====================================================================

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass input as-is in forward, sum gradients across TP ranks in backward."""
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce sum in forward, pass gradients as-is in backward."""
    @staticmethod
    def forward(ctx, input_):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def copy_to_tensor_parallel_region(x):
    return _CopyToModelParallelRegion.apply(x)


def reduce_from_tensor_parallel_region(x):
    return _ReduceFromModelParallelRegion.apply(x)


class ColumnParallelLinear(nn.Module):
    """
    Splits output dimension across TP ranks.
    $W \in \mathbb{R}^{H \times (O / TP)}$
    """
    def __init__(self, in_features, out_features, tp_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // tp_size
        
        if TE_AVAILABLE:
            self.linear = te.Linear(in_features, self.out_features_per_partition, bias=bias)
        else:
            self.linear = nn.Linear(in_features, self.out_features_per_partition, bias=bias)

    def forward(self, x):
        # Input is replicated across ranks
        x = copy_to_tensor_parallel_region(x)
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """
    Splits input dimension across TP ranks.
    $W \in \mathbb{R}^{(H / TP) \times O}$
    Applies All-Reduce on output.
    """
    def __init__(self, in_features, out_features, tp_size, bias=True):
        super().__init__()
        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features
        
        if TE_AVAILABLE:
            self.linear = te.Linear(self.in_features_per_partition, out_features, bias=bias)
        else:
            self.linear = nn.Linear(self.in_features_per_partition, out_features, bias=bias)

    def forward(self, x):
        # Linear projection on split input dimension
        out = self.linear(x)
        # Sum outputs across TP group
        return reduce_from_tensor_parallel_region(out)


# =====================================================================
# 2. Megatron Transformer Layer
# =====================================================================

class TensorParallelMLP(nn.Module):
    """
    Standard Megatron MLP:
    Column Parallel (Hidden -> 4*Hidden) -> GELU -> Row Parallel (4*Hidden -> Hidden)
    """
    def __init__(self, hidden_dim, ffn_dim, tp_size):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden_dim, ffn_dim, tp_size)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(ffn_dim, hidden_dim, tp_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TensorParallelAttention(nn.Module):
    """
    Tensor-Parallel Self-Attention:
    Column Parallel QKV projection -> Multi-Head Attn -> Row Parallel Output projection
    """
    def __init__(self, hidden_dim, num_heads, tp_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_dim // num_heads

        # Combined Q, K, V column-parallel projection
        self.qkv_proj = ColumnParallelLinear(hidden_dim, 3 * hidden_dim, tp_size)
        self.out_proj = RowParallelLinear(hidden_dim, hidden_dim, tp_size)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, S, 3 * (hidden_dim / tp_size)]
        
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        q = q.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        # FlashAttention / Scaled Dot-Product Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, tp_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = TensorParallelAttention(hidden_dim, num_heads, tp_size)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = TensorParallelMLP(hidden_dim, ffn_dim, tp_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, tp_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, tp_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = ColumnParallelLinear(hidden_dim, vocab_size, tp_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# =====================================================================
# 3. Synthetic Dataset & Dataset Loader
# =====================================================================

class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size, seq_len, total_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return tokens[:-1], tokens[1:]


# =====================================================================
# 4. Harness & Execution Loop
# =====================================================================

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def main():
    local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Model Hyperparameters
    vocab_size = 32000
    hidden_dim = 2048
    num_layers = 12
    num_heads = 16
    ffn_dim = 8192
    seq_len = 1024
    batch_size = 4
    num_steps = 100

    # Model Initialization
    model = GPTModel(vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, tp_size=world_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # Dataset
    dataset = RandomTokenDataset(vocab_size, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # FP8 Recipe Setup (E4M3 for forward, E5M2 for backward)
    fp8_recipe = None
    if TE_AVAILABLE:
        fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID)
        if local_rank == 0:
            print("[INFO] Transformer Engine detected. FP8 Precision enabled.")
    else:
        if local_rank == 0:
            print("[WARN] Transformer Engine NOT detected. Falling back to standard PyTorch precision.")

    model.train()
    start_time = time.time()

    for step, (x, y) in enumerate(dataloader):
        if step >= num_steps:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # FP8 Autocast Context
        if TE_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()

        if local_rank == 0 and step % 10 == 0:
            elapsed = time.time() - start_time
            tps = (step + 1) * batch_size * seq_len * world_size / elapsed
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Throughput: {tps:.2f} tokens/sec")

    if local_rank == 0:
        print("Pre-training harness run completed successfully.")


if __name__ == "__main__":
    main()