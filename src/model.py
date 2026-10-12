import torch
import torch.nn as nn
import torch.nn.functional as F
from src.parallel import ColumnParallelLinear, RowParallelLinear


class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, tp_size: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden_dim, ffn_dim, tp_size)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(ffn_dim, hidden_dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TensorParallelAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, tp_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = ColumnParallelLinear(hidden_dim, 3 * hidden_dim, tp_size)
        self.out_proj = RowParallelLinear(hidden_dim, hidden_dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, tp_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = TensorParallelAttention(hidden_dim, num_heads, tp_size)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = TensorParallelMLP(hidden_dim, ffn_dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        tp_size: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, tp_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = ColumnParallelLinear(hidden_dim, vocab_size, tp_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)