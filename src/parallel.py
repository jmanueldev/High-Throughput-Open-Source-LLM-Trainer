import torch
import torch.nn as nn
import torch.distributed as dist

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
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
    def __init__(self, in_features: int, out_features: int, tp_size: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // tp_size

        if TE_AVAILABLE:
            self.linear = te.Linear(in_features, self.out_features_per_partition, bias=bias)
        else:
            self.linear = nn.Linear(in_features, self.out_features_per_partition, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tensor_parallel_region(x)
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """
    Splits input dimension across TP ranks.
    $W \in \mathbb{R}^{(H / TP) \times O}$
    Applies All-Reduce on output.
    """
    def __init__(self, in_features: int, out_features: int, tp_size: int, bias: bool = True):
        super().__init__()
        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features

        if TE_AVAILABLE:
            self.linear = te.Linear(self.in_features_per_partition, out_features, bias=bias)
        else:
            self.linear = nn.Linear(self.in_features_per_partition, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return reduce_from_tensor_parallel_region(out)