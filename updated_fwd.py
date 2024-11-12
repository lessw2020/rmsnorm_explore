import math

from functools import partial
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def rms_forward_core(
    X,
    X_stride,
    Y,
    Y_stride,
    W,
    rstd,
    eps,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_index = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # input and output pointer indexing
    x_row_pointer = X + row_index * X_stride
    y_row_pointer = Y + row_index * Y_stride

    # load data
    x_row = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    w_row = tl.load(W + cols, mask=mask, other=0.0)
    x_row = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    w_row = tl.load(W + cols, mask=mask, other=0.0)

    # upscale
    x_row_fp32 = x_row_pointer.to(tl.float32)

    # compute RMS
    x_row_squared = x_row_fp32 * x_row_fp32
    mean_squared = tl.sum(x_row_squared, axis=0) / N
    rstd_row = 1.0 / tl.sqrt(mean_squared + eps)

    # store rstd
    tl.store(rstd + row_index, rstd_row)

    # normalize and scale
    y_row = (x_row_fp32 * rstd_row).to(x_row.dtype) * w_row

    # save output
    tl.store(y_row_pointer + cols, y_row, mask=mask)


class TritonFusedRMSNorm(torch.autograd.Function):
    """
    RMS Normalization layer using optimized Triton kernel
    Args:
        hidden_size: Size of hidden dimension
        eps: Small constant for numerical stability
        device: Device to place the layer on
        dtype: Data type of parameters
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonFusedRMSNorm.apply(x, self.weight, self.eps)


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Functional interface for fused RMS normalization
    Args:
        x: Input tensor
        weight: Weight tensor
        eps: Small constant for numerical stability
    Returns:
        Normalized tensor
    """
    return TritonFusedRMSNorm.apply(x, weight, eps)
