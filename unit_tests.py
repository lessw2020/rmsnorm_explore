from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from titan_fwd import FusedRMSNorm as TritonFusedRMSNorm
from torch import nn


class TorchRMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


def generate_test_cases():
    """Generate diverse test cases for thorough testing"""
    test_cases = []

    # Test different sizes
    sizes = [
        (1, 2048),  # Single sequence
        (8, 2048),  # Batch of sequences
        (2, 16, 1024),  # Multiple batches
        (4, 32, 2048),  # Larger hidden size
        (16, 8, 4096),  # Even larger hidden size
        (4, 8, 8192),  # Even larger hidden size
    ]

    # Test different dtypes
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    # Test different epsilon values
    eps_values = [1e-6, 1e-5, 1e-4]

    for size in sizes:
        for dtype in dtypes:
            for eps in eps_values:
                test_cases.append({"size": size, "dtype": dtype, "eps": eps})

    return test_cases


def setup_test(
    test_case: dict, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Setup test inputs and parameters"""
    size = test_case["size"]
    dtype = test_case["dtype"]
    eps = test_case["eps"]

    # Generate input with values between -2 and 2
    x = (torch.randn(*size, device=device) * 2).to(dtype)

    # Generate weights close to 1 for stability
    weight = (
        torch.ones(size[-1], device=device) + torch.randn(size[-1], device=device) * 0.1
    ).to(dtype)

    return x, weight, eps


def compute_relative_error(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute relative error between two tensors"""
    return torch.abs(a - b) / (torch.abs(b) + eps)


class TestRMSNorm:
    @pytest.mark.parametrize("test_case", generate_test_cases())
    def test_forward_accuracy(self, test_case):
        """Test forward pass accuracy against reference implementation"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, weight, eps = setup_test(test_case, device)

        # Initialize both implementations
        triton_norm = TritonFusedRMSNorm(
            hidden_size=test_case["size"][-1],
            eps=eps,
            device=device,
            dtype=test_case["dtype"],
        )
        torch_norm = (
            TorchRMSNorm(hidden_size=test_case["size"][-1], eps=eps)
            .to(device)
            .to(test_case["dtype"])
        )

        # Set same weights
        with torch.no_grad():
            triton_norm.weight.copy_(weight)
            torch_norm.weight.copy_(weight)

        # Compute outputs
        with torch.no_grad():
            out_triton = triton_norm(x)
            out_torch = torch_norm(x)

        # Compute relative error
        rel_error = compute_relative_error(out_triton, out_torch)
        max_rel_error = rel_error.max().item()

        # Define tolerance based on dtype
        rtol = 1e-5 if test_case["dtype"] == torch.float32 else 1e-1

        assert max_rel_error < rtol, (
            f"Forward relative error {max_rel_error} exceeds tolerance {rtol} "
            f"for test case: {test_case}"
        )

    '''@pytest.mark.parametrize("test_case", generate_test_cases())
    def test_backward_accuracy(self, test_case):
        """Test backward pass accuracy against reference implementation"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, weight, eps = setup_test(test_case, device)

        # Initialize both implementations
        triton_norm = RMSNorm(
            hidden_size=test_case["size"][-1],
            eps=eps,
            device=device,
            dtype=test_case["dtype"],
        )
        torch_norm = (
            TorchRMSNorm(hidden_size=test_case["size"][-1], eps=eps)
            .to(device)
            .to(test_case["dtype"])
        )

        # Set same weights and requires_grad
        with torch.no_grad():
            triton_norm.weight.copy_(weight)
            torch_norm.weight.copy_(weight)
        x.requires_grad = True
        x_clone = x.clone().detach().requires_grad_(True)

        # Forward pass
        out_triton = triton_norm(x)
        out_torch = torch_norm(x_clone)

        # Create gradient
        grad_output = torch.randn_like(out_triton)

        # Backward pass
        out_triton.backward(grad_output)
        out_torch.backward(grad_output)

        # Check input gradients
        rel_error_dx = compute_relative_error(x.grad, x_clone.grad)
        max_rel_error_dx = rel_error_dx.max().item()

        # Check weight gradients
        rel_error_dw = compute_relative_error(
            triton_norm.weight.grad, torch_norm.weight.grad
        )
        max_rel_error_dw = rel_error_dw.max().item()

        # Define tolerance based on dtype
        rtol = 1e-3 if test_case["dtype"] == torch.float16 else 1e-5

        assert max_rel_error_dx < rtol, (
            f"Backward (dx) relative error {max_rel_error_dx} exceeds tolerance {rtol} "
            f"for test case: {test_case}"
        )
        assert max_rel_error_dw < rtol, (
            f"Backward (dw) relative error {max_rel_error_dw} exceeds tolerance {rtol} "
            f"for test case: {test_case}"
        )
    
    @pytest.mark.parametrize("test_case", generate_test_cases())
    def test_numerical_gradient(self, test_case):
        """Test gradients using torch.autograd.gradcheck"""
        if test_case["dtype"] == torch.float16:
            pytest.skip("gradcheck requires float64/float32")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, weight, eps = setup_test(test_case, device)

        # Convert to float64 for numerical stability
        x = x.to(torch.float64)
        weight = weight.to(torch.float64)

        def rms_norm_func(x, weight):
            return fused_rms_norm(x, weight, eps)

        x.requires_grad_(True)
        weight.requires_grad_(True)

        # Reduce size for gradcheck as it's computationally expensive
        if x.numel() > 1024:
            x = x[..., :1024].contiguous()
            weight = weight[:1024].contiguous()

        torch.autograd.gradcheck(
            rms_norm_func, (x, weight), eps=1e-6, atol=1e-4, nondet_tol=0.0
        )
    '''

    @pytest.mark.parametrize("test_case", generate_test_cases())
    def test_performance(self, test_case):
        """Compare performance between implementations"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, weight, eps = setup_test(test_case, device)

        # Initialize both implementations
        triton_norm = TritonFusedRMSNorm(
            hidden_size=test_case["size"][-1],
            eps=eps,
            device=device,
            dtype=test_case["dtype"],
        )
        torch_norm = (
            TorchRMSNorm(hidden_size=test_case["size"][-1], eps=eps)
            .to(device)
            .to(test_case["dtype"])
        )

        # Set same weights
        with torch.no_grad():
            triton_norm.weight.copy_(weight)
            torch_norm.weight.copy_(weight)

        # Warmup
        for _ in range(10):
            triton_norm(x)
            torch_norm(x)

        torch.cuda.synchronize()

        # Benchmark Triton implementation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            triton_norm(x)
        end.record()

        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 100

        # Benchmark PyTorch implementation
        start.record()
        for _ in range(100):
            torch_norm(x)
        end.record()

        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end) / 100

        print(f"\nPerformance for test case {test_case}:")
        print(f"Triton: {triton_time:.3f} ms")
        print(f"PyTorch: {torch_time:.3f} ms")
        print(f"Speedup: {torch_time/triton_time:.2f}x")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
